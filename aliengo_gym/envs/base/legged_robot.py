# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from aliengo_gym import MINI_GYM_ROOT_DIR
from aliengo_gym.envs.base.base_task import BaseTask
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from aliengo_gym.utils.terrain import Terrain
# from .legged_robot_config import BaseCfg as Cfg
# from .fall_recovery_config import FallRecoveryConfig as Cfg
# from .fall_recovery_config_go1 import FallRecoveryConfig as Cfg
from .fall_recovery_config_tr import FallRecoveryConfig as Cfg
from pathlib import Path


class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        # self.debug_viz = True
        self.debug_viz = getattr(self.cfg.terrain, "debug_height_grid", False)
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = self.cfg.env.record_now
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

        self.debug_rewards = True
        self.debug_counter = 0

        self.max_video_frames = self.cfg.env.max_video_frames

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Store current torques as "last" BEFORE computing new ones
        # self.last_last_torques[:] = self.last_torques[:]
        self.last_torques[:] = self.torques[:]

        if getattr(self.cfg.env, "debug_hold_reset_pose", False):
            q_default = self._get_default_dof_pos_for_env_ids(
                torch.arange(self.num_envs, device=self.device)
            )

            q_hold = self.debug_hold_dof_pos[:, :actions.shape[1]]
            q_default = q_default[:, :actions.shape[1]]

            action_scale = self.cfg.control.action_scale

            if not torch.is_tensor(action_scale):
                action_scale = torch.ones_like(actions) * float(action_scale)
            else:
                action_scale = action_scale.to(self.device).view(1, -1)

            actions[:] = torch.clamp(
                (q_hold - q_default) / (action_scale + 1e-6),
                -1.0,
                1.0,
            )

        elif getattr(self.cfg.env, "debug_zero_actions", False):
            actions[:] = 0.0

        # ------------------------------------------------------------
        # Multi-zone state-based action limiting
        # ------------------------------------------------------------

        clip_actions = float(self.cfg.normalization.clip_actions)

        effective_clip = torch.full(
            (self.num_envs, 1),
            clip_actions,
            device=self.device,
            dtype=actions.dtype,
        )

        # Gravity in base frame.
        # Upright robot: g_now[:, 2] close to -1.
        # Fallen/side/inverted: closer to 0 or positive.
        g_now = quat_rotate_inverse(
            self.root_states[:, 3:7],
            self.gravity_vec,
        )

        z = self.root_states[:, 2]

        self.near_crouch_clip_frac = torch.tensor(
            0.0,
            device=self.device,
            dtype=torch.float,
        )

        self.near_stand_clip_frac = torch.tensor(
            0.0,
            device=self.device,
            dtype=torch.float,
        )

        near_crouch_action_clip = getattr(self.cfg.env, "near_crouch_action_clip", None)
        near_stand_action_clip = getattr(self.cfg.env, "near_stand_action_clip", None)

        # Transition zone:
        # robot is not fully standing yet, but it is no longer deeply fallen.
        # This prevents sudden full [-10, 10] actions during terminal collapse.
        if near_crouch_action_clip is not None:
            near_crouch = (
                (g_now[:, 2] < -0.45)
                & (z > 0.18)
            )

            self.near_crouch_clip_frac = near_crouch.float().mean()

            if near_crouch.any():
                effective_clip[near_crouch] = min(
                    float(near_crouch_action_clip),
                    clip_actions,
                )

        # Strict terminal stabilization zone.
        if near_stand_action_clip is not None:
            near_stand = (
                (g_now[:, 2] < -0.75)
                & (z > 0.27)
            )

            self.near_stand_clip_frac = near_stand.float().mean()

            if near_stand.any():
                effective_clip[near_stand] = min(
                    float(near_stand_action_clip),
                    clip_actions,
                )

        self.actions = torch.clamp(
            actions,
            -effective_clip,
            effective_clip,
        ).to(self.device)

        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # print("dones sum:", self.reset_buf.sum())
        # print("reward sample:", self.rew_buf[:5])

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # clear sparse reward buffer
        self.recovery_bonus_buf[:] = 0.0

        # if self.debug_rewards and self.common_step_counter % 1000 == 0:
        #     base_height = self.root_states[:, 2]

        #     print("POST STEP:")
        #     print("height mean:", base_height.mean().item())
        #     print("height min:", base_height.min().item())

        #     ang_vel = torch.norm(self.root_states[:, 10:13], dim=1)

        #     print("ang vel mean:", ang_vel.mean().item())

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # if self.common_step_counter % 200 == 0:

        #     env_id = 0

        #     g = self.projected_gravity[env_id]

        #     print("\n===== GRAVITY DEBUG =====")
        #     print("projected gravity:", g.cpu().numpy())
        #     print("base height:", self.root_states[env_id, 2].item())

        #     print("lin vel:",
        #           torch.norm(self.base_lin_vel[env_id]).item())

        #     print("ang vel:",
        #           torch.norm(self.base_ang_vel[env_id]).item())

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        self._post_physics_step_callback()

        # ------------------------------------------------------------
        # Temporary terminal post-reset stability diagnostic
        # ------------------------------------------------------------
        if (
            getattr(self.cfg.env, "debug_log_terminal_reset", False)
            and self.common_step_counter % 50 == 0
            and hasattr(self, "terminal_reset_buf")
        ):
            ids = self.terminal_reset_buf.bool()

            if ids.any():
                foot_contact = (
                    self.contact_forces[ids][:, self.feet_indices, 2] > 1.0
                )

                if len(self.base_contact_indices) > 0:
                    nonfoot_contact_force = torch.norm(
                        self.contact_forces[ids][:, self.base_contact_indices, :],
                        dim=-1,
                    )

                    nonfoot_contact_frac = (
                        nonfoot_contact_force.max(dim=1).values > 0.2
                    ).float().mean().item()
                else:
                    nonfoot_contact_frac = 0.0

                print(
                    "[TERMINAL POST-STEP]",
                    "n", int(ids.sum().item()),
                    "z mean", self.root_states[ids, 2].mean().item(),
                    "z min", self.root_states[ids, 2].min().item(),
                    "z max", self.root_states[ids, 2].max().item(),
                    "gz mean", self.projected_gravity[ids, 2].mean().item(),
                    "lin vel mean", torch.norm(self.base_lin_vel[ids, :2], dim=1).mean().item(),
                    "ang vel mean", torch.norm(self.base_ang_vel[ids], dim=1).mean().item(),
                    "foot contact mean", foot_contact.float().sum(dim=1).mean().item(),
                    "nonfoot contact frac", nonfoot_contact_frac,
                )

        # Compute Energy Consumption
        self.energy_consume = torch.sum(torch.abs(self.dof_vel * self.torques), dim=1).detach().clone()
        robot_mass = 12.0 if self.cfg.env.robot == "go1" else 21.5
        self.cot = torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1) / (
            robot_mass * 9.81 * torch.norm(self.base_lin_vel[:, 0:2], dim=1).clamp_min(1e-6)
        )
        self.dof_acc[:] = (self.dof_vel[:] - self.last_dof_vel[:]) / self.dt

        # update recovery_counter first
        self.check_recovery_success()

        # then terminate based on recovery_counter
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if env_ids.numel() > 0 and self.cfg.terrain.measure_heights:
            self.base_pos[env_ids] = self.root_states[env_ids, :3]
            self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

            train_reset_ids = env_ids[env_ids < self.num_train_envs]

            if train_reset_ids.numel() > 0:
                self.measured_heights[train_reset_ids] = self._get_heights(train_reset_ids, self.cfg)

            eval_reset_ids = env_ids[env_ids >= self.num_train_envs]

            if eval_reset_ids.numel() > 0 and self.eval_cfg is not None:
                self.measured_heights[eval_reset_ids] = self._get_heights(eval_reset_ids, self.eval_cfg)

        # if self.common_step_counter % 1000 == 0:
        #     forces = self.contact_forces[:, self.base_contact_indices, :]
        #     print("force mean:", torch.mean(torch.abs(forces[:, :, 2])).item())

        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        # self.last_torques[:] = self.torques[:]
        # self.last_last_torques[:] = self.last_torques[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        if self.cfg.env.record_video:
            self._render_headless()

    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     # Contact termination is effectively DISABLED (since terminate_after_contacts_on = [])
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
    #                                dim=1)
    #     self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    #
    #     if self.cfg.rewards.use_terminal_body_height:
    #         self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
    #                                < self.cfg.rewards.terminal_body_height
    #         self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    # def check_termination(self):
    #     self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    #     # Success termination: single source of truth
    #     if self.cfg.env.train_recovery:
    #         success_reset = self.recovery_counter >= self.cfg.rewards.recovery_success_steps
    #         self.reset_buf |= success_reset

    #     # Timeout
    #     self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
    #     self.reset_buf |= self.time_out_buf

    #     root_z = self.root_states[:, 2]
    #     lin_vel_norm = torch.norm(self.root_states[:, 7:10], dim=1)
    #     ang_vel_norm = torch.norm(self.root_states[:, 10:13], dim=1)

    #     bad_state = (
    #         torch.isnan(self.root_states).any(dim=1)
    #         | torch.isinf(self.root_states).any(dim=1)
    #         | torch.isnan(self.dof_pos).any(dim=1)
    #         | torch.isinf(self.dof_pos).any(dim=1)
    #         | torch.isnan(self.dof_vel).any(dim=1)
    #         | torch.isinf(self.dof_vel).any(dim=1)
    #         | (root_z < -1.0)
    #         | (root_z > 2.0)
    #         | (lin_vel_norm > 20.0)
    #         | (ang_vel_norm > 50.0)
    #     )

    #     self.bad_state_buf = bad_state
    #     self.reset_buf |= self.bad_state_buf
    #

    def check_termination(self):
        self.reset_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # ------------------------------------------------------------
        # Recovery success should be detected in check_recovery_success().
        # During terminal-stabilizer training, do NOT terminate on success.
        # The policy must learn to remain stable after reaching recovery.
        # ------------------------------------------------------------
        if not hasattr(self, "success_reset_buf"):
            self.success_reset_buf = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        self.success_reset_buf[:] = False

        if self.cfg.env.train_recovery:
            terminate_on_success = getattr(
                self.cfg.env,
                "terminate_on_recovery_success",
                False,
            )

            if terminate_on_success:
                success_reset = (
                    self.recovery_counter
                    >= self.cfg.rewards.recovery_success_steps
                )
                self.success_reset_buf[:] = success_reset
                self.reset_buf |= success_reset

        # ------------------------------------------------------------
        # Timeout
        # ------------------------------------------------------------
        self.time_out_buf = (
            self.episode_length_buf > self.cfg.env.max_episode_length
        )
        self.reset_buf |= self.time_out_buf

        # ------------------------------------------------------------
        # Bad / exploded physics state reset
        # ------------------------------------------------------------
        root_z = self.root_states[:, 2]
        lin_vel_norm = torch.norm(self.root_states[:, 7:10], dim=1)
        ang_vel_norm = torch.norm(self.root_states[:, 10:13], dim=1)

        bad_state = (
            torch.isnan(self.root_states).any(dim=1)
            | torch.isinf(self.root_states).any(dim=1)
            | torch.isnan(self.dof_pos).any(dim=1)
            | torch.isinf(self.dof_pos).any(dim=1)
            | torch.isnan(self.dof_vel).any(dim=1)
            | torch.isinf(self.dof_vel).any(dim=1)
            | (root_z < -1.0)
            | (root_z > 2.0)
            | (lin_vel_norm > 20.0)
            | (ang_vel_norm > 50.0)
        )

        self.bad_state_buf = bad_state
        self.reset_buf |= self.bad_state_buf

        # ------------------------------------------------------------
        # Debug logging
        # ------------------------------------------------------------
        if not hasattr(self, "extras"):
            self.extras = {}

        if "recovery_debug" not in self.extras:
            self.extras["recovery_debug"] = {}

        self.extras["recovery_debug"]["success_reset_frac"] = self.success_reset_buf.float().mean().item()
        self.extras["recovery_debug"]["timeout_frac"] = self.time_out_buf.float().mean().item()
        self.extras["recovery_debug"]["bad_state_frac"] = self.bad_state_buf.float().mean().item()
        self.extras["recovery_debug"]["root_z_min"] = root_z.min().item()
        self.extras["recovery_debug"]["root_z_max"] = root_z.max().item()
        self.extras["recovery_debug"]["lin_vel_norm_max"] = lin_vel_norm.max().item()
        self.extras["recovery_debug"]["ang_vel_norm_max"] = ang_vel_norm.max().item()

    # def log_recovery_tracking_metrics(
    #     self,
    #     posture_error,
    #     foot_contact,
    #     non_slipping_feet,
    #     handoff_ready,
    # ):
    #     """
    #     Logs continuous recovery tracking diagnostics.

    #     Foot order assumed:
    #         0 = FL, 1 = FR, 2 = RL, 3 = RR

    #     For Go1, default left-right foot separation is about 0.31 m.
    #     Use crossed_frac for severe crossing/collapse diagnostics.
    #     """

    #     if not hasattr(self, "extras"):
    #         self.extras = {}

    #     if "recovery_tracking" not in self.extras:
    #         self.extras["recovery_tracking"] = {}

    #     lin_vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)
    #     ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)

    #     base_height = self.root_states[:, 2]
    #     base_height_error = torch.abs(
    #         base_height - self.cfg.rewards.recovery_height_target
    #     )

    #     # Existing contact metrics
    #     foot_contact_count = foot_contact.float().sum(dim=1)
    #     non_slipping_foot_count = non_slipping_feet.float().sum(dim=1)

    #     base_contact_force = torch.norm(
    #         self.contact_forces[:, self.base_contact_indices, :],
    #         dim=-1
    #     )

    #     base_contact_env = (
    #         base_contact_force.max(dim=1).values
    #         > self.cfg.rewards.recovery_contact_force_threshold
    #     )

    #     # foot geometry metrics in base/body frame
    #     foot_pos_body = quat_rotate_inverse(
    #         self.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
    #         (self.foot_positions - self.base_pos.unsqueeze(1)).reshape(-1, 3)
    #     ).reshape(self.num_envs, 4, 3)

    #     # Body-frame lateral foot positions
    #     y = foot_pos_body[:, :, 1]

    #     # Foot order assumed: FL, FR, RL, RR
    #     front_sep = y[:, 0] - y[:, 1]   # FL_y - FR_y
    #     rear_sep = y[:, 2] - y[:, 3]    # RL_y - RR_y

    #     # Strict crossing: left foot goes to the right of right foot
    #     front_crossed_strict = front_sep < 0.0
    #     rear_crossed_strict = rear_sep < 0.0

    #     # Practical collapse/interlocking diagnostic.
    #     # Default Go1 separation is ~0.31 m, so <0.10 means badly collapsed.
    #     front_crossed = front_sep < 0.10
    #     rear_crossed = rear_sep < 0.10

    #     # also log foot x positions to detect rear feet too far forward
    #     x = foot_pos_body[:, :, 0]
    #     front_x_mean = 0.5 * (x[:, 0] + x[:, 1])
    #     rear_x_mean = 0.5 * (x[:, 2] + x[:, 3])

    #     base_contact_env_frac = base_contact_env.float().mean()

    #     late_gate = (
    #         (self.projected_gravity[:, 2] < -0.75)
    #         & (self.root_states[:, 2] > 0.27)
    #     )

    #     late_gate_count = late_gate.float().sum()

    #     late_base_contact_frac = (
    #         (base_contact_env & late_gate).float().sum()
    #         / late_gate_count.clamp_min(1.0)
    #     )

    #     self.extras["recovery_tracking"]["base_contact_env_frac"] = base_contact_env_frac.item()
    #     self.extras["recovery_tracking"]["late_base_contact_frac"] = late_base_contact_frac.item()
    #     self.extras["recovery_tracking"]["late_gate_frac"] = late_gate.float().mean().item()

    #     self.extras["recovery_tracking"]["posture_error_mean"] = posture_error.mean().item()
    #     self.extras["recovery_tracking"]["posture_error_min"] = posture_error.min().item()
    #     self.extras["recovery_tracking"]["posture_error_max"] = posture_error.max().item()

    #     self.extras["recovery_tracking"]["base_height_mean"] = base_height.mean().item()
    #     self.extras["recovery_tracking"]["base_height_error_mean"] = base_height_error.mean().item()

    #     self.extras["recovery_tracking"]["lin_vel_norm_mean"] = lin_vel_norm.mean().item()
    #     self.extras["recovery_tracking"]["ang_vel_norm_mean"] = ang_vel_norm.mean().item()

    #     self.extras["recovery_tracking"]["foot_contact_count_mean"] = foot_contact_count.mean().item()
    #     self.extras["recovery_tracking"]["non_slipping_foot_count_mean"] = non_slipping_foot_count.mean().item()

    #     self.extras["recovery_tracking"]["base_contact_force_mean"] = base_contact_force.mean().item()

    #     self.extras["recovery_tracking"]["handoff_ready_mean"] = handoff_ready.float().mean().item()

    #     # New geometry logs
    #     self.extras["recovery_tracking"]["front_sep_mean"] = front_sep.mean().item()
    #     self.extras["recovery_tracking"]["rear_sep_mean"] = rear_sep.mean().item()

    #     self.extras["recovery_tracking"]["front_sep_min"] = front_sep.min().item()
    #     self.extras["recovery_tracking"]["rear_sep_min"] = rear_sep.min().item()

    #     self.extras["recovery_tracking"]["front_crossed_frac"] = front_crossed.float().mean().item()
    #     self.extras["recovery_tracking"]["rear_crossed_frac"] = rear_crossed.float().mean().item()

    #     self.extras["recovery_tracking"]["front_crossed_strict_frac"] = front_crossed_strict.float().mean().item()
    #     self.extras["recovery_tracking"]["rear_crossed_strict_frac"] = rear_crossed_strict.float().mean().item()

    #     self.extras["recovery_tracking"]["front_x_mean"] = front_x_mean.mean().item()
    #     self.extras["recovery_tracking"]["rear_x_mean"] = rear_x_mean.mean().item()

    def log_recovery_tracking_metrics(
            self,
            posture_error,
            foot_contact,
            non_slipping_feet,
            handoff_ready,
        ):
            """
            Logs continuous recovery tracking diagnostics.

            Expected foot order:
                0 = FL, 1 = FR, 2 = RL, 3 = RR

            Important interpretation:
                - foot_contact means the foot has enough vertical contact force.
                - non_slipping_feet means the foot is in contact and its XY velocity is low.
                - slipping_feet means the foot is in contact but still moving/sliding.
                - late_* metrics are conditioned on near-upright + near-raised states.

            Why late metrics matter:
                In fall recovery, global averages include fallen/rolling states.
                Low global foot contact does not necessarily mean the robot never
                touches the ground. The late_* metrics answer:

                    "When the robot is near recovery, are the feet loaded and stable?"
            """

            if not hasattr(self, "extras"):
                self.extras = {}

            if "recovery_tracking" not in self.extras:
                self.extras["recovery_tracking"] = {}

            tracking = self.extras["recovery_tracking"]
            cfg = self.cfg.rewards

            # ------------------------------------------------------------
            # Thresholds
            # ------------------------------------------------------------
            foot_contact_threshold = getattr(
                cfg,
                "recovery_contact_force_threshold",
                1.0,
            )

            # Non-foot contact should usually use a lower threshold than foot contact.
            # This catches thigh/calf/trunk contact without requiring large impact force.
            nonfoot_contact_threshold = getattr(
                cfg,
                "nonfoot_contact_threshold",
                0.2,
            )

            # Useful support load threshold.
            # This is stricter than visual/contact detection.
            loaded_foot_force_threshold = getattr(
                cfg,
                "loaded_foot_force_threshold",
                3.0,
            )

            # Geometry diagnostic thresholds for Go1.
            # These are diagnostics, not necessarily reward thresholds.
            front_too_wide_threshold = getattr(
                cfg,
                "front_too_wide_threshold",
                0.42,
            )

            front_too_forward_threshold = getattr(
                cfg,
                "front_too_forward_threshold",
                0.28,
            )

            rear_too_narrow_threshold = getattr(
                cfg,
                "rear_too_narrow_threshold",
                0.10,
            )

            front_too_narrow_threshold = getattr(
                cfg,
                "front_too_narrow_threshold",
                0.10,
            )

            # ------------------------------------------------------------
            # Ensure expected boolean types
            # ------------------------------------------------------------
            foot_contact = foot_contact.bool()
            non_slipping_feet = non_slipping_feet.bool()
            handoff_ready = handoff_ready.bool()

            # ------------------------------------------------------------
            # Basic recovery state metrics
            # ------------------------------------------------------------
            lin_vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)
            ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)

            # base_height = self.root_states[:, 2]
            local_terrain_height = self._get_local_terrain_height()

            base_height = (
                self.root_states[:, 2]
                - local_terrain_height
            )
            target_height = self.cfg.rewards.recovery_height_target

            base_height_error = torch.abs(base_height - target_height)

            valid_height = (
                torch.isfinite(base_height)
                & (base_height > -1.0)
                & (base_height < 2.0)
            )

            g_z = self.projected_gravity[:, 2]

            foot_fz = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
            foot_xy_vel = torch.norm(self.foot_velocities[:, :, :2], dim=-1)

            loaded_feet = foot_fz > loaded_foot_force_threshold

            # Contacted but not non-slipping.
            # This is the most useful per-foot slip diagnostic.
            slipping_feet = foot_contact & (~non_slipping_feet)

            foot_contact_count = foot_contact.float().sum(dim=1)
            loaded_foot_count = loaded_feet.float().sum(dim=1)
            non_slipping_foot_count = non_slipping_feet.float().sum(dim=1)
            slipping_foot_count = slipping_feet.float().sum(dim=1)

            stable_contacts = (
                non_slipping_foot_count >= self.cfg.rewards.recovery_min_foot_contacts
            )

            # ------------------------------------------------------------
            # Non-foot contact metrics
            # ------------------------------------------------------------
            # self.base_contact_indices should contain all non-foot bodies:
            # trunk/base, hips, thighs, calves, etc.
            #
            # If it contains only base/trunk in your code, rename these metrics
            # accordingly. The metric names below are kept compatible with your
            # existing dashboard.
            # ------------------------------------------------------------
            if len(self.base_contact_indices) > 0:
                nonfoot_contact_force = torch.norm(
                    self.contact_forces[:, self.base_contact_indices, :],
                    dim=-1,
                )

                nonfoot_contact_env = (
                    nonfoot_contact_force.max(dim=1).values
                    > nonfoot_contact_threshold
                )

                nonfoot_contact_force_mean = nonfoot_contact_force.mean()
            else:
                nonfoot_contact_force = torch.zeros(
                    self.num_envs,
                    1,
                    device=self.device,
                )
                nonfoot_contact_env = torch.zeros(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )
                nonfoot_contact_force_mean = torch.tensor(
                    0.0,
                    device=self.device,
                )

            nonfoot_contact_env_frac = nonfoot_contact_env.float().mean()

            # ------------------------------------------------------------
            # Late-stage gate
            # ------------------------------------------------------------
            # Late gate means:
            #   robot is already reasonably upright and raised.
            #
            # This avoids treating expected fallen/rolling contact as a final
            # recovery failure.
            # ------------------------------------------------------------
            late_gate = (
                (g_z < -0.75)
                & (base_height > 0.27)
            )

            late_gate_float = late_gate.float()
            late_count = late_gate_float.sum().clamp_min(1.0)

            def late_scalar_mean(x):
                """
                Mean of a scalar per-env tensor over late-gate envs.
                Returns 0 if no env is in late gate.
                """
                return (x.float() * late_gate_float).sum() / late_count

            def late_vector_mean(x):
                """
                Mean of a [num_envs, 4] tensor over late-gate envs.
                Returns a [4] tensor.
                """
                return (
                    x.float() * late_gate_float.unsqueeze(1)
                ).sum(dim=0) / late_count

            late_nonfoot_contact_frac = late_scalar_mean(nonfoot_contact_env)

            late_foot_contact_count_mean = late_scalar_mean(foot_contact_count)
            late_loaded_foot_count_mean = late_scalar_mean(loaded_foot_count)
            late_non_slipping_foot_count_mean = late_scalar_mean(non_slipping_foot_count)
            late_slipping_foot_count_mean = late_scalar_mean(slipping_foot_count)
            late_stable_contacts_frac = late_scalar_mean(stable_contacts)

            # ------------------------------------------------------------
            # Foot geometry metrics in base/body frame
            # ------------------------------------------------------------
            foot_pos_body = quat_rotate_inverse(
                self.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
                (self.foot_positions - self.base_pos.unsqueeze(1)).reshape(-1, 3),
            ).reshape(self.num_envs, 4, 3)

            x = foot_pos_body[:, :, 0]
            y = foot_pos_body[:, :, 1]

            # Foot order: FL, FR, RL, RR
            front_sep = y[:, 0] - y[:, 1]
            rear_sep = y[:, 2] - y[:, 3]

            front_crossed_strict = front_sep < 0.0
            rear_crossed_strict = rear_sep < 0.0

            # Practical collapse/interlocking diagnostics.
            # Go1 nominal left-right foot separation is about 0.31 m.
            front_crossed = front_sep < front_too_narrow_threshold
            rear_crossed = rear_sep < rear_too_narrow_threshold

            front_x_mean = 0.5 * (x[:, 0] + x[:, 1])
            rear_x_mean = 0.5 * (x[:, 2] + x[:, 3])

            front_too_wide = front_sep > front_too_wide_threshold
            front_too_forward = front_x_mean > front_too_forward_threshold

            # ------------------------------------------------------------
            # Inferred state-distribution diagnostics
            # ------------------------------------------------------------
            # These help interpret whether current data mostly comes from fallen,
            # crouched, near-stand, or terminal-candidate states.
            # ------------------------------------------------------------
            fallen_like = (
                (g_z > -0.30)
                | (base_height < 0.18)
            )

            rollup_or_crouch = (
                (g_z <= -0.30)
                & (g_z > -0.75)
                & (base_height >= 0.18)
            )

            near_stand_unstable = (
                late_gate
                & (~stable_contacts)
                & (~handoff_ready)
            )

            handoff_candidate = handoff_ready

            # ------------------------------------------------------------
            # Global tracking metrics
            # ------------------------------------------------------------
            if hasattr(self, "recovery_reset_source_buf"):
                tracking["terminal_reset_frac"] = self.recovery_reset_source_buf.float().mean().item()

            tracking["posture_error_mean"] = posture_error.mean().item()
            tracking["posture_error_min"] = posture_error.min().item()
            tracking["posture_error_max"] = posture_error.max().item()

            if valid_height.any():
                base_height_valid = base_height[valid_height]
                base_height_error_valid = torch.abs(base_height_valid - target_height)

                tracking["base_height_mean"] = base_height_valid.mean().item()
                tracking["base_height_median"] = base_height_valid.median().item()
                tracking["base_height_error_mean"] = base_height_error_valid.mean().item()
            else:
                tracking["base_height_mean"] = 0.0
                tracking["base_height_median"] = 0.0
                tracking["base_height_error_mean"] = 0.0

            # keep raw diagnostics separately
            tracking["base_height_raw_min"] = base_height.min().item()
            tracking["base_height_raw_max"] = base_height.max().item()
            tracking["bad_height_frac"] = (~valid_height).float().mean().item()

            tracking["lin_vel_norm_mean"] = lin_vel_norm.mean().item()
            tracking["ang_vel_norm_mean"] = ang_vel_norm.mean().item()

            tracking["foot_contact_count_mean"] = foot_contact_count.mean().item()
            tracking["loaded_foot_count_mean"] = loaded_foot_count.mean().item()
            tracking["non_slipping_foot_count_mean"] = non_slipping_foot_count.mean().item()
            tracking["slipping_foot_count_mean"] = slipping_foot_count.mean().item()

            tracking["stable_contacts_frac"] = stable_contacts.float().mean().item()

            tracking["foot_fz_mean"] = foot_fz.mean().item()
            tracking["foot_xy_vel_mean"] = foot_xy_vel.mean().item()

            # Keep old dashboard-compatible names.
            tracking["base_contact_force_mean"] = nonfoot_contact_force_mean.item()
            tracking["base_contact_env_frac"] = nonfoot_contact_env_frac.item()
            tracking["late_base_contact_frac"] = late_nonfoot_contact_frac.item()

            # Clearer aliases.
            tracking["nonfoot_contact_env_frac"] = nonfoot_contact_env_frac.item()
            tracking["late_nonfoot_contact_frac"] = late_nonfoot_contact_frac.item()

            tracking["late_gate_frac"] = late_gate.float().mean().item()

            tracking["late_foot_contact_count_mean"] = late_foot_contact_count_mean.item()
            tracking["late_loaded_foot_count_mean"] = late_loaded_foot_count_mean.item()
            tracking["late_non_slipping_foot_count_mean"] = (
                late_non_slipping_foot_count_mean.item()
            )
            tracking["late_slipping_foot_count_mean"] = late_slipping_foot_count_mean.item()
            tracking["late_stable_contacts_frac"] = late_stable_contacts_frac.item()

            tracking["handoff_ready_mean"] = handoff_ready.float().mean().item()

            # ------------------------------------------------------------
            # Geometry tracking metrics
            # ------------------------------------------------------------
            tracking["front_sep_mean"] = front_sep.mean().item()
            tracking["rear_sep_mean"] = rear_sep.mean().item()

            tracking["front_sep_min"] = front_sep.min().item()
            tracking["rear_sep_min"] = rear_sep.min().item()

            tracking["front_crossed_frac"] = front_crossed.float().mean().item()
            tracking["rear_crossed_frac"] = rear_crossed.float().mean().item()

            tracking["front_crossed_strict_frac"] = front_crossed_strict.float().mean().item()
            tracking["rear_crossed_strict_frac"] = rear_crossed_strict.float().mean().item()

            tracking["front_x_mean"] = front_x_mean.mean().item()
            tracking["rear_x_mean"] = rear_x_mean.mean().item()

            tracking["front_too_wide_frac"] = front_too_wide.float().mean().item()
            tracking["front_too_forward_frac"] = front_too_forward.float().mean().item()

            tracking["late_front_sep_mean"] = late_scalar_mean(front_sep).item()
            tracking["late_rear_sep_mean"] = late_scalar_mean(rear_sep).item()

            tracking["late_front_x_mean"] = late_scalar_mean(front_x_mean).item()
            tracking["late_rear_x_mean"] = late_scalar_mean(rear_x_mean).item()

            tracking["late_front_crossed_frac"] = late_scalar_mean(front_crossed).item()
            tracking["late_rear_crossed_frac"] = late_scalar_mean(rear_crossed).item()

            tracking["late_front_crossed_strict_frac"] = (
                late_scalar_mean(front_crossed_strict).item()
            )
            tracking["late_rear_crossed_strict_frac"] = (
                late_scalar_mean(rear_crossed_strict).item()
            )

            tracking["late_front_too_wide_frac"] = late_scalar_mean(front_too_wide).item()
            tracking["late_front_too_forward_frac"] = late_scalar_mean(front_too_forward).item()

            tracking["near_crouch_clip_frac"] = getattr(
                self,
                "near_crouch_clip_frac",
                torch.tensor(0.0, device=self.device),
            ).item()

            tracking["near_stand_clip_frac"] = getattr(
                self,
                "near_stand_clip_frac",
                torch.tensor(0.0, device=self.device),
            ).item()

            # ------------------------------------------------------------
            # Per-foot contact/load/non-slip/slip diagnostics
            # ------------------------------------------------------------
            # These answer:
            #   - which foot is touching?
            #   - which foot is carrying load?
            #   - which foot is non-slipping?
            #   - which contacted foot is slipping?
            # ------------------------------------------------------------
            foot_names = ["FL", "FR", "RL", "RR"]

            per_foot_contact_frac = foot_contact.float().mean(dim=0)
            per_foot_loaded_frac = loaded_feet.float().mean(dim=0)
            per_foot_non_slip_frac = non_slipping_feet.float().mean(dim=0)
            per_foot_slipping_contact_frac = slipping_feet.float().mean(dim=0)

            per_foot_fz_mean = foot_fz.mean(dim=0)
            per_foot_xy_vel_mean = foot_xy_vel.mean(dim=0)

            late_per_foot_contact_frac = late_vector_mean(foot_contact)
            late_per_foot_loaded_frac = late_vector_mean(loaded_feet)
            late_per_foot_non_slip_frac = late_vector_mean(non_slipping_feet)
            late_per_foot_slipping_contact_frac = late_vector_mean(slipping_feet)

            late_per_foot_fz_mean = late_vector_mean(foot_fz)
            late_per_foot_xy_vel_mean = late_vector_mean(foot_xy_vel)

            for i, name in enumerate(foot_names):
                # Global per-foot metrics.
                tracking[f"{name}_contact_frac"] = per_foot_contact_frac[i].item()
                tracking[f"{name}_loaded_frac"] = per_foot_loaded_frac[i].item()
                tracking[f"{name}_non_slip_frac"] = per_foot_non_slip_frac[i].item()
                tracking[f"{name}_slipping_contact_frac"] = (
                    per_foot_slipping_contact_frac[i].item()
                )

                tracking[f"{name}_foot_fz_mean"] = per_foot_fz_mean[i].item()
                tracking[f"{name}_foot_xy_vel_mean"] = per_foot_xy_vel_mean[i].item()

                # Late-stage per-foot metrics.
                tracking[f"late_{name}_contact_frac"] = (
                    late_per_foot_contact_frac[i].item()
                )
                tracking[f"late_{name}_loaded_frac"] = (
                    late_per_foot_loaded_frac[i].item()
                )
                tracking[f"late_{name}_non_slip_frac"] = (
                    late_per_foot_non_slip_frac[i].item()
                )
                tracking[f"late_{name}_slipping_contact_frac"] = (
                    late_per_foot_slipping_contact_frac[i].item()
                )

                tracking[f"late_{name}_foot_fz_mean"] = late_per_foot_fz_mean[i].item()
                tracking[f"late_{name}_foot_xy_vel_mean"] = (
                    late_per_foot_xy_vel_mean[i].item()
                )

                # Per-foot body-frame position diagnostics.
                tracking[f"{name}_x_mean"] = x[:, i].mean().item()
                tracking[f"{name}_y_mean"] = y[:, i].mean().item()

                tracking[f"late_{name}_x_mean"] = late_scalar_mean(x[:, i]).item()
                tracking[f"late_{name}_y_mean"] = late_scalar_mean(y[:, i]).item()

            # ------------------------------------------------------------
            # State-distribution diagnostics
            # ------------------------------------------------------------
            tracking["state_fallen_like_frac"] = fallen_like.float().mean().item()
            tracking["state_rollup_or_crouch_frac"] = rollup_or_crouch.float().mean().item()
            tracking["state_near_stand_unstable_frac"] = (
                near_stand_unstable.float().mean().item()
            )
            tracking["state_handoff_candidate_frac"] = handoff_candidate.float().mean().item()

            # ------------------------------------------------------------
            # Reset-source diagnostics, if you have self.reset_source
            # ------------------------------------------------------------
            # Suggested convention:
            #   0 = full fallen reset
            #   1 = partial recovery / roll-up reset
            #   2 = near-stand / final-stand reset
            # reset_fallen_frac
            #    current fraction of the 4096 envs whose latest reset was fallen
            # reset_event_fallen_frac
            #    cumulative fraction of reset events that were fallen
            # ------------------------------------------------------------
            if hasattr(self, "reset_source"):
                reset_source = self.reset_source
                valid = reset_source >= 0
                valid_count = valid.float().sum().clamp_min(1.0)

                tracking["reset_fallen_frac"] = (
                    ((reset_source == 0) & valid).float().sum() / valid_count
                ).item()

                tracking["reset_near_stand_frac"] = (
                    ((reset_source == 1) & valid).float().sum() / valid_count
                ).item()

            if hasattr(self, "reset_source_event_counts"):
                counts = self.reset_source_event_counts.float()
                total_events = counts.sum().clamp_min(1.0)

                tracking["reset_event_fallen_frac"] = (counts[0] / total_events).item()
                tracking["reset_event_near_stand_frac"] = (counts[1] / total_events).item()
                tracking["reset_event_total"] = total_events.item()

    def compute_handoff_ready(self):
        """
        Diagnostic/runtime gate for switching from recovery policy to locomotion policy.

        This should be stricter than the training recovery-success gate.
        Do not use this as the sparse reward gate during training.
        """


        cfg = self.cfg.rewards

        foot_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > cfg.recovery_contact_force_threshold
        )

        foot_xy_vel = torch.norm(
            self.foot_velocities[:, :, :2],
            dim=-1
        )

        non_slipping_feet = foot_contact & (
            foot_xy_vel < 0.10
        )

        posture_error = torch.norm(
            self.dof_pos[:, :self.num_actuated_dof]
            - self.default_dof_pos[:, :self.num_actuated_dof],
            dim=1
        )

        relative_base_height = (
            self.root_states[:, 2]
            - self._get_local_terrain_height()
        )

        handoff_ready = (
                (self.projected_gravity[:, 2] < cfg.recovery_upright_threshold)  # usually -0.90
                & (relative_base_height > 0.30)                                  # stricter than 0.28 success
                & (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.25)
                & (torch.norm(self.base_ang_vel, dim=1) < 1.00)
                & (posture_error < 1.80)
                & (non_slipping_feet.sum(dim=1) >= 3)
            )

        return handoff_ready

    def _write_recovery_grid_snapshot(
            self,
            upright,
            stable_height,
            low_velocity,
            low_ang_vel,
            good_posture,
            stable_contacts,
            recovered,
            stable_recovery,
        ):
            """
            Writes per-environment recovery gate status for dashboard visualization.

            Status code:
                0 = not upright
                1 = upright, but height failed
                2 = upright + height, but velocity failed
                3 = velocity okay, but posture failed
                4 = posture okay, but contact/slip failed
                5 = instant recovered
                6 = stable recovered
            """

            # Write infrequently; do not write every physics step.
            if self.common_step_counter % 50 != 0:
                return

            status = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

            status[~upright] = 0

            status[upright & ~stable_height] = 1

            status[
                upright
                & stable_height
                & ~(low_velocity & low_ang_vel)
            ] = 2

            status[
                upright
                & stable_height
                & low_velocity
                & low_ang_vel
                & ~good_posture
            ] = 3

            status[
                upright
                & stable_height
                & low_velocity
                & low_ang_vel
                & good_posture
                & ~stable_contacts
            ] = 4

            status[recovered] = 5
            status[stable_recovery] = 6

            # Use your run/log directory if available.
            # Replace this with your actual run directory variable if you have one.
            import numpy as np
            from pathlib import Path

            if not hasattr(self, "recovery_grid_path"):
                # fallback: current working directory
                self.recovery_grid_path = Path("recovery_grid.npz")
                print(f"[WARN] recovery_grid_path was not set; using {self.recovery_grid_path.resolve()}")

            np.savez_compressed(
                self.recovery_grid_path,
                step=int(self.common_step_counter),
                status=status.detach().cpu().numpy(),
                upright=upright.detach().cpu().numpy(),
                stable_height=stable_height.detach().cpu().numpy(),
                low_velocity=low_velocity.detach().cpu().numpy(),
                low_ang_vel=low_ang_vel.detach().cpu().numpy(),
                good_posture=good_posture.detach().cpu().numpy(),
                stable_contacts=stable_contacts.detach().cpu().numpy(),
                recovered=recovered.detach().cpu().numpy(),
                stable_recovery=stable_recovery.detach().cpu().numpy(),
            )

    def _write_reset_source_grid_snapshot(self):
        """
        Writes per-env reset-source labels for dashboard visualization.

        Source code:
            -1 = uninitialized
             0 = fallen reset
             1 = terminal / near-stand reset
        """

        if self.common_step_counter % 50 != 0:
            return

        if not hasattr(self, "reset_source"):
            return

        import numpy as np
        from pathlib import Path

        if hasattr(self, "recovery_grid_path"):
            grid_path = Path(self.recovery_grid_path).with_name("reset_source_grid.npz")
        else:
            grid_path = Path("reset_source_grid.npz")

        reset_source = self.reset_source.detach().cpu().numpy()

        if hasattr(self, "terminal_reset_buf"):
            terminal_reset = self.terminal_reset_buf.detach().cpu().numpy()
        else:
            terminal_reset = np.zeros(self.num_envs, dtype=bool)

        if hasattr(self, "reset_source_event_counts"):
            event_counts = self.reset_source_event_counts.detach().cpu().numpy()
        else:
            event_counts = np.zeros(2, dtype=np.int64)

        np.savez_compressed(
            grid_path,
            step=int(self.common_step_counter),
            reset_source=reset_source,
            terminal_reset=terminal_reset,
            event_counts=event_counts,
        )

    def check_recovery_success(self):
        """
        Counts successful fall-recovery events.

        Recovery is detected using hard thresholds from cfg.rewards:
        upright orientation, sufficient body height, low base velocity,
        low angular velocity, standing-like posture, and stable foot contacts.
        The condition must hold for recovery_success_steps consecutive steps.
        """

        cfg = self.cfg.rewards
        g_z = self.projected_gravity[:, 2]

        upright = g_z < cfg.recovery_upright_threshold

        local_terrain_height = self._get_local_terrain_height()
        relative_base_height = (
            self.root_states[:, 2] - local_terrain_height
        )
        stable_height = (
            relative_base_height > cfg.recovery_height_success
        )

        # Horizontal velocity
        low_velocity = (
            torch.norm(self.base_lin_vel[:, :2], dim=1)
            < cfg.recovery_lin_vel_threshold
        )

        # Prefer world-frame vertical velocity.
        low_vertical_velocity = (
            torch.abs(self.root_states[:, 9])
            < cfg.recovery_vertical_vel_threshold
        )

        low_ang_vel = (
            torch.norm(self.base_ang_vel, dim=1)
            < cfg.recovery_ang_vel_threshold
        )

        posture_error = torch.norm(
            self.dof_pos - self.default_dof_pos,
            dim=1,
        )
        good_posture = (
            posture_error < cfg.recovery_posture_threshold
        )

        foot_contact = (
            self.contact_forces[:, self.feet_indices, 2]
            > cfg.recovery_contact_force_threshold
        )

        foot_xy_vel = torch.norm(
            self.foot_velocities[:, :, :2],
            dim=-1,
        )

        non_slipping_feet = foot_contact & (
            foot_xy_vel < cfg.recovery_foot_slip_vel_threshold
        )

        if getattr(cfg, "require_non_slipping_contacts", False):
            stable_contacts = (
                non_slipping_feet.sum(dim=1)
                >= cfg.recovery_min_foot_contacts
            )
        else:
            stable_contacts = (
                foot_contact.sum(dim=1)
                >= cfg.recovery_min_foot_contacts
            )

        # Reject recovery supported by trunk, hips, thighs or calves.
        if self.base_contact_indices.numel() > 0:
            nonfoot_force = torch.norm(
                self.contact_forces[:, self.base_contact_indices, :],
                dim=-1,
            )

            no_nonfoot_contact = (
                nonfoot_force.max(dim=1).values
                < cfg.recovery_nonfoot_contact_threshold
            )
        else:
            no_nonfoot_contact = torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        recovered = (
            upright
            & stable_height
            & low_velocity
            & low_vertical_velocity
            & low_ang_vel
            & good_posture
            & stable_contacts
            & no_nonfoot_contact
        )

        self.recovery_counter[recovered] += 1
        self.recovery_counter[~recovered] = 0

        stable_recovery = (self.recovery_counter >= cfg.recovery_success_steps)

        # Diagnostic / runtime handoff readiness
        handoff_ready = self.compute_handoff_ready()

        if hasattr(self, "terminal_reset_buf"):
            terminal_src = self.terminal_reset_buf.bool()
            fallen_src = ~terminal_src

            dbg = self.extras.setdefault("recovery_debug", {})

            if terminal_src.any():
                dbg["terminal_src_upright"] = upright[terminal_src].float().mean().item()
                dbg["terminal_src_height"] = stable_height[terminal_src].float().mean().item()
                dbg["terminal_src_low_vel"] = low_velocity[terminal_src].float().mean().item()
                dbg["terminal_src_low_ang_vel"] = low_ang_vel[terminal_src].float().mean().item()
                dbg["terminal_src_posture"] = good_posture[terminal_src].float().mean().item()
                dbg["terminal_src_stable_contacts"] = stable_contacts[terminal_src].float().mean().item()
                dbg["terminal_src_recovered"] = recovered[terminal_src].float().mean().item()
                dbg["terminal_src_handoff"] = handoff_ready[terminal_src].float().mean().item()

            if fallen_src.any():
                dbg["fallen_src_upright"] = upright[fallen_src].float().mean().item()
                dbg["fallen_src_height"] = stable_height[fallen_src].float().mean().item()
                dbg["fallen_src_low_vel"] = low_velocity[fallen_src].float().mean().item()
                dbg["fallen_src_low_ang_vel"] = low_ang_vel[fallen_src].float().mean().item()
                dbg["fallen_src_posture"] = good_posture[fallen_src].float().mean().item()
                dbg["fallen_src_stable_contacts"] = stable_contacts[fallen_src].float().mean().item()
                dbg["fallen_src_recovered"] = recovered[fallen_src].float().mean().item()
                dbg["fallen_src_handoff"] = handoff_ready[fallen_src].float().mean().item()

        self.log_recovery_tracking_metrics(
            posture_error=posture_error,
            foot_contact=foot_contact,
            non_slipping_feet=non_slipping_feet,
            handoff_ready=handoff_ready,
        )

        self.rollout_upright |= upright
        # Records whether each environment achieved a stable recovery during its episode
        self.rollout_recovered |= stable_recovery

        self.rollout_max_counter = torch.maximum(self.rollout_max_counter, self.recovery_counter.float())

        if not hasattr(self, "extras"):
            self.extras = {}

        if "recovery_debug" not in self.extras:
            self.extras["recovery_debug"] = {}

        self.extras["recovery_debug"]["recovery_rate_ema"] = self.recovery_rate_ema
        self.extras["recovery_debug"]["rollout_upright"] = self.rollout_upright.float().mean().item()
        self.extras["recovery_debug"]["rollout_recovered"] = self.rollout_recovered.float().mean().item()
        self.extras["recovery_debug"]["rollout_max_counter_mean"] = self.rollout_max_counter.mean().item()
        self.extras["recovery_debug"]["rollout_max_counter_max"] = self.rollout_max_counter.max().item()
        self.extras["recovery_debug"]["handoff_ready"] = handoff_ready.float().mean().item()
        self.extras["recovery_debug"]["upright"] = upright.float().mean().item()
        self.extras["recovery_debug"]["stable_height"] = stable_height.float().mean().item()
        self.extras["recovery_debug"]["low_velocity"] = low_velocity.float().mean().item()
        self.extras["recovery_debug"]["low_ang_vel"] = low_ang_vel.float().mean().item()
        self.extras["recovery_debug"]["good_posture"] = good_posture.float().mean().item()
        self.extras["recovery_debug"]["stable_contacts"] = stable_contacts.float().mean().item()
        self.extras["recovery_debug"]["recovered"] = recovered.float().mean().item()
        self.extras["recovery_debug"]["stable_recovery"] = stable_recovery.float().mean().item()
        self.extras["recovery_debug"]["recovery_counter_mean"] = self.recovery_counter.float().mean().item()
        self.extras["recovery_debug"]["recovery_counter_max"] = self.recovery_counter.max().float().item()

        # Orientation is now an evaluation/training bin, not a global
        # promote/demote curriculum phase.  The only promoted dimension is
        # terrain difficulty, represented by one frontier per terrain column.
        if hasattr(self, "terrain_sampling_stage"):
            stages = self.terrain_sampling_stage.float()
            self.extras["recovery_debug"]["sampling_stage_mean"] = (
                stages.mean().item()
            )
            self.extras["recovery_debug"]["sampling_stage_bootstrap_frac"] = (
                (stages == 0).float().mean().item()
            )
            self.extras["recovery_debug"]["sampling_stage_developing_frac"] = (
                (stages == 1).float().mean().item()
            )
            self.extras["recovery_debug"]["sampling_stage_mature_frac"] = (
                (stages == 2).float().mean().item()
            )

        self.extras["recovery_debug"]["recovery_rate_ema"] = (
            self.recovery_rate_ema
        )

        new_recovery = stable_recovery & (~self.recovered_flag)

        self.recovery_bonus_buf[new_recovery] = 1.0
        self.recovered_flag |= stable_recovery

        # self.episode_sums["recovery_success"] += new_recovery.float()
        self.extras["recovery_debug"]["new_recovery_event"] = new_recovery.float().mean().item()

        self._write_recovery_grid_snapshot(
            upright=upright,
            stable_height=stable_height,
            low_velocity=low_velocity,
            low_ang_vel=low_ang_vel,
            good_posture=good_posture,
            stable_contacts=stable_contacts,
            recovered=recovered,
            stable_recovery=stable_recovery,
        )

        self._write_reset_source_grid_snapshot()
        self._write_curriculum_grid_snapshot()


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # Must run before rollout_recovered, timeout and episode buffers
        # are cleared and before the root state is moved.
        if self.cfg.env.train_recovery:
            self._update_recovery_curricula(env_ids)

            # Select the terrain level, sampling role, and orientation bin for
            # the new episode.  This must happen after the old episode outcome
            # is consumed and before the new root state is initialized.
            self._sample_recovery_curriculum_tasks(env_ids)

        # Reset DOFs
        self._call_train_eval(self._reset_dofs, env_ids)

        # Reset ROOT STATE
        # if self.cfg.env.train_recovery:
        #     self._call_train_eval(self._reset_root_states_fall_recovery, env_ids)
        # else:
        #     self._call_train_eval(self._reset_root_states, env_ids)

        if self.cfg.env.train_recovery:
            self._call_train_eval(self._reset_root_states_fall_recovery, env_ids)
        else:
            self._call_train_eval(self._reset_root_states, env_ids)

        # Domain randomization
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        # Commands
        if self.cfg.env.train_recovery:
            self.commands[env_ids] = 0.0
            self.recovery_counter[env_ids] = 0.0
            self.recovered_flag[env_ids] = False
            self.rollout_upright[env_ids] = False
            self.rollout_recovered[env_ids] = False
            self.rollout_max_counter[env_ids] = 0
            self.last_torques[env_ids] = 0.   # prevents torque penalty spike at step 1
            self.torques[env_ids] = 0.
        else:
            self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            # for key in self.episode_sums.keys():
            #     self.extras["train/episode"]['rew_' + key] = torch.mean(
            #         self.episode_sums[key][train_env_ids])
            #     self.episode_sums[key][train_env_ids] = 0.
            #
            # for key in self.episode_sums.keys():
            #     if key == "recovery_success":
            #         self.extras["train/episode"]["recovery_success"] = torch.mean(
            #             self.episode_sums[key][train_env_ids]
            #         )
            #     else:
            #         self.extras["train/episode"]['rew_' + key] = torch.mean(
            #             self.episode_sums[key][train_env_ids]
            #         )
            #     self.episode_sums[key][train_env_ids] = 0.
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids]
                )
                self.episode_sums[key][train_env_ids] = 0.
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                # save the evaluation rollout result if not already saved
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())
        if self.cfg.commands.command_curriculum:
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]
            self.extras["train/episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            self.extras["train/episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            self.extras["train/episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            self.extras["train/episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            self.extras["train/episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            self.extras["train/episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            self.extras["train/episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            self.extras["train/episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            self.extras["train/episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
            self.extras["train/episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
            self.extras["train/episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            self.extras["train/episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            self.extras["train/episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            self.extras["train/episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            self.extras["train/episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            self.extras["train/episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            if self.cfg.commands.num_commands > 9:
                self.extras["train/episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                self.extras["train/episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
                                                                           curriculum.weights.shape[0]

            self.extras["train/episode"]["min_action"] = torch.min(self.actions)
            self.extras["train/episode"]["max_action"] = torch.max(self.actions)

            self.extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

        self._debug_log_terminal_reset_state(env_ids)

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # base position
        self.root_states[env_ids] = base_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _debug_log_terminal_reset_state(self, env_ids):
        if not getattr(self.cfg.env, "debug_log_terminal_reset", False):
            return

        if not hasattr(self, "terminal_reset_buf"):
            print("[TERMINAL RESET DEBUG] terminal_reset_buf does not exist")
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()

        # Safety: keep only valid env ids
        valid_env_mask = (env_ids >= 0) & (env_ids < self.num_envs)
        env_ids = env_ids[valid_env_mask]

        if env_ids.numel() == 0:
            return

        terminal_mask = self.terminal_reset_buf[env_ids].bool().flatten()
        terminal_env_ids = env_ids[terminal_mask]

        if terminal_env_ids.numel() == 0:
            print("[TERMINAL RESET] no terminal envs in this reset batch")
            return

        # Extra safety against bad env ids
        valid_terminal = (
            (terminal_env_ids >= 0)
            & (terminal_env_ids < self.root_states.shape[0])
            & (terminal_env_ids < self.dof_pos.shape[0])
            & (terminal_env_ids < self.dof_vel.shape[0])
        )
        terminal_env_ids = terminal_env_ids[valid_terminal]

        if terminal_env_ids.numel() == 0:
            print("[TERMINAL RESET DEBUG] terminal_env_ids invalid after filtering")
            return

        q = self.dof_pos[terminal_env_ids, :self.num_actuated_dof]

        # Correct handling of default_dof_pos shape
        if hasattr(self, "_get_default_dof_pos_for_env_ids"):
            q_default = self._get_default_dof_pos_for_env_ids(
                terminal_env_ids
            )[:, :self.num_actuated_dof]
        else:
            default = self.default_dof_pos

            if default.dim() == 1:
                q_default = default[:self.num_actuated_dof].unsqueeze(0).expand_as(q)
            elif default.shape[0] == 1:
                q_default = default[:, :self.num_actuated_dof].expand_as(q)
            else:
                q_default = default[terminal_env_ids, :self.num_actuated_dof]

        terminal_posture_error = torch.norm(q - q_default, dim=1)

        print(
            "[TERMINAL RESET]",
            "n", int(terminal_env_ids.numel()),
            "z mean", self.root_states[terminal_env_ids, 2].mean().item(),
            "z min", self.root_states[terminal_env_ids, 2].min().item(),
            "z max", self.root_states[terminal_env_ids, 2].max().item(),
            "posture err mean", terminal_posture_error.mean().item(),
            "posture err max", terminal_posture_error.max().item(),
            "dof vel max", self.dof_vel[terminal_env_ids, :self.num_actuated_dof].abs().max().item(),
            "lin vel max", self.root_states[terminal_env_ids, 7:10].norm(dim=1).max().item(),
            "ang vel max", self.root_states[terminal_env_ids, 10:13].norm(dim=1).max().item(),
        )

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]

            raw_rew = self.reward_functions[i]()              # RAW
            scaled_rew = raw_rew * self.reward_scales[name]   # SCALED

            self.rew_buf += scaled_rew

            if torch.sum(scaled_rew) >= 0:
                self.rew_buf_pos += scaled_rew
            elif torch.sum(scaled_rew) <= 0:
                self.rew_buf_neg += scaled_rew

            self.episode_sums[name] += scaled_rew

            # DEBUG LOGGING
            # if self.debug_rewards and self.debug_counter % 500 == 0:
            #     print(f"{name}: raw_mean={raw_rew.mean().item():.2e}, raw_max={raw_rew.max().item():.2e}, "
            #           f"scaled_mean={scaled_rew.mean().item():.2e}, scaled_max={scaled_rew.max().item():.2e}")

            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + scaled_rew
            else:
                self.command_sums[name] += scaled_rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)

        # self.rew_buf *= 1000.0
        self.episode_sums["total"] += self.rew_buf
        # print(f"Episodic sum: {self.episode_sums['total']}")

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += scaled_rew
            self.command_sums["termination"] += scaled_rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

        self.debug_counter += 1

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                             :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                                 :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        # if self.cfg.env.observe_desired_contact_states:
        #     self.obs_buf = torch.cat((self.obs_buf,
        #                               self.desired_contact_states), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # print(f"obs shape: {self.obs_buf.shape}")

        # build privileged obs

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_link_masses:

            # [N, 4]
            # [trunk_mass_norm, hip_mass_norm, thigh_mass_norm, calf_mass_norm]
            mass_groups_norm = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

            # 0) TRUNK / BASE MASS
            # Current absolute trunk mass
            trunk_mass = self.mass_groups[:, 0]

            trunk_range = torch.tensor(self.cfg.normalization.trunk_mass_range, device=self.device, dtype=torch.float)

            trunk_min = trunk_range[0]
            trunk_max = trunk_range[1]

            trunk_shift = 0.5 * (trunk_max + trunk_min)
            trunk_scale = 2.0 / (trunk_max - trunk_min + 1e-8)

            mass_groups_norm[:, 0] = (trunk_mass - trunk_shift) * trunk_scale

            # 1..3) HIP / THIGH / CALF
            # Use ABSOLUTE masses
            link_masses = self.mass_groups[:, 1:]   # [N, 3]

            # [[hip_min, hip_max],
            # [thigh_min, thigh_max],
            # [calf_min, calf_max]]
            lm_ranges = torch.tensor(self.cfg.normalization.link_mass_ranges, device=self.device, dtype=torch.float)

            lm_min = lm_ranges[:, 0]
            lm_max = lm_ranges[:, 1]

            lm_shift = 0.5 * (lm_max + lm_min)
            lm_scale = 2.0 / (lm_max - lm_min + 1e-8)

            mass_groups_norm[:, 1:] = (link_masses - lm_shift) * lm_scale

            # DEBUG
            # if self.common_step_counter % 100 == 0:

            #     print("\n===== LINK MASS DEBUG =====")

            #     for env_id in range(min(5, self.num_envs)):

            #         print(f"\nENV {env_id}")

            #         print("Current groups:")
            #         print(self.mass_groups[env_id])

            #         print("Normalized:")
            #         print(mass_groups_norm[env_id])

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, mass_groups_norm), dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, mass_groups_norm),dim=1)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        # if self.cfg.env.priv_observe_ground_friction:
        #     self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
        #     ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
        #         self.cfg.normalization.ground_friction_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.ground_friction_coeffs.unsqueeze(
        #                                              1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                         dim=1)
        #     self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
        #                                               (self.ground_friction_coeffs.unsqueeze(
        #                                               1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                               dim=1)

        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)

        if self.cfg.env.priv_observe_com_displacement:
            # randomized COM offset parameter injected into simulation
            com_displacements_scale, com_displacements_shift = get_scale_shift(self.cfg.normalization.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)

        if self.cfg.env.priv_observe_com_position:
            # center of mass position relative to the base frame
            body_positions_world = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, :, :3]
            body_masses = self.rigid_body_masses.unsqueeze(-1)
            # body_masses = self.body_masses.unsqueeze(0).unsqueeze(-1)
            total_mass = body_masses.sum(dim=1).clamp_min(1e-8)
            com_world = (body_positions_world * body_masses).sum(dim=1) / total_mass
            com_in_base = quat_rotate_inverse(self.base_quat, com_world - self.base_pos)
            com_position_scale, com_position_shift = get_scale_shift(self.cfg.normalization.com_position_range)
            normalized_com_position = (com_in_base - com_position_shift) * com_position_scale
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, normalized_com_position), dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, normalized_com_position), dim=1)


        # if self.cfg.env.priv_observe_Kp_factor:
        #     Kp_factor_scale, Kp_factor_shift = get_scale_shift(self.cfg.normalization.Kp_factor_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.Kp_factors - Kp_factor_shift) * Kp_factor_scale),
        #                                         dim=1)
        #     self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
        #                                               (self.Kp_factors - Kp_factor_shift) * Kp_factor_scale),
        #                                              dim=1)
        # if self.cfg.env.priv_observe_Kd_factor:
        #     Kd_factor_scale, Kd_factor_shift = get_scale_shift(self.cfg.normalization.Kd_factor_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.Kd_factors - Kd_factor_shift) * Kd_factor_scale),
        #                                         dim=1)
        #     self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
        #                                               (self.Kd_factors - Kd_factor_shift) * Kd_factor_scale),
        #                                              dim=1)

        if self.cfg.env.priv_observe_Kpd_factor:
            # Current implementation randomizes:
            # - one shared Kp factor per environment
            # - one shared Kd factor per environment
            #
            # Although tensors are shape [N, 12], all joint values are identical.
            # Therefore expose only 2 privileged dimensions:
            # [normalized_Kp_factor, normalized_Kd_factor]

            Kp_factor_scale, Kp_factor_shift = get_scale_shift(self.cfg.normalization.Kp_factor_range)
            Kd_factor_scale, Kd_factor_shift = get_scale_shift(self.cfg.normalization.Kd_factor_range)
            normalized_Kp_factor = (self.Kp_factors[:, 0:1] - Kp_factor_shift) * Kp_factor_scale
            normalized_Kd_factor = (self.Kd_factors[:, 0:1] - Kd_factor_shift) * Kd_factor_scale
            normalized_Kpd_factors = torch.cat((normalized_Kp_factor, normalized_Kd_factor), dim=1)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, normalized_Kpd_factors), dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, normalized_Kpd_factors), dim=1)

        if self.cfg.env.priv_observe_contact_forces:
            contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
            foot_contact_forces = self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (foot_contact_forces - contact_forces_shift) * contact_forces_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (foot_contact_forces - contact_forces_shift) * contact_forces_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_contact_states:
            foot_contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()  # [num_envs, 4]
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, foot_contact_states), dim=-1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, foot_contact_states), dim=-1)

        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        if self.cfg.env.priv_observe_heightmap:
            heights = self.measured_heights
            body_height = self.root_states[:, 2:3]

            # Unclamped base-to-terrain vertical clearance.
            heights_relative_raw = body_height - heights

            height_min, height_max = self.cfg.normalization.relative_height_range

            # Log before clamping so range violations remain visible.
            if self.common_step_counter % 50 == 0:
                tracking = self.extras.setdefault("recovery_tracking", {})

                outside_range = (
                    (heights_relative_raw < height_min)
                    | (heights_relative_raw > height_max)
                )

                tracking["heightmap_relative_min"] = heights_relative_raw.min().item()
                tracking["heightmap_relative_mean"] = heights_relative_raw.mean().item()
                tracking["heightmap_relative_max"] = heights_relative_raw.max().item()
                tracking["heightmap_outside_range_frac"] = outside_range.float().mean().item()

            heights_relative = torch.clamp(
                heights_relative_raw,
                height_min,
                height_max,
            )

            height_scale, height_shift = get_scale_shift(self.cfg.normalization.relative_height_range)
            heights_normalized = (heights_relative - height_shift) * height_scale

            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights_normalized), dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, heights_normalized), dim=1)

        # print(f"priv obs shape: {self.privileged_obs_buf.shape}")
        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()


    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target) \

    def set_main_agent_pose(self, loc, quat):
        self.root_states[0, 0:3] = torch.Tensor(loc)
        self.root_states[0, 3:7] = torch.Tensor(quat)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _call_train_eval(self, func, env_ids):

        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None: ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload

        if cfg.domain_rand.randomize_link_masses:

            self.trunk_masses[env_ids] = torch.rand(
                len(env_ids),
                device=self.device
            ) * (
                cfg.domain_rand.trunk_mass_range[1]
                - cfg.domain_rand.trunk_mass_range[0]
            ) + cfg.domain_rand.trunk_mass_range[0]

            self.hip_masses[env_ids] = torch.rand(
                len(env_ids),
                device=self.device
            ) * (
                cfg.domain_rand.hip_mass_range[1]
                - cfg.domain_rand.hip_mass_range[0]
            ) + cfg.domain_rand.hip_mass_range[0]

            self.thigh_masses[env_ids] = torch.rand(
                len(env_ids),
                device=self.device
            ) * (
                cfg.domain_rand.thigh_mass_range[1]
                - cfg.domain_rand.thigh_mass_range[0]
            ) + cfg.domain_rand.thigh_mass_range[0]

            self.calf_masses[env_ids] = torch.rand(
                len(env_ids),
                device=self.device
            ) * (
                cfg.domain_rand.calf_mass_range[1]
                - cfg.domain_rand.calf_mass_range[0]
            ) + cfg.domain_rand.calf_mass_range[0]

        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
            # print(f"Randomized Kp_factors: {self.Kp_factors}")

        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor
            # print(f"Randomized Kd_factors: {self.Kd_factors}")

    # def _process_rigid_body_props(self, props, env_id):
    #     self.default_body_mass = props[0].mass

    #     props[0].mass = self.default_body_mass + self.payloads[env_id]
    #     props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
    #                                self.com_displacements[env_id, 2])
    #     # Use per-env rigid body masses for CoM privileged feature
    #     self.rigid_body_masses[env_id] = torch.tensor([prop.mass for prop in props], dtype=torch.float,
    #                                                   device=self.device)
    #     return props

    def _process_rigid_body_props(self, props, env_id):

        # Store nominal/default masses ONCE
        if env_id == 0:

            nominal_body, nominal_hip, nominal_thigh, nominal_calf = \
                    self._compute_mass_groups_from_body_props(props)

            self.mass_groups_default = torch.tensor(
                [
                    nominal_body,
                    nominal_hip,
                    nominal_thigh,
                    nominal_calf
                ],
                device=self.device,
                dtype=torch.float
            )

            print("\n===== NOMINAL MASS GROUPS =====")
            print(self.mass_groups_default)

            print(f"Body Names: {self.body_names}")

            self.default_body_mass = props[0].mass

            # self.default_rigid_body_masses = torch.tensor(
            #     [p.mass for p in props],
            #     device=self.device,
            #     dtype=torch.float
            # )

        # Base / payload randomization
        payload = self.payloads[env_id].item()

        # Start from nominal base mass
        base_mass = self.default_body_mass + payload

        # Optional trunk mass override
        if self.cfg.domain_rand.randomize_link_masses:
            base_mass = self.trunk_masses[env_id].item() + payload

        props[0].mass = base_mass

        # COM randomization
        props[0].com = gymapi.Vec3(
            self.com_displacements[env_id, 0].item(),
            self.com_displacements[env_id, 1].item(),
            self.com_displacements[env_id, 2].item()
        )

        # Link mass randomization
        if self.cfg.domain_rand.randomize_link_masses:

            hip_mass = self.hip_masses[env_id].item()
            thigh_mass = self.thigh_masses[env_id].item()
            calf_mass = self.calf_masses[env_id].item()

            for i, p in enumerate(props):

                body_name = self.body_names[i].lower()

                # Skip base/trunk
                if "base" in body_name or "trunk" in body_name:
                    continue

                if "hip" in body_name:
                    p.mass = hip_mass

                elif "thigh" in body_name:
                    p.mass = thigh_mass

                elif "calf" in body_name:
                    p.mass = calf_mass

        # Store current rigid body masses
        self.rigid_body_masses[env_id] = torch.tensor(
            [p.mass for p in props],
            device=self.device,
            dtype=torch.float
        )


        # Compute grouped masses
        m_body, m_hip, m_thigh, m_calf = \
            self._compute_mass_groups_from_body_props(props)

        self.mass_groups[env_id, 0] = m_body
        self.mass_groups[env_id, 1] = m_hip
        self.mass_groups[env_id, 2] = m_thigh
        self.mass_groups[env_id, 3] = m_calf

        # DEBUG
        # if env_id < 3:

        #     print(f"\n===== ENV {env_id} MASS DEBUG =====")

        #     print("\n--- BASE ---")

        #     print(f"Default base mass : {self.default_body_mass:.4f}")

        #     print(f"Payload           : {payload:.4f}")

        #     if self.cfg.domain_rand.randomize_link_masses:
        #         print(f"Sampled trunk mass: {self.trunk_masses[env_id].item():.4f}")

        #     print(f"Final base mass   : {props[0].mass:.4f}")

        #     print("\n--- LINKS ---")

        #     if self.cfg.domain_rand.randomize_link_masses:

        #         print(f"Hip mass   : {hip_mass:.4f}")
        #         print(f"Thigh mass : {thigh_mass:.4f}")
        #         print(f"Calf mass  : {calf_mass:.4f}")

        #     print("\n--- GROUP MASSES ---")

        #     print(f"Body  group: {m_body:.4f}")
        #     print(f"Hip   group: {m_hip:.4f}")
        #     print(f"Thigh group: {m_thigh:.4f}")
        #     print(f"Calf  group: {m_calf:.4f}")

        #     print("\n--- ALL BODY MASSES ---")

        #     for i, p in enumerate(props):
        #         print(f"{i:02d} | {self.body_names[i]:20s} | {p.mass:.4f}")

        return props


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        if not self.cfg.env.train_recovery:
            self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)

        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        if not self.cfg.env.train_recovery:
            self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]

            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # def _reset_dofs(self, env_ids, cfg):
    #     """ Resets DOF position and velocities of selected environmments
    #     Positions are randomly selected within 0.5:1.5 x default positions.
    #     Velocities are set to zero.

    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """

    #     if self.cfg.env.train_recovery:
    #         self.dof_pos[env_ids] = self.default_dof_pos + 0.5 * torch.randn(len(env_ids), self.num_dof, device=self.device)
    #         self.dof_vel[env_ids] = 0.5 * torch.randn(len(env_ids), self.num_dof, device=self.device)
    #         # self.dof_vel[env_ids] = 0.
    #     else:
    #         self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
    #                                                                         device=self.device)
    #         self.dof_vel[env_ids] = 0.

    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                           gymtorch.unwrap_tensor(self.dof_state),
    #                                           gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs(self, env_ids, cfg):
        if self.cfg.env.train_recovery:
            noise = 0.1 * torch.randn(len(env_ids), self.num_dof, device=self.device)
            self.dof_pos[env_ids] = torch.clamp(self.default_dof_pos + noise, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            self.dof_vel[env_ids] = 0.
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    # def _reset_root_states_fall_recovery(self, env_ids, cfg):
    #     num_resets = len(env_ids)

    #     # 1. XY position from terrain origins
    #     self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2]

    #     # 2. Orientation curriculum
    #     progress = torch.clamp(
    #         torch.tensor(self.common_step_counter, device=self.device) / 5e6,
    #         0.0, 1.0
    #     )

    #     min_roll  = 0.6 + 0.4 * progress
    #     min_pitch = 0.4 + 0.4 * progress
    #     max_roll  = 0.8 + 1.2 * progress
    #     max_pitch = 0.5 + 0.8 * progress

    #     mag_roll  = min_roll  + (max_roll  - min_roll)  * torch.rand(num_resets, device=self.device) ** 0.5
    #     mag_pitch = min_pitch + (max_pitch - min_pitch) * torch.rand(num_resets, device=self.device) ** 0.5

    #     sign_roll  = torch.where(torch.rand(num_resets, device=self.device) > 0.5,  1.0, -1.0)
    #     sign_pitch = torch.where(torch.rand(num_resets, device=self.device) > 0.5,  1.0, -1.0)

    #     roll  = sign_roll  * mag_roll
    #     pitch = sign_pitch * mag_pitch
    #     yaw   = torch.rand(num_resets, device=self.device) * 2 * torch.pi - torch.pi

    #     quat    = quat_from_euler_xyz(roll, pitch, yaw)
    #     gravity = self.gravity_vec[env_ids]
    #     g       = quat_rotate_inverse(quat, gravity)

    #     # Reject near-upright (gz < -0.1) AND fully supine (gz > 0.8)
    #     # Keep sideways + partial falls: gz in [-0.1, 0.8]
    #     # mask = (g[:, 2] < -0.1) | (g[:, 2] > 0.8)
    #     # Use rollout_recovered as curriculum signal (already tracked in your debug)
    #     # recovery_rate = self.rollout_recovered.float().mean().item()
    #     recovery_rate = self.recovery_rate_ema

    #     if recovery_rate < 0.25:
    #         # Phase 1: supine only - easiest, symmetric
    #         mask = g[:, 2] < 0.1 #0.3
    #     elif recovery_rate < 0.55:
    #         # Phase 2: all fallen states including sideways
    #         mask = g[:, 2] < -0.1
    #     else:
    #         # Phase 3: exclude easy supine, force harder sideways
    #         mask = (g[:, 2] < -0.1) | (g[:, 2] > 0.8)

    #     max_iters = 20
    #     for _ in range(max_iters):
    #         if not mask.any():
    #             break
    #         idx = mask.nonzero(as_tuple=False).flatten()
    #         roll[idx]  = (torch.rand(len(idx), device=self.device) * 2 - 1) * torch.pi
    #         pitch[idx] = (torch.rand(len(idx), device=self.device) * 2 - 1) * torch.pi
    #         quat[idx]  = quat_from_euler_xyz(roll[idx], pitch[idx], yaw[idx])
    #         g[idx]     = quat_rotate_inverse(quat[idx], gravity[idx])
    #         # mask       = (g[:, 2] < -0.1) | (g[:, 2] > 0.8)
    #         if recovery_rate < 0.25:
    #             # Phase 1: supine only — easiest, symmetric
    #             mask = g[:, 2] < 0.3
    #         elif recovery_rate < 0.55:
    #             # Phase 2: all fallen states including sideways
    #             mask = g[:, 2] < -0.1
    #         else:
    #             # Phase 3: exclude easy supine, force harder sideways
    #             mask = (g[:, 2] < -0.1) | (g[:, 2] > 0.8)

    #     # 3. Height: low enough to land quickly, varied
    #     # Robot-aware spawn height
    #     if self.cfg.env.robot == "go1":
    #         self.root_states[env_ids, 2] = 0.15 + 0.06 * torch.rand(num_resets, device=self.device)
    #     else:  # aliengo
    #         self.root_states[env_ids, 2] = 0.18 + 0.08 * torch.rand(num_resets, device=self.device)

    #     # 4. Orientation
    #     self.root_states[env_ids, 3:7] = quat

    #     # 5. Velocities — small linear only, NO angular at reset
    #     vel = torch.zeros(num_resets, 6, device=self.device)
    #     vel[:, 0:3] = 0.05 * torch.randn(num_resets, 3, device=self.device)
    #     vel[:, 2]   = torch.clamp(vel[:, 2], -0.1, 0.0)
    #     # vel[:, 3:6] = 0  ← keep zero, don't add angular vel at reset
    #     self.root_states[env_ids, 7:13] = vel

    #     # 6. Push to sim
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)

    #     self.gym.set_dof_state_tensor_indexed(
    #         self.sim,
    #         gymtorch.unwrap_tensor(self.dof_state),
    #         gymtorch.unwrap_tensor(env_ids_int32),
    #         len(env_ids_int32)
    #     )

    #     self.gym.set_actor_root_state_tensor_indexed(
    #         self.sim,
    #         gymtorch.unwrap_tensor(self.root_states),
    #         gymtorch.unwrap_tensor(env_ids_int32),
    #         len(env_ids_int32)
    #     )

    #     # 7. Settle — enough steps for body to land and stop bouncing
    #     #    dt=0.005s × 25 steps = 125ms — sufficient for ~0.2m drop
    #     zero_torques = torch.zeros_like(self.torques)
    #     settle_steps = 20 if self.cfg.env.robot == "go1" else 25
    #     for _ in range(settle_steps):
    #         self.gym.set_dof_actuation_force_tensor(
    #             self.sim, gymtorch.unwrap_tensor(zero_torques)
    #         )
    #         self.gym.simulate(self.sim)
    #         self.gym.fetch_results(self.sim, True)

    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     self.gym.refresh_net_contact_force_tensor(self.sim)
    #     self.gym.refresh_dof_state_tensor(self.sim)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)

    #     # 8. Debug — recompute gz from fresh quat
    #     fresh_quat = self.root_states[env_ids, 3:7]
    #     g_post     = quat_rotate_inverse(fresh_quat, self.gravity_vec[env_ids])
    #     gz         = g_post[:, 2]

    #     sideways = ((gz > -0.7) & (gz < 0.3)).float().mean()
    #     fallen   = (gz >= 0.3).float().mean()
    #     upright  = (gz < -0.7).float().mean()
    #     print(f"[POST-SETTLE] upright={upright:.2f} sideways={sideways:.2f} fallen={fallen:.2f} | "
    #           f"gz min={gz.min():.3f} mean={gz.mean():.3f} max={gz.max():.3f}")

    #     if cfg.env.record_video and 0 in env_ids:
    #         if self.complete_video_frames is None:
    #             self.complete_video_frames = []
    #         else:
    #             self.complete_video_frames = self.video_frames[:]
    #         self.video_frames = []

    #     if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
    #         if self.complete_video_frames_eval is None:
    #             self.complete_video_frames_eval = []
    #         else:
    #             self.complete_video_frames_eval = self.video_frames_eval[:]
    #         self.video_frames_eval = []




    def _reset_root_states_fall_recovery_fallen(self, env_ids, cfg):
        if len(env_ids) == 0:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        num_resets = len(env_ids)

        # ------------------------------------------------------------
        # XY position
        # ------------------------------------------------------------
        self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2]

        # ------------------------------------------------------------
        # Sample candidate orientations for the episode's fixed bin.
        # A normalized Gaussian quaternion is uniform on SO(3); sampling
        # roll/pitch/yaw independently would bias the orientation density.
        # ------------------------------------------------------------
        orientation_bins = self.episode_orientation_bin[env_ids]
        quat = torch.randn(
            num_resets,
            4,
            device=self.device,
            dtype=self.root_states.dtype,
        )
        quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp_min(1e-8)
        gravity = self.gravity_vec[env_ids]
        projected_gravity = quat_rotate_inverse(quat, gravity)
        reject = self._get_orientation_rejection_mask(
            projected_gravity[:, 2],
            orientation_bins,
        )

        # ------------------------------------------------------------
        # Rejection sampling
        # ------------------------------------------------------------
        max_iterations = 32

        for _ in range(max_iterations):
            if not reject.any():
                break

            idx = reject.nonzero(as_tuple=False).flatten()
            count = idx.numel()
            new_quat = torch.randn(
                count,
                4,
                device=self.device,
                dtype=self.root_states.dtype,
            )
            new_quat = new_quat / torch.norm(
                new_quat,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            quat[idx] = new_quat
            projected_gravity[idx] = quat_rotate_inverse(quat[idx], gravity[idx])
            reject = self._get_orientation_rejection_mask(
                projected_gravity[:, 2],
                orientation_bins,
            )

        # ------------------------------------------------------------
        # Deterministic fallback
        # ------------------------------------------------------------
        if reject.any():
            idx = reject.nonzero(as_tuple=False).flatten()
            # Interior gravity-z targets for bins 0..3.  With pitch=0,
            # projected gravity z is -cos(roll).
            fallback_gz = torch.tensor(
                [0.85, 0.45, 0.0, -0.50],
                device=self.device,
                dtype=self.root_states.dtype,
            )[orientation_bins[idx]]
            fallback_roll = torch.acos(torch.clamp(-fallback_gz, -1.0, 1.0))
            fallback_pitch = torch.zeros_like(fallback_roll)
            fallback_yaw = (
                2.0 * torch.rand(idx.numel(), device=self.device) - 1.0
            ) * torch.pi
            quat[idx] = quat_from_euler_xyz(
                fallback_roll,
                fallback_pitch,
                fallback_yaw,
            )

        # Store the selected orientation before height sampling.
        self.root_states[env_ids, 3:7] = quat

        # _get_heights() currently uses self.base_quat rather
        # than root_states[:, 3:7].
        self.base_quat[env_ids] = quat

        # ------------------------------------------------------------
        # Terrain-relative spawn height
        # ------------------------------------------------------------

        # Z does not affect the height-map XY indices.
        self.root_states[env_ids, 2] = 0.0

        spawn_height_map = self._get_heights(env_ids, cfg)
        # Conservative local terrain elevation.
        local_spawn_height = torch.quantile(spawn_height_map, 0.8, dim=1)

        if cfg.env.robot == "go1":
            clearance = 0.15 + 0.06 * torch.rand(num_resets, device=self.device)
        else:
            clearance = 0.18 + 0.08 * torch.rand(num_resets, device=self.device)

        self.root_states[env_ids, 2] = local_spawn_height + clearance

        # Keep cached base state consistent with the reset state.
        self.base_pos[env_ids] = self.root_states[env_ids, 0:3]

        # ------------------------------------------------------------
        # Initial root velocities
        # ------------------------------------------------------------
        root_velocity = torch.zeros(num_resets, 6, device=self.device, dtype=self.root_states.dtype)

        root_velocity[:, 0:3] = 0.05 * torch.randn(num_resets, 3, device=self.device)

        # Do not launch the robot upward.
        root_velocity[:, 2] = torch.clamp(root_velocity[:, 2], min=-0.10, max=0.0)

        # Initial angular velocity remains zero.
        self.root_states[env_ids, 7:13] = root_velocity

        # ------------------------------------------------------------
        # Push reset state to Isaac Gym
        # ------------------------------------------------------------
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Do not call gym.simulate() here.
        # That would advance all environments outside the normal
        # policy-step/reward/observation pipeline.

        # ------------------------------------------------------------
        # Reset policy and controller history
        # ------------------------------------------------------------
        self._sync_recovery_control_history_after_reset(env_ids)

        # ------------------------------------------------------------
        # Reset-source diagnostics
        # ------------------------------------------------------------
        if not hasattr(self, "terminal_reset_buf"):
            self.terminal_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.terminal_reset_buf[env_ids] = False

        self._mark_recovery_reset_source(env_ids, source_value=0.0)

        # ------------------------------------------------------------
        # Orientation diagnostics
        # ------------------------------------------------------------
        sampled_gravity = quat_rotate_inverse(quat, gravity)
        gravity_z = sampled_gravity[:, 2]
        if self.common_step_counter % 1000 == 0:
            bin_counts = torch.bincount(orientation_bins, minlength=4).float()
            bin_fractions = bin_counts / float(max(num_resets, 1))
            print(
                "[RESET ORIENTATION]",
                f"bin fractions={bin_fractions.detach().cpu().tolist()}",
                f"| gz min={gravity_z.min().item():.3f}",
                f"mean={gravity_z.mean().item():.3f}",
                f"max={gravity_z.max().item():.3f}",
            )

        # ------------------------------------------------------------
        # Video bookkeeping
        # ------------------------------------------------------------
        if cfg.env.record_video and bool((env_ids == 0).any()):
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]

            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and bool((env_ids == self.num_train_envs).any()):
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = (
                    self.video_frames_eval[:]
                )

            self.video_frames_eval = []


    # def _reset_root_states_fall_recovery(self, env_ids, cfg):
    #     """
    #     Recovery reset dispatcher.

    #     Keeps the external reset call unchanged:

    #         self._call_train_eval(self._reset_root_states_fall_recovery, env_ids)

    #     Internally, it chooses between:
    #         0 = normal fallen-state reset
    #         1 = near-terminal standing reset

    #     For the next short stabilization run, use:
    #         cfg.env.terminal_stance_reset_prob = 1.0

    #     Later:
    #         0.7 terminal / 0.3 fallen
    #         0.4 terminal / 0.6 fallen
    #         0.0 terminal / 1.0 fallen
    #     """

    #     if len(env_ids) == 0:
    #         return

    #     terminal_prob = getattr(cfg.env, "terminal_stance_reset_prob", 0.0)
    #     terminal_prob = float(terminal_prob)

    #     if terminal_prob <= 0.0:
    #         self._reset_root_states_fall_recovery_fallen(env_ids, cfg)
    #         return

    #     if terminal_prob >= 1.0:
    #         self._reset_root_states_recovery_terminal_stance(env_ids, cfg)
    #         return

    #     use_terminal = (
    #         torch.rand(len(env_ids), device=self.device) < terminal_prob
    #     )

    #     terminal_env_ids = env_ids[use_terminal]
    #     fallen_env_ids = env_ids[~use_terminal]

    #     if len(terminal_env_ids) > 0:
    #         self._reset_root_states_recovery_terminal_stance(terminal_env_ids, cfg)

    #     if len(fallen_env_ids) > 0:
    #         self._reset_root_states_fall_recovery_fallen(fallen_env_ids, cfg)
    #

    def _reset_root_states_fall_recovery(self, env_ids, cfg):
        """
        Fall-recovery reset dispatcher.

        During terminal-stance debugging, force all envs into a clean terminal
        standing reset. During normal training, mix terminal and fallen resets.
        """

        if len(env_ids) == 0:
            return

        # ------------------------------------------------------------
        # DEBUG MODE:
        # Force clean terminal stance only.
        # This checks whether the terminal pose itself is physically valid.
        # ------------------------------------------------------------
        if getattr(cfg.env, "debug_clean_terminal_reset", False):
            self._reset_root_states_recovery_terminal_stance(env_ids, cfg)
            return

        # ------------------------------------------------------------
        # TRAINING MODE:
        # Mix terminal-stance resets and normal fallen resets.
        # Important: mask must have shape [len(env_ids)], not [num_envs].
        # ------------------------------------------------------------
        terminal_prob = getattr(cfg.env, "terminal_stance_reset_prob", 0.0)

        reset_sample = torch.rand(len(env_ids), device=self.device)
        terminal_mask = reset_sample < terminal_prob

        terminal_env_ids = env_ids[terminal_mask]
        fallen_env_ids = env_ids[~terminal_mask]

        if len(terminal_env_ids) > 0:
            self._reset_root_states_recovery_terminal_stance(terminal_env_ids, cfg)

        if len(fallen_env_ids) > 0:
            self._reset_root_states_fall_recovery_fallen(fallen_env_ids, cfg)


    def _get_default_dof_pos_for_env_ids(self, env_ids):
        """
        Returns default DOF positions with shape [len(env_ids), num_dof].

        Handles both cases:
            self.default_dof_pos shape [1, num_dof]
            self.default_dof_pos shape [num_envs, num_dof]
        """

        num_resets = len(env_ids)

        if self.default_dof_pos.ndim == 1:
            return self.default_dof_pos.unsqueeze(0).expand(num_resets, -1)

        if self.default_dof_pos.ndim == 2:
            if self.default_dof_pos.shape[0] == 1:
                return self.default_dof_pos.expand(num_resets, -1)

            if self.default_dof_pos.shape[0] == self.num_envs:
                return self.default_dof_pos[env_ids]

        raise RuntimeError(
            f"Unexpected default_dof_pos shape: {self.default_dof_pos.shape}"
        )


    # def _reset_root_states_recovery_terminal_stance(self, env_ids, cfg):
    #     """
    #     Near-terminal recovery reset.

    #     Purpose:
    #         Train the missing final skill:
    #             near-standing -> reduce foot slip -> stabilize contacts -> handoff-ready

    #     Important:
    #         Do NOT zero-torque settle here.
    #         Zero-torque settling is useful for fallen resets, but for near-standing
    #         resets it can immediately collapse the robot before the policy acts.
    #     """

    #     if len(env_ids) == 0:
    #         return

    #     num_resets = len(env_ids)
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)

    #     # ------------------------------------------------------------
    #     # 1. Base position
    #     # ------------------------------------------------------------
    #     self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2]

    #     if self.cfg.env.robot == "go1":
    #         # Near standing but not always exactly perfect.
    #         self.root_states[env_ids, 2] = torch_rand_float(
    #             0.305, 0.335, (num_resets, 1), device=self.device
    #         ).squeeze(1)
    #     else:
    #         self.root_states[env_ids, 2] = torch_rand_float(
    #             0.40, 0.46, (num_resets, 1), device=self.device
    #         ).squeeze(1)

    #     # ------------------------------------------------------------
    #     # 2. Base orientation
    #     # ------------------------------------------------------------
    #     roll = torch_rand_float(
    #         -0.08, 0.08, (num_resets, 1), device=self.device
    #     ).squeeze(1)

    #     pitch = torch_rand_float(
    #         -0.08, 0.08, (num_resets, 1), device=self.device
    #     ).squeeze(1)

    #     yaw = torch_rand_float(
    #         -torch.pi, torch.pi, (num_resets, 1), device=self.device
    #     ).squeeze(1)

    #     quat = quat_from_euler_xyz(roll, pitch, yaw)
    #     self.root_states[env_ids, 3:7] = quat

    #     # ------------------------------------------------------------
    #     # 3. Base velocity
    #     # ------------------------------------------------------------
    #     self.root_states[env_ids, 7:13] = 0.0

    #     # Small horizontal perturbation only.
    #     self.root_states[env_ids, 7:9] = torch_rand_float(
    #         -0.03, 0.03, (num_resets, 2), device=self.device
    #     )

    #     # Very small angular perturbation.
    #     self.root_states[env_ids, 10:13] = torch_rand_float(
    #         -0.08, 0.08, (num_resets, 3), device=self.device
    #     )

    #     # ------------------------------------------------------------
    #     # 4. Joint state
    #     # ------------------------------------------------------------
    #     q_nominal = self._get_default_dof_pos_for_env_ids(env_ids)

    #     # Easier terminal reset for first stability/debug run.
    #     q_noise = 0.01 * torch.randn(
    #         num_resets, self.num_dof, device=self.device
    #     )

    #     q_min = self.dof_pos_limits[:, 0].unsqueeze(0)
    #     q_max = self.dof_pos_limits[:, 1].unsqueeze(0)

    #     q_reset = torch.minimum(
    #         torch.maximum(q_nominal + q_noise, q_min),
    #         q_max,
    #     )

    #     self.dof_pos[env_ids] = q_reset
    #     self.dof_vel[env_ids] = 0.0

    #     # Store the exact reset pose so we can test whether this pose is physically stable
    #     # when the PD controller holds it.
    #     if not hasattr(self, "debug_hold_dof_pos"):
    #         self.debug_hold_dof_pos = torch.zeros_like(self.dof_pos)

    #     self.debug_hold_dof_pos[env_ids] = q_reset

    #     # ------------------------------------------------------------
    #     # 5. Push state to simulator
    #     # ------------------------------------------------------------
    #     self.gym.set_dof_state_tensor_indexed(
    #         self.sim,
    #         gymtorch.unwrap_tensor(self.dof_state),
    #         gymtorch.unwrap_tensor(env_ids_int32),
    #         len(env_ids_int32),
    #     )

    #     self.gym.set_actor_root_state_tensor_indexed(
    #         self.sim,
    #         gymtorch.unwrap_tensor(self.root_states),
    #         gymtorch.unwrap_tensor(env_ids_int32),
    #         len(env_ids_int32),
    #     )

    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     self.gym.refresh_net_contact_force_tensor(self.sim)
    #     self.gym.refresh_dof_state_tensor(self.sim)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)

    #     # ------------------------------------------------------------
    #     # 6. Critical: remove stale action/target history
    #     # ------------------------------------------------------------
    #     self._sync_recovery_control_history_after_reset(env_ids)

    #     # 1 = terminal reset. Useful for dashboard/debugging.
    #     self._mark_recovery_reset_source(env_ids, source_value=1.0)

    #     # ------------------------------------------------------------
    #     # 7. Video buffer handling
    #     # ------------------------------------------------------------
    #     if cfg.env.record_video and 0 in env_ids:
    #         if self.complete_video_frames is None:
    #             self.complete_video_frames = []
    #         else:
    #             self.complete_video_frames = self.video_frames[:]
    #         self.video_frames = []

    #     if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
    #         if self.complete_video_frames_eval is None:
    #             self.complete_video_frames_eval = []
    #         else:
    #             self.complete_video_frames_eval = self.video_frames_eval[:]
    #         self.video_frames_eval = []
    #

    def _reset_root_states_recovery_terminal_stance(self, env_ids, cfg):
        """
        Clean terminal-stance reset.

        Purpose:
            Create a physically valid near-handoff standing state.

        This is NOT a fallen reset. It should initialize the robot in an upright,
        feet-on-ground, no-body-contact standing pose.

        Use this first with:
            debug_hold_reset_pose = True
            debug_clean_terminal_reset = True
            terminal_stance_reset_prob = 1.0

        Expected result:
            stable_height high
            nonfoot_contact_env_frac near 0
            foot_contact_count_mean around 3-4
            base_height_mean around 0.30-0.34
        """

        if len(env_ids) == 0:
            return

        num_resets = len(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # ------------------------------------------------------------
        # 1. Root position: clean standing height.
        # ------------------------------------------------------------
        self.root_states[env_ids, 0:2] = self.env_origins[env_ids, 0:2]

        # Use a conservative standing height.
        # Your previous debug showed base_height_mean ~0.19, which is too low.
        self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + 0.34

        # ------------------------------------------------------------
        # 2. Root orientation: exactly upright.
        # Quaternion format: x, y, z, w
        # ------------------------------------------------------------
        upright_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=self.root_states.dtype,
        ).repeat(num_resets, 1)

        self.root_states[env_ids, 3:7] = upright_quat

        # ------------------------------------------------------------
        # 3. Root velocities: zero.
        # ------------------------------------------------------------
        self.root_states[env_ids, 7:13] = 0.0

        # ------------------------------------------------------------
        # 4. Joint pose: use default standing pose.
        # Do NOT use a crouched/low pose here.
        # ------------------------------------------------------------
        q_reset = self._get_default_dof_pos_for_env_ids(env_ids).clone()

        self.dof_pos[env_ids, :self.num_dof] = q_reset[:, :self.num_dof]
        self.dof_vel[env_ids, :self.num_dof] = 0.0

        # ------------------------------------------------------------
        # 5. Store exact reset pose for debug_hold_reset_pose.
        # step() will command this pose through the PD controller.
        # ------------------------------------------------------------
        if hasattr(self, "debug_hold_dof_pos"):
            self.debug_hold_dof_pos[env_ids] = q_reset

        # ------------------------------------------------------------
        # 6. Reset action and target histories.
        # This prevents artificial action-smoothness / PD transient spikes.
        # ------------------------------------------------------------
        n = min(self.num_actions, self.num_dof)

        if hasattr(self, "joint_pos_target"):
            self.joint_pos_target[env_ids, :n] = q_reset[:, :n]

        if hasattr(self, "last_joint_pos_target"):
            self.last_joint_pos_target[env_ids, :n] = q_reset[:, :n]

        if hasattr(self, "last_last_joint_pos_target"):
            self.last_last_joint_pos_target[env_ids, :n] = q_reset[:, :n]

        if hasattr(self, "actions"):
            self.actions[env_ids] = 0.0

        if hasattr(self, "last_actions"):
            self.last_actions[env_ids] = 0.0

        if hasattr(self, "last_last_actions"):
            self.last_last_actions[env_ids] = 0.0

        if hasattr(self, "last_dof_vel"):
            self.last_dof_vel[env_ids] = 0.0

        if hasattr(self, "torques"):
            self.torques[env_ids] = 0.0

        if hasattr(self, "last_torques"):
            self.last_torques[env_ids] = 0.0

        # ------------------------------------------------------------
        # 7. Mark reset source for dashboard.
        # ------------------------------------------------------------
        if not hasattr(self, "terminal_reset_buf"):
            self.terminal_reset_buf = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )

        self.terminal_reset_buf[env_ids] = True

        self._mark_recovery_reset_source(env_ids, source_value=1.0)  # 1 = terminal reset

        # ------------------------------------------------------------
        # 8. Push state to simulator.
        # Keep this if your reset functions directly update Isaac Gym tensors.
        # If your reset_idx already does this globally later, duplicate setting is
        # harmless but slightly redundant.
        # ------------------------------------------------------------
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )


    def _sync_recovery_control_history_after_reset(self, env_ids):
        """
        Clears stale policy/action/PD history after recovery reset.

        This is especially important for terminal-stance resets. Otherwise, the first
        action after reset may compare against old targets from the previous episode,
        creating an artificial jerk and immediate foot slip.
        """

        if len(env_ids) == 0:
            return

        q_reset = self.dof_pos[env_ids, :self.num_dof].clone()

        if hasattr(self, "actions"):
            self.actions[env_ids] = 0.0

        if hasattr(self, "last_actions"):
            self.last_actions[env_ids] = 0.0

        if hasattr(self, "last_last_actions"):
            self.last_last_actions[env_ids] = 0.0

        if hasattr(self, "joint_pos_target"):
            self.joint_pos_target[env_ids, :self.num_dof] = q_reset

        if hasattr(self, "last_joint_pos_target"):
            self.last_joint_pos_target[env_ids, :self.num_dof] = q_reset

        if hasattr(self, "last_last_joint_pos_target"):
            self.last_last_joint_pos_target[env_ids, :self.num_dof] = q_reset

        if hasattr(self, "last_dof_vel"):
            self.last_dof_vel[env_ids] = 0.0

        if hasattr(self, "last_contacts"):
            self.last_contacts[env_ids] = False

        if hasattr(self, "torques"):
            self.torques[env_ids] = 0.0

        if hasattr(self, "last_torques"):
            self.last_torques[env_ids] = 0.0

        if hasattr(self, "last_last_torques"):
            self.last_last_torques[env_ids] = 0.0

        # For actuator-network history, if used.
        if hasattr(self, "joint_pos_err_last"):
            self.joint_pos_err_last[env_ids] = 0.0

        if hasattr(self, "joint_pos_err_last_last"):
            self.joint_pos_err_last_last[env_ids] = 0.0

        if hasattr(self, "joint_vel_last"):
            self.joint_vel_last[env_ids] = 0.0

        if hasattr(self, "joint_vel_last_last"):
            self.joint_vel_last_last[env_ids] = 0.0


    def _mark_recovery_reset_source(self, env_ids, source_value):
        """
        Tracks reset source for debugging.

        source_value:
            0 = fallen recovery reset
            1 = terminal / near-stand reset
        """

        if len(env_ids) == 0:
            return

        source_int = int(source_value)

        if not hasattr(self, "recovery_reset_source_buf"):
            self.recovery_reset_source_buf = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            )

        if not hasattr(self, "reset_source"):
            self.reset_source = torch.full(
                (self.num_envs,),
                -1,
                device=self.device,
                dtype=torch.long,
            )

        if not hasattr(self, "reset_source_event_counts"):
            self.reset_source_event_counts = torch.zeros(
                2,
                device=self.device,
                dtype=torch.long,
            )

        self.recovery_reset_source_buf[env_ids] = float(source_int)
        self.reset_source[env_ids] = source_int

        if source_int in (0, 1):
            self.reset_source_event_counts[source_int] += len(env_ids)



    def _reset_root_states(self, env_ids, cfg):
            """ Resets ROOT states position and velocities of selected environmments
                Sets base position based on the curriculum
                Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            Args:
                env_ids (List[int]): Environemnt ids
            """
            # base position
            if self.custom_origins:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                                   cfg.terrain.x_init_range, (len(env_ids), 1),
                                                                   device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                                   cfg.terrain.y_init_range, (len(env_ids), 1),
                                                                   device=self.device)
                self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
                self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]

            # base yaws
            init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                         cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                         device=self.device)
            quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
            self.root_states[env_ids, 3:7] = quat

            # base velocities
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                               device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            if cfg.env.record_video and 0 in env_ids:
                if self.complete_video_frames is None:
                    self.complete_video_frames = []
                else:
                    self.complete_video_frames = self.video_frames[:]
                self.video_frames = []

            if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
                if self.complete_video_frames_eval is None:
                    self.complete_video_frames_eval = []
                else:
                    self.complete_video_frames_eval = self.video_frames_eval[:]
                self.video_frames_eval = []

    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            if cfg.env.train_recovery:
                env_ids_int32 = env_ids.to(torch.int32)

                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32)
                )
            else:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            if cfg.env.train_recovery:
                env_ids_int32 = env_ids.to(torch.int32)

                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32)
                )
            else:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)
        if self.cfg.env.observe_only_ang_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        self.debug_hold_dof_pos = torch.zeros_like(self.dof_pos)

        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        if self.cfg.env.train_recovery:
            self.recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.recovered_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            # =========================================================
            # Rollout-level recovery debug statistics
            # =========================================================
            # Has env EVER been upright during rollout?
            self.rollout_upright = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            # Has env EVER satisfied recovered condition?
            self.rollout_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Recovery rate over completed episodes, not control steps.
            self.recovery_rate_ema = 0.0
            self.recovery_ema_initialized = False

            # Orientation is categorical and is sampled uniformly on every
            # episode.  It is not promoted or demoted.
            #   0 = strongly inverted, 1 = partially inverted,
            #   2 = sideways,          3 = near-upright fallen.
            self.episode_orientation_bin = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )

            # Backward-compatible dashboard alias.  No code should interpret
            # this value as a global curriculum phase anymore.
            self.episode_orientation_phase = self.episode_orientation_bin

            # Exact task labels captured at episode initialization.  These
            # prevent stale episodes from voting after a frontier promotion.
            self.episode_terrain_type = torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            )
            self.episode_terrain_level = torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            )
            self.episode_sampler_group = torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            )

            # Maximum persistence counter reached during rollout
            self.rollout_max_counter = torch.zeros(self.num_envs, device=self.device)

            self.recovery_bonus_buf = torch.zeros(self.num_envs, device=self.device)

            self.reset_source = torch.full(
                (self.num_envs,),
                -1,
                device=self.device,
                dtype=torch.long,
            )

            # cumulative reset-event counts:
            # index 0 = fallen reset
            # index 1 = terminal / near-stand reset
            self.reset_source_event_counts = torch.zeros(
                2,
                device=self.device,
                dtype=torch.long,
            )

            self.near_crouch_clip_frac = torch.tensor(
                0.0,
                device=self.device,
                dtype=torch.float,
            )

            self.near_stand_clip_frac = torch.tensor(
                0.0,
                device=self.device,
                dtype=torch.float,
            )

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_last_torques = torch.zeros_like(self.torques)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])


        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )


        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)


        print("episode_length_s:", self.cfg.env.episode_length_s)
        print("sim dt:", self.sim_params.dt)
        print("control decimation:", self.cfg.control.decimation)

        dt_effective = self.sim_params.dt * self.cfg.control.decimation
        print("effective dt:", dt_effective)

        max_steps = int(self.cfg.env.episode_length_s / dt_effective)
        print("expected max_episode_length:", max_steps)

        # for torque smoothing
        # self.torque_history_len = 3  # N = 3–5 is ideal
        # self.torque_history = torch.zeros(
        #     self.torque_history_len,
        #     self.num_envs,
        #     self.num_dof,
        #     device=self.device
        # )
        # self.smoothed_torques = torch.zeros_like(self.torques)

        # Legacy per-environment counters retained for old dashboards.  The
        # active curriculum uses cell and frontier statistics initialized
        # below, not these counters.
        self.terrain_curriculum_successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.terrain_curriculum_trials = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.cfg.env.train_recovery:
            self._init_recovery_curriculum_buffers()

    def _init_recovery_curriculum_buffers(self):
        """Initialize the terrain-frontier x orientation-bin task grid."""

        terrain_cfg = self.cfg.terrain
        num_types = int(terrain_cfg.num_cols)
        num_levels = int(terrain_cfg.num_rows)
        num_bins = 4

        configured_max = int(getattr(
            terrain_cfg,
            "curriculum_max_level",
            num_levels - 1,
        ))
        self.recovery_curriculum_max_level = max(
            0,
            min(configured_max, num_levels - 1),
        )
        start_level = int(getattr(terrain_cfg, "recovery_start_level", 0))
        start_level = max(0, min(start_level, self.recovery_curriculum_max_level))

        # One promoted frontier per physical terrain column.  Columns can map
        # to repeated semantic terrain families; keeping separate frontiers
        # also captures realization-specific difficulty.
        self.terrain_frontier = torch.full(
            (num_types,),
            start_level,
            device=self.device,
            dtype=torch.long,
        )

        # Competence statistics for every (column, level, orientation-bin)
        # cell.  All sampler groups update these diagnostic statistics.
        grid_shape = (num_types, num_levels, num_bins)
        self.recovery_cell_success_ema = torch.zeros(
            grid_shape, device=self.device, dtype=torch.float
        )
        self.recovery_cell_trials = torch.zeros(
            grid_shape, device=self.device, dtype=torch.long
        )

        # Only current-frontier episodes from sampler group 0 contribute to
        # these promotion windows.
        self.frontier_window_successes = torch.zeros(
            (num_types, num_bins), device=self.device, dtype=torch.float
        )
        self.frontier_window_trials = torch.zeros(
            (num_types, num_bins), device=self.device, dtype=torch.long
        )
        self.frontier_pass_windows = torch.zeros(
            num_types, device=self.device, dtype=torch.long
        )
        self.frontier_last_mean_success = torch.zeros(
            num_types, device=self.device, dtype=torch.float
        )
        self.frontier_last_lower_quartile = torch.zeros(
            num_types, device=self.device, dtype=torch.float
        )
        self.frontier_review_count = torch.zeros(
            num_types, device=self.device, dtype=torch.long
        )

        # One sampling stage per physical terrain column:
        # 0 = bootstrap, 1 = developing, 2 = mature.
        self.terrain_sampling_stage = torch.zeros(
            num_types,
            device=self.device,
            dtype=torch.long,
        )
        self.recovery_total_promotions = 0

        # Training starts at the configured easy level.  Terrain columns stay
        # fixed; only the row/level is resampled at episode boundaries.
        if self.custom_origins and self.num_train_envs > 0:
            train_ids = torch.arange(
                self.num_train_envs,
                device=self.device,
                dtype=torch.long,
            )
            train_types = self.terrain_types[train_ids]
            self.terrain_levels[train_ids] = self.terrain_frontier[train_types]
            self.env_origins[train_ids] = terrain_cfg.terrain_origins[
                self.terrain_levels[train_ids],
                train_types,
            ]

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # link mass randomization
        self.trunk_masses = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.hip_masses = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.thigh_masses = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.calf_masses = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.mass_groups = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # defaults for normalization/reference
        self.mass_groups_default = torch.zeros(4, dtype=torch.float, device=self.device, requires_grad=False)

        self.rigid_body_masses = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device,
                                     requires_grad=False)

        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                           self.cfg.commands.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _get_recovery_sampling_probabilities(self, terrain_types):
        """
        Return per-environment sampling probabilities.

        Every physical terrain column owns an independent sampling stage.
        Environments assigned to different columns can therefore use different
        frontier/replay/next/coverage mixtures in the same reset batch.
        """

        cfg = self.cfg.terrain
        terrain_types = terrain_types.to(device=self.device, dtype=torch.long)

        if terrain_types.numel() == 0:
            empty = torch.zeros(0, device=self.device, dtype=torch.float)
            return empty, empty, empty, empty

        if (
            (terrain_types < 0).any()
            or (terrain_types >= self.terrain_sampling_stage.numel()).any()
        ):
            raise ValueError("terrain type is outside the configured column range")

        # ------------------------------------------------------------
        # Update each column's stage independently and monotonically.
        # ------------------------------------------------------------
        reviewed = self.frontier_review_count > 0
        bootstrap_threshold = float(getattr(
            cfg,
            "bootstrap_success_threshold",
            0.60,
        ))

        developing = (
            reviewed
            & (self.frontier_last_mean_success >= bootstrap_threshold)
        )
        self.terrain_sampling_stage[developing] = torch.maximum(
            self.terrain_sampling_stage[developing],
            torch.ones_like(self.terrain_sampling_stage[developing]),
        )

        mature_level = int(getattr(cfg, "mature_frontier_level", 3))
        mature = self.terrain_frontier >= mature_level
        self.terrain_sampling_stage[mature] = 2

        env_stages = self.terrain_sampling_stage[terrain_types]

        developing_probabilities = torch.tensor(
            [
                float(getattr(cfg, "developing_frontier_fraction", 0.75)),
                float(getattr(cfg, "developing_replay_fraction", 0.15)),
                float(getattr(cfg, "developing_next_fraction", 0.10)),
                float(getattr(cfg, "developing_coverage_fraction", 0.00)),
            ],
            device=self.device,
            dtype=torch.float,
        )
        mature_probabilities = torch.tensor(
            [
                float(getattr(cfg, "mature_frontier_fraction", 0.60)),
                float(getattr(cfg, "mature_replay_fraction", 0.25)),
                float(getattr(cfg, "mature_next_fraction", 0.10)),
                float(getattr(cfg, "mature_coverage_fraction", 0.05)),
            ],
            device=self.device,
            dtype=torch.float,
        )
        bootstrap_probabilities = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float,
        )

        configured_probabilities = torch.stack(
            (
                bootstrap_probabilities,
                developing_probabilities,
                mature_probabilities,
            ),
            dim=0,
        )

        if (configured_probabilities < 0.0).any():
            raise ValueError(
                "Recovery curriculum sampling weights must be nonnegative"
            )

        totals = configured_probabilities.sum(dim=1, keepdim=True)
        if (totals <= 0.0).any():
            raise ValueError(
                "Every recovery sampling stage must have positive total weight"
            )

        configured_probabilities = configured_probabilities / totals
        env_probabilities = configured_probabilities[env_stages]

        return (
            env_probabilities[:, 0],
            env_probabilities[:, 1],
            env_probabilities[:, 2],
            env_probabilities[:, 3],
        )

    def _sample_recovery_curriculum_tasks(self, env_ids):
        """
        Assign the next task independently at every episode reset.

        Sampler groups:
            0 frontier: current promoted level; eligible to vote.
            1 replay:   a strictly lower mastered level (or zero).
            2 next:     one level above the frontier, clipped at max.
            3 coverage: any terrain level, used only after maturation.

        Orientation bin is always uniform over 0..3 and never promoted.
        """

        if len(env_ids) == 0:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # Both training and evaluation resets receive a categorical fallen
        # orientation.  Only training environments participate in terrain
        # frontier sampling and statistics.
        self.episode_orientation_bin[env_ids] = torch.randint(
            0,
            4,
            (env_ids.numel(),),
            device=self.device,
        )

        train_ids = env_ids[env_ids < self.num_train_envs]
        if train_ids.numel() == 0:
            return

        terrain_types = self.terrain_types[train_ids]
        levels = self.terrain_levels[train_ids].clone()
        groups = torch.zeros_like(train_ids)

        cfg = self.cfg.terrain
        curriculum_enabled = (
            bool(getattr(cfg, "recovery_curriculum", False))
            and bool(cfg.curriculum)
            and bool(self.custom_origins)
        )

        if curriculum_enabled:
            frontier = self.terrain_frontier[terrain_types]
            p_frontier, p_replay, p_next, p_coverage = (
                self._get_recovery_sampling_probabilities(terrain_types)
            )
            u = torch.rand(train_ids.numel(), device=self.device)

            replay_group = (u >= p_frontier) & (u < p_frontier + p_replay)
            next_group = (
                (u >= p_frontier + p_replay)
                & (u < p_frontier + p_replay + p_next)
            )
            coverage_group = (
                (u >= p_frontier + p_replay + p_next)
                & (p_coverage > 0.0)
            )

            groups[replay_group] = 1
            groups[next_group] = 2
            groups[coverage_group] = 3

            # Frontier group is the default.
            levels = frontier.clone()

            # Strictly lower replay when possible.  At frontier zero this
            # correctly degenerates to level zero.
            if replay_group.any():
                replay_frontier = frontier[replay_group]
                replay_levels = torch.floor(
                    torch.rand(replay_frontier.numel(), device=self.device)
                    * replay_frontier.clamp_min(1).float()
                ).long()
                levels[replay_group] = replay_levels

            if next_group.any():
                levels[next_group] = torch.clamp(
                    frontier[next_group] + 1,
                    max=self.recovery_curriculum_max_level,
                )

            if coverage_group.any():
                levels[coverage_group] = torch.randint(
                    0,
                    self.recovery_curriculum_max_level + 1,
                    (int(coverage_group.sum().item()),),
                    device=self.device,
                )

            self.terrain_levels[train_ids] = levels
            self.env_origins[train_ids] = cfg.terrain_origins[
                levels,
                terrain_types,
            ]

        # Store exactly what this episode received.  Outcome accounting reads
        # these labels instead of mutable global/current curriculum state.
        self.episode_terrain_type[train_ids] = terrain_types
        self.episode_terrain_level[train_ids] = levels
        self.episode_sampler_group[train_ids] = groups

        tracking = self.extras.setdefault("recovery_tracking", {})
        env_stages = self.terrain_sampling_stage[terrain_types]
        tracking["sampling_stage_bootstrap_frac"] = (
            (env_stages == 0).float().mean().item()
        )
        tracking["sampling_stage_developing_frac"] = (
            (env_stages == 1).float().mean().item()
        )
        tracking["sampling_stage_mature_frac"] = (
            (env_stages == 2).float().mean().item()
        )
        tracking["sample_frontier_frac"] = (groups == 0).float().mean().item()
        tracking["sample_replay_frac"] = (groups == 1).float().mean().item()
        tracking["sample_next_frac"] = (groups == 2).float().mean().item()
        tracking["sample_coverage_frac"] = (groups == 3).float().mean().item()

    def _update_recovery_terrain_curriculum(self, ids, success):
        """
        Deprecated compatibility hook.

        Terrain updates are now performed by _update_recovery_curricula using
        balanced four-bin frontier windows.  In particular, failure never
        demotes a terrain level.
        """
        return

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from aliengo_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        # self.episode_sums["recovery_success"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        # self.episode_sums_eval["recovery_success"] = -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        # print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _compute_mass_groups_from_body_props(self, body_props):
        # choose mean for leg groups (stable scaling), body as sum (usually one trunk)
        def agg(indices, use_mean=False):
            if len(indices) == 0:
                return 0.0
            vals = [body_props[j].mass for j in indices]
            return float(np.mean(vals) if use_mean else np.sum(vals))

        m_body  = agg(self.mass_group_indices["body"],  use_mean=False)
        m_hip   = agg(self.mass_group_indices["hip"],   use_mean=True)
        m_thigh = agg(self.mass_group_indices["thigh"], use_mean=True)
        m_calf  = agg(self.mass_group_indices["calf"],  use_mean=True)
        return m_body, m_hip, m_thigh, m_calf

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        self.feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]

        # asset_body_props = self.gym.get_asset_rigid_body_properties(self.robot_asset)
        # self.body_masses = torch.tensor([prop.mass for prop in asset_body_props], device=self.device, dtype=torch.float)

        self.mass_group_indices = {
            "body":  [i for i, n in enumerate(self.body_names) if ("trunk" in n or n == "base")],
            "hip":   [i for i, n in enumerate(self.body_names) if ("_hip" in n)],
            "thigh": [i for i, n in enumerate(self.body_names) if ("_thigh" in n)],
            "calf":  [i for i, n in enumerate(self.body_names) if ("_calf" in n)],
        }
        print("mass_group_indices:", {k: len(v) for k, v in self.mass_group_indices.items()})


        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)

            # Compute m_true[i] from body_props using precomputed group indices
            # self.mass_groups[i, 0] = sum/mean masses for trunk/base
            # self.mass_groups[i, 1] = sum/mean masses for hips
            # self.mass_groups[i, 2] = sum/mean masses for thighs
            # self.mass_groups[i, 3] = sum/mean masses for calves

            # m_body, m_hip, m_thigh, m_calf = self._compute_mass_groups_from_body_props(body_props)
            # self.mass_groups[i, 0] = m_body
            # self.mass_groups[i, 1] = m_hip
            # self.mass_groups[i, 2] = m_thigh
            # self.mass_groups[i, 3] = m_calf

            # print(f"Link Masses: {self.mass_groups}")

            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         self.feet_names[i])

        print("feet_indices:", self.feet_indices)
        print("feet body names:", [self.body_names[i] for i in self.feet_indices])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.camera_props.enable_tensors = False # True (on gcp headless) #False (local)
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)

            if self.rendering_camera == -1:
                raise RuntimeError("❌ Camera creation failed — EGL not working")

            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

        #############
        all_bodies = torch.arange(self.num_bodies, device=self.device)
        # feet = torch.tensor(self.feet_indices, device=self.device)
        feet = self.feet_indices.to(device=self.device, dtype=torch.long)

        mask = torch.ones_like(all_bodies, dtype=torch.bool)
        mask[feet] = False

        self.base_contact_indices = all_bodies[mask]
        ############

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        if self.record_now:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]

            self.gym.set_camera_location(
                self.rendering_camera,
                self.envs[0],
                gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                gymapi.Vec3(bx, by, bz)
            )

            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            img = self.gym.get_camera_image(
                self.sim,
                self.envs[0],
                self.rendering_camera,
                gymapi.IMAGE_COLOR
            )

            # if img is not None:
            #     print("IMG SIZE:", img.size)

            if img is None or img.size == 0:
                return

            frame = img.reshape(
                (self.camera_props.height, self.camera_props.width, 4)
            )
            frame = frame[:, :, :3].astype(np.uint8)

            if len(self.video_frames) < self.max_video_frames:
                self.video_frames.append(frame)

            # if len(self.video_frames) < self.max_video_frames:
            #     print("APPEND FRAME")
            #     self.video_frames.append(frame)
            #     print("NEW LEN:", len(self.video_frames))

        # EVAL
        if self.record_now and self.eval_cfg is not None:
            bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                         self.root_states[self.num_train_envs, 2]

            self.gym.set_camera_location(
                self.rendering_camera_eval,
                self.envs[self.num_train_envs],
                gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                gymapi.Vec3(bx, by, bz)
            )

            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            img = self.gym.get_camera_image(
                self.sim,
                self.envs[self.num_train_envs],
                self.rendering_camera_eval,
                gymapi.IMAGE_COLOR
            )

            if img is None or img.size == 0:
                return

            frame = img.reshape(
                (self.camera_props.height, self.camera_props.width, 4)
            )
            frame = frame[:, :, :3].astype(np.uint8)

            if len(self.video_frames_eval) < self.max_video_frames:
                self.video_frames_eval.append(frame)

    def start_recording(self):
        # self.complete_video_frames = None
        self.video_frames = []
        self.record_now = True

    def start_recording_eval(self):
        # self.complete_video_frames_eval = None
        self.video_frames_eval = []
        self.record_eval_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def pause_recording_eval(self):
        self.complete_video_frames_eval = []
        self.video_frames_eval = []
        self.record_eval_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        if self.complete_video_frames_eval is None:
            return []
        return self.complete_video_frames_eval

    def stop_recording(self, tag="train", telegram_fn=None):
        import cv2, imageio, os
        from datetime import datetime

        self.record_now = False

        if len(self.video_frames) == 0:
            print("[WARN] No frames recorded")
            return

        os.makedirs("runs/videos", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_path = f"runs/videos/{tag}_{timestamp}.mp4"
        gif_path = f"runs/videos/{tag}_{timestamp}.gif"

        h, w, _ = self.video_frames[0].shape

        writer = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h),
        )

        for f in self.video_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        writer.release()

        imageio.mimsave(gif_path, self.video_frames[::2], fps=15)

        print(f"[Video Saved] {mp4_path}")

        if telegram_fn:
            try:
                telegram_fn(gif_path, caption=tag)
            except Exception as e:
                print("[WARNING] Telegram failed:", e)

        return mp4_path

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            xx = xx.to(self.device)
            yy = yy.to(self.device)
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    # def _draw_debug_vis(self):
    #     """ Draws visualizations for dubugging (slows down simulation a lot).
    #         Default behaviour: draws height measurement points
    #     """
    #     # draw height lines
    #     if not self.terrain.cfg.measure_heights:
    #         return
    #     self.gym.clear_lines(self.viewer)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
    #     for i in range(self.num_envs):
    #         base_pos = (self.root_states[i, :3]).cpu().numpy()
    #         heights = self.measured_heights[i].cpu().numpy()
    #         height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
    #                                        self.height_points[i]).cpu().numpy()
    #         for j in range(heights.shape[0]):
    #             x = height_points[j, 0] + base_pos[0]
    #             y = height_points[j, 1] + base_pos[1]
    #             z = heights[j]
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_debug_vis(self):
        """
        Draw height-sampling points using the original working visualization path.

        Colors:
            red     = outer grid boundary
            cyan    = center x/y axes
            magenta = exact center
            yellow  = remaining interior points
        """

        if not self.terrain.cfg.measure_heights:
            return

        if self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        radius = float(getattr(self.cfg.terrain, "debug_height_grid_point_radius", 0.025))

        nx = len(self.cfg.terrain.measured_points_x)
        ny = len(self.cfg.terrain.measured_points_y)

        boundary_geom = gymutil.WireframeSphereGeometry(radius, 4, 4, None, color=(1.0, 0.0, 0.0))
        axis_geom = gymutil.WireframeSphereGeometry(radius, 4, 4, None, color=(0.0, 1.0, 1.0))
        center_geom = gymutil.WireframeSphereGeometry(radius * 1.5, 6, 6, None, color=(1.0, 0.0, 1.0))
        interior_geom = gymutil.WireframeSphereGeometry(radius, 4, 4, None, color=(1.0, 1.0, 0.0))

        # -1 draws the grid for every environment.
        selected_env = int(getattr(self.cfg.terrain, "debug_height_grid_env_id", -1))

        if selected_env < 0:
            env_ids = range(self.num_envs)
        else:
            selected_env = max(0, min(selected_env, self.num_envs - 1))
            env_ids = [selected_env]

        center_i = nx // 2
        center_j = ny // 2

        z_visual_offset = 0.025

        for env_id in env_ids:
            base_pos = self.root_states[env_id, :3].detach().cpu().numpy()
            heights = self.measured_heights[env_id].detach().cpu().numpy()

            repeated_quat = self.base_quat[env_id].unsqueeze(0).repeat(heights.shape[0], 1)
            height_points = quat_apply_yaw(repeated_quat, self.height_points[env_id]).detach().cpu().numpy()

            for ix in range(nx):
                for iy in range(ny):
                    flat_idx = ix * ny + iy

                    is_boundary = ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1
                    is_center = ix == center_i and iy == center_j
                    is_center_axis = ix == center_i or iy == center_j

                    if is_center:
                        sphere_geom = center_geom
                    elif is_boundary:
                        sphere_geom = boundary_geom
                    elif is_center_axis:
                        sphere_geom = axis_geom
                    else:
                        sphere_geom = interior_geom

                    x = height_points[flat_idx, 0] + base_pos[0]
                    y = height_points[flat_idx, 1] + base_pos[1]
                    z = heights[flat_idx] + z_visual_offset

                    sphere_pose = gymapi.Transform(gymapi.Vec3(float(x), float(y), float(z)), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)


    def _init_height_points(self, env_ids, cfg):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.env.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.env.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.env.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.env.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale

    def _get_local_terrain_height(self):
        """
        Effective terrain height over the robot support footprint.
        Assumes self.measured_heights has shape [num_envs, num_points].
        """
        heights = self.measured_heights

        sorted_heights = torch.sort(
            heights,
            dim=1,
        ).values

        # Remove the lowest and highest 10%.
        trim = max(1, heights.shape[1] // 10)

        trimmed_heights = sorted_heights[:, trim:-trim]

        return trimmed_heights.mean(dim=1)

    def _update_recovery_orientation_curriculum(self, episode_success):
        """
        Deprecated compatibility hook.

        Orientation is now sampled uniformly as four fixed bins.  Keeping this
        no-op prevents an old external caller from reintroducing coupling with
        terrain promotion.
        """
        return False

    def _collect_recovery_episode_outcomes(self, env_ids):
        """
        Obtain final binary outcomes and immutable task labels.

        Terminal/near-stand reset episodes are deliberately excluded: they are
        useful for stabilizer training but are not fall-recovery evidence.
        """

        def empty_result(ids):
            empty_bool = torch.zeros(0, dtype=torch.bool, device=self.device)
            empty_long = torch.zeros(0, dtype=torch.long, device=self.device)
            return (
                ids,
                empty_bool,
                empty_long,
                empty_long,
                empty_long,
                empty_long,
            )

        train_ids = env_ids[env_ids < self.num_train_envs]

        if train_ids.numel() == 0:
            return empty_result(train_ids)

        # Exclude the artificial reset performed during env.reset().
        valid_episode = self.episode_length_buf[train_ids] > 0
        train_ids = train_ids[valid_episode]

        if train_ids.numel() == 0:
            return empty_result(train_ids)

        success_reset = self.success_reset_buf[train_ids].bool()
        timeout = self.time_out_buf[train_ids].bool()
        bad_state = self.bad_state_buf[train_ids].bool()

        # All normal episode endings in the current implementation.
        completed = success_reset | timeout | bad_state

        ids = train_ids[completed]

        if ids.numel() == 0:
            return empty_result(ids)

        # reset_source describes how the current (ending) episode began.
        # Only source 0 is a genuine fallen-state recovery task.
        ids = ids[self.reset_source[ids] == 0]

        if ids.numel() == 0:
            return empty_result(ids)

        bad_state = self.bad_state_buf[ids].bool()

        terminate_on_success = getattr(self.cfg.env, "terminate_on_recovery_success", False)

        if terminate_on_success:
            # Stable recovery directly terminates the episode.
            success = (
                self.success_reset_buf[ids].bool()
                & (~bad_state)
            )
        else:
            # For non-terminating stabilizer training, success means
            # that the robot is still stably recovered at timeout.
            stable_at_end = (
                self.recovery_counter[ids]
                >= self.cfg.rewards.recovery_success_steps
            )

            success = (
                self.time_out_buf[ids].bool()
                & stable_at_end
                & (~bad_state)
            )

        return (
            ids,
            success,
            self.episode_terrain_type[ids],
            self.episode_terrain_level[ids],
            self.episode_orientation_bin[ids],
            self.episode_sampler_group[ids],
        )

    def _update_recovery_curricula(self, env_ids):
        """
        Update the joint task grid and promote terrain frontiers.

        All genuine fallen episodes update competence diagnostics.  Only an
        episode sampled from group 0 at the column's *current* frontier is
        eligible to vote on promotion.  Therefore stale episodes, replay,
        next-level probes, and coverage samples cannot accidentally promote or
        demote the terrain curriculum.
        """

        (
            ids,
            success,
            terrain_types,
            terrain_levels,
            orientation_bins,
            sampler_groups,
        ) = self._collect_recovery_episode_outcomes(env_ids)

        if ids.numel() == 0:
            return

        cfg = self.cfg.terrain
        num_types = int(cfg.num_cols)
        num_levels = int(cfg.num_rows)
        num_bins = 4

        valid_labels = (
            (terrain_types >= 0)
            & (terrain_types < num_types)
            & (terrain_levels >= 0)
            & (terrain_levels < num_levels)
            & (orientation_bins >= 0)
            & (orientation_bins < num_bins)
        )

        if not valid_labels.all():
            ids = ids[valid_labels]
            success = success[valid_labels]
            terrain_types = terrain_types[valid_labels]
            terrain_levels = terrain_levels[valid_labels]
            orientation_bins = orientation_bins[valid_labels]
            sampler_groups = sampler_groups[valid_labels]

        if ids.numel() == 0:
            return

        # Global fallen-episode EMA is logging only; it does not control either
        # task dimension and therefore cannot couple the curricula.
        batch_recovery_rate = success.float().mean().item()
        global_ema_alpha = float(getattr(
            cfg,
            "recovery_rate_ema_alpha",
            0.10,
        ))
        if not self.recovery_ema_initialized:
            self.recovery_rate_ema = batch_recovery_rate
            self.recovery_ema_initialized = True
        else:
            self.recovery_rate_ema = (
                (1.0 - global_ema_alpha) * self.recovery_rate_ema
                + global_ema_alpha * batch_recovery_rate
            )

        # ------------------------------------------------------------
        # Cell competence statistics: all sampling groups contribute.
        # ------------------------------------------------------------
        flat_cell = (
            (terrain_types * num_levels + terrain_levels) * num_bins
            + orientation_bins
        )
        cell_count = torch.bincount(
            flat_cell,
            minlength=num_types * num_levels * num_bins,
        )
        cell_success = torch.bincount(
            flat_cell,
            weights=success.float(),
            minlength=num_types * num_levels * num_bins,
        )
        active_cells = cell_count > 0

        flat_trials = self.recovery_cell_trials.view(-1)
        flat_ema = self.recovery_cell_success_ema.view(-1)
        batch_rate = cell_success[active_cells] / cell_count[active_cells].float()
        old_trials = flat_trials[active_cells]
        alpha = float(getattr(cfg, "cell_success_ema_alpha", 0.10))
        old_ema = flat_ema[active_cells]
        updated_ema = (1.0 - alpha) * old_ema + alpha * batch_rate
        updated_ema = torch.where(old_trials == 0, batch_rate, updated_ema)
        flat_ema[active_cells] = updated_ema
        flat_trials += cell_count.long()

        curriculum_enabled = (
            bool(getattr(cfg, "recovery_curriculum", False))
            and bool(cfg.curriculum)
            and bool(self.custom_origins)
        )
        if not curriculum_enabled:
            return

        # ------------------------------------------------------------
        # Promotion evidence: frontier group + non-stale frontier only.
        # ------------------------------------------------------------
        current_frontier = self.terrain_frontier[terrain_types]
        vote = (
            (sampler_groups == 0)
            & (terrain_levels == current_frontier)
        )

        if vote.any():
            vote_key = terrain_types[vote] * num_bins + orientation_bins[vote]
            vote_count = torch.bincount(
                vote_key,
                minlength=num_types * num_bins,
            ).view(num_types, num_bins)
            vote_success = torch.bincount(
                vote_key,
                weights=success[vote].float(),
                minlength=num_types * num_bins,
            ).view(num_types, num_bins)
            self.frontier_window_trials += vote_count.long()
            self.frontier_window_successes += vote_success

        min_trials = max(
            1,
            int(getattr(cfg, "frontier_min_trials_per_bin", 64)),
        )
        ready = (self.frontier_window_trials >= min_trials).all(dim=1)
        ready_types = ready.nonzero(as_tuple=False).flatten()

        if ready_types.numel() == 0:
            return

        rates = (
            self.frontier_window_successes[ready_types]
            / self.frontier_window_trials[ready_types].float().clamp_min(1.0)
        )
        mean_rate = rates.mean(dim=1)
        # There are exactly four bins, so the lower empirical quartile with
        # "lower" interpolation is the worst bin.  This prevents three easy
        # orientations from hiding failure in the fourth.
        lower_quartile = torch.sort(rates, dim=1).values[:, 0]

        self.frontier_last_mean_success[ready_types] = mean_rate
        self.frontier_last_lower_quartile[ready_types] = lower_quartile
        self.frontier_review_count[ready_types] += 1

        mean_threshold = float(getattr(
            cfg,
            "frontier_mean_promote_threshold",
            0.80,
        ))
        quartile_threshold = float(getattr(
            cfg,
            "frontier_lower_quartile_threshold",
            0.70,
        ))
        passed = (
            (mean_rate >= mean_threshold)
            & (lower_quartile >= quartile_threshold)
        )

        self.frontier_pass_windows[ready_types] = torch.where(
            passed,
            self.frontier_pass_windows[ready_types] + 1,
            torch.zeros_like(self.frontier_pass_windows[ready_types]),
        )

        required_windows = max(
            1,
            int(getattr(cfg, "frontier_required_windows", 2)),
        )
        promotable = (
            self.frontier_pass_windows[ready_types] >= required_windows
        ) & (
            self.terrain_frontier[ready_types]
            < self.recovery_curriculum_max_level
        )
        promoted_types = ready_types[promotable]

        if promoted_types.numel() > 0:
            old_levels = self.terrain_frontier[promoted_types].clone()
            self.terrain_frontier[promoted_types] += 1
            self.frontier_pass_windows[promoted_types] = 0
            self.recovery_total_promotions += int(promoted_types.numel())
            print(
                "[RECOVERY TERRAIN FRONTIER]",
                f"columns={promoted_types.detach().cpu().tolist()}",
                f"levels={old_levels.detach().cpu().tolist()} -> "
                f"{self.terrain_frontier[promoted_types].detach().cpu().tolist()}",
            )

        # Each reviewed column starts a fresh, balanced four-bin window.
        self.frontier_window_successes[ready_types] = 0.0
        self.frontier_window_trials[ready_types] = 0

        tracking = self.extras.setdefault("recovery_tracking", {})
        tracking["fallen_episode_success_ema"] = float(self.recovery_rate_ema)
        tracking["frontier_level_mean"] = self.terrain_frontier.float().mean().item()
        tracking["frontier_level_min"] = self.terrain_frontier.min().item()
        tracking["frontier_level_max"] = self.terrain_frontier.max().item()
        tracking["frontier_review_mean_success"] = mean_rate.mean().item()
        tracking["frontier_review_lower_quartile"] = lower_quartile.mean().item()
        tracking["frontier_promotions_total"] = int(self.recovery_total_promotions)
        for bin_index in range(num_bins):
            tracking[f"frontier_bin_{bin_index}_success"] = (
                rates[:, bin_index].mean().item()
            )

    def _get_orientation_rejection_mask(self, gravity_z, orientation_bins):
        """
        Return True when a candidate is outside its requested fallen bin.

        projected-gravity interpretation:
            upright:  gravity_z ~= -1
            sideways: gravity_z ~=  0
            inverted: gravity_z ~= +1

        Bins:
            0 strongly inverted:    [ 0.70,  1.00]
            1 partially inverted:   [ 0.20,  0.70)
            2 sideways:             [-0.20,  0.20)
            3 near-upright fallen:  [-0.80, -0.20)
        """

        if gravity_z.shape != orientation_bins.shape:
            raise ValueError("gravity_z and orientation_bins must have equal shape")

        if ((orientation_bins < 0) | (orientation_bins > 3)).any():
            raise ValueError("orientation bin must be in [0, 3]")

        lower = torch.tensor(
            [0.70, 0.20, -0.20, -0.80],
            device=gravity_z.device,
            dtype=gravity_z.dtype,
        )[orientation_bins]
        upper = torch.tensor(
            [1.0001, 0.70, 0.20, -0.20],
            device=gravity_z.device,
            dtype=gravity_z.dtype,
        )[orientation_bins]

        return (gravity_z < lower) | (gravity_z >= upper)

    def _write_curriculum_grid_snapshot(self):
        """
        Write synchronized per-environment curriculum state for the dashboard.

        Saved values:
            terrain_level:
                Current terrain difficulty row for each training environment.

            terrain_type:
                Fixed terrain column/type assigned to each environment.

            episode_orientation_bin:
                Fixed orientation category sampled for the current episode.

            sampler_group:
                0 frontier, 1 replay, 2 next-level probe, 3 coverage.

            terrain_frontier:
                Promoted difficulty frontier for every physical terrain column.

            terrain_sampling_stage:
                Per-column stage: 0 bootstrap, 1 developing, 2 mature.

            cell_success_ema / cell_trials:
                Statistics for (terrain column, level, orientation bin).
        """

        # Match the existing recovery-grid write frequency.
        if self.common_step_counter % 50 != 0:
            return

        if not self.cfg.env.train_recovery:
            return

        required_attributes = (
            "terrain_levels",
            "terrain_types",
            "episode_orientation_bin",
            "episode_sampler_group",
            "terrain_frontier",
            "terrain_sampling_stage",
            "recovery_cell_success_ema",
            "recovery_cell_trials",
        )

        if not all(hasattr(self, name) for name in required_attributes):
            return


        if hasattr(self, "recovery_grid_path"):
            grid_path = Path(self.recovery_grid_path).with_name("curriculum_grid.npz")
        else:
            grid_path = Path("curriculum_grid.npz")

        num_train_envs = int(self.num_train_envs)

        # Write atomically so Dash does not read a partially written archive.
        temporary_path = grid_path.with_name(f"{grid_path.stem}.tmp.npz")

        np.savez_compressed(
            temporary_path,
            curriculum_schema_version=np.asarray(2, dtype=np.int16),
            step=np.asarray(int(self.common_step_counter), dtype=np.int64),
            terrain_level=self.terrain_levels[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            terrain_type=self.terrain_types[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            episode_orientation_bin=self.episode_orientation_bin[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            sampler_group=self.episode_sampler_group[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            terrain_frontier=self.terrain_frontier.detach().cpu().numpy().astype(np.int16),
            cell_success_ema=self.recovery_cell_success_ema.detach().cpu().numpy().astype(np.float32),
            cell_trials=self.recovery_cell_trials.detach().cpu().numpy().astype(np.int32),
            frontier_window_successes=self.frontier_window_successes.detach().cpu().numpy().astype(np.float32),
            frontier_window_trials=self.frontier_window_trials.detach().cpu().numpy().astype(np.int32),
            frontier_pass_windows=self.frontier_pass_windows.detach().cpu().numpy().astype(np.int16),
            frontier_review_count=self.frontier_review_count.detach().cpu().numpy().astype(np.int32),
            frontier_last_mean_success=self.frontier_last_mean_success.detach().cpu().numpy().astype(np.float32),
            frontier_last_worst_bin_success=self.frontier_last_lower_quartile.detach().cpu().numpy().astype(np.float32),
            terrain_sampling_stage=self.terrain_sampling_stage.detach().cpu().numpy().astype(np.int16),
            env_sampling_stage=self.terrain_sampling_stage[
                self.terrain_types[:num_train_envs]
            ].detach().cpu().numpy().astype(np.int16),
            # Backward-compatible key.  It is now a per-column vector rather
            # than the old global scalar.
            sampling_stage=self.terrain_sampling_stage.detach().cpu().numpy().astype(np.int16),
            # Backward-compatible keys for an older dashboard.  "phase" now
            # contains the bin ID and -1 means there is no global phase.
            episode_orientation_phase=self.episode_orientation_phase[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            global_orientation_phase=np.asarray(-1, dtype=np.int16),
            recovery_success_ema=np.asarray(float(self.recovery_rate_ema), dtype=np.float32),
            orientation_ema=np.asarray(float(self.recovery_rate_ema), dtype=np.float32),
            terrain_successes=self.terrain_curriculum_successes[:num_train_envs].detach().cpu().numpy().astype(np.float32),
            terrain_trials=self.terrain_curriculum_trials[:num_train_envs].detach().cpu().numpy().astype(np.int16),
            num_terrain_levels=np.asarray(int(self.cfg.terrain.num_rows), dtype=np.int16,),
            num_terrain_types=np.asarray(int(self.cfg.terrain.num_cols), dtype=np.int16),
            num_orientation_bins=np.asarray(4, dtype=np.int16),
            frontier_min_trials_per_bin=np.asarray(
                int(getattr(self.cfg.terrain, "frontier_min_trials_per_bin", 64)),
                dtype=np.int32,
            ),
            frontier_mean_promote_threshold=np.asarray(
                float(getattr(self.cfg.terrain, "frontier_mean_promote_threshold", 0.80)),
                dtype=np.float32,
            ),
            frontier_worst_bin_threshold=np.asarray(
                float(getattr(self.cfg.terrain, "frontier_lower_quartile_threshold", 0.70)),
                dtype=np.float32,
            ),
            frontier_required_windows=np.asarray(
                int(getattr(self.cfg.terrain, "frontier_required_windows", 2)),
                dtype=np.int16,
            ),
            curriculum_max_level=np.asarray(
                int(self.recovery_curriculum_max_level),
                dtype=np.int16,
            ),
            mature_frontier_level=np.asarray(
                int(getattr(self.cfg.terrain, "mature_frontier_level", 3)),
                dtype=np.int16,
            ),
        )

        temporary_path.replace(grid_path)
