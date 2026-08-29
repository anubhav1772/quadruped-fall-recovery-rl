# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

import numpy as np

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain
# from .legged_robot_config import Cfg
from .fall_recovery_config_tr import FallRecoveryConfig as recoveryCfg
from .go1_loco_config import LocoCfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LocoCfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
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
        self.loco_cfg = cfg              # locomotion
        self.recovery_cfg = recoveryCfg  # recovery
        self.eval_cfg = eval_cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict

        assert self.loco_cfg.control.control_type == self.recovery_cfg.control.control_type
        assert self.loco_cfg.control.action_scale == self.recovery_cfg.control.action_scale
        assert self.loco_cfg.control.hip_scale_reduction == self.recovery_cfg.control.hip_scale_reduction
        assert self.loco_cfg.control.decimation == self.recovery_cfg.control.decimation
        assert self.loco_cfg.env.num_actions == self.recovery_cfg.env.num_actions

        if eval_cfg is not None: self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        # self.rand_buffers_eval = self._init_custom_buffers__(self.num_eval_envs)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        # self._prepare_reward_function()
        (
            self.loco_reward_container,
            self.loco_reward_scales,
            self.loco_reward_functions,
            self.loco_reward_names,
            self.loco_episode_sums,
            self.loco_episode_sums_eval,
        ) = self._prepare_reward_function(self.loco_cfg)

        (
            self.recovery_reward_container,
            self.recovery_reward_scales,
            self.recovery_reward_functions,
            self.recovery_reward_names,
            self.recovery_episode_sums,
            self.recovery_episode_sums_eval,
        ) = self._prepare_reward_function(self.recovery_cfg)

        # Locomotion-only command statistics
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in (
                list(self.loco_reward_scales.keys())
                + [
                    "lin_vel_raw",
                    "ang_vel_raw",
                    "lin_vel_residual",
                    "ang_vel_residual",
                    "ep_timesteps",
                ]
            )
        }

        self.init_done = True
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        self.applied_recovery_mode = self.recovery_mode.clone()
        clip_actions = float(self.cfg.normalization.clip_actions)
        effective_clip = torch.full((self.num_envs, 1), clip_actions, device=self.device, dtype=actions.dtype)

        if self.recovery_mode.any():
            recovery_clip = float(self.recovery_cfg.normalization.clip_actions)
            effective_clip[self.recovery_mode] = min(recovery_clip, clip_actions)

            g_now = quat_rotate_inverse(self.root_states[:, 3:7], self.gravity_vec)
            relative_base_height = self.root_states[:, 2] - self._get_local_terrain_height()

            near_crouch_action_clip = getattr(self.recovery_cfg.env, "near_crouch_action_clip", None)
            near_stand_action_clip = getattr(self.recovery_cfg.env, "near_stand_action_clip", None)

            if near_crouch_action_clip is not None:
                near_crouch = self.recovery_mode & (g_now[:, 2] < -0.45) & (relative_base_height > 0.18)
                effective_clip[near_crouch] = min(float(near_crouch_action_clip), clip_actions)

            if near_stand_action_clip is not None:
                near_stand = self.recovery_mode & (g_now[:, 2] < -0.85) & (relative_base_height > 0.30)
                effective_clip[near_stand] = min(float(near_stand_action_clip), clip_actions)

        self.actions = torch.clamp(actions, -effective_clip, effective_clip).to(self.device)

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

        recovery_clip_obs = self.recovery_cfg.normalization.clip_observations
        self.recovery_obs_buf = torch.clip(self.recovery_obs_buf, -recovery_clip_obs, recovery_clip_obs)

        if self.recovery_privileged_obs_buf is not None:
            self.recovery_privileged_obs_buf = torch.clip(self.recovery_privileged_obs_buf, -recovery_clip_obs, recovery_clip_obs)

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

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]

        self._post_physics_step_callback()

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.recovery_measured_heights = self._get_recovery_heights(all_env_ids)

        self.check_recovery_success()
        self.update_recovery_mode()

        # compute observations, rewards, resets, ...
        self.check_termination()

        # Compute Energy Consumption
        self.power_consume = torch.sum(torch.abs(self.dof_vel * self.torques), dim=1)
        # self.cot = torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1) / (21.5 * 9.81 * torch.norm(self.base_lin_vel[:, 0:2], dim=1))
        # self.dof_acc[:] = (self.dof_vel[:] - self.last_dof_vel[:]) / self.dt

        # self.compute_reward()
        mode = self.applied_recovery_mode
        loco_mask = ~mode
        recovery_mask = mode

        self.rew_buf.zero_()

        if loco_mask.any():
            self.rew_buf += self.compute_locomotion_reward(loco_mask)

        if recovery_mask.any():
            self.rew_buf += self.compute_recovery_reward(recovery_mask)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if env_ids.numel() > 0:
            self.base_pos[env_ids] = self.root_states[env_ids, :3]
            self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

            self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
            self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
            self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

            self.recovery_measured_heights[env_ids] = self._get_recovery_heights(env_ids)

        self.obs_buf, self.privileged_obs_buf = self.compute_observations(self.loco_cfg)
        self.recovery_obs_buf, self.recovery_privileged_obs_buf = self.compute_observations(self.recovery_cfg)

        assert self.obs_buf.shape[1] == self.loco_cfg.env.num_observations
        assert self.privileged_obs_buf.shape[1] == self.loco_cfg.env.num_privileged_obs
        assert self.recovery_obs_buf.shape[1] == self.recovery_cfg.env.num_observations
        assert self.recovery_privileged_obs_buf.shape[1] == self.recovery_cfg.env.num_privileged_obs

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf

        relative_base_height = self.root_states[:, 2] - self._get_local_terrain_height()

        lin_vel_norm = torch.norm(self.root_states[:, 7:10], dim=1)
        ang_vel_norm = torch.norm(self.root_states[:, 10:13], dim=1)

        bad_state = (
            torch.isnan(self.root_states).any(dim=1)
            | torch.isinf(self.root_states).any(dim=1)
            | torch.isnan(self.dof_pos).any(dim=1)
            | torch.isinf(self.dof_pos).any(dim=1)
            | torch.isnan(self.dof_vel).any(dim=1)
            | torch.isinf(self.dof_vel).any(dim=1)
            | (relative_base_height < -1.0)
            | (relative_base_height > 2.0)
            | (lin_vel_norm > 20.0)
            | (ang_vel_norm > 50.0)
        )

        self.reset_buf |= bad_state


    def compute_fall_detected(self):
        if self.termination_contact_indices.numel() > 0:
            contact_fall = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        else:
            contact_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        relative_base_height = self.root_states[:, 2] - self._get_local_terrain_height()

        if self.cfg.rewards.use_terminal_body_height:
            height_fall = relative_base_height < self.cfg.rewards.terminal_body_height
        else:
            height_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        orientation_fall = -self.projected_gravity[:, 2] < 0.4

        return contact_fall | height_fall | orientation_fall


    def compute_handoff_ready(self):
        cfg = self.recovery_cfg.rewards

        foot_contact = self.contact_forces[:, self.feet_indices, 2] > cfg.recovery_contact_force_threshold
        foot_xy_vel = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        non_slipping_feet = foot_contact & (foot_xy_vel < 0.10)

        posture_error = torch.norm(self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:, :self.num_actuated_dof], dim=1)

        relative_base_height = self.root_states[:, 2] - self._get_local_terrain_height()

        nonfoot_force = torch.norm(self.contact_forces[:, self.base_contact_indices, :], dim=-1).max(dim=1).values
        no_nonfoot_contact = nonfoot_force < cfg.recovery_nonfoot_contact_threshold

        handoff_ready = (
            (self.projected_gravity[:, 2] < cfg.recovery_upright_threshold)
            & (relative_base_height > 0.30)
            & (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.15)
            & (torch.abs(self.base_lin_vel[:, 2]) < 0.10)
            & (torch.norm(self.base_ang_vel, dim=1) < 0.50)
            & (posture_error < 1.80)
            & (non_slipping_feet.sum(dim=1) >= 3)
            & no_nonfoot_contact
        )

        return handoff_ready


    def check_recovery_success(self):
        cfg = self.recovery_cfg.rewards

        upright = self.projected_gravity[:, 2] < cfg.recovery_upright_threshold

        local_terrain_height = self._get_local_terrain_height()
        relative_base_height = self.root_states[:, 2] - local_terrain_height
        stable_height = relative_base_height > cfg.recovery_height_success

        low_velocity = torch.norm(self.base_lin_vel[:, :2], dim=1) < cfg.recovery_lin_vel_threshold
        low_vertical_velocity = torch.abs(self.root_states[:, 9]) < cfg.recovery_vertical_vel_threshold
        low_ang_vel = torch.norm(self.base_ang_vel, dim=1) < cfg.recovery_ang_vel_threshold

        posture_error = torch.norm(self.dof_pos - self.default_dof_pos, dim=1)
        good_posture = posture_error < cfg.recovery_posture_threshold

        foot_contact = self.contact_forces[:, self.feet_indices, 2] > cfg.recovery_contact_force_threshold
        foot_xy_vel = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        non_slipping_feet = foot_contact & (foot_xy_vel < cfg.recovery_foot_slip_vel_threshold)

        if getattr(cfg, "require_non_slipping_contacts", False):
            stable_contacts = non_slipping_feet.sum(dim=1) >= cfg.recovery_min_foot_contacts
        else:
            stable_contacts = foot_contact.sum(dim=1) >= cfg.recovery_min_foot_contacts

        if self.base_contact_indices.numel() > 0:
            nonfoot_force = torch.norm(self.contact_forces[:, self.base_contact_indices, :], dim=-1)
            no_nonfoot_contact = nonfoot_force.max(dim=1).values < cfg.recovery_nonfoot_contact_threshold
        else:
            no_nonfoot_contact = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        recovered = upright & stable_height & low_velocity & low_vertical_velocity & low_ang_vel & good_posture & stable_contacts & no_nonfoot_contact

        recovered &= self.recovery_mode

        self.recovery_counter[recovered] += 1
        self.recovery_counter[~recovered] = 0

        stable_recovery = self.recovery_counter >= cfg.recovery_success_steps

        new_recovery = stable_recovery & (~self.recovered_flag)
        self.recovered_flag |= stable_recovery

        self.extras["recovery_success_frac"] = stable_recovery.float().mean().item()
        self.extras["new_recovery_frac"] = new_recovery.float().mean().item()

        return stable_recovery


    def update_recovery_mode(self):
        fall_detected = self.compute_fall_detected()

        new_recoveries = fall_detected & (~self.recovery_mode)

        self.recovery_mode[new_recoveries] = True
        self.recovery_counter[new_recoveries] = 0
        self.recovered_flag[new_recoveries] = False
        self.handoff_counter[new_recoveries] = 0

        handoff_ready = self.compute_handoff_ready()

        self.handoff_counter = torch.where(self.recovery_mode & handoff_ready, self.handoff_counter + 1, torch.zeros_like(self.handoff_counter))

        handoff_steps = self.recovery_cfg.rewards.recovery_success_steps
        new_handoffs = self.recovery_mode & (self.handoff_counter >= handoff_steps)

        self.recovery_mode[new_handoffs] = False
        self.recovery_counter[new_handoffs] = 0
        self.handoff_counter[new_handoffs] = 0

        self.extras["recovery_mode_frac"] = self.recovery_mode.float().mean().item()
        self.extras["fall_detected_frac"] = fall_detected.float().mean().item()
        self.extras["handoff_ready_frac"] = handoff_ready.float().mean().item()
        self.extras["new_handoff_frac"] = new_handoffs.float().mean().item()


    def _sync_control_history_after_reset(self, env_ids):
        if len(env_ids) == 0:
            return

        q_reset = self.dof_pos[env_ids, :self.num_dof].clone()

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0

        self.joint_pos_target[env_ids, :self.num_dof] = q_reset
        self.last_joint_pos_target[env_ids, :self.num_dof] = q_reset
        self.last_last_joint_pos_target[env_ids, :self.num_dof] = q_reset

        self.last_dof_vel[env_ids] = 0.0
        self.torques[env_ids] = 0.0

        if hasattr(self, "joint_pos_err_last"):
            self.joint_pos_err_last[env_ids] = 0.0

        if hasattr(self, "joint_pos_err_last_last"):
            self.joint_pos_err_last_last[env_ids] = 0.0

        if hasattr(self, "joint_vel_last"):
            self.joint_vel_last[env_ids] = 0.0

        if hasattr(self, "joint_vel_last_last"):
            self.joint_vel_last_last[env_ids] = 0.0


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

        self.recovery_mode[env_ids] = False
        self.recovery_counter[env_ids] = 0
        self.recovered_flag[env_ids] = False
        self.handoff_counter[env_ids] = 0

        # reset robot states
        self._resample_commands(env_ids)
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)
        self._sync_control_history_after_reset(env_ids)

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
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])
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

    def _compute_reward(self, cfg, reward_container, reward_functions, reward_names, reward_scales, episode_sums, active_mask, command_sums=None):
        rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        rew_buf_pos = torch.zeros_like(rew_buf)
        rew_buf_neg = torch.zeros_like(rew_buf)

        for i in range(len(reward_functions)):
            name = reward_names[i]

            rew = reward_functions[i]() * reward_scales[name]

            # Only this controller's environments contribute.
            rew = rew * active_mask.float()

            rew_buf += rew

            if torch.sum(rew) >= 0:
                rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                rew_buf_neg += rew

            episode_sums[name] += rew

            # Only locomotion passes command_sums.
            if command_sums is not None:
                if name in ["tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"]:
                    command_sums[name][active_mask] += (reward_scales[name] + rew[active_mask])
                else:
                    command_sums[name][active_mask] += rew[active_mask]

        if cfg.rewards.only_positive_rewards:
            rew_buf = torch.clip(rew_buf, min=0.)

        elif cfg.rewards.only_positive_rewards_ji22_style:
            rew_buf = rew_buf_pos * torch.exp(rew_buf_neg / cfg.rewards.sigma_rew_neg)

        episode_sums["total"] += rew_buf

        if "termination" in reward_scales:
            rew = reward_container._reward_termination() * reward_scales["termination"] * active_mask.float()
            rew_buf += rew
            episode_sums["termination"] += rew

            if command_sums is not None:
                command_sums["termination"][active_mask] += rew[active_mask]

        return rew_buf

    def compute_reward(self):
        mode = self.applied_recovery_mode

        loco_mask = ~mode
        recovery_mask = mode

        self.rew_buf.zero_()

        if loco_mask.any():
            loco_rew = self._compute_reward(
                cfg=self.loco_cfg,
                reward_container=self.loco_reward_container,
                reward_functions=self.loco_reward_functions,
                reward_names=self.loco_reward_names,
                reward_scales=self.loco_reward_scales,
                episode_sums=self.loco_episode_sums,
                active_mask=loco_mask,
                command_sums=self.command_sums,
            )

            self.rew_buf += loco_rew

        if recovery_mask.any():
            recovery_rew = self._compute_reward(
                cfg=self.recovery_cfg,
                reward_container=self.recovery_reward_container,
                reward_functions=self.recovery_reward_functions,
                reward_names=self.recovery_reward_names,
                reward_scales=self.recovery_reward_scales,
                episode_sums=self.recovery_episode_sums,
                active_mask=recovery_mask,
                command_sums=None,
            )

            self.rew_buf += recovery_rew

        # Locomotion command statistics only
        if loco_mask.any():
            self.command_sums["lin_vel_raw"][loco_mask] += self.base_lin_vel[loco_mask, 0]
            self.command_sums["ang_vel_raw"][loco_mask] += self.base_ang_vel[loco_mask, 2]
            self.command_sums["lin_vel_residual"][loco_mask] += (self.base_lin_vel[loco_mask, 0] - self.commands[loco_mask, 0]) ** 2
            self.command_sums["ang_vel_residual"][loco_mask] += (self.base_ang_vel[loco_mask, 2] - self.commands[loco_mask, 2]) ** 2
            self.command_sums["ep_timesteps"][loco_mask] += 1

    def compute_observations(self, cfg):
        """ Computes observations
        """
        obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                             :self.num_actuated_dof]) * cfg.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actuated_dof] * cfg.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        if cfg.env.observe_command:
            obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                                 :self.num_actuated_dof]) * cfg.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * cfg.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if cfg.env.observe_two_prev_actions:
            obs_buf = torch.cat((obs_buf,
                                self.last_actions), dim=-1)

        if cfg.env.observe_timing_parameter:
            obs_buf = torch.cat((obs_buf,
                                self.gait_indices.unsqueeze(1)), dim=-1)

        if cfg.env.observe_clock_inputs:
            obs_buf = torch.cat((obs_buf,
                                self.clock_inputs), dim=-1)

        if cfg.env.observe_desired_contact_states:
            obs_buf = torch.cat((obs_buf,
                                self.desired_contact_states), dim=-1)

        if cfg.env.observe_vel:
            if cfg.commands.global_reference:
                obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * cfg.obs_scales.lin_vel,
                                          self.base_ang_vel * cfg.obs_scales.ang_vel,
                                    obs_buf), dim=-1)
            else:
                obs_buf = torch.cat((self.base_lin_vel * cfg.obs_scales.lin_vel,
                                    self.base_ang_vel * cfg.obs_scales.ang_vel,
                                    obs_buf), dim=-1)

        if cfg.env.observe_only_ang_vel:
            obs_buf = torch.cat((self.base_ang_vel * cfg.obs_scales.ang_vel,
                                obs_buf), dim=-1)

        if cfg.env.observe_only_lin_vel:
            obs_buf = torch.cat((self.base_lin_vel * cfg.obs_scales.lin_vel,
                                obs_buf), dim=-1)

        if cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            obs_buf = torch.cat((obs_buf,
                                      heading), dim=-1)

        if cfg.env.observe_contact_states:
            obs_buf = torch.cat((obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        # print(f"obs shape: {obs_buf.shape}")

        # build privileged obs

        privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if cfg.env.priv_observe_link_masses:

            # [N, 4]
            # [trunk_mass_norm, hip_mass_norm, thigh_mass_norm, calf_mass_norm]
            mass_groups_norm = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

            # 0) TRUNK / BASE MASS
            # Current absolute trunk mass
            trunk_mass = self.mass_groups[:, 0]

            trunk_range = torch.tensor(cfg.normalization.trunk_mass_range, device=self.device, dtype=torch.float)

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
            lm_ranges = torch.tensor(cfg.normalization.link_mass_ranges, device=self.device, dtype=torch.float)

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

            privileged_obs_buf = torch.cat((privileged_obs_buf, mass_groups_norm), dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf, mass_groups_norm),dim=1)

        if cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(cfg.normalization.friction_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale),
                                            dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
        # if cfg.env.priv_observe_ground_friction:
        #     self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
        #     ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
        #         cfg.normalization.ground_friction_range)
        #     privileged_obs_buf = torch.cat((privileged_obs_buf,
        #                                          (self.ground_friction_coeffs.unsqueeze(
        #                                              1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                         dim=1)
        #     next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
        #                                               (self.ground_friction_coeffs.unsqueeze(
        #                                               1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
        #                                               dim=1)

        if cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(cfg.normalization.restitution_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(cfg.normalization.added_mass_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)

        if cfg.env.priv_observe_com_displacement:
            # randomized COM offset parameter injected into simulation
            com_displacements_scale, com_displacements_shift = get_scale_shift(cfg.normalization.com_displacement_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(cfg.normalization.motor_strength_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(cfg.normalization.motor_offset_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)

        if cfg.env.priv_observe_com_position:
            # center of mass position relative to the base frame
            body_positions_world = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, :, :3]
            body_masses = self.rigid_body_masses.unsqueeze(-1)
            # body_masses = self.body_masses.unsqueeze(0).unsqueeze(-1)
            total_mass = body_masses.sum(dim=1).clamp_min(1e-8)
            com_world = (body_positions_world * body_masses).sum(dim=1) / total_mass
            com_in_base = quat_rotate_inverse(self.base_quat, com_world - self.base_pos)
            com_position_scale, com_position_shift = get_scale_shift(cfg.normalization.com_position_range)
            normalized_com_position = (com_in_base - com_position_shift) * com_position_scale
            privileged_obs_buf = torch.cat((privileged_obs_buf, normalized_com_position), dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf, normalized_com_position), dim=1)


        # if cfg.env.priv_observe_Kp_factor:
        #     Kp_factor_scale, Kp_factor_shift = get_scale_shift(cfg.normalization.Kp_factor_range)
        #     privileged_obs_buf = torch.cat((privileged_obs_buf,
        #                                          (self.Kp_factors - Kp_factor_shift) * Kp_factor_scale),
        #                                         dim=1)
        #     next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
        #                                               (self.Kp_factors - Kp_factor_shift) * Kp_factor_scale),
        #                                              dim=1)
        # if cfg.env.priv_observe_Kd_factor:
        #     Kd_factor_scale, Kd_factor_shift = get_scale_shift(cfg.normalization.Kd_factor_range)
        #     privileged_obs_buf = torch.cat((privileged_obs_buf,
        #                                          (self.Kd_factors - Kd_factor_shift) * Kd_factor_scale),
        #                                         dim=1)
        #     next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
        #                                               (self.Kd_factors - Kd_factor_shift) * Kd_factor_scale),
        #                                              dim=1)

        if cfg.env.priv_observe_Kpd_factor:
            # Current implementation randomizes:
            # - one shared Kp factor per environment
            # - one shared Kd factor per environment
            #
            # Although tensors are shape [N, 12], all joint values are identical.
            # Therefore expose only 2 privileged dimensions:
            # [normalized_Kp_factor, normalized_Kd_factor]

            Kp_factor_scale, Kp_factor_shift = get_scale_shift(cfg.normalization.Kp_factor_range)
            Kd_factor_scale, Kd_factor_shift = get_scale_shift(cfg.normalization.Kd_factor_range)
            normalized_Kp_factor = (self.Kp_factors[:, 0:1] - Kp_factor_shift) * Kp_factor_scale
            normalized_Kd_factor = (self.Kd_factors[:, 0:1] - Kd_factor_shift) * Kd_factor_scale
            normalized_Kpd_factors = torch.cat((normalized_Kp_factor, normalized_Kd_factor), dim=1)
            privileged_obs_buf = torch.cat((privileged_obs_buf, normalized_Kpd_factors), dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf, normalized_Kpd_factors), dim=1)

        if cfg.env.priv_observe_contact_forces:
            contact_forces_scale, contact_forces_shift = get_scale_shift(cfg.normalization.contact_force_range)
            foot_contact_forces = self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 (foot_contact_forces - contact_forces_shift) * contact_forces_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      (foot_contact_forces - contact_forces_shift) * contact_forces_scale),
                                                     dim=1)
        if cfg.env.priv_observe_contact_states:
            foot_contact_states = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()  # [num_envs, 4]
            privileged_obs_buf = torch.cat((privileged_obs_buf, foot_contact_states), dim=-1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf, foot_contact_states), dim=-1)

        if cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(cfg.normalization.body_height_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(cfg.normalization.body_velocity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(cfg.normalization.gravity_range)
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                            (self.gravities - gravity_shift) / gravity_scale),
                                            dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf,
                                                (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if cfg.env.priv_observe_clock_inputs:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if cfg.env.priv_observe_desired_contact_states:
            privileged_obs_buf = torch.cat((privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        if cfg.env.priv_observe_heightmap:
            heights = self.recovery_measured_heights if getattr(cfg.env, "train_recovery", False) else self.measured_heights
            body_height = self.root_states[:, 2:3]

            # Unclamped base-to-terrain vertical clearance.
            heights_relative_raw = body_height - heights

            height_min, height_max = cfg.normalization.relative_height_range

            # Log before clamping so range violations remain visible.
            if self.common_step_counter % 50 == 0:
                tracking = self.extras.setdefault("recovery_tracking", {})

                outside_range = (heights_relative_raw < height_min) | (heights_relative_raw > height_max)

                tracking["heightmap_relative_min"] = heights_relative_raw.min().item()
                tracking["heightmap_relative_mean"] = heights_relative_raw.mean().item()
                tracking["heightmap_relative_max"] = heights_relative_raw.max().item()
                tracking["heightmap_outside_range_frac"] = outside_range.float().mean().item()

            heights_relative = torch.clamp(heights_relative_raw, height_min, height_max)

            height_scale, height_shift = get_scale_shift(cfg.normalization.relative_height_range)
            heights_normalized = (heights_relative - height_shift) * height_scale

            privileged_obs_buf = torch.cat((privileged_obs_buf, heights_normalized), dim=1)
            next_privileged_obs_buf = torch.cat((next_privileged_obs_buf, heights_normalized), dim=1)

        # print(f"priv obs shape: {self.privileged_obs_buf.shape}")
        assert privileged_obs_buf.shape[
                   1] == cfg.env.num_privileged_obs, f"num_privileged_obs ({cfg.env.num_privileged_obs}) != the number of privileged observations ({privileged_obs_buf.shape[1]}), you will discard data from the student!"

        return obs_buf, privileged_obs_buf

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

    # ------------- Callbacks --------------
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
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    # def _process_rigid_body_props(self, props, env_id):
    #     self.default_body_mass = props[0].mass

    #     props[0].mass = self.default_body_mass + self.payloads[env_id]
    #     props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
    #                                self.com_displacements[env_id, 2])
    #     return props


    def _process_rigid_body_props(self, props, env_id):
        if not hasattr(self, "default_body_mass"):
            self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]

        props[0].com = gymapi.Vec3(
            self.com_displacements[env_id, 0],
            self.com_displacements[env_id, 1],
            self.com_displacements[env_id, 2],
        )

        self.rigid_body_masses[env_id] = torch.tensor([p.mass for p in props], device=self.device, dtype=torch.float)

        m_body, m_hip, m_thigh, m_calf = self._compute_mass_groups_from_body_props(props)

        self.mass_groups[env_id, 0] = m_body
        self.mass_groups[env_id, 1] = m_hip
        self.mass_groups[env_id, 2] = m_thigh
        self.mass_groups[env_id, 3] = m_calf

        return props


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        locomotion_ids = (~self.recovery_mode).nonzero(as_tuple=False).flatten()
        self._call_train_eval(self._teleport_robots, locomotion_ids)

        # teleport robots to prevent falling off the edge
        # self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        # Freeze command resampling while an environment is recovering
        env_ids = env_ids[~self.recovery_mode[env_ids]]
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        # self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))
        self._call_train_eval(self._push_robots, locomotion_ids)

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

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
                                                                        3)  # shape: num_envs, num_bodies, xyz axis

        self.recovery_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.recovered_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.handoff_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.recovery_height_points = self._init_recovery_height_points()
        self.recovery_measured_heights = self._get_recovery_heights(all_env_ids)

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # , self.eval_cfg)
        self.add_noise = False

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
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

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # link mass randomization
        self.mass_groups = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.rigid_body_masses = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

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

    def _prepare_reward_function(self, cfg):
        """Prepare reward functions for one task/config."""

        from aliengo_gym.envs.rewards.locomotion_rewards import LocomotionRewards
        from aliengo_gym.envs.rewards.fall_recovery_rewards import FallRecoveryRewards

        reward_containers = {
            "LocomotionRewards": LocomotionRewards,
            "FallRecoveryRewards": FallRecoveryRewards,
        }

        reward_container = reward_containers[cfg.rewards.reward_container_name](self, cfg)

        # IMPORTANT: copy, otherwise you modify the config class dictionary.
        reward_scales = vars(cfg.reward_scales).copy()

        for key in list(reward_scales.keys()):
            if reward_scales[key] == 0:
                reward_scales.pop(key)
            else:
                reward_scales[key] *= self.dt

        reward_functions = []
        reward_names = []

        for name in reward_scales.keys():
            if name == "termination":
                continue

            reward_name = "_reward_" + name

            if not hasattr(reward_container, reward_name):
                print(f"Warning: reward {reward_name} has nonzero coefficient but was not found!")
            else:
                reward_names.append(name)
                reward_functions.append(getattr(reward_container, reward_name))

        episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in reward_scales.keys()
        }
        episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, equires_grad=False)
        episode_sums_eval = {
            name: -torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in reward_scales.keys()
        }
        episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        return (
            reward_container,
            reward_scales,
            reward_functions,
            reward_names,
            episode_sums,
            episode_sums_eval,
        )

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

        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

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

        # Video cameras: one probe camera per requested terrain column
        self.video_probe_env_ids = []
        self.video_probe_cameras = []
        self.video_probe_columns = []
        self.video_probe_index = -1

        self.record_env_id = 0
        self.rendering_camera = None

        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.camera_props.enable_tensors = False

            requested_columns = getattr(self.cfg.env, "video_probe_columns", list(range(self.cfg.terrain.num_cols)))

            for terrain_column in requested_columns:
                terrain_column = int(terrain_column)

                candidates = torch.nonzero(
                    self.terrain_types[:self.num_train_envs] == terrain_column,
                    as_tuple=False,
                ).flatten()

                if candidates.numel() == 0:
                    print(
                        f"[VIDEO WARNING] No training environment found "
                        f"for terrain column {terrain_column}"
                    )
                    continue

                # Select one representative environment from this column.
                candidate_index = candidates.numel() // 2
                env_id = int(candidates[candidate_index].item())

                camera = self.gym.create_camera_sensor(
                    self.envs[env_id],
                    self.camera_props,
                )

                if camera == -1:
                    raise RuntimeError(
                        f"Camera creation failed for video probe env {env_id}"
                    )

                origin = self.env_origins[env_id]
                ox = float(origin[0].item())
                oy = float(origin[1].item())
                oz = float(origin[2].item())

                self.gym.set_camera_location(
                    camera,
                    self.envs[env_id],
                    gymapi.Vec3(ox + 1.2, oy - 1.5, oz + 0.9),
                    gymapi.Vec3(ox, oy, oz),
                )

                self.video_probe_columns.append(terrain_column)
                self.video_probe_env_ids.append(env_id)
                self.video_probe_cameras.append(camera)

            if len(self.video_probe_cameras) == 0:
                raise RuntimeError("No video probe cameras were created")

            self.record_env_id = self.video_probe_env_ids[0]
            self.rendering_camera = self.video_probe_cameras[0]

            print("[VIDEO] Probe columns:", self.video_probe_columns)
            print("[VIDEO] Probe env IDs:", self.video_probe_env_ids)

            # Existing evaluation camera can remain if evaluation is used.
            if self.eval_cfg is not None:
                eval_env_id = self.num_train_envs

                self.rendering_camera_eval = self.gym.create_camera_sensor(
                    self.envs[eval_env_id],
                    self.camera_props,
                )

                self.gym.set_camera_location(
                    self.rendering_camera_eval,
                    self.envs[eval_env_id],
                    gymapi.Vec3(1.5, 1.0, 3.0),
                    gymapi.Vec3(0.0, 0.0, 0.0),
                )

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
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            self.video_frames.append(self.video_frame)

        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            if self.eval_cfg is not None:
                bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                             self.root_states[self.num_train_envs, 2]
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                self.video_frames_eval.append(self.video_frame_eval)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def start_recording_eval(self):
        self.complete_video_frames_eval = None
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

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

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


    def _init_recovery_height_points(self):
        cfg = self.recovery_cfg

        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)

        # Same ordering as the recovery training code.
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        num_points = grid_x.numel()

        cfg.env.num_height_points = num_points

        points = torch.zeros(self.num_envs, num_points, 3, device=self.device, requires_grad=False)

        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()

        return points

    def _sample_terrain_height_at_xy(self, xy_world):
        if xy_world.ndim != 3 or xy_world.shape[-1] != 2:
            raise ValueError(
                "xy_world must have shape [N, K, 2]"
            )

        num_envs, num_points, _ = xy_world.shape

        mesh_type = self.cfg.terrain.mesh_type

        if mesh_type == "plane":
            return torch.zeros(num_envs, num_points, device=self.device, dtype=self.root_states.dtype)

        if mesh_type == "none":
            raise RuntimeError("Cannot measure terrain height with mesh_type='none'.")

        grid_xy = (xy_world + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale

        px = torch.floor(grid_xy[..., 0]).long()
        py = torch.floor(grid_xy[..., 1]).long()

        px = torch.clamp(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clamp(py, 0, self.height_samples.shape[1] - 2)

        h00 = self.height_samples[px, py]
        h10 = self.height_samples[px + 1, py]
        h01 = self.height_samples[px, py + 1]
        sampled_height = torch.min(torch.min(h00, h10), h01)
        return sampled_height.to(self.root_states.dtype) * self.terrain.cfg.vertical_scale

    def _get_recovery_heights(self, env_ids):
        num_points = self.recovery_height_points.shape[1]

        points_world = (
            quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, num_points),
                self.recovery_height_points[env_ids],
            )
            + self.root_states[env_ids, :3].unsqueeze(1)
        )

        return self._sample_terrain_height_at_xy(points_world[:, :, :2])

    def _get_local_terrain_height(self):
        """
        Recovery support-aware terrain height.
        """

        heights = self.recovery_measured_heights

        if (not torch.is_tensor(heights) or heights.ndim != 2 or heights.shape[0] != self.num_envs):
            return torch.zeros(self.num_envs, device=self.device, dtype=self.root_states.dtype)

        # Terrain directly under the base.
        if not hasattr(self, "_recovery_center_height_index"):
            local_xy = self.recovery_height_points[0, :, :2]
            distance_sq = torch.sum(torch.square(local_xy), dim=1)
            self._recovery_center_height_index = int(torch.argmin(distance_sq).item())

        center_height = heights[:, self._recovery_center_height_index]

        if not (hasattr(self, "foot_positions") and hasattr(self, "contact_forces") and hasattr(self, "feet_indices")):
            return center_height

        # Terrain underneath each foot.
        foot_ground_height = self._sample_terrain_height_at_xy(self.foot_positions[:, :, :2])
        foot_fz = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
        contact_threshold = getattr(self.recovery_cfg.rewards, "recovery_contact_force_threshold", 1.0)
        load_score = torch.clamp((foot_fz - contact_threshold) / 12.0, 0.0, 1.0)
        load_sum = load_score.sum(dim=1)
        support_height = (load_score * foot_ground_height).sum(dim=1) / load_sum.clamp_min(1e-6)
        support_height = torch.where(load_sum > 1e-4, support_height, center_height)
        support_alpha = torch.clamp((load_sum - 0.5) / 2.5, 0.0, 1.0)
        local_terrain_height = (1.0 - support_alpha) * center_height + support_alpha * support_height

        # Same diagnostics as recovery implementation.
        self.local_height_center = center_height
        self.local_height_support = support_height
        self.local_height_support_alpha = support_alpha
        self.local_height_effective_load = load_sum
        self.local_height_foot_ground = foot_ground_height

        return local_terrain_height
