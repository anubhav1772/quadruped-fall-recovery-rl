import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

# base_contact      ->  non-foot contact
# stance_region     ->  foot geometry
# front_leg_error   ->  front-foot geometry
# stable_foot_support -> loaded and stationary feet

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def get_body_height(self):
        """
        Returns terrain-relative base height.
        """
        local_terrain_height = self.env._get_local_terrain_height()
        relative_base_height = self.env.root_states[:, 2] - local_terrain_height
        return relative_base_height

    # Helper functions:
    # - _upright_progress: soft orientation progress
    # - _height_progress: soft height progress
    # - _height_success: hard success height threshold
    # Dense rewards should use soft progress.
    # legged_robot.py success detection should use hard thresholds.
    def _upright_progress(self, sigma=None):
        """
        Soft measure of uprightness.
        Returns high value when projected gravity z is close to -1.
        Used for dense orientation shaping, not hard success detection.
        """
        gz = self.env.projected_gravity[:, 2]
        if sigma is None:
            sigma = self.env.cfg.rewards.upright_sigma_soft
        return torch.exp(-torch.square(gz + 1.0) / (2 * sigma ** 2))

    def _terminal_height_progress(self):
        """
        Stricter height progress for final recovery / handoff refinement.
        Gives little reward to low crouched states.
        """
        h = self.get_body_height()
        return torch.clamp((h - 0.26) / 0.07, 0.0, 1.0)

    def _height_progress(self):
        """
        Dense terrain-relative height progress.
        Returns 0 when body is too low and 1 near target standing height.
        This is deliberately softer than the hard recovery success threshold.
        """
        body_height = self.get_body_height()

        h_min = self.env.cfg.rewards.recovery_height_min
        h_target = self.env.cfg.rewards.recovery_height_target

        return torch.clamp((body_height - h_min) / (h_target - h_min), 0.0, 1.0)

    def _height_success(self):
        """
        Hard height check for true recovery success.
        Should match the stable_height condition used in legged_robot.py.
        """
        body_height = self.get_body_height()
        h_success = self.env.cfg.rewards.recovery_height_success
        return body_height > h_success

    def _foot_load_score(self):
        env = self.env
        cfg = env.cfg.rewards

        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

        loaded_threshold = getattr(cfg, "loaded_foot_force_threshold", 3.0)
        full_load_force = getattr(cfg, "full_load_force_threshold", 20.0)

        transition_width = max(full_load_force - loaded_threshold, 1.0e-6)

        return torch.clamp((foot_fz - loaded_threshold) / transition_width, 0.0, 1.0)

    def _smooth_support_count(self):
        """
        Smooth estimate of useful support feet.

        Gives credit only when a foot is:
        - carrying some vertical load
        - not sliding too fast in XY

        Returns approximately [0, 4].
        """
        env = self.env

        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

        # Softer than the previous (fz - 3) / 20.
        # This gives learning signal once the foot starts loading.
        load_score = torch.clamp((foot_fz - 1.0) / 12.0, 0.0, 1.0)

        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)
        no_slip_score = torch.exp(-2.0 * foot_xy_vel)

        return (load_score * no_slip_score).sum(dim=1)


    def _loaded_support_gate(self):
        loaded_count = self._foot_load_score().sum(dim=1)
        # Example:
        # Foot forces: [20, 20, 20, 0] N
        # Scores:      [1,  1,  1,  0]
        # loaded_count = 3
        # gate = 3 / 3 = 1
        # The gate reaches 1 when the summed support is equivalent
        # to approximately three fully loaded feet.
        return torch.clamp(loaded_count / 3.0, 0.0, 1.0)


    def _stable_support_gate(self):
        env = self.env
        cfg = env.cfg.rewards

        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)
        foot_contact = foot_fz > cfg.recovery_contact_force_threshold
        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)
        non_slipping_feet = foot_contact & (foot_xy_vel < cfg.recovery_foot_slip_vel_threshold)
        stable_count = non_slipping_feet.float().sum(dim=1)

        # Strict terminal support gate:
        # 0 below 2 stable feet, 1 at 3 or more stable feet.
        return torch.clamp((stable_count - 2.0) / 1.0, 0.0, 1.0)


    def _smooth_support_gate(self):
        """
        0 when support is poor, 1 when about 3 useful support feet exist.
        """
        smooth_support_count = self._smooth_support_count()
        return torch.clamp((smooth_support_count - 1.0) / 2.0, 0.0, 1.0)

    ######## dense success reward ########
    def _reward_recovery_progress(self):
        """
        Staged dense recovery reward:

        1. rotate toward upright,
        2. establish useful foot support,
        3. raise the body,
        4. combine all three.
        """
        env = self.env
        orientation_progress = torch.clamp((1.0 - env.projected_gravity[:, 2]) * 0.5, 0.0, 1.0)

        height_progress = self._height_progress()

        support_progress = torch.clamp(self._smooth_support_count() / 3.0, 0.0, 1.0)

        return (
            0.15 * orientation_progress
            + 0.30 * orientation_progress * height_progress
            + 0.25 * orientation_progress * support_progress
            + 0.30 * orientation_progress * height_progress * support_progress
        )

    ######## sparse success reward ########
    # def _reward_recovery_bonus(self):
    #     """
    #     Sparse one-shot success bonus.
    #     Fires only when legged_robot.py detects stable recovery and sets recovery_bonus_buf.
    #     """
    #     return self.env.recovery_bonus_buf

    def _reward_recovery_bonus(self):
        """
        Sparse recovery bonus.

        During terminal-stabilization training, this is usually disabled by setting
        recovery_bonus = 0.0 in reward_scales.

        When re-enabled later, the delay prevents the policy from receiving a success
        bonus immediately after being reset near standing. The robot must first survive
        for a short time after reset.
        """
        env = self.env

        delay_s = getattr(env.cfg.rewards, "recovery_bonus_delay_s", 0.0)
        delay_steps = int(delay_s / env.dt)

        old_enough = env.episode_length_buf >= delay_steps

        return env.recovery_bonus_buf * old_enough.float()


    ###############################################
    ############ ORIENTATION & POSTURE ############
    ###############################################

    def _reward_base_orientation(self):
        """
        Penalizes roll/pitch tilt using projected gravity x-y components.
        Lower value means the base is more upright.
        """
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    # def _reward_upright_orientation(self):
    #     """Rewards alignment of the base with gravity (upright posture)."""
    #     return torch.exp(
    #         -torch.square(self.env.projected_gravity[:, 2] + 1.0)
    #         / (2 * self.eps_orien ** 2)
    #     )
    #

    # def _reward_upright_orientation(self):
    #     """
    #     Rewards upright orientation using a stricter upright sigma.
    #     The reward is scaled by height progress and foot-contact support to reduce flying/flipping exploits.
    #     """
    #     upright_reward = self._upright_progress(self.env.cfg.rewards.upright_sigma_strict)
    #     # height_progress = self._height_progress()
    #     height_progress = self._terminal_height_progress()
    #     foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).sum(dim=1).float()
    #     contact_progress = torch.clamp(foot_contacts / 4.0, 0.0, 1.0)
    #     # 20% reward for becoming upright at all
    #     # +35% if the body is raised
    #     # +45% if feet are supporting the robot
    #     stability_factor = (
    #         0.20
    #         + 0.35 * height_progress
    #         + 0.45 * contact_progress
    #     )
    #     return upright_reward * stability_factor

    def _reward_upright_orientation(self):
        upright_reward = self._upright_progress(self.env.cfg.rewards.upright_sigma_strict)

        height_progress = self._height_progress()

        foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 5.0).sum(dim=1).float()
        contact_progress = torch.clamp(foot_contacts / 4.0, 0.0, 1.0)

        stability_factor = 0.30 + 0.35 * height_progress + 0.35 * contact_progress

        return upright_reward * stability_factor

    # Stage I
    # def _reward_height_alignment(self):
    #     """
    #     Robot must learn to rise. It is not ideal for subsequent Stage
    #     because it does not penalize excessive extension above 0.34 m.
    #
    #     height below target -> increasing reward
    #     height at target    -> maximum reward
    #     height above target -> same maximum reward
    #     """
    #     env = self.env
    #     upright_factor = torch.clamp(-env.projected_gravity[:, 2], 0.0, 1.0)

    #     height_progress = self._height_progress()
    #     loaded_support = self._loaded_support_gate()

    #     # Preserve a small height gradient before feet are fully loaded.
    #     support_factor = 0.20 + 0.80 * loaded_support

    #     return upright_factor * support_factor * height_progress

    # Stage II
    def _reward_height_alignment(self):
        """
        A target-centred reward.
        """
        env = self.env
        cfg = env.cfg.rewards

        body_height = self.get_body_height()
        target_height = cfg.recovery_height_target

        sigma = getattr(cfg, "height_alignment_sigma", 0.06)

        height_score = torch.exp(-torch.square((body_height - target_height) / sigma))

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.65) / 0.30, 0.0, 1.0)

        loaded_support = self._loaded_support_gate()
        support_factor = 0.20 + 0.80 * loaded_support

        return upright_gate * support_factor * height_score



    ###############################################
    ################ MOTOR CONTROL ################
    ###############################################

    def _reward_torques(self):
        """
        Penalizes large motor torques.
        Keeps recovery motions from becoming unnecessarily forceful.
        """
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_action(self):
        """
        Penalizes large action magnitudes.
        Acts as light regularization on policy output.
        """
        return torch.sum(torch.square(self.env.actions), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes large joint accelerations.
        Reduces extremely jerky recovery motions.
        """
        return torch.sum(torch.square(self.env.dof_acc), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high joint velocities.
        Should usually be weak or disabled during early recovery training.
        """
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    ###################################################
    ################ MOTION SMOOTHNESS ################
    ###################################################

    def _reward_action_smoothness_1(self):
        """
        Penalizes first-order action changes.
        Encourages smoother target joint commands.
        """
        diff = torch.square(
            self.env.joint_pos_target[:, :self.env.num_actuated_dof]
            - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof]
        )
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        """
        Penalizes second-order action changes.
        Discourages jerky command sequences.
        """
        diff = torch.square(
            self.env.joint_pos_target[:, :self.env.num_actuated_dof]
            - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof]
            + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof]
        )
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)
        return torch.sum(diff, dim=1)

    def _reward_joint_vel_limit(self):
        """
        Penalizes joint velocities above 0.8 rad/s.
        This is a safety/polishing term, not useful early if recovery needs fast motion.
        """
        return torch.sum(torch.clamp(torch.abs(self.env.dof_vel) - 0.8, min=0.0), dim=1)

    #############################################################
    ################ STABILITY AND CONFIGURATION ################
    #############################################################
    # Stage I
    # def _reward_base_ang_vel(self):
    #     """
    #     Penalizes high base angular velocity.
    #     Helps stabilize the robot after it starts standing.
    #     It penalizes angular velocity while the robot is:
    #     - rolling from its back;
    #     - pushing against the terrain;
    #     - transitioning through a sideways pose;
    #     - already standing.
    #     """
    #     return torch.sum(torch.square(self.env.base_ang_vel), dim=1)

    # Stage II
    def _reward_base_ang_vel(self):
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.60) / 0.30, 0.0, 1.0)

        body_height = self.get_body_height()
        height_gate = torch.clamp((body_height - 0.24) / 0.08, 0.0, 1.0)

        terminal_gate = upright_gate * height_gate

        # Retain a weak penalty while rolling, but apply the full
        # penalty after the body becomes upright and raised.
        penalty_weight = 0.10 + 0.90 * terminal_gate

        ang_vel_squared = torch.sum(torch.square(env.base_ang_vel), dim=1)

        return penalty_weight * ang_vel_squared


    def _reward_feet_slip(self):
        """
        Penalizes horizontal foot sliding while feet are in contact.
        Best enabled after basic recovery is learned.
        """
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_body_slip(self):
        """
        Penalizes trunk/base sliding while the body is in ground contact.
        Useful for polishing, but can restrict early recovery exploration.
        """
        # detect base contact
        forces = self.env.contact_forces[:, self.env.base_contact_indices, :]
        contact = (torch.norm(forces, dim=-1) > 1.0).any(dim=1)
        # horizontal base velocity
        body_xy_vel = torch.norm(self.env.base_lin_vel[:, :2], dim=1)
        # small body motion is ignored, only significant sliding gets penalized.
        slip = torch.clamp(body_xy_vel - 0.15, min=0.0)
        return contact.float() * torch.square(slip)

    # Stage I
    # def _reward_stable_foot_support(self):
    #     env = self.env

    #     upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.55) / 0.35, 0.0, 1.0)

    #     base_height = self.get_body_height()
    #     height_gate = torch.clamp((base_height - 0.25) / 0.08, 0.0, 1.0)

    #     foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

    #     # Reward loaded support, not just tiny contact.
    #     load_score = torch.clamp((foot_fz - 3.0) / 20.0, 0.0, 1.0)

    #     foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)

    #     # Smooth no-slip score.
    #     no_slip_score = torch.exp(-8.0 * foot_xy_vel)

    #     support_score = (load_score * no_slip_score).sum(dim=1) / 4.0

    #     foot_pos_body = quat_rotate_inverse(
    #         env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
    #         (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
    #     ).reshape(env.num_envs, 4, 3)

    #     y = foot_pos_body[:, :, 1]

    #     front_sep = torch.abs(y[:, 0] - y[:, 1])
    #     rear_sep = torch.abs(y[:, 2] - y[:, 3])

    #     front_violation = (torch.clamp(0.24 - front_sep, min=0.0) + torch.clamp(front_sep - 0.40, min=0.0))
    #     rear_violation = (torch.clamp(0.24 - rear_sep, min=0.0) + torch.clamp(rear_sep - 0.40, min=0.0))

    #     stance_gate = torch.exp(-10.0 * (0.5 * front_violation + 0.5 * rear_violation))

    #     # Non-foot/body contact gate
    #     nonfoot_force = torch.norm(env.contact_forces[:, env.base_contact_indices, :], dim=-1).max(dim=1).values
    #     body_contact_gate = torch.exp(-0.08 * nonfoot_force)

    #     return upright_gate * height_gate * stance_gate * body_contact_gate * support_score


    def _terminal_stable_support_count(self):
        env = self.env
        cfg = env.cfg.rewards

        # foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)
        # loaded_threshold = getattr(cfg, "loaded_foot_force_threshold", 3.0)
        # # 17 is the width of the transition from unloaded to fully loaded
        # # Fz ≤ 3 N   -> score 0
        # # Fz = 10 N  -> score 0.41
        # # Fz = 15 N  -> score 0.71
        # # Fz ≥ 20 N  -> score 1
        # full_load_force = getattr(cfg, "full_load_force_threshold", 17.0)
        # load_score = torch.clamp((foot_fz - loaded_threshold) / (full_load_force - loaded_threshold), 0.0, 1.0)
        load_score = self._foot_load_score()

        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)
        slip_threshold = getattr(cfg, "recovery_foot_slip_vel_threshold", 0.12)
        # Dense shaping can be softer than the hard success threshold,
        # but should remain connected to that configured threshold.
        slip_scale = max(2.0 * float(slip_threshold), 0.20)
        no_slip_score = 1.0 / (1.0 + torch.square(foot_xy_vel / slip_scale))

        return torch.sum(load_score * no_slip_score, dim=1)

    # Stage II
    def _reward_stable_foot_support(self):
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.65) / 0.30, 0.0, 1.0)

        body_height = self.get_body_height()
        height_gate = torch.clamp((body_height - 0.25) / 0.08, 0.0, 1.0)

        stable_count = self._terminal_stable_support_count()

        # 1 stable foot -> 0
        # 2 stable feet -> 0.5
        # 3+ stable feet -> 1
        # Stage I-III
        # support_score = torch.clamp((stable_count - 1.0) / 2.0, 0.0, 1.0)
        # Stage IV
        support_score = torch.clamp((stable_count - 1.0) / 2.5, 0.0, 1.0)

        return upright_gate * height_gate * support_score


    def _reward_rear_leg_crossing(self):
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.26) / 0.07, 0.0, 1.0)

        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)

        y = foot_pos_body[:, :, 1]

        front_sep = y[:, 0] - y[:, 1]
        rear_sep = y[:, 2] - y[:, 3]

        front_violation = torch.clamp(0.16 - front_sep, min=0.0)
        rear_violation = torch.clamp(0.20 - rear_sep, min=0.0)

        return upright_gate * height_gate * (0.5 * front_violation + 2.0 * rear_violation)


    def _reward_asymmetry(self):
        """
        Penalizes asymmetric joint configurations.
        Optional regularizer; use carefully because recovery may require asymmetric motions.
        """
        return torch.std(self.env.dof_pos[:, :self.env.num_actuated_dof], dim=1)

    # def _reward_feet_on_ground(self):
    #     """
    #     Rewards stable 3-4 foot support, gradually activated as the robot becomes upright.
    #     Prevents rewarding meaningless foot contact while fully fallen.
    #     """
    #     contact_forces = self.env.contact_forces[:, self.env.feet_indices, :]
    #     contact_norm = torch.norm(contact_forces, dim=-1)
    #     contacts = (contact_norm > 1.0).float()
    #     num_contacts = torch.sum(contacts, dim=1)
    #     g_z = self.env.projected_gravity[:, 2]
    #     # 0 when not upright enough, gradually increases as robot becomes upright
    #     # g_z = -0.3 -> upright_factor = 0.0
    #     # g_z = -0.6 -> upright_factor = 0.5
    #     # g_z = -0.9 -> upright_factor = 1.0
    #     upright_factor = torch.clamp((-g_z - 0.3) / 0.6, 0.0, 1.0)
    #     contact_reward = torch.clamp(num_contacts - 2, min=0.0) / 2.0
    #     return contact_reward * upright_factor

    # def _reward_feet_on_ground(self):
    #     env = self.env

    #     # Activate only when the robot is already near upright.
    #     upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.65) / 0.30, 0.0, 1.0)

    #     base_height = self.get_body_height()
    #     # Activate mostly once the body is raised.
    #     height_gate = torch.clamp((base_height - 0.26) / 0.07, 0.0, 1.0)

    #     foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

    #     # Soft contact score, not strict no-slip.
    #     contact_score = torch.clamp((foot_fz - 1.0) / 10.0, 0.0, 1.0)
    #     contact_count_soft = contact_score.sum(dim=1)

    #     # 1 foot -> little reward, 3 feet -> high reward.
    #     support_contact_progress = torch.clamp((contact_count_soft - 1.0) / 2.0, 0.0, 1.0)

    #     return upright_gate * height_gate * support_contact_progress

    def _reward_feet_on_ground(self):
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.65) / 0.30, 0.0, 1.0)

        height_gate = 0.20 + 0.80 * self._height_progress()

        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)
        contact_score = torch.clamp((foot_fz - 1.0) / 12.0, 0.0, 1.0)
        contact_count_soft = contact_score.sum(dim=1)

        support_contact_progress = torch.clamp((contact_count_soft - 1.0) / 2.0, 0.0, 1.0)

        return upright_gate * height_gate * support_contact_progress


    # def _reward_posture(self):
    #     """
    #     Rewards joints being close to the standing posture.
    #     The reward gradually activates as the robot approaches upright orientation.
    #     """
    #     q = self.env.dof_pos[:, :self.env.num_actuated_dof]
    #     q_stand = self.env.default_dof_pos[:, :self.env.num_actuated_dof]
    #     posture_error = torch.mean(torch.square(q - q_stand), dim=1)
    #     r_posture = torch.exp(-2.0 * posture_error)
    #     g_z = self.env.projected_gravity[:, 2]
    #     # g_z = -0.4 -> posture factor = 0
    #     # g_z = -0.65 -> posture factor = 0.5
    #     # g_z = -0.9 -> posture factor = 1
    #     upright_factor = torch.clamp((-g_z - 0.4) / 0.5, 0.0, 1.0)
    #     return r_posture * upright_factor

    def _reward_posture(self):
        """
        Rewards joints being close to the standing posture.
        The reward gradually activates as the robot approaches upright orientation.

        -g_z ≤ 0.40: no posture reward.
        0.40 < -g_z < 0.90: progressively activates.
        -g_z ≥ 0.90: fully upright-gated.
        At low height: only 20% of the posture signal is available.
        As body height rises: it smoothly increases to 100%.
        """
        env = self.env

        q = env.dof_pos[:, :env.num_actuated_dof]
        q_stand = env.default_dof_pos[:, :env.num_actuated_dof]

        posture_error = torch.norm(q - q_stand, dim=1)
        posture_score = torch.exp(-0.6 * posture_error)

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.40) / 0.50, 0.0, 1.0)

        height_gate = 0.20 + 0.80 * self._height_progress()

        return upright_gate * height_gate * posture_score


    def _reward_base_lin_vel(self):
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        body_height = self.get_body_height()
        height_gate = torch.clamp((body_height - 0.27) / 0.06, 0.0, 1.0)

        stable_count = self._terminal_stable_support_count()
        support_gate = torch.clamp((stable_count - 1.5) / 1.0, 0.0, 1.0)

        terminal_gate = upright_gate * height_gate * support_gate

        lin_vel_cost = torch.sum(torch.square(env.base_lin_vel), dim=1)

        return terminal_gate * lin_vel_cost


    def _reward_base_height(self):
        """
        Raw target-height reward without upright/contact gating.
        For fall recovery, usually keep this disabled and use height_alignment instead.
        """
        body_height = self.get_body_height()
        target_height = self.env.cfg.rewards.base_height_target
        return torch.exp(-torch.square(target_height - body_height))

    #############################################################
    ########################## SAFETY ###########################
    #############################################################

    def _reward_base_contact(self):
        env = self.env
        cfg = env.cfg.rewards

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.55) / 0.35, 0.0, 1.0)

        contact_threshold = getattr(cfg, "recovery_nonfoot_contact_threshold", 5.0)
        forces = env.contact_forces[:, env.base_contact_indices, :]
        force_norm = torch.norm(forces, dim=-1)
        nonfoot_contact = (force_norm.max(dim=1).values > contact_threshold).float()

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.24) / 0.08, 0.0, 1.0)

        return upright_gate * height_gate * nonfoot_contact

    def _reward_dof_pos_limits(self):
        """
        Penalizes joint position limit violations.
        Safety term to avoid unrealistic or damaging joint configurations.
        """
        q = self.env.dof_pos
        q_min = self.env.dof_pos_limits[:, 0]
        q_max = self.env.dof_pos_limits[:, 1]
        violations = ((q < q_min) | (q > q_max)).float()
        return torch.sum(violations, dim=1)

    def _reward_rear_leg_separation(self):
            env = self.env

            upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

            base_height = self.get_body_height()
            height_gate = torch.clamp((base_height - 0.26) / 0.07, 0.0, 1.0)

            foot_pos_body = quat_rotate_inverse(
                env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
                (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
            ).reshape(env.num_envs, 4, 3)

            y = foot_pos_body[:, :, 1]

            # Foot order: FL, FR, RL, RR
            front_sep = y[:, 0] - y[:, 1]
            rear_sep = y[:, 2] - y[:, 3]

            front_sep_err = torch.clamp(0.24 - front_sep, min=0.0)
            rear_sep_err  = torch.clamp(0.24 - rear_sep, min=0.0)

            sep_err = 0.75 * front_sep_err + 0.25 * rear_sep_err

            return upright_gate * height_gate * torch.exp(-8.0 * sep_err)


    def _reward_feet_under_body(self):
        """
        Rewards nominal foot placement under/around the robot body during the
        late recovery phase.

        The reward is intended as a final-stance shaping term. It first computes
        two soft gates: an uprightness gate based on the projected gravity z-axis,
        and a height gate based on the base height. This prevents the reward from
        strongly constraining the legs while the robot is still rolling or pushing
        itself up from a fallen state.

        Foot positions are converted from world coordinates to the robot base frame.
        The body-frame horizontal foot positions are then compared against nominal
        target XY locations for the four feet in FL, FR, RL, RR order. Contacted
        feet are weighted more strongly than non-contacted feet, since the placement
        of supporting feet matters more for forming a stable stance.

        Returns:
            torch.Tensor: Per-environment reward in [0, 1], high when the robot is
            sufficiently upright, sufficiently raised, and its feet are close to the
            nominal body-frame stance positions.
        """
        upright_gate = torch.clamp((-self.env.projected_gravity[:, 2] - 0.5) / 0.5, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.18) / 0.12, 0.0, 1.0)

        contact = (self.env.contact_forces[:, self.env.feet_indices, 2] > self.env.cfg.rewards.recovery_contact_force_threshold).float()

        foot_pos_body = quat_rotate_inverse(
            self.env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (self.env.foot_positions - self.env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(self.env.num_envs, 4, 3)

        target_xy = torch.tensor([
            [ 0.1725,  0.1574],
            [ 0.1725, -0.1574],
            [-0.2652,  0.1565],
            [-0.2652, -0.1565],
        ], device=self.env.device)

        err = torch.norm(foot_pos_body[:, :, :2] - target_xy.unsqueeze(0), dim=-1)
        weighted_err = (err * (0.3 + 0.7 * contact)).mean(dim=1)

        return upright_gate * height_gate * torch.exp(-3.0 * weighted_err)

    def _reward_stance_region(self):
        """
        Rewards valid final foot-support geometry in the robot base frame.

        This version is stricter on fore-aft foot placement, especially rear feet.
        It is intended for terminal recovery refinement after the robot can already
        reach upright/high states.

        Foot order:
            0 = FL, 1 = FR, 2 = RL, 3 = RR
        """

        env = self.env

        # Active only after the robot is reasonably upright.
        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        base_height = self.get_body_height()
        # Active only near terminal recovery height.
        height_gate = torch.clamp((base_height - 0.26) / 0.07, 0.0, 1.0)

        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)

        x = foot_pos_body[:, :, 0]
        y = foot_pos_body[:, :, 1]

        # Wider terminal stance boxes.
        # Nominal Go1-like measured stance:
        # FL: [ 0.1725,  0.1574]
        # FR: [ 0.1725, -0.1574]
        # RL: [-0.2652,  0.1565]
        # RR: [-0.2652, -0.1565]
        #
        # These bounds are slightly wider than nominal to avoid over-constraining
        # recovery, but still penalize rear feet drifting too far forward.

        if not hasattr(self, "_stance_x_min") or self._stance_x_min.device != env.device:
            self._stance_x_min = torch.tensor([0.08, 0.08, -0.36, -0.36], device=env.device)
            self._stance_x_max = torch.tensor([0.28, 0.28, -0.16, -0.16], device=env.device)
            self._stance_y_min = torch.tensor([0.10, -0.22, 0.10, -0.22], device=env.device)
            self._stance_y_max = torch.tensor([0.22, -0.10, 0.22, -0.10], device=env.device)

            # Rear feet get higher weight because rear-foot placement has been
            # the main terminal-support failure.
            self._stance_x_weight = torch.tensor([1.5, 1.5, 2.5, 2.5], device=env.device)
            self._stance_y_weight = torch.tensor([1.0, 1.0, 1.5, 1.5], device=env.device)

        x_violation = (
            torch.clamp(self._stance_x_min.unsqueeze(0) - x, min=0.0)
            + torch.clamp(x - self._stance_x_max.unsqueeze(0), min=0.0)
        )

        y_violation = (
            torch.clamp(self._stance_y_min.unsqueeze(0) - y, min=0.0)
            + torch.clamp(y - self._stance_y_max.unsqueeze(0), min=0.0)
        )

        weight_sum = self._stance_x_weight.sum() + self._stance_y_weight.sum()

        weighted_box_err = (
            (self._stance_x_weight.unsqueeze(0) * x_violation).sum(dim=1)
            + (self._stance_y_weight.unsqueeze(0) * y_violation).sum(dim=1)
        ) / weight_sum

        # Explicitly penalize rear feet being too far forward.
        # This targets the observed rear_x_mean ≈ -0.09 to -0.12 failure.
        rear_forward_violation = 0.5 * (torch.clamp(x[:, 2] - (-0.16), min=0.0) + torch.clamp(x[:, 3] - (-0.16), min=0.0))
        # Also mildly penalize front feet being too far forward.
        front_forward_violation = 0.5 * (torch.clamp(x[:, 0] - 0.28, min=0.0) + torch.clamp(x[:, 1] - 0.28, min=0.0))
        stance_err = weighted_box_err + 2.0 * rear_forward_violation + 0.75 * front_forward_violation

        return upright_gate * height_gate * torch.exp(-6.0 * stance_err)

    def _reward_stand_still_action(self):
        """
        Penalize sustained nonzero terminal actions after the robot is upright and raised.
        A weak penalty remains active before stable support is fully established,
        then increases as stable multi-foot support develops.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.80) / 0.15, 0.0, 1.0)

        body_height = self.get_body_height()
        height_gate = torch.clamp((body_height - 0.27) / 0.04, 0.0, 1.0)

        stable_count = self._terminal_stable_support_count()
        support_gate = torch.clamp((stable_count - 1.5) / 1.0, 0.0, 1.0)
        # 25% terminal damping before stable support is established.
        # Progressively reaches 100% as support becomes stable.
        support_factor = 0.25 + 0.75 * support_gate

        action_cost = torch.sum(torch.square(env.actions[:, :env.num_actuated_dof]), dim=1)

        return upright_gate * height_gate * support_factor * action_cost

    def _reward_loaded_foot_support(self):
        """
        Rewards loaded foot support during late recovery.

        This is different from non-slip support:
            loaded support = foot carries vertical force
            non-slip support = loaded/contacted foot is also stationary

        Purpose:
            First teach the policy to use feet as support points.
            Do not rely only on feet_slip, because slip penalty can be avoided
            by reducing foot contact.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.65) / 0.30, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.25) / 0.08, 0.0, 1.0)

        # foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

        # loaded_threshold = getattr(env.cfg.rewards, "loaded_foot_force_threshold", 3.0)

        # # Smooth load score.
        # # Weak/tiny contacts get little reward; real support contacts get more.
        # load_score = torch.clamp((foot_fz - loaded_threshold) / 17.0, 0.0, 1.0)
        load_score = self._foot_load_score()
        loaded_count = load_score.sum(dim=1)

        # Reward progress from roughly 1 loaded foot to 3 loaded feet.
        # Stage I-III
        # support_progress = torch.clamp((loaded_count - 1.0) / 2.0, 0.0, 1.0)
        # Stage IV
        support_progress = torch.clamp((loaded_count - 1.0) / 2.5, 0.0, 1.0)

        return upright_gate * height_gate * support_progress

    def _reward_late_nonfoot_contact(self):
        """
        Penalizes non-foot contact only in the late recovery phase.

        This prevents the robot from using trunk/thigh/calf contact as a fake
        terminal support mode after it is already upright and raised.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.25) / 0.08, 0.0, 1.0)

        nonfoot_threshold = getattr(env.cfg.rewards, "recovery_nonfoot_contact_threshold", 5.0)

        forces = env.contact_forces[:, env.base_contact_indices, :]
        contact_force = torch.norm(forces, dim=-1)
        nonfoot_contact = (contact_force.max(dim=1).values > nonfoot_threshold).float()

        return upright_gate * height_gate * nonfoot_contact

    def _reward_loaded_foot_slip(self):
        """
        Penalizes horizontal sliding of feet that are actually carrying load.

        This targets the current failure mode:
            feet are visually on the ground,
            feet are loaded,
            but the contact points are sliding.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.27) / 0.06, 0.0, 1.0)

        gate = upright_gate * height_gate

        # foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)
        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)

        # loaded_threshold = getattr(env.cfg.rewards, "loaded_foot_force_threshold", 3.0)
        slip_threshold = getattr(env.cfg.rewards, "recovery_foot_slip_vel_threshold", 0.12)

        # load_weight = torch.clamp((foot_fz - loaded_threshold) / 20.0, 0.0, 1.0)
        load_weight = self._foot_load_score()
        slip_speed = torch.clamp(foot_xy_vel - slip_threshold, min=0.0)

        return gate * torch.sum(load_weight * torch.square(slip_speed), dim=1)

    def _reward_support_deficit(self):
        """
        Penalizes having fewer than 3 smooth loaded/non-slipping support feet
        during the late recovery phase.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.70) / 0.25, 0.0, 1.0)

        body_height = self.get_body_height()
        # height <= 0.27 m       -> reward inactive
        # 0.27 m < height < 0.33 m -> gradually activates
        # height ≥ 0.33 m       -> fully active
        height_gate = torch.clamp((body_height - 0.27) / 0.06, 0.0, 1.0)
        # Equivalent to:
        # terminal_height_start = 0.27
        # terminal_height_full = 0.33
        # height_gate = torch.clamp((body_height - terminal_height_start) / (terminal_height_full - terminal_height_start), 0.0, 1.0)

        stable_count = self._terminal_stable_support_count()
        deficit = torch.clamp(3.0 - stable_count, min=0.0) / 3.0

        return upright_gate * height_gate * torch.square(deficit)

    def _reward_front_leg_error(self):
        """
        Penalizes front feet being too wide and too far forward during terminal
        recovery.

        Current observed failure:
            front feet act like wide forward braces instead of moving under the body.
        """
        env = self.env

        upright_gate = torch.clamp((-env.projected_gravity[:, 2] - 0.75) / 0.20, 0.0, 1.0)

        base_height = self.get_body_height()
        height_gate = torch.clamp((base_height - 0.28) / 0.05, 0.0, 1.0)

        gate = upright_gate * height_gate

        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(env.num_envs, 4, 3)

        x = foot_pos_body[:, :, 0]
        y = foot_pos_body[:, :, 1]

        # Foot order: FL, FR, RL, RR
        front_sep = y[:, 0] - y[:, 1]
        front_x_mean = 0.5 * (x[:, 0] + x[:, 1])
        front_y_center = 0.5 * (y[:, 0] + y[:, 1])

        # Penalize only outside a reasonable terminal stance.
        width_err = torch.clamp(front_sep - 0.42, min=0.0) / 0.20
        forward_err = torch.clamp(front_x_mean - 0.28, min=0.0) / 0.20
        center_err = torch.abs(front_y_center) / 0.15

        err = (
            1.5 * torch.square(width_err)
            + 1.0 * torch.square(forward_err)
            + 0.3 * torch.square(center_err)
        )

        return gate * torch.clamp(err, 0.0, 4.0)

    def _reward_terminal_action_prior(self):
        """
        Encourages small actions once the robot is upright and high enough
        to be near terminal recovery.

        This version intentionally does not require good posture/contact,
        because those are the quantities we are trying to recover.
        """

        cfg = self.env.cfg.rewards

        # Upright gate: active before perfect upright.
        upright_score = torch.clamp((-self.env.projected_gravity[:, 2] - 0.45) / 0.45, 0.0, 1.0)

        base_height = self.get_body_height()
        # Height gate: active once the robot is raised enough.
        height_score = torch.clamp((base_height - 0.22) / 0.10, 0.0, 1.0)

        terminal_gate = upright_score * height_score

        # Start loose. Tighten later after stable behavior emerges.
        sigma = getattr(cfg, "terminal_action_prior_sigma", 3.0)

        action_cost = torch.mean((self.env.actions / sigma) ** 2, dim=1)

        # Non-saturating reward. Unlike exp(-cost), this still gives signal
        # when actions are large.
        reward = terminal_gate / (1.0 + action_cost)

        return reward
