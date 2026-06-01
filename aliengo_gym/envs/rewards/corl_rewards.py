import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

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
        h = self.env.root_states[:, 2]
        return torch.clamp((h - 0.26) / 0.07, 0.0, 1.0)

    def _height_progress(self):
        """
        Soft normalized body-height progress from recovery_height_min to recovery_height_target.
        Returns 0 when body is too low and 1 near target standing height.
        """
        body_height = self.env.root_states[:, 2]
        h_min = self.env.cfg.rewards.recovery_height_min
        h_target = self.env.cfg.rewards.recovery_height_target
        return torch.clamp((body_height - h_min) / (h_target - h_min), 0.0, 1.0)

    def _height_success(self):
        """
        Hard height check for true recovery success.
        Should match the stable_height condition used in legged_robot.py.
        """
        body_height = self.env.root_states[:, 2]
        h_success = self.env.cfg.rewards.recovery_height_success
        return body_height > h_success

    ######## dense success reward ########
    # def _reward_recovery_progress(self):
    #     gz = self.env.projected_gravity[:, 2]
    #     upright = gz < -0.9
    #     if self.env.cfg.env.robot == "go1":
    #         height = self.env.root_states[:, 2] > 0.28
    #     else:
    #         height = self.env.root_states[:, 2] > 0.42
    #     low_vel     = torch.norm(self.env.base_lin_vel[:, :2], dim=1) < 0.3
    #     low_ang_vel = torch.norm(self.env.base_ang_vel, dim=1) < 1.2
    #     posture_error = torch.norm(self.env.dof_pos - self.env.default_dof_pos, dim=1)
    #     good_posture  = posture_error < 2.0
    #     foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).sum(dim=1)
    #     stable_contacts = foot_contacts >= 3
    #     success = upright & height & low_vel & low_ang_vel & good_posture & stable_contacts
    #     return success.float()

    # def _reward_recovery_progress(self):
    #     """
    #     Soft combined recovery-progress reward.
    #     Rewards the robot only when it is upright, sufficiently raised, and supported by 3-4 feet.
    #     This is dense shaping, not the final success condition.
    #     """
    #     upright = self._upright_progress(self.env.cfg.rewards.upright_sigma_soft)
    #     height_progress = self._height_progress()
    #     # foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).sum(dim=1).float()
    #     # support_progress = torch.clamp(foot_contacts / 4.0, 0.0, 1.0)
    #     # 0, 1, 2 foot contacts -> 0 reward
    #     # 3 foot contacts       -> 0.5 reward
    #     # 4 foot contacts       -> 1.0 reward
    #     # support_progress = torch.clamp((foot_contacts - 2.0) / 2.0, 0.0, 1.0)

    #     # FINETUNE
    #     foot_contact = (
    #         self.env.contact_forces[:, self.env.feet_indices, 2] > self.env.cfg.rewards.recovery_contact_force_threshold
    #     )

    #     foot_xy_vel = torch.norm(self.env.foot_velocities[:, :, :2], dim=-1)

    #     non_slipping_feet = foot_contact & (
    #         foot_xy_vel < self.env.cfg.rewards.recovery_foot_slip_vel_threshold
    #     )

    #     support_progress = torch.clamp(
    #         (non_slipping_feet.sum(dim=1).float() - 2.0) / 2.0,
    #         0.0,
    #         1.0
    #     )
    #     return upright * height_progress * support_progress
    #

    def _reward_recovery_progress(self):
        """
        Dense recovery progress reward.

        Rewards upright + height + smooth loaded non-slipping support.
        Unlike the hard success condition, this gives gradient even when feet are
        partially loaded or still slipping.
        """

        env = self.env

        upright = self._upright_progress(env.cfg.rewards.upright_sigma_soft)
        # height_progress = self._height_progress()
        height_progress = self._terminal_height_progress()

        # Vertical foot load
        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

        # Smooth load score.
        # Tiny contacts get little reward; feet carrying real load get more.
        load_score = torch.clamp((foot_fz - 3.0) / 20.0, 0.0, 1.0)

        # Horizontal foot speed
        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)

        # Smooth no-slip score.
        # v = 0.00 -> 1.00
        # v = 0.10 -> about 0.45
        # v = 0.20 -> about 0.20
        no_slip_score = torch.exp(-8.0 * foot_xy_vel)

        # Smooth support score in [0, 4]
        smooth_support_count = (load_score * no_slip_score).sum(dim=1)

        # Progress toward roughly 3 stable support feet.
        support_progress = torch.clamp(
            smooth_support_count / 3.0,
            0.0,
            1.0
        )

        return upright * height_progress * support_progress

    ######## sparse success reward ########
    def _reward_recovery_bonus(self):
        """
        Sparse one-shot success bonus.
        Fires only when legged_robot.py detects stable recovery and sets recovery_bonus_buf.
        """
        return self.env.recovery_bonus_buf


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

    def _reward_upright_orientation(self):
        """
        Rewards upright orientation using a stricter upright sigma.
        The reward is scaled by height progress and foot-contact support to reduce flying/flipping exploits.
        """
        upright_reward = self._upright_progress(self.env.cfg.rewards.upright_sigma_strict)
        # height_progress = self._height_progress()
        height_progress = self._terminal_height_progress()
        foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).sum(dim=1).float()
        contact_progress = torch.clamp(foot_contacts / 4.0, 0.0, 1.0)
        # 20% reward for becoming upright at all
        # +35% if the body is raised
        # +45% if feet are supporting the robot
        stability_factor = (
            0.20
            + 0.35 * height_progress
            + 0.45 * contact_progress
        )
        return upright_reward * stability_factor

    # def _reward_height_alignment(self):
    #     """Rewards maintaining the desired base height."""
    #     body_height = self.env.root_states[:, 2]
    #     target_height = self.env.cfg.rewards.base_height_target
    #     g_z = self.env.projected_gravity[:,2]
    #     # prevents the agent from farming height while upside down
    #     upright_factor = torch.clamp(-g_z, 0.0, 1.0)
    #     return upright_factor * torch.exp(
    #         -torch.square(target_height - body_height)
    #     )

    # def _reward_height_alignment(self):
    #     """
    #     Rewards body-height recovery only when the robot is reasonably upright.
    #     Uses height_progress instead of raw Gaussian height tracking for clearer recovery shaping.
    #     """
    #     # low height     -> low reward
    #     # standing height -> high reward
    #     g_z = self.env.projected_gravity[:, 2]
    #     upright_factor = torch.clamp(-g_z, 0.0, 1.0)
    #     height_progress = self._height_progress()
    #     return upright_factor * height_progress

    def _reward_height_alignment(self):
        g_z = self.env.projected_gravity[:, 2]
        upright_factor = torch.clamp(-g_z, 0.0, 1.0)

        height_progress = self._terminal_height_progress()

        return upright_factor * height_progress


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
    #
    def _reward_base_ang_vel(self):
        """
        Penalizes high base angular velocity.
        Helps stabilize the robot after it starts standing.
        """
        return torch.sum(torch.square(self.env.base_ang_vel), dim=1)

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

    # def _reward_stable_foot_support(self):
    #     # upright-gated no-slip support
    #     # Active only near upright; avoids punishing useful rolling/self-righting.
    #     upright_gate = torch.clamp(
    #         (-self.env.projected_gravity[:, 2] - 0.5) / 0.5,
    #         0.0,
    #         1.0
    #     )

    #     foot_contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0
    #     foot_xy_vel = torch.norm(self.env.foot_velocities[:, :, :2], dim=-1)

    #     # Reward contacted feet that are not sliding.
    #     no_slip = torch.exp(-10.0 * foot_xy_vel) * foot_contact.float()

    #     return upright_gate * no_slip.mean(dim=1)

    # def _reward_stable_foot_support(self):

    #     upright_gate = torch.clamp((-self.env.projected_gravity[:, 2] - 0.55) / 0.35, 0.0, 1.0)
    #     height_gate = torch.clamp((self.env.root_states[:, 2] - 0.25) / 0.08, 0.0, 1.0)

    #     foot_contact = (self.env.contact_forces[:, self.env.feet_indices, 2] > self.env.cfg.rewards.recovery_contact_force_threshold).float()

    #     foot_xy_vel = torch.norm(self.env.foot_velocities[:, :, :2], dim=-1)

    #     # Smooth score: high when contacted feet are nearly stationary.
    #     no_slip_score = torch.exp(-8.0 * foot_xy_vel)

    #     support_score = (foot_contact * no_slip_score).sum(dim=1) / 4.0

    #     return upright_gate * height_gate * support_score

    def _reward_stable_foot_support(self):
        env = self.env

        upright_gate = torch.clamp(
            (-env.projected_gravity[:, 2] - 0.55) / 0.35,
            0.0,
            1.0
        )

        height_gate = torch.clamp(
            (env.root_states[:, 2] - 0.25) / 0.08,
            0.0,
            1.0
        )

        foot_fz = env.contact_forces[:, env.feet_indices, 2].clamp_min(0.0)

        # Reward loaded support, not just tiny contact.
        load_score = torch.clamp((foot_fz - 3.0) / 20.0, 0.0, 1.0)

        foot_xy_vel = torch.norm(env.foot_velocities[:, :, :2], dim=-1)

        # Smooth no-slip score.
        no_slip_score = torch.exp(-8.0 * foot_xy_vel)

        support_score = (load_score * no_slip_score).sum(dim=1) / 4.0

        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)

        y = foot_pos_body[:, :, 1]

        front_sep = y[:, 0] - y[:, 1]
        rear_sep = y[:, 2] - y[:, 3]

        front_violation = torch.clamp(0.18 - front_sep, min=0.0)
        rear_violation = torch.clamp(0.22 - rear_sep, min=0.0)

        stance_gate = torch.exp(
            -8.0 * (0.5 * front_violation + 1.5 * rear_violation)
        )

        # Non-foot/body contact gate
        nonfoot_force = torch.norm(
            env.contact_forces[:, env.base_contact_indices, :],
            dim=-1
        ).max(dim=1).values

        body_contact_gate = torch.exp(-0.08 * nonfoot_force)

        return upright_gate * height_gate * stance_gate * body_contact_gate * support_score

    def _reward_rear_leg_crossing(self):
        env = self.env

        upright_gate = torch.clamp(
            (-env.projected_gravity[:, 2] - 0.70) / 0.25,
            0.0,
            1.0
        )

        height_gate = torch.clamp(
            (env.root_states[:, 2] - 0.26) / 0.07,
            0.0,
            1.0
        )

        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)

        y = foot_pos_body[:, :, 1]

        front_sep = y[:, 0] - y[:, 1]
        rear_sep = y[:, 2] - y[:, 3]

        front_violation = torch.clamp(0.16 - front_sep, min=0.0)
        rear_violation = torch.clamp(0.20 - rear_sep, min=0.0)

        return upright_gate * height_gate * (
            0.5 * front_violation + 2.0 * rear_violation
        )


    def _reward_asymmetry(self):
        """
        Penalizes asymmetric joint configurations.
        Optional regularizer; use carefully because recovery may require asymmetric motions.
        """
        return torch.std(self.env.dof_pos[:, :self.env.num_actuated_dof], dim=1)

    def _reward_feet_on_ground(self):
        """
        Rewards stable 3-4 foot support, gradually activated as the robot becomes upright.
        Prevents rewarding meaningless foot contact while fully fallen.
        """
        contact_forces = self.env.contact_forces[:, self.env.feet_indices, :]
        contact_norm = torch.norm(contact_forces, dim=-1)
        contacts = (contact_norm > 1.0).float()
        num_contacts = torch.sum(contacts, dim=1)
        g_z = self.env.projected_gravity[:, 2]
        # 0 when not upright enough, gradually increases as robot becomes upright
        # g_z = -0.3 -> upright_factor = 0.0
        # g_z = -0.6 -> upright_factor = 0.5
        # g_z = -0.9 -> upright_factor = 1.0
        upright_factor = torch.clamp((-g_z - 0.3) / 0.6, 0.0, 1.0)
        contact_reward = torch.clamp(num_contacts - 2, min=0.0) / 2.0
        return contact_reward * upright_factor

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
        # FINETUNE
        q = self.env.dof_pos[:, :self.env.num_actuated_dof]
        q_stand = self.env.default_dof_pos[:, :self.env.num_actuated_dof]

        posture_error = torch.norm(q - q_stand, dim=1)

        r_posture = torch.exp(-1.5 * posture_error)

        g_z = self.env.projected_gravity[:, 2]
        upright_factor = torch.clamp((-g_z - 0.4) / 0.5, 0.0, 1.0)

        # height_gate = torch.clamp(
        #     (self.env.root_states[:, 2] - 0.18) / 0.12,
        #     0.0,
        #     1.0
        # )

        height_gate = torch.clamp(
            (self.env.root_states[:, 2] - 0.26) / 0.07,
            0.0,
            1.0
        )

        return r_posture * upright_factor * height_gate



    def _reward_base_height(self):
        """
        Raw target-height reward without upright/contact gating.
        For fall recovery, usually keep this disabled and use height_alignment instead.
        """
        body_height = self.env.root_states[:, 2]
        target_height = self.env.cfg.rewards.base_height_target
        return torch.exp(-torch.square(target_height - body_height))

    #############################################################
    ########################## SAFETY ###########################
    #############################################################

    # def _reward_base_contact(self):
    #     """
    #     Penalizes non-foot body contact with the ground.
    #     Keep disabled during early recovery because the robot starts fallen.
    #     """
    #     forces = self.env.contact_forces[:, self.env.base_contact_indices, :]
    #     force_norm = torch.norm(forces, dim=-1)
    #     return (force_norm > 0.2).any(dim=1).float()

    def _reward_base_contact(self):
        # fallen/rolling phase: non-foot contact is expected
        # near-standing phase: non-foot contact is bad
        env = self.env

        forces = env.contact_forces[:, env.base_contact_indices, :]
        force_norm = torch.norm(forces, dim=-1)
        nonfoot_contact = (force_norm > 0.2).any(dim=1).float()

        upright_gate = torch.clamp(
            (-env.projected_gravity[:, 2] - 0.55) / 0.35,
            0.0,
            1.0
        )

        height_gate = torch.clamp(
            (env.root_states[:, 2] - 0.24) / 0.08,
            0.0,
            1.0
        )

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

            upright_gate = torch.clamp(
                (-env.projected_gravity[:, 2] - 0.70) / 0.25,
                0.0,
                1.0
            )

            # height_gate = torch.clamp(
            #     (env.root_states[:, 2] - 0.23) / 0.10,
            #     0.0,
            #     1.0
            # )

            height_gate = torch.clamp(
                (env.root_states[:, 2] - 0.26) / 0.07,
                0.0,
                1.0
            )

            foot_pos_body = quat_rotate_inverse(
                env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
                (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
            ).reshape(env.num_envs, 4, 3)

            y = foot_pos_body[:, :, 1]

            # Foot order: FL, FR, RL, RR
            front_sep = y[:, 0] - y[:, 1]
            rear_sep = y[:, 2] - y[:, 3]

            front_sep_err = torch.clamp(0.22 - front_sep, min=0.0)
            rear_sep_err = torch.clamp(0.24 - rear_sep, min=0.0)

            sep_err = 0.25 * front_sep_err + 0.75 * rear_sep_err

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
        upright_gate = torch.clamp(
            (-self.env.projected_gravity[:, 2] - 0.5) / 0.5,
            0.0,
            1.0
        )

        height_gate = torch.clamp(
            (self.env.root_states[:, 2] - 0.18) / 0.12,
            0.0,
            1.0
        )

        contact = (
            self.env.contact_forces[:, self.env.feet_indices, 2]
            > self.env.cfg.rewards.recovery_contact_force_threshold
        ).float()

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

        err = torch.norm(
            foot_pos_body[:, :, :2] - target_xy.unsqueeze(0),
            dim=-1
        )

        weighted_err = (err * (0.3 + 0.7 * contact)).mean(dim=1)

        return upright_gate * height_gate * torch.exp(-3.0 * weighted_err)

    def _reward_stance_region(self):
        """
        Rewards a valid final foot-support geometry in the robot base frame.

        This reward is active mainly in the late recovery phase. It checks that
        each foot lies in a measured-default-centered stance region and adds a
        small anti-collapse term for left/right foot separation, with stronger
        emphasis on the rear feet because the observed failure mode is rear-leg
        interlocking.

        Foot order is assumed to be:
            0 = FL, 1 = FR, 2 = RL, 3 = RR
        """

        env = self.env

        # Activate only after the robot is reasonably upright.
        upright_gate = torch.clamp(
            (-env.projected_gravity[:, 2] - 0.70) / 0.25,
            0.0,
            1.0
        )

        # Activate only once the body has started rising.
        # height_gate = torch.clamp(
        #     (env.root_states[:, 2] - 0.23) / 0.10,
        #     0.0,
        #     1.0
        # )

        height_gate = torch.clamp(
            (env.root_states[:, 2] - 0.26) / 0.07,
            0.0,
            1.0
        )

        # Convert foot positions from world frame to body/base frame.
        foot_pos_body = quat_rotate_inverse(
            env.base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4),
            (env.foot_positions - env.base_pos.unsqueeze(1)).reshape(-1, 3)
        ).reshape(env.num_envs, 4, 3)

        xy = foot_pos_body[:, :, :2]
        x = xy[:, :, 0]
        y = xy[:, :, 1]

        # ------------------------------------------------------------------
        # Measured static default stance from recovery env:
        #
        # FL: [ 0.1725,  0.1574]
        # FR: [ 0.1725, -0.1574]
        # RL: [-0.2652,  0.1565]
        # RR: [-0.2652, -0.1565]
        #
        # These bounds are centered around that pose but still allow tolerance.
        # ------------------------------------------------------------------

        # Foot order: FL, FR, RL, RR
        x_min = torch.tensor([ 0.12,  0.12, -0.33, -0.33], device=env.device)
        x_max = torch.tensor([ 0.23,  0.23, -0.21, -0.21], device=env.device)

        y_min = torch.tensor([ 0.13, -0.19,  0.13, -0.19], device=env.device)
        y_max = torch.tensor([ 0.19, -0.13,  0.19, -0.13], device=env.device)

        x_violation = (
            torch.clamp(x_min.unsqueeze(0) - x, min=0.0)
            + torch.clamp(x - x_max.unsqueeze(0), min=0.0)
        )

        y_violation = (
            torch.clamp(y_min.unsqueeze(0) - y, min=0.0)
            + torch.clamp(y - y_max.unsqueeze(0), min=0.0)
        )

        # Per-foot box violation. Zero if the foot is inside its valid region.
        box_err = (x_violation + y_violation).mean(dim=1)

        return upright_gate * height_gate * torch.exp(-5.0 * box_err)

    def _reward_stand_still_action(self):
        env = self.env

        upright_gate = torch.clamp(
            (-env.projected_gravity[:, 2] - 0.80) / 0.15,
            0.0,
            1.0
        )

        # Activate only near real standing height, not at 0.25 m crouch
        height_gate = torch.clamp(
            (env.root_states[:, 2] - 0.30) / 0.04,
            0.0,
            1.0
        )

        near_stand_gate = upright_gate * height_gate

        return near_stand_gate * torch.sum(
            torch.square(env.actions[:, :env.num_actuated_dof]),
            dim=1
        )
