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
    # def _reward_recovery_success(self):
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

    def _reward_recovery_success(self):
        """
        Soft combined recovery-progress reward.
        Rewards the robot only when it is upright, sufficiently raised, and supported by 3-4 feet.
        This is dense shaping, not the final success condition.
        """
        upright = self._upright_progress(self.env.cfg.rewards.upright_sigma_soft)
        height_progress = self._height_progress()
        foot_contacts = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).sum(dim=1).float()
        # support_progress = torch.clamp(foot_contacts / 4.0, 0.0, 1.0)
        # 0, 1, 2 foot contacts -> 0 reward
        # 3 foot contacts       -> 0.5 reward
        # 4 foot contacts       -> 1.0 reward
        support_progress = torch.clamp((foot_contacts - 2.0) / 2.0, 0.0, 1.0)
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
        height_progress = self._height_progress()
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

    def _reward_height_alignment(self):
        """
        Rewards body-height recovery only when the robot is reasonably upright.
        Uses height_progress instead of raw Gaussian height tracking for clearer recovery shaping.
        """
        # low height     -> low reward
        # standing height -> high reward
        g_z = self.env.projected_gravity[:, 2]
        upright_factor = torch.clamp(-g_z, 0.0, 1.0)
        height_progress = self._height_progress()
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
    #     Rewards standing posture when approximately upright.
    #     Paper:
    #     exp(-(q-qstand)^2) if |gz + 1| < eps
    #     """
    #     q = self.env.dof_pos[:, :self.env.num_actuated_dof]

    #     q_stand = self.env.default_dof_pos[
    #         :, :self.env.num_actuated_dof
    #     ]

    #     posture_error = torch.sum(
    #         torch.square(q - q_stand),
    #         dim=1
    #     )

    #     r_posture = torch.exp(-posture_error)

    #     g_z = self.env.projected_gravity[:, 2]

    #     # posture reward activates roughly when −1 ≤ g_z ≤ (−1+eps_posture)
    #     upright = (
    #         torch.abs(g_z + 1.0) < self.env.cfg.rewards.posture_upright_epsilon
    #     ).float()

    #     return r_posture * upright

    def _reward_posture(self):
        """
        Rewards joints being close to the standing posture.
        The reward gradually activates as the robot approaches upright orientation.
        """
        q = self.env.dof_pos[:, :self.env.num_actuated_dof]
        q_stand = self.env.default_dof_pos[:, :self.env.num_actuated_dof]
        posture_error = torch.mean(torch.square(q - q_stand), dim=1)
        r_posture = torch.exp(-2.0 * posture_error)
        g_z = self.env.projected_gravity[:, 2]
        # g_z = -0.4 -> posture factor = 0
        # g_z = -0.65 -> posture factor = 0.5
        # g_z = -0.9 -> posture factor = 1
        upright_factor = torch.clamp((-g_z - 0.4) / 0.5, 0.0, 1.0)
        return r_posture * upright_factor


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

    def _reward_base_contact(self):
        """
        Penalizes non-foot body contact with the ground.
        Keep disabled during early recovery because the robot starts fallen.
        """
        forces = self.env.contact_forces[:, self.env.base_contact_indices, :]
        force_norm = torch.norm(forces, dim=-1)
        return (force_norm > 0.2).any(dim=1).float()

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
