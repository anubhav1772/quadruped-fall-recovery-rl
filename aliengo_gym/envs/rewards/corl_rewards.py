import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env
        self.eps_orien = 0.25
        self.eps_posture = 0.20 #0.25

    def load_env(self, env):
        self.env = env

    ######## dense success reward ########
    def _reward_recovery_success(self):

        g_z = self.env.projected_gravity[:, 2]

        upright = g_z < -0.9

        if self.env.cfg.env.robot == "go1":
            height = self.env.root_states[:, 2] > 0.28
        else:
            height = self.env.root_states[:, 2] > 0.42

        low_vel = torch.norm(
            self.env.base_lin_vel[:, :2],
            dim=1
        ) < 0.5

        success = upright & height #& low_vel

        return success.float()

    ######## sparse success reward ########
    def _reward_recovery_bonus(self):
        return self.env.recovery_bonus_buf


    ###############################################
    ############ ORIENTATION & POSTURE ############
    ###############################################

    def _reward_base_orientation(self):
        """Penalize base tilt (roll/pitch deviation from upright)."""
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
        Rewards upright orientation only when the robot exhibits
        stable standing characteristics, including sufficient body
        height and multi-foot ground support. This prevents
        reward exploitation through transient flipping or
        upside-down rolling motions.
        """

        g_z = self.env.projected_gravity[:, 2]

        upright_reward = torch.exp(
            -torch.square(g_z + 1.0)
            / (2 * self.eps_orien ** 2)
        )

        body_height = self.env.root_states[:, 2]

        if self.env.cfg.env.robot == "go1":
            stable_height = (body_height > 0.24).float()
        else:
            stable_height = (body_height > 0.40).float()

        foot_contacts = (
            self.env.contact_forces[
                :, self.env.feet_indices, 2
            ] > 1.0
        ).sum(dim=1).float()

        stable_contacts = foot_contacts / 4.0

        stability_factor = (
            0.25
            + 0.35 * stable_height
            + 0.40 * stable_contacts
        )

        return upright_reward * stability_factor

    def _reward_height_alignment(self):
        """Rewards maintaining the desired base height."""
        body_height = self.env.root_states[:, 2]
        target_height = self.env.cfg.rewards.base_height_target

        g_z = self.env.projected_gravity[:,2]

        # prevents the agent from farming height while upside down
        upright_factor = torch.clamp(-g_z, 0.0, 1.0)

        return upright_factor * torch.exp(
            -torch.square(target_height - body_height)
        )

        # return torch.exp(
        #     -torch.square(target_height - body_height)
        # )

    ###############################################
    ################ MOTOR CONTROL ################
    ###############################################

    def _reward_torques(self):
        """Penalizes large joint torques."""
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_action(self):
        """Penalizes large action magnitudes."""
        return torch.sum(torch.square(self.env.actions), dim=1)

    def _reward_dof_acc(self):
        """Penalizes joint accelerations."""
        return torch.sum(torch.square(self.env.dof_acc), dim=1)

    def _reward_dof_vel(self):
        """Penalizes joint velocities."""
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    ###################################################
    ################ MOTION SMOOTHNESS ################
    ###################################################

    def _reward_action_smoothness_1(self):
        """Penalizes rapid changes in actions."""
        diff = torch.square(
            self.env.joint_pos_target[:, :self.env.num_actuated_dof]
            - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof]
        )

        diff = diff * (
            self.env.last_actions[:, :self.env.num_dof] != 0
        )

        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        """Penalizes jerky actions via second-order differences."""
        diff = torch.square(
            self.env.joint_pos_target[:, :self.env.num_actuated_dof]
            - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof]
            + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof]
        )

        diff = diff * (
            self.env.last_actions[:, :self.env.num_dof] != 0
        )

        diff = diff * (
            self.env.last_last_actions[:, :self.env.num_dof] != 0
        )

        return torch.sum(diff, dim=1)

    def _reward_joint_vel_limit(self):
        """Penalizes joint velocities exceeding threshold."""
        return torch.sum(
            torch.clamp(
                torch.abs(self.env.dof_vel) - 0.8,
                min=0.0
            ),
            dim=1
        )

    #############################################################
    ################ STABILITY AND CONFIGURATION ################
    #############################################################
    #
    def _reward_base_ang_vel(self):
        return torch.sum(
            torch.square(self.env.base_ang_vel),
            dim=1
        )

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_body_slip(self):
        """
        Penalizes horizontal sliding of the robot base
        while the trunk is in contact with the ground.
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
        return torch.std(
            self.env.dof_pos[:, :self.env.num_actuated_dof],
            dim=1
        )

    # def _reward_feet_on_ground(self):
    #     """Rewards foot-ground contacts."""
    #     contact_forces = self.env.contact_forces[
    #         :, self.env.feet_indices, :
    #     ]

    #     contact_norm = torch.norm(contact_forces, dim=-1)

    #     contacts = (contact_norm > 1.0).float()

    #     return torch.sum(contacts, dim=1)

    def _reward_feet_on_ground(self):
        """
        Rewards stable foot support only when approximately upright.
        """

        contact_forces = self.env.contact_forces[
            :, self.env.feet_indices, :
        ]

        contact_norm = torch.norm(contact_forces, dim=-1)

        contacts = (contact_norm > 1.0).float()

        num_contacts = torch.sum(contacts, dim=1)

        g_z = self.env.projected_gravity[:, 2]

        upright = (g_z < -0.7).float()

        return (
            torch.clamp(num_contacts - 2, min=0.0)
            / 2.0
        ) * upright

    def _reward_posture(self):
        """
        Rewards standing posture when approximately upright.
        Paper:
        exp(-(q-qstand)^2) if |gz + 1| < eps
        """
        q = self.env.dof_pos[:, :self.env.num_actuated_dof]

        q_stand = self.env.default_dof_pos[
            :, :self.env.num_actuated_dof
        ]

        posture_error = torch.sum(
            torch.square(q - q_stand),
            dim=1
        )

        r_posture = torch.exp(-posture_error)

        g_z = self.env.projected_gravity[:, 2]

        # posture reward activates roughly when −1 ≤ g_z ≤ (−1+eps_posture)
        upright = (
            torch.abs(g_z + 1.0) < self.eps_posture
        ).float()

        return r_posture * upright

    def _reward_base_height(self):
        """Rewards maintaining target base height."""
        body_height = self.env.root_states[:, 2]

        target_height = self.env.cfg.rewards.base_height_target

        return torch.exp(
            -torch.square(target_height - body_height)
        )

    #############################################################
    ########################## SAFETY ###########################
    #############################################################

    def _reward_base_contact(self):
        """Penalizes base-ground contact."""
        forces = self.env.contact_forces[
            :, self.env.base_contact_indices, :
        ]

        force_norm = torch.norm(forces, dim=-1)

        return (force_norm > 0.2).any(dim=1).float()

    def _reward_dof_pos_limits(self):
        """Penalizes joint limit violations."""
        q = self.env.dof_pos

        q_min = self.env.dof_pos_limits[:, 0]
        q_max = self.env.dof_pos_limits[:, 1]

        violations = (
            (q < q_min) | (q > q_max)
        ).float()

        return torch.sum(violations, dim=1)
