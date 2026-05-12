import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env
        self.eps_orien = 0.25
        self.eps_posture = 0.25

    def load_env(self, env):
        self.env = env

    ###############################################
    ############ ORIENTATION & POSTURE ############
    ###############################################

    def _reward_base_orientation(self):
        """Penalize base tilt (roll/pitch deviation from upright)."""
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_upright_orientation(self):
        """Rewards alignment of the base with gravity (upright posture)."""
        return torch.exp(
            -torch.square(self.env.projected_gravity[:, 2] + 1.0)
            / (2 * self.eps_orien ** 2)
        )

    def _reward_height_alignment(self):
        """Rewards maintaining the desired base height."""
        body_height = self.env.root_states[:, 2]
        target_height = self.env.cfg.rewards.base_height_target

        return torch.exp(
            -torch.square(target_height - body_height)
        )

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

    def _reward_ang_vel_limit(self):
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

    def _reward_asymmetry(self):
        return torch.std(
            self.env.dof_pos[:, :self.env.num_actuated_dof],
            dim=1
        )

    def _reward_feet_on_ground(self):
        """Rewards foot-ground contacts."""
        contact_forces = self.env.contact_forces[
            :, self.env.feet_indices, :
        ]

        contact_norm = torch.norm(contact_forces, dim=-1)

        contacts = (contact_norm > 1.0).float()

        return torch.sum(contacts, dim=1)

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

# class CoRLRewards:
#     def __init__(self, env):
#         self.env = env
#         self.eps = 0.5
#         # print(self.env.commands.shape)

#     def load_env(self, env):
#         self.env = env

#     ###############################################
#     ############ ORIENTATION & POSTURE ############
#     ###############################################

#     def _reward_base_orientation(self):
#         '''Penalize base tilt (roll/pitch deviation from upright).
#         '''
#         # print(f"Reward base orient: {torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)}")
#         return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

#     # def _reward_upright_orientation(self):
#     #     '''Rewards alignment of the base with gravity (upright posture).
#     #     '''
#     #     return torch.exp(-torch.square(self.env.projected_gravity[:, 2] + 1.0) / (2 * self.eps**2))
#     #
#     def _reward_upright_orientation(self):

#         upright_reward = torch.exp(-torch.square(self.env.projected_gravity[:, 2] + 1.0) / (2 * self.eps**2))
#         low_ang_vel = (torch.norm(self.env.base_ang_vel, dim=1) < 1.0).float()
#         return upright_reward * low_ang_vel

#     def _reward_height_alignment(self):
#         '''Rewards maintaining the desired base height relative to terrain.
#         '''
#         body_height = self.env.root_states[:, 2]
#         # For rough terrain (height terrain-relative)
#         # body_height = self.env.root_states[:, 2] - self.env.measured_heights
#         target_height = self.env.cfg.rewards.base_height_target
#         return torch.exp(-torch.square(target_height - body_height))

#     ###############################################
#     ################ MOTOR CONTROL ################
#     ###############################################

#     def _reward_torques(self):
#         '''Penalizes large joint torques to encourage energy-efficient control.
#         '''
#         return torch.sum(torch.square(self.env.torques), dim=1)

#     def _reward_action(self):
#         '''Penalizes large action magnitudes to avoid aggressive commands.
#         '''
#         return torch.sum(torch.square(self.env.actions), dim=1)

#     def _reward_dof_acc(self):
#         '''Penalizes high joint accelerations for smoother motion.
#         '''
#         # return torch.sum(torch.square(self.env.dof_acc), dim=1)
#         # # softer scaling
#         # acc = self.env.dof_acc
#         # acc_penalty = torch.mean((acc / 50.0)**2, dim=1)

#         acc = torch.clamp(self.env.dof_acc, -200, 200)
#         acc_penalty = torch.mean(acc**2, dim=1)
#         return acc_penalty

#     def _reward_dof_vel(self):
#         '''Penalizes high joint velocities to reduce excessive movement.
#         '''
#         # return torch.sum(torch.square(self.env.dof_vel), dim=1)
#         vel = torch.clamp(self.env.dof_vel, -50, 50)
#         vel_penalty = torch.mean(vel**2, dim=1)
#         return vel_penalty

#     ###################################################
#     ################ MOTION SMOOTHNESS ################
#     ###################################################

#     def _reward_action_smoothness_1(self):
#         '''Penalizes rapid changes in actions (first-order smoothness).
#         '''
#         diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
#         diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
#         return torch.sum(diff, dim=1)

#     def _reward_action_smoothness_2(self):
#         '''Penalizes jerky actions via second-order differences.
#         '''
#         diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
#         diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
#         diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
#         return torch.sum(diff, dim=1)

#     # def _reward_ang_vel_limit(self):
#     #     '''Penalizes joint velocities exceeding a predefined threshold.
#     #     '''
#     #     return torch.sum(torch.clamp(torch.abs(self.env.dof_vel) - 0.8, min=0.0), dim=1)

#     def _reward_ang_vel_limit(self):
#         """penalizes fast motion only after recovery.
#         """
#         penalty = torch.sum(
#             torch.clamp(torch.abs(self.env.dof_vel) - 0.8, min=0.0),
#             dim=1
#         )

#         g_z = self.env.projected_gravity[:, 2]
#         upright = (g_z < -0.7).float()

#         return penalty * upright

#     #############################################################
#     ################ STABILITY AND CONFIGURATION ################
#     #############################################################

#     def _reward_asymmetry(self):
#         return torch.std(self.env.dof_pos[:, :self.env.num_actuated_dof], dim=1)

#     def _reward_feet_on_ground(self):
#         '''Rewards maintaining foot contacts with the ground for stability.
#         '''
#         contact_forces = self.env.contact_forces[:, self.env.feet_indices, :]
#         contact_norm = torch.norm(contact_forces, dim=-1)   # (N, 4)
#         contacts = (contact_norm > 1.0).float()             # threshold 1–5N
#         upright = (self.env.projected_gravity[:, 2] < -0.9).float()

#         return torch.sum(contacts, dim=1) * upright
#         # return torch.sum(contacts, dim=1)

#     def _reward_posture(self):
#         """Counts successful fall-recovery events per episode.
#         A recovery is detected when the robot is upright
#         (projected gravity z < -0.9), reaches a stable height
#         (base height > 0.25 m), and has low linear velocity
#         (‖v‖ < 0.2 m/s). Only the first occurrence per episode
#         is counted using a recovery flag to avoid multiple counts.
#         """
#         q = self.env.dof_pos[:, :self.env.num_actuated_dof]
#         q_stand = self.env.default_dof_pos[:, :self.env.num_actuated_dof]
#         posture_error = torch.sum((q - q_stand)**2, dim=1)
#         r_posture = torch.exp(-posture_error/2.0)
#         g_z = self.env.projected_gravity[:, 2]

#         # Only reward posture when upright
#         upright = (g_z < -0.7).float()
#         return r_posture * upright

#     def _reward_base_height(self):
#         '''Rewards maintaining the target base height above ground.

#         Provide a smooth exponential reward that peaks at the target height,
#         encouraging the robot to stand up during recovery and maintain
#         standing height during normal locomotion.
#         '''
#         body_height = self.env.root_states[:, 2]
#         target_height = self.env.cfg.rewards.base_height_target

#         upright = (self.env.projected_gravity[:, 2] < -0.8).float()

#         reward = torch.exp(
#             -torch.square(target_height - body_height) / 0.04
#         )

#         return reward * upright

#         # return torch.exp(-torch.square(target_height - body_height) / 0.04)


#     #############################################################
#     ########################## SAFETY ###########################
#     #############################################################

#     # def _reward_base_contact(self):
#     #     '''Penalizes contact between the robot base and the ground.
#     #     '''
#     #     forces = self.env.contact_forces[:, self.env.termination_contact_indices, :]  # (N, K, 3)
#     #     force_norm = torch.norm(forces, dim=-1)  # (N, K)
#     #     return (force_norm > 1.0).any(dim=1).float()  # if ANY base body touches

#     def _reward_base_contact(self):
#         forces = self.env.contact_forces[:, self.env.base_contact_indices, :]
#         force_norm = torch.norm(forces, dim=-1)
#         return (force_norm > 0.2).any(dim=1).float()

#     def _reward_dof_pos_limits(self):
#         '''Penalizes joint positions exceeding their allowable limits.
#         '''
#         q = self.env.dof_pos
#         q_min = self.env.dof_pos_limits[:, 0]
#         q_max = self.env.dof_pos_limits[:, 1]

#         violations = ((q < q_min) | (q > q_max)).float()
#         return torch.sum(violations, dim=1)

#     # def _reward_dof_pos_limits(self):
#     #     q = self.env.dof_pos
#     #     q_min = self.env.dof_pos_limits[:, 0]
#     #     q_max = self.env.dof_pos_limits[:, 1]
#     #
#     #     lower_violation = torch.clamp(q_min - q, min=0.0)
#     #     upper_violation = torch.clamp(q - q_max, min=0.0)
#     #
#     #     return torch.sum(lower_violation + upper_violation, dim=1)
