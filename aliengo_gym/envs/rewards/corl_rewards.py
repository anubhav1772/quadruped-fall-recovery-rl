import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env
        self.eps = 0.1
        # print(self.env.commands.shape)

    def load_env(self, env):
        self.env = env

    ###############################################
    ############ ORIENTATION & POSTURE ############
    ###############################################

    def _reward_base_orientation(self):
        '''Penalize base tilt (roll/pitch deviation from upright).
        '''
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_upright_orientation(self):
        '''Rewards alignment of the base with gravity (upright posture).
        '''
        return torch.exp(-torch.square(self.env.projected_gravity[:, 2] + 1.0) / (2 * self.eps**2))

    # def _reward_height_alignment(self):
    #     # return torch.exp(-(self.env.cfg.rewards.base_height_target - self.env.base_pos[:, 2])**2)
    #     reference_heights = 0
    #     body_height = self.env.base_pos[:, 2] - reference_heights
    #     jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
    #     return torch.exp(-torch.square(jump_height_target - body_height))

    def _reward_height_alignment(self):
        '''Rewards maintaining the desired base height relative to terrain.
        '''
        body_height = self.env.base_pos[:, 2]
        # For rough terrain (height terrain-relative)
        body_height = self.env.base_pos[:, 2] - self.env.measured_heights
        target_height = self.env.cfg.rewards.base_height_target
        return torch.exp(-torch.square(target_height - body_height))

    ###############################################
    ################ MOTOR CONTROL ################
    ###############################################

    def _reward_torques(self):
        '''Penalizes large joint torques to encourage energy-efficient control.
        '''
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_action(self):
        '''Penalizes large action magnitudes to avoid aggressive commands.
        '''
        return torch.sum(torch.square(self.env.actions), dim=1)

    def _reward_dof_acc(self):
        '''Penalizes high joint accelerations for smoother motion.
        '''
        return torch.sum(torch.square(self.env.dof_acc), dim=1)

    def _reward_dof_vel(self):
        '''Penalizes high joint velocities to reduce excessive movement.
        '''
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    ###################################################
    ################ MOTION SMOOTHNESS ################
    ###################################################

    def _reward_action_smoothness_1(self):
        '''Penalizes rapid changes in actions (first-order smoothness).
        '''
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        '''Penalizes jerky actions via second-order differences.
        '''
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_ang_vel_limit(self):
        '''Penalizes joint velocities exceeding a predefined threshold.
        '''
        return -torch.sum(torch.clamp(torch.abs(self.env.dof_vel) - 0.8, min=0.0), dim=1)

    #############################################################
    ################ STABILITY AND CONFIGURATION ################
    #############################################################

    def _reward_feet_on_ground(self):
        '''Rewards maintaining foot contacts with the ground for stability.
        '''
        contact_forces = self.env.contact_forces[:, self.env.feet_indices, :]
        contact_norm = torch.norm(contact_forces, dim=-1)   # (N, 4)
        contacts = (contact_norm > 1.0).float()             # threshold 1–5N
        return torch.sum(contacts, dim=1)

    def _reward_posture(self):
        '''Rewards tracking a desired standing joint configuration when upright.
        '''
        q = self.env.dof_pos[:, :self.env.num_actuated_dof]
        q_stand = self.env.default_dof_pos[:, :self.env.num_actuated_dof]
        posture_error = torch.sum((q - q_stand)**2, dim=1)
        r_posture = torch.exp(-posture_error)
        g_z = self.env.projected_gravity[:, 2]
        upright_mask = (torch.abs(g_z + 1.0) < self.eps).float()
        return r_posture * upright_mask

    #############################################################
    ########################## SAFETY ###########################
    #############################################################

    def _reward_base_contact(self):
        '''Penalizes contact between the robot base and the ground.
        '''
        forces = self.env.contact_forces[:, self.env.termination_contact_indices, :]  # (N, K, 3)
        force_norm = torch.norm(forces, dim=-1)  # (N, K)
        return (force_norm > 1.0).any(dim=1).float()  # if ANY base body touches

    def _reward_dof_pos_limits(self):
        '''Penalizes joint positions exceeding their allowable limits.
        '''
        q = self.env.dof_pos
        q_min = self.env.dof_pos_limits[:, 0]
        q_max = self.env.dof_pos_limits[:, 1]

        violations = ((q < q_min) | (q > q_max)).float()
        return torch.sum(violations, dim=1)

    # def _reward_dof_pos_limits(self):
    #     q = self.env.dof_pos
    #     q_min = self.env.dof_pos_limits[:, 0]
    #     q_max = self.env.dof_pos_limits[:, 1]
    #
    #     lower_violation = torch.clamp(q_min - q, min=0.0)
    #     upper_violation = torch.clamp(q - q_max, min=0.0)
    #
    #     return torch.sum(lower_violation + upper_violation, dim=1)


    ###########################################################################################################
    ###########################################################################################################


    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)
    #
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.env.base_lin_vel[:, 2])
    #
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
    #                      dim=1)

    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    #     out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def _reward_jump(self):
    #     reference_heights = 0
    #     body_height = self.env.base_pos[:, 2] - reference_heights
    #     jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
    #     reward = torch.square(body_height - jump_height_target)
    #     return reward

    # def _reward_jump(self):
    #     vz = self.env.base_lin_vel[:, 2]
    #     return vz * vz

    # def _reward_tracking_contacts_shaped_force(self):
    #     foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
    #     desired_contact = self.env.desired_contact_states
    #
    #     reward = 0
    #     for i in range(4):
    #         reward += - (1 - desired_contact[:, i]) * (
    #                     1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
    #     return reward / 4

    # def _reward_tracking_contacts_shaped_vel(self):
    #     foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
    #     desired_contact = self.env.desired_contact_states
    #     reward = 0
    #     for i in range(4):
    #         reward += - (desired_contact[:, i] * (
    #                     1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
    #     return reward / 4

    # def _reward_dof_pos(self):
    #     # Penalize dof positions
    #     return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
    #
    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.env.dof_vel), dim=1)

    # def _reward_feet_slip(self):
    #     contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.env.last_contacts)
    #     self.env.last_contacts = contact
    #     foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
    #     rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
    #     return rew_slip
    #
    # def _reward_feet_contact_vel(self):
    #     reference_heights = 0
    #     near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
    #     foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
    #     rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
    #     return rew_contact_vel
    #
    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
    #                                  dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    #
    # def _reward_feet_clearance_cmd_linear(self):
    #     phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    #     foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
    #     target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
    #     rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
    #     return torch.sum(rew_foot_clearance, dim=1)
    #
    # def _reward_feet_impact_vel(self):
    #     prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
    #     contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0
    #
    #     rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))
    #
    #     return torch.sum(rew_foot_impact_vel, dim=1)
    #
    #
    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
    #                      dim=1)
    #
    # def _reward_orientation_control(self):
    #     # Penalize non flat base orientation
    #     roll_pitch_commands = self.env.commands[:, 10:12]
    #     quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
    #                                      torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
    #     quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
    #                                       torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))
    #
    #     desired_base_quat = quat_mul(quat_roll, quat_pitch)
    #     desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)
    #
    #     return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    #
    # def _reward_raibert_heuristic(self):
    #     cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
    #     footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
    #     for i in range(4):
    #         footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
    #                                                           cur_footsteps_translated[:, i, :])
    #
    #     # nominal positions: [FR, FL, RR, RL]
    #     if self.env.cfg.commands.num_commands >= 13:
    #         desired_stance_width = self.env.commands[:, 12:13]
    #         desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
    #     else:
    #         desired_stance_width = 0.3
    #         desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)
    #
    #     if self.env.cfg.commands.num_commands >= 14:
    #         desired_stance_length = self.env.commands[:, 13:14]
    #         desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
    #     else:
    #         desired_stance_length = 0.45
    #         desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)
    #
    #     # raibert offsets
    #     phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
    #     frequencies = self.env.commands[:, 4]
    #     durations = self.env.commands[:, 8]
    #     x_vel_des = self.env.commands[:, 0:1]
    #     yaw_vel_des = self.env.commands[:, 2:3]
    #
    #     y_vel_des = yaw_vel_des * desired_stance_length / 2
    #     # desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    #     desired_ys_offset = phases * y_vel_des * (durations.unsqueeze(1) / frequencies.unsqueeze(1))
    #     desired_ys_offset[:, 2:4] *= -1
    #     # desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))
    #     desired_xs_offset = phases * x_vel_des * (durations.unsqueeze(1) / frequencies.unsqueeze(1))
    #
    #     desired_ys_nom = desired_ys_nom + desired_ys_offset
    #     desired_xs_nom = desired_xs_nom + desired_xs_offset
    #
    #     desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
    #
    #     err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
    #
    #     reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
    #
    #     return reward
    #
    # def _reward_diagonal_phase(self):
    #     # foot_indices: [num_envs, 4] in [0, 1)
    #     foot_phase = self.env.foot_indices
    #
    #     FL, FR, RL, RR = 0, 1, 2, 3
    #
    #     # diagonal legs should be IN phase
    #     diag1 = torch.cos(2 * torch.pi * (foot_phase[:, FL] - foot_phase[:, RR]))
    #     diag2 = torch.cos(2 * torch.pi * (foot_phase[:, FR] - foot_phase[:, RL]))
    #
    #     # average diagonal alignment
    #     return 0.5 * (diag1 + diag2)
    #
    #
    # def _reward_tracking_norm_lin_vel(self):
    #     # commanded and actual linear velocities
    #     v_cmd = self.env.commands[:, :2]
    #     v     = self.env.base_lin_vel[:, :2]
    #
    #     # Euclidean tracking error
    #     err = torch.norm(v_cmd - v, dim=1)
    #
    #     # normalize by command magnitude
    #     cmd_mag = torch.norm(v_cmd, dim=1)
    #
    #     # relative error (avoid divide by zero)
    #     rel_err = err / (cmd_mag + 0.1)
    #
    #     return torch.exp(-rel_err)
    #
    # def _reward_speed_ratio(self):
    #     eps = 1e-3
    #     # alpha = self.cfg.rewards.speed_ratio_alpha
    #     alpha = 0.45 #0.7
    #
    #     v_xy      = self.env.base_lin_vel[:, :2]
    #     v_cmd_xy = self.env.commands[:, :2]
    #
    #     cmd_norm = torch.norm(v_cmd_xy, dim=1)
    #     valid = cmd_norm > 0.2
    #
    #     cmd_dir = torch.zeros_like(v_cmd_xy)
    #     cmd_dir[valid] = v_cmd_xy[valid] / (cmd_norm[valid].unsqueeze(1) + eps)
    #
    #     speed_along_cmd = torch.sum(v_xy * cmd_dir, dim=1)
    #     speed_along_cmd = torch.clamp(speed_along_cmd, min=0.0)
    #
    #     speed_ratio = torch.zeros_like(cmd_norm)
    #     speed_ratio[valid] = speed_along_cmd[valid] / (cmd_norm[valid] + eps)
    #
    #     self.speed_ratio = speed_ratio
    #     self.speed_ok = speed_ratio >= alpha
    #
    #     penalty = torch.relu(alpha - speed_ratio)
    #     penalty = torch.clamp(penalty, max=0.05)
    #
    #     return penalty
    #
    # def _reward_action_rate(self):
    #     diff = self.env.actions - self.env.last_actions
    #     return torch.mean(diff * diff, dim=1)
    #
    # def _reward_energy(self):
    #     m = 21.5  # kg
    #     g = 9.81  # m/s
    #     vmax = self.env.vel_x_limit[1]
    #     # actual_vel = torch.abs(self.env.base_lin_vel[:, 0])
    #     cmd_vel = torch.abs(self.env.commands[:, 0])
    #
    #     # normalize power
    #     power_norm = self.env.energy_consume / (m*g*vmax)
    #     return power_norm / (cmd_vel + 0.15)
    #     # return power_norm / (actual_vel + 0.15)
    #
    #
    # def _reward_power(self):
    #     # return instantaneous power
    #     # torques = self.env.torques                          # [N, 12]
    #     # vel = self.env.dof_vel                              # [N, 12]
    #
    #     # # Instantaneous power for each joint
    #     # power = torques * vel                               # [N, 12]
    #
    #     # # Total power (sum across joints) - can be positive or negative
    #     # total_power = torch.sum(power, dim=1)    # [N]
    #     # power_consumed = torch.clamp(total_power, min=0)
    #
    #     return self.env.energy_consume
    #
    # def _reward_vel_smoothness(self):
    #     jvel = self.env.dof_vel                              # [N, 12]
    #     jvel_last = self.env.last_dof_vel                    # [N, 12]
    #
    #     diff = (jvel - jvel_last)**2
    #
    #     return torch.sum(diff, dim=1)
    #
    # def _reward_torque_smoothness(self):
    #     # Penalize jerky motions
    #     torque_changes = torch.norm(self.env.torques - self.env.last_torques, dim=1)
    #     return torque_changes
    #
    # # def _reward_stand_still(self):
    # #     # From Legged Gym
    # #     # https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
    # #     # Penalize motion at zero commands
    # #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    #
    # def _reward_stand_still(self):
    #     action_penalty = torch.sum(torch.abs(self.env.actions), dim=1)
    #     near_zero_cmd = (torch.norm(self.env.commands[:, :3], dim=1) < 0.1).float()
    #     return action_penalty * near_zero_cmd
    #
    # def _reward_save_energy(self):
    #     """Stable CoT calculation with velocity clamping."""
    #     P = torch.sum(torch.abs(self.env.torques * self.env.dof_vel), dim=1)
    #
    #     # CLAMP velocity to avoid instability
    #     # But also track what we're penalizing
    #     actual_vel = torch.abs(self.env.base_lin_vel[:, 0])
    #     cmd_vel = torch.abs(self.env.commands[:, 0])
    #
    #     # If actual velocity is very low (< 0.05 m/s), use commanded velocity
    #     # This prevents cheating by standing still
    #     v_for_cot = torch.where(
    #         actual_vel < 0.05,
    #         torch.clamp(cmd_vel, min=0.1),  # Use commanded, min 0.1
    #         torch.clamp(actual_vel, min=0.1)  # Use actual, min 0.1
    #     )
    #
    #     # Stable CoT calculation
    #     CoT = P / (21.5 * 9.81 * v_for_cot)
    #
    #     return CoT
    #
    #
    # def _reward_energy_cot(self):
    #     """CoT-based energy efficiency reward."""
    #     # Constants
    #     m = 21.5  # kg
    #     g = 9.81  # m/s²
    #     eps = 0.1
    #
    #     # Get norm of vx and vy
    #     vxy = torch.norm(self.env.base_lin_vel[:, 0:2], dim=1)
    #
    #     # Calculate total energy consumed (from torques and velocities)
    #     # power = self.env.torques * self.env.dof_vel
    #     # energy_rate = torch.sum(power, dim=1)  # J/s = W
    #
    #     # Instantaneous CoT
    #     # We are using instantaneous power and instantaneous speed
    #     CoT = self.env.energy_consume / (m * g * torch.clamp(vxy, min=eps))
    #     # print(f"vxy: {vxy}, Energy Consume: {torch.mean(self.env.energy_consume)}, \
    #     #         Torques {self.env.torques}, \
    #     #         Mean CoT: {torch.mean(CoT)}")
    #     return CoT
