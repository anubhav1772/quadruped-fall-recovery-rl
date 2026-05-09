from .legged_robot_config import BaseCfg


class FallRecoveryConfig(BaseCfg):
    """Configuration for fall recovery policy training.
    Inherits from BaseCfg and overrides settings optimized for learning
    a fall recovery controller as described in the AFR paper.
    """

    class env(BaseCfg.env):
        train_recovery = True
        num_observations = 42  # ang_vel(3) in base + observe_only_ang_vel adds 3 more + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12)
        num_scalar_observations = 42
        num_privileged_obs = 48
        episode_length_s = 15  # longer episodes to allow full recovery sequences (was 3 s)
        observe_only_ang_vel = True  # prepend ang_vel again for emphasis in recovery task (gives 45 obs total)

        record_video = True
        priv_observe_link_masses = True

    class domain_rand(BaseCfg.domain_rand):
        trunk_mass_range = [-2.0, 2.0]
        hip_mass_range   = [0.3, 0.7]
        thigh_mass_range = [0.15, 0.5]
        calf_mass_range  = [0.05, 0.15]
        randomize_link_masses = True

    class reward_scales(BaseCfg.reward_scales):
        # Recovery-priority rewards (significantly increased vs locomotion)
        upright_orientation = 5.0  # was 2.0 — primary recovery objective
        feet_on_ground = 0.5  # was 0.3 — stable stance during recovery
        base_height = 2.0  # was 0.0 — reward for standing up
        posture = 5.0  # was 1.0 — tracking default joint pose (active during recovery)
        base_contact = -0.2
        asymmetry = 0.05
        # Motion quality penalties (kept similar to locomotion)
        torques = -5e-4
        action = -5e-3
        dof_vel = -5e-4
        dof_acc = -2.5e-6
        ang_vel_limit = -0.02
        base_orientation = -0.5
        action_smoothness_1 = -0.01
        action_smoothness_2 = -0.01
        # Locomotion-specific rewards (disabled for pure recovery)
        tracking_lin_vel = 0.0
        tracking_ang_vel = 0.0
        feet_air_time = 0.0

    class normalization(BaseCfg.normalization):

        trunk_mass_range = [7.0, 12.5]

        link_mass_ranges = [
            [0.3, 0.7],    # hip
            [0.15, 0.5],   # thigh
            [0.05, 0.15],  # calf
        ]
