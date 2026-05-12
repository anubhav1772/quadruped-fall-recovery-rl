from .legged_robot_config import BaseCfg


class FallRecoveryConfig(BaseCfg):
    """Configuration for fall recovery policy training.
    Inherits from BaseCfg and overrides settings optimized for learning
    a fall recovery controller as described in the AFR paper.
    """

    class env(BaseCfg.env):
        robot = "aliengo"
        train_recovery = True
        num_observations = 42  # ang_vel(3) in base + observe_only_ang_vel adds 3 more + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12)
        num_scalar_observations = 42
        num_privileged_obs = 26
        episode_length_s = 6  # longer episodes to allow full recovery sequences (6/4*0.005 = 300)
        observe_only_ang_vel = True  # prepend ang_vel again for emphasis in recovery task (gives 45 obs total)

        record_video = True
        max_video_frames = 250

        priv_observe_friction = True
        priv_observe_ground_friction = False
        priv_observe_link_masses = True
        priv_observe_com_displacement = True
        priv_observe_Kpd_factor = True
        priv_observe_contact_forces = True
        priv_observe_contact_states = True
        priv_observe_heightmap = False

    class domain_rand(BaseCfg.domain_rand):
        trunk_mass_range = [19.0, 21.5]
        hip_mass_range   = [0.4, 0.7]
        thigh_mass_range = [1.0 2.5]
        calf_mass_range  = [0.15, 0.45]

        added_mass_range = [-1., 3.]
        randomize_link_masses = True

    class reward_scales(BaseCfg.reward_scales):
        # Recovery-priority rewards (significantly increased vs locomotion)
        # upright_orientation = 4.0  # was 2.0 — primary recovery objective
        # feet_on_ground = 0.3  # was 0.3 — stable stance during recovery
        # base_height = 1.5  # was 0.0 — reward for standing up
        # posture = 3.0  # was 1.0 — tracking default joint pose (active during recovery)
        # base_contact = -0.2
        # asymmetry = 0.0 #0.05
        # # Motion quality penalties (kept similar to locomotion)
        # torques = -5e-4
        # action = -1e-2
        # dof_vel = -5e-3 #5e-4
        # dof_acc = -2.5e-6
        # ang_vel_limit = -0.1 #-0.02
        # base_orientation = -0.5
        # action_smoothness_1 = -0.02
        # action_smoothness_2 = -0.02

        upright_orientation = 6.0
        feet_on_ground = 0.5
        base_height = 2.0
        posture = 3.0

        base_contact = -0.2
        base_orientation = -0.5

        torques = -1e-4
        action = -2e-3
        dof_vel = -1e-3
        dof_acc = -1e-6

        ang_vel_limit = -0.02

        action_smoothness_1 = -0.005
        action_smoothness_2 = -0.005

    class rewards(BaseCfg.rewards):
        base_height_target = 0.45

    class normalization(BaseCfg.normalization):
        com_displacement_range = [-0.05, 0.05]
        trunk_mass_range = [18.0, 24.5]

        link_mass_ranges = [
            [0.4, 0.7],    # hip
            [1.0, 2.5],    # thigh
            [0.15, 0.45],  # calf
        ]
