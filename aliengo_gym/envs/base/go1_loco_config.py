# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from .legged_robot_config import BaseCfg

class LocoCfg(BaseCfg):
    """Configuration for the Go1 locomotion policy.

    Inherits shared/default settings from BaseCfg and overrides only the
    values that differ for locomotion.
    """

    class init_state(BaseCfg.init_state):
        pos = [0.0, 0.0, 0.34]

    class control(BaseCfg.control):
        control_type = "actuator_net"
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}


    class asset(BaseCfg.asset):
        file = "{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]


    class env(BaseCfg.env):
        num_observations = 58
        num_scalar_observations = 58
        num_privileged_obs = 2

        observe_gaits = True

        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_command = True
        observe_gait_commands = True
        observe_clock_inputs = True
        observe_desired_contact_states = False

        # Additional flags referenced by the shared observation builder.
        priv_observe_Kpd_factor = False
        priv_observe_restitution = True
        priv_observe_joint_friction = True


    class terrain(BaseCfg.terrain):
        mesh_type = "trimesh"

        num_rows = 30
        num_cols = 30

        # 100% random/noisy terrain family used by the original locomotion config.
        terrain_proportions = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        teleport_robots = True
        teleport_thresh = 2.0

        center_robots = False
        center_span = 5


    class commands(BaseCfg.commands):
        command_curriculum = True
        distributional_commands = True

        lin_vel_x = [-2.5, 2.5]
        lin_vel_y = [-1.0, 1.0]

        gait_frequency_cmd_range = [2.0, 4.0]
        gait_duration_cmd_range = [0.5, 0.5]

        limit_vel_y = [-1.5, 1.5]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_duration = [0.5, 0.5]

        binary_phases = True
        balance_gait_distribution = True
        gaitwise_curricula = True


    class domain_rand(BaseCfg.domain_rand):
        randomize_Kp_factor = False
        randomize_Kd_factor = False

        randomize_gravity = True
        randomize_lag_timesteps = True


    class rewards(BaseCfg.rewards):
        only_positive_rewards_ji22_style = True
        reward_container_name = "LocomotionRewards"

        use_terminal_body_height = True
        use_terminal_roll_pitch = False
        terminal_body_ori = 0.5


    class reward_scales(BaseCfg.reward_scales):
        # Locomotion objectives.
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5

        lin_vel_z = -0.02
        ang_vel_xy = -0.001

        torques = -0.0001
        dof_vel = -0.0001
        dof_acc = -2.5e-7

        collision = -5.0
        action_rate = -0.01

        tracking_contacts_shaped_force = 4.0
        tracking_contacts_shaped_vel = 4.0

        jump = 10.0

        dof_pos_limits = -10.0
        feet_slip = -0.04
        feet_clearance_cmd_linear = -30.0

        action_smoothness_1 = -0.1
        action_smoothness_2 = -0.1

        raibert_heuristic = -10.0
        orientation_control = -5.0

        # BaseCfg contains recovery-specific non-zero reward terms that were
        # absent from the original standalone LocoCfg. Explicitly neutralize
        # them so inheritance does not change the locomotion reward.
        action = 0.0
        ang_vel_limit = 0.0
        base_contact = 0.0
        feet_on_ground = 0.0
        posture = 0.0
        base_orientation = 0.0
        upright_orientation = 0.0
        height_alignment = 0.0

    class noise(LocoCfg.noise):
        add_noise = False

    class sim(BaseCfg.sim):
        class physx(BaseCfg.sim.physx):
            # Preserve the original locomotion configuration exactly.
            max_gpu_contact_pairs = 2 ** 23
