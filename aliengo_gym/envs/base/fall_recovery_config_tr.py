from .legged_robot_config import BaseCfg


class FallRecoveryConfig(BaseCfg):
    """Configuration for fall recovery policy training.
    Inherits from BaseCfg and overrides settings optimized for learning
    a fall recovery controller as described in the AFR paper.
    """

    class init_state(BaseCfg.init_state):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m] #.6
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5  # [rad]
        }

    class control(BaseCfg.control):
        control_type = 'actuator_net' #'P'  # P: position, V: velocity, T: torques # actuator_net
        # PD Drive parameters:
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(BaseCfg.asset):
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        foot_name = "foot"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments =  False # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class env(BaseCfg.env):
        train_recovery = True

        debug_log_height_comparison = False

        # Episodes terminate after stable recovery
        terminate_on_recovery_success = True

        # Full fallen-state training
        terminal_stance_reset_prob = 0.15 #0.05
        orientation_probs = [0.50, 0.20, 0.20, 0.10]

        debug_clean_terminal_reset = False
        debug_zero_actions = False
        debug_hold_reset_pose = False

        # Do not restrict recovery exploration initially
        terminal_action_clip = None

        near_crouch_action_clip = None #2.0
        near_stand_action_clip = None #0.90

        debug_log_terminal_reset = False

        robot = "go1"
        num_observations = 42  # ang_vel(3) in base + observe_only_ang_vel adds 3 more + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12)
        num_scalar_observations = 42
        num_privileged_obs = 26 + 176
        # policy_dt = 0.005 * 4 = 0.02 s
        # episode_steps = 9 / 0.02 = 450
        episode_length_s = 9  # 9 / (0.005 * 4) = 450 control steps
        observe_only_ang_vel = True  # prepend ang_vel again for emphasis in recovery task (gives 45 obs total)

        record_video = True
        max_video_frames = 400 # (for episode length of 450)

        # Physical terrain columns used for video recording.
        # One different column is recorded every video interval.
        # video_probe_columns = list(range(20))

        # one representative per enabled terrain family instead of all 20 physical columns
        video_probe_columns = [
            0,   # descending smooth slope
            9,   # stairs direction 1
            10,  # stairs direction 2
            # 11,  # discrete obstacles
            # 13,  # stepping stones
            14,  # random terrain
            16,  # half-flat/half-rough
            5,   # rough slope
            2,   # smooth slope
        ]

        priv_observe_friction = True
        priv_observe_ground_friction = False
        priv_observe_link_masses = True
        priv_observe_com_displacement = True
        priv_observe_Kpd_factor = True
        priv_observe_contact_forces = True
        priv_observe_contact_states = True
        priv_observe_heightmap = True

    class terrain(BaseCfg.terrain):
        mesh_type = "trimesh" # none, plane, heightfield or trimesh
        curriculum = True

        # Recovery-driven terrain progression.
        recovery_curriculum = True

        num_rows = 10       # difficulty levels: 0 – 9
        num_cols = 20       # physical terrain columns

        recovery_start_level = 0
        curriculum_max_level = 9 # (num_row - 1)

        static_friction = 1.0
        dynamic_friction = 1.0

        measure_heights = True

        # Visualize terrain height-sampling points in the Isaac Gym viewer
        # Disable(False) during normal training to avoid rendering overhead
        debug_height_grid = False
        debug_height_grid_env_id = -1
        debug_height_grid_point_radius = 0.025

        measured_points_x = [
            -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
            0.00,
            0.05,  0.10,  0.15,  0.20,  0.25,  0.30,  0.35,
        ]

        measured_points_y = [
            -0.25, -0.20, -0.15, -0.10, -0.05,
            0.00,
            0.05,  0.10,  0.15,  0.20,  0.25,
        ]

        terrain_noise_magnitude = 0.03

        # Do not let center sampling override min/max terrain levels
        center_robots = False

        terrain_proportions = [
            0.15,  # smooth slope
            0.30,  # rough slope
            0.05,  # stairs up
            0.05,  # stairs down
            0.10,  # discrete obstacles
            0.05,  # stepping stones
            0.00,  # gap
            0.00,  # pillar
            0.10,  # random noise
            0.20,  # half-flat half-rough
        ]

        # terrain_proportions = [
        #     0.15,  # smooth slope: 3 columns
        #     0.25,  # rough slope: 5 columns
        #     0.05,  # stairs up: 1 column
        #     0.05,  # stairs down: 1 column
        #     0.10,  # discrete obstacles: 2 columns
        #     0.05,  # stepping stones: 1 column
        #     0.05,  # gap: 1 column
        #     0.05,  # pillars: 1 column
        #     0.10,  # random noise: 2 columns
        #     0.15,  # half-flat half-rough: 3 columns
        # ]

        # Promotion evidence
        frontier_min_trials_per_bin = 32
        frontier_mean_promote_threshold = 0.60   #0.80
        frontier_lower_quartile_threshold = 0.50 #0.70
        frontier_required_windows = 2

        # Cell competence logging
        cell_success_ema_alpha = 0.10
        recovery_rate_ema_alpha = 0.10

        # Per-terrain-column sampling-stage transitions.  A column enters the
        # developing stage after it has a reviewed frontier window with at
        # least this mean success.  It enters mature independently when that
        # column's own frontier reaches mature_frontier_level.
        bootstrap_success_threshold = 0.60
        mature_frontier_level = 3

        # Developing stage
        developing_frontier_fraction = 0.75
        developing_replay_fraction = 0.15
        developing_next_fraction = 0.10
        developing_coverage_fraction = 0.00

        # Mature stage
        mature_frontier_fraction = 0.60
        mature_replay_fraction = 0.25
        mature_next_fraction = 0.10
        mature_coverage_fraction = 0.05

        # Fall-recovery XY spawn distribution
        # keeps 25% easy center samples while making 75% genuinely off-center
        recovery_center_spawn_fraction = 0.25
        recovery_center_spawn_jitter = 0.15

        # Off-center annulus, safely inside the 5 m × 5 m terrain cell
        recovery_spawn_min_radius = 0.55
        recovery_spawn_max_radius = 1.25

    class domain_rand(BaseCfg.domain_rand):
        # trunk_mass_range = [4.0, 28.0]
        # hip_mass_range   = [0.3, 0.7]
        # thigh_mass_range = [0.4, 4.0]
        # calf_mass_range  = [0.1, 0.8]

        trunk_mass_range = [9.5,12.5]

        hip_mass_range   = [0.45, 0.65]
        thigh_mass_range = [1.2, 2.0]
        calf_mass_range  = [0.18, 0.40]

        added_mass_range = [-1., 3.]
        randomize_link_masses = True

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]

        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.5, 1.8]

        randomize_restitution = False

        randomize_base_mass = True
        randomize_motor_strength = False

    class normalization(BaseCfg.normalization):
        clip_actions = 10.0
        com_displacement_range = [-0.05, 0.05]
        friction_range = [0.5, 1.8]
        Kp_factor_range = [0.9, 1.1]
        Kd_factor_range = [0.9, 1.1]
        # trunk_mass_range = [1.5, 30.5]

        # link_mass_ranges = [
        #     [0.3, 0.7],    # hip
        #     [0.4, 4.0],   # thigh
        #     [0.1, 0.8],  # calf
        # ]
        #
        trunk_mass_range = [8.5, 15.5]

        link_mass_ranges = [
            [0.45, 0.65],    # hip
            [1.2, 2.0],   # thigh
            [0.18, 0.40],  # calf
        ]
        relative_height_range = [-0.2, 0.8]

    class rewards(BaseCfg.rewards):
        base_height_target = 0.34

        recovery_height_min = 0.05
        recovery_height_success = 0.28
        recovery_height_target = 0.34

        loaded_foot_force_threshold = 3.0   # score begins increasing
        full_load_force_threshold = 20.0    # score reaches 1
        # minimum vertical ground-reaction force required
        # for a foot to be counted as a valid contact
        recovery_contact_force_threshold = 12.0 #1.0
        recovery_min_foot_contacts = 3

        upright_sigma_strict = 0.25
        upright_sigma_soft = 0.35

        recovery_upright_threshold = -0.9
        recovery_lin_vel_threshold = 0.3
        recovery_ang_vel_threshold = 1.2
        recovery_posture_threshold = 2.0

        # recovery_success_steps = 10     # Stage I: 0.2 seconds (10x0.02), dt = 0.02 = 0.005 x 4
        recovery_success_steps = 20     # Stage II
        require_non_slipping_contacts = True #False

        # Retained for diagnostics and later stages
        recovery_foot_slip_vel_threshold = 0.12

        recovery_vertical_vel_threshold = 0.15
        recovery_nonfoot_contact_threshold = 5.0
        height_alignment_sigma = 0.06

        recovery_bonus_delay_s = 0.5

    # Stage I
    # class reward_scales(BaseCfg.reward_scales):
    #     # Main objective
    #     recovery_bonus = 5000.0     # effective 100.0 (recovery_bonus * dt)
    #     recovery_progress = 20.0

    #     # Recovery shaping
    #     upright_orientation = 3.0
    #     height_alignment = 2.0
    #     posture = 3.0
    #     feet_on_ground = 1.0

    #     # Stability
    #     base_orientation = 0.0
    #     base_ang_vel = -1.0e-2

    #     # Weak regularization
    #     action = -1.0e-3
    #     torques = -2.0e-4
    #     dof_acc = -5.0e-8
    #     dof_vel = -1.0e-4

    #     # Safety
    #     dof_pos_limits = -0.1
    #     joint_vel_limit = 0.0
    #     base_contact = -0.5

    #     # Exploration-safe slip/smoothness
    #     body_slip = -5.0e-3
    #     feet_slip = -5.0e-3
    #     action_smoothness_1 = -1.0e-3
    #     action_smoothness_2 = -2.0e-4

    #     # Late-gated, so it does not suppress rolling
    #     loaded_foot_slip = -1.0

    #     # Disable terminal-refinement objectives in Stage 1
    #     stand_still_action = 0.0
    #     late_nonfoot_contact = 0.0
    #     support_deficit = 0.0
    #     front_leg_error = 0.0
    #     loaded_foot_support = 0.0
    #     stable_foot_support = 0.0
    #     stance_region = 0.0
    #     rear_leg_separation = 0.0
    #     rear_leg_crossing = 0.0
    #     terminal_action_prior = 0.0
    #     base_height = 0.0

    # Stage II
    class reward_scales(BaseCfg.reward_scales):
        # Main objective
        recovery_bonus = 5000.0     # effective 100.0 (recovery_bonus * dt)
        recovery_progress = 8.0

        # Final-pose shaping
        upright_orientation = 3.0
        height_alignment = 2.0
        posture = 3.0
        feet_on_ground = 1.0

        # Stability
        base_orientation = 0.0
        base_ang_vel = -0.03

        # Motor regularization
        action = -1.0e-3
        torques = -2.0e-4
        dof_acc = -5.0e-8
        dof_vel = -1.0e-4
        ang_vel_limit = 0.0

        # Safety
        dof_pos_limits = -0.1
        joint_vel_limit = 0.0
        base_contact = -0.5

        # Exploration-safe regularization
        body_slip = -5.0e-3
        feet_slip = -5.0e-3
        action_smoothness_1 = -1.0e-3
        action_smoothness_2 = -2.0e-4

        # Balance positive support against slip penalty
        loaded_foot_slip = -0.25

        # Keep morphology-specific terminal shaping disabled
        late_nonfoot_contact = 0.0
        terminal_action_prior = 0.0
        base_height = 0.0
        # Your data showed excessive front width
        front_leg_error = 0.0 #-0.5
        # Gentle stance refinement
        stance_region = 0.0 #0.2

        # Establish loaded, stable support
        loaded_foot_support = 2.0
        support_deficit = -1.0
        stand_still_action = -1.0e-3
        stable_foot_support = 5.0

        # Rear geometry is not currently the bottleneck
        rear_leg_separation = 0.0
        rear_leg_crossing = 0.0
