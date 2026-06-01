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
        robot = "go1"
        num_observations = 42  # ang_vel(3) in base + observe_only_ang_vel adds 3 more + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12)
        num_scalar_observations = 42
        num_privileged_obs = 26
        episode_length_s = 9  # longer episodes to allow full recovery sequences (9/4*0.005 = 450)
        observe_only_ang_vel = True  # prepend ang_vel again for emphasis in recovery task (gives 45 obs total)

        record_video = True
        max_video_frames = 400 # (for episode length of 450)

        priv_observe_friction = True
        priv_observe_ground_friction = False
        priv_observe_link_masses = True
        priv_observe_com_displacement = True
        priv_observe_Kpd_factor = True
        priv_observe_contact_forces = True
        priv_observe_contact_states = True
        priv_observe_heightmap = False

    class terrain(BaseCfg.terrain):
        mesh_type = "trimesh" # none, plane, heightfield or trimesh
        static_friction = 1.0
        dynamic_friction = 1.0

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

    class normalization(BaseCfg.normalization):
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

    class rewards(BaseCfg.rewards):
        base_height_target = 0.34

        # fall-recovery height thresholds
        recovery_height_min = 0.18       # no/low height reward below this
        recovery_height_success = 0.28   # strict success gate
        recovery_height_target = 0.34    # full standing height reward

        upright_sigma_strict = 0.25
        upright_sigma_soft = 0.35

        # Hard recovery success thresholds
        recovery_upright_threshold = -0.9
        recovery_lin_vel_threshold = 0.3
        recovery_ang_vel_threshold = 1.2
        recovery_posture_threshold = 2.0
        recovery_contact_force_threshold = 1.0
        recovery_min_foot_contacts = 3
        # recovery_success_steps = 10
        recovery_success_steps = 5

        # FINETUNE
        nonfoot_contact_threshold = 0.2
        recovery_foot_slip_vel_threshold = 0.12
        # recovery_success_steps = 20
        # recovery_ang_vel_threshold = 0.8
        # recovery_lin_vel_threshold = 0.2
        # recovery_posture_threshold = 1.6
        # recovery_min_foot_contacts = 3

        # Optional shaping thresholds
        # feet_contact_upright_threshold = -0.7
        # posture_upright_epsilon = 0.20


    # class reward_scales(BaseCfg.reward_scales):

    #     # Sparse recovery success
    #     recovery_bonus = 200.0

    #     # Soft dense recovery progress
    #     recovery_progress = 30.0

    #     # Standing/recovery shaping
    #     upright_orientation = 6.0
    #     height_alignment = 2.0
    #     feet_on_ground = 2.0
    #     posture = 6.0

    #     # Stability
    #     base_orientation = -0.3
    #     base_ang_vel = -0.02

    #     # Motor regularization
    #     action = -1.0e-3
    #     torques = -2.0e-4
    #     dof_acc = -5.0e-8
    #     dof_vel = -1.0e-4

    #     # Safety / physical realism
    #     dof_pos_limits = -0.1
    #     joint_vel_limit = -5.0e-4
    #     base_contact = -0.05

    #     # Slip / smoothness, kept nonzero but weak
    #     feet_slip = -5.0e-3
    #     body_slip = -5.0e-3
    #     action_smoothness_1 = -1.0e-3
    #     action_smoothness_2 = -2.0e-4

    #     base_height = 0.0

    #     # FINETUNE
    #     # recovery_progress = 15.0   # reduce from 30.0
    #     # base_ang_vel = -5.0e-2     # from -0.02
    #     # feet_slip = -2.0e-2        # from -5e-3
    #     # stable_foot_support = 1.0
    #     # height_alignment = 4.0
    #     # posture = 8.0
    #     # feet_on_ground = 1.0
    #     # base_contact = -0.08

    class reward_scales(BaseCfg.reward_scales):
        recovery_bonus = 200.0
        recovery_progress = 8.0

        upright_orientation = 3.0
        height_alignment = 6.0
        posture = 10.0
        feet_on_ground = 0.0

        base_orientation = -0.3
        base_ang_vel = -7.5e-2

        action = -2.0e-4
        stand_still_action = 0.0

        torques = -5.0e-5
        dof_acc = -5.0e-8
        dof_vel = -1.0e-4

        dof_pos_limits = -0.1
        joint_vel_limit = -5.0e-4
        base_contact = -0.20

        feet_slip = -5.0e-2
        body_slip = -5.0e-3
        action_smoothness_1 = -1.0e-3
        action_smoothness_2 = -2.0e-4

        stable_foot_support = 5.0
        stance_region = 0.2
        rear_leg_separation = 0.25
        rear_leg_crossing = -5.0

        base_height = 0.0




    # class reward_scales(BaseCfg.reward_scales):

    #     # Sparse recovery progress
    #     recovery_bonus = 200.0

    #     # Soft dense progress
    #     recovery_success = 30.0

    #     # standing rewards
    #     upright_orientation = 6.0
    #     height_alignment = 2.0
    #     feet_on_ground = 2.0
    #     posture = 6.0

    #     # Light orientation/stability penalties
    #     base_orientation = -0.5
    #     base_ang_vel = -0.02

    #     # Light motor regularization
    #     action = -2.0e-3
    #     torques = -5.0e-4
    #     dof_acc = -1.0e-7
    #     dof_pos_limits = -0.3

    #     base_contact = -0.2
    #     feet_slip = -0.02
    #     body_slip = -0.02
    #     dof_vel = -5.0e-4
    #     joint_vel_limit = -2.0e-3
    #     action_smoothness_1 = -0.005
    #     action_smoothness_2 = -0.001
    #     base_height = 0.0
