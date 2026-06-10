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

        # Terminal-stabilizer stage:
        # keep episodes alive after recovery so the policy learns to hold stance.
        terminate_on_recovery_success = True

        # # Disable reset-source-based action clipping.
        # # Action limits are now selected only from the robot's physical state.
        # terminal_action_clip = None #0.9 #0.8 #0.50 #0.30

        # # Multi-zone state-based action limits.
        # # Deep fallen / rolling states keep the global action range [-clip_actions, clip_actions].
        # # Partial upright/crouch states are limited to avoid violent terminal collapse.
        # near_crouch_action_clip = 2.0

        # # Near-standing states use a tighter stabilizing action range.
        # near_stand_action_clip = 0.90

        # # Optional diagnostic overrides.
        # # debug_hold_reset_pose: bypass policy and command the reset joint pose.
        # # debug_zero_actions: bypass policy and send zero actions.
        # # debug_log_hold_action: log the hold-pose action for inspection.
        # debug_hold_reset_pose = False
        # debug_zero_actions = False
        # debug_log_hold_action = False

        # # Terminal-only stabilization stage.
        # # 1.0 means every reset starts from the terminal/near-stand reset distribution.
        # # No fallen-state resets are sampled in this stage.
        # terminal_stance_reset_prob = 0.98 #1.0

        # # If True, remove reset noise from the terminal stance reset.
        # # Keep False for normal terminal-reset finetuning.
        # debug_clean_terminal_reset = False
        # debug_log_terminal_reset = True
        #

        terminal_stance_reset_prob = 1.0
        debug_clean_terminal_reset = False
        debug_zero_actions = False
        debug_hold_reset_pose = False

        terminal_action_clip = None
        near_crouch_action_clip = 2.0
        near_stand_action_clip = 0.90

        debug_log_terminal_reset = False

        # ========================
        # DEBUG TERMINAL STATE
        # ========================
        # train_recovery = True

        # # No reset-source-based clipping during reset sanity check.
        # terminal_action_clip = None

        # # Disable state-based clipping for this diagnostic.
        # # With debug_zero_actions=True, these should not matter, but keeping them
        # # disabled makes the test easier to interpret.
        # near_crouch_action_clip = None
        # near_stand_action_clip = None

        # # Reset sanity diagnostic:
        # # bypass policy and send zero actions after reset.
        # debug_hold_reset_pose = False
        # debug_zero_actions = True
        # debug_log_hold_action = False

        # # Use only terminal / near-stand resets.
        # terminal_stance_reset_prob = 1.0

        # # Remove terminal-reset noise for the first diagnostic.
        # debug_clean_terminal_reset = True

        # # Print terminal reset state after reset.
        # debug_log_terminal_reset = True

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
        randomize_link_masses = False #True

        randomize_com_displacement = False #True
        com_displacement_range = [-0.05, 0.05]

        randomize_Kp_factor = False #True
        Kp_factor_range = [0.9, 1.1]

        randomize_Kd_factor = False #True
        Kd_factor_range = [0.9, 1.1]

        randomize_friction = False #True
        friction_range = [0.5, 1.8]

        randomize_restitution = False

        randomize_base_mass = False
        randomize_motor_strength = False

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

    # class rewards(BaseCfg.rewards):
    #     base_height_target = 0.34

    #     # fall-recovery height thresholds
    #     recovery_height_min = 0.18       # no/low height reward below this
    #     recovery_height_success = 0.28   # strict success gate
    #     recovery_height_target = 0.34    # full standing height reward

    #     upright_sigma_strict = 0.25
    #     upright_sigma_soft = 0.35

    #     # Hard recovery success thresholds
    #     recovery_upright_threshold = -0.9
    #     recovery_lin_vel_threshold = 0.3
    #     recovery_ang_vel_threshold = 1.2
    #     recovery_posture_threshold = 2.0
    #     recovery_contact_force_threshold = 1.0
    #     recovery_min_foot_contacts = 3
    #     # recovery_success_steps = 10
    #     recovery_success_steps = 20

    #     # FINETUNE
    #     recovery_foot_slip_vel_threshold = 0.12
    #     # recovery_success_steps = 20
    #     # recovery_ang_vel_threshold = 0.8
    #     # recovery_lin_vel_threshold = 0.2
    #     # recovery_posture_threshold = 1.6
    #     # recovery_min_foot_contacts = 3

    #     # Optional shaping thresholds
    #     # feet_contact_upright_threshold = -0.7
    #     # posture_upright_epsilon = 0.20
    #     #

    #     # Tracking / diagnostics thresholds
    #     nonfoot_contact_threshold = 0.2
    #     loaded_foot_force_threshold = 3.0

    #     # Go1 terminal geometry diagnostic thresholds
    #     front_too_wide_threshold = 0.42
    #     front_too_forward_threshold = 0.28
    #     front_too_narrow_threshold = 0.10
    #     rear_too_narrow_threshold = 0.10

    class rewards(BaseCfg.rewards):
        base_height_target = 0.34

        recovery_height_min = 0.18
        recovery_height_success = 0.28
        recovery_height_target = 0.34

        upright_sigma_strict = 0.25
        upright_sigma_soft = 0.35

        # recovery_lin_vel_threshold = 0.5
        # recovery_ang_vel_threshold = 2.0
        # recovery_posture_threshold = 2.3
        recovery_foot_slip_vel_threshold = 0.12
        recovery_min_foot_contacts = 3
        recovery_success_steps = 25

        recovery_upright_threshold = -0.9
        recovery_lin_vel_threshold = 0.3
        recovery_ang_vel_threshold = 1.2
        recovery_posture_threshold = 2.0
        recovery_contact_force_threshold = 1.0

        # Used when recovery_bonus is restored later.
        recovery_bonus_delay_s = 0.75

        # Tracking / diagnostics thresholds
        nonfoot_contact_threshold = 0.2
        loaded_foot_force_threshold = 3.0

        # Go1 terminal geometry diagnostic thresholds
        front_too_wide_threshold = 0.42
        front_too_forward_threshold = 0.28
        front_too_narrow_threshold = 0.10
        rear_too_narrow_threshold = 0.10

        terminal_action_prior_sigma = 3.0


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

    # class reward_scales(BaseCfg.reward_scales):
    #     recovery_bonus = 200.0
    #     recovery_progress = 8.0

    #     upright_orientation = 3.0
    #     height_alignment = 6.0
    #     posture = 10.0
    #     feet_on_ground = 0.0

    #     base_orientation = -0.3
    #     base_ang_vel = -7.5e-2

    #     action = -2.0e-4
    #     stand_still_action = 0.0

    #     torques = -5.0e-5
    #     dof_acc = -5.0e-8
    #     dof_vel = -1.0e-4

    #     dof_pos_limits = -0.1
    #     joint_vel_limit = -5.0e-4
    #     base_contact = -0.20

    #     feet_slip = -5.0e-2
    #     body_slip = -5.0e-3
    #     action_smoothness_1 = -1.0e-3
    #     action_smoothness_2 = -2.0e-4

    #     stable_foot_support = 5.0
    #     stance_region = 0.2
    #     rear_leg_separation = 0.25
    #     rear_leg_crossing = -5.0

    #     base_height = 0.0
    #

    # class reward_scales(BaseCfg.reward_scales):
    #     recovery_bonus = 200.0
    #     recovery_progress = 6.0

    #     upright_orientation = 2.5
    #     height_alignment = 4.0

    #     # Reduce posture pressure.
    #     # Too much posture reward makes the robot try to match nominal joint angles
    #     # before the feet have formed a stable support polygon.
    #     posture = 10.0

    #     feet_on_ground = 0.0

    #     base_orientation = -0.3

    #     # Keep angular velocity penalty moderate.
    #     # Stronger values can suppress corrective recovery motion.
    #     base_ang_vel = -8.0e-2

    #     action = -1.5e-4
    #     stand_still_action = 0.0

    #     torques = -5.0e-5
    #     dof_acc = -5.0e-8
    #     dof_vel = -1.0e-4

    #     dof_pos_limits = -0.05
    #     joint_vel_limit = -5.0e-4

    #     # General non-foot contact penalty.
    #     # Keep moderate because the robot starts from fallen states.
    #     base_contact = -5.0e-2

    #     # Terminal non-foot contact penalty.
    #     # This should only activate in the late/upright phase.
    #     late_nonfoot_contact = -1.0

    #     # Keep global foot slip penalty mild.
    #     # If too strong, the policy may avoid forming contacts.
    #     feet_slip = -2.0e-2

    #     # Penalize slip only when the foot is actually loaded.
    #     # Reduce from -1e-1; that was too harsh.
    #     loaded_foot_slip = -5.0e-2

    #     # Make missing stable support more important.
    #     # This directly attacks the current issue: only ~1.8 loaded feet and
    #     # <1 non-slipping foot.
    #     support_deficit = -4.0

    #     # Front-leg geometry correction should be weak initially.
    #     # Strong front-leg error can fight the recovery motion before the robot
    #     # learns stable support.
    #     front_leg_error = -0.3

    #     body_slip = -5.0e-3
    #     action_smoothness_1 = -5.0e-4
    #     action_smoothness_2 = -1.0e-4

    #     # Reward hierarchy:
    #     # 1. first get feet loaded,
    #     # 2. then make loaded feet non-slipping.
    #     loaded_foot_support = 4.0
    #     stable_foot_support = 8.0

    #     # Geometry terms should shape the terminal stance, not dominate learning.
    #     stance_region = 0.5
    #     rear_leg_separation = 0.15
    #     rear_leg_crossing = -1.0

    #     base_height = 0.0
    #

    class reward_scales(BaseCfg.reward_scales):
        # Disable sparse bonus temporarily during terminal stabilization.
        # Restore to 200.0 after stable terminal standing improves.
        recovery_bonus = 25.0
        recovery_progress = 6.0

        upright_orientation = 2.5
        height_alignment = 4.0

        # Lower posture pressure. Strong posture reward can fight foot correction.
        posture = 8.0

        feet_on_ground = 0.0

        base_orientation = -0.3
        base_ang_vel = -0.12 #-1.5e-1

        action = -5.0e-4
        stand_still_action = -5.0e-3

        torques = -5.0e-5
        dof_acc = -5.0e-8
        dof_vel = -1.0e-4

        dof_pos_limits = -0.05
        joint_vel_limit = -5.0e-4

        base_contact = -5.0e-2
        late_nonfoot_contact = -1.0

        # Keep normal slip penalty moderate.
        feet_slip = -0.5 #-0.25 #-5.0e-2

        # Penalize slipping specifically while feet are loaded.
        loaded_foot_slip = -0.15 #-0.10

        # Strongly penalize having fewer than about 3 stable feet.
        support_deficit = -4.0

        # Penalize front legs being too wide / too far forward, but not too strongly yet.
        front_leg_error = -1.0

        body_slip = -3.0e-2
        action_smoothness_1 = -5.0e-4
        action_smoothness_2 = -1.0e-4

        # First reward loaded support, then stable non-slipping support.
        loaded_foot_support = 2.0 #1.5 #4.0
        stable_foot_support = 4.0 #3.0 #15.0

        # Geometry is useful, but should not dominate terminal stabilization.
        stance_region = 1.5
        rear_leg_separation = 0.20
        rear_leg_crossing = -1.0

        base_height = 0.0

        # terminal_action_prior = -0.2
    #


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
