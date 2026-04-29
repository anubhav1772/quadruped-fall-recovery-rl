from .legged_robot_config import BaseCfg


class FallRecoveryConfig(BaseCfg):
    """Configuration for fall recovery policy training.
    Inherits from BaseCfg and overrides settings optimized for learning
    a fall recovery controller as described in the AFR paper.
    Key changes vs locomotion config:
      - Longer episodes to allow time for recovery sequences
      - Higher reward scales for orientation and height to prioritize recovery
      - Angular velocity included in observation twice (base + flag) for 45-dim obs
      - Termination only on irrecoverable collapses (not on tilted/recovering states)
    """

    class env(BaseCfg.env):
        train_recovery = True
        num_observations = 42  # ang_vel(3) in base + observe_only_ang_vel adds 3 more + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12)
        num_scalar_observations = 42
        episode_length_s = 15  # longer episodes to allow full recovery sequences (was 3 s)
        observe_only_ang_vel = True  # prepend ang_vel again for emphasis in recovery task (gives 45 obs total)

        record_video = False

    class reward_scales(BaseCfg.reward_scales):
        # Recovery-priority rewards (significantly increased vs locomotion)
        upright_orientation = 10.0  # was 2.0 — primary recovery objective
        feet_on_ground = 2.0  # was 0.3 — stable stance during recovery
        base_height = 2.0  # was 0.0 — reward for standing up
        posture = 5.0  # was 1.0 — tracking default joint pose (active during recovery)
        # Motion quality penalties (kept similar to locomotion)
        torques = -5e-4
        action = -1e-2
        dof_vel = -1e-2
        dof_acc = -2.5e-6
        action_smoothness_1 = -0.05
        action_smoothness_2 = -0.05
        # Locomotion-specific rewards (disabled for pure recovery)
        tracking_lin_vel = 0.0
        tracking_ang_vel = 0.0
        feet_air_time = 0.0
