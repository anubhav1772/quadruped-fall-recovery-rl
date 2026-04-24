"""Training script for the fall recovery policy.

Trains a fall recovery controller using PPO with curriculum-based initialization
where the robot starts in fallen poses and must learn to stand back up.

The observation space follows the AFR paper:
    ot = (omega, g, q, qdot, a_prev)  [45-dim with FallRecoveryConfig]

Usage:
    python scripts/train_fall_recovery.py --headless=True --num_envs=4096

Monitor training with:
    - train/episode/rew_recovery_success/mean  — fraction of steps spent recovered
    - train/episode/rew_upright_orientation/mean — orientation reward
    - train/episode/rew_base_height/mean        — height reward
"""


def train_fall_recovery(headless=True, num_envs=None):

    import isaacgym
    assert isaacgym
    import torch

    from aliengo_gym.envs.base.legged_robot_config import FallRecoveryConfig as Cfg
    from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from aliengo_gym_learn.ppo_cse import Runner
    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from aliengo_gym_learn.ppo_cse.actor_critic import AC_Args
    from aliengo_gym_learn.ppo_cse.ppo import PPO_Args
    from aliengo_gym_learn.ppo_cse import RunnerArgs

    if num_envs is not None:
        Cfg.env.num_envs = num_envs

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # Log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=50000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from aliengo_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'fall-recovery/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts:
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_recovery_success/mean
                  xKey: iterations
                - yKey: train/episode/rew_upright_orientation/mean
                  xKey: iterations
                - yKey: train/episode/rew_base_height/mean
                  xKey: iterations
                - yKey: train/episode/rew_posture/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_on_ground/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)

    # To see environment rendering, set headless=False
    train_fall_recovery(headless=True)
