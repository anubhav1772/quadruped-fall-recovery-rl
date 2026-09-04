def train_aliengo(headless=True):

    import isaacgym
    assert isaacgym
    import torch

    # from aliengo_gym.envs.base.fall_recovery_config_go1 import FallRecoveryConfig as Cfg
    from aliengo_gym.envs.base.fall_recovery_config_tr import FallRecoveryConfig as Cfg
    # from aliengo_gym.envs.base.fall_recovery_config import FallRecoveryConfig as Cfg
    # from aliengo_gym.envs.aliengo.aliengo_config import config_aliengo
    from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from aliengo_gym_learn.ppo_cse import Runner
    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from aliengo_gym_learn.ppo_cse.actor_critic import AC_Args
    from aliengo_gym_learn.ppo_cse.ppo import PPO_Args
    from aliengo_gym_learn.ppo_cse import RunnerArgs

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    # runner.learn(num_learning_iterations=20000, init_at_random_ep_len=True, eval_freq=100)
    runner.learn(num_learning_iterations=35000, init_at_random_ep_len=False, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from aliengo_gym import MINI_GYM_ROOT_DIR
    import subprocess
    import os

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )


    run_dir = Path(logger.root) / logger.prefix
    checkpoint_dir = run_dir / "checkpoints"

    msg = (
        f"📁 Run Directory:\n{run_dir}\n\n"
        # f"💾 Checkpoints:\n{checkpoint_dir}"
    )

    try:
        subprocess.run(
            [
                "python",
                "tracebot/send_telegram.py",
                msg
            ],
            check=False
        )
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

    logger.log_text("""
                charts:
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/recovery_success/mean
                  xKey: iterations
                - yKey: train/episode/rew_upright_orientation/mean
                  xKey: iterations
                - yKey: train/episode/rew_height_alignment/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_on_ground/mean
                  xKey: iterations
                - yKey: train/episode/rew_posture/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_slip/mean
                  xKey: iterations
                - yKey: train/episode/rew_body_slip/mean
                  xKey: iterations
                - yKey: recovery/stable_contacts/mean
                  xKey: iterations
                - yKey: recovery/stable_recovery/mean
                  xKey: iterations
                - yKey: recovery/recovery_counter_max/mean
                  xKey: iterations
                - yKey: adaptation_loss/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_aliengo(headless=False)
