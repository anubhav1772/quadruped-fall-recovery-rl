import isaacgym
assert isaacgym

import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from argparse import ArgumentParser
import os

from aliengo_gym.envs import *
from aliengo_gym.envs.base.fall_recovery_config import FallRecoveryConfig as BaseCfg
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        return action

    return policy

def load_env(logdir, headless=True):
    logdir = "/home/anubhav1772/Github/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-04-28/train_fall_recovery/113519.359542"

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(BaseCfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(BaseCfg, key), key2, value2)

    # FORCE RECOVERY MODE
    BaseCfg.env.train_recovery = True

    BaseCfg.domain_rand.push_robots = False
    BaseCfg.domain_rand.randomize_friction = False
    BaseCfg.domain_rand.randomize_gravity = False
    BaseCfg.domain_rand.randomize_restitution = False
    BaseCfg.domain_rand.randomize_motor_offset = False
    BaseCfg.domain_rand.randomize_motor_strength = False
    BaseCfg.domain_rand.randomize_ground_friction = False
    BaseCfg.domain_rand.randomize_base_mass = False

    BaseCfg.env.num_envs = 1
    BaseCfg.env.num_recording_envs = 1

    BaseCfg.terrain.mesh_type = "plane"

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=BaseCfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)

    return env, policy


# Force Fall
def force_fall(env):
    """
    Force a deterministic fallen state
    """
    env_ids = torch.tensor([0], device=env.device)

    # Use trained reset
    env.env._reset_root_states_fall_recovery(env_ids, env.env.cfg)

    # refresh observation after manual reset
    obs = env.get_observations()
    return obs


# Recovery Check
def check_recovered(env):

    height = env.root_states[0, 2].item()

    # projected gravity Z ~ 1 when upright
    upright = env.projected_gravity[0, 2].item()

    vel = torch.norm(env.base_lin_vel[0]).item()

    return (height > 0.25) and (upright > 0.9) and (vel < 0.5)


def eval_recovery(model_dir, num_episodes=50, max_steps=400, headless=True):

    env, policy = load_env(model_dir, headless=headless)

    success = 0
    recovery_steps = []

    for ep in range(num_episodes):

        obs = env.reset()

        # FORCE FALL
        obs = force_fall(env)

        recovered = False
        stable_counter = 0

        for step in range(max_steps):

            with torch.no_grad():
                actions = policy(obs)

            # zero commands for recovery
            env.commands[:] = 0.0

            obs, _, _, _ = env.step(actions)

            if check_recovered(env):
                stable_counter += 1

                # require persistence (important)
                if stable_counter > 20 and not recovered:
                    success += 1
                    recovery_steps.append(step)
                    recovered = True
            else:
                stable_counter = 0

        print(f"Episode {ep:03d}: {'SUCCESS' if recovered else 'FAIL'}")

    print("\n===== RESULTS =====")
    print(f"Success rate: {success}/{num_episodes} = {success/num_episodes:.2f}")

    if recovery_steps:
        print(f"Avg recovery steps: {np.mean(recovery_steps):.2f}")
        print(f"Avg recovery time: {np.mean(recovery_steps) * env.dt:.2f} sec")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=False, default="/home/anubhav1772/Github/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-04-28/train_fall_recovery/113519.359542")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--headless", default=False)
    args = parser.parse_args()

    eval_recovery(
        model_dir=args.model_dir,
        num_episodes=args.episodes,
        headless=args.headless
    )
