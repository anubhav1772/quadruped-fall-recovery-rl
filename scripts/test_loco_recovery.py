import argparse
from pathlib import Path

import isaacgym
assert isaacgym

import torch
from isaacgym import gymtorch

from go1_gym.envs.base.go1_loco_config import LocoCfg

# IMPORTANT:
# This version must inherit from
# legged_robot_loco_recovery.LeggedRobot
from go1_gym.envs.go1.velocity_tracking_loco_recovery import VelocityTrackingEasyEnv

# Policy loading
def load_jit_student(run_dir, device):
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"

    body_path = checkpoint_dir / "body_latest.jit"
    adaptation_path = checkpoint_dir / "adaptation_module_latest.jit"

    assert body_path.exists(), f"Missing: {body_path}"
    assert adaptation_path.exists(), f"Missing: {adaptation_path}"

    body = torch.jit.load(str(body_path), map_location=device).eval()
    adaptation = torch.jit.load(str(adaptation_path), map_location=device).eval()

    def policy(obs, obs_history):
        latent = adaptation(obs_history)
        # Match ActorCritic.act_student():
        # actor input = current obs + estimated privileged context
        actions = body(torch.cat((obs, latent), dim=-1))
        return actions

    return policy


# Separate observation histories
def create_history(num_envs, obs_dim, history_len, device):
    return torch.zeros(num_envs, obs_dim * history_len, device=device, dtype=torch.float)


def push_history(history, obs):
    """
    [o(t-H+1), ..., o(t-1)] -> [o(t-H+2), ..., o(t)]
    """
    obs_dim = obs.shape[1]
    return torch.cat((history[:, obs_dim:], obs), dim=-1)


# Locomotion command
def set_locomotion_command(env, x_vel=0.5, y_vel=0.0, yaw_vel=0.0, gait_name="trotting"):
    gaits = {
        "pronking": [0.0, 0.0, 0.0],
        "trotting": [0.5, 0.0, 0.0],
        "bounding": [0.0, 0.5, 0.0],
        "pacing":   [0.0, 0.0, 0.5],
    }

    gait = torch.tensor(gaits[gait_name], dtype=env.commands.dtype, device=env.device)

    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    # body height
    env.commands[:, 3] = 0.0

    # gait frequency
    env.commands[:, 4] = 3.0

    # phase / offset / bound
    env.commands[:, 5:8] = gait

    # gait duration
    env.commands[:, 8] = 0.5

    # foot swing height
    env.commands[:, 9] = 0.08

    # pitch / roll
    env.commands[:, 10] = 0.0
    env.commands[:, 11] = 0.0

    # stance width / length
    env.commands[:, 12] = 0.25
    env.commands[:, 13] = 0.40

    # auxiliary reward command
    env.commands[:, 14] = 0.0

# Deterministic fall
def inject_fall(env, lateral_vel=2.5, roll_rate=6.0):
    """
    Apply a deterministic disturbance to all envs.

    We change base velocity rather than directly teleporting the
    robot into a fallen orientation, so the fall develops through
    the simulator dynamics.
    """
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # world-frame lateral velocity
    env.root_states[env_ids, 8] = lateral_vel

    # world-frame angular velocity about x
    env.root_states[env_ids, 10] = roll_rate

    env_ids_int32 = env_ids.to(torch.int32)

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--loco-run-dir", type=str, required=True)
    parser.add_argument("--recovery-run-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1000)

    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--x-vel", type=float, default=0.5)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)

    parser.add_argument(
        "--gait",
        type=str,
        default="trotting",
        choices=["pronking", "trotting", "bounding", "pacing"],
    )
    parser.add_argument("--force-fall-step", type=int, default=150, help="-1 disables forced fall")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Evaluation configuration
    LocoCfg.env.num_envs = args.num_envs
    LocoCfg.env.record_video = False

    # Disable DR for the first deterministic integration test.
    LocoCfg.domain_rand.push_robots = False
    LocoCfg.domain_rand.randomize_friction = False
    LocoCfg.domain_rand.randomize_gravity = False
    LocoCfg.domain_rand.randomize_restitution = False
    LocoCfg.domain_rand.randomize_motor_offset = False
    LocoCfg.domain_rand.randomize_motor_strength = False
    LocoCfg.domain_rand.randomize_base_mass = False
    LocoCfg.domain_rand.randomize_Kp_factor = False
    LocoCfg.domain_rand.randomize_Kd_factor = False
    LocoCfg.domain_rand.randomize_com_displacement = False
    LocoCfg.domain_rand.randomize_joint_friction = False

    # Important for first integration test:
    # recovery was not trained with locomotion's randomized action delay.
    LocoCfg.domain_rand.randomize_lag_timesteps = False

    # Prevent random command changes during this evaluation.
    LocoCfg.commands.resampling_time = 1e9

    # Environment
    env = VelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=LocoCfg)

    # DO NOT HistoryWrapper(env)
    base_env = env

    # Load both independently-trained policies
    loco_policy = load_jit_student(args.loco_run_dir, device)
    recovery_policy = load_jit_student(args.recovery_run_dir, device)

    # Reset
    env.reset()

    set_locomotion_command(
        base_env,
        x_vel=args.x_vel,
        y_vel=args.y_vel,
        yaw_vel=args.yaw_vel,
        gait_name=args.gait,
    )

    # Recompute observations because we replaced the randomly
    # sampled command immediately after reset.
    base_env.obs_buf, base_env.privileged_obs_buf = base_env.compute_observations(base_env.loco_cfg)
    base_env.recovery_obs_buf, base_env.recovery_privileged_obs_buf = base_env.compute_observations(base_env.recovery_cfg)

    loco_obs = base_env.obs_buf
    recovery_obs = base_env.recovery_obs_buf

    # Two independent histories
    loco_history_len = base_env.loco_cfg.env.num_observation_history
    recovery_history_len = base_env.recovery_cfg.env.num_observation_history

    loco_history = create_history(base_env.num_envs, loco_obs.shape[1], loco_history_len, base_env.device)
    recovery_history = create_history(base_env.num_envs, recovery_obs.shape[1], recovery_history_len, base_env.device)

    # Put current state into the newest slot.
    loco_history = push_history(loco_history, loco_obs)
    recovery_history = push_history(recovery_history, recovery_obs)

    ever_recovery = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    ever_handoff = torch.zeros_like(ever_recovery)

    previous_mode = base_env.recovery_mode.clone()

    # Evaluation loop
    with torch.inference_mode():
        for step in range(args.num_steps):
            # forced disturbance
            if (args.force_fall_step >= 0 and step == args.force_fall_step):
                print(f"\n[step {step}] injecting fall")
                inject_fall(base_env)

            # Evaluate BOTH policies for ALL environments
            loco_actions = loco_policy(loco_obs, loco_history)
            recovery_actions = recovery_policy(recovery_obs, recovery_history)

            # Per-env controller selection
            actions = torch.where(base_env.recovery_mode[:, None], recovery_actions, loco_actions)
            # Physics step
            _, _, rewards, dones, infos = env.step(actions)

            # Keep locomotion command fixed.
            set_locomotion_command(
                base_env,
                x_vel=args.x_vel,
                y_vel=args.y_vel,
                yaw_vel=args.yaw_vel,
                gait_name=args.gait,
            )

            # The integrated env has already computed both
            # observation streams.
            loco_obs = base_env.obs_buf
            recovery_obs = base_env.recovery_obs_buf

            # Handle true simulator/episode resets
            done_mask = dones.bool()

            if done_mask.any():
                loco_history[done_mask] = 0.0
                recovery_history[done_mask] = 0.0

            # Update BOTH shadow histories
            loco_history = push_history(loco_history, loco_obs)
            recovery_history = push_history(recovery_history, recovery_obs)

            # Track mode transitions
            current_mode = base_env.recovery_mode.clone()

            entered_recovery = (~previous_mode) & current_mode
            left_recovery = previous_mode & (~current_mode)

            ever_recovery |= entered_recovery
            ever_handoff |= left_recovery

            if entered_recovery.any():
                print(f"[step {step}] entered recovery: {entered_recovery.nonzero().flatten().tolist()}")

            if left_recovery.any():
                print(f"[step {step}] handoff -> locomotion: {left_recovery.nonzero().flatten().tolist()}")

            previous_mode = current_mode

            if step % 25 == 0:
                print(
                    f"step={step:04d} | "
                    f"recovery_mode={current_mode.float().mean().item():.2f} | "
                    f"fall={infos.get('fall_detected_frac', 0.0):.2f} | "
                    f"handoff_ready={infos.get('handoff_ready_frac', 0.0):.2f} | "
                    f"new_handoff={infos.get('new_handoff_frac', 0.0):.2f} | "
                    f"vx={base_env.base_lin_vel[:, 0].mean().item():.3f}"
                )

    print("\n========== Integrated evaluation ==========")
    print(
        f"entered_recovery_rate: "
        f"{ever_recovery.float().mean().item():.3f}"
    )
    print(
        f"successful_handoff_rate: "
        f"{ever_handoff.float().mean().item():.3f}"
    )
    print("===========================================\n")


if __name__ == "__main__":
    main()
