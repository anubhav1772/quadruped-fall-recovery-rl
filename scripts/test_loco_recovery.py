import argparse
from pathlib import Path

import isaacgym
assert isaacgym

import torch
from isaacgym import gymtorch

from aliengo_gym.envs.base.go1_loco_config import LocoCfg
# IMPORTANT:
# This version must inherit from
# legged_robot_loco_recovery.LeggedRobot
from aliengo_gym.envs.aliengo.velocity_tracking_loco_recovery import VelocityTrackingEasyEnv
import pickle as pkl


def load_loco_saved_cfg(run_dir):
    params_path = Path(run_dir) / "parameters.pkl"
    assert params_path.exists(), f"Missing locomotion parameters.pkl: {params_path}"

    with params_path.open("rb") as f:
        saved = pkl.load(f)

    saved_cfg = saved["Cfg"]

    for section_name, values in saved_cfg.items():
        if not hasattr(LocoCfg, section_name):
            continue

        section = getattr(LocoCfg, section_name)

        if isinstance(values, dict):
            for key, value in values.items():
                if not key.startswith("_"):
                    setattr(section, key, value)

    return saved


# Recovery Policy Loading
def load_recovery_jit_student(run_dir, device):
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"

    body_path = checkpoint_dir / "body_latest.jit"
    adaptation_path = checkpoint_dir / "adaptation_module_latest.jit"

    body = torch.jit.load(str(body_path), map_location=device).to(device).eval()
    adaptation = torch.jit.load(str(adaptation_path), map_location=device).to(device).eval()

    def policy(obs, obs_history):
        obs = obs.to(device)
        obs_history = obs_history.to(device)
        ee_output = adaptation(obs_history)
        actions = body(torch.cat((obs, ee_output), dim=-1))
        return actions

    return policy


# Locomotion Policy Loading
def load_loco_jit_student(run_dir, device):
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"

    body_path = checkpoint_dir / "body_latest.jit"
    adaptation_path = checkpoint_dir / "adaptation_module_latest.jit"

    assert body_path.exists(), f"Missing: {body_path}"
    assert adaptation_path.exists(), f"Missing: {adaptation_path}"

    body = torch.jit.load(str(body_path), map_location=device).to(device).eval()
    adaptation = torch.jit.load(str(adaptation_path), map_location=device).to(device).eval()

    def policy(obs, obs_history):
        obs_history = obs_history.to(device)
        latent = adaptation(obs_history)
        actions = body(torch.cat((obs_history, latent), dim=-1))
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
def inject_fall(env, lateral_vel=3.5, roll_rate=6.0):
    """Inject a deterministic velocity disturbance without teleporting pose.
    """
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # world-frame lateral velocity
    # Push toward +y
    env.root_states[env_ids, 8] += lateral_vel  # move strongly toward +y / left

    # world-frame angular velocity about x
    # env.root_states[env_ids, 10] += roll_rate   # positive roll about +x

    # Roll toward the same (+y) side.
    env.root_states[env_ids, 10] -= roll_rate

    env_ids_int32 = env_ids.to(torch.int32)

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--loco-run-dir", type=str, required=True)
    parser.add_argument(
        "--loco-run-dir",
        type=str,
        default="runs/gait-conditioned-agility/2026-02-10/train/194644.419603")
    # parser.add_argument("--recovery-run-dir", type=str, required=True)
    parser.add_argument(
        "--recovery-run-dir",
        type=str,
        default="runs/gait-conditioned-agility/2026-08-24/train_fall_recovery/105731.979852")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=500)

    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--x-vel", type=float, default=0.8)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)

    parser.add_argument(
        "--gait",
        type=str,
        default="trotting",
        choices=["pronking", "trotting", "bounding", "pacing"])
    parser.add_argument("--force-fall-step", type=int, default=-1, help="-1 disables forced fall")

    args = parser.parse_args()

    device = torch.device(args.device)

    load_loco_saved_cfg(args.loco_run_dir)
    LocoCfg.rewards.reward_container_name = "LocomotionRewards"

    # Evaluation configuration
    LocoCfg.env.num_envs = args.num_envs
    LocoCfg.env.record_video = False


    LocoCfg.terrain.num_rows = 5
    LocoCfg.terrain.num_cols = 5
    LocoCfg.terrain.border_size = 0
    LocoCfg.terrain.center_robots = True
    LocoCfg.terrain.center_span = 1

    LocoCfg.terrain.teleport_robots = True

    LocoCfg.terrain.mesh_type = "trimesh"
    LocoCfg.terrain.terrain_proportions = [
        0.20,  # smooth slope
        0.20,  # rough slope
        0.00,  # stairs up
        0.20,  # stairs down
        0.00,  # discrete obstacles
        0.00,  # stepping stones
        0.00,  # gap
        0.00,  # pillar
        0.20,  # random noise
        0.20,  # half-flat half-rough
    ]
    LocoCfg.terrain.terrain_noise_magnitude = 0.03

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
    # LocoCfg.domain_rand.randomize_lag_timesteps = False

    # Prevent random command changes during this evaluation.
    # LocoCfg.commands.resampling_time = 1e9

    LocoCfg.domain_rand.lag_timesteps = 6
    LocoCfg.domain_rand.randomize_lag_timesteps = True
    LocoCfg.control.control_type = "actuator_net"

    # Environment
    env = VelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=LocoCfg)

    # DO NOT HistoryWrapper(env)
    base_env = env

    # Load both independently-trained policies
    loco_policy = load_loco_jit_student(args.loco_run_dir, device)
    recovery_policy = load_recovery_jit_student(args.recovery_run_dir, device)

    # Reset
    env.reset()

    base_env.recovery_mode[:] = False
    base_env.recovery_counter[:] = 0
    base_env.recovered_flag[:] = False
    base_env.handoff_counter[:] = 0
    base_env.recovery_bonus_buf[:] = 0.0

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

    with torch.inference_mode():
        for step in range(args.num_steps):

            # Forced disturbance
            if args.force_fall_step >= 0 and step == args.force_fall_step:
                if base_env.recovery_mode.any():
                    print(
                        f"\n[step {step}] skipping forced fall: "
                        "robot is already in recovery"
                    )
                else:
                    print(f"\n[step {step}] injecting fall")
                    inject_fall(base_env)

            # Mode used to select the controller this step
            action_mode = base_env.recovery_mode.clone()

            # Evaluate BOTH policies
            loco_actions = loco_policy(loco_obs, loco_history)
            recovery_actions = recovery_policy(recovery_obs, recovery_history)

            # Per-env controller selection
            actions = torch.where(
                action_mode[:, None].to(loco_actions.device),
                recovery_actions,
                loco_actions,
            ).to(base_env.device)
            actions = loco_actions.to(base_env.device)

            # Physics step
            _, rewards, dones, infos = env.step(actions)

            # Keep locomotion command fixed
            set_locomotion_command(
                base_env,
                x_vel=args.x_vel,
                y_vel=args.y_vel,
                yaw_vel=args.yaw_vel,
                gait_name=args.gait,
            )

            # Integrated env computed both observation streams
            loco_obs = base_env.obs_buf
            recovery_obs = base_env.recovery_obs_buf

            # Handle true episode resets
            done_mask = dones.bool()

            if done_mask.any():
                loco_history[done_mask] = 0.0
                recovery_history[done_mask] = 0.0

            # Update BOTH policy histories
            loco_history = push_history(loco_history, loco_obs)
            recovery_history = push_history(recovery_history, recovery_obs)

            # Mode after physics/state-machine update.
            # This controller will be selected next step.
            current_mode = base_env.recovery_mode.clone()
            entered_recovery = (~previous_mode) & current_mode

            # Don't count a true episode reset as a handoff
            left_recovery = previous_mode & (~current_mode) & (~done_mask)
            ever_recovery |= entered_recovery
            ever_handoff |= left_recovery

            if entered_recovery.any():
                print(f"[step {step}] >>> entered recovery: {entered_recovery.nonzero().flatten().tolist()}")

            if left_recovery.any():
                print(f"[step {step}] <<< handoff -> locomotion: {left_recovery.nonzero().flatten().tolist()}")

            previous_mode = current_mode

            if step % 25 == 0:
                controller = "RECOVERY" if action_mode[0].item() else "LOCO"

                print(
                    f"step={step:04d} | "
                    f"controller={controller} | "
                    f"recovery_mode="
                    f"{current_mode.float().mean().item():.2f} | "
                    f"fall="
                    f"{infos.get('fall_detected_frac', 0.0):.2f} | "
                    f"done={done_mask.float().mean().item():.2f} | "
                    f"vx="
                    f"{base_env.base_lin_vel[:, 0].mean().item():.3f}"
                )

    print(f"entered_recovery_rate: {ever_recovery.float().mean().item():.3f}")
    print(f"successful_handoff_rate: {ever_handoff.float().mean().item():.3f}")

if __name__ == "__main__":
    main()
