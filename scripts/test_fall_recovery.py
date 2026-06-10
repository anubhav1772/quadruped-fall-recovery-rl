"""
Deterministic fall-recovery evaluation script with close viewer camera.

Purpose
--------
Load a trained fall-recovery actor-critic checkpoint and evaluate the
deterministic student/mean action, not sampled PPO actions.

Default behavior:
    - one robot/env
    - GUI mode unless --headless is passed
    - full fallen resets with terminal_stance_reset_prob = 0.0
    - close camera following robot 0
    - separate handoff_ready counter for deployment-style validation

Command
--------
python scripts/test_fall_recovery.py \
  --run-dir /home/anubhav1772/Github/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-05/train_fall_recovery/142349.605195 \
  --checkpoint last \
  --num-envs 1 \
  --num-steps 450 \
  --terminal-stance-reset-prob 0.0 \
  --handoff-success-steps 25 \
  --camera-distance 1.15 \
  --camera-height 0.45 \
  --print-every 10

"""
import isaacgym
assert isaacgym

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

import glob
import pickle as pkl

from aliengo_gym.envs.base.fall_recovery_config_go1 import FallRecoveryConfig as Cfg
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv
from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper
from aliengo_gym_learn.ppo_cse.actor_critic import ActorCritic, AC_Args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    ckpt = parser.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--checkpoint-path", type=str, default=None, help="Full path to ac_weights_*.pt")
    ckpt.add_argument("--run-dir", type=str, default=None, help="Run directory containing checkpoints/ac_weights_*.pt")

    parser.add_argument("--checkpoint", type=str, default="last", help="Checkpoint name when --run-dir is used. Use 'last' or an iteration number, e.g. 39200.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=450)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Viewer:
    # If omitted -> headless=False -> Isaac Gym viewer opens.
    # If passed -> headless=True -> no viewer.
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--max-video-frames", type=int, default=400)

    # 0.0 = evaluate full fall recovery.
    # 1.0 = evaluate only terminal stance stabilization.
    parser.add_argument("--terminal-stance-reset-prob", type=float, default=0.0)

    # Usually False for evaluation, because we want to keep measuring
    # whether handoff remains stable after recovery.
    parser.add_argument("--terminate-on-success", action="store_true")

    parser.add_argument("--handoff-success-steps", type=int, default=25)
    parser.add_argument("--log-csv", type=str, default=None)
    parser.add_argument("--print-every", type=int, default=25)

    # Close camera follow for one-robot GUI evaluation.
    parser.add_argument("--camera-follow", dest="camera_follow", action="store_true", default=True)
    parser.add_argument("--no-camera-follow", dest="camera_follow", action="store_false")
    parser.add_argument("--camera-distance", type=float, default=1.15, help="Camera distance from robot 0. Smaller = closer.")
    parser.add_argument("--camera-height", type=float, default=0.45, help="Camera height above robot base.")
    parser.add_argument("--camera-update-every", type=int, default=5, help="Update viewer camera every N policy steps.")

    return parser.parse_args()


# def load_policy(logdir, device="cpu"):
#     import torch
#     from pathlib import Path

#     logdir = Path(logdir)

#     body = torch.jit.load(str(logdir / "checkpoints" / "body_latest.jit"), map_location=device)
#     adaptation_module = torch.jit.load(
#         str(logdir / "checkpoints" / "adaptation_module_latest.jit"),
#         map_location=device,
#     )

#     body.eval()
#     adaptation_module.eval()

#     def policy(obs, info={}):
#         with torch.inference_mode():
#             obs_now = obs["obs"].to(device)
#             obs_history = obs["obs_history"].to(device)

#             ee_output = adaptation_module(obs_history)
#             action = body(torch.cat((obs_now, ee_output), dim=-1))

#             info["ee_output"] = ee_output.detach().cpu()

#         return action

#     return policy

def load_saved_params(run_dir, Cfg, AC_Args=None):
    """
    Restore the exact training config saved in parameters.pkl.

    This should be called before env creation and before ActorCritic creation.
    """
    params_path = Path(run_dir).expanduser().resolve() / "parameters.pkl"
    assert params_path.exists(), f"parameters.pkl not found: {params_path}"

    with params_path.open("rb") as file:
        pkl_cfg = pkl.load(file)

    # Restore Cfg
    if "Cfg" in pkl_cfg:
        saved_cfg = pkl_cfg["Cfg"]

        for section_name, section_values in saved_cfg.items():
            if not hasattr(Cfg, section_name):
                continue

            section_obj = getattr(Cfg, section_name)

            if isinstance(section_values, dict):
                for key, value in section_values.items():
                    setattr(section_obj, key, value)
            else:
                setattr(Cfg, section_name, section_values)

    # Restore ActorCritic architecture args
    if AC_Args is not None and "AC_Args" in pkl_cfg:
        for key, value in pkl_cfg["AC_Args"].items():
            if not key.startswith("_"):
                setattr(AC_Args, key, value)

    return pkl_cfg

def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return Path(args.checkpoint_path).expanduser().resolve()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.checkpoint == "last" or args.checkpoint == "-1":
        return run_dir / "checkpoints" / "ac_weights_last.pt"

    return run_dir / "checkpoints" / f"ac_weights_{int(args.checkpoint):06d}.pt"


def unwrap_env(env: Any) -> Any:
    """
    Unwrap HistoryWrapper-like wrappers to access the base Isaac Gym env.
    """
    base = env
    visited = set()

    while hasattr(base, "env") and id(base) not in visited:
        visited.add(id(base))
        base = base.env

    return base


def get_obs_dict(env: Any) -> Dict[str, torch.Tensor]:
    obs = env.get_observations()

    if not isinstance(obs, dict):
        raise RuntimeError(
            "Expected HistoryWrapper.get_observations() to return a dict "
            "with keys 'obs', 'privileged_obs', and 'obs_history'. "
            "Make sure env = HistoryWrapper(env)."
        )

    required = ["obs", "privileged_obs", "obs_history"]
    missing = [k for k in required if k not in obs]
    if missing:
        raise RuntimeError(f"Observation dict is missing keys: {missing}")

    return obs


def unpack_step(ret: Any, env: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Handle both:
        HistoryWrapper.step(...) -> obs_dict, rewards, dones, infos
    and raw env:
        env.step(...) -> obs, privileged_obs, rewards, dones, infos
    """
    if isinstance(ret, tuple) and len(ret) == 4:
        obs_dict, rewards, dones, infos = ret
        return obs_dict, rewards, dones, infos

    if isinstance(ret, tuple) and len(ret) == 5:
        _obs, _priv_obs, rewards, dones, infos = ret
        obs_dict = get_obs_dict(env)
        return obs_dict, rewards, dones, infos

    raise RuntimeError(
        f"Unexpected env.step return format: {type(ret)} "
        f"length={len(ret) if isinstance(ret, tuple) else 'NA'}"
    )


def compute_handoff_ready_fallback(base_env: Any) -> torch.Tensor:
    """
    Deployment-style fallback handoff gate if base_env.compute_handoff_ready()
    is not available.

    Criterion:
        upright + sufficient height + low planar velocity + low angular velocity
        + at least N loaded non-slipping feet.
    """
    cfg = base_env.cfg.rewards

    g_z = base_env.projected_gravity[:, 2]

    upright_thr = getattr(cfg, "recovery_upright_threshold", -0.85)
    height_thr = getattr(cfg, "recovery_height_success", 0.28)
    lin_vel_thr = getattr(cfg, "recovery_lin_vel_threshold", 0.25)
    ang_vel_thr = getattr(cfg, "recovery_ang_vel_threshold", 1.0)
    contact_force_thr = getattr(cfg, "recovery_contact_force_threshold", 1.0)
    slip_vel_thr = getattr(cfg, "recovery_foot_slip_vel_threshold", 0.12)
    min_contacts = getattr(cfg, "recovery_min_foot_contacts", 3)

    upright = g_z < upright_thr
    height_ok = base_env.root_states[:, 2] > height_thr
    low_xy_vel = torch.norm(base_env.base_lin_vel[:, :2], dim=1) < lin_vel_thr
    low_ang_vel = torch.norm(base_env.base_ang_vel, dim=1) < ang_vel_thr

    foot_contact = (
        base_env.contact_forces[:, base_env.feet_indices, 2]
        > contact_force_thr
    )

    foot_xy_vel = torch.norm(base_env.foot_velocities[:, :, :2], dim=-1)

    non_slipping_feet = foot_contact & (foot_xy_vel < slip_vel_thr)
    stable_contacts = non_slipping_feet.sum(dim=1) >= min_contacts

    return upright & height_ok & low_xy_vel & low_ang_vel & stable_contacts


def compute_handoff_ready(base_env: Any) -> torch.Tensor:
    if hasattr(base_env, "compute_handoff_ready"):
        return base_env.compute_handoff_ready()
    return compute_handoff_ready_fallback(base_env)


def update_viewer_camera(base_env: Any, args: argparse.Namespace) -> None:
    """
    Keep Isaac Gym viewer camera close to robot 0.

    This calls base_env.set_camera(), which should internally use
    gym.viewer_camera_look_at(...).

    Only active when:
        - args.headless == False
        - args.camera_follow == True
        - viewer exists
    """
    if args.headless or not args.camera_follow:
        return

    if not hasattr(base_env, "set_camera"):
        return

    if getattr(base_env, "viewer", None) is None:
        return

    if not hasattr(base_env, "root_states") or base_env.root_states is None:
        return

    p = base_env.root_states[0, 0:3].detach().cpu()
    x, y, z = float(p[0]), float(p[1]), float(p[2])

    d = float(args.camera_distance)
    h = float(args.camera_height)

    # Oblique close view.
    # If too far: reduce --camera-distance.
    # If too low: increase --camera-height.
    cam_pos = [
        x + 0.65 * d,
        y - d,
        max(0.35, z + h),
    ]

    cam_target = [
        x,
        y,
        z + 0.10,
    ]

    base_env.set_camera(cam_pos, cam_target)


def get_debug_value(infos: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    if not isinstance(infos, dict):
        return default

    dbg = infos.get("recovery_debug", {})
    if not isinstance(dbg, dict):
        return default

    val = dbg.get(key, default)

    if torch.is_tensor(val):
        return float(val.detach().float().mean().cpu().item())

    if isinstance(val, (float, int)):
        return float(val)

    return default


def actor_deterministic_action(actor_critic: Any, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
    """
    Use deterministic student action.

    Preferred:
        actor_critic.act_student(obs, obs_history)

    Internally, act_student() computes the adaptation/estimator output from obs_history,
    concatenates it with the current obs, and passes that into the actor body.
    """
    if hasattr(actor_critic, "act_student"):
        return actor_critic.act_student(obs, obs_history)

    raise RuntimeError(
        "actor_critic.act_student(...) not found. "
        "For deterministic evaluation, add/use act_student in ActorCritic."
    )


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    device = torch.device(args.device)

    # ------------------------------------------------------------
    # Load saved training parameters FIRST
    # ------------------------------------------------------------
    # This restores the exact Cfg and AC_Args used by this run.
    # Important for comparing policies from different training stages.
    if args.run_dir is not None:
        pkl_cfg = load_saved_params(
            run_dir=args.run_dir,
            Cfg=Cfg,
            AC_Args=AC_Args,
        )
        print(pkl_cfg)
    else:
        pkl_cfg = None
        print("[WARN] --run-dir not provided; using current Cfg and AC_Args from source files.")

    # ------------------------------------------------------------
    # Apply evaluation overrides after loading parameters.pkl
    # ------------------------------------------------------------
    Cfg.env.train_recovery = True
    Cfg.env.num_envs = int(args.num_envs)

    Cfg.env.record_video = bool(args.record_video)
    Cfg.env.max_video_frames = int(args.max_video_frames)

    # Force full-fall evaluation by default.
    # This must come AFTER loading parameters.pkl, otherwise saved config
    # may overwrite it back to terminal_stance_reset_prob = 1.0.
    Cfg.env.terminal_stance_reset_prob = float(args.terminal_stance_reset_prob)

    # Keep episodes alive after recovery unless explicitly requested.
    Cfg.env.terminate_on_recovery_success = bool(args.terminate_on_success)

    # Disable debug action/reset overrides for evaluation
    Cfg.env.debug_zero_actions = False
    Cfg.env.debug_hold_reset_pose = False
    Cfg.env.debug_clean_terminal_reset = False

    if hasattr(Cfg.rewards, "handoff_success_steps"):
        Cfg.rewards.handoff_success_steps = int(args.handoff_success_steps)

    # ------------------------------------------------------------
    # Print final config actually used for env creation
    # ------------------------------------------------------------
    print("\n========== Deterministic fall-recovery evaluation ==========")
    print(f"checkpoint: {checkpoint_path}")
    print(f"num_envs: {Cfg.env.num_envs}")
    print(f"num_steps: {args.num_steps}")
    print(f"device: {args.device}")
    print(f"headless: {args.headless}")
    print(f"record_video: {Cfg.env.record_video}")
    print(f"terminal_stance_reset_prob: {Cfg.env.terminal_stance_reset_prob}")
    print(f"terminate_on_recovery_success: {Cfg.env.terminate_on_recovery_success}")
    print(f"handoff_success_steps: {args.handoff_success_steps}")
    print(f"camera_follow: {args.camera_follow}")
    print(f"camera_distance: {args.camera_distance}")
    print(f"camera_height: {args.camera_height}")

    # Useful sanity checks
    print("\n[RECOVERY THRESHOLDS]")
    print(f"recovery_success_steps: {getattr(Cfg.rewards, 'recovery_success_steps', None)}")
    print(f"recovery_upright_threshold: {getattr(Cfg.rewards, 'recovery_upright_threshold', None)}")
    print(f"recovery_height_success: {getattr(Cfg.rewards, 'recovery_height_success', None)}")
    print(f"recovery_lin_vel_threshold: {getattr(Cfg.rewards, 'recovery_lin_vel_threshold', None)}")
    print(f"recovery_ang_vel_threshold: {getattr(Cfg.rewards, 'recovery_ang_vel_threshold', None)}")
    print(f"recovery_min_foot_contacts: {getattr(Cfg.rewards, 'recovery_min_foot_contacts', None)}")
    print(f"recovery_foot_slip_vel_threshold: {getattr(Cfg.rewards, 'recovery_foot_slip_vel_threshold', None)}")

    print("\n[ACTOR CRITIC ARGS]")
    print(f"actor_hidden_dims: {AC_Args.actor_hidden_dims}")
    print(f"critic_hidden_dims: {AC_Args.critic_hidden_dims}")
    print(f"adaptation_module_branch_hidden_dims: {AC_Args.adaptation_module_branch_hidden_dims}")
    print(f"estimator_mass_dim: {getattr(AC_Args, 'estimator_mass_dim', None)}")
    print("===========================================================\n")

    # ------------------------------------------------------------
    # Create env after loading params and applying overrides
    # ------------------------------------------------------------
    env = VelocityTrackingEasyEnv(
        sim_device=args.device,
        headless=args.headless,
        cfg=Cfg,
    )

    env = HistoryWrapper(env)
    base_env = unwrap_env(env)

    # ------------------------------------------------------------
    # Create ActorCritic after loading saved AC_Args
    # ------------------------------------------------------------
    actor_critic = ActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_obs_history,
        env.num_actions,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    missing_keys, unexpected_keys = actor_critic.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print("[load_state_dict] Missing keys:")
        for k in missing_keys:
            print("  ", k)

    if unexpected_keys:
        print("[load_state_dict] Unexpected keys:")
        for k in unexpected_keys:
            print("  ", k)

    actor_critic.eval()
    for p in actor_critic.parameters():
        p.requires_grad_(False)

    # Reset all envs. This also creates the first observation history.
    env.reset()
    obs_dict = get_obs_dict(env)

    obs = obs_dict["obs"].to(device)
    obs_history = obs_dict["obs_history"].to(device)

    update_viewer_camera(base_env, args)

    handoff_counter = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    ever_handoff_stable = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    first_handoff_step = torch.full((base_env.num_envs,), fill_value=-1, dtype=torch.long, device=base_env.device)

    csv_path = Path(args.log_csv).expanduser().resolve() if args.log_csv else None
    csv_file = None
    writer = None

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "reward_mean",
                "done_frac",
                "handoff_ready_frac",
                "handoff_stable_frac",
                "handoff_counter_mean",
                "handoff_counter_max",
                "upright",
                "stable_height",
                "low_velocity",
                "low_ang_vel",
                "stable_contacts",
                "recovered",
                "stable_recovery",
                "rollout_recovered",
                "timeout_frac",
                "bad_state_frac",
            ],
        )

        writer.writeheader()

    t0 = time.time()

    with torch.inference_mode():
        for step in range(int(args.num_steps)):
            actions = actor_deterministic_action(actor_critic, obs, obs_history)
            ret = env.step(actions)
            obs_dict, rewards, dones, infos = unpack_step(ret, env)
            obs = obs_dict["obs"].to(device)
            obs_history = obs_dict["obs_history"].to(device)

            if step % max(1, int(args.camera_update_every)) == 0:
                update_viewer_camera(base_env, args)

            handoff_ready = compute_handoff_ready(base_env).to(base_env.device)

            handoff_counter[handoff_ready] += 1
            handoff_counter[~handoff_ready] = 0

            # Reset counter for envs that reset.
            handoff_counter[dones.bool()] = 0

            handoff_stable = handoff_counter >= int(args.handoff_success_steps)

            newly_stable = handoff_stable & (~ever_handoff_stable)
            first_handoff_step[newly_stable] = step
            ever_handoff_stable |= handoff_stable

            if (step % int(args.print_every) == 0) or (step == int(args.num_steps) - 1):
                reward_mean = rewards.float().mean().item()
                done_frac = dones.float().mean().item()
                handoff_ready_frac = handoff_ready.float().mean().item()
                handoff_stable_frac = handoff_stable.float().mean().item()
                counter_mean = handoff_counter.float().mean().item()
                counter_max = handoff_counter.max().item()

                print(
                    f"step={step:04d} | "
                    f"rew={reward_mean:8.3f} | "
                    f"done={done_frac:5.2f} | "
                    f"handoff_ready={handoff_ready_frac:5.2f} | "
                    f"handoff_stable={handoff_stable_frac:5.2f} | "
                    f"counter_mean={counter_mean:6.2f} | "
                    f"counter_max={counter_max:3d} | "
                    f"upright={get_debug_value(infos, 'upright'):5.2f} | "
                    f"height={get_debug_value(infos, 'stable_height'):5.2f} | "
                    f"contacts={get_debug_value(infos, 'stable_contacts'):5.2f}"
                )

            if writer is not None:
                writer.writerow(
                    {
                        "step": step,
                        "reward_mean": rewards.float().mean().item(),
                        "done_frac": dones.float().mean().item(),
                        "handoff_ready_frac": handoff_ready.float().mean().item(),
                        "handoff_stable_frac": handoff_stable.float().mean().item(),
                        "handoff_counter_mean": handoff_counter.float().mean().item(),
                        "handoff_counter_max": handoff_counter.max().item(),
                        "upright": get_debug_value(infos, "upright"),
                        "stable_height": get_debug_value(infos, "stable_height"),
                        "low_velocity": get_debug_value(infos, "low_velocity"),
                        "low_ang_vel": get_debug_value(infos, "low_ang_vel"),
                        "stable_contacts": get_debug_value(infos, "stable_contacts"),
                        "recovered": get_debug_value(infos, "recovered"),
                        "stable_recovery": get_debug_value(infos, "stable_recovery"),
                        "rollout_recovered": get_debug_value(infos, "rollout_recovered"),
                        "timeout_frac": get_debug_value(infos, "timeout_frac"),
                        "bad_state_frac": get_debug_value(infos, "bad_state_frac"),
                    }
                )

    elapsed = time.time() - t0

    success_rate = ever_handoff_stable.float().mean().item()

    valid_times = first_handoff_step[first_handoff_step >= 0]
    mean_first_step = (
        valid_times.float().mean().item()
        if valid_times.numel() > 0
        else float("nan")
    )

    print("\n================ Evaluation summary ================")
    print(f"ever_handoff_stable_rate: {success_rate:.3f}")
    print(f"mean_first_handoff_step: {mean_first_step:.1f}")
    print(f"elapsed_sec: {elapsed:.1f}")

    if csv_path is not None:
        print(f"csv_log: {csv_path}")

    print("====================================================\n")

    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
