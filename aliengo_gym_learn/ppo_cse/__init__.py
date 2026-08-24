import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage
from tracebot.send_telegram import safe_send_gif
# from send_telegram import safe_send_gif


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from aliengo_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


# class RunnerArgs(PrefixProto, cli=False):
#     # runner
#     algorithm_class_name = 'RMA'
#     num_steps_per_env = 24
#     max_iterations = 1500

#     # logging
#     save_interval = 400
#     save_video_interval = 300
#     log_freq = 10

#     # recovery policy resume
#     resume = True
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-05-18/train_fall_recovery/215844.509038" #None
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-01/train_fall_recovery/073922.812308"
#     # terminal_action_clip = 0.30
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-03/train_fall_recovery/101441.711985"
#     # terminal_action_clip = 0.50
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-03/train_fall_recovery/104741.561249"
#     # terminal_action_clip = 0.80
#     #resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-03/train_fall_recovery/113046.764231"
#     # # terminal_action_clip = 0.90
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-03/train_fall_recovery/141847.429995"

#     terminal_stance_reset_prob = 1.0
#     # resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-04/train_fall_recovery/094324.831941"
#     resume_path = "/home/ros20_doc/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-06-04/train_fall_recovery/123700.778927"

#     checkpoint = "last"          # "last" or iteration number, e.g. 8717
#     resume_optimizer = False     # keep False for recovery fine-tuning
#     resume_iteration = 39200 #35600 #34400 #31600 #29600 #28800 #28400 #0         # set manually if you want logs to continue from old iter

# STAGE I
# class RunnerArgs(PrefixProto, cli=False):
#     algorithm_class_name = "RMA"
#     num_steps_per_env = 24
#     max_iterations = 1500

#     save_interval = 400
#     save_video_interval = 300
#     log_freq = 10

#     resume = False
#     resume_path = None
#     checkpoint = "last"
#     resume_optimizer = False
#     resume_iteration = 0

# STAGE II
class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24
    max_iterations = 1500

    # logging
    save_interval = 400
    save_video_interval = 300
    log_freq = 10

    # recovery policy resume
    resume = True
    # resume_path = "/home/everythingelse1772/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-08-14/train_fall_recovery/080722.513527"
    # resume_path = "/home/everythingelse1772/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-08-15/train_fall_recovery/073618.354140"
    # resume_path = "/home/everythingelse1772/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-08-18/train_fall_recovery/063240.180386"
    # resume_path = "/home/everythingelse1772/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-08-19/train_fall_recovery/060729.262985"
    resume_path = "/home/everythingelse1772/Projects/quadruped-fall-recovery-rl/runs/gait-conditioned-agility/2026-08-23/train_fall_recovery/070942.908284"


    checkpoint = 23200 #19200 #18000 #14000 #6400            # "last" or iteration number, e.g. 8717
    resume_optimizer = False     # keep False for recovery fine-tuning
    resume_iteration = 23201 #19201 #18001 #14001 #6401      # set manually if you want logs to continue from old iter

class Runner:

    # def __init__(self, env, device='cpu'):
    #     from .ppo import PPO

    #     self.device = device
    #     self.env = env

    #     if self.env.cfg.env.priv_observe_heightmap:
    #         num_height_obs = (
    #             len(self.env.cfg.terrain.measured_points_x)
    #             * len(self.env.cfg.terrain.measured_points_y)
    #         )
    #     else:
    #         num_height_obs = 0

    #     actor_critic = ActorCritic(
    #         self.env.num_obs,
    #         self.env.num_privileged_obs,
    #         self.env.num_obs_history,
    #         self.env.num_actions,
    #         num_height_obs=num_height_obs,
    #     ).to(self.device)

    #     # Recovery policy resume
    #     if RunnerArgs.resume:
    #         from pathlib import Path
    #         import torch

    #         resume_dir = Path(RunnerArgs.resume_path).expanduser().resolve()

    #         if RunnerArgs.checkpoint == "last" or RunnerArgs.checkpoint == -1:
    #             checkpoint_path = resume_dir / "checkpoints" / "ac_weights_last.pt"
    #         else:
    #             checkpoint_path = resume_dir / "checkpoints" / f"ac_weights_{int(RunnerArgs.checkpoint):06d}.pt"

    #         assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    #         print(f"[Recovery Resume] Loading actor-critic from: {checkpoint_path}")

    #         state_dict = torch.load(checkpoint_path, map_location=self.device)
    #         missing_keys, unexpected_keys = actor_critic.load_state_dict(state_dict, strict=False)

    #         if len(missing_keys) > 0:
    #             print("[Recovery Resume] Missing keys:")
    #             for k in missing_keys:
    #                 print("  ", k)

    #         if len(unexpected_keys) > 0:
    #             print("[Recovery Resume] Unexpected keys:")
    #             for k in unexpected_keys:
    #                 print("  ", k)

    #         print("[Recovery Resume] Actor-critic weights loaded.")

    #         # # Reduce exploration for final-stand refinement.
    #         # # This prevents the old recovery policy from injecting
    #         # # large stochastic actions near the standing pose.
    #         # with torch.no_grad():
    #         #     actor_critic.std.data[:] = torch.clamp(
    #         #         actor_critic.std.data,
    #         #         max=0.15,
    #         #     )
    #         #
    #         # with torch.no_grad():
    #         #     actor_critic.std.clamp_(min=0.02, max=0.05)

    #         # print(
    #         #     "[Recovery Resume] Clamped action std:",
    #         #     "mean =", actor_critic.std.mean().item(),
    #         #     "max =", actor_critic.std.max().item(),
    #         # )
    #         #

    #     # with torch.no_grad():
    #     #     print(
    #     #         "Action std:",
    #     #         actor_critic.std.mean().item(),
    #     #         actor_critic.std.min().item(),
    #     #         actor_critic.std.max().item(),
    #     #     )

    #     self.alg = PPO(actor_critic, device=self.device)
    #     self.num_steps_per_env = RunnerArgs.num_steps_per_env

    #     # init storage and model
    #     self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
    #                           [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

    #     # self.tot_timesteps = 0
    #     # self.tot_time = 0
    #     # self.current_learning_iteration = 0
    #     # self.last_recording_it = 0

    #     if RunnerArgs.resume:
    #         self.current_learning_iteration = RunnerArgs.resume_iteration
    #         self.tot_timesteps = (
    #             RunnerArgs.resume_iteration
    #             * self.num_steps_per_env
    #             * self.env.num_envs
    #         )
    #         self.last_recording_it = RunnerArgs.resume_iteration
    #     else:
    #         self.current_learning_iteration = 0
    #         self.tot_timesteps = 0
    #         self.last_recording_it = 0

    #     self.tot_time = 0

    #     self.env.reset()

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        # Height-map observation dimension
        if self.env.cfg.env.priv_observe_heightmap:
            num_height_obs = (
                len(self.env.cfg.terrain.measured_points_x)
                * len(self.env.cfg.terrain.measured_points_y)
            )
        else:
            num_height_obs = 0

        # Construct NEW actor-critic
        #
        # New observation configuration:
        #
        #   observe_vel = False
        #   observe_only_ang_vel = True
        #   observe_only_lin_vel = True
        #
        # old obs = 42D:
        #   [ang_vel(3), gravity(3), dof_pos(12), dof_vel(12), actions(12)]
        #
        # new obs = 45D:
        #   [lin_vel(3), ang_vel(3), gravity(3), dof_pos(12), dof_vel(12), actions(12)]
        actor_critic = ActorCritic(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_obs_history,
            self.env.num_actions,
            num_height_obs=num_height_obs,
        ).to(self.device)

        # Recovery policy resume
        if RunnerArgs.resume:
            from pathlib import Path

            resume_dir = Path(RunnerArgs.resume_path).expanduser().resolve()

            if (RunnerArgs.checkpoint == "last" or RunnerArgs.checkpoint == -1):
                checkpoint_path = resume_dir / "checkpoints" / "ac_weights_last.pt"
            else:
                checkpoint_path = resume_dir / "checkpoints" / f"ac_weights_{int(RunnerArgs.checkpoint):06d}.pt"

            assert checkpoint_path.exists(), (f"Checkpoint not found: {checkpoint_path}")
            print(f"[Recovery Resume] Loading actor-critic from: " f"{checkpoint_path}")

            # Load OLD 42D checkpoint
            old_state_dict = torch.load(checkpoint_path, map_location=self.device)

            # Current NEW 45D state dict.
            new_state_dict = actor_critic.state_dict()

            # Observation migration settings
            OLD_OBS_DIM = 42
            NEW_OBS_DIM = 45
            NEW_LIN_VEL_DIM = 3

            assert self.env.num_obs == NEW_OBS_DIM, (
                "Expected new observation dimension to be 45, "
                f"but env.num_obs={self.env.num_obs}. "
                "Check num_observations / observation flags."
            )

            # These layers changed because their inputs depend directly
            # on observation dimension.
            special_keys = {
                "actor_body.0.weight",
                "critic_body.0.weight",
                "adaptation_module.trunk.0.weight",
            }

            print("[Recovery Resume] Migrating observation space " f"{OLD_OBS_DIM}D -> {NEW_OBS_DIM}D")

            # 1. Copy every parameter whose shape did NOT change
            for key, old_tensor in old_state_dict.items():

                if key in special_keys:
                    continue

                if key not in new_state_dict:
                    print("[Recovery Resume] Skipping unexpected " f"checkpoint key: {key}")
                    continue

                if old_tensor.shape != new_state_dict[key].shape:
                    raise RuntimeError(
                        "\nUnexpected checkpoint shape mismatch.\n"
                        f"Key: {key}\n"
                        f"Old: {tuple(old_tensor.shape)}\n"
                        f"New: {tuple(new_state_dict[key].shape)}\n"
                        "\nOnly the actor first layer, critic first "
                        "layer, and adaptation first layer should "
                        "change during the 42D -> 45D migration."
                    )

                new_state_dict[key] = old_tensor

            # 2. Migrate ACTOR first layer
            #
            # OLD: [ old_obs(42) | predicted privileged context ]
            # NEW: [ lin_vel(3) | old_obs(42) | predicted privileged context ]
            #
            # New lin_vel weights start at ZERO.
            actor_key = "actor_body.0.weight"

            assert actor_key in old_state_dict, f"{actor_key} missing from old checkpoint"

            old_actor_w = old_state_dict[actor_key]
            new_actor_w = new_state_dict[actor_key].clone()

            assert (
                new_actor_w.shape[0] == old_actor_w.shape[0]
            ), (
                "Actor hidden dimension changed. "
                f"old={tuple(old_actor_w.shape)}, "
                f"new={tuple(new_actor_w.shape)}"
            )

            assert (
                new_actor_w.shape[1]
                == old_actor_w.shape[1] + NEW_LIN_VEL_DIM
            ), (
                "Actor input did not increase by exactly 3 dimensions.\n"
                f"old={tuple(old_actor_w.shape)}\n"
                f"new={tuple(new_actor_w.shape)}"
            )

            # Start from zero for safety.
            new_actor_w.zero_()

            # New lin_vel columns:
            # new_actor_w[:, 0:3]
            # deliberately remain zero.
            # Copy OLD observation weights:
            # old [0:42] -> new [3:45]
            new_actor_w[:, NEW_LIN_VEL_DIM:NEW_OBS_DIM] = old_actor_w[:, :OLD_OBS_DIM]

            # Copy actor privileged/adaptation-context weights unchanged.
            new_actor_w[:, NEW_OBS_DIM:] = old_actor_w[:, OLD_OBS_DIM:]

            new_state_dict[actor_key] = new_actor_w

            print(
                "[Recovery Resume] Migrated actor input layer:"
                f" {tuple(old_actor_w.shape)}"
                f" -> {tuple(new_actor_w.shape)}"
            )

            # 3. Migrate CRITIC first layer
            # Current critic input:
            # [obs, dynamics_privileged, height_map_latent]
            # Only obs grows by 3 dimensions.
            critic_key = "critic_body.0.weight"

            assert critic_key in old_state_dict, f"{critic_key} missing from old checkpoint"

            old_critic_w = old_state_dict[critic_key]
            new_critic_w = new_state_dict[critic_key].clone()

            assert (
                new_critic_w.shape[0]
                == old_critic_w.shape[0]
            ), (
                "Critic hidden dimension changed. "
                f"old={tuple(old_critic_w.shape)}, "
                f"new={tuple(new_critic_w.shape)}"
            )

            assert (
                new_critic_w.shape[1]
                == old_critic_w.shape[1] + NEW_LIN_VEL_DIM
            ), (
                "Critic input did not increase by exactly 3 dimensions.\n"
                f"old={tuple(old_critic_w.shape)}\n"
                f"new={tuple(new_critic_w.shape)}"
            )

            new_critic_w.zero_()

            # New linear-velocity columns remain zero.
            # Copy OLD observation weights.
            new_critic_w[:, NEW_LIN_VEL_DIM:NEW_OBS_DIM] = old_critic_w[:, :OLD_OBS_DIM]

            # Copy privileged + height-map latent weights unchanged.
            new_critic_w[:, NEW_OBS_DIM:] = old_critic_w[:, OLD_OBS_DIM:]

            new_state_dict[critic_key] = new_critic_w

            print(
                "[Recovery Resume] Migrated critic input layer:"
                f" {tuple(old_critic_w.shape)}"
                f" -> {tuple(new_critic_w.shape)}"
            )

            # 4. Migrate ADAPTATION-MODULE history input
            #
            # OLD history: [obs42(t-H), obs42(t-H+1), ..., obs42(t)]
            # NEW history: [obs45(t-H), obs45(t-H+1), ..., obs45(t)]
            # Since lin_vel is at the BEGINNING of every new frame,
            # we insert 3 zero-weight inputs PER HISTORY FRAME.
            adaptation_key = "adaptation_module.trunk.0.weight"

            assert adaptation_key in old_state_dict, f"{adaptation_key} missing from old checkpoint"

            old_adapt_w = old_state_dict[adaptation_key]
            new_adapt_w = new_state_dict[adaptation_key].clone()

            assert (
                old_adapt_w.shape[1] % OLD_OBS_DIM == 0
            ), (
                "Old adaptation input dimension is not divisible "
                f"by OLD_OBS_DIM={OLD_OBS_DIM}.\n"
                f"Shape: {tuple(old_adapt_w.shape)}"
            )

            history_len = old_adapt_w.shape[1] // OLD_OBS_DIM
            expected_old_history_dim =  history_len * OLD_OBS_DIM
            expected_new_history_dim = history_len * NEW_OBS_DIM

            assert old_adapt_w.shape[1] == expected_old_history_dim

            assert (
                new_adapt_w.shape[1]
                == expected_new_history_dim
            ), (
                "New observation-history dimension is inconsistent "
                "with the old checkpoint.\n"
                f"history_len={history_len}\n"
                f"expected={expected_new_history_dim}\n"
                f"actual={new_adapt_w.shape[1]}"
            )

            assert (
                new_adapt_w.shape[1]
                == self.env.num_obs_history
            ), (
                "ActorCritic adaptation input and environment "
                "observation-history dimensions disagree.\n"
                f"network={new_adapt_w.shape[1]}\n"
                f"env={self.env.num_obs_history}"
            )

            # New linear velocity history connections start at zero.
            new_adapt_w.zero_()

            for history_idx in range(history_len):

                old_start = history_idx * OLD_OBS_DIM
                old_end = old_start + OLD_OBS_DIM
                new_start = history_idx * NEW_OBS_DIM

                # Leave:
                # new_start : new_start + 3
                # at zero because these are the new lin_vel inputs.
                # Copy old 42D observation immediately after.
                new_adapt_w[:, new_start + NEW_LIN_VEL_DIM: new_start + NEW_LIN_VEL_DIM + OLD_OBS_DIM] = old_adapt_w[:, old_start:old_end]

            new_state_dict[adaptation_key] = new_adapt_w

            print(
                "[Recovery Resume] Migrated adaptation input layer:"
                f" {tuple(old_adapt_w.shape)}"
                f" -> {tuple(new_adapt_w.shape)}"
            )

            print("[Recovery Resume] Observation history length:", history_len)

            # 5. Strictly load migrated checkpoint
            # At this point every tensor should exactly match the new
            # architecture.
            actor_critic.load_state_dict(new_state_dict, strict=True)

            print("[Recovery Resume] Actor-critic weights migrated successfully.")
            print("[Recovery Resume] New linear-velocity pathways initialized to zero.")

            # Reset exploration noise for the new fine-tuning stage.
            #
            # The previous checkpoint learned an excessively large std
            # (~8-15). Preserve the learned policy mean, but restart
            # exploration from the original nominal std.
            with torch.no_grad():
                loaded_std = actor_critic.std.detach().clone()

                actor_critic.std.fill_(1.0)

            print(
                "[Recovery Resume] Reset action std:"
                f" old mean={loaded_std.mean().item():.4f},"
                f" old min={loaded_std.min().item():.4f},"
                f" old max={loaded_std.max().item():.4f}"
            )

            print(
                "[Recovery Resume] New action std:"
                f" mean={actor_critic.std.mean().item():.4f},"
                f" min={actor_critic.std.min().item():.4f},"
                f" max={actor_critic.std.max().item():.4f}"
            )

            # Sanity checks
            with torch.no_grad():
                actor_lin_vel_weight_norm = torch.norm(actor_critic.actor_body[0].weight[:, :NEW_LIN_VEL_DIM]).item()
                critic_lin_vel_weight_norm = torch.norm(actor_critic.critic_body[0].weight[:, :NEW_LIN_VEL_DIM]).item()

            print("[Recovery Resume] Actor new lin_vel " f"weight norm: {actor_lin_vel_weight_norm:.8f}")
            print("[Recovery Resume] Critic new lin_vel " f"weight norm: {critic_lin_vel_weight_norm:.8f}")

            assert actor_lin_vel_weight_norm == 0.0
            assert critic_lin_vel_weight_norm == 0.0

            print("[Recovery Resume] Observation migration checks passed.")


            # Optional action-std inspection
            with torch.no_grad():
                print(
                    "[Recovery Resume] Action std:",
                    "mean =",
                    actor_critic.std.mean().item(),
                    "min =",
                    actor_critic.std.min().item(),
                    "max =",
                    actor_critic.std.max().item(),
                )

        # PPO
        # Construct PPO AFTER checkpoint migration.
        # This gives the migrated model a fresh optimizer state.
        self.alg = PPO(actor_critic, device=self.device)

        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # Rollout storage
        self.alg.init_storage(
            self.env.num_train_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_obs_history],
            [self.env.num_actions],
        )

        # Iteration / timestep bookkeeping
        if RunnerArgs.resume:
            self.current_learning_iteration = RunnerArgs.resume_iteration
            self.tot_timesteps = RunnerArgs.resume_iteration * self.num_steps_per_env * self.env.num_envs
            self.last_recording_it = RunnerArgs.resume_iteration

        else:
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.last_recording_it = 0

        self.tot_time = 0

        # Reset environment
        self.env.reset()


    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        # actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                        #                                                  privileged_obs[num_train_envs:])
                        actions_eval = self.alg.actor_critic.act_teacher(obs[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])

                    else:
                        # actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                        actions_eval = self.alg.actor_critic.act_student(obs[num_train_envs:],
                                                                        obs_history[num_train_envs:])
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    # if 'curriculum' in infos:

                    cur_reward_sum += rewards
                    cur_episode_length += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)

                    new_ids_train = new_ids[new_ids < num_train_envs]
                    rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                    cur_reward_sum[new_ids_train] = 0
                    cur_episode_length[new_ids_train] = 0

                    new_ids_eval = new_ids[new_ids >= num_train_envs]
                    rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                    lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                    cur_reward_sum[new_ids_eval] = 0
                    cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])
                self.alg.compute_returns(obs[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                         path=f"curriculum/distribution.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_mass_mse, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_mass_loss = mean_mass_mse,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            # if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
            #     # if it % Config.log_freq == 0:
            #     logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
            #     logger.job_running()
            #
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):

                reward_names = [
                    k for k in self.env.episode_sums.keys()
                    if k not in ["total", "recovery_success"]
                ]

                abs_total = 0.0

                for name in reward_names:
                    abs_total += torch.mean(
                        torch.abs(self.env.episode_sums[name])
                    )

                print("\n===== Reward Contribution =====")

                reward_contrib_metrics = {}

                for name in reward_names:

                    mean_abs = torch.mean(
                        torch.abs(self.env.episode_sums[name])
                    )

                    contrib = mean_abs / (abs_total + 1e-8)

                    contrib_percent = 100 * contrib.item()

                    print(
                        f"{name:<25}: "
                        f"{contrib_percent:6.2f}%"
                    )

                    # store for dashboard
                    reward_contrib_metrics[f"reward_contrib/{name}"] = contrib_percent

                # =========================================================
                # Recovery debug statistics
                # =========================================================

                recovery_metrics = {}

                if hasattr(self.env, "extras") and "recovery_debug" in self.env.extras:

                    dbg = self.env.extras["recovery_debug"]

                    print("\n===== Recovery Debug =====")

                    for k, v in dbg.items():

                        if torch.is_tensor(v):
                            v = v.item()

                        print(f"{k:<25}: {v:8.4f}")

                        # store for dashboard
                        recovery_metrics[f"recovery/{k}"] = v


                # =========================================================
                # Recovery tracking statistics
                # =========================================================

                recovery_tracking_metrics = {}

                if hasattr(self.env, "extras") and "recovery_tracking" in self.env.extras:

                    tracking = self.env.extras["recovery_tracking"]

                    print("\n===== Recovery Tracking =====")

                    for k, v in tracking.items():

                        if torch.is_tensor(v):
                            v = v.item()

                        print(f"{k:<30}: {v:8.4f}")

                        # store for dashboard
                        recovery_tracking_metrics[f"recovery_tracking/{k}"] = v


                # =========================================================
                # Store all metrics
                # =========================================================

                logger.store_metrics(
                    **reward_contrib_metrics,
                    **recovery_metrics,
                    **recovery_tracking_metrics
                )

                logger.log_metrics_summary(
                    key_values={
                        "timesteps": self.tot_timesteps,
                        "iterations": it
                    }
                )

                logger.job_running()

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            # self.current_learning_iteration += num_learning_iterations
            self.current_learning_iteration = it + 1

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)


    def log_video(self, it):
        # START RECORDING
        if (
            not self.env.record_now and
            it - self.last_recording_it >= RunnerArgs.save_video_interval
        ):
            print(f"[VIDEO] Start recording at iter {it}")
            self.env.start_recording()

            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()

            self.last_recording_it = it

        # print(f"Num vid frames {len(self.env.video_frames)}")
        # STOP + SAVE RECORDING
        if (
            self.env.record_now
            and len(self.env.video_frames) >= self.env.max_video_frames
        ):
            env_id = int(self.env.record_env_id)

            terrain_column = int(
                self.env.terrain_types[env_id].item()
            )
            terrain_level = int(
                self.env.terrain_levels[env_id].item()
            )
            orientation_bin = int(
                self.env.episode_orientation_bin[env_id].item()
            )
            sampler_group = int(
                self.env.episode_sampler_group[env_id].item()
            )

            tag = (
                f"iter_{it}"
                f"_column_{terrain_column}"
                f"_level_{terrain_level}"
                f"_bin_{orientation_bin}"
                f"_group_{sampler_group}"
            )

            print(f"[VIDEO] Saving {tag}")

            try:
                self.env.stop_recording(
                    tag=tag,
                    telegram_fn=safe_send_gif,
                )
            except Exception as e:
                print(f"[VIDEO ERROR] {e}")
                self.env.stop_recording(
                    tag=tag,
                    telegram_fn=None,
                )


    # def log_video(self, it):
    #     if it - self.last_recording_it >= RunnerArgs.save_video_interval:
    #         self.env.start_recording()
    #         if self.env.num_eval_envs > 0:
    #             self.env.start_recording_eval()
    #         print("START RECORDING...")
    #         self.last_recording_it = it

    #     # frames = self.env.get_complete_frames()
    #     # if len(frames) > 0:
    #     #     self.env.pause_recording()
    #     #     print("LOGGING VIDEO")
    #     #     logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
    #     #
    #     # if self.env.record_now and len(self.env.video_frames) > 0:
    #     # ---- Stop + save when enough frames collected ----
    #     if self.env.record_now and len(self.env.video_frames) >= self.env.max_video_frames:
    #         print("SAVING VIDEO...r)")

    #         self.env.stop_recording(
    #             tag=f"iter_{it}",
    #             telegram_fn=send_gif   # import this at top
    #         )

    #         # self.env.stop_recording(
    #         #     tag=f"iter_{it}",
    #         #     telegram_fn=None
    #         # )

    #     # if self.env.num_eval_envs > 0:
    #     #     frames = self.env.get_complete_frames_eval()
    #     #     if len(frames) > 0:
    #     #         self.env.pause_recording_eval()
    #     #         print("LOGGING EVAL VIDEO")
    #     #         logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
