import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from aliengo_gym_learn.ppo_cse import ActorCritic
from aliengo_gym_learn.ppo_cse import RolloutStorage
from aliengo_gym_learn.ppo_cse import caches


# class PPO_Args(PrefixProto):
#     # algorithm
#     value_loss_coef = 1.0
#     use_clipped_value_loss = True
#     # clip_param = 0.2
#     # # entropy_coef = 0.005 #0.01
#     # num_learning_epochs = 5
#     num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
#     # learning_rate = 1e-3
#     # adaptation_module_learning_rate = 1e-3

#     # FINETUNE
#     entropy_coef = 0.0002 #0.0005
#     learning_rate = 2e-4 #5e-4 #3e-4
#     adaptation_module_learning_rate = 1e-4 #3e-6 #3e-4

#     num_adaptation_module_substeps = 1
#     schedule = 'adaptive'  # could be adaptive, fixed
#     gamma = 0.99
#     lam = 0.95
#     # desired_kl = 0.01
#     # max_grad_norm = 1.

#     clip_param = 0.10
#     num_learning_epochs = 2
#     # desired_kl = 0.005       # if your PPO uses adaptive LR / KL stop
#     max_grad_norm = 0.5

#     # use_mass_regression_loss = True

#     # If adaptation overfits masses and ignores other privileged info, set 0.5
#     # In case of poor mass prediction, set it to 2.0
#     # mass_regression_coef = 1.0

#     selective_adaptation_module_loss = True
#     # freeze_adaptation_module = False

#     # conservative settings
#     entropy_coef = 0.0
#     learning_rate = 1e-5
#     adaptation_module_learning_rate = 0.0

#     freeze_adaptation_module = True
#     use_mass_regression_loss = False
#     mass_regression_coef = 0.0
#     desired_kl = 0.002
#

class PPO_Args(PrefixProto):
    value_loss_coef = 1.0
    use_clipped_value_loss = True

    clip_param = 0.2

    entropy_coef = 0.01 # Stage I
    # entropy_coef = 0.001 # Stage II

    num_learning_epochs = 5
    num_mini_batches = 4

    learning_rate = 3e-4
    adaptation_module_learning_rate = 3e-4
    num_adaptation_module_substeps = 1

    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    freeze_adaptation_module = False

    use_mass_regression_loss = True
    mass_regression_coef = 1.0

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)

        # Freeze adaptation module for recovery fine-tuning
        if getattr(PPO_Args, "freeze_adaptation_module", False):
            for p in self.actor_critic.adaptation_module.parameters():
                p.requires_grad_(False)

            print("[PPO] Adaptation module frozen.")

        self.storage = None  # initialized later

        print(
            f"[PPO INIT] learning_rate={PPO_Args.learning_rate}, "
            f"adaptation_lr={PPO_Args.adaptation_module_learning_rate}, "
            f"entropy_coef={PPO_Args.entropy_coef}"
        )

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
            lr=PPO_Args.learning_rate,
        )
        # self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
        #                                               lr=PPO_Args.adaptation_module_learning_rate)

        # Only create adaptation optimizer if adaptation is not frozen
        if getattr(PPO_Args, "freeze_adaptation_module", False):
            self.adaptation_module_optimizer = None
        else:
            self.adaptation_module_optimizer = optim.Adam(
                self.actor_critic.adaptation_module.parameters(),
                lr=PPO_Args.adaptation_module_learning_rate
            )

        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.decoder.parameters(),
                                                lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # print("rewards shape:", rewards.shape)
        # print("rewards mean:", rewards.mean().item())
        # print("rewards max:", rewards.max().item())

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_adaptation_module_loss = 0.0
        mean_decoder_loss = 0.0
        mean_decoder_loss_student = 0.0
        mean_adaptation_module_test_loss = 0.0
        mean_decoder_test_loss = 0.0
        mean_decoder_test_loss_student = 0.0
        mean_mass_regression_loss = 0.0
        mean_mass_regression_test_loss = 0.0

        train_adaptation = (
            not getattr(PPO_Args, "freeze_adaptation_module", False)
            and PPO_Args.num_adaptation_module_substeps > 0
            and self.adaptation_module_optimizer is not None
        )

        generator = self.storage.mini_batch_generator(
            PPO_Args.num_mini_batches,
            PPO_Args.num_learning_epochs,
        )

        for (
            obs_batch,
            critic_obs_batch,
            privileged_obs_batch,
            obs_history_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            masks_batch,
        ) in generator:

            # ------------------------------------------------------------
            # PPO actor-critic update
            # ------------------------------------------------------------
            self.actor_critic.act(
                obs_batch,
                obs_history_batch,
                masks=masks_batch,
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(
                obs_batch,
                privileged_obs_batch,
                masks=masks_batch,
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ------------------------------------------------------------
            # Adaptive learning-rate schedule using KL
            # ------------------------------------------------------------
            if PPO_Args.desired_kl is not None and PPO_Args.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )

                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(
                            1.0e-5,
                            self.learning_rate / 1.5,
                        )
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            1.0e-2,
                            self.learning_rate * 1.5,
                        )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ------------------------------------------------------------
            # PPO losses
            # ------------------------------------------------------------
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - PPO_Args.clip_param,
                1.0 + PPO_Args.clip_param,
            )

            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(
                    -PPO_Args.clip_param,
                    PPO_Args.clip_param,
                )

                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + PPO_Args.value_loss_coef * value_loss
                - PPO_Args.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
                PPO_Args.max_grad_norm,
            )

            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # ------------------------------------------------------------
            # Optional adaptation-module update
            #
            # For recovery fine-tuning with freeze_adaptation_module=True,
            # this whole block is skipped. This is required because
            # self.adaptation_module_optimizer is None when frozen.
            # ------------------------------------------------------------
            if train_adaptation:
                data_size = privileged_obs_batch.shape[0]
                num_train = int(data_size // 5 * 4)

                for _ in range(PPO_Args.num_adaptation_module_substeps):
                    adaptation_pred = self.actor_critic.adaptation_module(
                        obs_history_batch
                    )

                    # with torch.no_grad():
                        # adaptation_target = privileged_obs_batch

                    # Only the first 26 privileged values are adaptation targets.
                    # The remaining values are the raw height map for the critic encoder.
                    with torch.no_grad():
                        adaptation_target = privileged_obs_batch[
                            :, :self.actor_critic.num_adaptation_obs
                        ]

                    if adaptation_pred.shape != adaptation_target.shape:
                        raise RuntimeError(
                            "Adaptation prediction/target shape mismatch: "
                            f"prediction={tuple(adaptation_pred.shape)}, "
                            f"target={tuple(adaptation_target.shape)}, "
                            f"num_adaptation_obs={self.actor_critic.num_adaptation_obs}, "
                            f"full_privileged_dim={privileged_obs_batch.shape[-1]}"
                        )

                    mass_dim = self.actor_critic.estimator_mass_dim

                    if mass_dim > 0:
                        mass_pred = adaptation_pred[:, :mass_dim]
                        latent_pred = adaptation_pred[:, mass_dim:]

                        mass_target = adaptation_target[:, :mass_dim]
                        latent_target = adaptation_target[:, mass_dim:]
                    else:
                        mass_pred = None
                        latent_pred = adaptation_pred

                        mass_target = None
                        latent_target = adaptation_target

                    if latent_pred.shape[1] > 0:
                        latent_loss = F.mse_loss(
                            latent_pred[:num_train],
                            latent_target[:num_train],
                        )
                        latent_test_loss = F.mse_loss(
                            latent_pred[num_train:],
                            latent_target[num_train:],
                        )
                    else:
                        latent_loss = torch.zeros((), device=self.device)
                        latent_test_loss = torch.zeros((), device=self.device)

                    if PPO_Args.use_mass_regression_loss and mass_dim > 0:
                        mass_reg_loss = F.mse_loss(
                            mass_pred[:num_train],
                            mass_target[:num_train],
                        )
                        mass_reg_test_loss = F.mse_loss(
                            mass_pred[num_train:],
                            mass_target[num_train:],
                        )
                    else:
                        mass_reg_loss = torch.zeros((), device=self.device)
                        mass_reg_test_loss = torch.zeros((), device=self.device)

                    adaptation_loss = (
                        latent_loss
                        + PPO_Args.mass_regression_coef * mass_reg_loss
                    )

                    adaptation_test_loss = (
                        latent_test_loss
                        + PPO_Args.mass_regression_coef * mass_reg_test_loss
                    )

                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()

                    mean_adaptation_module_loss += adaptation_loss.item()
                    mean_adaptation_module_test_loss += adaptation_test_loss.item()

                    if PPO_Args.use_mass_regression_loss and mass_dim > 0:
                        mean_mass_regression_loss += mass_reg_loss.item()
                        mean_mass_regression_test_loss += mass_reg_test_loss.item()

        # ------------------------------------------------------------
        # Normalize logging values
        # ------------------------------------------------------------
        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        if train_adaptation:
            adapt_updates = num_updates * PPO_Args.num_adaptation_module_substeps
            mean_adaptation_module_loss /= adapt_updates
            mean_mass_regression_loss /= adapt_updates
            mean_adaptation_module_test_loss /= adapt_updates
        else:
            mean_adaptation_module_loss = 0.0
            mean_mass_regression_loss = 0.0
            mean_adaptation_module_test_loss = 0.0

        # Decoder is not updated in this PPO file's active code path.
        mean_decoder_loss = 0.0
        mean_decoder_loss_student = 0.0
        mean_decoder_test_loss = 0.0
        mean_decoder_test_loss_student = 0.0

        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_adaptation_module_loss,
            mean_mass_regression_loss,
            mean_decoder_loss,
            mean_decoder_loss_student,
            mean_adaptation_module_test_loss,
            mean_decoder_test_loss,
            mean_decoder_test_loss_student,
        )
