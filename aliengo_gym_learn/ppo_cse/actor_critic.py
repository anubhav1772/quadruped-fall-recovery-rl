import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]
    estimator_mass_dim = 4 # (base/trunk, hip, thigh, calf)

    use_decoder = False


class EstimatorEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, mass_dim, activation):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        self.trunk = nn.Sequential(*layers)

        self.output_dim = output_dim
        self.mass_dim = min(max(mass_dim, 0), output_dim)
        self.latent_dim = output_dim - self.mass_dim

        if self.mass_dim > 0:
            self.mass_head = nn.Linear(hidden_dims[-1], self.mass_dim)
        else:
            self.mass_head = None
        self.latent_head = nn.Linear(hidden_dims[-1], self.latent_dim)

    def forward_heads(self, observation_history):
        features = self.trunk(observation_history)
        mass_hat = self.mass_head(features) if self.mass_head is not None else None
        latent_hat = self.latent_head(features)
        return mass_hat, latent_hat

    def forward(self, observation_history):
        mass_hat, latent_hat = self.forward_heads(observation_history)
        if mass_hat is None:
            return latent_hat
        return torch.cat((mass_hat, latent_hat), dim=-1)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        # adaptation_module_layers = []
        # adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        # adaptation_module_layers.append(activation)
        # for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
        #     if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
        #         adaptation_module_layers.append(
        #             nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
        #     else:
        #         adaptation_module_layers.append(
        #             nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
        #                       AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
        #         adaptation_module_layers.append(activation)
        # self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        self.estimator_mass_dim = min(max(AC_Args.estimator_mass_dim, 0), self.num_privileged_obs)
        # Input of EE is obs_history
        self.adaptation_module = EstimatorEncoder(
            input_dim=self.num_obs_history,
            hidden_dims=AC_Args.adaptation_module_branch_hidden_dims,
            output_dim=self.num_privileged_obs,
            mass_dim=self.estimator_mass_dim,
            activation=activation,
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, observation_history):
        # Estimator Encoder output (concatenated predicted link masses and latent vector)
        ee_output = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((obs, ee_output), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs, observation_history, **kwargs):
        self.update_distribution(obs, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(
            ob["obs"],
            ob["privileged_obs"],
            policy_info=policy_info
        )

    def act_inference(self, ob, policy_info={}):
        return self.act_student(
            ob["obs"],
            ob["obs_history"],
            policy_info=policy_info
        )

    def act_student(self, obs, observation_history, policy_info={}):
        ee_output = self.adaptation_module(observation_history)
        actor_input = torch.cat((obs, ee_output), dim=-1)
        actions_mean = self.actor_body(actor_input)
        policy_info["ee_output"] = ee_output.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, obs, privileged_info, policy_info={}):
        actor_input = torch.cat((obs, privileged_info), dim=-1)
        actions_mean = self.actor_body(actor_input)
        policy_info["ee_output"] = privileged_info
        return actions_mean

    def evaluate(self, obs, privileged_observations, **kwargs):
        critic_input = torch.cat((obs, privileged_observations), dim=-1)
        value = self.critic_body(critic_input)
        return value

    def get_student_estimator_output(self, observation_history):
        return self.adaptation_module(observation_history)

    def get_estimator_outputs(self, observation_history):
        return self.adaptation_module.forward_heads(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
