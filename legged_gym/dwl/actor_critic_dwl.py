
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCriticDWL(nn.Module):
    is_recurrent = False
    def __init__(self,  num_short_obs,
                        num_proprio_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        in_channels = 20,
                        kernel_size=[3, 2],
                        filter_size=[16, 8],
                        stride_size=[1, 1],
                        lh_output_dim=32,
                        activation='elu',
                        init_noise_std=1.0,
                        min_noise_std=0.2,
                        max_noise_std=1.5,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDWL, self).__init__()

        activation = get_activation(activation)
        
        mlp_input_dim_a = num_short_obs + lh_output_dim
        mlp_input_dim_c = num_critic_obs

        self.num_short_obs = num_short_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
        #define long_history CNN
        long_history_layers = []
        self.in_channels = in_channels
        cnn_output_dim = num_proprio_obs
        for out_channels, kernel_size, stride_size in zip(filter_size, kernel_size, stride_size):
            long_history_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride_size))
            long_history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel_size + stride_size) // stride_size
            in_channels = out_channels
        cnn_output_dim *= out_channels
        long_history_layers.append(nn.Flatten())
        long_history_layers.append(nn.Linear(cnn_output_dim, 128))
        long_history_layers.append(nn.ELU())
        long_history_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*long_history_layers)
        print(f"long_history CNN: {self.long_history}")
        
        self.num_proprio_obs = num_proprio_obs

    
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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # Prevent exploration from growing so large that nearly every action
        # saturates the controller.  Clamp the parameter itself so the bound
        # also takes effect when resuming an older checkpoint with a large std.
        with torch.no_grad():
            self.std.clamp_(self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        short_history = observations[...,-self.num_short_obs:]
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, compressed_long_history),dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        short_history = observations[...,-self.num_short_obs:]
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, compressed_long_history),dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    


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
