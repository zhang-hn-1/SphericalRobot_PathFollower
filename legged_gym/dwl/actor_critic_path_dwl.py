"""Temporal-history mixture-of-experts actor for Rotunbot path following."""

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(name):
    activations = {
        "elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU,
        "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError("Unsupported activation: " + str(name))
    return activations[name]()


class ActorCriticPathDWL(nn.Module):
    """Use independent straight/left/right policy heads over shared history."""

    is_recurrent = False

    def __init__(
        self, num_short_obs, num_proprio_obs, num_critic_obs, num_actions,
        actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
        history_frames=20, kernel_size=(3, 2), filter_size=(32, 16),
        stride_size=(1, 1), lh_output_dim=32, path_obs_dim=43,
        activation="elu", init_noise_std=0.35, min_noise_std=0.08,
        max_noise_std=0.5, turn_gate_threshold=0.04,
        freeze_history_encoder=True, trainable_experts="all",
        zero_init_actor_output=False, **kwargs,
    ):
        super().__init__()
        if kwargs:
            print("ActorCriticPathDWL ignored policy arguments: " + str(sorted(kwargs.keys())))
        self.num_short_obs = int(num_short_obs)
        self.num_proprio_obs = int(num_proprio_obs)
        self.history_frames = int(history_frames)
        self.path_obs_dim = int(path_obs_dim)
        self.history_obs_dim = self.history_frames * self.num_proprio_obs
        self.turn_gate_threshold = float(turn_gate_threshold)

        history_layers = []
        cnn_output_dim = self.history_frames
        channels = self.num_proprio_obs
        for out_channels, kernel, stride in zip(filter_size, kernel_size, stride_size):
            history_layers.append(nn.Conv1d(channels, out_channels, kernel, stride))
            history_layers.append(nn.ReLU())
            cnn_output_dim = (cnn_output_dim - kernel + stride) // stride
            channels = out_channels
        cnn_output_dim *= channels
        history_layers.extend([
            nn.Flatten(), nn.Linear(cnn_output_dim, 128), nn.ELU(),
            nn.Linear(128, lh_output_dim),
        ])
        self.long_history = nn.Sequential(*history_layers)
        if freeze_history_encoder:
            for parameter in self.long_history.parameters():
                parameter.requires_grad_(False)

        actor_input_dim = self.num_short_obs + lh_output_dim + self.path_obs_dim
        self.actors = nn.ModuleDict({
            name: self._mlp(actor_input_dim, list(actor_hidden_dims), num_actions, activation, True)
            for name in ("straight", "left", "right")
        })
        if zero_init_actor_output:
            for expert in self.actors.values():
                output = next(layer for layer in reversed(expert) if isinstance(layer, nn.Linear))
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
        trainable_experts = str(trainable_experts)
        if trainable_experts != "all":
            requested = {item.strip() for item in trainable_experts.split(",") if item.strip()}
            unknown = requested.difference(self.actors.keys())
            if unknown:
                raise ValueError("Unknown trainable experts: " + str(sorted(unknown)))
            for name, expert in self.actors.items():
                if name not in requested:
                    for parameter in expert.parameters():
                        parameter.requires_grad_(False)
        self.trainable_experts = trainable_experts
        self.critic = self._mlp(num_critic_obs, list(critic_hidden_dims), 1, activation, False)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.distribution = None
        Normal.set_default_validate_args = False

        print("Path history CNN (frozen=%s):" % freeze_history_encoder, self.long_history)
        print("Path actor experts (trainable=%s):" % self.trainable_experts, self.actors)
        print("Path critic MLP:", self.critic)

    @staticmethod
    def _mlp(input_dim, hidden_dims, output_dim, activation, bound_output):
        dims = [input_dim] + hidden_dims
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(dims[-1], output_dim))
        if bound_output:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _actor_input(self, observations):
        expected = self.history_obs_dim + self.path_obs_dim
        if observations.shape[-1] != expected:
            raise RuntimeError(f"Expected {expected} actor observations, got {observations.shape[-1]}")
        history = observations[..., :self.history_obs_dim]
        path = observations[..., self.history_obs_dim:]
        short_history = history[..., -self.num_short_obs:]
        compressed = self.long_history(
            history.reshape(-1, self.history_frames, self.num_proprio_obs).transpose(1, 2)
        )
        return torch.cat((short_history, compressed, path), dim=-1)

    def _route_masks(self, observations):
        path = observations[..., self.history_obs_dim:]
        preview = path[..., :40].reshape(-1, 10, 4)
        if self.path_obs_dim >= 44:
            # V5 appends 2*kappa and routes experts from desired geometry,
            # so tracking errors cannot accidentally switch turn experts.
            turn_signal = 0.5 * path[..., -1]
        else:
            turn_signal = (
                0.30 * preview[:, :3, 1].mean(dim=1)
                + 0.70 * preview[:, :3, 3].mean(dim=1)
            )
        left = turn_signal > self.turn_gate_threshold
        right = turn_signal < -self.turn_gate_threshold
        straight = ~(left | right)
        return straight.float()[:, None], left.float()[:, None], right.float()[:, None]

    def _actor_mean(self, observations):
        actor_input = self._actor_input(observations)
        straight_mask, left_mask, right_mask = self._route_masks(observations)
        return (
            straight_mask * self.actors["straight"](actor_input)
            + left_mask * self.actors["left"](actor_input)
            + right_mask * self.actors["right"](actor_input)
        )

    def update_distribution(self, observations):
        mean = self._actor_mean(observations)
        with torch.no_grad():
            self.std.clamp_(self.min_noise_std, self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self._actor_mean(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        return None

    def forward(self):
        raise NotImplementedError