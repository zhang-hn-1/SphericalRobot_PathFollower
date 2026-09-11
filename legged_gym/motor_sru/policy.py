"""Motor policy preserving the master-new point policy at initialization.

The 50 Hz body branch is unchanged. A 5 Hz visual branch uses the official
attention and SRU star cell, with explicit caller-owned recurrent state. This
is a new single-policy motor PPO experiment, not the official SRU navigator.
"""

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL
from .vendor.attention import CrossAttentionFuseModule
from .vendor.lstm_sru import LSTM_SRU


State = Tuple[torch.Tensor, torch.Tensor]


class MotorSRUPolicy(nn.Module):
    """Body380 + depth2560 + fresh1; critic body63 + geometry28 + time1.

    ``episode_starts`` marks the observation *after* reset, before its decision.
    A fresh flag of zero holds memory, even at a new episode (where it is zero).
    No state, action distribution, or dropout mask is retained on this module.
    """

    is_recurrent = True
    num_actor_obs = 2941
    num_critic_obs = 92
    num_actions = 2
    body_obs_dim = 380
    body_critic_dim = 63
    num_proprio_obs = 19
    num_short_obs = 95
    depth_shape = (64, 5, 8)

    def __init__(self, hidden_size=512, init_noise_std=(0.3, 0.15),
                 min_noise_std=(0.1, 0.05), max_noise_std=(0.5, 0.25)):
        super().__init__()
        if int(hidden_size) != hidden_size or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self._validate_exploration(init_noise_std, min_noise_std, max_noise_std)
        self.hidden_size = int(hidden_size)
        body = ActorCriticDWL(
            num_short_obs=95, num_proprio_obs=19, num_critic_obs=63,
            num_actions=2, actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128], in_channels=20,
            kernel_size=[3, 2], filter_size=[16, 8], stride_size=[1, 1],
            lh_output_dim=16, activation="elu", init_noise_std=0.5,
            min_noise_std=0.2, max_noise_std=1.5,
        )
        # Preserve names and tensor dimensions of every original parameter.
        self.actor = body.actor
        self.critic = body.critic
        self.long_history = body.long_history
        self.std = body.std
        self._baseline_keys = frozenset(self.state_dict().keys())
        self.attention = CrossAttentionFuseModule(
            image_dim=64, info_dim=19, num_heads=4, spatial_dims=(1, 5, 8)
        )
        self.sru = LSTM_SRU(input_size=64 + 19, hidden_size=self.hidden_size,
                            num_layers=1)
        self.visual_projection = nn.Linear(self.hidden_size, 512, bias=False)
        self.critic_extra_projection = nn.Linear(29, 512, bias=False)
        nn.init.zeros_(self.visual_projection.weight)
        nn.init.zeros_(self.critic_extra_projection.weight)
        self.baseline_metadata = None
        self.exploration_metadata = None
        self.initialize_exploration(init_noise_std, min_noise_std, max_noise_std)

    @staticmethod
    def _validate_exploration(initial, lower, upper):
        values = []
        for name, value in (("init", initial), ("min", lower), ("max", upper)):
            array = torch.as_tensor(value, dtype=torch.float64, device="cpu")
            if array.ndim == 0:
                array = array.repeat(2)
            if array.shape != (2,) or not bool(torch.isfinite(array).all()):
                raise ValueError("Normal std {} must contain two finite values".format(name))
            values.append(array)
        initial, lower, upper = values
        if not bool(((lower > 0) & (lower <= initial) & (initial <= upper)).all()):
            raise ValueError("Normal std must satisfy 0 < min <= init <= max per joint")
        return tuple(tuple(float(x) for x in array.tolist()) for array in values)

    def initialize_exploration(self, init_noise_std=(0.3, 0.15),
                               min_noise_std=(0.1, 0.05), max_noise_std=(0.5, 0.25)):
        """Explicit fresh-run override; never call after restoring a trained policy.

        Keep the baseline's ``std`` parameter name and optimizer identity. The
        returned JSON data records a deliberate exploration change separately
        from loading the original actor/critic tensors.
        """
        initial, lower, upper = self._validate_exploration(
            init_noise_std, min_noise_std, max_noise_std)
        previous = self.std.detach().cpu().tolist()
        with torch.no_grad():
            self.std.copy_(self.std.new_tensor(initial))
        self.min_noise_std, self.max_noise_std = lower, upper
        metadata = dict(init=list(initial), min=list(lower), max=list(upper),
                        previous_std=previous,
                        baseline_std_override=self.baseline_metadata is not None)
        self.exploration_metadata = metadata
        if self.baseline_metadata is not None:
            self.baseline_metadata["exploration_override"] = dict(metadata)
        return dict(metadata)

    def initial_state(self, batch, device) -> State:
        if int(batch) != batch or batch <= 0:
            raise ValueError("batch must be a positive integer")
        shape = (1, int(batch), self.hidden_size)
        return (torch.zeros(shape, device=device, dtype=self.std.dtype),
                torch.zeros(shape, device=device, dtype=self.std.dtype))

    def _validate_obs(self, obs, critic_obs):
        if obs.ndim != 2 or obs.shape[1] != self.num_actor_obs:
            raise ValueError("actor observation must have shape [B,2941]")
        if critic_obs.ndim != 2 or critic_obs.shape != (obs.shape[0], self.num_critic_obs):
            raise ValueError("critic observation must have shape [B,92]")
        if obs.device != self.std.device or critic_obs.device != obs.device:
            raise ValueError("observations and policy must use the same device")
        if obs.dtype != self.std.dtype or critic_obs.dtype != self.std.dtype:
            raise ValueError("observations and policy must use the same dtype")

    def _masked_state(self, state, starts, batch) -> State:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must be (h,c)")
        for item in state:
            if item.shape != (1, batch, self.hidden_size):
                raise ValueError("state must have shape [1,B,H]")
            if item.device != self.std.device or item.dtype != self.std.dtype:
                raise ValueError("state and policy must use the same device and dtype")
        if starts.shape != (batch,) or starts.device != self.std.device:
            raise ValueError("episode_starts must have shape [B] on the policy device")
        if starts.dtype != torch.bool:
            raise ValueError("episode_starts must be boolean")
        # where, rather than in-place zeroing, both preserves caller state and
        # cuts gradients to the previous episode, including its held memory.
        mask = starts.reshape(1, batch, 1)
        return tuple(torch.where(mask, torch.zeros_like(item), item) for item in state)

    def value(self, critic_obs):
        """Feed-forward value bootstrap; never evaluates or advances the actor."""
        if critic_obs.ndim != 2 or critic_obs.shape[1] != self.num_critic_obs:
            raise ValueError("critic observation must have shape [B,92]")
        x = (self.critic[0](critic_obs[:, :self.body_critic_dim])
             + self.critic_extra_projection(critic_obs[:, self.body_critic_dim:]))
        for index, layer in enumerate(self.critic):
            if index:
                x = layer(x)
        return x

    def step(self, obs, critic_obs, state: State, episode_starts):
        self._validate_obs(obs, critic_obs)
        batch = obs.shape[0]
        h, c = self._masked_state(state, episode_starts, batch)
        flag = obs[:, -1]
        if not bool(torch.all((flag == 0) | (flag == 1))):
            raise ValueError("fresh_depth_flag must be exactly zero or one")
        indices = torch.nonzero(flag == 1, as_tuple=False).flatten()
        if indices.numel():
            selected = obs.index_select(0, indices)
            body = selected[:, :self.body_obs_dim]
            current = body[:, -self.num_proprio_obs:]
            depth = selected[:, self.body_obs_dim:-1].reshape(-1, *self.depth_shape)
            visual = self.attention(depth, current)
            memory_input = torch.cat((visual, current), dim=-1).unsqueeze(0)
            _, (fresh_h, fresh_c) = self.sru(
                memory_input, (h.index_select(1, indices), c.index_select(1, indices))
            )
            # Functional index_copy preserves gradients for fresh and held envs.
            h = h.index_copy(1, indices, fresh_h)
            c = c.index_copy(1, indices, fresh_c)

        body = obs[:, :self.body_obs_dim]
        history = self.long_history(body.reshape(batch, 20, 19))
        actor_input = torch.cat((body[:, -self.num_short_obs:], history), dim=-1)
        mean = self.actor[0](actor_input) + self.visual_projection(h[0])
        for index, layer in enumerate(self.actor):
            if index:
                mean = layer(mean)
        return mean, self.value(critic_obs), (h, c)

    def sequence(self, obs, critic_obs, initial_state: State, episode_starts):
        """Recompute chronological sequences; no detach and no hidden resets.

        Any padded steps must be masked by the PPO caller when forming losses.
        Real episode starts are reset here, so a sequence may span episodes.
        """
        if obs.ndim != 3 or obs.shape[0] == 0 or obs.shape[2] != self.num_actor_obs:
            raise ValueError("sequence actor observation must have shape [T,B,2941], T>0")
        if critic_obs.shape != (obs.shape[0], obs.shape[1], self.num_critic_obs):
            raise ValueError("sequence critic observation must have shape [T,B,92]")
        if episode_starts.shape != obs.shape[:2]:
            raise ValueError("sequence episode_starts must have shape [T,B]")
        means, values = [], []
        state = initial_state
        for index in range(obs.shape[0]):
            mean, value, state = self.step(obs[index], critic_obs[index], state,
                                           episode_starts[index])
            means.append(mean)
            values.append(value)
        return torch.stack(means), torch.stack(values), state

    def distribution(self, means):
        """Unsquashed Normal; only std is clamped, separately for each joint."""
        if means.shape[-1] != self.num_actions:
            raise ValueError("Normal means must have two action components")
        with torch.no_grad():
            lower = self.std.new_tensor(self.min_noise_std)
            upper = self.std.new_tensor(self.max_noise_std)
            self.std.copy_(torch.minimum(torch.maximum(self.std, lower), upper))
        return Normal(means, means * 0.0 + self.std)

    def load_baseline(self, path) -> Dict:
        """Load only the trusted point-policy tensors, after strict preflight.

        This is initialization, not resume: no optimizer, rollout, environment,
        RNG, or iteration state is restored. New projections must still be zero.
        """
        path = Path(path)
        checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("baseline checkpoint has no model_state_dict")
        weights = checkpoint["model_state_dict"]
        if not isinstance(weights, dict) or set(weights) != self._baseline_keys:
            incoming = set(weights) if isinstance(weights, dict) else set()
            raise ValueError("baseline key mismatch: missing={}, unexpected={}".format(
                sorted(self._baseline_keys - incoming), sorted(incoming - self._baseline_keys)))
        current = self.state_dict()
        for name in sorted(self._baseline_keys):
            tensor = weights[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != current[name].shape
                    or tensor.dtype != current[name].dtype):
                raise ValueError("baseline shape or dtype mismatch: " + name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("nonfinite baseline tensor: " + name)
        if not bool((weights["std"] > 0).all()):
            raise ValueError("baseline std must be positive")
        if (torch.count_nonzero(self.visual_projection.weight).item()
                or torch.count_nonzero(self.critic_extra_projection.weight).item()):
            raise ValueError("load_baseline requires zero fusion; use a full checkpoint for resume")
        # Build a complete state dict so strict=True also validates all new keys.
        merged = dict(current)
        merged.update(weights)
        self.load_state_dict(merged, strict=True)
        self.exploration_metadata = None
        self.baseline_metadata = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "iteration": int(checkpoint.get("iter", 0)),
            "loaded_keys": sorted(self._baseline_keys),
            "new_keys": sorted(set(current) - self._baseline_keys),
            "optimizer_loaded": False,
            "loaded_std": weights["std"].detach().cpu().tolist(),
            "exploration_override": None,
        }
        return dict(self.baseline_metadata)
