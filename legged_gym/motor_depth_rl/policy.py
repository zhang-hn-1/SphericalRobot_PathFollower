"""Trainable raw-depth motor policy; independent of the frozen-encoder SRU run.

step/sequence return pre-tanh Normal location, value, and caller-owned (h,c).
Use deterministic_action(location) for evaluation. During PPO collection save
the pre-tanh sample and pass it back to log_prob; never use a clipped Normal
likelihood for the bounded action. Physical joint scaling belongs to the env.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL


class TanhNormal:
    """Elementwise tanh push-forward of a Normal, with stable exact log density.

    entropy() uses exact base entropy plus one reparameterized Monte Carlo
    estimate of E[log|tanh'|]; it is differentiable but not an analytic entropy.
    KL between distributions with this same bijection equals base-Normal KL.
    No event dimensions are reduced: callers sum over the two action columns.
    """
    def __init__(self, location, scale):
        self.base_loc, self.base_scale = torch.broadcast_tensors(location, scale)
        self.base_dist = Normal(self.base_loc, self.base_scale, validate_args=False)
        self.scale = self.base_scale  # Compatibility: this is PRE-tanh std.

    @staticmethod
    def _log_jacobian(pre_tanh):
        return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))

    def sample_with_pre_tanh(self, sample_shape=torch.Size()):
        with torch.no_grad():
            latent = self.base_dist.sample(sample_shape)
            return latent.tanh(), latent

    def rsample_with_pre_tanh(self, sample_shape=torch.Size()):
        latent = self.base_dist.rsample(sample_shape)
        return latent.tanh(), latent

    def sample(self, sample_shape=torch.Size()):
        return self.sample_with_pre_tanh(sample_shape)[0]

    def rsample(self, sample_shape=torch.Size()):
        return self.rsample_with_pre_tanh(sample_shape)[0]

    def log_prob(self, action, pre_tanh_value=None):
        if pre_tanh_value is None:
            # Convenience for interior actions. Saved latent is required for
            # exact PPO reuse when float32 tanh has rounded to +/-1.
            eps = torch.finfo(action.dtype).eps
            bounded = action.clamp(-1.0 + eps, 1.0 - eps)
            pre_tanh_value = 0.5 * (bounded.log1p() - (-bounded).log1p())
        return (self.base_dist.log_prob(pre_tanh_value)
                - self._log_jacobian(pre_tanh_value))

    def entropy(self, num_samples=1):
        if int(num_samples) != num_samples or num_samples <= 0:
            raise ValueError("entropy num_samples must be a positive integer")
        latent = self.base_dist.rsample(torch.Size([int(num_samples)]))
        return self.base_dist.entropy() + self._log_jacobian(latent).mean(dim=0)

    def kl_to(self, other):
        return (torch.log(other.base_scale / self.base_scale)
                + (self.base_scale.square() + (self.base_loc - other.base_loc).square())
                / (2.0 * other.base_scale.square()) - 0.5)


class RawDepthCNN(nn.Module):
    """Metric depth plus validity mask; no pretrained/frozen feature cache."""
    def __init__(self, far_plane=10.0):
        super().__init__()
        if not math.isfinite(far_plane) or far_plane <= 0:
            raise ValueError("far_plane must be finite and positive")
        self.far_plane = float(far_plane)
        self.network = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(32 * 5 * 8, 64), nn.ELU(),
        )

    def forward(self, depth, blind=False):
        if depth.ndim != 4 or tuple(depth.shape[1:]) != (1, 40, 64):
            raise ValueError("raw depth must have shape [B,1,40,64]")
        valid = torch.isfinite(depth) & (depth > 0.0) & (depth <= self.far_plane)
        metric = torch.where(valid, depth, torch.zeros_like(depth)) / self.far_plane
        inputs = torch.cat((metric, valid.to(depth.dtype)), dim=1)
        if blind:
            inputs = torch.zeros_like(inputs)
        return self.network(inputs)


class MotorDepthPolicy(nn.Module):
    is_recurrent = True
    num_actor_obs = 2943
    body_obs_dim = 380
    body_critic_dim = 63
    num_proprio_obs = 19
    num_short_obs = 95
    num_actions = 2
    depth_shape = (1, 40, 64)
    action_semantics = "tanh_normalized_joint_targets"

    def __init__(self, critic_extra_dim=125, hidden_size=128, far_plane=10.0,
                 init_noise_std=(0.35, 0.35), min_noise_std=(0.08, 0.08),
                 max_noise_std=(0.8, 0.8), initialization_mode="bounded_residual",
                 reset_critic_head=True, anchor_limit=.8, input_mode="depth"):
        super().__init__()
        if int(critic_extra_dim) != critic_extra_dim or critic_extra_dim <= 0:
            raise ValueError("critic_extra_dim must be a positive integer")
        if int(hidden_size) != hidden_size or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = int(hidden_size)
        self.critic_extra_dim = int(critic_extra_dim)
        self.num_critic_obs = self.body_critic_dim + self.critic_extra_dim
        if initialization_mode not in ("bounded_residual", "reset_head"):
            raise ValueError("Unknown initialization_mode")
        if input_mode not in ("depth", "blind"):
            raise ValueError("input_mode must be depth or blind")
        self.input_mode = input_mode
        if not math.isfinite(anchor_limit) or not 0 < anchor_limit < 1:
            raise ValueError("anchor_limit must be strictly between zero and one")
        self.initialization_mode = initialization_mode
        self.reset_critic_head = bool(reset_critic_head)
        self.anchor_limit = float(anchor_limit)
        initial, lower, upper = self._validate_stds(init_noise_std, min_noise_std, max_noise_std)
        body = ActorCriticDWL(95, 19, 63, 2, actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128], in_channels=20, kernel_size=[3, 2],
            filter_size=[16, 8], stride_size=[1, 1], lh_output_dim=16,
            activation="elu", init_noise_std=.5, min_noise_std=.2, max_noise_std=1.5)
        self._source_baseline_keys = frozenset(body.state_dict().keys())
        # Original ActorCriticDWL reuses one ELU object at several indices.
        # children() deduplicates that object and would silently drop activations.
        self.actor_body = nn.Sequential(*(body.actor[i] for i in range(len(body.actor) - 1)))
        self.baseline_action_head = body.actor[-1] if initialization_mode == "bounded_residual" else None
        self.long_history = body.long_history
        self.body_feature_norm = nn.LayerNorm(128, elementwise_affine=False)
        self.critic = body.critic
        if self.reset_critic_head:
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
            nn.init.zeros_(self.critic[-1].bias)
        self.critic_extra_projection = nn.Linear(self.critic_extra_dim, 512, bias=False)
        nn.init.zeros_(self.critic_extra_projection.weight)
        self.depth_encoder = RawDepthCNN(far_plane)
        self.gru = nn.GRU(64 + 19 + 2, self.hidden_size, num_layers=1)
        self.actor_head = nn.Linear(128 + self.hidden_size + 2, 2)
        if initialization_mode == "bounded_residual":
            nn.init.zeros_(self.actor_head.weight)
        else:
            nn.init.orthogonal_(self.actor_head.weight, gain=.01)
        nn.init.zeros_(self.actor_head.bias)
        self.log_std = nn.Parameter(torch.tensor(initial).log())
        self.min_noise_std, self.max_noise_std = lower, upper
        self.baseline_metadata = None

    @staticmethod
    def _validate_stds(initial, lower, upper):
        arrays = []
        for value in (initial, lower, upper):
            value = torch.as_tensor(value, dtype=torch.float64, device="cpu")
            if value.ndim == 0:
                value = value.repeat(2)
            if value.shape != (2,) or not bool(torch.isfinite(value).all()):
                raise ValueError("std specification must contain two finite values")
            arrays.append(value)
        if not bool(((arrays[1] > 0) & (arrays[1] <= arrays[0]) & (arrays[0] <= arrays[2])).all()):
            raise ValueError("std must satisfy 0 < min <= init <= max")
        return tuple(tuple(float(x) for x in a.tolist()) for a in arrays)

    @property
    def std(self):
        # Clamp without mutating parameters in inference. This is Normal std
        # before tanh, not the standard deviation of the executed target.
        low = self.log_std.new_tensor(self.min_noise_std).log()
        high = self.log_std.new_tensor(self.max_noise_std).log()
        return torch.maximum(torch.minimum(self.log_std, high), low).exp()

    def initialize_exploration(self, init_noise_std=(.35, .35),
                               min_noise_std=(.08, .08), max_noise_std=(.8, .8)):
        """Explicit fresh-run pre-tanh exploration; call before checkpoint load."""
        initial, lower, upper = self._validate_stds(
            init_noise_std, min_noise_std, max_noise_std)
        previous = self.std.detach().cpu().tolist()
        with torch.no_grad():
            self.log_std.copy_(self.log_std.new_tensor(initial).log())
        self.min_noise_std, self.max_noise_std = lower, upper
        metadata = dict(init=list(initial), min=list(lower), max=list(upper),
            previous_std=previous, coordinates="pre-tanh Normal",
            source_baseline_std=None if self.baseline_metadata is None
                else self.baseline_metadata.get("source_baseline_std"),
            baseline_std_override=self.baseline_metadata is not None)
        if self.baseline_metadata is not None:
            self.baseline_metadata["exploration_override"] = dict(metadata)
        return metadata

    def parameter_groups(self):
        groups = dict(
            actor_body=list(self.actor_body.parameters()) + list(self.long_history.parameters()),
            head=list(self.actor_head.parameters()),
            visual=list(self.depth_encoder.parameters()) + list(self.gru.parameters()),
            critic=list(self.critic.parameters()) + list(self.critic_extra_projection.parameters()),
            std=[self.log_std],
        )
        if self.baseline_action_head is not None:
            groups["actor_body"] += list(self.baseline_action_head.parameters())
        identifiers = [id(p) for values in groups.values() for p in values]
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != {id(p) for p in self.parameters()}:
            raise RuntimeError("Parameter groups must cover every parameter exactly once")
        return groups

    def initial_state(self, batch, device):
        if int(batch) != batch or batch <= 0:
            raise ValueError("batch must be a positive integer")
        shape = (1, int(batch), self.hidden_size)
        return tuple(torch.zeros(shape, device=device, dtype=self.log_std.dtype) for _ in range(2))

    def _mask_state(self, state, starts, batch):
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must be (h,c); c is a zero GRU compatibility placeholder")
        for item in state:
            if (item.shape != (1, batch, self.hidden_size) or item.device != self.log_std.device
                    or item.dtype != self.log_std.dtype):
                raise ValueError("state shape/device/dtype mismatch")
        if starts.shape != (batch,) or starts.device != self.log_std.device or starts.dtype != torch.bool:
            raise ValueError("episode_starts must be bool [B] on the policy device")
        return torch.where(starts.reshape(1, batch, 1), torch.zeros_like(state[0]), state[0])

    def value(self, critic_obs):
        if critic_obs.ndim != 2 or critic_obs.shape[1] != self.num_critic_obs:
            raise ValueError("critic observation dimension mismatch")
        hidden = (self.critic[0](critic_obs[:, :63])
                  + self.critic_extra_projection(critic_obs[:, 63:]))
        for index in range(1, len(self.critic)):
            hidden = self.critic[index](hidden)
        return hidden

    def step(self, obs, critic_obs, state, episode_starts):
        if obs.ndim != 2 or obs.shape[1] != self.num_actor_obs:
            raise ValueError("actor observation must have shape [B,2943]")
        if critic_obs.shape != (obs.shape[0], self.num_critic_obs):
            raise ValueError("critic observation dimension mismatch")
        if any(x.device != self.log_std.device or x.dtype != self.log_std.dtype for x in (obs, critic_obs)):
            raise ValueError("observation device/dtype must match policy")
        flag = obs[:, -1]
        if not bool(((flag == 0) | (flag == 1)).all()):
            raise ValueError("fresh flag must be zero or one")
        batch = obs.shape[0]
        h = self._mask_state(state, episode_starts, batch)
        body = obs[:, :380]
        targets = obs[:, 2940:2942]
        fresh = torch.nonzero(flag == 1, as_tuple=False).flatten()
        if fresh.numel():
            raw = obs.index_select(0, fresh)[:, 380:2940].reshape(-1, *self.depth_shape)
            features = self.depth_encoder(raw, blind=self.input_mode == "blind")
            inputs = torch.cat((features, body.index_select(0, fresh)[:, -19:],
                                targets.index_select(0, fresh)), dim=-1).unsqueeze(0)
            _, updated = self.gru(inputs, h.index_select(1, fresh))
            h = h.index_copy(1, fresh, updated)
        history = self.long_history(body.reshape(batch, 20, 19))
        raw_body_features = self.actor_body(torch.cat((body[:, -95:], history), dim=-1))
        body_features = self.body_feature_norm(raw_body_features)
        location = self.actor_head(torch.cat((body_features, h[0], targets), dim=-1))
        if self.baseline_action_head is not None:
            raw_anchor = self.baseline_action_head(raw_body_features)
            normalized_anchor = (raw_anchor / raw_anchor.new_tensor([3., .9])).clamp(
                -self.anchor_limit, self.anchor_limit)
            anchor_location = .5 * (normalized_anchor.log1p() - (-normalized_anchor).log1p())
            location = location + anchor_location
        return location, self.value(critic_obs), (h, torch.zeros_like(h))

    def sequence(self, obs, critic_obs, initial_state, episode_starts):
        if obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[2] != self.num_actor_obs:
            raise ValueError("actor sequence must be [T,B,2943], T>0")
        timesteps, batch = obs.shape[:2]
        if critic_obs.shape != (timesteps, batch, self.num_critic_obs):
            raise ValueError("critic sequence dimension mismatch")
        if episode_starts.shape != (timesteps, batch) or episode_starts.dtype != torch.bool:
            raise ValueError("start sequence must be bool [T,B]")
        if episode_starts.device != self.log_std.device:
            raise ValueError("episode_starts must use the policy device")
        if any(x.device != self.log_std.device or x.dtype != self.log_std.dtype for x in (obs, critic_obs)):
            raise ValueError("observation device/dtype must match policy")
        # _mask_state validates the tuple and applies the first reset exactly once.
        h = self._mask_state(initial_state, episode_starts[0], batch)
        markers = torch.stack((obs[:, :, -1], episode_starts.to(obs.dtype)), dim=-1).detach().cpu()
        fresh_cpu = markers[:, :, 0]
        if not bool(((fresh_cpu == 0) | (fresh_cpu == 1)).all()):
            raise ValueError("fresh flag must be zero or one")
        counts = fresh_cpu.sum(dim=1).to(torch.long).tolist()
        has_starts = markers[:, :, 1].bool().any(dim=1).tolist()
        flat_indices_cpu = torch.nonzero(fresh_cpu.reshape(-1) == 1, as_tuple=False).flatten()
        flat_indices = flat_indices_cpu.to(device=obs.device)
        env_indices = flat_indices.remainder(batch)

        flat_obs = obs.reshape(timesteps * batch, self.num_actor_obs)
        body = flat_obs[:, :380]
        targets = flat_obs[:, 2940:2942]
        history = self.long_history(body.reshape(timesteps * batch, 20, 19))
        raw_body_features = self.actor_body(torch.cat((body[:, -95:], history), dim=-1))
        body_features = self.body_feature_norm(raw_body_features)
        if flat_indices_cpu.numel():
            selected = flat_obs.index_select(0, flat_indices)
            depth_features = self.depth_encoder(selected[:, 380:2940].reshape(-1, *self.depth_shape),
                                                blind=self.input_mode == "blind")
            recurrent_inputs = torch.cat((depth_features, selected[:, 361:380],
                                          selected[:, 2940:2942]), dim=-1)
        else:
            recurrent_inputs = None
        memories, offset = [], 0
        for t in range(timesteps):
            if t and has_starts[t]:
                mask = episode_starts[t].reshape(1, batch, 1)
                h = torch.where(mask, torch.zeros_like(h), h)
            count = counts[t]
            if count:
                indices = env_indices[offset:offset + count]
                inputs = recurrent_inputs[offset:offset + count].unsqueeze(0)
                _, updated = self.gru(inputs, h.index_select(1, indices))
                h = h.index_copy(1, indices, updated)
                offset += count
            memories.append(h[0])
        memory = torch.stack(memories).reshape(timesteps * batch, self.hidden_size)
        locations = self.actor_head(torch.cat((body_features, memory, targets), dim=-1))
        if self.baseline_action_head is not None:
            raw_anchor = self.baseline_action_head(raw_body_features)
            anchor = (raw_anchor / raw_anchor.new_tensor([3., .9])).clamp(-self.anchor_limit, self.anchor_limit)
            locations = locations + .5 * (anchor.log1p() - (-anchor).log1p())
        values = self.value(critic_obs.reshape(timesteps * batch, self.num_critic_obs))
        return (locations.reshape(timesteps, batch, 2),
                values.reshape(timesteps, batch, 1), (h, torch.zeros_like(h)))

    def distribution(self, locations):
        if locations.shape[-1] != 2:
            raise ValueError("Normal locations must have two action columns")
        return TanhNormal(locations, self.std)

    @staticmethod
    def deterministic_action(locations):
        """Tanh of base-Normal center; not an analytic transformed mean/mode."""
        return locations.tanh()

    def load_baseline(self, path):
        """Initialize the chosen body anchor; deliberately NOT action-equivalent."""
        if self.baseline_metadata is not None:
            raise ValueError("Baseline initialization already applied; resume uses a full new checkpoint")
        path = Path(path)
        checkpoint = torch.load(str(path), map_location="cpu")
        weights = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(weights, dict) or set(weights) != self._source_baseline_keys:
            raise ValueError("Expected exactly the trusted original model800 state keys")
        mappings = {}
        for key in self.state_dict():
            if key.startswith("actor_body."):
                mappings[key] = "actor." + key[len("actor_body."):]
            elif key.startswith("baseline_action_head."):
                mappings[key] = "actor.6." + key[len("baseline_action_head."):]
            elif key.startswith("critic.") and not (self.reset_critic_head and key.startswith("critic.6.")):
                mappings[key] = key
            elif key.startswith("long_history."):
                mappings[key] = key
        current = self.state_dict()
        for target, source in mappings.items():
            tensor = weights[source]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != current[target].shape
                    or tensor.dtype != current[target].dtype or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Invalid baseline tensor: " + source)
        for key in ("actor.6.weight", "actor.6.bias", "std"):
            if not isinstance(weights[key], torch.Tensor) or not bool(torch.isfinite(weights[key]).all()):
                raise ValueError("Invalid discarded baseline tensor: " + key)
        merged = dict(current)
        merged.update({target: weights[source] for target, source in mappings.items()})
        self.load_state_dict(merged, strict=True)
        discarded = sorted(set(weights) - set(mappings.values()))
        self.baseline_metadata = dict(
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            iteration=int(checkpoint.get("iter", 0)), optimizer_loaded=False,
            source_baseline_std=weights["std"].detach().cpu().tolist(),
            action_equivalent_to_baseline=False, body_hidden_tensors_preserved=True,
            initialization_mode=self.initialization_mode, input_mode=self.input_mode,
            loaded_keys=mappings, discarded_keys=discarded,
            new_action_head="Linear(128 + GRU{} + 2 previous targets, 2); {} weight, zero bias".format(
                self.hidden_size, "zero" if self.initialization_mode == "bounded_residual" else "orthogonal gain 0.01"),
            bounded_residual_anchor="atanh(clamp(original_raw/[3,0.9], -{0}, {0})); add new head before tanh".format(self.anchor_limit)
                if self.initialization_mode == "bounded_residual" else None,
            anchor_limit=self.anchor_limit if self.initialization_mode == "bounded_residual" else None,
            anchor_interpretation="Only high normalized magnitudes are capped; interior targets are not multiplied by 0.8.",
            critic_output_head_reset=self.reset_critic_head,
            body_feature_normalization="parameter-free LayerNorm128 before the new action head",
            new_visual_branch="trainable raw depth+validity CNN64 and GRU{}; no frozen VAE".format(self.hidden_size),
            action_distribution="tanh Normal in normalized joint-target coordinates [-1,1]",
            std_before_tanh=self.std.detach().cpu().tolist(),
            critic_extra_dim=self.critic_extra_dim,
        )
        return dict(self.baseline_metadata)
