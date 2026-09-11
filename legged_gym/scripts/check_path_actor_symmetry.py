"""Deterministic unit check for the Rotunbot path actor soft reflection map."""
import torch
from legged_gym.dwl.actor_critic_path_dwl import ActorCriticPathDWL

torch.manual_seed(7)
net = ActorCriticPathDWL(
    num_short_obs=90, num_proprio_obs=18, num_critic_obs=28,
    num_actions=2, history_frames=20, path_obs_dim=43,
    filter_size=(32, 16), kernel_size=(3, 2), stride_size=(1, 1),
    lh_output_dim=32,
)
obs = torch.randn(17, 403)
mirrored_obs = obs * net.observation_mirror_sign
action = net.act_inference(obs)
mirrored_action = net.act_inference(mirrored_obs) * net.action_mirror_sign
raw = net.actor(net._actor_input(obs))
raw_mirrored = net.actor(net._actor_input(mirrored_obs)) * net.action_mirror_sign
raw_asymmetry = raw - raw_mirrored
policy_asymmetry = action - mirrored_action
expected = raw_asymmetry * net.mirror_residual_scale
error = torch.max(torch.abs(policy_asymmetry - expected)).item()
print({
    "observation_shape": list(obs.shape),
    "actor_input_dim": net.actor[0].in_features,
    "action_shape": list(action.shape),
    "soft_symmetry_max_error": error,
    "residual_scales": net.mirror_residual_scale.tolist(),
    "max_abs_action": torch.max(torch.abs(action)).item(),
})
assert net.actor[0].in_features == 165
assert error < 1.0e-6
assert torch.max(torch.abs(action)).item() <= 1.0