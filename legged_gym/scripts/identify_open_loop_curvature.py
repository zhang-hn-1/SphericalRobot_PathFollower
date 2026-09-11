"""Measure Rotunbot path curvature under a grid of constant joint targets."""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401; must precede torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def fit_heading_curvature(xy, dt_sample):
    """Return travelled length, mean speed, and d(velocity heading)/ds."""
    delta = np.diff(xy, axis=0)
    step_length = np.linalg.norm(delta, axis=-1)
    # Reject tiny steps whose direction is numerical noise.
    valid = step_length > 2.0e-4
    heading = np.unwrap(np.arctan2(delta[..., 1], delta[..., 0]), axis=0)
    arc = np.cumsum(step_length, axis=0)
    results = []
    for env_id in range(xy.shape[1]):
        mask = valid[:, env_id]
        s = arc[:, env_id][mask]
        h = heading[:, env_id][mask]
        if len(s) < 8 or s[-1] - s[0] < 0.25:
            curvature = np.nan
        else:
            centered = s - s.mean()
            curvature = float(np.sum(centered * (h - h.mean())) / np.sum(centered ** 2))
        total = float(step_length[:, env_id].sum())
        speed = total / (dt_sample * step_length.shape[0])
        results.append((total, speed, curvature))
    return results


def main():
    args = get_args()
    torch.manual_seed(7101)
    np.random.seed(7101)
    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    first_values = [-0.90, -0.65, -0.40, 0.40, 0.65, 0.90]
    second_values = [-0.90, -0.60, -0.30, 0.00, 0.30, 0.60, 0.90]
    repeats = 6
    pairs = [(a1, a2) for a1 in first_values for a2 in second_values for _ in range(repeats)]
    env_cfg.env.num_envs = len(pairs)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.path.success_remaining_length = -1.0
    env_cfg.path.success_endpoint_distance = -1.0
    env_cfg.path.deviation_termination = 100.0
    env_cfg.path.unstable_gravity_z = 2.0
    env_cfg.env.episode_length_s = 30.0
    args.seed = 7101
    env_cfg.seed = 7101
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)

    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    env.compute_observations()
    actions = torch.tensor(pairs, dtype=torch.float, device=env.device)
    samples = []
    warmup_steps = 100
    total_steps = 600
    sample_stride = 5
    with torch.inference_mode():
        for step in range(total_steps):
            env.step(actions)
            if step >= warmup_steps and (step - warmup_steps) % sample_stride == 0:
                samples.append(env.root_states[:, :2].detach().cpu().numpy().copy())
    xy = np.stack(samples, axis=0)
    raw = fit_heading_curvature(xy, env.dt * sample_stride)

    rows = []
    for (a1, a2), (distance, speed, curvature) in zip(pairs, raw):
        rows.append({
            "action_joint1": a1,
            "action_joint2": a2,
            "distance_m": distance,
            "mean_speed_mps": speed,
            "trajectory_curvature_inv_m": curvature,
            "turn_radius_m": abs(1.0 / curvature) if np.isfinite(curvature) and abs(curvature) > 1.0e-4 else None,
        })

    grouped = []
    for a1 in first_values:
        for a2 in second_values:
            subset = [r for r in rows if r["action_joint1"] == a1 and r["action_joint2"] == a2]
            curvatures = np.array([r["trajectory_curvature_inv_m"] for r in subset], dtype=float)
            speeds = np.array([r["mean_speed_mps"] for r in subset], dtype=float)
            finite = np.isfinite(curvatures)
            grouped.append({
                "action_joint1": a1,
                "action_joint2": a2,
                "median_speed_mps": float(np.median(speeds)),
                "median_curvature_inv_m": float(np.median(curvatures[finite])) if finite.any() else None,
                "curvature_p10_inv_m": float(np.percentile(curvatures[finite], 10)) if finite.any() else None,
                "curvature_p90_inv_m": float(np.percentile(curvatures[finite], 90)) if finite.any() else None,
                "valid_repeats": int(finite.sum()),
            })

    out = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/system_identification/open_loop_grid")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (out / "grouped.json").open("w") as handle:
        json.dump(grouped, handle, indent=2)
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
