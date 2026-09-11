"""Grid-search a deterministic geometric controller before residual PPO."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    torch.manual_seed(7201)
    np.random.seed(7201)

    # Values are deliberately broad. Each tuple is evaluated on several
    # independently randomized R3 paths in the same GPU simulation.
    configs = []
    for drive in (0.50, 0.65):
        for stop_distance in (0.8, 1.2, 1.6):
            for k_cross in (0.6, 1.2):
                for k_heading in (0.8, 1.5):
                    configs.append((drive, stop_distance, k_cross, k_heading))
    repeats = 16
    assignment = [cfg for cfg in configs for _ in range(repeats)]

    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs = len(assignment)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    args.seed = 7201
    env_cfg.seed = 7201
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    env.path_curriculum_stage = 2
    env.forced_path_type = 2
    env.forced_curvature = -1.0 / 3.0
    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    env.compute_observations()

    parameters = torch.tensor(assignment, dtype=torch.float, device=env.device)
    drive = parameters[:, 0]
    stop_distance = parameters[:, 1]
    k_cross = parameters[:, 2]
    k_heading = parameters[:, 3]
    records = []
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    max_steps = int(env.max_episode_length) + 5

    with torch.inference_mode():
        for step in range(max_steps):
            # The system-identification grid gives forward speed ~= 1.18*a1.
            cruise_speed = 1.18 * drive
            distance_scale = torch.clamp(env.path_endpoint_distance / stop_distance, 0.0, 1.0)
            desired_speed = cruise_speed * distance_scale
            forward_speed = env.base_lin_vel[:, 0]
            a1 = desired_speed / 1.18 + 0.65 * (desired_speed - forward_speed)

            desired_curvature = env._lookahead_curvature()
            curvature_command = (
                desired_curvature
                - k_cross * env.path_cross_track
                - k_heading * env.path_heading_error
            )
            # At a1 around 0.5--0.65, measured kappa ~= -gain*a2.
            gain = 0.47 + (drive - 0.40) * (0.69 - 0.47) / 0.25
            a2 = -curvature_command / gain
            actions = torch.stack((a1, a2), dim=1).clamp(-1.0, 1.0)
            _, _, _, dones, _ = env.step(actions)
            done_ids = torch.nonzero(dones & active, as_tuple=False).flatten()
            for idx in done_ids.tolist():
                cfg = assignment[idx]
                records.append({
                    "drive": cfg[0], "stop_distance": cfg[1],
                    "k_cross": cfg[2], "k_heading": cfg[3],
                    "success": int(env.terminal_success[idx].item()),
                    "reason": int(env.terminal_reason[idx].item()),
                    "cross_track_m": float(env.terminal_cross_track[idx].item()),
                    "endpoint_distance_m": float(env.terminal_endpoint_distance[idx].item()),
                    "terminal_speed_mps": float(env.terminal_speed[idx].item()),
                    "steps": step + 1,
                })
            active[done_ids] = False
            if not bool(active.any().item()):
                break

    grouped = []
    buckets = defaultdict(list)
    for row in records:
        key = (row["drive"], row["stop_distance"], row["k_cross"], row["k_heading"])
        buckets[key].append(row)
    for cfg in configs:
        rows = buckets[cfg]
        grouped.append({
            "drive": cfg[0], "stop_distance": cfg[1],
            "k_cross": cfg[2], "k_heading": cfg[3],
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
            "median_cross_track_m": float(np.median([r["cross_track_m"] for r in rows])) if rows else math.nan,
            "median_endpoint_distance_m": float(np.median([r["endpoint_distance_m"] for r in rows])) if rows else math.nan,
            "median_terminal_speed_mps": float(np.median([r["terminal_speed_mps"] for r in rows])) if rows else math.nan,
            "reason_counts": {str(reason): sum(r["reason"] == reason for r in rows) for reason in range(5)},
        })
    grouped.sort(key=lambda row: (row["success_rate"], -row["median_cross_track_m"]), reverse=True)

    out = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/r3_grid")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    with (out / "summary.json").open("w") as handle:
        json.dump(grouped, handle, indent=2)
    print(json.dumps(grouped[:12], indent=2))


if __name__ == "__main__":
    main()
