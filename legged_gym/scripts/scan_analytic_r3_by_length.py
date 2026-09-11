"""Tune R3 steering and stopping while exposing failures by path length."""

import json
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
    torch.manual_seed(7701)
    np.random.seed(7701)
    configs = []
    for stop_distance in (1.2, 1.6, 2.0):
        for k_cross in (0.3, 0.6):
            for k_heading in (0.8, 1.2, 1.5):
                for gain_mode in (0.60, 0.69, -1.0):
                    configs.append((stop_distance, k_cross, k_heading, gain_mode))
    repeats = 4
    assignment = [cfg for cfg in configs for _ in range(repeats)]

    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs = len(assignment)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    args.seed = 7701
    env_cfg.seed = 7701
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    env.path_curriculum_stage = 2
    env.forced_path_type = 2
    env.forced_curvature = -1.0 / 3.0
    params = torch.tensor(assignment, dtype=torch.float, device=env.device)
    stop_distance, k_cross, k_heading, gain_mode = [params[:, i] for i in range(4)]
    drive = 0.65
    all_ids = torch.arange(env.num_envs, device=env.device)
    records = []

    for length in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
        env.cfg.path.length_range_stage2 = [length, length]
        env.reset_idx(all_ids)
        env.compute_observations()
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        with torch.no_grad():
            for step in range(int(env.max_episode_length) + 5):
                cruise_speed = 1.18 * drive
                desired_speed = cruise_speed * torch.clamp(
                    env.path_endpoint_distance / stop_distance, 0.0, 1.0
                )
                forward_speed = env.base_lin_vel[:, 0]
                a1 = desired_speed / 1.18 + 0.65 * (desired_speed - forward_speed)
                command = (
                    env._lookahead_curvature()
                    - k_cross * env.path_cross_track
                    - k_heading * env.path_heading_error
                )
                dynamic_gain = torch.clamp(0.092 + 0.92 * torch.abs(a1), 0.25, 0.75)
                gain = torch.where(gain_mode < 0.0, dynamic_gain, gain_mode)
                a2 = -command / gain
                actions = torch.stack((a1, a2), dim=1).clamp(-1.0, 1.0)
                _, _, _, dones, _ = env.step(actions)
                done_ids = torch.nonzero(dones & active, as_tuple=False).flatten()
                for idx in done_ids.tolist():
                    cfg = assignment[idx]
                    records.append({
                        "length": length, "stop_distance": cfg[0],
                        "k_cross": cfg[1], "k_heading": cfg[2],
                        "gain_mode": cfg[3],
                        "success": int(env.terminal_success[idx].item()),
                        "reason": int(env.terminal_reason[idx].item()),
                        "cross_track_m": float(env.terminal_cross_track[idx].item()),
                        "endpoint_distance_m": float(env.terminal_endpoint_distance[idx].item()),
                        "terminal_speed_mps": float(env.terminal_speed[idx].item()),
                    })
                active[done_ids] = False
                if not bool(active.any().item()):
                    break

    by_config = defaultdict(list)
    for row in records:
        key = (row["stop_distance"], row["k_cross"], row["k_heading"], row["gain_mode"])
        by_config[key].append(row)
    summary = []
    for cfg in configs:
        rows = by_config[cfg]
        rates = {}
        for length in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
            subset = [r for r in rows if r["length"] == length]
            rates[str(length)] = float(np.mean([r["success"] for r in subset]))
        summary.append({
            "stop_distance": cfg[0], "k_cross": cfg[1],
            "k_heading": cfg[2], "gain_mode": "dynamic" if cfg[3] < 0 else cfg[3],
            "overall_success": float(np.mean([r["success"] for r in rows])),
            "minimum_length_success": min(rates.values()),
            "success_by_length": rates,
            "median_cross_track_m": float(np.median([r["cross_track_m"] for r in rows])),
            "median_endpoint_distance_m": float(np.median([r["endpoint_distance_m"] for r in rows])),
        })
    summary.sort(key=lambda r: (r["minimum_length_success"], r["overall_success"]), reverse=True)
    out = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/r3_length_scan")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "episodes.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(summary[:15], indent=2))


if __name__ == "__main__":
    main()
