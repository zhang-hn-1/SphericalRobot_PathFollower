"""Deterministic fixed-seed evaluation for Rotunbot geometric path tracking."""

import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # must precede torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TYPE_NAMES = {0: "straight", 1: "left_arc", 2: "right_arc", 3: "s_curve"}
REASON_NAMES = {0: "success", 1: "timeout", 2: "deviation", 3: "unstable", 4: "out_of_bounds"}


def checkpoint_iteration(path):
    match = re.search(r"model_(\d+)\.pt$", str(path))
    return int(match.group(1)) if match else -1


def resolve_checkpoint():
    explicit = os.environ.get("PATH_CHECKPOINT")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list(Path(LEGGED_GYM_ROOT_DIR).glob("logs/rotunbot_path/*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError("No rotunbot_path checkpoints found")
    return max(candidates, key=lambda p: (p.parent.stat().st_mtime, checkpoint_iteration(p)))


def main():
    args = get_args()
    checkpoint = resolve_checkpoint()
    stage = int(os.environ.get("PATH_EVAL_STAGE", "0"))
    episodes_requested = int(os.environ.get("PATH_EVAL_EPISODES", "256"))
    seed = int(os.environ.get("PATH_EVAL_SEED", "1100")) + stage
    output_root = Path(os.environ.get(
        "PATH_EVAL_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / "evaluation"),
    ))
    forced_tag = os.environ.get("PATH_EVAL_TYPE", "mixed")
    curvature_tag = os.environ.get("PATH_EVAL_CURVATURE", "random").replace(".", "p")
    split_tag = os.environ.get("PATH_LAYOUT_SPLIT", "train").strip().lower()
    # The split is part of the directory name: the same checkpoint evaluated on
    # train and on held-out curvature would otherwise share a path and the
    # second run would silently overwrite the first.
    output_dir = output_root / checkpoint.parent.name / f"model_{checkpoint_iteration(checkpoint)}" / f"stage_{stage}_{forced_tag}_k{curvature_tag}_{split_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    args.seed = seed
    env_cfg.seed = seed
    train_cfg.seed = seed
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 256
    env_cfg.noise.add_noise = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = False

    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    stochastic = os.environ.get("PATH_EVAL_STOCHASTIC", "0") == "1"

    env.cfg.path.curriculum_enabled = False
    env.path_curriculum_stage = stage
    forced_type = os.environ.get("PATH_EVAL_TYPE")
    if forced_type is not None:
        type_lookup = {"straight": 0, "left_arc": 1, "right_arc": 2, "s_curve": 3}
        env.forced_path_type = type_lookup[forced_type]
    forced_curvature = os.environ.get("PATH_EVAL_CURVATURE")
    if forced_curvature is not None:
        env.forced_curvature = float(forced_curvature)
    all_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_ids)
    env.compute_observations()
    obs = env.get_observations()

    initial_path = env.path_xy[0].detach().cpu().numpy().copy()
    initial_last = int(env.path_last_index[0].item())
    trace_xy = [env.root_states[0, :2].detach().cpu().numpy().copy()]
    trace_actions = []
    trace_done = False
    records = []
    max_steps = int(env.max_episode_length * 4)

    with torch.inference_mode():
        for step in range(max_steps):
            actions = runner.alg.actor_critic.act(obs) if stochastic else policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            if not trace_done:
                if bool(dones[0].item()):
                    position = env.terminal_position[0]
                    trace_done = True
                else:
                    position = env.root_states[0, :2]
                trace_xy.append(position.detach().cpu().numpy().copy())
                trace_actions.append(torch.clamp(actions[0], -1.0, 1.0).detach().cpu().numpy().copy())
            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            if len(done_ids) > 0:
                for idx in done_ids.tolist():
                    record = {
                        "success": int(env.terminal_success[idx].item()),
                        "reason": REASON_NAMES[int(env.terminal_reason[idx].item())],
                        "path_type": TYPE_NAMES[int(env.terminal_path_type[idx].item())],
                        "curvature": float(env.terminal_path_curvature[idx].item()),
                        "path_length_m": float(env.terminal_path_length[idx].item()),
                        "remaining_m": float(env.terminal_path_remaining[idx].item()),
                        "abs_cross_track_m": float(env.terminal_cross_track[idx].item()),
                        "endpoint_distance_m": float(env.terminal_endpoint_distance[idx].item()),
                        "terminal_speed_mps": float(env.terminal_speed[idx].item()),
                        "termination_step": step + 1,
                    }
                    records.append(record)
                    if len(records) >= episodes_requested:
                        break
            if len(records) >= episodes_requested:
                break

    if not records:
        raise RuntimeError("Evaluation produced no completed episodes")
    records = records[:episodes_requested]
    success = np.array([row["success"] for row in records], dtype=float)
    cross = np.array([row["abs_cross_track_m"] for row in records], dtype=float)
    endpoint = np.array([row["endpoint_distance_m"] for row in records], dtype=float)
    speed = np.array([row["terminal_speed_mps"] for row in records], dtype=float)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration(checkpoint),
        "stage": stage,
        "forced_path_type": os.environ.get("PATH_EVAL_TYPE"),
        "forced_curvature": os.environ.get("PATH_EVAL_CURVATURE"),
        "layout_split": os.environ.get("PATH_LAYOUT_SPLIT", "train"),
        "seed": seed,
        "episodes": len(records),
        "stochastic_actions": stochastic,
        "success_rate": float(success.mean()),
        "cross_track_median_m": float(np.median(cross)),
        "cross_track_p95_m": float(np.percentile(cross, 95)),
        "endpoint_distance_median_m": float(np.median(endpoint)),
        "terminal_speed_median_mps": float(np.median(speed)),
        "terminal_speed_p95_mps": float(np.percentile(speed, 95)),
        "by_path_type": {},
        # The curriculum gate and every earlier evaluation only reported path
        # type, which averages the hardest curvature into a passing mean.  This
        # breakdown keeps (path type x curvature) separate.
        "by_type_curvature": {},
        "failure_reasons": {},
    }
    for name in TYPE_NAMES.values():
        subset = [row for row in records if row["path_type"] == name]
        if subset:
            summary["by_path_type"][name] = {
                "episodes": len(subset),
                "success_rate": float(np.mean([row["success"] for row in subset])),
            }
    for reason in REASON_NAMES.values():
        count = sum(row["reason"] == reason for row in records)
        if count:
            summary["failure_reasons"][reason] = count

    buckets = {}
    for row in records:
        buckets.setdefault((row["path_type"], round(row["curvature"], 4)), []).append(row)
    for (name, curvature), subset in sorted(buckets.items()):
        reasons = {}
        for reason in REASON_NAMES.values():
            count = sum(row["reason"] == reason for row in subset)
            if count:
                reasons[reason] = count
        summary["by_type_curvature"][f"{name}@k{curvature:.4g}"] = {
            "path_type": name,
            "curvature": curvature,
            "episodes": len(subset),
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "failure_reasons": reasons,
        }

    if len(trace_actions):
        summary["trace_env0_action_mean"] = np.asarray(trace_actions).mean(axis=0).tolist()
        summary["trace_env0_action_abs_p95"] = np.percentile(np.abs(np.asarray(trace_actions)), 95, axis=0).tolist()
        summary["trace_env0_displacement_m"] = float(np.linalg.norm(trace_xy[-1] - trace_xy[0]))
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    trace_xy = np.asarray(trace_xy)
    trace_actions = np.asarray(trace_actions)
    np.savez(
        output_dir / "trace_env0.npz",
        path_xy=initial_path[: initial_last + 1],
        executed_xy=trace_xy,
        actions=trace_actions,
    )
    origin = initial_path[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(initial_path[: initial_last + 1, 0] - origin[0], initial_path[: initial_last + 1, 1] - origin[1], "k--", label="reference path")
    axes[0].plot(trace_xy[:, 0] - origin[0], trace_xy[:, 1] - origin[1], "b", label="mean-action robot")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title(f"stage {stage}, success={summary['success_rate']:.1%}")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    if len(trace_actions):
        axes[1].plot(trace_actions[:, 0], label="joint1 action")
        axes[1].plot(trace_actions[:, 1], label="joint2 action")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel("normalized mean action")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_env0.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()





