"""Evaluate the Rotunbot geometric follower on exported NeuPAN Acker S* windows.

Each planning step contains an 11-state local path [x, y, yaw].  The evaluator
removes repeated terminal states, resamples by arc length at the environment's
0.05 m spacing, rigidly aligns the first state with each simulated robot, and
runs exactly one episode per local path.  No NeuPAN speed or action is used.
"""

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


REASONS = {0: "success", 1: "timeout", 2: "deviation", 3: "unstable", 4: "out_of_bounds"}


def wrap_np(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def prepare_path(states, spacing, max_samples):
    """Return uniformly sampled local x/y/yaw/curvature and an audit record."""
    states = np.asarray(states, dtype=np.float64)
    finite = np.all(np.isfinite(states[:, :3]), axis=1)
    states = states[finite]
    if len(states) == 0:
        raise ValueError("S* window has no finite [x,y,yaw] states")

    # Repeated goal states are valid NeuPAN padding; remove zero-length segments.
    if len(states) > 1:
        segment = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
        keep = np.r_[True, segment > 1.0e-5]
        states = states[keep]

    x0, y0, yaw0 = states[0, :3]
    local_xy = states[:, :2] - np.array([x0, y0])
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    local_xy = np.column_stack(
        (c0 * local_xy[:, 0] + s0 * local_xy[:, 1],
         -s0 * local_xy[:, 0] + c0 * local_xy[:, 1])
    )
    local_yaw = np.unwrap(states[:, 2]) - yaw0

    if len(states) == 1:
        sample_xy = np.zeros((1, 2), dtype=np.float64)
        sample_yaw = np.zeros(1, dtype=np.float64)
        sample_s = np.zeros(1, dtype=np.float64)
    else:
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(local_xy, axis=0), axis=1))]
        # Keep the fixed 0.05 m convention used by the environment.  A residual
        # segment shorter than one sample is intentionally discarded.
        intervals = min(int(np.floor(cumulative[-1] / spacing + 1.0e-7)), max_samples - 1)
        sample_s = np.arange(intervals + 1, dtype=np.float64) * spacing
        sample_xy = np.column_stack(
            (np.interp(sample_s, cumulative, local_xy[:, 0]),
             np.interp(sample_s, cumulative, local_xy[:, 1]))
        )
        sample_yaw = np.interp(sample_s, cumulative, local_yaw)

    if len(sample_s) >= 3:
        curvature = np.gradient(sample_yaw, spacing, edge_order=2)
    elif len(sample_s) == 2:
        curvature = np.full(2, (sample_yaw[1] - sample_yaw[0]) / spacing)
    else:
        curvature = np.zeros(1, dtype=np.float64)
    curvature = np.clip(curvature, -2.0, 2.0)

    meaningful = curvature[np.abs(curvature) > 0.03]
    if len(meaningful) == 0:
        path_type = 0
    elif np.any(meaningful > 0) and np.any(meaningful < 0):
        path_type = 3
    elif np.median(meaningful) > 0:
        path_type = 1
    else:
        path_type = 2

    audit = {
        "raw_states": int(len(states)),
        "length_m": float(sample_s[-1]),
        "peak_abs_curvature_1pm": float(np.max(np.abs(curvature))),
        "path_type": int(path_type),
    }
    return sample_xy, sample_yaw, curvature, path_type, audit


def inject_paths(env, prepared):
    """Overwrite generated paths after reset, aligned to each environment root."""
    count = len(prepared)
    max_samples = env.path_xy.shape[1]
    spacing = float(env.cfg.path.sample_spacing)
    ids = torch.arange(count, device=env.device)
    q = env.root_states[ids, 3:7]
    root_yaw = torch.atan2(
        2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
        1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
    )

    env.path_xy[ids] = env.root_states[ids, None, :2]
    env.path_yaw[ids] = root_yaw[:, None]
    env.path_curvature[ids] = 0.0
    for idx, (xy, yaw, curvature, path_type, _) in enumerate(prepared):
        n = len(xy)
        c, s = torch.cos(root_yaw[idx]), torch.sin(root_yaw[idx])
        xy_t = torch.as_tensor(xy, device=env.device, dtype=env.path_xy.dtype)
        yaw_t = torch.as_tensor(yaw, device=env.device, dtype=env.path_yaw.dtype)
        k_t = torch.as_tensor(curvature, device=env.device, dtype=env.path_curvature.dtype)
        world = torch.empty_like(xy_t)
        world[:, 0] = env.root_states[idx, 0] + c * xy_t[:, 0] - s * xy_t[:, 1]
        world[:, 1] = env.root_states[idx, 1] + s * xy_t[:, 0] + c * xy_t[:, 1]
        env.path_xy[idx, :n] = world
        env.path_yaw[idx, :n] = root_yaw[idx] + yaw_t
        env.path_curvature[idx, :n] = k_t
        if n < max_samples:
            env.path_xy[idx, n:] = world[-1]
            env.path_yaw[idx, n:] = root_yaw[idx] + yaw_t[-1]
        env.path_last_index[idx] = n - 1
        env.path_length[idx] = (n - 1) * spacing
        env.path_type[idx] = path_type
        env.path_index[idx] = 0
        env.path_s[idx] = 0.0
        env.last_path_s[idx] = 0.0
    env._update_path_state()
    env.compute_observations()


def plot_examples(output, scenario, prepared, histories, records, source_indices):
    successes = [i for i, r in enumerate(records) if r and r["success"]]
    failures = [i for i, r in enumerate(records) if r and not r["success"]]
    selected = successes[:4] + failures[:4]
    if not selected:
        return
    cols = 4
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, idx in zip(axes.flat, selected):
        ax.set_visible(True)
        xy = prepared[idx][0]
        actual = np.asarray(histories[idx])
        ax.plot(xy[:, 0], xy[:, 1], "k--", lw=2, label="NeuPAN $S^*$")
        if len(actual):
            ax.plot(actual[:, 0], actual[:, 1], color="#1976d2", lw=2, label="Rotunbot")
            ax.scatter(actual[-1, 0], actual[-1, 1], c="#d32f2f", s=25)
        r = records[idx]
        ax.set_title(
            f"window {source_indices[idx]} | {'OK' if r['success'] else r['reason']}\n"
            f"L={r['length']:.2f} m, e={r['cross']:.2f} m, v={r['speed']:.2f} m/s"
        )
        ax.axis("equal")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"NeuPAN Acker local paths tracked by Rotunbot: {scenario}")
    fig.tight_layout()
    fig.savefig(output / f"{scenario}_examples.png", dpi=170)
    plt.close(fig)


def main():
    args = get_args()
    seed = int(os.environ.get("NEUPAN_EVAL_SEED", "7911"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    source = Path(os.environ["NEUPAN_SSTAR"])
    output = Path(os.environ.get("NEUPAN_EVAL_OUTPUT", str(PROJECT_ROOT / "artifacts/path_follower/neupan_sstar")))
    output.mkdir(parents=True, exist_ok=True)
    scenario = source.parent.name
    data = np.load(source)
    raw = np.asarray(data["s_star"])
    # Official NeuPAN exports [planning_step, state_dimension, horizon].
    # Accept the conventional [planning_step, horizon, state_dimension] too.
    if raw.ndim == 3 and raw.shape[1] in (3, 4, 5) and raw.shape[2] > raw.shape[1]:
        raw = np.transpose(raw, (0, 2, 1))
    if raw.ndim != 3 or raw.shape[1] < 2 or raw.shape[2] < 3:
        raise ValueError(f"Expected s_star [N,H,>=3], got {raw.shape}")

    spacing = 0.05
    max_samples = 201
    prepared_all = [prepare_path(window[:, :3], spacing, max_samples) for window in raw]
    # Empty/sub-0.20 m goal padding is reported by the audit but omitted from the
    # dynamic test because it requires no path-following action.
    source_indices = [i for i, p in enumerate(prepared_all) if p[4]["length_m"] >= 0.20]
    max_windows = int(os.environ.get("NEUPAN_EVAL_MAX_WINDOWS", "0"))
    if max_windows > 0 and len(source_indices) > max_windows:
        pick = np.linspace(0, len(source_indices) - 1, max_windows).round().astype(int)
        source_indices = [source_indices[i] for i in pick]
    prepared = [prepared_all[i] for i in source_indices]

    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs = len(prepared)
    env_cfg.env.episode_length_s = float(os.environ.get("NEUPAN_EVAL_EPISODE_LENGTH_S", "80.0"))
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    args.seed = seed
    env_cfg.seed = seed
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    inject_paths(env, prepared)

    records = [None] * len(prepared)
    histories = [[] for _ in prepared]
    active = np.ones(len(prepared), dtype=bool)
    root0 = env.root_states[:, :2].detach().cpu().numpy().copy()
    q0 = env.root_states[:, 3:7]
    yaw0 = torch.atan2(
        2.0 * (q0[:, 3] * q0[:, 2] + q0[:, 0] * q0[:, 1]),
        1.0 - 2.0 * (q0[:, 1].square() + q0[:, 2].square()),
    ).detach().cpu().numpy()

    with torch.no_grad():
        for step in range(int(env.max_episode_length) * 4):
            world = env.root_states[:, :2].detach().cpu().numpy()
            delta = world - root0
            c, s = np.cos(yaw0), np.sin(yaw0)
            local = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]))
            for idx in np.flatnonzero(active):
                histories[idx].append(local[idx].copy())
            actions = env._analytic_action_prior()
            _, _, _, dones, _ = env.step(actions)
            for idx in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                if not active[idx]:
                    continue
                records[idx] = {
                    "source_window": int(source_indices[idx]),
                    "success": int(env.terminal_success[idx]),
                    "reason": REASONS[int(env.terminal_reason[idx])],
                    "length": float(env.terminal_path_length[idx]),
                    "cross": float(env.terminal_cross_track[idx]),
                    "endpoint": float(env.terminal_endpoint_distance[idx]),
                    "speed": float(env.terminal_speed[idx]),
                    "steps": int(step + 1),
                    **prepared[idx][4],
                }
                active[idx] = False
            if not np.any(active):
                break

    # Make incomplete episodes explicit instead of silently dropping them.
    for idx in np.flatnonzero(active):
        records[idx] = {
            "source_window": int(source_indices[idx]),
            "success": 0,
            "reason": "evaluator_limit",
            "length": float(prepared[idx][4]["length_m"]),
            "cross": float(abs(env.path_cross_track[idx])),
            "endpoint": float(env.path_endpoint_distance[idx]),
            "speed": float(torch.linalg.vector_norm(env.root_states[idx, 7:9])),
            "steps": int(env.max_episode_length) * 4,
            **prepared[idx][4],
        }

    success = np.asarray([r["success"] for r in records], dtype=np.float64)
    summary = {
        "scenario": scenario,
        "source": str(source),
        "raw_windows": int(len(raw)),
        "evaluated_windows": int(len(records)),
        "omitted_sub_0p20m_windows": int(len(raw) - len(source_indices)),
        "success_rate": float(success.mean()),
        "success_count": int(success.sum()),
        "failure_reasons": dict(Counter(r["reason"] for r in records)),
        "cross_track_median_m": float(np.median([r["cross"] for r in records])),
        "endpoint_distance_median_m": float(np.median([r["endpoint"] for r in records])),
        "terminal_speed_median_mps": float(np.median([r["speed"] for r in records])),
        "path_length_range_m": [float(min(r["length_m"] for r in records)), float(max(r["length_m"] for r in records))],
        "peak_abs_curvature_max_1pm": float(max(r["peak_abs_curvature_1pm"] for r in records)),
        "path_type_counts": dict(Counter(str(r["path_type"]) for r in records)),
    }
    (output / f"{scenario}_metrics.json").write_text(json.dumps(summary, indent=2))
    (output / f"{scenario}_episodes.json").write_text(json.dumps(records, indent=2))
    np.savez_compressed(
        output / f"{scenario}_trajectories.npz",
        source_indices=np.asarray(source_indices),
        desired=np.asarray([p[0] for p in prepared], dtype=object),
        actual=np.asarray([np.asarray(h) for h in histories], dtype=object),
    )
    plot_examples(output, scenario, prepared, histories, records, source_indices)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



