"""Track complete NeuPAN Acker executed paths with rolling 10 m path windows."""
import json
import math
import os
import sys
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


def load_path(csv_path, spacing=0.05):
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    states = np.column_stack((raw["x"], raw["y"], raw["yaw"])).astype(np.float64)
    finite = np.all(np.isfinite(states), axis=1)
    states = states[finite]
    segment = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
    states = states[np.r_[True, segment > 1.0e-5]]
    origin = states[0, :2].copy()
    yaw0 = float(states[0, 2])
    delta = states[:, :2] - origin
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    xy = np.column_stack((c0 * delta[:, 0] + s0 * delta[:, 1],
                          -s0 * delta[:, 0] + c0 * delta[:, 1]))
    yaw = np.unwrap(states[:, 2]) - yaw0
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    intervals = int(np.floor(cumulative[-1] / spacing + 1.0e-7))
    sample_s = np.arange(intervals + 1) * spacing
    xy_s = np.column_stack((np.interp(sample_s, cumulative, xy[:, 0]),
                            np.interp(sample_s, cumulative, xy[:, 1])))
    yaw_s = np.interp(sample_s, cumulative, yaw)
    curvature = np.gradient(yaw_s, spacing, edge_order=2)
    curvature = np.clip(curvature, -2.0, 2.0)
    return {"xy": xy_s, "yaw": yaw_s, "curvature": curvature,
            "length": float(sample_s[-1]), "source_points": int(len(states)),
            "peak_curvature": float(np.max(np.abs(curvature)))}


def classify(curvature):
    meaningful = curvature[np.abs(curvature) > 0.03]
    if len(meaningful) == 0:
        return 0
    if np.any(meaningful > 0) and np.any(meaningful < 0):
        return 3
    return 1 if np.median(meaningful) > 0 else 2


def world_from_local(xy, root_xy, root_yaw):
    c, s = math.cos(root_yaw), math.sin(root_yaw)
    return np.column_stack((root_xy[0] + c * xy[:, 0] - s * xy[:, 1],
                            root_xy[1] + s * xy[:, 0] + c * xy[:, 1]))


def local_from_world(xy, root_xy, root_yaw):
    delta = xy - root_xy
    c, s = math.cos(root_yaw), math.sin(root_yaw)
    return np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]))


def inject_segment(env, idx, path, start, root_xy, root_yaw):
    max_samples = env.path_xy.shape[1]
    end = min(start + max_samples, len(path["xy"]))
    xy = path["xy"][start:end]
    yaw = path["yaw"][start:end]
    curvature = path["curvature"][start:end]
    world = world_from_local(xy, root_xy, root_yaw)
    n = len(xy)
    xy_t = torch.as_tensor(world, device=env.device, dtype=env.path_xy.dtype)
    yaw_t = torch.as_tensor(root_yaw + yaw, device=env.device, dtype=env.path_yaw.dtype)
    k_t = torch.as_tensor(curvature, device=env.device, dtype=env.path_curvature.dtype)
    env.path_xy[idx] = xy_t[-1]
    env.path_yaw[idx] = yaw_t[-1]
    env.path_curvature[idx] = 0.0
    env.path_xy[idx, :n] = xy_t
    env.path_yaw[idx, :n] = yaw_t
    env.path_curvature[idx, :n] = k_t
    env.path_last_index[idx] = n - 1
    env.path_length[idx] = (n - 1) * float(env.cfg.path.sample_spacing)
    env.path_type[idx] = classify(curvature)
    env.path_index[idx] = 0
    env.path_s[idx] = 0.0
    env.last_path_s[idx] = 0.0
    return end


def nearest_forward(path_xy, point, cursor, back=10, forward=120):
    lo = max(0, cursor - back)
    hi = min(len(path_xy), cursor + forward + 1)
    distance2 = np.sum((path_xy[lo:hi] - point) ** 2, axis=1)
    return max(cursor, lo + int(np.argmin(distance2)))


def main():
    args = get_args()
    seed = int(os.environ.get("NEUPAN_GLOBAL_SEED", "7921"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    source_root = Path(os.environ.get(
        "NEUPAN_OUTPUT_ROOT", "/data/lzq/workspace/NeuPAN_official_repro_20260910/outputs"))
    output = Path(os.environ.get(
        "NEUPAN_GLOBAL_OUTPUT", str(PROJECT_ROOT / "artifacts/path_follower/neupan_global_paths")))
    output.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(source_root.glob("*_acker_official/executed_path.csv"))
    selected = {x.strip() for x in os.environ.get("NEUPAN_GLOBAL_SCENARIOS", "").split(",") if x.strip()}
    if selected:
        csv_paths = [path for path in csv_paths if path.parent.name in selected]
    if not csv_paths:
        raise FileNotFoundError(f"No *_acker_official/executed_path.csv under {source_root}")
    paths = [load_path(path) for path in csv_paths]
    names = [path.parent.name for path in csv_paths]

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs = len(paths)
    env_cfg.env.episode_length_s = float(os.environ.get("NEUPAN_GLOBAL_EPISODE_LENGTH_S", "400.0"))
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.path.out_of_bounds_radius = 100.0
    env_cfg.path.prior_use_pure_pursuit = os.environ.get("NEUPAN_GLOBAL_CONTROLLER", "") == "pure_pursuit"
    env_cfg.path.prior_use_local_curvature_schedule = os.environ.get("NEUPAN_GLOBAL_CONTROLLER", "") == "local_schedule"
    env_cfg.path.prior_pp_lookahead = float(os.environ.get("NEUPAN_PP_LOOKAHEAD", "1.20"))
    env_cfg.path.prior_pp_min_drive = float(os.environ.get("NEUPAN_PP_MIN_DRIVE", "0.22"))
    env_cfg.path.prior_pp_max_drive = float(os.environ.get("NEUPAN_PP_MAX_DRIVE", "0.48"))
    env_cfg.path.prior_pp_max_curvature = float(os.environ.get("NEUPAN_PP_MAX_CURVATURE", "0.50"))
    args.seed = seed
    env_cfg.seed = seed
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    checkpoint = os.environ.get("NEUPAN_GLOBAL_CHECKPOINT")
    policy = None
    if checkpoint:
        runner, _ = task_registry.make_alg_runner(
            env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None
        )
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)

    root0 = env.root_states[:, :2].detach().cpu().numpy().copy()
    q0 = env.root_states[:, 3:7]
    yaw0 = torch.atan2(2.0 * (q0[:, 3] * q0[:, 2] + q0[:, 0] * q0[:, 1]),
                       1.0 - 2.0 * (q0[:, 1].square() + q0[:, 2].square())).detach().cpu().numpy()
    cursors = np.zeros(len(paths), dtype=np.int64)
    segment_starts = np.zeros(len(paths), dtype=np.int64)
    segment_ends = np.zeros(len(paths), dtype=np.int64)
    for idx, path in enumerate(paths):
        segment_ends[idx] = inject_segment(env, idx, path, 0, root0[idx], yaw0[idx])
    env._update_path_state()
    env.compute_observations()

    active = np.ones(len(paths), dtype=bool)
    records = [None] * len(paths)
    history = [[] for _ in paths]
    cross_history = [[] for _ in paths]
    reload_stride = int(round(float(os.environ.get("NEUPAN_GLOBAL_RELOAD_M", "1.0")) / 0.05))

    with torch.no_grad():
        for step in range(int(env.max_episode_length) + 1):
            world = env.root_states[:, :2].detach().cpu().numpy()
            local = np.vstack([local_from_world(world[i:i+1], root0[i], yaw0[i])[0]
                               for i in range(len(paths))])
            changed = False
            for idx in np.flatnonzero(active):
                cursors[idx] = nearest_forward(paths[idx]["xy"], local[idx], int(cursors[idx]))
                normal = np.array([-math.sin(paths[idx]["yaw"][cursors[idx]]),
                                   math.cos(paths[idx]["yaw"][cursors[idx]])])
                cross = float(np.dot(local[idx] - paths[idx]["xy"][cursors[idx]], normal))
                cross_history[idx].append(abs(cross))
                if step % 10 == 0:
                    history[idx].append(local[idx].copy())
                remaining_after_segment = len(paths[idx]["xy"]) - segment_ends[idx]
                if (cursors[idx] - segment_starts[idx] >= reload_stride
                        and remaining_after_segment > 0):
                    segment_starts[idx] = max(0, int(cursors[idx]) - 5)
                    segment_ends[idx] = inject_segment(env, idx, paths[idx], int(segment_starts[idx]),
                                                       root0[idx], yaw0[idx])
                    changed = True
            if changed:
                env._update_path_state()
                env.compute_observations()

            actions = policy(env.get_observations()) if policy is not None else env._analytic_action_prior()
            _, _, _, dones, _ = env.step(actions)
            for idx in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                if not active[idx]:
                    continue
                terminal_world = env.terminal_position[idx].detach().cpu().numpy()[None, :]
                terminal_local = local_from_world(terminal_world, root0[idx], yaw0[idx])[0]
                cursors[idx] = nearest_forward(paths[idx]["xy"], terminal_local, int(cursors[idx]))
                endpoint = float(np.linalg.norm(terminal_local - paths[idx]["xy"][-1]))
                records[idx] = {
                    "scenario": names[idx],
                    "success": int(env.terminal_success[idx]),
                    "reason": REASONS[int(env.terminal_reason[idx])],
                    "global_length_m": paths[idx]["length"],
                    "progress_m": float(cursors[idx] * 0.05),
                    "progress_fraction": float(cursors[idx] / max(len(paths[idx]["xy"]) - 1, 1)),
                    "endpoint_distance_m": endpoint,
                    "terminal_speed_mps": float(env.terminal_speed[idx]),
                    "cross_track_median_m": float(np.median(cross_history[idx])),
                    "cross_track_p95_m": float(np.quantile(cross_history[idx], 0.95)),
                    "cross_track_max_m": float(np.max(cross_history[idx])),
                    "peak_abs_curvature_1pm": paths[idx]["peak_curvature"],
                    "steps": int(step + 1),
                }
                active[idx] = False
            if not np.any(active):
                break

    for idx in np.flatnonzero(active):
        world = env.root_states[idx, :2].detach().cpu().numpy()[None, :]
        terminal_local = local_from_world(world, root0[idx], yaw0[idx])[0]
        records[idx] = {
            "scenario": names[idx], "success": 0, "reason": "evaluator_limit",
            "global_length_m": paths[idx]["length"], "progress_m": float(cursors[idx] * 0.05),
            "progress_fraction": float(cursors[idx] / max(len(paths[idx]["xy"]) - 1, 1)),
            "endpoint_distance_m": float(np.linalg.norm(terminal_local - paths[idx]["xy"][-1])),
            "terminal_speed_mps": float(torch.linalg.vector_norm(env.root_states[idx, 7:9])),
            "cross_track_median_m": float(np.median(cross_history[idx])),
            "cross_track_p95_m": float(np.quantile(cross_history[idx], 0.95)),
            "cross_track_max_m": float(np.max(cross_history[idx])),
            "peak_abs_curvature_1pm": paths[idx]["peak_curvature"], "steps": int(env.max_episode_length),
        }

    summary = {
        "evaluation": "continuous executed-car path with rolling 10 m geometric windows",
        "checkpoint": checkpoint,
        "speed_reference_used": False,
        "yaw_at_goal_required": False,
        "success_endpoint_m": float(env.cfg.path.success_endpoint_distance),
        "success_speed_mps": float(env.cfg.path.success_speed),
        "success_count": int(sum(r["success"] for r in records)),
        "path_count": len(records),
        "results": records,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output / "trajectories.npz", names=np.asarray(names),
                        desired=np.asarray([p["xy"] for p in paths], dtype=object),
                        actual=np.asarray([np.asarray(h) for h in history], dtype=object))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, name, path, actual, record in zip(axes.flat, names, paths, history, records):
        ax.set_visible(True)
        actual = np.asarray(actual)
        ax.plot(path["xy"][:, 0], path["xy"][:, 1], "k--", lw=1.6, label="NeuPAN car path")
        if len(actual):
            ax.plot(actual[:, 0], actual[:, 1], color="#1976d2", lw=1.5, label="Rotunbot")
        ax.set_title(f"{name}: {'OK' if record['success'] else record['reason']}\n"
                     f"progress={record['progress_fraction']:.1%}, p95={record['cross_track_p95_m']:.2f} m")
        ax.axis("equal")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Rotunbot tracking complete NeuPAN Acker executed paths (no speed reference)")
    fig.tight_layout()
    fig.savefig(output / "trajectory_overview.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()




