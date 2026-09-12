"""Track real NeuPAN plan windows through the external-path interface.

This is the end-to-end check that the NeuPAN-facing contract works.  It loads the
144 windows exported from the official Ackermann reproduction
(``non_obs_acker_official_trajectories.npz``: 201 points at 0.05 m = 10 m per
window, in the frame of the robot pose at the start of the window) and installs
each one through ``RotunbotPath.set_external_path``.

An earlier, script-private injection of the same kind of paths scored 0.7%
success with a median endpoint distance of 14.4 m, i.e. the ball did not travel
at all.  That number is the baseline this script is meant to beat, and any
remaining gap is a frame/scale or controller issue rather than a plumbing
question.

    PATH_PATH_SOURCE=external ./run_local_script.sh \
        legged_gym/scripts/run_neupan_windows.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

REASON_NAMES = {0: "success", 1: "timeout", 2: "deviation", 3: "unstable", 4: "out_of_bounds"}


def load_windows(path, limit, min_curvature):
    """Return [n, M, 2] windows, optionally restricted to curved ones.

    The exported set is dominated by straight parking-garage segments (median
    peak |k| = 0.00), so testing only the first N windows exercises almost no
    turning.  Filtering by peak curvature is what makes the test meaningful.
    """
    data = np.load(path, allow_pickle=True)
    desired = np.asarray(data["desired"], dtype=float)
    indices = np.asarray(data["source_indices"])
    if min_curvature > 0.0:
        keep = []
        for i, window in enumerate(desired):
            delta = np.diff(window, axis=0)
            heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
            peak = np.abs(np.gradient(heading, 0.05)).max()
            if peak >= min_curvature:
                keep.append(i)
        desired = desired[keep]
        indices = indices[keep]
        print(f"curvature filter >= {min_curvature}: kept {len(keep)} windows")
    return desired[:limit], indices[:limit]


def main():
    args = get_args()
    seed = int(os.environ.get("PATH_NEUPAN_SEED", "4200"))
    windows_path = Path(os.environ.get(
        "PATH_NEUPAN_WINDOWS",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / "neupan_sstar_v1" /
            "non_obs_acker_official_trajectories.npz")))
    limit = int(os.environ.get("PATH_NEUPAN_ENVS", "144"))
    episode_length_s = float(os.environ.get("PATH_NEUPAN_EPISODE_S", "150"))
    min_curvature = float(os.environ.get("PATH_NEUPAN_MIN_CURVATURE", "0.0"))

    if not windows_path.is_file():
        raise SystemExit(f"windows not found: {windows_path}")
    if str(os.environ.get("PATH_PATH_SOURCE", "")) != "external":
        raise SystemExit("PATH_PATH_SOURCE=external is required so the internal "
                         "generator stands down")

    windows, indices = load_windows(windows_path, limit, min_curvature)
    n, points, _ = windows.shape
    print(f"loaded {n} windows x {points} points "
          f"({(points - 1) * 0.05:.1f} m at 0.05 m spacing)")

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    if str(env_cfg.path.path_source) != "external":
        raise SystemExit("config did not pick up PATH_PATH_SOURCE=external")
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg.env.num_envs = n
    env_cfg.env.episode_length_s = episode_length_s
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.path.out_of_bounds_radius = 1.0e4
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)

    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    env.compute_observations()

    # Window frames.  The exported windows all start at (0, 0) and run 10 m, but
    # their axes are NOT the robot's body axes: across the 144 windows the first
    # segment points anywhere from -140 to +141 degrees, and 118 of them differ
    # from the robot's heading by more than 30 degrees.  Installing them with a
    # translation alone makes the path run sideways relative to the ball, which
    # turns hard, exceeds the 1.5 m cross-track limit and terminates as
    # `deviation` - the same failure that scored 0.7% under the earlier
    # script-side injection.
    #
    # A follower needs one invariant at installation: the path's first segment
    # must be aligned with the robot's heading.  Rotate each window by its own
    # initial tangent, then place it at the robot with the robot's yaw.
    origins = env.root_states[:, :2].clone()
    q = env.root_states[:, 3:7]
    robot_yaw = torch.atan2(
        2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
        1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
    )
    local = torch.as_tensor(windows, dtype=env.path_xy.dtype, device=env.device)
    tangent = local[:, 1] - local[:, 0]
    first_heading = torch.atan2(tangent[:, 1], tangent[:, 0])
    angle = robot_yaw - first_heading
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    world = torch.empty_like(local)
    world[:, :, 0] = cos_a[:, None] * local[:, :, 0] - sin_a[:, None] * local[:, :, 1]
    world[:, :, 1] = sin_a[:, None] * local[:, :, 0] + cos_a[:, None] * local[:, :, 1]
    world = world + origins[:, None, :]

    installed = env.set_external_path(ids, world)
    env._update_path_state()
    env.compute_observations()
    obs = env.get_observations()
    print(f"installed; longest path {installed} samples "
          f"({(installed - 1) * 0.05:.2f} m), "
          f"median {float(torch.median(env.path_length)):.2f} m")

    records = {}
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    budget = int(env.max_episode_length * 2)
    with torch.no_grad():
        for step in range(budget):
            obs, _, _, dones, _ = env.step(zero)
            for idx in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                if idx in records:
                    continue
                records[idx] = {
                    "window": int(indices[idx]),
                    "success": int(env.terminal_success[idx].item()),
                    "reason": REASON_NAMES[int(env.terminal_reason[idx].item())],
                    "cross_track_m": float(env.terminal_cross_track[idx].item()),
                    "endpoint_m": float(env.terminal_endpoint_distance[idx].item()),
                    "speed_mps": float(env.terminal_speed[idx].item()),
                    "steps": step + 1,
                }
            if len(records) >= env.num_envs:
                break

    rows = list(records.values())
    success = np.array([r["success"] for r in rows], dtype=float)
    endpoints = np.array([r["endpoint_m"] for r in rows])
    cross = np.array([r["cross_track_m"] for r in rows])
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summary = {
        "windows": env.num_envs,
        "completed": len(rows),
        "episode_length_s": episode_length_s,
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "failure_reasons": reasons,
        "cross_track_median_m": float(np.median(cross)) if len(rows) else None,
        "endpoint_median_m": float(np.median(endpoints)) if len(rows) else None,
        "reference_from_earlier_injection": {
            "success_rate": 0.006944444444444444,
            "endpoint_median_m": 14.408471584320068,
            "cross_track_median_m": 0.7460684180259705,
        },
    }
    out = Path(os.environ.get(
        "PATH_NEUPAN_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / "neupan_windows_tracked")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "episodes.csv").open("w") as handle:
        handle.write("window,success,reason,cross_track_m,endpoint_m,speed_mps,steps\n")
        for r in sorted(rows, key=lambda r: r["window"]):
            handle.write(f"{r['window']},{r['success']},{r['reason']},"
                         f"{r['cross_track_m']},{r['endpoint_m']},{r['speed_mps']},{r['steps']}\n")

    print()
    print(f"success {summary['success_rate']:.1%}  ({int(success.sum())}/{len(rows)})")
    print(f"  failure reasons      {reasons}")
    print(f"  endpoint median      {summary['endpoint_median_m']:.2f} m  "
          f"(earlier injection: 14.41 m)")
    print(f"  cross-track median   {summary['cross_track_median_m']:.3f} m  "
          f"(earlier injection: 0.746 m)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
