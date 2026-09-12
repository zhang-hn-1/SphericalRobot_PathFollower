"""Separate longitudinal failure from lateral-offset failure on sharp S-curves.

Success requires three things at once: remaining arc length <= 0.20 m, straight
line distance to the final path sample <= 0.20 m, and speed <= 0.10 m/s.  On
s_curve at |k| = 0.40 the ball is measured spending a third of its steps inside
the last metre, moving at 0.1-0.19 m/s at full joint-2 tilt with 0.2-0.37 m of
cross-track.  That pattern is consistent with two very different stories:

* it never gets to the end of the path (a longitudinal failure), or
* it gets to the end of the path and stands 0.3 m to the side of the final
  sample, which the endpoint criterion counts as failure (a lateral-offset
  failure, and arguably a specification question rather than a controller bug).

This script classifies every terminating episode by which criterion it failed,
so the remaining gap can be attributed instead of guessed at.

    PATH_PORT_TYPE=s_curve PATH_PORT_CURVATURE=0.40 ./run_local_script.sh \
        legged_gym/scripts/attribute_path_failure.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

TYPE_LOOKUP = {"straight": 0, "left_arc": 1, "right_arc": 2, "s_curve": 3}
REASON_NAMES = {0: "success", 1: "timeout", 2: "deviation", 3: "unstable", 4: "out_of_bounds"}


def classify(remaining, endpoint, speed, limits):
    """Which acceptance criteria the episode met."""
    ok_len = remaining <= limits["remaining"]
    ok_end = endpoint <= limits["endpoint"]
    ok_speed = speed <= limits["speed"]
    if ok_len and ok_end and ok_speed:
        return "success (全部满足)"
    if ok_len and ok_speed and not ok_end:
        return "只差横向: 走完且停稳, 但离终点样本 > 阈值"
    if ok_len and ok_end and not ok_speed:
        return "只差末速"
    if not ok_len:
        return "纵向未走完 (剩余弧长 > 阈值)"
    return "多项不满足"


def main():
    args = get_args()
    seed = int(os.environ.get("PATH_PORT_SEED", "4200"))
    stage = int(os.environ.get("PATH_PORT_STAGE", "6"))
    type_name = os.environ.get("PATH_PORT_TYPE", "s_curve")
    curvature = float(os.environ.get("PATH_PORT_CURVATURE", "0.40"))
    episodes = int(os.environ.get("PATH_PORT_ENVS", "256"))

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg.env.num_envs = episodes
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)

    env.cfg.path.curriculum_enabled = False
    env.path_curriculum_stage = stage
    env.forced_path_type = TYPE_LOOKUP[type_name]
    env.forced_curvature = curvature
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    env.compute_observations()

    limits = {
        "remaining": float(env.cfg.path.success_remaining_length),
        "endpoint": float(env.cfg.path.success_endpoint_distance),
        "speed": float(env.cfg.path.success_speed),
    }
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    records = {}
    with torch.no_grad():
        for step in range(int(env.max_episode_length * 2)):
            done = env.step(zero)[3]
            for idx in torch.nonzero(done, as_tuple=False).flatten().tolist():
                if idx in records:
                    continue
                records[idx] = {
                    "remaining": float(env.terminal_path_remaining[idx].item()),
                    "endpoint": float(env.terminal_endpoint_distance[idx].item()),
                    "speed": float(env.terminal_speed[idx].item()),
                    "cross_track": float(env.terminal_cross_track[idx].item()),
                    "reason": REASON_NAMES[int(env.terminal_reason[idx].item())],
                    "success": int(env.terminal_success[idx].item()),
                }
            if len(records) >= env.num_envs:
                break

    rows = list(records.values())
    n = len(rows)
    print(f"{type_name} k={curvature:+.3f}  {n} episodes, stage {stage}")
    print(f"判据: 剩余 <= {limits['remaining']} m, 到终点样本 <= {limits['endpoint']} m, "
          f"末速 <= {limits['speed']} m/s\n")

    outcomes = Counter(classify(r["remaining"], r["endpoint"], r["speed"], limits)
                       for r in rows)
    print(f"{'归类':44}{'n':>6}{'占比':>8}")
    for name, count in outcomes.most_common():
        print(f"{name:44}{count:>6}{count / n:>8.1%}")
    reasons = Counter(r["reason"] for r in rows)
    print(f"\n环境给出的终止原因: {dict(reasons)}")

    offsets = np.array([r["cross_track"] for r in rows])
    endpoint = np.array([r["endpoint"] for r in rows])
    length_gap = np.array([r["remaining"] for r in rows])
    print(f"\n横向误差 |xtrack|:  中位数 {np.median(offsets):.3f} m  "
          f"p90 {np.percentile(offsets, 90):.3f} m")
    print(f"到终点直线距离:     中位数 {np.median(endpoint):.3f} m")
    print(f"剩余弧长:           中位数 {np.median(length_gap):.3f} m")

    # How much would the rate move if only the endpoint tolerance changed?
    print(f"\n只放宽『到终点距离』阈值会怎样（其余判据不动）:")
    len_ok = length_gap <= limits["remaining"]
    speed_ok = np.array([r["speed"] for r in rows]) <= limits["speed"]
    for tol in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        rate = float(np.mean(len_ok & speed_ok & (endpoint <= tol)))
        print(f"   tolerance {tol:.2f} m -> {rate:.1%}")

    # Failure-only statistics: the pooled medians hide which criterion binds.
    failed = [r for r in rows if not r["success"]]
    if failed:
        f_len = np.array([r["remaining"] for r in failed])
        f_end = np.array([r["endpoint"] for r in failed])
        f_speed = np.array([r["speed"] for r in failed])
        f_cross = np.array([r["cross_track"] for r in failed])
        print(f"\n失败样本 ({len(failed)} 个) 的判据分布:")
        print(f"  剩余弧长 <= {limits['remaining']}:  {np.mean(f_len <= limits['remaining']):6.1%}   "
              f"中位数 {np.median(f_len):.3f} m")
        print(f"  到终点   <= {limits['endpoint']}:  {np.mean(f_end <= limits['endpoint']):6.1%}   "
              f"中位数 {np.median(f_end):.3f} m")
        print(f"  末速     <= {limits['speed']}:  {np.mean(f_speed <= limits['speed']):6.1%}   "
              f"中位数 {np.median(f_speed):.3f} m/s")
        print(f"  |横向|   <= 1.50 m:  {np.mean(f_cross <= 1.50):6.1%}   "
              f"中位数 {np.median(f_cross):.3f} m")

    out = Path(os.environ.get(
        "PATH_PORT_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" /
            f"failure_attribution_{type_name}_k{curvature:+.3f}".replace(".", "p"))))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "path_type": type_name, "curvature": curvature, "episodes": n,
        "limits": limits, "outcomes": dict(outcomes), "reasons": dict(reasons),
        "cross_track_median_m": float(np.median(offsets)),
        "endpoint_median_m": float(np.median(endpoint)),
        "remaining_median_m": float(np.median(length_gap)),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
