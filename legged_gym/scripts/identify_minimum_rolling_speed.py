"""Measure the smallest joint-1 command that produces sustained rolling.

Why
---

Stage-6 failures driven by the analytic prior are an endgame stall: episode
records show the ball stopping a median of 0.43-0.52 m short of the path end
with terminal speed ~0.00 m/s, then waiting out the 40 s budget.  Cross-track
error on the successful episodes is only 0.03-0.07 m, so tracking is fine and
the failure is longitudinal.

The prior commands a drive speed that ramps linearly to zero as the remaining
distance goes to zero (``desired = cruise * clamp(d / stop_distance, 0, 1)``),
converted to a joint-1 target velocity.  If that requested target falls below
the smallest command the mechanism can convert into rolling, the ball stops
early and can never close the last half metre.  Sweeping the deceleration ramp
did not help (0.45 -> 81.8%, 0.90 -> 81.6%, 2.40 -> 74.8%), which is consistent
with a hard floor rather than a tuning problem.

What this measures
------------------

For a grid of constant joint-1 target velocities, held for several seconds with
joint 2 at a fixed angle, the steady-state travel speed.  The interesting output
is the smallest command whose steady speed is clearly non-zero, and the shape of
the command-to-speed map near that threshold.  Compare the result with the
joint-1 targets the prior actually issues in the last metre before the endpoint.

Open loop: the analytic prior is off (PATH_USE_ACTION_PRIOR unset), so the
action passed in is the command the servo receives.  Output is written as JSON
next to the CSV so later analysis does not have to re-run the simulation.

    ./run_local_script.sh legged_gym/scripts/identify_minimum_rolling_speed.py
"""

import csv
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

# Joint-1 targets in rad/s, concentrated where the current prior operates late in
# an episode (drive 0.25 and 0.45 map to roughly 0.3 and 0.53 rad/s of cruise,
# and the endgame ramp divides that by the remaining-distance fraction).
FIRST_TARGETS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.55, 0.80]
# Joint-2 angles in rad; 0 is straight, the others load the mechanism.  Override
# with PATH_ID_ANGLES to probe whether a large tilt blocks propulsion, which is
# what the S-curve endgame looks like (|joint-2 command| pinned at its limit).
SECOND_ANGLES = [
    float(value) for value in
    os.environ.get("PATH_ID_ANGLES", "0.0,0.30,-0.30").split(",")
]
REPEATS = 4


def main():
    args = get_args()
    seed = int(os.environ.get("PATH_ID_SEED", "7101"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    first_scale = float(env_cfg.control.first_actionScale)
    second_scale = float(env_cfg.control.second_actionScale)

    pairs = [(a1, a2) for a1 in FIRST_TARGETS for a2 in SECOND_ANGLES for _ in range(REPEATS)]
    env_cfg.env.num_envs = len(pairs)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.path.curriculum_enabled = False
    # Never terminate: every environment must run the full window so the
    # steady-state speed is measured over the same interval for every command.
    env_cfg.path.success_remaining_length = -1.0
    env_cfg.path.success_endpoint_distance = -1.0
    env_cfg.path.deviation_termination = 1.0e6
    env_cfg.path.unstable_gravity_z = 2.0
    env_cfg.env.episode_length_s = 30.0
    args.seed = seed
    env_cfg.seed = seed
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)

    # Actions are normalised; the servo receives action * scale.
    actions = torch.tensor(
        [[a1 / first_scale, a2 / second_scale] for a1, a2 in pairs],
        dtype=torch.float, device=env.device,
    )

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()

    warmup_steps = int(os.environ.get("PATH_ID_WARMUP", "150"))
    measure_steps = int(os.environ.get("PATH_ID_MEASURE", "500"))
    trace = []
    with torch.no_grad():
        for step in range(warmup_steps + measure_steps):
            env.step(actions)
            if step >= warmup_steps:
                trace.append(env.root_states[:, :2].detach().cpu().numpy().copy())

    xy = np.stack(trace, axis=0)  # [steps, envs, 2]
    window_seconds = measure_steps * env.dt
    displacement = xy[-1] - xy[0]
    travel = np.linalg.norm(displacement, axis=-1)
    steady_speed = travel / window_seconds
    # Mean speed over the window, which also catches a run that moves late.
    step_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    mean_speed = step_lengths.sum(axis=0) / window_seconds

    rows = []
    for index, (a1, a2) in enumerate(pairs):
        rows.append({
            "joint1_target_rad_s": a1,
            "joint2_angle_rad": a2,
            "net_displacement_m": float(travel[index]),
            "steady_speed_mps": float(steady_speed[index]),
            "mean_speed_mps": float(mean_speed[index]),
        })

    out = Path(os.environ.get(
        "PATH_ID_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / "minimum_rolling_speed")))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate per (joint-1 target, joint-2 angle): median steady speed.
    summary = {"dt": env.dt, "window_seconds": window_seconds, "repeats": REPEATS, "grid": []}
    print(f"{'joint1 rad/s':>12} {'joint2 rad':>11} {'median steady m/s':>18} {'min':>8} {'max':>8}")
    for a1 in FIRST_TARGETS:
        for a2 in SECOND_ANGLES:
            subset = [r for r in rows
                      if abs(r["joint1_target_rad_s"] - a1) < 1e-9
                      and abs(r["joint2_angle_rad"] - a2) < 1e-9]
            speeds = np.array([r["steady_speed_mps"] for r in subset])
            entry = {
                "joint1_target_rad_s": a1,
                "joint2_angle_rad": a2,
                "steady_speed_median_mps": float(np.median(speeds)),
                "steady_speed_min_mps": float(speeds.min()),
                "steady_speed_max_mps": float(speeds.max()),
                "net_displacement_median_m": float(np.median(
                    [r["net_displacement_m"] for r in subset])),
                "episodes": len(subset),
            }
            summary["grid"].append(entry)
            print(f"{a1:>12.3f} {a2:>11.2f} {entry['steady_speed_median_mps']:>18.4f} "
                  f"{entry['steady_speed_min_mps']:>8.4f} {entry['steady_speed_max_mps']:>8.4f}")

    # The smallest command whose median steady speed clears 1 cm/s, per angle.
    summary["threshold"] = {}
    for a2 in SECOND_ANGLES:
        moving = [e for e in summary["grid"]
                  if e["joint2_angle_rad"] == a2 and e["steady_speed_median_mps"] > 0.01]
        summary["threshold"][f"joint2_{a2:+.2f}"] = (
            min(e["joint1_target_rad_s"] for e in moving) if moving else None)
    print()
    print("smallest joint-1 target with median steady speed > 0.01 m/s:")
    for key, value in summary["threshold"].items():
        print(f"  {key}: {value if value is not None else 'none on the grid'}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'summary.json'} and {out / 'samples.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
