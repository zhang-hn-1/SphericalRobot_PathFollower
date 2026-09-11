"""Trace what the analytic prior requests as it approaches the path endpoint.

The two obvious explanations for the stage-6 endgame stall have both been
refuted by measurement:

* the deceleration ramp is not the lever - sweeping ``prior_normal_stop_distance``
  over 0.45/0.90/2.40 gives 81.8%/81.6%/74.8% overall;
* there is no minimum rolling speed - an open-loop sweep
  (identify_minimum_rolling_speed.py) shows a strictly linear command-to-speed
  map down to 0.02 rad/s, i.e. 8 mm/s, with no dead band.

So the ball is able to creep arbitrarily slowly, yet failing episodes stop
0.43-0.52 m short at ~0.00 m/s.  That leaves the command itself: either the
prior requests zero too early, or it requests motion and the mechanism does not
deliver it in the endgame configuration (steering active, cross-track and
heading corrections loaded).

This script bins the last metres of an episode by remaining distance and reports
the commanded joint-1 target, the achieved forward speed, the cross-track and
heading errors, and the commanded joint-2 angle.  A command that collapses while
distance remains means the prior is stopping early; a command that stays up while
speed stays at zero means the mechanism cannot execute it in that configuration.

The prior is evaluated with a zero-initialised actor, so the action it produces
is exactly what the servo receives.

    PATH_TRACE_TYPE=left_arc PATH_TRACE_CURVATURE=0.25 \
        ./run_local_script.sh legged_gym/scripts/trace_prior_endgame.py
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

TYPE_LOOKUP = {"straight": 0, "left_arc": 1, "right_arc": 2, "s_curve": 3}
BIN_WIDTH = 0.10
MAX_DISTANCE = 3.0


def main():
    args = get_args()
    seed = int(os.environ.get("PATH_TRACE_SEED", "4200"))
    stage = int(os.environ.get("PATH_TRACE_STAGE", "6"))
    type_name = os.environ.get("PATH_TRACE_TYPE", "left_arc")
    curvature = float(os.environ.get("PATH_TRACE_CURVATURE", "0.25"))
    episodes = int(os.environ.get("PATH_TRACE_ENVS", "128"))

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    if not bool(getattr(env_cfg.control, "use_path_action_prior", False)):
        raise SystemExit("PATH_USE_ACTION_PRIOR=1 is required")
    if not bool(getattr(train_cfg.policy, "zero_init_actor_output", False)):
        raise SystemExit("PATH_ZERO_INIT_ACTOR=1 is required")

    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg.env.num_envs = episodes
    env_cfg.noise.add_noise = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)

    probe = env.get_observations()
    with torch.no_grad():
        if float(policy(probe).abs().max().item()) != 0.0:
            raise SystemExit("actor output is not zero; the action would not equal the prior")

    first_scale = float(env.cfg.control.first_actionScale)
    second_scale = float(env.cfg.control.second_actionScale)

    env.cfg.path.curriculum_enabled = False
    env.path_curriculum_stage = stage
    env.forced_path_type = TYPE_LOOKUP[type_name]
    env.forced_curvature = curvature
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()

    zero_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    samples = []
    # Episode ends when every environment has terminated; cap the run.
    step_budget = int(env.max_episode_length * 2)
    with torch.no_grad():
        for step in range(step_budget):
            # Read the prior's request before stepping: it is a pure function of
            # the current state, so this is exactly the command the servo gets.
            prior = env._analytic_action_prior()
            distance = env.path_endpoint_distance
            keep = distance < MAX_DISTANCE
            if bool(keep.any()):
                index = keep.nonzero(as_tuple=False).flatten()
                speed = torch.linalg.vector_norm(env.root_states[:, 7:9], dim=1)
                samples.append(torch.stack([
                    distance[index],
                    prior[index, 0] * first_scale,   # commanded joint-1 target, rad/s
                    prior[index, 1] * second_scale,  # commanded joint-2 angle, rad
                    speed[index],
                    env.path_cross_track[index].abs(),
                    env.path_heading_error[index].abs(),
                    env.path_remaining[index],
                ], dim=1).cpu().numpy())
            done = env.step(zero_action)[3]
            if bool(done.all()):
                break

    if not samples:
        raise SystemExit("no samples collected below the distance threshold")
    data = np.concatenate(samples, axis=0)
    columns = ["endpoint_distance_m", "commanded_joint1_rad_s", "commanded_joint2_rad",
               "speed_mps", "abs_cross_track_m", "abs_heading_error_rad", "remaining_m"]

    out = Path(os.environ.get(
        "PATH_TRACE_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" /
            f"prior_endgame_trace_{type_name}_k{curvature:+.3f}".replace(".", "p"))))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(data.tolist())

    print(f"{type_name} k={curvature:+.3f}  {episodes} episodes  {len(data)} samples")
    print(f"{'distance bin':>16} {'n':>7} {'cmd j1 rad/s':>13} {'speed m/s':>10} "
          f"{'j2 rad':>8} {'xtrack':>8} {'heading':>8}")
    rows = []
    edges = np.arange(0.0, MAX_DISTANCE + BIN_WIDTH, BIN_WIDTH)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (data[:, 0] >= low) & (data[:, 0] < high)
        if mask.sum() < 5:
            continue
        subset = data[mask]
        entry = {
            "distance_low_m": float(low), "distance_high_m": float(high),
            "samples": int(mask.sum()),
            "commanded_joint1_rad_s": float(np.median(subset[:, 1])),
            "speed_mps": float(np.median(subset[:, 3])),
            "commanded_joint2_rad": float(np.median(subset[:, 2])),
            "abs_cross_track_m": float(np.median(subset[:, 4])),
            "abs_heading_error_rad": float(np.median(subset[:, 5])),
        }
        rows.append(entry)
        print(f"{low:>7.1f}-{high:<8.1f} {entry['samples']:>7} "
              f"{entry['commanded_joint1_rad_s']:>13.4f} {entry['speed_mps']:>10.4f} "
              f"{entry['commanded_joint2_rad']:>8.3f} {entry['abs_cross_track_m']:>8.3f} "
              f"{entry['abs_heading_error_rad']:>8.3f}")

    (out / "bins.json").write_text(json.dumps({
        "path_type": type_name, "curvature": curvature, "episodes": episodes,
        "bin_width_m": BIN_WIDTH, "bins": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
