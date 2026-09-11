"""Sweep the analytic path-following prior inside a single IsaacGym process.

Why this exists
---------------

The learned residual was measured to contribute nothing over 125 PPO updates,
so the analytic prior *is* the policy.  The prior is a deterministic controller
whose parameters live in ``cfg.path.*`` and are re-read on every step, so many
configurations can be evaluated in one process.  Rebuilding the environment per
configuration costs ~40 s of IsaacGym start-up; this harness pays it once.

Design decisions that matter
----------------------------

* **Paired evaluation.**  The RNG is reseeded before every rollout, so each
  configuration sees byte-identical paths and identical initial poses.  The
  comparison is paired, which is what makes 256-512 episodes enough to resolve
  differences of a point or two.  Without it the sampled curvature mixture moves
  the headline number by ~15 points and swamps the effect being searched for.
* **One episode per environment, from the initial reset.**  Only the first
  terminating episode of each environment is recorded, so every episode comes
  from the same reset and therefore the same path set.  Letting environments
  recycle would let fast configurations contribute more episodes than slow ones.
* **No checkpoint.**  ``PATH_ZERO_INIT_ACTOR=1`` zeroes both weight and bias of
  every expert's output layer at construction, so the policy output is exactly
  zero and the action equals the prior.  No weights need to be loaded.
* **The objective is the worst bucket, not the average.**  Success is aggregated
  per (path type, curvature); the score of a configuration is the minimum over
  buckets that have at least ``--min-bucket-n`` episodes.  Optimising the
  average would reward a configuration that is excellent on the curvatures the
  seed happened to draw.

Usage
-----

Every sweep setting is an environment variable, matching the convention of the
rest of this project; command-line flags are left to `get_args()`.

    PATH_SWEEP_KIND=baseline ./run_local_sweep.sh --num_envs 256
    PATH_SWEEP_KIND=screen   ./run_local_sweep.sh --num_envs 512
    PATH_SWEEP_CONFIGS=my_configs.json PATH_SWEEP_LABEL=my_sweep ./run_local_sweep.sh

    PATH_SWEEP_KIND                       baseline | screen       (default baseline)
    PATH_SWEEP_CONFIGS                    JSON file: [{"param": value}, ...]
    PATH_SWEEP_LABEL / PATH_SWEEP_OUTPUT  where results.jsonl is written
    PATH_SWEEP_DRY_RUN=1                  print the schedule, build nothing
    PATH_SWEEP_MIN_BUCKET_N               buckets smaller than this are ignored
                                          when scoring the worst case (default 8)
    PATH_SWEEP_FORCED_TYPE                int 0-3, force one path type
    PATH_SWEEP_FORCED_CURVATURE           force one signed curvature
    PATH_EVAL_STAGE / PATH_EVAL_SEED      stage and RNG seed (defaults 6 / 4200)
    PATH_SWEEP_MAX_WALL                   per-configuration wall clock guard, s

Results are appended to ``<out>/results.jsonl`` one configuration per line, so
an interrupted sweep resumes by skipping configurations already present.
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: E402  must precede torch

import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

TYPE_NAMES = {0: "straight", 1: "left_arc", 2: "right_arc", 3: "s_curve"}
REASON_NAMES = {0: "success", 1: "timeout", 2: "deviation", 3: "unstable", 4: "out_of_bounds"}

# How many of the worst (path type x curvature) buckets the smoothed objective
# averages.  A single worst bucket is dominated by whichever bucket happens to
# contain the fewest episodes.
SMOOTH_BUCKETS = 3

# Single-variable perturbations for the first screening pass.  Steps are chosen
# to be large enough to move the behaviour but small enough to stay physical.
SCREEN_STEPS = {
    "prior_normal_drive": 0.10,
    "prior_normal_stop_distance": 0.30,
    "prior_normal_cross_track_kp": 0.10,
    "prior_normal_heading_kp": 0.50,
    "prior_tight_drive": 0.10,
    "prior_tight_stop_distance": 0.60,
    "prior_tight_cross_track_kp": 0.10,
    "prior_tight_heading_kp": 0.50,
    "prior_speed_kp": 0.20,
    "prior_gain_offset": 0.03,
    "prior_gain_slope": 0.20,
    "prior_gain_min": 0.06,
    "prior_gain_max": 0.10,
    "prior_tight_curvature": 0.05,
    "prior_r2_curvature": 0.05,
    # Endgame homing.  Only useful in combination with prior_endpoint_floor (which
    # keeps the drive alive once the arc-length remaining saturates), and the
    # blend distance has a sharp optimum: 0.25 m gives 41.5% on the worst bucket,
    # 0.50 m and 1.00 m cut the path badly and drop the overall rate to 47%.
    "prior_endpoint_blend_distance": 0.05,
    "prior_endpoint_max_curvature": 0.10,
    # S-curve regime.  These were missing from the schedule entirely, which is
    # exactly why the search stalled with s_curve at |k|=0.4 as the worst bucket:
    # its failures are `deviation`, and the gain that resists lateral drift is
    # prior_s_*_cross_track_kp, which the search could not touch.  Note the
    # shipped value for the long-curve heading gain is zero.
    "prior_s_cross_track_kp": 0.05,
    "prior_s_heading_kp": 0.30,
    "prior_s_mid_cross_track_kp": 0.05,
    "prior_s_mid_heading_kp": 0.30,
    "prior_s_mid_drive": 0.05,
    "prior_s_mid_stop_distance": 0.30,
    "prior_s_short_drive": 0.05,
    "prior_s_long_drive": 0.05,
    "prior_s_tight_short_cross_track_kp": 0.05,
    "prior_s_tight_short_heading_kp": 0.20,
    "prior_s_tight_long_cross_track_kp": 0.05,
    "prior_s_tight_long_heading_kp": 0.20,
    "prior_s_regular_long_cross_track_kp": 0.05,
    "prior_s_regular_long_heading_kp": 0.20,
    "prior_s_lookahead": 0.20,
}


def base_defaults(env_cfg=None):
    """Current values of the swept parameters, read from a live config."""
    if env_cfg is None:
        env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    return {name: float(getattr(env_cfg.path, name)) for name in SCREEN_STEPS}


def build_schedule(kind, defaults):
    """Return a list of (label, overrides) pairs."""
    schedule = [("baseline", {})]
    if kind == "baseline":
        return schedule
    if kind == "screen":
        for name, step in SCREEN_STEPS.items():
            for sign, tag in ((+1, "plus"), (-1, "minus")):
                value = defaults[name] + sign * step
                if name in ("prior_gain_min",) and value <= 0:
                    continue
                if name in ("prior_tight_curvature", "prior_r2_curvature") and not 0 < value < 1:
                    continue
                schedule.append((f"{name}_{tag}", {name: value}))
        return schedule
    raise SystemExit(f"unknown sweep kind: {kind}")


def load_schedule(path):
    configs = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(entry.get("label", f"config_{i}"), {k: v for k, v in entry.items() if k != "label"})
            for i, entry in enumerate(configs)]


def apply_overrides(env, overrides, defaults):
    """Set prior parameters on the live config, restoring untouched defaults."""
    for name, value in defaults.items():
        setattr(env.cfg.path, name, value)
    for name, value in overrides.items():
        if not hasattr(env.cfg.path, name):
            raise KeyError(f"cfg.path has no parameter {name!r}")
        setattr(env.cfg.path, name, value)


def rollout(env, policy, args):
    """One paired rollout: one episode from each environment, same paths always."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.cfg.path.curriculum_enabled = False
    env.path_curriculum_stage = args.stage
    env.forced_path_type = args.forced_type
    env.forced_curvature = args.forced_curvature

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()
    obs = env.get_observations()

    records = {}
    steps = 0
    step_budget = int(env.max_episode_length * 4)
    wall_start = time.time()
    # no_grad rather than inference_mode: this harness reuses one environment for
    # many rollouts, and tensors created inside inference_mode cannot be updated
    # in place afterwards, which the environment's reset path does.
    with torch.no_grad():
        while len(records) < env.num_envs and steps < step_budget:
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            steps += 1
            done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
            for idx in done_ids:
                if idx in records:
                    continue
                records[idx] = {
                    "path_type": TYPE_NAMES[int(env.terminal_path_type[idx].item())],
                    "curvature": float(env.terminal_path_curvature[idx].item()),
                    "success": int(env.terminal_success[idx].item()),
                    "reason": REASON_NAMES[int(env.terminal_reason[idx].item())],
                    "endpoint": float(env.terminal_endpoint_distance[idx].item()),
                    "speed": float(env.terminal_speed[idx].item()),
                    "cross_track": float(env.terminal_cross_track[idx].item()),
                    "path_length": float(env.terminal_path_length[idx].item()),
                }
            if time.time() - wall_start > args.max_wall_seconds:
                break
    return list(records.values()), steps


def summarise(records, min_bucket_n):
    buckets = {}
    for row in records:
        key = (row["path_type"], round(row["curvature"], 4))
        buckets.setdefault(key, []).append(row)

    by_bucket = {}
    for (path_type, curvature), rows in sorted(buckets.items()):
        success = [r["success"] for r in rows]
        reasons = {}
        for name in REASON_NAMES.values():
            count = sum(1 for r in rows if r["reason"] == name)
            if count:
                reasons[name] = count
        by_bucket[f"{path_type}@k{curvature:.4g}"] = {
            "path_type": path_type,
            "curvature": curvature,
            "episodes": len(rows),
            "success_rate": float(np.mean(success)),
            "endpoint_median_m": float(np.median([r["endpoint"] for r in rows])),
            "speed_median_mps": float(np.median([r["speed"] for r in rows])),
            "failure_reasons": reasons,
        }

    eligible = sorted(
        (v["success_rate"] for v in by_bucket.values() if v["episodes"] >= min_bucket_n)
    )
    overall = float(np.mean([r["success"] for r in records])) if records else 0.0
    # The single worst bucket quantises badly: with ~20 buckets over a few
    # hundred episodes, one bucket can hold 7 episodes and every result is then a
    # multiple of 1/7.  Averaging the worst few buckets keeps the minimax intent
    # while making the score continuous enough to hill-climb on.
    smoothed = float(np.mean(eligible[:SMOOTH_BUCKETS])) if eligible else 0.0
    return {
        "episodes": len(records),
        "success_rate": overall,
        "worst_bucket_rate": float(min(eligible)) if eligible else 0.0,
        "smoothed_worst_rate": smoothed,
        "eligible_buckets": len(eligible),
        "by_type_curvature": by_bucket,
    }


def main():
    kind = os.environ.get("PATH_SWEEP_KIND", "baseline").strip()
    configs_file = os.environ.get("PATH_SWEEP_CONFIGS")
    label = os.environ.get("PATH_SWEEP_LABEL", "prior_sweep")
    stage = int(os.environ.get("PATH_EVAL_STAGE", "6"))
    seed = int(os.environ.get("PATH_EVAL_SEED", "4200"))
    forced_type = os.environ.get("PATH_SWEEP_FORCED_TYPE")
    forced_type = int(forced_type) if forced_type not in (None, "") else None
    forced_curvature = os.environ.get("PATH_SWEEP_FORCED_CURVATURE")
    forced_curvature = float(forced_curvature) if forced_curvature not in (None, "") else None
    min_bucket_n = int(os.environ.get("PATH_SWEEP_MIN_BUCKET_N", "8"))
    max_wall = float(os.environ.get("PATH_SWEEP_MAX_WALL", "900"))
    dry_run = os.environ.get("PATH_SWEEP_DRY_RUN", "0") == "1"

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")

    if configs_file:
        schedule = load_schedule(configs_file)
    else:
        schedule = build_schedule(kind, base_defaults(env_cfg))

    if dry_run:
        print(f"{len(schedule)} configurations:")
        for name, overrides in schedule:
            print(f"  {name}: {overrides}")
        return 0

    # The whole approach assumes the policy output is exactly zero so that the
    # action equals the prior.  Both switches must be on before the environment
    # and the actor-critic are constructed.
    if not bool(getattr(env_cfg.control, "use_path_action_prior", False)):
        raise SystemExit("PATH_USE_ACTION_PRIOR=1 is required: without it the "
                         "action is the raw policy output, not the prior")
    if not bool(getattr(train_cfg.policy, "zero_init_actor_output", False)):
        raise SystemExit("PATH_ZERO_INIT_ACTOR=1 is required: a randomly "
                         "initialised actor would add a non-zero residual and "
                         "the sweep would not be measuring the prior alone")

    out = Path(os.environ.get(
        "PATH_SWEEP_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / label),
    ))
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["label"])
        print(f"resuming: {len(done)} configurations already recorded")

    pending = [(name, overrides) for name, overrides in schedule if name not in done]
    if not pending:
        print("nothing to do; every configuration already has a result")
        return 0

    # Environment and runner are built once and reused for every configuration.
    cli_args = get_args()
    cli_args.headless = True
    env_cfg.seed = seed
    env_cfg.noise.add_noise = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.domain_rand.push_robots = False
    if cli_args.num_envs is not None:
        env_cfg.env.num_envs = cli_args.num_envs
    env, _ = task_registry.make_env(name="rotunbot_path", args=cli_args, env_cfg=env_cfg)

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=cli_args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)

    defaults = base_defaults(env.cfg)

    # Prove the residual is inert rather than assuming it: the actor output must
    # be zero on a real observation.
    probe = env.get_observations()
    with torch.no_grad():
        residual = policy(probe)
    residual_abs_max = float(residual.abs().max().item())
    print(f"num_envs={env.num_envs}  residual_scale=({env.cfg.path.prior_residual_scale_1},"
          f"{env.cfg.path.prior_residual_scale_2})  max|actor output|={residual_abs_max:.3e}")
    if residual_abs_max != 0.0:
        raise SystemExit("actor output is not zero; the action would not equal the prior")

    settings = SimpleNamespace(
        seed=seed, stage=stage, forced_type=forced_type,
        forced_curvature=forced_curvature, max_wall_seconds=max_wall)

    stop_file = out / "STOP"
    for index, (name, overrides) in enumerate(pending):
        if stop_file.exists():
            print(f"STOP file present, halting before {name}")
            break
        apply_overrides(env, overrides, defaults)
        started = time.time()
        records, steps = rollout(env, policy, settings)
        elapsed = time.time() - started
        summary = summarise(records, min_bucket_n)
        summary.update({"label": name, "overrides": overrides, "steps": steps,
                        "seconds": round(elapsed, 1)})
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(f"[{index + 1}/{len(pending)}] {name:38} "
              f"overall={summary['success_rate']:6.1%}  "
              f"worst_bucket={summary['worst_bucket_rate']:6.1%} "
              f"({summary['eligible_buckets']} buckets)  {elapsed:5.1f}s")
        sys.stdout.flush()

    print(f"results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
