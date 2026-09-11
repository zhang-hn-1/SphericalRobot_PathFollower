"""Budgeted coordinate-descent search over the analytic path-following prior.

Companion to ``sweep_path_prior.py``; reuses its rollout, summarising and
override machinery.  Runs until a wall-clock budget is exhausted, improving the
worst (path type x curvature) bucket one parameter at a time.

Why coordinate descent on the worst bucket
------------------------------------------

* The learned residual was measured inert, so the prior is the policy and the
  prior's ~15 scalars are the only real lever.
* A paired rollout costs ~12 s at 512 environments, so a night affords a few
  thousand evaluations - far more than a hand-written schedule can use.
* Scoring by the average would reward a configuration that is excellent on
  whichever curvatures the seed happened to draw.  The worst bucket is what
  "path following works everywhere" actually means.

Robustness for unattended running
---------------------------------

* Every evaluation is appended to ``results.jsonl`` and the cache is keyed by the
  canonical parameter tuple, so restarting replays instantly from cache and
  continues.  An interrupted night loses at most one evaluation.
* ``PATH_SEARCH_BUDGET_SECONDS`` bounds the whole run; only real (uncached)
  rollouts consume budget.
* ``STOP`` in the output directory halts cleanly between evaluations.
* Step sizes shrink on failure to improve and the search stops when every step
  falls below tolerance, so it terminates on its own rather than spinning.

Environment
-----------

    PATH_SWEEP_OUTPUT            output directory (results.jsonl + summary)
    PATH_SEARCH_BUDGET_SECONDS   wall-clock budget for real evaluations (default 28800)
    PATH_SEARCH_SEED             seed used for the search rollouts (default 4200)
    PATH_SEARCH_TOPK             how many local optima to report (default 5)
    PATH_EVAL_STAGE              curriculum stage (default 6)
    plus everything sweep_path_prior.py reads

Run through the launcher so the prior switches are set:

    PATH_SWEEP_MODE=search PATH_SEARCH_BUDGET_SECONDS=28800 \
        ./run_local_sweep.sh --num_envs 512
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaacgym  # noqa: E402  must precede torch

import torch  # noqa: E402

# Imported before legged_gym.utils: the env registration must initialize first,
# otherwise legged_gym.utils.task_registry hits its circular import of
# legged_gym.envs.  Every other entry point in this tree does the same.
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402
from sweep_path_prior import (  # noqa: E402
    SCREEN_STEPS, SMOOTH_BUCKETS, apply_overrides, base_defaults, rollout, summarise,
)

# Bounds keep the search physical: negative gains would invert control, and the
# curvature regime split must stay ordered and inside the unit interval.
LOWER = {
    "prior_normal_drive": 0.05, "prior_tight_drive": 0.05,
    "prior_normal_stop_distance": 0.20, "prior_tight_stop_distance": 0.40,
    "prior_normal_cross_track_kp": 0.0, "prior_tight_cross_track_kp": 0.0,
    "prior_normal_heading_kp": 0.0, "prior_tight_heading_kp": 0.0,
    "prior_speed_kp": 0.05, "prior_gain_offset": 0.0, "prior_gain_slope": 0.05,
    "prior_gain_min": 0.02, "prior_gain_max": 0.10,
    "prior_tight_curvature": 0.05, "prior_r2_curvature": 0.10,
    "prior_endpoint_blend_distance": 0.02, "prior_endpoint_max_curvature": 0.10,
}
UPPER = {
    "prior_normal_drive": 0.60, "prior_tight_drive": 0.80,
    "prior_normal_stop_distance": 3.00, "prior_tight_stop_distance": 4.00,
    "prior_normal_cross_track_kp": 1.00, "prior_tight_cross_track_kp": 1.00,
    "prior_normal_heading_kp": 4.00, "prior_tight_heading_kp": 4.00,
    "prior_speed_kp": 2.00, "prior_gain_offset": 0.30, "prior_gain_slope": 2.00,
    "prior_gain_min": 0.60, "prior_gain_max": 1.20,
    "prior_tight_curvature": 0.60, "prior_r2_curvature": 0.80,
}
TOLERANCE = 1e-3
MIN_IMPROVEMENT = 0.005


def canonical(overrides):
    parts = []
    for key in sorted(overrides):
        value = overrides[key]
        parts.append(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}")
    return "|".join(parts) or "baseline"


def valid(overrides):
    for name, value in overrides.items():
        if name not in LOWER:
            continue
        if not (LOWER[name] <= value <= UPPER[name]):
            return False
    tight = overrides.get("prior_tight_curvature")
    r2 = overrides.get("prior_r2_curvature")
    if tight is not None and r2 is not None and tight >= r2:
        return False
    low = overrides.get("prior_gain_min")
    high = overrides.get("prior_gain_max")
    if low is not None and high is not None and low >= high:
        return False
    return True


def objective(entry):
    """Ranking key: smoothed worst-bucket rate, then overall."""
    smoothed = entry.get("smoothed_worst_rate")
    if smoothed is None:  # results written before the smoothed score existed
        smoothed = entry["worst_bucket_rate"]
    return smoothed, entry["success_rate"]


def main():
    label = os.environ.get("PATH_SWEEP_LABEL", "prior_search")
    budget = float(os.environ.get("PATH_SEARCH_BUDGET_SECONDS", "28800"))
    seed = int(os.environ.get("PATH_SEARCH_SEED", os.environ.get("PATH_EVAL_SEED", "4200")))
    stage = int(os.environ.get("PATH_EVAL_STAGE", "6"))
    # A higher floor than the screen uses: the objective is a worst-bucket
    # average, so buckets with a handful of episodes quantise it into steps of
    # 1/n and the search climbs noise.
    min_bucket_n = int(os.environ.get("PATH_SWEEP_MIN_BUCKET_N", "16"))
    topk = int(os.environ.get("PATH_SEARCH_TOPK", "5"))
    forced_type = os.environ.get("PATH_SWEEP_FORCED_TYPE") or None
    forced_curvature = os.environ.get("PATH_SWEEP_FORCED_CURVATURE") or None

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    if not bool(getattr(env_cfg.control, "use_path_action_prior", False)):
        raise SystemExit("PATH_USE_ACTION_PRIOR=1 is required")
    if not bool(getattr(train_cfg.policy, "zero_init_actor_output", False)):
        raise SystemExit("PATH_ZERO_INIT_ACTOR=1 is required")

    out = Path(os.environ.get(
        "PATH_SWEEP_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" / label)))
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    cache = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[canonical(entry["overrides"])] = entry
        print(f"resumed {len(cache)} cached evaluations")

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
    settings = SimpleNamespace(
        seed=seed, stage=stage,
        forced_type=None if forced_type is None else int(forced_type),
        forced_curvature=None if forced_curvature is None else float(forced_curvature),
        max_wall_seconds=float(os.environ.get("PATH_SWEEP_MAX_WALL", "900")),
    )
    stop_file = out / "STOP"

    spent = 0.0
    evaluated = 0

    def score(overrides):
        """Return the (possibly cached) result for a configuration."""
        nonlocal spent, evaluated
        key = canonical(overrides)
        if key in cache:
            return cache[key]
        if stop_file.exists():
            raise KeyboardInterrupt("STOP file present")
        apply_overrides(env, overrides, defaults)
        started = time.time()
        records, steps = rollout(env, policy, settings)
        elapsed = time.time() - started
        summary = summarise(records, min_bucket_n)
        summary.update({"label": key, "overrides": dict(overrides), "steps": steps,
                        "seconds": round(elapsed, 1), "cumulative_seconds": round(spent + elapsed, 1)})
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        cache[key] = summary
        spent += elapsed
        evaluated += 1
        print(f"  {evaluated:4d} eval  smoothed={summary['smoothed_worst_rate']:6.1%} "
              f"worst={summary['worst_bucket_rate']:6.1%} "
              f"overall={summary['success_rate']:6.1%}  {elapsed:5.1f}s  "
              f"budget {spent/60:5.1f}/{budget/60:.0f}min", flush=True)
        return summary

    try:
        best_overrides = {}
        best_worst, best_overall = objective(score(best_overrides))
        steps = dict(SCREEN_STEPS)
        round_index = 0
        while spent < budget:
            round_index += 1
            improved = False
            for name in SCREEN_STEPS:
                if spent >= budget:
                    break
                for sign in (+1, -1):
                    candidate = dict(best_overrides)
                    candidate[name] = candidate.get(name, defaults[name]) + sign * steps[name]
                    if not valid(candidate):
                        continue
                    worst, overall = objective(score(candidate))
                    if worst >= best_worst + MIN_IMPROVEMENT:
                        print(f"  accepted {name} {sign:+d} -> smoothed={worst:.1%} "
                              f"overall={overall:.1%}", flush=True)
                        best_overrides[name] = candidate[name]
                        best_worst, best_overall = worst, overall
                        improved = True
                        break
            if not improved:
                steps = {k: v * 0.5 for k, v in steps.items()}
                largest = max(steps.values())
                print(f"  round {round_index}: no improvement, halving steps "
                      f"(largest {largest:.4f})", flush=True)
                if largest < TOLERANCE:
                    print("  converged", flush=True)
                    break
    except KeyboardInterrupt as exc:
        print(f"halting: {exc}", flush=True)

    ranked = sorted(cache.values(), key=objective, reverse=True)
    summary_path = out / "summary.md"
    lines = ["# Analytic-prior search summary", "",
             f"Evaluations this session: {evaluated}; cached total: {len(cache)}; "
             f"search wall clock: {spent/60:.1f} min of {budget/60:.0f} min budget.", "",
             f"## Top {topk} configurations by smoothed worst-bucket score", "",
             f"`smoothed` is the mean of the {SMOOTH_BUCKETS} worst eligible "
             "(path type x curvature) buckets; `worst` is the single worst.", "",
             "| smoothed | worst | overall | overrides |", "|---|---|---|---|"]
    for entry in ranked[:topk]:
        overrides = ", ".join(f"{k}={v:.4g}" for k, v in sorted(entry["overrides"].items())) or "defaults"
        smoothed = entry.get("smoothed_worst_rate", entry["worst_bucket_rate"])
        lines.append(f"| {smoothed:.1%} | {entry['worst_bucket_rate']:.1%} | "
                     f"{entry['success_rate']:.1%} | {overrides} |")
    lines += ["", "## Per-bucket detail of the best configuration", "",
              "| bucket | n | success | endpoint | speed | failures |", "|---|---|---|---|---|---|"]
    if ranked:
        for key, bucket in sorted(ranked[0]["by_type_curvature"].items()):
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(bucket["failure_reasons"].items())
                                if k != "success") or "-"
            lines.append(f"| {key} | {bucket['episodes']} | {bucket['success_rate']:.1%} | "
                         f"{bucket['endpoint_median_m']:.3f} | {bucket['speed_median_mps']:.3f} | {reasons} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")
    if ranked:
        print(f"best worst-bucket={ranked[0]['worst_bucket_rate']:.1%} "
              f"overall={ranked[0]['success_rate']:.1%}")
        print("best overrides:", json.dumps(ranked[0]["overrides"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
