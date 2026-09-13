# Rotunbot geometric path follower

A spherical robot (Rotunbot) follows a given geometric path: **no required travel
speed, no terminal yaw** — the task is "stay on the path and come to rest at its
end".

Built on `legged_gym` (ETH Zurich). The upstream README is preserved as
[`README_legged_gym.md`](README_legged_gym.md); this file describes the path
follower.

## Interface

| | |
|---|---|
| actions | joint-1 **target velocity** (±3 rad/s) and joint-2 **target angle** (±0.5236 rad), 50 Hz |
| observation | 20 frames × 19-D yaw-invariant state + 10×4 local path preview + 3 endpoint features = **423-D** |
| policy | hand-tuned analytic prior + bounded neural residual. The actor output layer is zero-initialised, so with `PATH_ZERO_INIT_ACTOR=1` the action equals the prior **exactly** |
| success | remaining arc length ≤ 0.20 m **and** distance to the final path sample ≤ 0.20 m **and** terminal speed ≤ 0.10 m/s |
| failure | cross-track > 1.50 m, tilt, out of bounds, or 40 s timeout |

## Status

Measured on **held-out curvature** (values that appear in no training stage) at
two independent seeds, 2048 environments, one paired episode per environment:

| | overall | worst (path type × curvature) bucket |
|---|---|---|
| shipped default | 78.9% | 35.1% |
| **recommended configuration** | **87.0% / 86.5%** | **76.2% / 68.4%** |

The worst bucket is the number that matters: it says whether path following works
everywhere rather than on average. See
[`artifacts/path_follower/report_20260912/REPORT.md`](artifacts/path_follower/report_20260912/REPORT.md)
for the write-up and five figures.

Remaining gap: `s_curve` at \|κ\| = 0.40–0.45 is still the weakest bucket
(76% held out). Everything else is at 85% or above.

## Findings worth knowing before reading the logs

* **The headline success rate is not a stable statistic.** The same checkpoint
  scores 96.9% or 81.6% depending only on the random seed, because each episode
  draws its curvature from a discrete set and the mixture moves the average.
  Quote the (path type × curvature) table, never the headline number.
* **The learned residual measured inert.** Over 125 PPO updates on top of the
  prior the score does not move (81.6 / 81.8 / 81.2 / 82.8 / 82.6%). The prior
  alone is the policy.
* **Endgame stall root cause:** the speed ramp used arc length remaining, which
  saturates at the last path sample, while success is measured as straight-line
  distance to that sample. A ball carrying 0.4–0.5 m of lateral error therefore
  parked 0.5 m short, outside the window, until the episode timed out.
* **Overshoot:** the endpoint floor added to fix that then *accelerated* a ball
  that had passed the endpoint, because the endpoint distance grows again after
  an overshoot. Measured on `s_curve` at κ=0.40, all 123 failures had reached the
  end of the path and sat a median 1.38 m past it at 0.30 m/s.
  `PATH_PRIOR_OVERSHOOT_STOP` cuts the drive once the robot is beyond the
  endpoint along the final tangent.
* **Six mechanisms were tested and refuted** for the sharp S-curve bottleneck:
  deceleration ramp, minimum rolling speed, preview phase error, drive authority,
  steering anti-windup, and randomised initial offsets. Also refuted: the idea
  that full tilt blocks propulsion (top speed falls only 20% at full tilt).
* **Planner paths need the yaw column.** Passing xy alone makes the environment
  differentiate polyline tangents, so every vertex becomes a curvature spike:
  3.6% success with xy only against 99.3% with a reconstructed yaw, on the
  `convex` scenario. An xy-only result measures the reconstruction.

## Running it

Everything is driven through the launchers; see [`LOCAL_SETUP.md`](LOCAL_SETUP.md)
for the environment constraints (NumPy ≥ 1.24 shims, `ninja`, `distutils`,
Ada/sm_89 TorchScript) and the planner-facing contract.

```bash
./run_local.sh --num_envs 2048 --max_iterations 300          # train
./run_local_eval.sh                                          # evaluate a checkpoint
PATH_SWEEP_MODE=search ./run_local_sweep.sh --num_envs 2048   # search the prior
./run_local_script.sh legged_gym/scripts/attribute_path_failure.py
```

`legged_gym/tests/test_path_env_contract.py` is a CPU-only contract test
(observation dimensions, joint limits, generalisation split, planner interface).

## Layout

| path | contents |
|---|---|
| `artifacts/path_follower/report_20260912/` | progress report, figures, figure generator |
| `artifacts/path_follower/RECOMMENDED_PRIOR.md` | recommended prior configuration + reproduction |
| `artifacts/path_follower/RESULTS_SUMMARY.md` | aggregated evaluations, per bucket |
| `PATH_FOLLOWER_LOG.md` | chronological experiment log, including what was refuted |
| `legged_gym/scripts/` | training, evaluation, identification, search tooling |
| `compat/` | local-machine shims; not needed on the original workstation |

All behaviour changes are gated behind environment variables that default to off,
so results recorded before them remain reproducible.
