#!/usr/bin/env bash
# Single-variable sweep of the analytic prior's deceleration-ramp scale.
#
# `desired speed = cruise * clamp(endpoint_distance / stop_distance, 0, 1)`, so
# a larger stop_distance asks the ball to slow down earlier.  The prior uses
# 0.90 m for |curvature| < 0.38 and 2.40 m above that, which correlates with the
# measured pattern that gentle-curvature arcs stall 0.4-0.5 m short of the
# endpoint while sharper arcs close to within 0.14 m.
#
# This script exists to test that correlation.  It is a single-variable test:
# only PATH_PRIOR_NORMAL_STOP changes, and each value writes to its own output
# directory (an earlier ad-hoc sweep silently overwrote one value with another
# because they shared a directory).
#
# Result (n=512, seed 4206, pure prior, V6 model_0):
#   stop    train   test
#   0.45    81.8%   71.3%
#   0.90    81.6%   71.7%   <- shipped default, in ablation_prior_vs_ppo/
#   2.40    74.8%   65.6%
# 0.45 and 0.90 are within noise of each other; 2.40 is clearly worse.  The ramp
# scale is therefore not the lever that controls the endgame stall.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${PATH_PRIOR_CHECKPOINT:-$ROOT/logs/rotunbot_path/Sep11_08-30-20_geometric_path_v6_safe_from_scratch/model_0.pt}"

export PATH_TRAIN_STAGE=6
export PATH_USE_ACTION_PRIOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_ZERO_INIT_ACTOR=1
export PATH_FREEZE_HISTORY=0
export PATH_INIT_STD=0.05
export PATH_MIN_STD=0.02
export PATH_MAX_STD=0.10
export PATH_EVAL_STAGE=6
export PATH_EVAL_EPISODES="${PATH_EVAL_EPISODES:-512}"
export PATH_EVAL_SEED=4200

for stop in ${PATH_PRIOR_STOP_VALUES:-0.45 0.90 2.40}; do
  tag="stop${stop//./p}"
  out="$ROOT/artifacts/path_follower/prior_stop_distance_sweep_$tag"
  mkdir -p "$out"
  for split in train test; do
    echo "=== stop=$stop split=$split -> $tag"
    PATH_CHECKPOINT="$CHECKPOINT" \
    PATH_PRIOR_NORMAL_STOP="$stop" \
    PATH_LAYOUT_SPLIT="$split" \
    PATH_EVAL_OUTPUT="$out" \
      "$ROOT/run_local_eval.sh" >"$out/${split}.log" 2>&1
  done
done

echo
echo "Regenerate the summary with: python3 legged_gym/scripts/summarize_path_evaluations.py"
