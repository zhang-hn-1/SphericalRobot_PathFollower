#!/usr/bin/env bash
# Ablation: analytic prior alone vs prior + learned PPO residual.
#
# The archived V6 run (launch_v6_safe_from_scratch.sh) starts from a zero-
# initialised actor, so model_0.pt IS the analytic prior with no learned
# contribution, and later checkpoints are prior + residual.  Comparing them on
# the same paths at the same seed isolates what PPO added.
#
# Each checkpoint is evaluated on both sides of the generalisation split:
#   train - the curvature values the run was trained on
#   test  - held-out curvature {0.15, 0.30, 0.45}, never seen in any stage list
#
# Results land in artifacts/path_follower/ablation_prior_vs_ppo/ and are picked
# up by legged_gym/scripts/summarize_path_evaluations.py.
#
# Episode counts of 512 are used because the archived scans used 32-128, which
# is only +/- 8-15 points of confidence.
#
# PATH_ABLATION_CHECKPOINTS selects which iterations to evaluate (default "0 50":
# the zero-initialised actor, i.e. the bare prior, and one trained checkpoint).
# Use "0 50 75 100 125" to check whether training eventually collapses the prior
# the way the V5b run did.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/logs/rotunbot_path/Sep11_08-30-20_geometric_path_v6_safe_from_scratch"
OUT="$ROOT/artifacts/path_follower/ablation_prior_vs_ppo"
mkdir -p "$OUT"

# The environment configuration must match launch_v6_safe_from_scratch.sh, or
# the actor tensors will not load.
export PATH_TRAIN_STAGE=6
export PATH_USE_ACTION_PRIOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_ZERO_INIT_ACTOR=1
export PATH_FREEZE_HISTORY=0
export PATH_INIT_STD=0.05
export PATH_MIN_STD=0.02
export PATH_MAX_STD=0.10
export PATH_PPO_LR=1.0e-5
export PATH_PPO_CLIP=0.05
export PATH_PPO_EPOCHS=1
export PATH_ENTROPY=0.0
export PATH_EVAL_STAGE=6
export PATH_EVAL_EPISODES="${PATH_EVAL_EPISODES:-512}"
export PATH_EVAL_OUTPUT="$OUT"

for checkpoint in ${PATH_ABLATION_CHECKPOINTS:-0 50}; do
  if [ "$checkpoint" = 0 ]; then
    policy="prior"
  else
    policy="prior_plus_ppo"
  fi
  for split in train test; do
    label="model_${checkpoint}_${policy}_${split}"
    echo "=== $label"
    PATH_CHECKPOINT="$RUN_DIR/model_${checkpoint}.pt" \
    PATH_LAYOUT_SPLIT="$split" \
    PATH_EVAL_SEED=4200 \
      "$ROOT/run_local_eval.sh" >"$OUT/$label.log" 2>&1
    echo "    done"
  done
done

echo
echo "Now regenerate the summary with:"
echo "  python3 legged_gym/scripts/summarize_path_evaluations.py"
