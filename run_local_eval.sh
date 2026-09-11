#!/usr/bin/env bash
# Machine-local evaluation launcher for the geometric path follower.
#
# Replaces the archived eval_*.sh scripts, which hardcode the original
# workstation paths.  See local_env.sh and LOCAL_SETUP.md.
#
# Environment (all optional):
#   PATH_CHECKPOINT       checkpoint .pt to evaluate (default: newest under logs/)
#   PATH_EVAL_STAGE       curriculum stage to generate paths from   (default: 0)
#   PATH_EVAL_EPISODES    completed episodes to collect             (default: 256)
#   PATH_EVAL_SEED        base seed                                 (default: 1100)
#   PATH_EVAL_TYPE        straight|left_arc|right_arc|s_curve       (default: mixed)
#   PATH_EVAL_CURVATURE   force one signed curvature, e.g. -0.3333333333
#   PATH_EVAL_STOCHASTIC  1 to sample actions instead of using the mean (default: 0)
#   PATH_LAYOUT_SPLIT     train|test; test draws unseen curvature   (default: train)
#   PATH_EVAL_OUTPUT      output root (default: artifacts/path_follower/evaluation)
#
# Any PATH_* variable the env config reads (PATH_USE_ACTION_PRIOR,
# PATH_V5_CURVATURE_OBS, PATH_RESIDUAL_SCALE_*, ...) must match the
# configuration the checkpoint was trained with, or the actor shapes will not
# load.  The archived launch_*.sh files record the values each run used.
#
# Example: the held-out generalisation split at n=512 for a prior+residual run:
#
#   PATH_CHECKPOINT=logs/rotunbot_path/<run>/model_50.pt \
#   PATH_USE_ACTION_PRIOR=1 PATH_V5_CURVATURE_OBS=1 PATH_ZERO_INIT_ACTOR=1 \
#   PATH_FREEZE_HISTORY=0 PATH_EVAL_STAGE=6 PATH_EVAL_EPISODES=512 \
#   PATH_LAYOUT_SPLIT=test ./run_local_eval.sh
set -euo pipefail

# shellcheck source=local_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"

exec "$PYTHON" legged_gym/scripts/evaluate_path_follower.py \
  --task rotunbot_path \
  --headless \
  --sim_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  --rl_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  "$@"
