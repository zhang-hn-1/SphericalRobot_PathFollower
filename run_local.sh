#!/usr/bin/env bash
# Machine-local training launcher.  See local_env.sh for the environment setup
# and LOCAL_SETUP.md for the constraints this machine needs.
#
# Every experiment setting is passed through as an environment variable, exactly
# as the archived launch_*.sh scripts recorded it, e.g. to replay V6:
#
#   PATH_RUN_NAME=geometric_path_v6_safe_from_scratch PATH_TRAIN_STAGE=6 \
#   PATH_USE_ACTION_PRIOR=1 PATH_V5_CURVATURE_OBS=1 PATH_ZERO_INIT_ACTOR=1 \
#   PATH_FREEZE_HISTORY=0 PATH_INIT_STD=0.05 PATH_MIN_STD=0.02 \
#   PATH_MAX_STD=0.10 PATH_PPO_LR=1.0e-5 PATH_PPO_CLIP=0.05 \
#   PATH_PPO_EPOCHS=1 PATH_ENTROPY=0.0 \
#   ./run_local.sh --num_envs 2048 --max_iterations 300
set -euo pipefail

# shellcheck source=local_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"

exec "$PYTHON" legged_gym/scripts/train_path.py \
  --task rotunbot_path \
  --headless \
  --sim_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  --rl_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  "$@"
