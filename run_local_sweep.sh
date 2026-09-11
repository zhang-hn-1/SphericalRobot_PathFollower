#!/usr/bin/env bash
# Machine-local launcher for the analytic-prior parameter sweep.
#
# Sets the configuration the sweep requires and forwards everything else to
# sweep_path_prior.py.  See LOCAL_SETUP.md for the environment constraints.
#
# Example:
#   PATH_SWEEP_KIND=baseline ./run_local_sweep.sh --num_envs 256
#   PATH_SWEEP_KIND=screen   ./run_local_sweep.sh --num_envs 512
#   PATH_SWEEP_CONFIGS=my_configs.json PATH_SWEEP_LABEL=my_sweep ./run_local_sweep.sh
#   PATH_SWEEP_MODE=search PATH_SEARCH_BUDGET_SECONDS=28800 ./run_local_sweep.sh --num_envs 512
#
# PATH_SWEEP_MODE=sweep (default) runs a fixed schedule once; =search runs the
# budgeted coordinate-descent optimiser under the same output directory.
# Sweep settings are environment variables (see the module docstrings in
# legged_gym/scripts/sweep_path_prior.py and search_path_prior.py); command-line
# flags are IsaacGym's.
set -euo pipefail

# shellcheck source=local_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"

# The sweep measures the analytic prior alone, so the residual must be exactly
# zero.  PATH_ZERO_INIT_ACTOR zeroes both weight and bias of every expert's
# output layer at construction; the harness aborts if it detects otherwise.
export PATH_USE_ACTION_PRIOR=1
export PATH_ZERO_INIT_ACTOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_FREEZE_HISTORY=0
export PATH_TRAIN_STAGE="${PATH_TRAIN_STAGE:-6}"

if [[ "${PATH_SWEEP_MODE:-sweep}" == "search" ]]; then
  SCRIPT=legged_gym/scripts/search_path_prior.py
else
  SCRIPT=legged_gym/scripts/sweep_path_prior.py
fi

exec "$PYTHON" "$SCRIPT" \
  --task rotunbot_path \
  --headless \
  --sim_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  --rl_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  "$@"
