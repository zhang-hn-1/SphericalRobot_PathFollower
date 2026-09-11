#!/usr/bin/env bash
# Run any script in this checkout with the local environment applied.
#
#   ./run_local_script.sh legged_gym/scripts/identify_minimum_rolling_speed.py
#
# The identification and analysis scripts all need the same interpreter,
# PYTHONPATH and compat shims as training; local_env.sh provides them.
# IsaacGym flags after the script path are forwarded to the script itself.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

SCRIPT="$1"
shift

# shellcheck source=local_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/local_env.sh"

exec "$PYTHON" "$SCRIPT" \
  --task rotunbot_path \
  --headless \
  --sim_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  --rl_device "cuda:${CUDA_VISIBLE_DEVICES}" \
  "$@"
