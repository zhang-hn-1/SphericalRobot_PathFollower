#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
PYTHON=/data/lzq/conda_envs/isaacgym/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$PYTHON" legged_gym/scripts/train_path.py \
  --task rotunbot_path \
  --headless \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  "$@"


