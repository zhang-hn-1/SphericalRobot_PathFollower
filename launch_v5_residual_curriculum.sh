#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
OUT="$ROOT/artifacts/path_follower/v5b_residual_from_scratch"
mkdir -p "$OUT"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PATH_RUN_NAME=geometric_path_v5b_residual_from_scratch
export PATH_TRAIN_STAGE=2
export PATH_USE_ACTION_PRIOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_ZERO_INIT_ACTOR=1
export PATH_FREEZE_HISTORY=0
export PATH_INIT_STD=0.15
export PATH_MIN_STD=0.03
export PATH_MAX_STD=0.30
export PATH_PPO_LR=1.0e-4
export PATH_PPO_CLIP=0.20
export PATH_PPO_EPOCHS=5
export PATH_ENTROPY=1.0e-3

exec /data/lzq/conda_envs/isaacgym/bin/python legged_gym/scripts/train_path.py \
  --task rotunbot_path --headless --sim_device cuda:0 --rl_device cuda:0 \
  --num_envs 2048 --max_iterations 1500 >"$OUT/train.log" 2>&1
