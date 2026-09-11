#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
OUT="$ROOT/artifacts/path_follower/v7_variable_residual_from_scratch"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PATH_RUN_NAME=geometric_path_v7_variable_residual_from_scratch
export PATH_TRAIN_STAGE=6
export PATH_TRAIN_TYPE=s_curve
export PATH_VARIABLE_CURVE_PROBABILITY=1
export PATH_USE_ACTION_PRIOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_ZERO_INIT_ACTOR=1
export PATH_FREEZE_HISTORY=0
export PATH_RESIDUAL_SCALE_1=0.02
export PATH_RESIDUAL_SCALE_2=0.08
export PATH_RESIDUAL_PENALTY=-0.50
export PATH_INIT_STD=0.10
export PATH_MIN_STD=0.03
export PATH_MAX_STD=0.15
export PATH_PPO_LR=1.0e-5
export PATH_PPO_CLIP=0.05
export PATH_PPO_EPOCHS=1
export PATH_ENTROPY=0.0
exec /data/lzq/conda_envs/isaacgym/bin/python legged_gym/scripts/train_path.py \
  --task rotunbot_path --headless --sim_device cuda:0 --rl_device cuda:0 \
  --num_envs 2048 --max_iterations 200 >"$OUT/train.log" 2>&1
