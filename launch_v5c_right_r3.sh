#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
OUT="$ROOT/artifacts/path_follower/v5c_right_r3_from_model50"
mkdir -p "$OUT"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PATH_RUN_NAME=v5c_right_r3_from_model50
export PATH_TRAIN_STAGE=2
export PATH_TRAIN_TYPE=right_arc
export PATH_TRAIN_CURVATURE=-0.3333333333
export PATH_TRAIN_EXPERT=right
export PATH_USE_ACTION_PRIOR=1
export PATH_V5_CURVATURE_OBS=1
export PATH_ZERO_INIT_ACTOR=0
export PATH_FREEZE_HISTORY=1
export PATH_INIT_STD=0.05
export PATH_MIN_STD=0.03
export PATH_MAX_STD=0.08
export PATH_PPO_LR=2.0e-5
export PATH_PPO_CLIP=0.10
export PATH_PPO_EPOCHS=2
export PATH_ENTROPY=1.0e-4
export PATH_LOAD_OPTIMIZER=0

exec /data/lzq/conda_envs/isaacgym/bin/python legged_gym/scripts/train_path.py \
  --task rotunbot_path --headless --sim_device cuda:0 --rl_device cuda:0 \
  --num_envs 2048 --resume \
  --load_run Sep11_03-52-46_geometric_path_v5b_residual_from_scratch \
  --checkpoint 50 --max_iterations 250 >"$OUT/train.log" 2>&1
