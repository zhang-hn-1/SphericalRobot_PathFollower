#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN_DIR="$ROOT/artifacts/path_follower/v1_direction_resume1000"
PID_FILE="$RUN_DIR/train.pid"
LOG_FILE="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR"
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=0 ./train_path_follower.sh --num_envs 2048 --resume --load_run Sep11_01-01-34_geometric_path_v1_from_scratch --checkpoint 1000 --max_iterations 4000 >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started pid=$pid log=$LOG_FILE"
