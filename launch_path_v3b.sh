#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN_DIR="$ROOT/artifacts/path_follower/v3b_soft_symmetric_from_scratch"
PID_FILE="$RUN_DIR/train.pid"
LOG_FILE="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "training already running: pid=$old_pid"
    exit 0
  fi
fi
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=1 PATH_LOAD_OPTIMIZER=1 ./train_path_follower.sh --num_envs 2048 --max_iterations 6000 >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started pid=$pid log=$LOG_FILE initialization=random"