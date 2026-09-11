#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN_DIR="$ROOT/artifacts/path_follower/v4c_right_r3_specialist"
PID_FILE="$RUN_DIR/train.pid"
LOG_FILE="$RUN_DIR/train.log"
SOURCE_RUN=v4b_stage2_seed_from_verified_model600
mkdir -p "$RUN_DIR"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "training already running: pid=$old_pid"
    exit 0
  fi
fi
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES=1 PATH_LOAD_OPTIMIZER=0 PATH_TRAIN_TYPE=right_arc PATH_TRAIN_CURVATURE=-0.3333333333 PATH_TRAIN_EXPERT=right PATH_RUN_NAME=v4c_right_r3_specialist ./train_path_follower.sh --num_envs 2048 --resume --load_run "$SOURCE_RUN" --checkpoint 600 --max_iterations 300 >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started pid=$pid log=$LOG_FILE source=$SOURCE_RUN/model_600.pt task=right_R3 expert=right"