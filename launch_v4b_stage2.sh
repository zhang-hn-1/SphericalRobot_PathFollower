#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN_DIR="$ROOT/artifacts/path_follower/v4b_moe_stage2_freshopt"
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
nohup env CUDA_VISIBLE_DEVICES=1 PATH_LOAD_OPTIMIZER=0 ./train_path_follower.sh --num_envs 2048 --resume --load_run "$SOURCE_RUN" --checkpoint 600 --max_iterations 5400 >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "started pid=$pid log=$LOG_FILE source=$SOURCE_RUN/model_600.pt optimizer=fresh"