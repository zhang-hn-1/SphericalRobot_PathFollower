#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN="$ROOT/logs/rotunbot_path/Sep11_03-23-01_v4d_right_r3_feedforward"
OUT="$ROOT/artifacts/path_follower/v4d_right_r3_feedforward/checkpoint_scan"
PY=/data/lzq/conda_envs/isaacgym/bin/python
mkdir -p "$OUT"

evaluate_one() {
  local gpu="$1"
  local checkpoint="$2"
  local seed="$3"
  local log="$OUT/model_${checkpoint}_right_r3_seed${seed}.log"
  cd "$ROOT"
  PYTHONPATH="$ROOT" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PATH_CHECKPOINT="$RUN/model_${checkpoint}.pt" \
  PATH_EVAL_STAGE=2 \
  PATH_EVAL_TYPE=right_arc \
  PATH_EVAL_CURVATURE=-0.3333333333 \
  PATH_EVAL_EPISODES=128 \
  PATH_EVAL_SEED="$seed" \
  "$PY" legged_gym/scripts/evaluate_path_follower.py \
    --task rotunbot_path --headless --num_envs 128 \
    --sim_device cuda:0 --rl_device cuda:0 >"$log" 2>&1
}

(
  evaluate_one 0 650 6200
  evaluate_one 0 725 6201
) &
pid0=$!

(
  evaluate_one 2 675 6202
  evaluate_one 2 750 6203
) &
pid2=$!

(
  evaluate_one 3 700 6204
  evaluate_one 3 775 6205
) &
pid3=$!

wait "$pid0"
wait "$pid2"
wait "$pid3"

grep -H -E '"success_rate"|"collision_rate"|"timeout_rate"|"endpoint_distance_mean"|"final_speed_mean"' "$OUT"/*.log > "$OUT/summary.txt" || true
cat "$OUT/summary.txt"
