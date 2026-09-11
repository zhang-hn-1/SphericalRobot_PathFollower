#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN="$ROOT/logs/rotunbot_path/Sep11_03-52-46_geometric_path_v5b_residual_from_scratch"
OUT="$ROOT/artifacts/path_follower/v5b_residual_from_scratch/checkpoint_eval"
PY=/data/lzq/conda_envs/isaacgym/bin/python
mkdir -p "$OUT"

evaluate_one() {
  local gpu="$1" model="$2" type="$3" curvature="$4" seed="$5"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" PATH_USE_ACTION_PRIOR=1 \
  PATH_V5_CURVATURE_OBS=1 PATH_CHECKPOINT="$RUN/model_${model}.pt" \
  PATH_EVAL_STAGE=2 PATH_EVAL_TYPE="$type" PATH_EVAL_CURVATURE="$curvature" \
  PATH_EVAL_EPISODES=64 PATH_EVAL_SEED="$seed" \
  "$PY" legged_gym/scripts/evaluate_path_follower.py --task rotunbot_path \
    --headless --num_envs 64 --sim_device cuda:0 --rl_device cuda:0 \
    >"$OUT/model_${model}_${type}_${curvature}.log" 2>&1
}

models=(50 75 100)
(for m in "${models[@]}"; do evaluate_one 0 "$m" straight 0 $((7500+m)); done) & p0=$!
(for m in "${models[@]}"; do evaluate_one 2 "$m" left_arc 0.3333333333 $((7500+m)); done) & p2=$!
(for m in "${models[@]}"; do evaluate_one 3 "$m" right_arc -0.3333333333 $((7500+m)); done) & p3=$!
wait "$p0"; wait "$p2"; wait "$p3"
grep -H -E '"success_rate"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"' "$OUT"/*.log || true
