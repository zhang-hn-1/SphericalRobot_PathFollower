#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN="$ROOT/logs/rotunbot_path/Sep11_03-52-46_geometric_path_v5b_residual_from_scratch"
OUT="$ROOT/artifacts/path_follower/v5b_residual_from_scratch/early_eval"
PY=/data/lzq/conda_envs/isaacgym/bin/python
mkdir -p "$OUT"

evaluate_one() {
  local gpu="$1" model="$2" type="$3" curvature="$4" seed="$5"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" \
  PATH_USE_ACTION_PRIOR=1 PATH_V5_CURVATURE_OBS=1 \
  PATH_CHECKPOINT="$RUN/model_${model}.pt" PATH_EVAL_STAGE=2 \
  PATH_EVAL_TYPE="$type" PATH_EVAL_CURVATURE="$curvature" \
  PATH_EVAL_EPISODES=128 PATH_EVAL_SEED="$seed" \
  "$PY" legged_gym/scripts/evaluate_path_follower.py --task rotunbot_path \
    --headless --num_envs 128 --sim_device cuda:0 --rl_device cuda:0 \
    >"$OUT/model_${model}_${type}_${curvature}.log" 2>&1
}

(evaluate_one 0 0 straight 0 7400; evaluate_one 0 25 straight 0 7400) & p0=$!
(evaluate_one 2 0 left_arc 0.3333333333 7400; evaluate_one 2 25 left_arc 0.3333333333 7400) & p2=$!
(evaluate_one 3 0 right_arc -0.3333333333 7400; evaluate_one 3 25 right_arc -0.3333333333 7400) & p3=$!
wait "$p0"; wait "$p2"; wait "$p3"
grep -H -E '"success_rate"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"' "$OUT"/*.log || true
