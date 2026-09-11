#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
PY=/data/lzq/conda_envs/isaacgym/bin/python
OUT="$ROOT/artifacts/path_follower/analytic_controller/v5d_piecewise_logs"
mkdir -p "$OUT"
run_one() {
  local gpu="$1" type="$2" curvature="$3"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" PATH_EVAL_STAGE=2 \
  PATH_EVAL_TYPE="$type" PATH_EVAL_CURVATURE="$curvature" PATH_EVAL_EPISODES=256 \
  PATH_EVAL_SEED=7900 "$PY" legged_gym/scripts/evaluate_analytic_prior.py \
  --task rotunbot_path --headless --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 \
  >"$OUT/${type}_${curvature}.log" 2>&1
}
run_one 0 straight 0 & p0=$!
run_one 2 left_arc 0.3333333333 & p2=$!
run_one 3 right_arc -0.3333333333 & p3=$!
wait "$p0"; wait "$p2"; wait "$p3"
grep -H -E '"success_rate"|"failure_reasons"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"' "$OUT"/*.log || true
