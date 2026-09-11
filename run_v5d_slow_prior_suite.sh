#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
PY=/data/lzq/conda_envs/isaacgym/bin/python
OUT="$ROOT/artifacts/path_follower/analytic_controller/v5d_slow_suite_logs"
mkdir -p "$OUT"
run_one() {
  local gpu="$1" stage="$2" type="$3" curvature="$4" seed="$5"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" PATH_EVAL_STAGE="$stage" \
  PATH_EVAL_TYPE="$type" PATH_EVAL_CURVATURE="$curvature" PATH_EVAL_EPISODES=256 \
  PATH_EVAL_SEED="$seed" "$PY" legged_gym/scripts/evaluate_analytic_prior.py \
  --task rotunbot_path --headless --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 \
  >"$OUT/stage${stage}_${type}_${curvature}.log" 2>&1
}
(run_one 0 2 straight 0 8000; run_one 0 4 left_arc 0.5 8001; run_one 0 6 s_curve 0.5 8002) & p0=$!
(run_one 2 2 left_arc 0.3333333333 8000; run_one 2 4 right_arc -0.5 8001; run_one 2 3 left_arc 0.4 8002) & p2=$!
(run_one 3 2 right_arc -0.3333333333 8000; run_one 3 5 s_curve 0.3333333333 8001; run_one 3 3 right_arc -0.4 8002) & p3=$!
wait "$p0"; wait "$p2"; wait "$p3"
grep -H -E '"success_rate"|"failure_reasons"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"' "$OUT"/*.log || true
