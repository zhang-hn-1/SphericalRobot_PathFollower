#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
RUN="$ROOT/logs/rotunbot_path/Sep11_04-01-00_v5c_right_r3_from_model50"
OUT="$ROOT/artifacts/path_follower/v5c_right_r3_from_model50/fixed_eval"
PY=/data/lzq/conda_envs/isaacgym/bin/python
mkdir -p "$OUT"
run_one() {
  local gpu="$1" model="$2"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" PATH_USE_ACTION_PRIOR=1 \
  PATH_V5_CURVATURE_OBS=1 PATH_CHECKPOINT="$RUN/model_${model}.pt" \
  PATH_EVAL_STAGE=2 PATH_EVAL_TYPE=right_arc PATH_EVAL_CURVATURE=-0.3333333333 \
  PATH_EVAL_EPISODES=256 PATH_EVAL_SEED=7600 \
  "$PY" legged_gym/scripts/evaluate_path_follower.py --task rotunbot_path \
    --headless --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 \
    >"$OUT/model_${model}_right_r3.log" 2>&1
}
run_one 2 75 & p2=$!
run_one 3 100 & p3=$!
wait "$p2"; wait "$p3"
grep -H -E '"success_rate"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"|"failure_reasons"' "$OUT"/*.log || true
