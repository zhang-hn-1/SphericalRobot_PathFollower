#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/lzq/workspace/SphericalRobot_PathFollower_20260911
PY=/data/lzq/conda_envs/isaacgym/bin/python
OUT="$ROOT/artifacts/path_follower/v4_action_diagnosis"
mkdir -p "$OUT"

run_eval() {
  local gpu="$1" checkpoint="$2" label="$3" stochastic="$4" type="$5" curvature="$6" seed="$7"
  cd "$ROOT"
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" \
  PATH_CHECKPOINT="$checkpoint" PATH_EVAL_STAGE=2 PATH_EVAL_TYPE="$type" \
  PATH_EVAL_CURVATURE="$curvature" PATH_EVAL_EPISODES=256 PATH_EVAL_SEED="$seed" \
  PATH_EVAL_STOCHASTIC="$stochastic" \
  "$PY" legged_gym/scripts/evaluate_path_follower.py --task rotunbot_path \
    --headless --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 \
    >"$OUT/${label}.log" 2>&1
}

BASE="$ROOT/logs/rotunbot_path/Sep11_02-59-28_geometric_path_v4_moe_continual/model_600.pt"
V4D="$ROOT/logs/rotunbot_path/Sep11_03-23-01_v4d_right_r3_feedforward/model_775.pt"
run_eval 0 "$BASE" base600_mean_right_r3 0 right_arc -0.3333333333 6300 & p0=$!
run_eval 2 "$V4D" v4d775_sample_right_r3 1 right_arc -0.3333333333 6300 & p2=$!
run_eval 3 "$V4D" v4d775_mean_right_r4 0 right_arc -0.25 6300 & p3=$!
wait "$p0"; wait "$p2"; wait "$p3"

grep -H -E '"stochastic_actions"|"success_rate"|"cross_track_median_m"|"endpoint_distance_median_m"|"terminal_speed_median_mps"|"failure_reasons"|"trace_env0_action_mean"' "$OUT"/*.log || true
