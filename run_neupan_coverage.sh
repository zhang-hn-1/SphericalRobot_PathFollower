#!/usr/bin/env bash
# Fill the two untested NeuPAN path families on this machine.
#
# Coverage before this script (see artifacts/path_follower/report_20260912):
#   geometric paths        - tested exhaustively
#   NeuPAN local S* windows- only 22 curved windows of one scenario, locally
#   NeuPAN global paths    - never run locally; the archived results came from the
#                            original workstation (/data/lzq is absent here)
#
# Inputs: the two official evaluators read files that only exist on the original
# workstation, so legged_gym/scripts/rebuild_neupan_local_inputs.py reconstructs
# them from the ``desired`` arrays the archived runs saved.  Read that module's
# docstring for what the reconstruction can and cannot recover -- in particular
# the global planner yaw is gone and is re-derived, validated against the
# archived curvature peaks.
#
# Prior configuration: the documented recommended prior
# (artifacts/path_follower/RECOMMENDED_PRIOR.md) plus the endgame switches, i.e.
# the configuration the project's headline numbers use.
#
# Note the two invariants that differ between the runners:
#   * run_neupan_windows.py passes zero actions and lets RotunbotPath.step add
#     the prior -> it needs PATH_USE_ACTION_PRIOR=1.
#   * evaluate_neupan_{sstar,executed_paths}.py pass the prior themselves -> the
#     switch must stay off or the prior is applied twice.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="$ROOT/artifacts/path_follower"

# Recommended prior + endgame fix.
export PATH_PRIOR_ENDPOINT_FLOOR=1
export PATH_ENDPOINT_PP=1
export PATH_ENDPOINT_PP_DISTANCE=0.25
export PATH_ENDPOINT_BLEND_DISTANCE=0.2
export PATH_GAIN_MIN=0.24
export PATH_NORMAL_DRIVE=0.35
export PATH_NORMAL_HEADING_KP=2
export PATH_S_HEADING_KP=1
export PATH_S_MID_HEADING_KP=1.3
export PATH_S_TIGHT_SHORT_CROSS_TRACK_KP=0.05
export PATH_TIGHT_HEADING_KP=0.8

SCENARIOS="convex_obs_acker_official corridor_acker_official non_obs_acker_official pf_acker_official pf_obs_acker_official"
DO_LOCAL="${DO_LOCAL:-1}"
DO_GLOBAL="${DO_GLOBAL:-1}"

if [[ "$DO_LOCAL" == "1" ]]; then
  echo "=========== local S* windows, all scenarios, every window ==========="
  for scenario in $SCENARIOS; do
    npz="$ART/neupan_local_inputs/${scenario}_windows.npz"
    [[ -f "$npz" ]] || { echo "missing $npz (run rebuild_neupan_local_inputs.py)"; exit 1; }
    echo "--- $scenario"
    PATH_PATH_SOURCE=external \
    PATH_USE_ACTION_PRIOR=1 \
    PATH_NEUPAN_WINDOWS="$npz" \
    PATH_NEUPAN_ENVS=100000 \
    PATH_NEUPAN_EPISODE_S=150 \
    PATH_NEUPAN_SEED=4200 \
    PATH_NEUPAN_OUTPUT="$ART/neupan_local_all/$scenario" \
      bash "$ROOT/run_local_script.sh" legged_gym/scripts/run_neupan_windows.py \
      2>&1 | grep -E "loaded|installed|success |failure reasons|endpoint median|cross-track median|wrote|Error|Traceback"
  done
fi

if [[ "$DO_GLOBAL" == "1" ]]; then
  for variant in savgol tangent; do
    if [[ "$variant" == "savgol" ]]; then
      src="$ART/neupan_global_inputs"; out="$ART/neupan_global_rebuilt"
    else
      src="$ART/neupan_global_inputs_tangent"; out="$ART/neupan_global_rebuilt_tangent"
    fi
    echo "=========== global executed paths ($variant yaw) ==========="
    NEUPAN_OUTPUT_ROOT="$src" \
    NEUPAN_GLOBAL_OUTPUT="$out" \
    NEUPAN_GLOBAL_EPISODE_LENGTH_S="${NEUPAN_GLOBAL_EPISODE_LENGTH_S:-700}" \
      bash "$ROOT/run_local_script.sh" legged_gym/scripts/evaluate_neupan_executed_paths.py \
      2>&1 | grep -E "Traceback|Error|error" || true
    python3 - "$out/metrics.json" "$variant" <<'PY'
import json, sys
path, variant = sys.argv[1], sys.argv[2]
m = json.load(open(path))
print("  variant=%s  success %d/%d" % (variant, m["success_count"], m["path_count"]))
for r in m["results"]:
    print("    %-28s %-9s progress=%5.1f%% endpoint=%6.2f m  x-track med/p95=%5.2f/%5.2f m  peak|k|=%.2f"
          % (r["scenario"], r["reason"], 100 * r["progress_fraction"],
             r["endpoint_distance_m"], r["cross_track_median_m"], r["cross_track_p95_m"],
             r["peak_abs_curvature_1pm"]))
PY
  done
fi

echo
echo "done. Local: $ART/neupan_local_all/*/summary.json   Global: $ART/neupan_global_rebuilt*/metrics.json"
