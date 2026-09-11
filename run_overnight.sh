#!/usr/bin/env bash
# One night, unattended: search the analytic prior, then verify the winner out of
# sample, then summarise and commit.
#
# The learned residual was measured inert (V6 model_0 through model_125 are
# indistinguishable at n=512), so the prior is the policy and its scalars are the
# lever.  A paired rollout costs ~15 s at 2048 environments, so a night affords
# roughly 1500-1900 evaluations.
#
# Stages
#   1. coordinate-descent search, scored by the smoothed worst (path type x
#      curvature) bucket, on the training curvature set
#   2. re-evaluate the best configurations at a different seed, on the training
#      set and on held-out curvature, to check the gain is not seed- or
#      mixture-specific
#   3. regenerate the results summary and commit
#
# Robustness
#   * every evaluation is appended to results.jsonl and cached by canonical
#     parameter tuple, so re-running resumes in seconds and an interrupted night
#     loses at most one evaluation
#   * create artifacts/path_follower/prior_search_night/STOP to stop cleanly
#   * each stage is skippable via PATH_OVERNIGHT_SKIP (e.g. "2 3")
#
# Environment
#   PATH_OVERNIGHT_BUDGET   search budget in seconds (default 28800 = 8 h)
#   PATH_OVERNIGHT_SEED     seed for the verification stage (default 7777)
#   PATH_OVERNIGHT_TOPK     how many search winners to verify (default 3)
#
# Results
#   artifacts/path_follower/prior_search_night/summary.md    search winners
#   artifacts/path_follower/prior_verified_{train,test}/     out-of-sample check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDGET="${PATH_OVERNIGHT_BUDGET:-28800}"
VERIFY_SEED="${PATH_OVERNIGHT_SEED:-7777}"
TOPK="${PATH_OVERNIGHT_TOPK:-3}"
SEARCH_OUT="$ROOT/artifacts/path_follower/prior_search_night"
SKIP="${PATH_OVERNIGHT_SKIP:-}"

# The endgame fix found by trace_prior_endgame.py: keep the drive alive once the
# arc-length remaining saturates, and home on the endpoint over the last 0.25 m.
# Without the blend distance this defaults to 2.0 m, which cuts the path and
# costs 23 points overall.
export PATH_PRIOR_ENDPOINT_FLOOR=1
export PATH_ENDPOINT_PP=1
export PATH_ENDPOINT_PP_DISTANCE=0.25
export PATH_SWEEP_MIN_BUCKET_N=16

has_stage() { [[ " $SKIP " != *" $1 "* ]]; }

echo "=== stage 1: search (budget ${BUDGET}s, ~$((BUDGET / 15)) evaluations)"
if has_stage 1; then
  PATH_SWEEP_MODE=search \
  PATH_SWEEP_LABEL=prior_search_night \
  PATH_SWEEP_OUTPUT="$SEARCH_OUT" \
  PATH_SEARCH_BUDGET_SECONDS="$BUDGET" \
  PATH_SEARCH_TOPK="$TOPK" \
    "$ROOT/run_local_sweep.sh" --num_envs 2048
else
  echo "skipped"
fi

echo
echo "=== stage 2: out-of-sample verification at seed $VERIFY_SEED"
if has_stage 2; then
  # Build the verification schedule from the top configurations the search kept,
  # plus the shipped defaults as a reference point.  Pure stdlib, so the system
  # python3 is enough.
  python3 - "$SEARCH_OUT/results.jsonl" "$SEARCH_OUT/verify_configs.json" "$TOPK" <<'PY'
import json, sys
results_path, out_path, topk = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(results_path) if line.strip()]
def score(entry):
    smoothed = entry.get("smoothed_worst_rate", entry["worst_bucket_rate"])
    return smoothed, entry["success_rate"]
ranked = sorted(rows, key=score, reverse=True)
schedule = [{"label": "defaults_reference"}]
seen = {frozenset()}
for entry in ranked:
    key = frozenset(entry["overrides"].items())
    if key in seen:
        continue
    seen.add(key)
    schedule.append({"label": "search_best_%d" % (len(schedule)), **entry["overrides"]})
    if len(schedule) > topk:
        break
json.dump(schedule, open(out_path, "w"), indent=2)
print("verification schedule:", [entry["label"] for entry in schedule])
PY

  for split in train test; do
    echo "--- split=$split"
    PATH_SWEEP_CONFIGS="$SEARCH_OUT/verify_configs.json" \
    PATH_SWEEP_LABEL="prior_verified_$split" \
    PATH_SWEEP_OUTPUT="$ROOT/artifacts/path_follower/prior_verified_$split" \
    PATH_EVAL_SEED="$VERIFY_SEED" \
    PATH_LAYOUT_SPLIT="$split" \
      "$ROOT/run_local_sweep.sh" --num_envs 2048
  done
else
  echo "skipped"
fi

echo
echo "=== stage 3: summary and commit"
if has_stage 3; then
  ( cd "$ROOT" && python3 legged_gym/scripts/summarize_path_evaluations.py ) || true
  ( cd "$ROOT" && git add -A && git commit -q -m "Overnight analytic-prior search results

Search winners: $SEARCH_OUT/summary.md
Out-of-sample verification: artifacts/path_follower/prior_verified_{train,test}/" \
      && git log --oneline -1 ) || echo "nothing to commit"
else
  echo "skipped"
fi

echo
echo "done. Read $SEARCH_OUT/summary.md first."
