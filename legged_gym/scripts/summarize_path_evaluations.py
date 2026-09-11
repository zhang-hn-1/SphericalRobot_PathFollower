"""Aggregate every path-follower evaluation into one results table.

Walks ``artifacts/path_follower/**/metrics.json`` and writes a markdown report
whose central table is broken down by (path type x curvature).  That breakdown
is the point: the curriculum gate and the earlier evaluations only reported
path type, and a path-type average hides a broken curvature behind two working
ones (right arc at k=-0.25 scores 100% while k=-0.333 scores 6.6%, yet both are
one ``right_arc`` number).

Curvature is recovered two ways:

* evaluations run after the breakdown was added record ``by_type_curvature``;
* archived evaluations that forced a single curvature record it in
  ``forced_curvature`` (and in the output directory name), which recovers the
  bucket exactly.  Episodes from ``mixed``/``random`` runs before the change
  have no recorded curvature and are reported only in the per-run table.

Pure stdlib, no Isaac Gym.  Run:

    python3 legged_gym/scripts/summarize_path_evaluations.py [output.md]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "path_follower" / "RESULTS_SUMMARY.md"

RUN_RE = re.compile(r"logs/rotunbot_path/([^/]+)/model_(\d+)")
# Newer evaluations append the layout split: stage_2_right_arc_k-0p3333_test
CURVATURE_IN_DIR_RE = re.compile(r"_k(-?\d+(?:p\d+)?)(?:_(?:train|test))?$")
CAMEL_STAGE_RE = re.compile(r"^Sep\d\d_\d\d-\d\d-\d\d_")


def short_run_name(run):
    name = CAMEL_STAGE_RE.sub("", run)
    for prefix in ("geometric_path_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_from_scratch", "").replace("_continual", "")


def curvature_from_directory(directory):
    match = CURVATURE_IN_DIR_RE.search(directory.name)
    if match is None:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def path_type_from_directory(directory):
    match = re.search(r"_(straight|left_arc|right_arc|s_curve)_k", directory.name)
    return match.group(1) if match else None


def load_records():
    records = []
    for path in sorted(ROOT.glob("artifacts/path_follower/**/metrics.json")):
        if "neupan" in str(path):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        match = RUN_RE.search(data.get("checkpoint", ""))
        if match is None:
            continue
        forced_curvature = data.get("forced_curvature")
        bucket_curvature = None
        if forced_curvature not in (None, "random", "None"):
            try:
                bucket_curvature = float(forced_curvature)
            except (TypeError, ValueError):
                bucket_curvature = None
        if bucket_curvature is None:
            bucket_curvature = curvature_from_directory(path.parent)
        forced_type = data.get("forced_path_type") or path_type_from_directory(path.parent)
        # Directory names recorded the arc curvature as a magnitude for part of
        # the project's history (stage_1_right_arc_k0p25) and as a signed value
        # later (stage_2_right_arc_k-0p3333333333).  A right arc is negative by
        # construction, so normalise the sign before bucketing or the same case
        # splits across two rows.
        if bucket_curvature is not None and forced_type == "right_arc" and bucket_curvature > 0:
            bucket_curvature = -bucket_curvature
        records.append({
            "run": short_run_name(match.group(1)),
            "iteration": int(match.group(2)),
            "stage": data.get("stage"),
            "layout_split": data.get("layout_split", "train"),
            "episodes": data.get("episodes"),
            "success": data.get("success_rate"),
            "forced_path_type": forced_type,
            "curvature": bucket_curvature,
            "by_type_curvature": data.get("by_type_curvature"),
            "failure_reasons": data.get("failure_reasons") or {},
            "stochastic": data.get("stochastic_actions"),
        })
    return records


def best_per_bucket(records):
    buckets = defaultdict(list)
    for record in records:
        if record["by_type_curvature"]:
            for key, entry in record["by_type_curvature"].items():
                buckets[(entry["path_type"], round(entry["curvature"], 4))].append({
                    "run": record["run"],
                    "iteration": record["iteration"],
                    "success": entry["success_rate"],
                    "episodes": entry["episodes"],
                    "split": record["layout_split"],
                })
        elif record["curvature"] is not None and record["forced_path_type"]:
            buckets[(record["forced_path_type"], round(record["curvature"], 4))].append({
                "run": record["run"],
                "iteration": record["iteration"],
                "success": record["success"],
                "episodes": record["episodes"],
                "split": record["layout_split"],
            })
    return buckets


def write_report(records, output):
    buckets = best_per_bucket(records)
    lines = []
    lines.append("# Rotunbot path-follower results summary")
    lines.append("")
    lines.append("Generated by `legged_gym/scripts/summarize_path_evaluations.py` from")
    lines.append(f"{len(records)} evaluation runs under `artifacts/path_follower/`.")
    lines.append("")
    lines.append("Success requires all three of endpoint distance <= 0.20 m,")
    lines.append("remaining path <= 0.20 m, and terminal speed <= 0.10 m/s.")
    lines.append("")
    lines.append("## Best result per (path type x curvature)")
    lines.append("")
    lines.append("Curvature is signed; positive is a left arc, negative a right arc.")
    lines.append("`n` is the episode count of the best run, `split` its layout split.")
    lines.append("")
    lines.append("| path type | curvature | best success | run | iter | n | split |")
    lines.append("|---|---|---|---|---|---|---|")
    for (path_type, curvature) in sorted(buckets, key=lambda k: (k[0], abs(k[1]), k[1])):
        entries = sorted(buckets[(path_type, curvature)], key=lambda e: e["success"], reverse=True)
        best = entries[0]
        lines.append(
            f"| {path_type} | {curvature:+.4g} | **{best['success']:.1%}** | "
            f"{best['run']} | {best['iteration']} | {best['episodes']} | {best['split']} |"
        )
    lines.append("")
    lines.append("## Per-run detail")
    lines.append("")
    lines.append("| run | iter | stage | case | split | n | success | failure reasons |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for record in sorted(records, key=lambda r: (r["run"], r["iteration"])):
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(record["failure_reasons"].items())
            if name != "success"
        ) or "-"
        case = record["forced_path_type"] or "mixed"
        if record["curvature"] is not None:
            case = f"{case} k={record['curvature']:+.4g}"
        if record["stochastic"]:
            case += " (sampled)"
        lines.append(
            f"| {record['run']} | {record['iteration']} | {record['stage']} | "
            f"{case} | {record['layout_split']} | {record['episodes']} | "
            f"{record['success']:.1%} | {reasons} |"
        )
    lines.append("")

    totals = Counter()
    for record in records:
        for name, count in record["failure_reasons"].items():
            if name != "success":
                totals[name] += count
    lines.append("## Failure taxonomy (all evaluations pooled)")
    lines.append("")
    lines.append("| reason | episodes |")
    lines.append("|---|---|")
    for name, count in totals.most_common():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## Caveats to read these numbers with")
    lines.append("")
    lines.append("- `split=train` is the distribution the policies were trained on. A run")
    lines.append("  with no `test` row has no measured generalisation.")
    lines.append("- Episode counts of 32-128 give roughly +/- 8-15 point confidence")
    lines.append("  intervals; only compare runs at equal n.")
    lines.append("- Successful arc episodes end at a median endpoint distance of about")
    lines.append("  0.199 m against a 0.20 m threshold, so arc success is")
    lines.append("  threshold-hugging rather than comfortably accurate.")
    lines.append("- Failures are bimodal: either the episode tracks the path, or it")
    lines.append("  reaches the 1.50 m cross-track termination. There is little in between.")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    records = load_records()
    if not records:
        print("no evaluations found", file=sys.stderr)
        return 1
    path = write_report(records, output)
    print(f"wrote {path} from {len(records)} evaluations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
