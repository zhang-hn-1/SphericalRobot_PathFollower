"""Rebuild the NeuPAN evaluation inputs this machine is missing.

The two official evaluators read inputs that live only on the original
workstation (``/data/lzq/workspace/NeuPAN_official_repro_20260910``), so
neither can be re-run here as-is:

* ``evaluate_neupan_sstar.py`` reads ``s_star.npz`` (``[step, horizon, state]``).
* ``evaluate_neupan_executed_paths.py`` reads ``*_acker_official/executed_path.csv``.

Both evaluators, however, saved what they parsed into their own
``trajectories.npz``: ``desired`` is the arc-resampled path each one consumed.
That is enough to reconstruct equivalent inputs locally, which is what this
script does.  Provenance and the exact reconstruction, per family:

**Global** (``executed_path.csv``).  ``desired`` there is ``load_path()``'s
``xy_s``: the executed car path already resampled at the 0.05 m convention,
expressed in the frame of its own first state.  ``load_path()`` needs
``x, y, yaw``, and **only x/y survived into the archive** - the planner yaw
column was dropped.  Yaw therefore has to be reconstructed, and the choice
matters: the path is a polyline whose vertices sit every 0.4 m, so differencing
the tangents directly turns each vertex into a 2.9-4.5 1/m curvature spike,
against an archived planner peak of 0.21-0.52 1/m (a car cannot exceed ~0.5).
``savgol_yaw()`` instead smooths x(s), y(s) with a Savitzky-Golay filter and
differentiates; a 49-sample (2.45 m) window reproduces all five archived peaks
to within 5-11% with a single setting, which is the validation that the
reconstruction is on the planner's scale rather than the polyline's.  It is
still a reconstruction, not the planner yaw, so global numbers are indicative
rather than directly comparable to the archived ones.  ``--yaw tangent`` emits
the unsmoothed variant for a sensitivity check.

**Local** (``s_star.npz``).  ``evaluate_neupan_sstar.py``/``run_neupan_windows.py``
consume ``desired`` directly as a ``[n, M, 2]`` window array, so the ragged
windows only need padding to a common ``M``.  The official export pads exactly
this way - repeated goal states - and ``set_external_path`` resamples by arc
length, so a repeated final point adds a zero-length segment that is dropped.
Padding therefore reproduces the upstream convention rather than inventing one.

    python3 legged_gym/scripts/rebuild_neupan_local_inputs.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "path_follower"

# The archived run whose ``desired`` arrays are used as the reference.  All the
# neupan_global_* runs carry byte-identical references (checked in the summary),
# so any of them serves; this one is the plain unmodified baseline.
GLOBAL_SOURCE = ARTIFACTS / "neupan_global_v1" / "trajectories.npz"
# Second-generation windows: the first generation kept the raw 10 m horizon
# including its repeated-goal padding, which overstates every path as 10 m.
LOCAL_SOURCE_DIR = ARTIFACTS / "neupan_sstar_v2"


def unwrapped_heading(xy):
    delta = np.diff(xy, axis=0)
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    # One heading per segment; repeat the first so len(yaw) == len(xy).
    return np.r_[heading[0], heading]


# Validated against the archived planner peaks: with this window one setting
# reproduces convex/corridor/non_obs/pf/pf_obs = 0.49/0.22/0.51/0.44/0.58 1/m
# against archived 0.52/0.21/0.52/0.45/0.52.  See the module docstring.
SAVGOL_WINDOW = 49


def savgol_yaw(xy, window=SAVGOL_WINDOW, spacing=0.05):
    """Recover a planner-scale yaw from an xy-only polyline.

    Direct tangent differencing spikes at every polyline vertex (see the module
    docstring), so the positions are smoothed in arc length first.  The filter
    must be short relative to a manoeuvre and long relative to the vertex
    spacing; ``SAVGOL_WINDOW`` was chosen by matching the archived curvature
    peaks, not by eye.
    """
    from scipy.signal import savgol_filter

    if len(xy) < window:
        window = len(xy) - 1 if len(xy) % 2 == 0 else len(xy)
    window = max(window, 5)
    xs = savgol_filter(xy[:, 0], window, 3, mode="interp")
    ys = savgol_filter(xy[:, 1], window, 3, mode="interp")
    heading = np.unwrap(np.arctan2(np.gradient(ys, spacing), np.gradient(xs, spacing)))
    return heading


def rebuild_global(output_dir, yaw_mode="savgol"):
    data = np.load(GLOBAL_SOURCE, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    written = []
    for name, xy in zip(names, data["desired"]):
        xy = np.asarray(xy, dtype=np.float64)
        yaw = unwrapped_heading(xy) if yaw_mode == "tangent" else savgol_yaw(xy)
        scenario_dir = output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        csv_path = scenario_dir / "executed_path.csv"
        with csv_path.open("w") as handle:
            handle.write("x,y,yaw\n")
            for (x, y), psi in zip(xy, yaw):
                handle.write(f"{x:.9f},{y:.9f},{psi:.9f}\n")
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        written.append({"scenario": name, "points": int(len(xy)), "length_m": length,
                        "yaw_mode": yaw_mode,
                        "reconstructed_peak_curvature_1pm":
                            float(np.abs(np.gradient(yaw, 0.05)).max()),
                        "csv": str(csv_path.relative_to(PROJECT_ROOT))})
    return written


def rebuild_local(output_dir):
    written = []
    for source in sorted(LOCAL_SOURCE_DIR.glob("*_trajectories.npz")):
        data = np.load(source, allow_pickle=True)
        scenario = source.name[: -len("_trajectories.npz")]
        windows = [np.asarray(w, dtype=np.float64) for w in data["desired"]]
        indices = np.asarray(data["source_indices"]) if "source_indices" in data.files \
            else np.arange(len(windows))
        points = max(len(w) for w in windows)
        padded = np.zeros((len(windows), points, 2), dtype=np.float32)
        for row, window in enumerate(windows):
            padded[row, : len(window)] = window
            padded[row, len(window):] = window[-1]  # official padding convention
        lengths = np.array([np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
                            for w in windows])
        out = output_dir / f"{scenario}_windows.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, desired=padded, source_indices=indices)
        written.append({"scenario": scenario, "windows": len(windows),
                        "padded_points": int(points),
                        "length_m_median": float(np.median(lengths)),
                        "length_m_max": float(lengths.max()),
                        "npz": str(out.relative_to(PROJECT_ROOT))})
    return written


# ``peak_abs_curvature_1pm`` from the archived neupan_global_v1/metrics.json.
# The yaw reconstruction is validated against these, since it is the only
# curvature reference the archive still carries.
ARCHIVED_PEAK_CURVATURE = {
    "convex_obs_acker_official": 0.5191359082183047,
    "corridor_acker_official": 0.20880085578357382,
    "non_obs_acker_official": 0.5191359082183175,
    "pf_acker_official": 0.4467184386945013,
    "pf_obs_acker_official": 0.5191359082183578,
}


def main():
    global_out = ARTIFACTS / "neupan_global_inputs"
    tangent_out = ARTIFACTS / "neupan_global_inputs_tangent"
    local_out = ARTIFACTS / "neupan_local_inputs"
    global_rows = rebuild_global(global_out, "savgol")
    tangent_rows = rebuild_global(tangent_out, "tangent")
    local_rows = rebuild_local(local_out)
    manifest = {"global_source": str(GLOBAL_SOURCE.relative_to(PROJECT_ROOT)),
                "local_source": str(LOCAL_SOURCE_DIR.relative_to(PROJECT_ROOT)),
                "savgol_window": SAVGOL_WINDOW,
                "archived_peak_curvature_1pm": ARCHIVED_PEAK_CURVATURE,
                "global": global_rows, "global_tangent": tangent_rows,
                "local": local_rows}
    (ARTIFACTS / "neupan_rebuilt_inputs.json").write_text(json.dumps(manifest, indent=2))

    tangent_by_name = {row["scenario"]: row for row in tangent_rows}
    print("rebuilt global executed paths ->", global_out)
    print("  %-28s %5s %8s %10s %10s %10s" % ("scenario", "pts", "length_m",
                                                "peak_k", "archived", "tangent_k"))
    for row in global_rows:
        archived = ARCHIVED_PEAK_CURVATURE.get(row["scenario"], float("nan"))
        print("  %-28s %5d %8.2f %10.3f %10.3f %10.3f"
              % (row["scenario"], row["points"], row["length_m"],
                 row["reconstructed_peak_curvature_1pm"], archived,
                 tangent_by_name[row["scenario"]]["reconstructed_peak_curvature_1pm"]))
    print("rebuilt local S* windows ->", local_out)
    for row in local_rows:
        print("  %-28s %4d windows  padded to %3d pts  len median %5.2f m max %5.2f m"
              % (row["scenario"], row["windows"], row["padded_points"],
                 row["length_m_median"], row["length_m_max"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
