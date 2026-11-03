#!/usr/bin/env python3
"""Calibrate split-conformal prediction radii for the trajectory predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.learned_predictor import HORIZONS, HistoryPredictor, POSITION_SCALE  # noqa: E402
from train_trajectory_predictor import (WAYPOINTS, metrics, sample_windows, stack)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-trajectories", type=int, default=60)
    parser.add_argument("--samples-per-trajectory", type=int, default=100)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--alphas", default="0.10,0.05")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    config = payload.get("config", {})
    use_absolute = bool(config.get("use_absolute", True))
    model = HistoryPredictor(
        input_dim=int(config.get("input_dim", 9)),
        context_dim=int(config.get("context_dim", 4)),
        hidden=int(config.get("hidden", 64)),
        horizons=int(config.get("horizons", len(HORIZONS))))
    model.load_state_dict(state)
    model.eval()
    windows = []
    for index in range(args.val_trajectories):
        windows.extend(sample_windows(
            args.seed + index, args.samples_per_trajectory,
            route=WAYPOINTS, use_absolute=use_absolute))
    data = stack(windows)
    _result, prediction, error = metrics(model, data, device=torch.device("cpu"))
    alphas = [float(value) for value in args.alphas.split(",")]
    radii = {}
    for alpha in alphas:
        radii[f"{1.0 - alpha:.2f}"] = {
            f"{horizon:.1f}s": round(float(np.quantile(error[:, index], 1.0 - alpha)), 4)
            for index, horizon in enumerate(HORIZONS)
        }
    output = dict(checkpoint=str(args.checkpoint), use_absolute=use_absolute,
                  samples=len(windows), alphas=alphas, radii=radii)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
