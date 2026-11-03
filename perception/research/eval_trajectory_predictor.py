#!/usr/bin/env python3
"""Evaluate a trajectory predictor on seen and unseen routes.

The checkpoint's ``use_absolute`` flag decides the input layout, so the same
script can score the route-aware model and the route-blind ablation.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/eval_trajectory_predictor.py \
        --checkpoint /home/jason/perception_data/runs/trajectory_predictor_v2/predictor.pt \
        --out /home/jason/perception_data/runs/trajectory_predictor_eval_absolute.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.learned_predictor import (  # noqa: E402
    CONTEXT_DIM, HORIZONS, INPUT_DIM, RELATIVE_CONTEXT_DIM,
    RELATIVE_INPUT_DIM, HistoryPredictor)
from train_trajectory_predictor import (  # noqa: E402
    UNSEEN_ROUTES, WAYPOINTS, metrics, sample_windows, stack)


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    use_absolute = bool(config.get("use_absolute", True))
    model = HistoryPredictor(
        input_dim=int(config.get(
            "input_dim", INPUT_DIM if use_absolute else RELATIVE_INPUT_DIM)),
        context_dim=int(config.get(
            "context_dim", CONTEXT_DIM if use_absolute else RELATIVE_CONTEXT_DIM)),
        hidden=int(config.get("hidden", 64)),
        horizons=int(config.get("horizons", len(HORIZONS))),
    )
    model.load_state_dict(state)
    return model.to(device).eval(), use_absolute, config


def evaluate_route(model, route, use_absolute, trajectories, samples,
                   seed, device):
    windows = []
    for index in range(trajectories):
        windows.extend(sample_windows(
            seed + index, samples, route=route,
            use_absolute=use_absolute))
    if not windows:
        return dict(error="no samples")
    data = stack(windows)
    result, _prediction, _error = metrics(model, data, device)
    result["samples"] = len(windows)
    result["route_waypoints"] = [
        [round(float(point[0]), 3), round(float(point[1]), 3)]
        for point in route
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-trajectories", type=int, default=60)
    parser.add_argument("--samples-per-trajectory", type=int, default=100)
    parser.add_argument("--seed", type=int, default=900000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_absolute, config = load_model(args.checkpoint, device)
    routes = {"same": WAYPOINTS, **UNSEEN_ROUTES}
    results = {}
    for name, route in routes.items():
        results[name] = evaluate_route(
            model, route, use_absolute, args.val_trajectories,
            args.samples_per_trajectory, args.seed, device)
    payload = dict(
        checkpoint=str(args.checkpoint),
        use_absolute=use_absolute,
        config=config,
        routes=results,
    )
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
