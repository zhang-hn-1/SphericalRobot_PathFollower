#!/usr/bin/env python3
"""Train a supervised future-position predictor from target history.

Data are generated from the same rectangular route used by the stage demo,
with random direction, speed, phase and detection noise.  The model sees only
measured world positions (the same data available in ``TargetBelief``) and
predicts displacement at 0.5/1.0/2.0 s.  A held-out set of trajectory seeds
measures it against a constant-velocity extrapolation.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/train_trajectory_predictor.py \
        --out /home/jason/perception_data/runs/trajectory_predictor_v1 \
        --epochs 30 --train-trajectories 300 --val-trajectories 60
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from perception.learned_predictor import (  # noqa: E402
    CONTEXT_DIM, HORIZONS, HISTORY_STEPS, INPUT_DIM, HistoryPredictor,
    POSITION_SCALE, make_sequence, _history_arrays)

WAYPOINTS = np.array([
    [-1.20, -1.10],
    [1.20, -1.10],
    [1.20, -3.20],
    [-1.20, -3.20],
], dtype=np.float64)
DT = 0.1
TARGET_SPEED = 0.40


def route_length():
    return float(sum(
        np.linalg.norm(WAYPOINTS[(i + 1) % len(WAYPOINTS)] - WAYPOINTS[i])
        for i in range(len(WAYPOINTS))))


def route_point(distance):
    points = WAYPOINTS
    total = route_length()
    distance = float(distance) % total
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if distance <= length:
            return start + segment / length * distance
        distance -= length
    return points[0].copy()


def make_trajectory(seed, duration=20.0):
    """Return clean positions and a measured/noisy version at 10 Hz."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, DT)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    base_speed = TARGET_SPEED * float(rng.uniform(0.70, 1.25))
    phase = float(rng.uniform(0.0, route_length()))
    # Smooth speed modulation: enough variation that a single last velocity is
    # not always the best predictor, but the route corners remain dominant.
    speed = base_speed * (
        1.0 + 0.12 * np.sin(2.0 * np.pi * times / 5.0 + rng.uniform(0, 6.28)))
    speed = np.clip(speed, 0.18, 0.80)
    travelled = np.cumsum(speed) * DT
    clean = np.asarray([
        route_point(phase + direction * value) for value in travelled])
    noise = float(rng.uniform(0.015, 0.075))
    measured = clean + rng.normal(0.0, noise, clean.shape)
    keep = rng.random(len(times)) > float(rng.uniform(0.0, 0.12))
    keep[0] = True
    measured = measured[keep]
    measured_times = times[keep]
    return times, clean, measured_times, measured


def interpolate(clean_times, clean, query):
    if query <= clean_times[0]:
        return clean[0].copy()
    if query >= clean_times[-1]:
        return clean[-1].copy()
    index = int(np.searchsorted(clean_times, query))
    low, high = index - 1, index
    span = clean_times[high] - clean_times[low]
    alpha = 0.0 if span <= 1e-12 else (query - clean_times[low]) / span
    return clean[low] * (1.0 - alpha) + clean[high] * alpha


def sample_windows(seed, count, duration=20.0):
    times, clean, measured_times, measured = make_trajectory(seed, duration)
    samples = []
    if len(measured_times) < HISTORY_STEPS + 2:
        return samples
    if len(measured_times) <= HISTORY_STEPS + 26:
        return samples
    for _ in range(count):
        anchor_index = int(np.random.default_rng(
            seed * 100003 + int(_)).integers(
                HISTORY_STEPS + 1, len(measured_times) - 25))
        anchor_time = float(measured_times[anchor_index])
        anchor = measured[anchor_index]
        history = list(zip(measured_times[:anchor_index + 1],
                           measured[:anchor_index + 1]))
        # Keep a maximum history window; the encoder pads short inputs.
        history = history[-HISTORY_STEPS:]
        if len(history) < 3:
            continue
        displacements = []
        valid = True
        for horizon in HORIZONS:
            future = anchor_time + horizon
            if future > times[-1]:
                valid = False
                break
            target = interpolate(times, clean, future)
            displacements.append(target - anchor)
        if not valid:
            continue
        anchor_velocity = np.zeros(2)
        smoothed_velocity = np.zeros(2)
        if len(history) >= 2:
            dt = max(1e-6, history[-1][0] - history[-2][0])
            anchor_velocity = (history[-1][1] - history[-2][1]) / dt
        if len(history) >= 3:
            history_times = np.asarray([item[0] for item in history])
            history_positions = np.asarray([item[1] for item in history])
            centered = history_times - history_times.mean()
            design = np.column_stack([centered, np.ones_like(centered)])
            smoothed_velocity = np.array([
                np.linalg.lstsq(
                    design, history_positions[:, axis], rcond=None)[0][0]
                for axis in range(2)
            ])
        sequence, context = make_sequence(history, anchor_velocity)
        if sequence is None:
            continue
        samples.append(dict(
            sequence=sequence,
            context=context,
            target=np.asarray(displacements, dtype=np.float32),
            baseline=np.asarray([
                anchor_velocity * horizon for horizon in HORIZONS
            ], dtype=np.float32),
            baseline_ls=np.asarray([
                smoothed_velocity * horizon for horizon in HORIZONS
            ], dtype=np.float32),
            corner=bool(min(
                np.linalg.norm(anchor - point) for point in WAYPOINTS) < 0.75),
        ))
    return samples


def stack(samples):
    return (np.asarray([item["sequence"] for item in samples]),
            np.asarray([item["context"] for item in samples]),
            np.asarray([item["target"] for item in samples]),
            np.asarray([item["baseline"] for item in samples]),
            np.asarray([item["baseline_ls"] for item in samples]),
            np.asarray([item["corner"] for item in samples], dtype=bool))


def metrics(model, data, device):
    sequence, context, target, baseline, baseline_ls, corner = data
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(sequence).to(device),
            torch.from_numpy(context).to(device),
        ).cpu().numpy().reshape(-1, len(HORIZONS), 2) * POSITION_SCALE
    error = np.linalg.norm(prediction - target, axis=2)
    baseline_error = np.linalg.norm(baseline - target, axis=2)
    baseline_ls_error = np.linalg.norm(baseline_ls - target, axis=2)
    result = {}
    for index, horizon in enumerate(HORIZONS):
        result["%.1fs" % horizon] = {
            "model_mae_m": round(float(error[:, index].mean()), 4),
            "constant_velocity_mae_m": round(
                float(baseline_error[:, index].mean()), 4),
            "smoothed_constant_velocity_mae_m": round(
                float(baseline_ls_error[:, index].mean()), 4),
            "corner_model_mae_m": (round(float(error[corner, index].mean()), 4)
                                   if corner.any() else None),
            "corner_constant_velocity_mae_m": (
                round(float(baseline_error[corner, index].mean()), 4)
                if corner.any() else None),
            "corner_fraction": round(float(corner.mean()), 4),
        }
    result["overall_model_mae_m"] = round(float(error.mean()), 4)
    result["overall_constant_velocity_mae_m"] = round(
        float(baseline_error.mean()), 4)
    result["overall_smoothed_constant_velocity_mae_m"] = round(
        float(baseline_ls_error.mean()), 4)
    return result, prediction, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-trajectories", type=int, default=300)
    parser.add_argument("--val-trajectories", type=int, default=60)
    parser.add_argument("--samples-per-trajectory", type=int, default=100)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_samples = []
    for index in range(args.train_trajectories):
        train_samples.extend(sample_windows(
            args.seed + index, args.samples_per_trajectory, args.duration))
    val_samples = []
    for index in range(args.val_trajectories):
        val_samples.extend(sample_windows(
            args.seed + 100000 + index, args.samples_per_trajectory,
            args.duration))
    if not train_samples or not val_samples:
        raise SystemExit("no samples generated")
    train = stack(train_samples)
    val = stack(val_samples)
    train_t = tuple(torch.from_numpy(item).to(device)
                    if item.dtype != bool else torch.from_numpy(item).to(device)
                    for item in train)
    val_t = tuple(torch.from_numpy(item).to(device)
                  if item.dtype != bool else torch.from_numpy(item).to(device)
                  for item in val)

    model = HistoryPredictor(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = None
    best_score = float("inf")
    history = []
    weights = torch.tensor([1.0, 1.0, 1.35], device=device)
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        permutation = torch.randperm(len(train_t[0]))
        total = 0.0
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start:start + args.batch_size]
            prediction = model(train_t[0][batch], train_t[1][batch])
            prediction = prediction.reshape(-1, len(HORIZONS), 2)
            target = train_t[2][batch] / POSITION_SCALE
            loss = (((prediction - target) ** 2).mean(dim=2)
                    * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        evaluation, _prediction, _error = metrics(
            model, val, device)
        score = evaluation["overall_model_mae_m"]
        entry = dict(epoch=epoch,
                     train_loss=round(total / len(permutation), 5),
                     val=evaluation,
                     seconds=round(time.time() - started, 1))
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score < best_score:
            best_score = score
            best = dict(
                model=model.state_dict(),
                config=dict(input_dim=INPUT_DIM, context_dim=CONTEXT_DIM,
                            hidden=args.hidden, horizons=len(HORIZONS)),
            )
            torch.save(best, out / "predictor.pt")
            (out / "best_metrics.json").write_text(json.dumps(
                dict(epoch=epoch, score=score, **evaluation), indent=2))
    if best is None:
        raise SystemExit("training produced no checkpoint")
    (out / "metrics.json").write_text(json.dumps(
        dict(train_trajectories=args.train_trajectories,
             val_trajectories=args.val_trajectories,
             train_samples=len(train_samples),
             val_samples=len(val_samples),
             best_score=best_score,
             history=history[-1:]), indent=2))
    (out / "config.json").write_text(json.dumps(dict(
        input_dim=INPUT_DIM, context_dim=CONTEXT_DIM, hidden=args.hidden,
        horizons=HORIZONS, history_steps=HISTORY_STEPS,
        position_scale=POSITION_SCALE, seed=args.seed,
        samples_per_trajectory=args.samples_per_trajectory,
        duration=args.duration), indent=2))
    print(json.dumps(dict(model=str(out / "predictor.pt"),
                          best=json.loads(
                              (out / "best_metrics.json").read_text())),
                     indent=2))


if __name__ == "__main__":
    main()
