#!/usr/bin/env python3
"""Train a tiny candidate scorer (VideoJev-lite) for target association.

Features are computed once per detection from the target track state, then a
linear model scores "this detection belongs to the target".  The model can be
loaded by ``TargetPerceptionLayer(candidate_scorer=...)``.

    python3 perception/research/train_candidate_scorer.py \
        --out /home/jason/perception_data/runs/candidate_scorer_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.eval_target_perception import (  # noqa: E402
    Scenario, _appearance, _bbox_for, _distractor_position,
    _target_position)
from perception.target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402


def hard_scenarios():
    base = [
        Scenario("similar_crossing", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.5),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.5),
                 similar_appearance=True),
        Scenario("similar_crossing_fast", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.8),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.8),
                 similar_appearance=True),
        Scenario("similar_crossing_noisy", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.55),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.55),
                 similar_appearance=True),
        Scenario("crossing", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.5),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.5)),
        Scenario("dropout_1s", target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(3.0, 4.0)),
    ]
    return base


def collect(scenarios, seeds, noise):
    rows, labels = [], []
    for scenario in scenarios:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            target_appearance = np.array([1.0, 0.0, 0.0])
            distractor_appearance = (np.array([0.98, 0.20, 0.0])
                                     if scenario.similar_appearance
                                     else np.array([0.0, 1.0, 0.0]))
            distractor_appearance = (
                distractor_appearance / np.linalg.norm(distractor_appearance))
            layer = TargetPerceptionLayer(
                init_bbox=_bbox_for(_target_position(scenario, 0.0)),
                init_appearance=target_appearance,
                max_prediction_s=1.0, max_memory_s=5.0, accel_std=0.6)
            steps = int(round(scenario.duration / 0.1)) + 1
            for step in range(steps):
                t = round(step * 0.1, 6)
                target_xy = _target_position(scenario, t)
                distractor_xy = _distractor_position(scenario, t)
                target_visible = (not scenario.target_dropout
                                  or not (scenario.target_dropout[0] <= t
                                          < scenario.target_dropout[1]))
                detections = []
                is_target = []
                if target_visible:
                    detections.append(PersonDetection(
                        position_xy=target_xy + rng.normal(0.0, noise, 2),
                        bbox=_bbox_for(target_xy),
                        appearance=_appearance(target_appearance, rng),
                        score=float(np.clip(0.9 + rng.normal(0.0, 0.01),
                                            0.05, 0.999))))
                    is_target.append(True)
                detections.append(PersonDetection(
                    position_xy=distractor_xy + rng.normal(0.0, noise, 2),
                    bbox=_bbox_for(distractor_xy),
                    appearance=_appearance(distractor_appearance, rng),
                    score=float(np.clip(0.9 + rng.normal(0.0, 0.01),
                                        0.05, 0.999))))
                is_target.append(False)
                track = layer._target_track()
                if track is not None:
                    for index, detection in enumerate(detections):
                        rows.append(layer._candidate_features(
                            track, detection, t))
                        labels.append(1.0 if is_target[index] else 0.0)
                layer.update(detections, t)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def fit_logistic(features, labels, epochs=500, lr=0.1):
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    x = (features - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = x @ weights + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = probability - labels
        weights -= lr * (x.T @ error / len(x) + 1e-4 * weights)
        bias -= lr * error.mean()
    return mean, scale, weights, bias


def accuracy(features, labels, params):
    mean, scale, weights, bias = params
    logits = ((features - mean) / scale) @ weights + bias
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    prediction = probability >= 0.5
    target = labels >= 0.5
    tp = int(np.sum(prediction & target))
    tn = int(np.sum(~prediction & ~target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    return dict(accuracy=round((tp + tn) / max(1, len(labels)), 4),
                precision=round(tp / max(1, tp + fp), 4),
                recall=round(tp / max(1, tp + fn), 4),
                positive= int(np.sum(target)), negative=int(np.sum(~target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--train-seeds", type=int, default=50)
    parser.add_argument("--val-seeds", type=int, default=50)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = collect(hard_scenarios(), range(args.train_seeds), args.noise)
    val = collect(hard_scenarios(), range(1000, 1000 + args.val_seeds),
                  args.noise)
    params = fit_logistic(*train)
    train_metrics = accuracy(train[0], train[1], params)
    val_metrics = accuracy(val[0], val[1], params)
    payload = dict(
        mean=params[0].tolist(), scale=params[1].tolist(),
        weights=params[2].tolist(), bias=float(params[3]),
        train=train_metrics, val=val_metrics,
        positives=int(train[1].sum()), total=int(len(train[1])),
    )
    (out / "scorer.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(dict(train=train_metrics, val=val_metrics,
                          positives=payload["positives"],
                          total=payload["total"]), indent=2))


if __name__ == "__main__":
    main()
