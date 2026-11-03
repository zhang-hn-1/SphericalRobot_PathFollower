#!/usr/bin/env python3
"""A tiny learned candidate scorer for target association.

This is the lightweight analogue of a video candidate-scoring head: one scalar
score from a handful of per-candidate features (appearance, motion, score,
box-size consistency).  It is deliberately a linear model, so it can run on
the camera-side CPU without a GPU.

Training is in ``research/train_candidate_scorer.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "appearance_anchor",
    "appearance_target",
    "motion_normalized",
    "distance_over_gate",
    "gap_s",
    "detection_score",
    "area_ratio",
    "close_target",
)


class CandidateScorer:
    def __init__(self, mean, scale, weights, bias):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.bias = float(bias)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        return cls(payload["mean"], payload["scale"],
                   payload["weights"], payload["bias"])

    def __call__(self, features):
        return self.predict(features)

    def predict(self, features):
        value = np.asarray(features, dtype=np.float64).reshape(-1)
        if value.size != self.weights.size:
            raise ValueError("candidate feature size mismatch")
        normalized = (value - self.mean) / np.maximum(self.scale, 1e-9)
        logit = float(np.dot(normalized, self.weights) + self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
