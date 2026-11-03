#!/usr/bin/env python3
"""Supervised future-position predictor for a lost target.

The predictor consumes the last measured world-frame positions and predicts
the target displacement at 0.5, 1.0 and 2.0 s.  It is deliberately separate
from the RL controller: the supervised model learns target dynamics, while the
controller learns whether to follow the prediction and how fast.

Training lives in ``research/train_trajectory_predictor.py``.  The checkpoint
contains the model state and its small config, so inference has no dependency
on the training script.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HORIZONS = (0.5, 1.0, 2.0)
HISTORY_STEPS = 12
INPUT_DIM = 9
CONTEXT_DIM = 4
RELATIVE_INPUT_DIM = 6
RELATIVE_CONTEXT_DIM = 2
POSITION_SCALE = 2.0


def _as_xy(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array[:2]).all():
        return None
    return array[:2].copy()


def _history_arrays(history):
    """Return sorted (times, positions) from either dicts or (t, xy)."""
    times, positions = [], []
    for item in history or []:
        if isinstance(item, dict):
            when = item.get("t")
            where = item.get("position_xy")
        else:
            try:
                when, where = item
            except (TypeError, ValueError):
                continue
        point = _as_xy(where)
        if when is None or point is None:
            continue
        times.append(float(when))
        positions.append(point)
    if not times:
        return np.zeros(0), np.zeros((0, 2))
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(positions)[order]


def make_sequence(history, last_velocity=None, use_absolute=True):
    """Build the fixed-length GRU input and the last-step context.

    Missing history is padded with the last measured point.  The last input
    row has valid=1 and zero relative displacement.  Features are bounded so
    the network cannot be dominated by a single noisy detection.

    ``use_absolute=False`` removes the world-position features and the
    absolute anchor from the context.  The resulting model is the
    route-blind ablation: it can only use relative history and velocity.
    """
    times, positions = _history_arrays(history)
    if len(times) == 0:
        return None, None
    times = times[-HISTORY_STEPS:]
    positions = positions[-HISTORY_STEPS:]
    anchor_time = float(times[-1])
    anchor = positions[-1]
    input_dim = INPUT_DIM if use_absolute else RELATIVE_INPUT_DIM
    steps = np.zeros((HISTORY_STEPS, input_dim), dtype=np.float32)
    start = HISTORY_STEPS - len(times)
    for index in range(len(times)):
        relative_t = float(times[index] - anchor_time)
        delta = (positions[index] - anchor) / POSITION_SCALE
        if index > 0:
            dt = max(1e-6, float(times[index] - times[index - 1]))
            velocity = (positions[index] - positions[index - 1]) / dt
        else:
            velocity = np.zeros(2)
        if last_velocity is not None and index == len(times) - 1:
            velocity = np.asarray(last_velocity, dtype=np.float64).reshape(2)
        features = [
            float(np.clip(delta[0], -2.0, 2.0)),
            float(np.clip(delta[1], -2.0, 2.0)),
            float(np.clip(velocity[0], -2.0, 2.0)),
            float(np.clip(velocity[1], -2.0, 2.0)),
            float(np.clip(relative_t, -2.0, 2.0)),
        ]
        if use_absolute:
            features.extend([
                float(np.clip(positions[index, 0] / 3.0, -2.0, 2.0)),
                float(np.clip(positions[index, 1] / 3.0, -2.0, 2.0)),
                1.0,
            ])
        features.append(float(1.0 if index == len(times) - 1 else 0.0))
        steps[start + index] = np.asarray(features, dtype=np.float32)
    del start
    if last_velocity is None:
        velocity = np.zeros(2)
    else:
        velocity = np.asarray(last_velocity, dtype=np.float64).reshape(2)
    context_values = []
    if use_absolute:
        context_values.extend([
            float(np.clip(anchor[0] / 3.0, -2.0, 2.0)),
            float(np.clip(anchor[1] / 3.0, -2.0, 2.0)),
        ])
    context_values.extend([
        float(np.clip(velocity[0], -2.0, 2.0)),
        float(np.clip(velocity[1], -2.0, 2.0)),
    ])
    context = np.asarray(context_values, dtype=np.float32)
    return steps, context


class HistoryPredictor(nn.Module):
    """Small GRU regressor: history + route/position context -> displacement."""

    def __init__(self, input_dim=INPUT_DIM, context_dim=CONTEXT_DIM,
                 hidden=64, horizons=len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden), nn.ELU(),
            nn.Linear(hidden, 2 * horizons),
        )

    def forward(self, sequence, context):
        encoded, _ = self.gru(sequence)
        return self.head(torch.cat([encoded[:, -1], context], dim=-1))


class TrajectoryPredictor:
    """Inference wrapper with interpolation and a safe analytic fallback."""

    def __init__(self, checkpoint=None, device="cpu"):
        self.device = torch.device(device)
        state = {}
        config = {}
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu")
            if isinstance(payload, dict) and "model" in payload:
                state = payload["model"]
                config = payload.get("config", {})
            else:
                state = payload
        self.use_absolute = bool(config.get("use_absolute", True))
        self.model = HistoryPredictor(
            input_dim=int(config.get(
                "input_dim", INPUT_DIM if self.use_absolute
                else RELATIVE_INPUT_DIM)),
            context_dim=int(config.get(
                "context_dim", CONTEXT_DIM if self.use_absolute
                else RELATIVE_CONTEXT_DIM)),
            hidden=int(config.get("hidden", 64)),
            horizons=int(config.get("horizons", len(HORIZONS))),
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict_displacements(self, history, last_velocity=None):
        sequence, context = make_sequence(
            history, last_velocity, use_absolute=self.use_absolute)
        if sequence is None:
            return None
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(sequence).unsqueeze(0).to(self.device),
                torch.from_numpy(context).unsqueeze(0).to(self.device),
            )
        values = output[0].detach().cpu().numpy().reshape(-1, 2)
        return values * POSITION_SCALE

    def predict_at(self, history, horizon_s, last_velocity=None):
        """World displacement from the last measured point at ``horizon_s``."""
        times, positions = _history_arrays(history)
        if len(times) == 0:
            return None
        horizon = max(0.0, float(horizon_s))
        if horizon <= 1e-9:
            return np.zeros(2)
        displacements = self.predict_displacements(history, last_velocity)
        if displacements is None:
            return None
        if horizon <= HORIZONS[0]:
            scale = horizon / HORIZONS[0]
            return displacements[0] * scale
        if horizon >= HORIZONS[-1]:
            return displacements[-1].copy()
        for index in range(len(HORIZONS) - 1):
            low, high = HORIZONS[index], HORIZONS[index + 1]
            if low <= horizon <= high:
                alpha = (horizon - low) / (high - low)
                return displacements[index] * (1.0 - alpha) + \
                    displacements[index + 1] * alpha
        return displacements[-1].copy()

    def predict_belief(self, belief, horizon_s=None):
        """Predict a future world point from a ``TargetBelief``.

        ``belief.trajectory_history`` contains measured positions only, so the
        anchor is the last measurement.  If the model is unavailable or the
        history is too short, fall back to the layer's constant-velocity
        prediction; this keeps the integration safe in live use.
        """
        history = getattr(belief, "trajectory_history", None)
        times, positions = _history_arrays(history)
        last_velocity = getattr(belief, "velocity_xy", None)
        if len(times) < 2:
            if getattr(belief, "position_xy", None) is None:
                return None
            return _as_xy(belief.position_xy)
        anchor = positions[-1]
        age = (getattr(belief, "measurement_age_s", None)
               if horizon_s is None else horizon_s)
        if age is None:
            age = 0.0
        if age <= 1e-9:
            return anchor
        displacement = self.predict_at(history, age, last_velocity)
        if displacement is None:
            if last_velocity is None:
                return anchor
            return anchor + np.asarray(last_velocity) * age
        return anchor + displacement
