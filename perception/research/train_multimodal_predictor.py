#!/usr/bin/env python3
"""Trajectron++-style multimodal trajectory predictor (lightweight MDN).

This is a deliberate reduction of Trajectron++: same history encoder and data
as the single-mode predictor, but with K trajectory modes and a mode
probability.  It is used to check whether multimodality helps before adding a
full graph/heterogeneous-agent implementation.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/train_multimodal_predictor.py \
        --out /home/jason/perception_data/runs/multimodal_predictor_v1 \
        --modes 3 --epochs 30
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
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.learned_predictor import (  # noqa: E402
    CONTEXT_DIM, HORIZONS, INPUT_DIM, POSITION_SCALE)
from train_trajectory_predictor import (  # noqa: E402
    UNSEEN_ROUTES, WAYPOINTS, sample_windows, stack)


class MixturePredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, context_dim=CONTEXT_DIM,
                 hidden=64, modes=3, horizons=len(HORIZONS)):
        super().__init__()
        self.modes = modes
        self.horizons = horizons
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden), nn.ELU(),
            nn.Linear(hidden, 2 * modes * horizons + modes))

    def forward(self, sequence, context):
        encoded, _ = self.gru(sequence)
        output = self.head(torch.cat([encoded[:, -1], context], dim=-1))
        modes = output[:, :2 * self.modes * self.horizons].reshape(
            -1, self.modes, self.horizons, 2)
        logits = output[:, 2 * self.modes * self.horizons:]
        return modes, logits


def predict(model, sequence, context, device):
    model.eval()
    with torch.no_grad():
        modes, logits = model(
            torch.from_numpy(sequence).to(device),
            torch.from_numpy(context).to(device))
    return (modes.cpu().numpy() * POSITION_SCALE,
            logits.cpu().numpy())


def evaluate(model, data, device, sigma=0.25):
    sequence, context, target, baseline, baseline_ls, _corner = data
    modes, logits = predict(model, sequence, context, device)
    target = target[:, None, :, :]
    distance = np.linalg.norm(modes - target, axis=3)
    best = distance.min(axis=1)
    chosen = distance[np.arange(len(target)), logits.argmax(axis=1)]
    result = {}
    for index, horizon in enumerate(HORIZONS):
        result["%.1fs" % horizon] = {
            "min_ade_m": round(float(best[:, index].mean()), 4),
            "chosen_mode_ade_m": round(float(chosen[:, index].mean()), 4),
        }
    result["min_ade_overall_m"] = round(float(best.mean()), 4)
    result["chosen_mode_ade_overall_m"] = round(float(chosen.mean()), 4)
    return result


def train(model, data, epochs, batch_size, lr, device, sigma=0.25):
    sequence, context, target, _baseline, _baseline_ls, _corner = data
    obs_t = torch.from_numpy(sequence).to(device)
    ctx_t = torch.from_numpy(context).to(device)
    target_t = torch.from_numpy(target).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(obs_t))
        total = 0.0
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            modes, logits = model(obs_t[batch], ctx_t[batch])
            target_b = target_t[batch].unsqueeze(1)
            distance = ((modes - target_b) ** 2).sum(dim=3)
            logp = (-0.5 * distance / (sigma ** 2)
                    - math.log(2.0 * math.pi * sigma ** 2))
            logmix = F.log_softmax(logits, dim=1).unsqueeze(2)
            loss = -torch.logsumexp(logmix + logp, dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        print(json.dumps(dict(epoch=epoch,
                              loss=round(total / len(permutation), 6))),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--train-trajectories", type=int, default=300)
    parser.add_argument("--val-trajectories", type=int, default=60)
    parser.add_argument("--samples-per-trajectory", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--no-absolute", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_absolute = not args.no_absolute
    input_dim = INPUT_DIM if use_absolute else 6
    context_dim = CONTEXT_DIM if use_absolute else 2

    train_samples, val_samples = [], []
    for index in range(args.train_trajectories):
        train_samples.extend(sample_windows(
            args.seed + index, args.samples_per_trajectory,
            use_absolute=use_absolute))
    for index in range(args.val_trajectories):
        val_samples.extend(sample_windows(
            args.seed + 100000 + index, args.samples_per_trajectory,
            use_absolute=use_absolute))
    train_data = stack(train_samples)
    val_data = stack(val_samples)
    model = MixturePredictor(
        input_dim=input_dim, context_dim=context_dim,
        hidden=args.hidden, modes=args.modes).to(device)
    train(model, train_data, args.epochs, args.batch_size, args.lr, device)
    results = {}
    for name, route in {"same": WAYPOINTS, **UNSEEN_ROUTES}.items():
        samples = []
        for index in range(args.val_trajectories):
            samples.extend(sample_windows(
                args.seed + 300000 + index, args.samples_per_trajectory,
                route=route, use_absolute=use_absolute))
        results[name] = evaluate(model, stack(samples), device)
    payload = dict(
        model=model.state_dict(),
        config=dict(input_dim=input_dim, context_dim=context_dim,
                    hidden=args.hidden, modes=args.modes,
                    horizons=list(HORIZONS), use_absolute=use_absolute,
                    position_scale=POSITION_SCALE),
    )
    torch.save(payload, out / "multimodal.pt")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(json.dumps(payload["config"], indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
