#!/usr/bin/env python3
"""Compare target association with/without the tiny candidate scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.candidate_scorer import CandidateScorer  # noqa: E402
from perception.eval_target_perception import (  # noqa: E402
    run_scenario, scenarios)
from perception.research.train_candidate_scorer import hard_scenarios  # noqa: E402


def aggregate(runs):
    frames = sum(run.frames for run in runs)
    correct = sum(run.identity_correct for run in runs)
    wrong = sum(run.wrong for run in runs)
    return dict(frames=frames, identity=round(correct / max(1, frames), 5),
                wrong=wrong)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--weight", type=float, default=0.25)
    args = parser.parse_args()
    scorer = CandidateScorer.load(args.scorer)
    scene_list = scenarios() + hard_scenarios()
    results = {}
    for scenario in scene_list:
        base = [run_scenario(scenario, seed, position_noise=args.noise)
                for seed in range(1000, 1000 + args.seeds)]
        scored = [run_scenario(
            scenario, seed, position_noise=args.noise,
            candidate_scorer=scorer, candidate_gate=args.gate,
            candidate_weight=args.weight)
            for seed in range(1000, 1000 + args.seeds)]
        results[scenario.name] = dict(
            baseline=aggregate(base), scored=aggregate(scored))
    payload = dict(scorer=args.scorer, gate=args.gate, weight=args.weight,
                   noise=args.noise, seeds=args.seeds, results=results)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
