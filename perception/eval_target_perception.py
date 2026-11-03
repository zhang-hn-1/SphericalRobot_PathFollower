#!/usr/bin/env python3
"""Synthetic benchmark for the target-person layer.

This isolates the tracker from the detector and the simulator.  It answers the
questions the perception layer is supposed to answer:

* Does the target identity survive a distractor crossing?
* Does a similar-looking distractor steal the identity?
* When the target disappears for one second, is the output still a useful
  position, or is it immediately reported as lost?
* After a longer disappearance, is the same identity re-acquired?

Detections are the true positions plus Gaussian noise.  Ground truth is used
only for scoring, never for association.  This is not a substitute for the
rendered Isaac Gym evaluation; it is the fast regression test.

    python3 perception/eval_target_perception.py \
        --out /tmp/target_perception_bench.json --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402


# ------------------------------------------------------------------ scenes


@dataclass
class Scenario:
    name: str
    duration: float = 8.0
    target_start: tuple = (5.0, -1.5)
    target_velocity: tuple = (0.0, 0.45)
    distractor_start: tuple = (4.5, 2.0)
    distractor_velocity: tuple = (0.0, 0.0)
    target_dropout: tuple = ()  # (start, end) seconds
    similar_appearance: bool = False
    turn_at: float | None = None
    turned_velocity: tuple = (0.45, 0.0)
    max_memory_s: float = 5.0
    note: str = ""


def scenarios():
    return [
        Scenario("clean", target_start=(5.0, -1.5),
                 target_velocity=(0.0, 0.45),
                 distractor_start=(6.0, 2.0),
                 note="no crossing, no dropout"),
        Scenario("crossing", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.5),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.5),
                 note="distinct appearance, head-on crossing"),
        Scenario("similar_crossing", target_start=(5.0, -2.0),
                 target_velocity=(0.0, 0.5),
                 distractor_start=(5.0, 2.0),
                 distractor_velocity=(0.0, -0.5),
                 similar_appearance=True,
                 note="near-identical appearance, head-on crossing"),
        Scenario("dropout_1s", target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(3.0, 4.0),
                 note="target invisible for 1.0 s, constant velocity"),
        Scenario("dropout_turn", target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(3.0, 4.0),
                 turn_at=3.0, turned_velocity=(0.4, 0.0),
                 note="target turns 90 deg while invisible"),
        Scenario("dropout_2s", target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(3.0, 5.0),
                 note="target invisible for 2.0 s: prediction then memory"),
        Scenario("long_loss_reid", duration=10.0, target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(3.0, 8.0),
                 note="target invisible for 5 s; memory expires, re-ID by appearance"),
        Scenario("reid_after_memory", duration=8.0, target_start=(5.0, -1.0),
                 target_velocity=(0.0, 0.4),
                 distractor_start=(4.5, 2.5),
                 target_dropout=(2.0, 5.0), max_memory_s=2.0,
                 note="track is pruned, then re-identified by appearance"),
    ]


def _target_position(scenario: Scenario, t: float) -> np.ndarray:
    start = np.asarray(scenario.target_start, dtype=np.float64)
    velocity = np.asarray(scenario.target_velocity, dtype=np.float64)
    if scenario.turn_at is None or t < scenario.turn_at:
        return start + velocity * t
    turn_position = start + velocity * scenario.turn_at
    return turn_position + np.asarray(
        scenario.turned_velocity, dtype=np.float64) * (t - scenario.turn_at)


def _distractor_position(scenario: Scenario, t: float) -> np.ndarray:
    return (np.asarray(scenario.distractor_start, dtype=np.float64)
            + np.asarray(scenario.distractor_velocity, dtype=np.float64) * t)


def _bbox_for(position: np.ndarray) -> np.ndarray:
    x, y = float(position[0]), float(position[1])
    return np.array([100.0 + 20.0 * x, 200.0 + 20.0 * y,
                     116.0 + 20.0 * x, 340.0 + 20.0 * y])


def _appearance(base, rng):
    value = np.asarray(base, dtype=np.float64) + rng.normal(0.0, 0.02, 3)
    norm = np.linalg.norm(value)
    return value / norm if norm > 0 else value


# ------------------------------------------------------------------ rollout


@dataclass
class Rollout:
    scenario: str
    seed: int
    frames: int = 0
    measured: int = 0
    predicted: int = 0
    lost: int = 0
    wrong: int = 0
    identity_correct: int = 0
    visible_errors: list = field(default_factory=list)
    predicted_errors: list = field(default_factory=list)
    dropout_errors: dict = field(default_factory=dict)
    dropout_status: dict = field(default_factory=dict)
    reacquisition_s: float | None = None
    same_track: bool | None = None
    reidentified: bool = False
    track_ids: list = field(default_factory=list)


def run_scenario(scenario: Scenario, seed: int,
                 position_noise: float = 0.06,
                 detection_confidence: float = 0.9,
                 candidate_scorer=None,
                 candidate_gate: float = 0.0,
                 candidate_weight: float = 0.0) -> Rollout:
    rng = np.random.default_rng(seed)
    target_appearance = np.array([1.0, 0.0, 0.0])
    if scenario.similar_appearance:
        distractor_appearance = np.array([0.98, 0.20, 0.0])
    else:
        distractor_appearance = np.array([0.0, 1.0, 0.0])
    distractor_appearance = distractor_appearance / np.linalg.norm(
        distractor_appearance)

    layer = TargetPerceptionLayer(
        init_bbox=_bbox_for(_target_position(scenario, 0.0)),
        init_appearance=target_appearance,
        max_prediction_s=1.0,
        max_memory_s=scenario.max_memory_s,
        accel_std=0.6,
        candidate_scorer=candidate_scorer,
        candidate_gate=candidate_gate,
        candidate_weight=candidate_weight,
    )
    result = Rollout(scenario=scenario.name, seed=seed)
    visible_at_start = True
    if scenario.target_dropout:
        visible_at_start = not (scenario.target_dropout[0] <= 0.0
                                < scenario.target_dropout[1])
    before_dropout_id = None
    reacquisition_started = None
    dropout_start = scenario.target_dropout[0] if scenario.target_dropout else None
    dropout_end = scenario.target_dropout[1] if scenario.target_dropout else None

    steps = int(round(scenario.duration / 0.1)) + 1
    for step in range(steps):
        t = round(step * 0.1, 6)
        target_xy = _target_position(scenario, t)
        distractor_xy = _distractor_position(scenario, t)
        target_visible = (not scenario.target_dropout
                          or not (dropout_start <= t < dropout_end))

        detections = []
        if target_visible:
            noisy = target_xy + rng.normal(0.0, position_noise, 2)
            detections.append(PersonDetection(
                position_xy=noisy,
                bbox=_bbox_for(target_xy),
                appearance=_appearance(target_appearance, rng),
                score=float(np.clip(detection_confidence
                                    + rng.normal(0.0, 0.01), 0.05, 0.999)),
            ))
        noisy = distractor_xy + rng.normal(0.0, position_noise, 2)
        detections.append(PersonDetection(
            position_xy=noisy,
            bbox=_bbox_for(distractor_xy),
            appearance=_appearance(distractor_appearance, rng),
            score=float(np.clip(detection_confidence
                                + rng.normal(0.0, 0.01), 0.05, 0.999)),
        ))

        belief = layer.update(detections, t)
        result.frames += 1
        if belief.track_id is not None:
            result.track_ids.append(belief.track_id)
        if belief.status == "measured":
            result.measured += 1
        elif belief.status == "predicted":
            result.predicted += 1
        else:
            result.lost += 1

        if belief.position_xy is None:
            continue
        target_error = float(np.linalg.norm(belief.position_xy - target_xy))
        distractor_error = float(np.linalg.norm(belief.position_xy - distractor_xy))
        if distractor_error < target_error:
            result.wrong += 1
        else:
            result.identity_correct += 1
        if belief.status == "measured" and target_visible:
            result.visible_errors.append(target_error)
        if belief.status == "predicted":
            result.predicted_errors.append(target_error)

        # Prediction error at fixed ages inside a dropout.  At 10 Hz the
        # 0.5 s / 1.0 s samples land on frames.
        if scenario.target_dropout:
            age = t - dropout_start
            for checkpoint in (0.5, 1.0):
                if abs(age - checkpoint) < 1e-6:
                    result.dropout_errors[f"{checkpoint:.1f}"] = target_error
                    result.dropout_status[f"{checkpoint:.1f}"] = belief.status
            if t < dropout_start and belief.track_id is not None:
                before_dropout_id = belief.track_id
            if t >= dropout_end and belief.status == "measured":
                if reacquisition_started is None:
                    reacquisition_started = t
                    result.reacquisition_s = t - dropout_end
                    result.same_track = (before_dropout_id == belief.track_id)
                    result.reidentified = bool(belief.reidentified)

    result.track_ids = sorted(set(result.track_ids))
    return result


# ------------------------------------------------------------------ report


def aggregate(rollouts):
    total = sum(r.frames for r in rollouts)
    measured = sum(r.measured for r in rollouts)
    predicted = sum(r.predicted for r in rollouts)
    lost = sum(r.lost for r in rollouts)
    correct = sum(r.identity_correct for r in rollouts)
    wrong = sum(r.wrong for r in rollouts)
    visible_errors = [value for r in rollouts for value in r.visible_errors]
    predicted_errors = [value for r in rollouts for value in r.predicted_errors]
    dropout_05 = [r.dropout_errors.get("0.5") for r in rollouts
                  if r.dropout_errors.get("0.5") is not None]
    dropout_10 = [r.dropout_errors.get("1.0") for r in rollouts
                  if r.dropout_errors.get("1.0") is not None]
    reacq = [r.reacquisition_s for r in rollouts
             if r.reacquisition_s is not None]
    same = [r.same_track for r in rollouts if r.same_track is not None]
    reidentified = sum(1 for r in rollouts if r.reidentified)
    return dict(
        runs=len(rollouts),
        frames=total,
        measured_rate=measured / max(1, total),
        predicted_rate=predicted / max(1, total),
        lost_rate=lost / max(1, total),
        identity_accuracy=correct / max(1, correct + wrong),
        wrong_person_rate=wrong / max(1, total),
        visible_position_rmse=float(np.sqrt(np.mean(
            np.square(visible_errors)))) if visible_errors else None,
        predicted_position_rmse=float(np.sqrt(np.mean(
            np.square(predicted_errors)))) if predicted_errors else None,
        dropout_error_0_5s=float(np.mean(dropout_05)) if dropout_05 else None,
        dropout_error_1_0s=float(np.mean(dropout_10)) if dropout_10 else None,
        reacquisition_s=float(np.mean(reacq)) if reacq else None,
        same_track_rate=float(np.mean(same)) if same else None,
        reidentified_rate=(reidentified / len(rollouts)) if rollouts else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--position-noise", type=float, default=0.06)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    print(f"{'scenario':22s} {'id_acc':>7s} {'wrong':>7s} {'lost':>7s} "
          f"{'pred@1s':>8s} {'reacq':>7s}")
    for scenario in scenarios():
        rollouts = [run_scenario(scenario, seed, args.position_noise)
                    for seed in range(args.seeds)]
        summary = aggregate(rollouts)
        all_results[scenario.name] = dict(
            note=scenario.note,
            summary=summary,
            runs=[r.__dict__ for r in rollouts],
        )
        def fmt(value, digits=2):
            return "--" if value is None else f"{value:.{digits}f}"
        print(f"{scenario.name:22s} {fmt(summary['identity_accuracy']):>7s} "
              f"{fmt(summary['wrong_person_rate']):>7s} "
              f"{fmt(summary['lost_rate']):>7s} "
              f"{fmt(summary['dropout_error_1_0s']):>8s} "
              f"{fmt(summary['reacquisition_s']):>7s}")

    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
