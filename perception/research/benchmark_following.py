#!/usr/bin/env python3
"""Unified multi-route benchmark for the predictive follow stack.

Every policy is evaluated on the same random-route episodes with the same
visibility/dropout protocol.  The 2D environment has no identity module, so
identity is reported separately by the ReplicaCAD stage; here we report
coverage, occlusion, success, collisions, path length, control effort and
policy/step latency.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/benchmark_following.py \
        --out .../benchmark_random.json --randomize --episodes 50 \
        --absolute-predictor .../trajectory_predictor_v2/predictor.pt \
        --relative-predictor .../trajectory_predictor_relative_v1/predictor.pt \
        --ppo .../predictive_follow_ppo_stage_v9/actor.pt \
        --teacher .../teacher_student_ppo_v1/actor.pt \
        --sac .../predictive_follow_sac_v2/actor.pt \
        --residual .../predictive_follow_ppo_residual_v1/actor.pt \
        --relative-ppo .../predictive_follow_ppo_stage_relative_v1/actor.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.research.predictive_follow_env import PredictiveFollowEnv  # noqa: E402
from perception.research.train_predictive_follow import (  # noqa: E402
    Actor, apply_action, rule_action)


def load_actor(path):
    actor = Actor(15)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    return actor.eval()


def make_policy(name, actor_path, residual=False, residual_scale=0.4):
    if name == "rule":
        return lambda obs, env: rule_action(env)
    actor = load_actor(actor_path)

    def policy(obs, env):
        with torch.no_grad():
            raw = float(actor(torch.from_numpy(obs).unsqueeze(0))[0])
        if residual:
            return apply_action(raw, env, True, residual_scale)
        return float(np.clip(raw, -1.0, 1.0))

    return policy


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def evaluate(policy, predictor, episodes, episode_s, seed0, randomize,
             dropout, noise):
    env = PredictiveFollowEnv(
        episode_s=episode_s, seed=seed0, randomize=randomize,
        stage_visibility=False, detection_dropout=dropout,
        measurement_noise=noise)
    env.set_predictor(predictor)
    rows = []
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        hidden_run = 0
        longest_hidden = 0
        policy_times, step_times = [], []
        geometric_visible = 0
        while not done:
            started = time.perf_counter()
            action = policy(obs, env)
            action_time = time.perf_counter()
            obs, _reward, done, info = env.step(action)
            finished = time.perf_counter()
            policy_times.append((action_time - started) * 1000.0)
            step_times.append((finished - action_time) * 1000.0)
            geometric_visible += int(env.geometric_visible)
            if info["visible"]:
                hidden_run = 0
            else:
                hidden_run += 1
                longest_hidden = max(longest_hidden, hidden_run)
        steps = env.steps
        visible = env.visible_frames
        rows.append(dict(
            seed=seed0 + episode,
            steps=steps,
            visible_rate=visible / max(1, steps),
            geometric_visible_rate=geometric_visible / max(1, steps),
            collision_frames=env.collisions,
            success=bool(env.collisions == 0 and visible / max(1, steps) >= 0.5),
            reacquired=bool(env.reacquired),
            mean_distance=env.distance_sum / max(1, steps),
            path_length=env.path_length,
            control_effort=env.control_effort,
            min_clearance=env.min_clearance,
            longest_hidden_s=longest_hidden * 0.1,
            policy_ms_mean=float(np.mean(policy_times)),
            policy_ms_p95=percentile(policy_times, 95),
            step_ms_mean=float(np.mean(step_times)),
            step_ms_p95=percentile(step_times, 95),
        ))
    values = {key: np.asarray([row[key] for row in rows], dtype=float)
              for key in ("visible_rate", "geometric_visible_rate",
                          "collision_frames", "success", "reacquired",
                          "mean_distance", "path_length", "control_effort",
                          "min_clearance", "longest_hidden_s",
                          "policy_ms_mean", "policy_ms_p95",
                          "step_ms_mean", "step_ms_p95")}
    return dict(
        episodes=episodes,
        mean_visible_rate=round(float(values["visible_rate"].mean()), 4),
        mean_geometric_visible_rate=round(
            float(values["geometric_visible_rate"].mean()), 4),
        success_rate=round(float(values["success"].mean()), 4),
        reacquired_rate=round(float(values["reacquired"].mean()), 4),
        collision_frames=int(values["collision_frames"].sum()),
        collision_episodes=int((values["collision_frames"] > 0).sum()),
        mean_distance=round(float(values["mean_distance"].mean()), 3),
        mean_path_length=round(float(values["path_length"].mean()), 3),
        mean_control_effort=round(float(values["control_effort"].mean()), 3),
        mean_min_clearance=round(float(values["min_clearance"].mean()), 3),
        mean_longest_hidden_s=round(
            float(values["longest_hidden_s"].mean()), 2),
        policy_ms_mean=round(float(values["policy_ms_mean"].mean()), 4),
        policy_ms_p95=round(float(values["policy_ms_p95"].mean()), 4),
        step_ms_mean=round(float(values["step_ms_mean"].mean()), 4),
        step_ms_p95=round(float(values["step_ms_p95"].mean()), 4),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--absolute-predictor", required=True)
    parser.add_argument("--relative-predictor", default="")
    parser.add_argument("--ppo", default="")
    parser.add_argument("--teacher", default="")
    parser.add_argument("--sac", default="")
    parser.add_argument("--residual", default="")
    parser.add_argument("--relative-ppo", default="")
    args = parser.parse_args()
    from perception.learned_predictor import TrajectoryPredictor
    absolute = TrajectoryPredictor(args.absolute_predictor)
    relative = (TrajectoryPredictor(args.relative_predictor)
                if args.relative_predictor else None)
    configs = [("rule_abs", "rule", absolute, None, False)]
    if relative is not None:
        configs.append(("rule_rel", "rule", relative, None, False))
    if args.ppo:
        configs.append(("ppo_abs", "ppo", absolute, args.ppo, False))
    if args.teacher:
        configs.append(("teacher_abs", "teacher", absolute,
                        args.teacher, False))
    if args.sac:
        configs.append(("sac_abs", "sac", absolute, args.sac, False))
    if args.residual:
        configs.append(("residual_abs", "residual", absolute,
                        args.residual, True))
    if relative is not None and args.relative_ppo:
        configs.append(("ppo_rel", "ppo", relative, args.relative_ppo, False))
    results = {}
    for name, kind, predictor, path, residual in configs:
        policy = make_policy(kind, path, residual=residual)
        results[name] = evaluate(
            policy, predictor, args.episodes, args.episode_s, args.seed,
            args.randomize, args.dropout, args.noise)
        print(json.dumps(dict(config=name, **{
            key: value for key, value in results[name].items()
            if key != "rows"}), indent=2), flush=True)
    payload = dict(protocol=dict(
        episodes=args.episodes, episode_s=args.episode_s,
        seed=args.seed, randomize=args.randomize,
        dropout=args.dropout, noise=args.noise), results=results)
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
