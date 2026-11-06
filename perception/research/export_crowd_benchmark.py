#!/usr/bin/env python3
"""Export crowd benchmark scenarios, replays and per-episode statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception.research.crowd_follow_env import CrowdFollowEnv  # noqa: E402
from perception.research.train_crowd_attention import (  # noqa: E402
    AttentionActor, safety_action)
from perception.research.train_predictive_follow import (  # noqa: E402
    Actor, rule_action)


def make_actor(path, kind):
    if kind == 'teacher':
        actor = Actor(15)
        actor.load_state_dict(torch.load(path, map_location='cpu'))
        return lambda obs: float(actor(torch.from_numpy(obs).unsqueeze(0))[0])
    actor = AttentionActor()
    actor.load_state_dict(torch.load(path, map_location='cpu'))
    actor.eval()
    return lambda obs: float(actor(torch.from_numpy(obs).unsqueeze(0))[0])


def run_episode(env, policy, safety=False, use_social=False):
    obs = env.reset(seed=env._episode_seed)
    rows = []
    done = False
    while not done:
        social = (env.social_observation() if use_social
                  else env._base_obs)
        action = policy(social) if policy is not None else rule_action(env)
        if safety:
            action = safety_action(env, action)
        obs, reward, done, info = env.step(action)
        rows.append(dict(
            t=float(env.t),
            robot=[round(env.robot_x, 4), round(env.robot_y, 4)],
            target=[round(float(env.target[0]), 4),
                    round(float(env.target[1]), 4)],
            pedestrians=[[round(float(p['position'][0]), 4),
                          round(float(p['position'][1]), 4)]
                         for p in env.pedestrians],
            action=round(float(action), 4),
            visible=bool(info['visible']),
            distance=round(float(info['distance']), 4),
            pedestrian_collision=bool(info['pedestrian_collision']),
            min_ped_distance=round(float(info['min_ped_distance']), 4),
        ))
    distances = [row['distance'] for row in rows]
    return dict(
        frames=len(rows),
        success=int(env.collisions == 0
                    and env.pedestrian_collisions == 0
                    and float(np.mean(distances)) < 4.0),
        visible_rate=float(np.mean([row['visible'] for row in rows])),
        static_collisions=int(env.collisions),
        pedestrian_collisions=int(env.pedestrian_collisions),
        mean_distance=float(np.mean(distances)),
        min_ped_distance=float(env.min_ped_distance),
    ), rows


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(max(1, len(values)))
    return dict(mean=round(mean, 4), ci95=[round(mean - half, 4),
                                           round(mean + half, 4)], n=int(len(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=25)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--attention', default='')
    parser.add_argument('--teacher', default='')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = CrowdFollowEnv(episode_s=30, seed=args.seed, randomize=True,
                         detection_dropout=0.30, measurement_noise=0.05)
    policies = {'rule': None}
    if args.teacher:
        policies['teacher'] = make_actor(args.teacher, 'teacher')
    if args.attention:
        policies['attention_safety'] = make_actor(args.attention, 'attention')
    episode_records = []
    stats_by_policy = {}
    for name, policy in policies.items():
        results = []
        replay_path = out / f'replay_{name}.jsonl'
        with replay_path.open('w') as replay:
            for episode in range(args.episodes):
                env._episode_seed = args.seed + episode
                result, rows = run_episode(
                    env, policy,
                    safety=(name == 'attention_safety'),
                    use_social=(name == 'attention_safety'))
                results.append(result)
                meta = env.scenario_metadata()
                meta.update(dict(episode=episode, seed=env._episode_seed,
                                 policy=name))
                episode_records.append(meta)
                replay.write(json.dumps(dict(episode=episode,
                                             seed=env._episode_seed,
                                             scenario=meta,
                                             result=result,
                                             rows=rows)) + '\n')
        stats_by_policy[name] = dict(
            episodes=args.episodes,
            success_rate=stats([r['success'] for r in results]),
            visible_rate=stats([r['visible_rate'] for r in results]),
            pedestrian_collisions=int(sum(r['pedestrian_collisions']
                                          for r in results)),
            static_collisions=int(sum(r['static_collisions']
                                      for r in results)),
            mean_distance=stats([r['mean_distance'] for r in results]),
            min_ped_distance=stats([r['min_ped_distance'] for r in results]),
        )
    with (out / 'scenarios.jsonl').open('w') as handle:
        for record in episode_records[::max(1, len(policies))]:
            handle.write(json.dumps(record) + '\n')
    (out / 'statistics.json').write_text(json.dumps(
        dict(protocol=dict(episodes=args.episodes, seed=args.seed,
                           episode_s=30, dynamic_pedestrians=3,
                           detection_dropout=0.30, measurement_noise=0.05),
             policies=stats_by_policy), indent=2))
    print(json.dumps(stats_by_policy, indent=2))


if __name__ == '__main__':
    main()
