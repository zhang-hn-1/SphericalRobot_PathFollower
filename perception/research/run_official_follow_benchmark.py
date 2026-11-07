"""Unified 2D target-following benchmark with official CrowdNav policies."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO / "perception/research/official_crowdnav"
sys.path.insert(0, str(OFFICIAL))

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_nav.policy.policy_factory import policy_factory  # noqa: E402


class RulePolicy:
    name = "rule"
    kinematics = "holonomic"
    trainable = False
    multiagent_training = False

    def set_phase(self, phase):
        return

    def set_device(self, device):
        return

    def set_env(self, env):
        return

    def set_epsilon(self, epsilon):
        return

    def predict(self, state):
        self_state = state.self_state
        delta = np.array([self_state.gx - self_state.px,
                          self_state.gy - self_state.py])
        speed = min(0.6, float(np.linalg.norm(delta)))
        if speed < 0.05:
            return ActionXY(0.0, 0.0)
        return ActionXY(*(delta / np.linalg.norm(delta) * speed))


def point_at_distance(path, distance):
    distance = float(distance) % float(sum(
        np.linalg.norm(path[(i + 1) % len(path)] - path[i])
        for i in range(len(path))))
    for index in range(len(path)):
        start = path[index]
        end = path[(index + 1) % len(path)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if distance <= length:
            unit = segment / max(length, 1e-9)
            return start + unit * distance, unit
        distance -= length
    return path[0], np.array([1.0, 0.0])


def compute_route_progress(path, point):
    best_error = float("inf")
    best = 0.0
    travelled = 0.0
    for index in range(len(path)):
        start = path[index]
        end = path[(index + 1) % len(path)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        unit = segment / length
        projection = float(np.clip(np.dot(point - start, unit), 0.0, length))
        error = float(np.linalg.norm(point - (start + unit * projection)))
        if error < best_error:
            best_error = error
            best = travelled + projection
        travelled += length
    return best


def safe_action(robot, humans, action, horizon=0.60, margin=0.25):
    scales = [1.0, 0.7, 0.4, 0.0]
    for scale in scales:
        candidate = ActionXY(action.vx * scale, action.vy * scale)
        future_robot = np.array([robot.px + candidate.vx * horizon,
                                 robot.py + candidate.vy * horizon])
        risk = False
        for human in humans:
            future_human = np.array([human.px + human.vx * horizon,
                                     human.py + human.vy * horizon])
            distance = float(np.linalg.norm(future_robot - future_human))
            if distance < robot.radius + human.radius + margin:
                risk = True
                break
        if not risk:
            return candidate
    return ActionXY(0.0, 0.0)


def make_env():
    import configparser
    config_path = OFFICIAL / "checkpoints/env20.config"
    policy_config_path = OFFICIAL / "checkpoints/policy.config"
    env_config = configparser.RawConfigParser()
    env_config.read(config_path)
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_path)
    env = CrowdSim()
    env.configure(env_config)
    return env, env_config, policy_config


def make_policy(name, env, env_config, policy_config):
    if name == "rule":
        return RulePolicy()
    policy = policy_factory[name]()
    policy.configure(policy_config)
    if name == "sarl":
        model_path = OFFICIAL / "checkpoints/crowdnav_il_model.pth"
        import torch
        policy.get_model().load_state_dict(torch.load(model_path,
                                                      map_location="cpu"))
    policy.set_phase("test")
    policy.set_device("cpu")
    policy.set_env(env)
    return policy


def evaluate(policy_name, episodes, seed, use_safety):
    env, env_config, policy_config = make_env()
    robot = Robot(env_config, "robot")
    policy = make_policy(policy_name, env, env_config, policy_config)
    robot.set_policy(policy)
    env.set_robot(robot)
    route = np.asarray([
        [-1.00, -1.00], [1.00, -1.00], [1.00, 1.00], [-1.00, 1.00]
    ], dtype=np.float64)
    route_length = float(sum(np.linalg.norm(
        route[(i + 1) % len(route)] - route[i]) for i in range(len(route))))
    speed = 0.45
    results = []
    rows = []
    for episode in range(episodes):
        env.case_counter["test"] = episode
        env.reset("test")
        robot.set(0.0, -4.0, -1.25, -1.25, 0.0, 0.0, math.pi / 2)
        done = False
        episode_rows = []
        distances = []
        collisions = 0
        while not done:
            t = env.global_time
            target_point, tangent = point_at_distance(route, speed * t)
            lookahead_point, _ = point_at_distance(route,
                                                   speed * (t + 6.0))
            robot.gx, robot.gy = lookahead_point[0], lookahead_point[1]
            ob = [human.get_observable_state() for human in env.humans]
            action = robot.act(ob)
            if use_safety:
                action = safe_action(robot, env.humans, action)
            ob, reward, done, info = env.step(action)
            distances.append(float(np.linalg.norm(
                np.array([robot.px, robot.py]) - target_point)))
            collisions += int(type(info).__name__ == "Collision")
            episode_rows.append(dict(
                t=round(t, 3),
                robot=[round(robot.px, 4), round(robot.py, 4)],
                target=[round(float(target_point[0]), 4),
                        round(float(target_point[1]), 4)],
                action=[round(float(action.vx), 4),
                        round(float(action.vy), 4)],
                humans=[[round(human.px, 4), round(human.py, 4)]
                        for human in env.humans],
                distance=round(distances[-1], 4),
                status=type(info).__name__,
            ))
        route_progress = compute_route_progress(route, np.array([robot.gx,
                                                          robot.gy])) / route_length
        result = dict(
            episode=episode,
            frames=len(episode_rows),
            success=int(collisions == 0 and float(np.mean(distances)) < 3.0
                        and route_progress > 0.9),
            collisions=int(collisions),
            mean_distance=round(float(np.mean(distances)), 4),
            min_distance=round(float(np.min(distances)), 4),
            route_progress=round(route_progress, 4),
        )
        results.append(result)
        rows.append(dict(episode=episode, result=result, rows=episode_rows))
    successes = np.asarray([item["success"] for item in results],
                           dtype=np.float64)
    collisions = int(sum(item["collisions"] for item in results))
    mean_distance = float(np.mean([item["mean_distance"]
                                   for item in results]))
    half = 1.96 * (successes.std(ddof=1) if len(successes) > 1 else 0.0
                   ) / math.sqrt(len(successes))
    summary = dict(
        policy=policy_name,
        use_safety=use_safety,
        episodes=episodes,
        success_rate=round(float(successes.mean()), 4),
        success_ci95=[round(float(successes.mean() - half), 4),
                      round(float(successes.mean() + half), 4)],
        collisions=collisions,
        mean_distance=round(mean_distance, 4),
    )
    return summary, results, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for policy_name in ("rule", "orca", "sarl"):
        for use_safety in (False, True):
            summary, results, rows = evaluate(
                policy_name, args.episodes, args.seed, use_safety)
            summaries.append(summary)
            suffix = "safety" if use_safety else "base"
            (out / f"{policy_name}_{suffix}_results.json").write_text(
                json.dumps(results, indent=2))
            with (out / f"{policy_name}_{suffix}_replay.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(json.dumps(summary), flush=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
