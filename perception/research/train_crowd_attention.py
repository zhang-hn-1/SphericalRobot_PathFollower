#!/usr/bin/env python3
"""PPO with social attention on the multi-pedestrian 2D benchmark."""

from __future__ import annotations

import argparse
import json
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
from perception.research.crowd_follow_env import CrowdFollowEnv  # noqa: E402
from perception.research.train_predictive_follow import (  # noqa: E402
    Actor, rule_action)


def safety_action(env, proposed):
    """Predict 0.35 s ahead and veto a collision-prone lateral command."""
    def risk(action):
        robot = np.array([env.robot_x + 0.6 * float(action) * 0.35,
                          env.robot_y])
        minimum = float("inf")
        for pedestrian in env.pedestrians:
            future = (pedestrian["position"]
                      + pedestrian["velocity"] * 0.35)
            minimum = min(minimum, float(np.linalg.norm(future - robot)))
        return minimum
    proposed = float(np.clip(proposed, -1.0, 1.0))
    if risk(proposed) >= 0.55:
        return proposed
    candidates = [0.0, -0.35, 0.35, -0.70, 0.70]
    best = max(candidates, key=risk)
    return best if risk(best) >= 0.45 else 0.0


class AttentionActor(nn.Module):
    def __init__(self, base_dim=15, ped_dim=4, hidden=64):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(base_dim, hidden), nn.ELU())
        self.ped = nn.Sequential(nn.Linear(ped_dim, hidden), nn.ELU())
        self.query = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ELU(),
                                  nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.tensor([-0.8]))

    def forward(self, observation):
        base = observation[:, :15]
        peds = observation[:, 15:].reshape(-1, 3, 4)
        base_h = self.base(base)
        ped_h = self.ped(peds)
        query = self.query(base_h).unsqueeze(1)
        score = torch.softmax((ped_h * query).sum(dim=-1, keepdim=True), dim=1)
        context = (score * ped_h).sum(dim=1)
        return self.head(torch.cat([base_h, context], dim=-1)).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim=27, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, observation):
        return self.net(observation).squeeze(-1)


def evaluate(policy, episodes=50, seed0=1000, kind="attention",
             safety=False):
    env = CrowdFollowEnv(episode_s=30, seed=seed0, randomize=True,
                         detection_dropout=0.30, measurement_noise=0.05)
    distances, clears, ped_min = [], [], []
    visible = steps = collisions = ped_collisions = successes = 0
    for episode in range(episodes):
        base = env.reset(seed=seed0 + episode)
        done = False
        episode_distances = []
        while not done:
            with torch.no_grad():
                obs = env.social_observation() if kind != "base" else base
                action = float(policy(torch.from_numpy(obs).unsqueeze(0))[0]
                               if not isinstance(policy, tuple) else 0.0)
            if safety and kind == "attention":
                action = safety_action(env, action)
            base, reward, done, info = env.step(float(np.clip(action, -1, 1)))
            steps += 1
            visible += int(info["visible"])
            episode_distances.append(info["distance"])
        distances.append(float(np.mean(episode_distances)))
        collisions += env.collisions
        ped_collisions += env.pedestrian_collisions
        ped_min.append(env.min_ped_distance)
        successes += int(env.collisions == 0
                         and env.pedestrian_collisions == 0
                         and np.mean(episode_distances) < 4.0)
    return dict(episodes=episodes,
                success_rate=round(successes / max(1, episodes), 4),
                visible_rate=round(visible / max(1, steps), 4),
                static_collisions=collisions,
                pedestrian_collisions=ped_collisions,
                mean_distance=round(float(np.mean(distances)), 3),
                min_ped_distance=round(float(np.min(ped_min)), 3))


def evaluate_rule(episodes=50, seed0=1000):
    env = CrowdFollowEnv(episode_s=30, seed=seed0, randomize=True,
                         detection_dropout=0.30, measurement_noise=0.05)
    distances, ped_min = [], []
    visible = steps = static = ped = successes = 0
    for episode in range(episodes):
        env.reset(seed=seed0 + episode)
        done = False
        episode_distances = []
        while not done:
            _, _, done, info = env.step(rule_action(env))
            steps += 1
            visible += int(info["visible"])
            episode_distances.append(info["distance"])
        distances.append(float(np.mean(episode_distances)))
        static += env.collisions
        ped += env.pedestrian_collisions
        ped_min.append(env.min_ped_distance)
        successes += int(env.collisions == 0 and env.pedestrian_collisions == 0
                         and np.mean(episode_distances) < 4.0)
    return dict(episodes=episodes, success_rate=round(successes / episodes, 4),
                visible_rate=round(visible / max(1, steps), 4),
                static_collisions=static, pedestrian_collisions=ped,
                mean_distance=round(float(np.mean(distances)), 3),
                min_ped_distance=round(float(np.min(ped_min)), 3))


def evaluate_actor(path, episodes=50, seed0=1000):
    actor = Actor(15)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    actor.eval()
    env = CrowdFollowEnv(episode_s=30, seed=seed0, randomize=True,
                         detection_dropout=0.30, measurement_noise=0.05)
    distances, ped_min = [], []
    visible = steps = static = ped = successes = 0
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        episode_distances = []
        while not done:
            with torch.no_grad():
                action = float(actor(torch.from_numpy(obs).unsqueeze(0))[0])
            obs, _, done, info = env.step(float(np.clip(action, -1, 1)))
            steps += 1
            visible += int(info["visible"])
            episode_distances.append(info["distance"])
        distances.append(float(np.mean(episode_distances)))
        static += env.collisions
        ped += env.pedestrian_collisions
        ped_min.append(env.min_ped_distance)
        successes += int(env.collisions == 0 and env.pedestrian_collisions == 0
                         and np.mean(episode_distances) < 4.0)
    return dict(episodes=episodes, success_rate=round(successes / episodes, 4),
                visible_rate=round(visible / max(1, steps), 4),
                static_collisions=static, pedestrian_collisions=ped,
                mean_distance=round(float(np.mean(distances)), 3),
                min_ped_distance=round(float(np.min(ped_min)), 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher", default="")
    parser.add_argument("--ppo", default="")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CrowdFollowEnv(episode_s=args.episode_s, seed=args.seed,
                         randomize=True, detection_dropout=0.30,
                         measurement_noise=0.05)
    actor = AttentionActor().to(device)
    critic = Critic().to(device)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    metrics = out / "metrics.jsonl"
    best = -1.0
    for iteration in range(args.iters):
        obs_b, act_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], []
        for episode in range(args.episodes):
            env.reset(seed=args.seed * 100000 + iteration * 100 + episode)
            obs = env.social_observation()
            done = False
            while not done:
                with torch.no_grad():
                    mean = actor(torch.from_numpy(obs).unsqueeze(0).to(device))
                    std = actor.log_std.exp().expand_as(mean)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum()
                    value = critic(torch.from_numpy(obs).unsqueeze(0).to(device))
                base_obs, reward, done, _ = env.step(float(action.clamp(-1, 1)))
                obs_b.append(obs); act_b.append(float(action))
                logp_b.append(float(logp)); val_b.append(float(value))
                rew_b.append(float(reward)); done_b.append(1.0 if done else 0.0)
                obs = env.social_observation()
        values = np.asarray(val_b + [0.0], dtype=np.float32)
        rewards = np.asarray(rew_b, dtype=np.float32)
        dones = np.asarray(done_b, dtype=np.float32)
        adv = np.zeros_like(rewards)
        last = 0.0
        for index in reversed(range(len(rewards))):
            nxt = 0.0 if dones[index] else values[index + 1]
            delta = rewards[index] + 0.99 * nxt - values[index]
            adv[index] = last = delta + 0.99 * 0.95 * (1.0 - dones[index]) * last
        returns = (adv + np.asarray(val_b, dtype=np.float32)).astype(
            np.float32)
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        obs_t = torch.from_numpy(np.asarray(obs_b, dtype=np.float32)).to(device)
        act_t = torch.from_numpy(np.asarray(act_b, dtype=np.float32)).to(device)
        old_logp = torch.tensor(logp_b, dtype=torch.float32, device=device)
        adv_t = torch.from_numpy(adv).to(device)
        ret_t = torch.from_numpy(returns).to(device)
        for _ in range(4):
            indices = torch.randperm(len(obs_t))
            for start in range(0, len(indices), 256):
                batch = indices[start:start + 256]
                mean = actor(obs_t[batch])
                std = actor.log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_t[batch])
                ratio = (logp - old_logp[batch]).exp()
                policy_loss = -torch.min(
                    ratio * adv_t[batch],
                    ratio.clamp(0.8, 1.2) * adv_t[batch]).mean()
                value_loss = F.mse_loss(critic(obs_t[batch]), ret_t[batch])
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()
        if iteration % 10 == 0 or iteration == args.iters - 1:
            learned = evaluate(lambda x: actor(x.to(device)).cpu(),
                               episodes=20, seed0=9000)
            entry = dict(iter=iteration, learned=learned)
            print(json.dumps(entry), flush=True)
            metrics.open("a").write(json.dumps(entry) + "\n")
            if learned["success_rate"] > best:
                best = learned["success_rate"]
                torch.save(actor.state_dict(), out / "actor.pt")
    if not (out / "actor.pt").exists():
        torch.save(actor.state_dict(), out / "actor.pt")
    results = dict(
        rule=evaluate_rule(),
        attention=evaluate(lambda x: actor(x.to(device)).cpu(),
                           episodes=50, seed0=1000),
        attention_safety=evaluate(lambda x: actor(x.to(device)).cpu(),
                                  episodes=50, seed0=1000, safety=True),
    )
    if args.teacher:
        results["teacher"] = evaluate_actor(args.teacher)
    if args.ppo:
        results["ppo"] = evaluate_actor(args.ppo)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
