#!/usr/bin/env python3
"""PPO for the predictive lost-target follow policy.

The environment is the fast 2D abstraction in ``predictive_follow_env.py``.
The policy sees target geometry, age, velocity confidence, prediction
uncertainty and clearance, and outputs one continuous lateral-speed command.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/train_predictive_follow.py \
        --out /home/jason/perception_data/runs/predictive_follow_ppo_v1 \
        --iters 120 --episodes 8
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictive_follow_env import (PredictiveFollowEnv,  # noqa: E402
                                   ROBOT_LANE_Y)


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.tensor([-0.8]))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def distribution(self, x):
        mean = self.forward(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        adv[t] = last = delta + gamma * lam * (1.0 - dones[t]) * last
    return adv


def rule_action(env):
    age = max(0.0, env.t - env.last_measurement_t)
    if env.visible:
        target_x = env.target[0]
    else:
        target_x = env.last_position[0] + env.last_velocity[0] * age
    vx = float(np.clip(0.8 * (target_x - env.robot_x), -0.55, 0.55))
    return vx / 0.6


def collect_bc_data(env, episodes=20, seed0=7000):
    observations = []
    actions = []
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        while not done:
            action = rule_action(env)
            observations.append(obs)
            actions.append(action)
            obs, _reward, done, _info = env.step(action)
    return (np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32))


def train_bc(actor, observations, actions, steps=1000, lr=1e-3,
             batch_size=256):
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
    obs_t = torch.from_numpy(observations)
    act_t = torch.from_numpy(actions)
    for _step in range(steps):
        indices = torch.randperm(len(obs_t))[:batch_size]
        loss = F.mse_loss(actor(obs_t[indices]), act_t[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        residual = actor(obs_t) - act_t
        rmse = float(torch.sqrt(torch.mean(residual ** 2)))
    return rmse


def evaluate(policy, episodes=20, seed0=9000):
    env = PredictiveFollowEnv()
    visible = hidden = collisions = reacquired = 0
    distances = []
    returns = []
    clears = []
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                action = float(policy(torch.from_numpy(obs).unsqueeze(0)))
            action = float(np.clip(action, -1.0, 1.0))
            obs, reward, done, info = env.step(action)
            total += reward
            visible += int(info["visible"])
            hidden += int(not info["visible"])
            collisions += int(info["collision"])
            distances.append(info["distance"])
            clears.append(info["clearance"])
        reacquired += int(env.reacquired)
        returns.append(total)
    return dict(
        visible_rate=visible / max(1, visible + hidden),
        hidden_frames=hidden,
        collisions=collisions,
        reacquired_episodes=reacquired,
        episodes=episodes,
        mean_distance=round(float(np.mean(distances)), 3),
        min_clearance=round(float(np.min(clears)), 3),
        mean_return=round(float(np.mean(returns)), 3),
    )


def evaluate_rule(episodes=20, seed0=9000):
    env = PredictiveFollowEnv()
    visible = hidden = collisions = reacquired = 0
    distances = []
    returns = []
    clears = []
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        total = 0.0
        while not done:
            obs, reward, done, info = env.step(rule_action(env))
            total += reward
            visible += int(info["visible"])
            hidden += int(not info["visible"])
            collisions += int(info["collision"])
            distances.append(info["distance"])
            clears.append(info["clearance"])
        reacquired += int(env.reacquired)
        returns.append(total)
    return dict(
        visible_rate=visible / max(1, visible + hidden),
        hidden_frames=hidden,
        collisions=collisions,
        reacquired_episodes=reacquired,
        episodes=episodes,
        mean_distance=round(float(np.mean(distances)), 3),
        min_clearance=round(float(np.min(clears)), 3),
        mean_return=round(float(np.mean(returns)), 3),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-episodes", type=int, default=0,
                        help="rule demonstrations used to warm-start the actor")
    parser.add_argument("--bc-steps", type=int, default=0,
                        help="supervised updates before PPO")
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = PredictiveFollowEnv(episode_s=args.episode_s)
    actor = Actor(env.obs_dim)
    critic = Critic(env.obs_dim)
    if args.bc_episodes > 0 and args.bc_steps > 0:
        bc_obs, bc_act = collect_bc_data(
            env, episodes=args.bc_episodes, seed0=args.seed * 100000 + 7000)
        bc_rmse = train_bc(actor, bc_obs, bc_act, steps=args.bc_steps,
                           lr=args.bc_lr)
        (out / "bc.json").write_text(json.dumps(dict(
            episodes=args.bc_episodes, steps=args.bc_steps,
            rmse=round(bc_rmse, 5),
            observations=len(bc_obs)), indent=2))
        print(json.dumps(dict(bc=dict(episodes=args.bc_episodes,
                                      steps=args.bc_steps,
                                      rmse=round(bc_rmse, 5)))), flush=True)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    metrics_path = out / "metrics.jsonl"
    best = None
    best_score = -1e9

    for iteration in range(args.iters):
        started = time.time()
        obs_b, act_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], []
        ep_returns = []
        for episode in range(args.episodes):
            obs = env.reset(seed=args.seed * 100000 + iteration * 100 + episode)
            done = False
            episode_return = 0.0
            while not done:
                obs_b.append(obs)
                with torch.no_grad():
                    dist = actor.distribution(torch.from_numpy(obs).unsqueeze(0))
                    action = dist.sample()
                    logp_b.append(float(dist.log_prob(action)))
                    val_b.append(float(critic(
                        torch.from_numpy(obs).unsqueeze(0))))
                obs, reward, done, _info = env.step(float(action))
                act_b.append(float(action))
                rew_b.append(float(reward))
                done_b.append(1.0 if done else 0.0)
                episode_return += reward
            ep_returns.append(episode_return)

        values = np.array(val_b + [0.0], dtype=np.float32)
        rewards = np.array(rew_b, dtype=np.float32)
        dones = np.array(done_b, dtype=np.float32)
        adv = gae(rewards, values, dones)
        returns = adv + np.array(val_b, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.from_numpy(np.array(obs_b, dtype=np.float32))
        act_t = torch.from_numpy(np.array(act_b, dtype=np.float32))
        logp_old = torch.tensor(logp_b, dtype=torch.float32)
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(returns)

        for _epoch in range(4):
            indices = torch.randperm(len(obs_t))
            for start in range(0, len(indices), 256):
                batch = indices[start:start + 256]
                dist = actor.distribution(obs_t[batch])
                logp = dist.log_prob(act_t[batch])
                ratio = (logp - logp_old[batch]).exp()
                policy_loss = -torch.min(
                    ratio * adv_t[batch],
                    ratio.clamp(0.8, 1.2) * adv_t[batch]).mean()
                value_loss = F.mse_loss(
                    critic(obs_t[batch]), ret_t[batch])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()

        entry = dict(
            iter=iteration,
            mean_return=round(float(np.mean(ep_returns)), 3),
            seconds=round(time.time() - started, 1),
        )
        if iteration % 10 == 0 or iteration == args.iters - 1:
            rule = evaluate_rule()
            learned = evaluate(actor)
            entry["rule"] = rule
            entry["learned"] = learned
            score = learned["visible_rate"] - 0.01 * learned["collisions"]
            if score > best_score:
                best_score = score
                best = actor.state_dict()
                torch.save(best, out / "actor.pt")
                (out / "best.json").write_text(json.dumps(
                    dict(iter=iteration, score=score, learned=learned,
                         rule=rule), indent=2))
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    if best is None:
        torch.save(actor.state_dict(), out / "actor.pt")
    (out / "config.json").write_text(json.dumps(dict(
        obs_dim=env.obs_dim, action_dim=env.action_dim,
        episode_s=args.episode_s, iters=args.iters,
        episodes=args.episodes, seed=args.seed,
        bc_episodes=args.bc_episodes, bc_steps=args.bc_steps), indent=2))
    print(json.dumps(dict(model=str(out / "actor.pt"),
                          best=json.loads((out / "best.json").read_text())
                          if (out / "best.json").exists() else None),
                     indent=2))


if __name__ == "__main__":
    main()
