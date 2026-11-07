"""PPO with social attention on the 2D crowd-following benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from perception.research.crowd_follow_2d_env import CrowdFollow2DEnv  # noqa: E402


class AttentionActor2D(nn.Module):
    def __init__(self, base_dim=15, ped_dim=4, hidden=64, action_dim=2):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(base_dim, hidden), nn.ELU())
        self.ped = nn.Sequential(nn.Linear(ped_dim, hidden), nn.ELU())
        self.query = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ELU(),
                                  nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.tensor([-0.8, -0.8]))

    def forward(self, observation):
        base = observation[:, :15]
        peds = observation[:, 15:].reshape(-1, 3, 4)
        base_h = self.base(base)
        ped_h = self.ped(peds)
        query = self.query(base_h).unsqueeze(1)
        score = torch.softmax((ped_h * query).sum(dim=-1, keepdim=True), dim=1)
        context = (score * ped_h).sum(dim=1)
        return self.head(torch.cat([base_h, context], dim=-1))


class Critic(nn.Module):
    def __init__(self, obs_dim=27, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, observation):
        return self.net(observation).squeeze(-1)


def evaluate(actor, episodes=50, seed0=1000, device=torch.device("cpu")):
    env = CrowdFollow2DEnv(episode_s=30, seed=seed0)
    distances, ped_min = [], []
    successes = static = ped = 0
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        episode_distances = []
        while not done:
            with torch.no_grad():
                mean = actor(torch.from_numpy(obs).unsqueeze(0).to(device))[0]
                std = actor.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().clamp(-1, 1).cpu().numpy()
            obs, _, done, info = env.step(action)
            episode_distances.append(info["distance"])
        distances.append(float(np.mean(episode_distances)))
        static += env.static_collisions
        ped += env.pedestrian_collisions
        ped_min.append(env.min_ped_distance)
        successes += int(env.static_collisions == 0
                         and env.pedestrian_collisions == 0
                         and np.mean(episode_distances) < 3.0)
    return dict(episodes=episodes,
                success_rate=round(successes / episodes, 4),
                static_collisions=static,
                pedestrian_collisions=ped,
                mean_distance=round(float(np.mean(distances)), 4),
                min_ped_distance=round(float(np.min(ped_min)), 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CrowdFollow2DEnv(episode_s=args.episode_s, seed=args.seed)
    actor = AttentionActor2D().to(device)
    critic = Critic().to(device)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    best = -1.0
    metrics = out / "metrics.jsonl"
    for iteration in range(args.iters):
        obs_b, act_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], []
        for episode in range(args.episodes):
            obs = env.reset(seed=args.seed * 100000 + iteration * 100 + episode)
            done = False
            while not done:
                with torch.no_grad():
                    mean = actor(torch.from_numpy(obs).unsqueeze(0).to(device))
                    std = actor.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum()
                    value = critic(torch.from_numpy(obs).unsqueeze(0).to(device))
                obs, reward, done, _ = env.step(action.detach().cpu().numpy())
                obs_b.append(obs); act_b.append(action.detach().cpu().numpy())
                logp_b.append(float(logp)); val_b.append(float(value))
                rew_b.append(float(reward)); done_b.append(1.0 if done else 0.0)
        values = np.asarray(val_b + [0.0], dtype=np.float32)
        rewards = np.asarray(rew_b, dtype=np.float32)
        dones = np.asarray(done_b, dtype=np.float32)
        adv = np.zeros_like(rewards)
        last = 0.0
        for index in reversed(range(len(rewards))):
            nxt = 0.0 if dones[index] else values[index + 1]
            delta = rewards[index] + 0.99 * nxt - values[index]
            adv[index] = last = delta + 0.99 * 0.95 * (1.0 - dones[index]) * last
        returns = (adv + np.asarray(val_b, dtype=np.float32)).astype(np.float32)
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
                std = actor.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_t[batch]).sum(dim=-1)
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
            learned = evaluate(actor, episodes=20, seed0=9000, device=device)
            entry = dict(iter=iteration, learned=learned)
            print(json.dumps(entry), flush=True)
            metrics.open("a").write(json.dumps(entry) + "\n")
            if learned["success_rate"] > best:
                best = learned["success_rate"]
                torch.save(actor.state_dict(), out / "actor.pt")
    if not (out / "actor.pt").exists():
        torch.save(actor.state_dict(), out / "actor.pt")
    orca_distances, orca_ped, orca_static, orca_success = [], 0, 0, 0
    env = CrowdFollow2DEnv(episode_s=30, seed=1000)
    for episode in range(50):
        env.reset(seed=1000 + episode); done = False
        episode_distances = []
        while not done:
            obs, _, done, info = env.step(env.orca_action())
            episode_distances.append(info["distance"])
        orca_distances.append(float(np.mean(episode_distances)))
        orca_ped += env.pedestrian_collisions
        orca_static += env.static_collisions
        orca_success += int(env.static_collisions == 0
                            and env.pedestrian_collisions == 0
                            and np.mean(episode_distances) < 3.0)
    results = dict(
        attention=evaluate(actor, episodes=50, seed0=1000, device=device),
        orca=dict(episodes=50,
                  success_rate=round(orca_success / 50, 4),
                  static_collisions=orca_static,
                  pedestrian_collisions=orca_ped,
                  mean_distance=round(float(np.mean(orca_distances)), 4),
                  min_ped_distance=None),
    )
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
