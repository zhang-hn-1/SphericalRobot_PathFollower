#!/usr/bin/env python3
"""SAC baseline for the predictive person-following task.

This is an off-policy alternative to PPO.  It uses the same environment,
frozen supervised predictor and evaluation protocol as
``train_predictive_follow.py``, so the comparison is apples-to-apples.
The saved actor has the same ``net``/``log_std`` layout as the PPO Actor and
can be loaded by ``stage_follow.py --policy``.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/train_predictive_follow_sac.py \
        --out /home/jason/perception_data/runs/predictive_follow_sac_v1 \
        --steps 60000 --predictor .../predictor.pt \
        --stage-visibility --detection-dropout 0.30 --measurement-noise 0.05
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
from perception.research.predictive_follow_env import PredictiveFollowEnv  # noqa: E402
from perception.research.train_predictive_follow import (  # noqa: E402
    Actor, Critic, evaluate, evaluate_rule, rule_action)


class SoftActor(nn.Module):
    """Tanh-squashed Gaussian; the underlying Actor state is stage-compatible."""

    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.actor = Actor(obs_dim, hidden=hidden)

    def forward(self, x):
        return torch.tanh(self.actor(x))

    def sample(self, x):
        mean = self.actor(x)
        std = self.actor.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return action.unsqueeze(-1), log_prob.sum(dim=-1, keepdim=True)


class SoftCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=1, hidden=64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, obs, action):
        value = torch.cat([obs, action], dim=-1)
        return self.q1(value), self.q2(value)


class Replay:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        index = self.index
        self.obs[index] = obs
        self.action[index] = np.asarray(action, dtype=np.float32).reshape(1)
        self.reward[index] = reward
        self.next_obs[index] = next_obs
        self.done[index] = done
        self.index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, device):
        indices = np.random.randint(0, self.size, size=batch)
        return tuple(torch.from_numpy(array[indices]).to(device)
                     for array in (self.obs, self.action, self.reward,
                                   self.next_obs, self.done))


def evaluate_actor(actor, episodes, seed0, predictor, env_kwargs):
    return evaluate(
        lambda obs: actor(obs.to(next(actor.parameters()).device)).cpu(),
        episodes=episodes, seed0=seed0,
        predictor=predictor, env_kwargs=env_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=80000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--predictor", default="")
    parser.add_argument("--stage-visibility", action="store_true")
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument("--measurement-noise", type=float, default=0.03)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_kwargs = dict(
        episode_s=args.episode_s,
        stage_visibility=args.stage_visibility,
        detection_dropout=args.detection_dropout,
        measurement_noise=args.measurement_noise,
    )
    env = PredictiveFollowEnv(**env_kwargs)
    predictor = None
    if args.predictor:
        from perception.learned_predictor import TrajectoryPredictor
        predictor = TrajectoryPredictor(args.predictor)
        env.set_predictor(predictor)

    actor = SoftActor(env.obs_dim, hidden=args.hidden).to(device)
    critic = SoftCritic(env.obs_dim, hidden=args.hidden).to(device)
    critic_target = SoftCritic(env.obs_dim, hidden=args.hidden).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor([-1.6], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.AdamW([log_alpha], lr=args.lr)
    target_entropy = -1.0
    replay = Replay(args.capacity, env.obs_dim, 1)
    metrics_path = out / "metrics.jsonl"
    best = None
    best_score = -1e9
    obs = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_count = 0
    started = time.time()

    for step in range(args.steps):
        if step < args.warmup:
            action = float(rule_action(env))
        else:
            with torch.no_grad():
                action = float(actor(torch.from_numpy(obs).unsqueeze(0).to(device))[0])
        next_obs, reward, done, _info = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        episode_return += reward
        obs = next_obs
        if done:
            episode_count += 1
            obs = env.reset(seed=args.seed + step + 1)
            episode_return = 0.0

        if step >= args.warmup and replay.size >= args.batch:
            obs_b, action_b, reward_b, next_obs_b, done_b = replay.sample(
                args.batch, device)
            with torch.no_grad():
                next_action, next_logp = actor.sample(next_obs_b)
                next_q1, next_q2 = critic_target(next_obs_b, next_action)
                next_q = torch.min(next_q1, next_q2) - log_alpha.exp() * next_logp
                target_q = reward_b + args.gamma * (1.0 - done_b) * next_q
            q1, q2 = critic(obs_b, action_b)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            action_new, logp = actor.sample(obs_b)
            q1_new, q2_new = critic(obs_b, action_new)
            actor_loss = (log_alpha.exp() * logp
                          - torch.min(q1_new, q2_new)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            with torch.no_grad():
                log_alpha.clamp_(-4.0, 1.0)

            with torch.no_grad():
                for target_param, param in zip(
                        critic_target.parameters(), critic.parameters()):
                    target_param.mul_(1.0 - args.tau).add_(args.tau * param)

        if step > 0 and step % args.eval_every == 0:
            rule = evaluate_rule(
                episodes=args.eval_episodes, predictor=predictor,
                env_kwargs=env_kwargs)
            learned = evaluate_actor(
                actor, args.eval_episodes, 9000, predictor, env_kwargs)
            entry = dict(step=step, episodes=episode_count,
                         seconds=round(time.time() - started, 1),
                         actor_loss=round(float(actor_loss), 4),
                         critic_loss=round(float(critic_loss), 4),
                         alpha=round(float(log_alpha.exp()), 4),
                         rule=rule, learned=learned)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            if learned["visible_rate"] > best_score:
                best_score = learned["visible_rate"]
                best = actor.actor.state_dict()
                torch.save(best, out / "actor.pt")
                (out / "best.json").write_text(json.dumps(
                    dict(step=step, learned=learned, rule=rule), indent=2))

    if best is None:
        torch.save(actor.actor.state_dict(), out / "actor.pt")
    (out / "config.json").write_text(json.dumps(dict(
        obs_dim=env.obs_dim, action_dim=1, steps=args.steps,
        warmup=args.warmup, batch=args.batch, capacity=args.capacity,
        hidden=args.hidden, lr=args.lr, gamma=args.gamma, tau=args.tau,
        seed=args.seed, episode_s=args.episode_s,
        predictor=args.predictor or None,
        stage_visibility=args.stage_visibility,
        detection_dropout=args.detection_dropout,
        measurement_noise=args.measurement_noise), indent=2))
    print(json.dumps(dict(model=str(out / "actor.pt"),
                          best_score=best_score),
                     indent=2))


if __name__ == "__main__":
    main()
