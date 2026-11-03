#!/usr/bin/env python3
"""Teacher-student / DAgger pilot for predictive person following.

Teacher: PPO with a privileged observation (clean target position/velocity,
no detection dropout).
Student: the normal 15D observation containing measured history and the
frozen supervised predictor, trained by imitation of the teacher and then
optionally fine-tuned by PPO.

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/research/train_teacher_student.py \
        --teacher /home/jason/perception_data/runs/teacher_privileged_ppo_v1/actor.pt \
        --predictor .../trajectory_predictor_v2/predictor.pt \
        --out /home/jason/perception_data/runs/teacher_student_v1 \
        --dagger-iters 2 --bc-steps 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
from perception.research.predictive_follow_env import PredictiveFollowEnv  # noqa: E402
from perception.research.train_predictive_follow import (  # noqa: E402
    Actor, evaluate, rule_action)


def load_actor(path, obs_dim, device="cpu"):
    actor = Actor(obs_dim)
    actor.load_state_dict(torch.load(path, map_location="cpu"))
    return actor.to(device).eval()


def collect_teacher_labels(env, teacher, student, episodes, seed0,
                           device="cpu"):
    observations, actions = [], []
    teacher.eval()
    if student is not None:
        student.eval()
    for episode in range(episodes):
        obs = env.reset(seed=seed0 + episode)
        done = False
        while not done:
            privileged = env.privileged_observation()
            with torch.no_grad():
                teacher_action = float(teacher(
                    torch.from_numpy(privileged).unsqueeze(0).to(device))[0])
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(np.clip(teacher_action, -1.0, 1.0))
            if student is None:
                action = rule_action(env)
            else:
                with torch.no_grad():
                    action = float(student(
                        torch.from_numpy(obs).unsqueeze(0).to(device))[0])
            obs, _reward, done, _info = env.step(float(action))
    return (np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32))


def train_student(observations, actions, steps, batch_size, lr, seed, device):
    torch.manual_seed(seed)
    student = Actor(observations.shape[1]).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    obs_t = torch.from_numpy(observations).to(device)
    act_t = torch.from_numpy(actions).to(device)
    for _ in range(steps):
        indices = torch.randperm(len(obs_t))[:batch_size]
        loss = F.mse_loss(student(obs_t[indices]), act_t[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        residual = student(obs_t) - act_t
        rmse = float(torch.sqrt(torch.mean(residual ** 2)))
    return student.eval(), rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictor", default="")
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--bc-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detection-dropout", type=float, default=0.30)
    parser.add_argument("--measurement-noise", type=float, default=0.05)
    parser.add_argument("--no-stage-visibility", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_kwargs = dict(
        episode_s=args.episode_s,
        stage_visibility=not args.no_stage_visibility,
        detection_dropout=args.detection_dropout,
        measurement_noise=args.measurement_noise,
    )
    env = PredictiveFollowEnv(**env_kwargs)
    predictor = None
    if args.predictor:
        from perception.learned_predictor import TrajectoryPredictor
        predictor = TrajectoryPredictor(args.predictor)
        env.set_predictor(predictor)
    teacher = load_actor(args.teacher, env.obs_dim, device)
    student = None
    all_obs, all_actions = [], []
    history = []
    for iteration in range(max(1, args.dagger_iters)):
        obs, actions = collect_teacher_labels(
            env, teacher, student, args.episodes,
            args.seed * 100000 + iteration * 1000, device)
        all_obs.append(obs)
        all_actions.append(actions)
        observations = np.concatenate(all_obs, axis=0)
        labels = np.concatenate(all_actions, axis=0)
        student, rmse = train_student(
            observations, labels, args.bc_steps, args.batch_size, args.lr,
            args.seed + iteration, device)
        eval_result = evaluate(
            lambda obs: student(obs.to(device)).cpu(), episodes=20,
            seed0=9000, predictor=predictor, env_kwargs=env_kwargs)
        entry = dict(iteration=iteration, samples=len(observations),
                     bc_rmse=round(rmse, 5), student=eval_result)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save(student.state_dict(), out / f"student_{iteration}.pt")
        (out / "metrics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in history) + "\n")
    if student is None:
        raise SystemExit("no student trained")
    torch.save(student.state_dict(), out / "student.pt")
    (out / "config.json").write_text(json.dumps(dict(
        teacher=args.teacher, predictor=args.predictor or None,
        dagger_iters=args.dagger_iters, episodes=args.episodes,
        bc_steps=args.bc_steps, seed=args.seed, env=env_kwargs), indent=2))
    (out / "summary.json").write_text(json.dumps(dict(history=history),
                                                  indent=2))
    print(json.dumps(dict(student=str(out / "student.pt"),
                          last=history[-1]), indent=2))


if __name__ == "__main__":
    main()
