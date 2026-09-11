# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import csv
import json
import subprocess
import sys
import time

import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


CHECKPOINT_START = 300
CHECKPOINT_END = 15000
CHECKPOINT_INTERVAL = 50
CHECKPOINT_POLL_SECONDS = 5
SCREENING_EPISODES = 10
SCREENING_SUCCESS_CUTOFF = 6
PLOT_EVALUATION = False


def _checkpoint_file(run_dir, checkpoint):
    return os.path.join(os.path.abspath(run_dir), f"model_{int(checkpoint)}.pt")


def _wait_for_checkpoint(run_dir, checkpoint):
    """Wait until a checkpoint has appeared and its file size is stable."""
    model_path = _checkpoint_file(run_dir, checkpoint)
    last_size = -1
    stable_polls = 0
    while True:
        if os.path.isfile(model_path):
            try:
                current_size = os.path.getsize(model_path)
            except OSError:
                current_size = -1
            if current_size > 0 and current_size == last_size:
                stable_polls += 1
            else:
                stable_polls = 0
            last_size = current_size
            if stable_polls >= 1:
                return model_path

        print(
            f"Waiting for checkpoint {checkpoint}: {model_path} "
            f"(poll every {CHECKPOINT_POLL_SECONDS}s)"
        )
        time.sleep(CHECKPOINT_POLL_SECONDS)


def _run_checkpoint_in_child_process(checkpoint):
    """Run one checkpoint in a fresh process to avoid Isaac Gym GPU leaks."""
    child_env = os.environ.copy()
    child_env["ROTUNBOT_SINGLE_CHECKPOINT"] = str(int(checkpoint))
    command = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    return subprocess.run(command, env=child_env).returncode


def _write_checkpoint_summary(run_dir, summary):
    """Write one JSON file and update the run-level CSV summary."""
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint = int(summary["checkpoint"])
    json_path = os.path.join(run_dir, f"evaluation_checkpoint_{checkpoint:05d}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    csv_path = os.path.join(run_dir, "evaluation_summary.csv")
    fieldnames = list(summary.keys())
    existing_rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))
    existing_rows = [
        row for row in existing_rows
        if int(row.get("checkpoint", -1)) != checkpoint
    ]
    existing_rows = [
        {key: row.get(key, "") for key in fieldnames}
        for row in existing_rows
    ]
    existing_rows.append({key: summary[key] for key in fieldnames})
    existing_rows.sort(key=lambda row: int(row["checkpoint"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    return json_path, csv_path


def evaluate_checkpoint(args, checkpoint):
    """Evaluate one checkpoint and save its complete metrics and trajectories."""
    # task_registry uses args.checkpoint when it loads the runner.
    args.checkpoint = int(checkpoint)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Evaluation is intentionally sequential: one environment produces one
    # complete episode at a time, so the requested episode count is exact and
    # every episode starts from the same world origin.
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    # Always evaluate the complete target distribution, never the training
    # target curriculum.
    env_cfg.commands.target_curriculum = False
    # Formal paper evaluation: both distance and speed must satisfy the
    # success condition.  Use 0.30 m only for a separate exploratory run.
    env_cfg.evaluation.target_error_threshold = 0.40

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    def close_environment():
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    # DWLOnPolicyRunner resets the environment while it is constructed.  Read
    # observations after that reset so the history and simulator state match.
    obs = env.get_observations()

    logger = Logger(env.dt)
    # Use the 40-trial evaluation protocol explicitly in play.py as well as
    # in the config, so an older remote config cannot silently run 200 trials.
    target_eval_episodes = 40
    print(
        "Evaluation success thresholds: "
        f"distance <= {env_cfg.evaluation.target_error_threshold:.2f} m, "
        f"speed <= {env_cfg.evaluation.stop_velocity_threshold:.2f} m/s"
    )

    completed_episodes = 0
    episode_metrics = []
    robot_index = 0
    joint_index = min(1, env.num_dof - 1)
    total_steps = 0
    # A timeout is a valid completed episode.  This guard only protects
    # against a broken reset/termination implementation.
    max_total_steps = int(target_eval_episodes) * (int(env.max_episode_length) + 2)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    all_trajectories = []
    all_starts = []
    all_goals = []

    start_marker = gymutil.WireframeSphereGeometry(
        0.12, 8, 8, None, color=(0.0, 1.0, 0.0)
    )
    goal_marker = gymutil.WireframeSphereGeometry(
        0.12, 8, 8, None, color=(1.0, 0.0, 0.0)
    )

    # Drawing every simulator step would create tens of thousands of viewer
    # line segments.  This only down-samples the visualization; the complete
    # trajectories are still kept in all_trajectories and saved to NPZ below.
    trajectory_draw_stride = 5

    def trajectory_to_lines(trajectory):
        """Convert an XY trajectory into ground-level Isaac Gym line data."""
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2:
            return None, None

        sampled = trajectory[::trajectory_draw_stride]
        if not np.array_equal(sampled[-1], trajectory[-1]):
            sampled = np.concatenate((sampled, trajectory[-1:]), axis=0)

        vertices = np.zeros((2 * (sampled.shape[0] - 1), 3), dtype=np.float32)
        vertices[0::2, :2] = sampled[:-1, :2]
        vertices[1::2, :2] = sampled[1:, :2]
        vertices[:, 2] = 0.035
        colors = np.zeros((sampled.shape[0] - 1, 3), dtype=np.float32)
        return np.ascontiguousarray(vertices), colors

    def draw_marker(geometry, position):
        pose = gymapi.Transform(
            p=gymapi.Vec3(float(position[0]), float(position[1]), 0.12),
            r=None,
        )
        gymutil.draw_lines(geometry, env.gym, env.viewer, env.envs[0], pose)

    def draw_evaluation_boundary():
        """Draw the fixed [-5, 5] x [-5, 5] evaluation square."""
        vertices = np.asarray(
            [
                [-5.0, -5.0, 0.025], [5.0, -5.0, 0.025],
                [5.0, -5.0, 0.025], [5.0, 5.0, 0.025],
                [5.0, 5.0, 0.025], [-5.0, 5.0, 0.025],
                [-5.0, 5.0, 0.025], [-5.0, -5.0, 0.025],
            ],
            dtype=np.float32,
        )
        colors = np.tile(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            (4, 1),
        )
        env.gym.add_lines(
            env.viewer,
            env.envs[0],
            4,
            np.ascontiguousarray(vertices),
            np.ascontiguousarray(colors),
        )

    def draw_all_visuals(current_episode=None):
        """Draw every saved trajectory plus all starts/goals in the viewer."""
        if env.viewer is None:
            return

        env.gym.clear_lines(env.viewer)
        draw_evaluation_boundary()

        # Completed trajectories: green for success and orange for failure.
        for index, trajectory in enumerate(all_trajectories):
            vertices, colors = trajectory_to_lines(trajectory)
            if vertices is None:
                continue
            if index < len(episode_metrics) and episode_metrics[index]["success"]:
                color = (0.0, 1.0, 0.0)
            else:
                color = (1.0, 0.45, 0.0)
            colors[:] = color
            env.gym.add_lines(
                env.viewer,
                env.envs[0],
                colors.shape[0],
                vertices,
                np.ascontiguousarray(colors),
            )

        # Current trajectory: cyan, so it is easy to distinguish from the
        # completed episodes while the test is still running.
        if current_episode is not None:
            current_positions = np.asarray(current_episode["positions"], dtype=np.float32)
            vertices, colors = trajectory_to_lines(current_positions)
            if vertices is not None:
                colors[:] = (0.0, 0.8, 1.0)
                env.gym.add_lines(
                    env.viewer,
                    env.envs[0],
                    colors.shape[0],
                    vertices,
                    np.ascontiguousarray(colors),
                )

        # Redraw all markers after clear_lines; otherwise they disappear as
        # soon as the next simulator step starts.
        for start in all_starts:
            draw_marker(start_marker, start)
        for goal in all_goals:
            draw_marker(goal_marker, goal)
        if current_episode is not None:
            draw_marker(start_marker, current_episode["start_pos"].detach().cpu().numpy())
            draw_marker(goal_marker, current_episode["goal_pos"].detach().cpu().numpy())

    def begin_episode():
        start_pos = env.root_states[0, :2].detach().clone()
        goal_pos = env.commands[0, :2].detach().clone()
        initial_yaw = float(env.base_euler_tensor[0, 2].detach().cpu().item())
        return {
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "previous_pos": start_pos.clone(),
            "path_length": 0.0,
            "steps": 0,
            "initial_yaw": initial_yaw,
            "closest_distance": float(torch.linalg.norm(goal_pos - start_pos).item()),
            "balance_sum": 0.0,
            "positions": [start_pos.detach().cpu().numpy().copy()],
        }

    episode = begin_episode()

    while completed_episodes < int(target_eval_episodes) and total_steps < max_total_steps:
        draw_all_visuals(episode)
        # Log the first episode for a trajectory/state diagnostic plot.
        if completed_episodes == 0:
            logger.log_states(
                {
                    "base_pos_x": env.root_states[robot_index, 0].item(),
                    "base_pos_y": env.root_states[robot_index, 1].item(),
                    "ref_pos_x": env.commands[robot_index, 0].item(),
                    "ref_pos_y": env.commands[robot_index, 1].item(),
                    "goal_dist": env.goal_dist[robot_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                }
            )

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        total_steps += 1
        episode["steps"] += 1

        done = bool(dones[0].item())
        if done:
            # The task cached the terminal pose before the automatic reset.
            terminal_pos = env.terminal_position[0].detach()
            goal_distance_sample = env.terminal_goal_dist[0].detach()
            balance_reward_sample = env.terminal_balance_reward[0].detach()
        else:
            terminal_pos = env.root_states[0, :2].detach()
            goal_distance_sample = env.goal_dist[0].detach()
            # Paper Table II balance reward (pitch/yaw angular rates).
            balance_reward_sample = torch.exp(
                -torch.sum(torch.square(env.base_ang_vel[0, 1:3]))
            )
        episode["path_length"] += float(
            torch.linalg.norm(terminal_pos - episode["previous_pos"]).item()
        )
        episode["previous_pos"] = terminal_pos.clone()
        episode["positions"].append(terminal_pos.cpu().numpy().copy())
        episode["closest_distance"] = min(
            episode["closest_distance"], float(goal_distance_sample.item())
        )
        episode["balance_sum"] += float(balance_reward_sample.item())

        if done:
            final_distance = float(env.terminal_goal_dist[0].item())
            final_speed = float(env.terminal_speed[0].item())
            success = bool(env.success_buf[0].item())
            shortest_path = float(
                torch.linalg.norm(episode["goal_pos"] - episode["start_pos"]).item()
            )
            shortest_path = max(shortest_path, 1.0e-6)
            path_length = max(episode["path_length"], shortest_path)
            spl = shortest_path / path_length if success else 0.0
            # Paper Table III: Closest Distance to Goal (meters).
            cls = episode["closest_distance"]
            # Paper Balance Metric: mean balance reward over the trajectory,
            # reported as a percentage.  It is not part of the SR flag.
            balance_metric = (
                episode["balance_sum"] / max(episode["steps"], 1) * 100.0
            )
            arrived = final_distance <= float(env_cfg.evaluation.target_error_threshold)
            stopped = final_speed <= float(env_cfg.evaluation.stop_velocity_threshold)

            if success:
                termination = "success"
            elif bool(env.terminal_timeout[0].item()):
                termination = "timeout"
            elif bool(env.terminal_unstable[0].item()):
                termination = "unstable"
            elif bool(env.terminal_out_of_bounds[0].item()):
                termination = "out_of_bounds"
            elif arrived and not stopped:
                termination = "arrived_but_moving"
            else:
                termination = "other"

            episode_metrics.append(
                {
                    "success": success,
                    "spl": spl,
                    "cls": cls,
                    "balance_metric": balance_metric,
                    "path_length": episode["path_length"],
                    "shortest_path": shortest_path,
                    "time_s": episode["steps"] * env.dt,
                    "final_distance": final_distance,
                    "final_speed": final_speed,
                    "initial_yaw": episode["initial_yaw"],
                    "arrived": arrived,
                    "stopped": stopped,
                    "termination": termination,
                }
            )
            all_trajectories.append(np.asarray(episode["positions"], dtype=np.float32))
            all_starts.append(episode["start_pos"].detach().cpu().numpy().copy())
            all_goals.append(episode["goal_pos"].detach().cpu().numpy().copy())
            completed_episodes += 1

            if infos.get("episode"):
                logger.log_rewards(infos["episode"], 1)
            if completed_episodes % 10 == 0 or completed_episodes == int(target_eval_episodes):
                print(f"Evaluated {completed_episodes}/{int(target_eval_episodes)} episodes")

            if completed_episodes == SCREENING_EPISODES:
                screening_successes = int(
                    sum(metric["success"] for metric in episode_metrics)
                )
                if screening_successes <= SCREENING_SUCCESS_CUTOFF:
                    print(
                        f"Checkpoint {checkpoint} rejected by screening: "
                        f"{screening_successes}/{SCREENING_EPISODES} successes "
                        f"(threshold requires > {SCREENING_SUCCESS_CUTOFF}). "
                        "No evaluation data will be saved."
                    )
                    close_environment()
                    return None
                print(
                    f"Checkpoint {checkpoint} passed screening: "
                    f"{screening_successes}/{SCREENING_EPISODES} successes; "
                    f"continuing to {target_eval_episodes} episodes."
                )

            if completed_episodes < int(target_eval_episodes):
                # LeggedRobot has already reset the single environment.  The
                # returned observation is the first observation of the next
                # episode, so start collecting its metrics immediately.
                episode = begin_episode()

        if RECORD_FRAMES:
            if total_steps % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

    if completed_episodes < int(target_eval_episodes):
        raise RuntimeError(
            f"Evaluation stopped after {total_steps} steps with "
            f"only {completed_episodes}/{int(target_eval_episodes)} completed episodes."
        )

    success_values = np.asarray([m["success"] for m in episode_metrics], dtype=np.float32)
    spl_values = np.asarray([m["spl"] for m in episode_metrics], dtype=np.float32)
    cls_values = np.asarray([m["cls"] for m in episode_metrics], dtype=np.float32)
    balance_values = np.asarray([m["balance_metric"] for m in episode_metrics], dtype=np.float32)
    path_values = np.asarray([m["path_length"] for m in episode_metrics], dtype=np.float32)
    shortest_values = np.asarray([m["shortest_path"] for m in episode_metrics], dtype=np.float32)
    time_values = np.asarray([m["time_s"] for m in episode_metrics], dtype=np.float32)
    final_distance_values = np.asarray([m["final_distance"] for m in episode_metrics], dtype=np.float32)
    final_speed_values = np.asarray([m["final_speed"] for m in episode_metrics], dtype=np.float32)
    yaw_values = np.asarray([m["initial_yaw"] for m in episode_metrics], dtype=np.float32)
    successful = success_values > 0.5

    print("\nSequential point-to-point evaluation")
    print(f"Episodes: {len(episode_metrics)}")
    print(f"SR: {int(success_values.sum())}/{len(episode_metrics)} = {success_values.mean():.4%}")
    print(f"SPL: {spl_values.mean():.6f}")
    print(f"CLS (closest distance to goal): {cls_values.mean():.6f} m")
    print(f"Balance Metric (mean paper balance reward): {balance_values.mean():.4f}%")
    print(f"Average path length: {path_values.mean():.4f} m")
    print(f"Average shortest path: {shortest_values.mean():.4f} m")
    print(f"Average final distance: {final_distance_values.mean():.4f} m")
    print(f"Average final speed: {final_speed_values.mean():.4f} m/s")
    print(f"Arrival rate (distance only): {np.mean([m['arrived'] for m in episode_metrics]):.4%}")
    print(f"Stopped rate (speed only): {np.mean([m['stopped'] for m in episode_metrics]):.4%}")
    if np.any(successful):
        print(f"Successful average time: {time_values[successful].mean():.4f} s")
        print(f"Successful average final distance: {final_distance_values[successful].mean():.4f} m")
        print(f"Successful average final speed: {final_speed_values[successful].mean():.4f} m/s")
    circular_mean_yaw = np.arctan2(np.mean(np.sin(yaw_values)), np.mean(np.cos(yaw_values)))
    print(
        "Initial yaw check: "
        f"min={yaw_values.min():.3f}, max={yaw_values.max():.3f}, "
        f"std={yaw_values.std():.3f}, circular_mean={circular_mean_yaw:.3f} rad"
    )
    termination_counts = {}
    for metric in episode_metrics:
        key = metric["termination"]
        termination_counts[key] = termination_counts.get(key, 0) + 1
    print("Termination counts: " + ", ".join(
        f"{key}={value}" for key, value in sorted(termination_counts.items())
    ))

    # Save every complete trajectory, not only the first one used by Logger.
    # Object arrays preserve the different episode lengths without padding.
    trajectory_path = None
    if getattr(args, "load_run", None) not in (None, "", "-1"):
        trajectory_dir = os.path.abspath(args.load_run)
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_path = os.path.join(
            trajectory_dir,
            f"evaluation_checkpoint_{int(checkpoint):05d}.npz",
        )
        trajectory_objects = np.empty(len(all_trajectories), dtype=object)
        for index, trajectory in enumerate(all_trajectories):
            trajectory_objects[index] = trajectory
        np.savez_compressed(
            trajectory_path,
            checkpoint=np.asarray([int(checkpoint)], dtype=np.int64),
            trajectories=trajectory_objects,
            starts=np.asarray(all_starts, dtype=np.float32),
            goals=np.asarray(all_goals, dtype=np.float32),
            success=success_values,
            path_length=path_values,
            shortest_path=shortest_values,
            cls=cls_values,
            spl=spl_values,
            balance_metric=balance_values,
            time_s=time_values,
            final_distance=final_distance_values,
            final_speed=final_speed_values,
            initial_yaw=yaw_values,
            arrived=np.asarray([m["arrived"] for m in episode_metrics], dtype=np.bool_),
            stopped=np.asarray([m["stopped"] for m in episode_metrics], dtype=np.bool_),
            termination=np.asarray(
                [m["termination"] for m in episode_metrics], dtype=object
            ),
        )
        print(f"Saved all trajectories to: {trajectory_path}")

    summary = {
        "checkpoint": int(checkpoint),
        "episodes": int(len(episode_metrics)),
        "success_count": int(success_values.sum()),
        "success_rate": float(success_values.mean()),
        "spl": float(spl_values.mean()),
        "cls_m": float(cls_values.mean()),
        "balance_metric_percent": float(balance_values.mean()),
        "average_path_length_m": float(path_values.mean()),
        "average_shortest_path_m": float(shortest_values.mean()),
        "average_final_distance_m": float(final_distance_values.mean()),
        "average_final_speed_mps": float(final_speed_values.mean()),
        "arrival_rate": float(np.mean([m["arrived"] for m in episode_metrics])),
        "stopped_rate": float(np.mean([m["stopped"] for m in episode_metrics])),
        "termination_counts": json.dumps(termination_counts, ensure_ascii=False),
        "successful_average_time_s": float(time_values[successful].mean())
        if np.any(successful) else None,
    }
    if getattr(args, "load_run", None) not in (None, "", "-1"):
        json_path, csv_path = _write_checkpoint_summary(args.load_run, summary)
        print(f"Saved checkpoint summary to: {json_path}")
        print(f"Updated cumulative summary: {csv_path}")

    if PLOT_EVALUATION and logger.state_log.get("base_pos_x"):
        logger.plot_trajectories()

    # Leave the final viewer frame containing every completed trajectory and
    # every start/goal marker, including the last episode.
    draw_all_visuals()

    # This script creates a fresh simulator for each checkpoint.  Release the
    # previous viewer/simulator before waiting for the next model file.
    if EXPORT_POLICY:
        path = os.path.join(
            os.path.abspath(args.load_run),
            "exported",
            f"checkpoint_{int(checkpoint):05d}",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    close_environment()
    return summary


def play(args):
    """Continuously evaluate model_0, model_50, ..., model_15000.

    The training process writes checkpoints every 50 iterations.  If the next
    checkpoint is not present yet, this process waits and polls the run folder.
    """
    if not getattr(args, "load_run", None) or args.load_run == "-1":
        raise ValueError(
            "Continuous checkpoint evaluation requires --load_run to be the "
            "absolute training run directory."
        )

    run_dir = os.path.abspath(args.load_run)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_start = CHECKPOINT_START
    if args.checkpoint not in (None, -1, CHECKPOINT_START):
        print(
            f"Auto-evaluation is configured to start at checkpoint "
            f"{CHECKPOINT_START}; ignoring --checkpoint {args.checkpoint}."
        )

    checkpoints = range(
        checkpoint_start,
        CHECKPOINT_END + 1,
        CHECKPOINT_INTERVAL,
    )
    for checkpoint in checkpoints:
        npz_path = os.path.join(
            run_dir, f"evaluation_checkpoint_{checkpoint:05d}.npz"
        )
        json_path = os.path.join(
            run_dir, f"evaluation_checkpoint_{checkpoint:05d}.json"
        )
        if os.path.isfile(npz_path) and os.path.isfile(json_path):
            print(f"Checkpoint {checkpoint} already evaluated; skipping.")
            continue

        model_path = _wait_for_checkpoint(run_dir, checkpoint)
        print(f"\n===== Evaluating checkpoint {checkpoint}: {model_path} =====")
        return_code = _run_checkpoint_in_child_process(checkpoint)
        if return_code != 0:
            print(
                f"Checkpoint {checkpoint} evaluation process exited with "
                f"code {return_code}; no result was saved. Continuing."
            )

    print(
        f"\nFinished continuous evaluation through checkpoint "
        f"{CHECKPOINT_END}. Results are in: {run_dir}"
    )

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    single_checkpoint = os.environ.get("ROTUNBOT_SINGLE_CHECKPOINT")
    if single_checkpoint is not None:
        evaluate_checkpoint(args, int(single_checkpoint))
    else:
        play(args)
