"""Identify a conservative speed--curvature envelope for Rotunbot.

This script does not use a learned policy.  It holds the two low-level
commands of ``rotunbot_vel_clean`` fixed:

    action[0] -> joint-1 velocity target (normalised)
    action[1] -> joint-2 position target (normalised)

For every point in the action grid it measures the ball-center speed and the
heading rate of the ground trajectory.  The measured curvature is

    kappa = trajectory_ang_vel / trajectory_speed.

It writes two files:

    rotunbot_curvature_sweep.csv  raw measurements for every action pair
    rotunbot_kmax_v.csv           conservative binned envelope k_max(v)

The result is a first calibration of the current low-level actuator model. It
is not a proof of global feasibility; repeat it for every friction/terrain
condition that matters and validate the final envelope with a separate run.
"""

import csv
import json
import math
import os
import sys
from datetime import datetime

import isaacgym  # noqa: F401: registers Isaac Gym before importing environments
from isaacgym import gymutil
import torch

from legged_gym.envs import *  # noqa: F401,F403: task registration side effect
from legged_gym.utils import class_to_dict, set_seed, task_registry


def parse_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "rotunbot_vel_clean",
            "help": "registered Rotunbot task",
        },
        {"name": "--headless", "action": "store_true", "default": False},
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "device used by the policy side; simulation uses --sim_device",
        },
        {"name": "--num_envs", "type": int, "default": None},
        {"name": "--seed", "type": int, "default": 4},
        {
            "name": "--first_action_min",
            "type": float,
            "default": -1.0,
            "help": "normalised joint-1 velocity target lower bound",
        },
        {
            "name": "--first_action_max",
            "type": float,
            "default": 1.0,
            "help": "normalised joint-1 velocity target upper bound",
        },
        {
            "name": "--second_action_min",
            "type": float,
            "default": -1.0,
            "help": "normalised joint-2 position target lower bound",
        },
        {
            "name": "--second_action_max",
            "type": float,
            "default": 1.0,
            "help": "normalised joint-2 position target upper bound",
        },
        {"name": "--first_points", "type": int, "default":  nine_or_default(9)},
        {"name": "--second_points", "type": int, "default":  nine_or_default(9)},
        {
            "name": "--settle_seconds",
            "type": float,
            "default": 2.0,
            "help": "discard this initial transient for each fixed-action trial",
        },
        {
            "name": "--measure_seconds",
            "type": float,
            "default": 2.0,
            "help": "measurement duration after settling",
        },
        {
            "name": "--min_speed",
            "type": float,
            "default": 0.05,
            "help": "minimum speed for a meaningful curvature estimate [m/s]",
        },
        {
            "name": "--max_speed_cv",
            "type": float,
            "default": 0.35,
            "help": "maximum speed coefficient of variation for feasibility",
        },
        {
            "name": "--max_lateral_ratio",
            "type": float,
            "default": 0.35,
            "help": "95-percentile |lateral velocity|/speed threshold",
        },
        {
            "name": "--max_torque_ratio",
            "type": float,
            "default": 0.98,
            "help": "95-percentile torque/limit threshold",
        },
        {
            "name": "--speed_bins",
            "type": int,
            "default": 12,
            "help": "number of measured-speed bins used for k_max(v)",
        },
        {
            "name": "--safety_factor",
            "type": float,
            "default": 0.80,
            "help": "multiply the measured envelope by this factor",
        },
        {
            "name": "--enforce_monotonic_envelope",
            "action": "store_true",
            "default": False,
            "help": (
                "force k_max to be non-increasing with speed; disabled by "
                "default because Rotunbot data does not support this assumption"
            ),
        },
        {
            "name": "--output_dir",
            "type": str,
            "default": "",
            "help": "output directory; default is logs/rotunbot_curvature_id/<timestamp>",
        },
    ]

    args = gymutil.parse_arguments(
        description="Rotunbot speed-curvature identification",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def nine_or_default(value):
    """Keep a named default expression easy to find in the command line table."""
    return value


def linspace_tensor(start, stop, count, device):
    if count < 2:
        return torch.tensor([float(start)], dtype=torch.float32, device=device)
    return torch.linspace(float(start), float(stop), int(count), device=device)


def quantile_or_nan(values, q):
    """Return a scalar quantile, or NaN when no finite samples are present."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float(torch.quantile(finite, q).item())


def build_action_grid(args, device):
    first = linspace_tensor(
        args.first_action_min, args.first_action_max, args.first_points, device
    )
    second = linspace_tensor(
        args.second_action_min, args.second_action_max, args.second_points, device
    )
    first_grid, second_grid = torch.meshgrid(first, second, indexing="ij")
    actions = torch.stack((first_grid.reshape(-1), second_grid.reshape(-1)), dim=1)
    return actions


def prepare_environment(args, num_envs, total_seconds):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = int(num_envs)
    env_cfg.env.episode_length_s = max(float(total_seconds) + 1.0, 5.0)
    env_cfg.commands.manual_command_mode = True
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s + 1.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_link_com = False
    env_cfg.domain_rand.randomize_base_inertia = False
    env_cfg.domain_rand.randomize_link_inertia = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.add_dof_lag = False
    env_cfg.domain_rand.add_imu_lag = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False

    # ``make_env`` calls update_cfg_from_args and may override num_envs when
    # --num_envs is explicitly supplied. For this script the grid size wins.
    original_num_envs = args.num_envs
    args.num_envs = None
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    args.num_envs = original_num_envs
    return env, env_cfg


def run_sweep(env, env_cfg, actions, args):
    device = env.device
    num_envs = actions.shape[0]
    settle_steps = max(1, int(round(args.settle_seconds / env.dt)))
    measure_steps = max(1, int(round(args.measure_seconds / env.dt)))
    total_steps = settle_steps + measure_steps

    # Reset first. The clean task's trajectory buffers are initialized by the
    # reset step, and all commands are then held at zero during direct tests.
    env.reset()
    env.commands.zero_()

    speed_samples = torch.empty((measure_steps, num_envs), device=device)
    curvature_samples = torch.empty((measure_steps, num_envs), device=device)
    lateral_ratio_samples = torch.empty((measure_steps, num_envs), device=device)
    torque_ratio_samples = torch.empty((measure_steps, num_envs), device=device)
    forward_velocity_samples = torch.empty((measure_steps, num_envs), device=device)

    early_termination = torch.zeros(num_envs, dtype=torch.bool, device=device)
    torque_limits = torch.tensor(
        [env_cfg.control.torque_limits_1, env_cfg.control.torque_limits_2],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1.0e-6)

    with torch.no_grad():
        for step in range(total_steps):
            env.commands.zero_()
            _, _, _, dones, _ = env.step(actions)

            # A timeout is not a physical failure. Contact termination before
            # the requested measurement is treated as an infeasible action.
            timeout = getattr(env, "time_out_buf", torch.zeros_like(dones))
            if step < total_steps - 1:
                early_termination |= dones.bool() & ~timeout.bool()

            if step < settle_steps:
                continue

            index = step - settle_steps
            speed = env.trajectory_speed.detach()
            yaw_rate = env.trajectory_ang_vel.detach()
            valid_speed = speed > float(args.min_speed)
            curvature = torch.where(
                valid_speed,
                yaw_rate / speed.clamp_min(float(args.min_speed)),
                torch.zeros_like(yaw_rate),
            )
            lateral_ratio = torch.abs(env.trajectory_lin_vel[:, 1]) / speed.clamp_min(
                float(args.min_speed)
            )
            torque_ratio = torch.max(
                torch.abs(env.torques) / torque_limits.unsqueeze(0), dim=1
            ).values

            speed_samples[index] = speed
            curvature_samples[index] = curvature
            lateral_ratio_samples[index] = lateral_ratio
            torque_ratio_samples[index] = torque_ratio
            forward_velocity_samples[index] = env.trajectory_lin_vel[:, 0].detach()

    speed_mean = speed_samples.mean(dim=0)
    speed_std = speed_samples.std(dim=0, unbiased=False)
    speed_cv = speed_std / speed_mean.clamp_min(float(args.min_speed))
    measured_kappa = curvature_samples.median(dim=0).values
    abs_kappa = torch.abs(measured_kappa)
    lateral_ratio_95 = torch.quantile(lateral_ratio_samples, 0.95, dim=0)
    torque_ratio_95 = torch.quantile(torque_ratio_samples, 0.95, dim=0)
    forward_velocity = forward_velocity_samples.mean(dim=0)

    feasible = (
        (speed_mean > float(args.min_speed))
        & (speed_cv <= float(args.max_speed_cv))
        & (lateral_ratio_95 <= float(args.max_lateral_ratio))
        & (torque_ratio_95 <= float(args.max_torque_ratio))
        & ~early_termination
    )

    rows = []
    for i in range(num_envs):
        rows.append(
            {
                "action_first": float(actions[i, 0].item()),
                "action_second": float(actions[i, 1].item()),
                "v_measured_mps": float(speed_mean[i].item()),
                "v_forward_mps": float(forward_velocity[i].item()),
                "speed_cv": float(speed_cv[i].item()),
                "kappa_measured_1_per_m": float(measured_kappa[i].item()),
                "turn_radius_m": (
                    float(1.0 / abs_kappa[i].item())
                    if abs_kappa[i].item() > 1.0e-6
                    else float("inf")
                ),
                "lateral_ratio_95": float(lateral_ratio_95[i].item()),
                "torque_ratio_95": float(torque_ratio_95[i].item()),
                "early_termination": bool(early_termination[i].item()),
                "feasible": bool(feasible[i].item()),
            }
        )
    return rows, total_steps


def make_envelope(rows, args):
    feasible_rows = [
        row
        for row in rows
        if row["feasible"]
        and row["v_measured_mps"] > float(args.min_speed)
        and math.isfinite(row["kappa_measured_1_per_m"])
    ]
    if not feasible_rows:
        return []

    speeds = [row["v_measured_mps"] for row in feasible_rows]
    min_speed = min(speeds)
    max_speed = max(speeds)
    if max_speed <= min_speed:
        return []

    bins = torch.linspace(float(min_speed), float(max_speed), args.speed_bins + 1)
    raw = []
    for bin_index in range(args.speed_bins):
        low = float(bins[bin_index].item())
        high = float(bins[bin_index + 1].item())
        if bin_index == args.speed_bins - 1:
            selected = [r for r in feasible_rows if low <= r["v_measured_mps"] <= high]
        else:
            selected = [r for r in feasible_rows if low <= r["v_measured_mps"] < high]
        if not selected:
            continue
        center = sum(r["v_measured_mps"] for r in selected) / len(selected)
        raw.append(
            {
                "v_center_mps": center,
                "kappa_max_raw_1_per_m": max(
                    abs(r["kappa_measured_1_per_m"]) for r in selected
                ),
                "num_samples": len(selected),
            }
        )

    # Retain the independently measured value in every speed bin by default.
    # A cumulative minimum is available only when a separate experiment has
    # justified the assumption that curvature capacity decreases with speed.
    raw.sort(key=lambda row: row["v_center_mps"])
    running = float("inf")
    for row in raw:
        candidate = row["kappa_max_raw_1_per_m"]
        if args.enforce_monotonic_envelope:
            running = min(running, candidate)
            candidate = running
        row["kappa_max_1_per_m"] = float(candidate * args.safety_factor)
        row["omega_max_radps"] = float(row["v_center_mps"] * row["kappa_max_1_per_m"])
        row["turn_radius_min_m"] = (
            float(1.0 / row["kappa_max_1_per_m"])
            if row["kappa_max_1_per_m"] > 1.0e-6
            else float("inf")
        )
    return raw


def write_outputs(output_dir, rows, envelope, metadata):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "rotunbot_curvature_sweep.csv")
    env_path = os.path.join(output_dir, "rotunbot_kmax_v.csv")
    json_path = os.path.join(output_dir, "rotunbot_curvature_id_metadata.json")

    raw_fields = list(rows[0].keys()) if rows else []
    with open(raw_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        if raw_fields:
            writer.writeheader()
            writer.writerows(rows)

    env_fields = list(envelope[0].keys()) if envelope else []
    with open(env_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=env_fields)
        if env_fields:
            writer.writeheader()
            writer.writerows(envelope)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return raw_path, env_path, json_path


def main():
    args = parse_args()
    if args.first_points < 1 or args.second_points < 1:
        raise ValueError("first_points and second_points must be positive")
    if args.speed_bins < 1:
        raise ValueError("speed_bins must be positive")
    if not 0.0 < args.safety_factor <= 1.0:
        raise ValueError("safety_factor must be in (0, 1]")

    seed = args.seed
    set_seed(seed)
    action_count = int(args.first_points) * int(args.second_points)
    total_seconds = float(args.settle_seconds) + float(args.measure_seconds)
    env, env_cfg = prepare_environment(args, action_count, total_seconds)
    actions = build_action_grid(args, env.device)

    try:
        rows, total_steps = run_sweep(env, env_cfg, actions, args)
        envelope = make_envelope(rows, args)
        if args.output_dir:
            output_dir = os.path.abspath(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                "logs", "rotunbot_curvature_id", timestamp
            )
            output_dir = os.path.abspath(output_dir)

        metadata = {
            "task": args.task,
            "num_action_pairs": action_count,
            "settle_steps": total_steps - int(round(args.measure_seconds / env.dt)),
            "measure_steps": int(round(args.measure_seconds / env.dt)),
            "dt": float(env.dt),
            "min_speed_mps": args.min_speed,
            "safety_factor": args.safety_factor,
            "enforce_monotonic_envelope": args.enforce_monotonic_envelope,
            "num_feasible_samples": sum(row["feasible"] for row in rows),
            "note": "kappa = trajectory_ang_vel / trajectory_speed; low-speed rows are excluded",
        }
        raw_path, env_path, json_path = write_outputs(
            output_dir, rows, envelope, metadata
        )
        print(f"Raw sweep: {raw_path}")
        print(f"k_max(v) envelope: {env_path}")
        print(f"Metadata: {json_path}")
        print(f"Feasible action pairs: {metadata['num_feasible_samples']}/{action_count}")
        if envelope:
            print("Conservative envelope samples:")
            for row in envelope:
                print(
                    "  v={:.3f} m/s, k_max={:.3f} 1/m, omega_max={:.3f} rad/s".format(
                        row["v_center_mps"],
                        row["kappa_max_1_per_m"],
                        row["omega_max_radps"],
                    )
                )
        else:
            print("No feasible measured samples; inspect the raw CSV and relax/tune thresholds.")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
