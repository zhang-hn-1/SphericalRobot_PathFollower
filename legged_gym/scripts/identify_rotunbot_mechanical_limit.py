"""Search for sustained Rotunbot circles without using a learned policy.

The older constant-action sweep only measures what a fixed lateral lean can
do. Rotunbot steering can require coordinated time-varying motion, so this
script additionally searches phase-coupled periodic low-level commands:

    a0(t) = drive_bias + drive_amplitude * sin(2*pi*f*t)
    a1(t) = steer_bias + steer_amplitude * sin(2*pi*f*t + phase)

Each candidate controls one Isaac Gym environment directly. A candidate is
called a sustained circle only if it keeps the requested speed range, turns
mostly in one direction, accumulates enough heading change, has bounded yaw
rate variation, and agrees with a circle fitted to the ball-center path.

Finding a stable radius below a proposed limit proves reachability in this
nominal simulator. Not finding one is not a proof of mechanical impossibility;
it only bounds this finite controller family and search budget.
"""

import csv
import json
import math
import os
from datetime import datetime

import isaacgym  # noqa: F401: register Isaac Gym before environments
from isaacgym import gymutil
import torch

from legged_gym.envs import *  # noqa: F401,F403: task registration side effect
from legged_gym.utils import set_seed, task_registry


def parse_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "rotunbot_vel_clean",
            "help": "registered Rotunbot task",
        },
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 4},
        {
            "name": "--num_candidates",
            "type": int,
            "default": 2048,
            "help": "number of parallel low-level controllers",
        },
        {
            "name": "--constant_points",
            "type": int,
            "default": 11,
            "help": "constant-action grid width included before periodic samples",
        },
        {
            "name": "--constant_drive_action_max",
            "type": float,
            "default": 0.50,
            "help": "absolute normalized joint-0 action in the locked-steer grid",
        },
        {
            "name": "--preload_seconds",
            "type": float,
            "default": 2.0,
            "help": "hold the lateral angle before starting joint-0 motion",
        },
        {
            "name": "--steer_position_limit_rad",
            "type": float,
            "default": 0.5236,
            "help": "mechanical lateral-joint target used by action +/-1 [rad]",
        },
        {"name": "--settle_seconds", "type": float, "default": 4.0},
        {"name": "--measure_seconds", "type": float, "default": 12.0},
        {
            "name": "--target_speed_min",
            "type": float,
            "default": 0.15,
            "help": "lower center-speed bound for reported mechanical limit [m/s]",
        },
        {
            "name": "--target_speed_max",
            "type": float,
            "default": 0.25,
            "help": "upper center-speed bound for reported mechanical limit [m/s]",
        },
        {"name": "--drive_bias_min", "type": float, "default": 0.05},
        {"name": "--drive_bias_max", "type": float, "default": 0.45},
        {"name": "--drive_amplitude_max", "type": float, "default": 0.25},
        {"name": "--steer_bias_max", "type": float, "default": 0.80},
        {"name": "--steer_amplitude_max", "type": float, "default": 0.70},
        {"name": "--frequency_min_hz", "type": float, "default": 0.20},
        {"name": "--frequency_max_hz", "type": float, "default": 2.00},
        {"name": "--min_valid_fraction", "type": float, "default": 0.90},
        {"name": "--max_speed_cv", "type": float, "default": 0.35},
        {"name": "--max_yaw_rate_cv", "type": float, "default": 0.50},
        {
            "name": "--min_turn_sign_consistency",
            "type": float,
            "default": 0.85,
        },
        {"name": "--min_heading_change_deg", "type": float, "default": 45.0},
        {
            "name": "--max_circle_residual_ratio",
            "type": float,
            "default": 0.15,
        },
        {
            "name": "--max_radius_agreement_ratio",
            "type": float,
            "default": 0.50,
        },
        {"name": "--max_lateral_ratio", "type": float, "default": 0.60},
        {"name": "--max_target_saturation", "type": float, "default": 0.10},
        {"name": "--max_joint_position_ratio", "type": float, "default": 0.98},
        {"name": "--max_torque_ratio", "type": float, "default": 0.98},
        {
            "name": "--radius_hypothesis_m",
            "type": float,
            "default": 3.0,
            "help": "radius whose mechanical-limit hypothesis should be tested",
        },
        {
            "name": "--output_dir",
            "type": str,
            "default": "",
            "help": "default: logs/rotunbot_mechanical_limit/<timestamp>",
        },
    ]
    args = gymutil.parse_arguments(
        description="Rotunbot direct low-level mechanical-limit search",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def validate_args(args):
    if args.num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if args.constant_points < 1:
        raise ValueError("constant_points must be positive")
    if not 0.0 < args.constant_drive_action_max <= 1.0:
        raise ValueError("constant_drive_action_max must lie in (0, 1]")
    if args.preload_seconds < 0.0:
        raise ValueError("preload_seconds cannot be negative")
    if not 0.0 < args.steer_position_limit_rad <= 0.5236:
        raise ValueError("steer_position_limit_rad must lie in (0, 0.5236]")
    if args.settle_seconds <= 0.0 or args.measure_seconds <= 0.0:
        raise ValueError("settle_seconds and measure_seconds must be positive")
    if not 0.0 < args.target_speed_min < args.target_speed_max:
        raise ValueError("target speed bounds must satisfy 0 < min < max")
    if not 0.0 <= args.drive_bias_min <= args.drive_bias_max <= 1.0:
        raise ValueError("drive bias bounds must lie in [0, 1]")
    if not 0.0 <= args.frequency_min_hz <= args.frequency_max_hz:
        raise ValueError("frequency bounds are invalid")


def prepare_environment(args, total_seconds):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = int(args.num_candidates)
    env_cfg.env.episode_length_s = max(float(total_seconds) + 2.0, 5.0)
    # The clean controller uses 0.45 rad during policy training. Mechanical
    # reachability should exercise the full URDF joint limit (0.5236 rad).
    env_cfg.control.second_actionScale = float(args.steer_position_limit_rad)
    env_cfg.control.second_pos_limits = float(args.steer_position_limit_rad)
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

    # The candidate count defines the environment count for this direct test.
    original_num_envs = getattr(args, "num_envs", None)
    args.num_envs = None
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    args.num_envs = original_num_envs
    return env, env_cfg


def uniform(generator, count, low, high):
    return low + (high - low) * torch.rand(count, generator=generator)


def build_candidates(args, device):
    """Include a constant grid, then fill the budget with mirrored periodic trials."""
    count = int(args.num_candidates)
    constant_width = min(int(args.constant_points), int(math.sqrt(count)))
    constant_drive_axis = torch.linspace(
        -float(args.constant_drive_action_max),
        float(args.constant_drive_action_max),
        constant_width,
    )
    constant_steer_axis = torch.linspace(-1.0, 1.0, constant_width)
    constant_drive, constant_steer = torch.meshgrid(
        constant_drive_axis, constant_steer_axis, indexing="ij"
    )
    constant_count = min(count, constant_drive.numel())

    drive_bias = torch.zeros(count)
    drive_amplitude = torch.zeros(count)
    steer_bias = torch.zeros(count)
    steer_amplitude = torch.zeros(count)
    frequency_hz = torch.zeros(count)
    phase_rad = torch.zeros(count)

    drive_bias[:constant_count] = constant_drive.reshape(-1)[:constant_count]
    steer_bias[:constant_count] = constant_steer.reshape(-1)[:constant_count]

    remaining = count - constant_count
    if remaining > 0:
        generator = torch.Generator().manual_seed(int(args.seed) + 1701)
        base_count = int(math.ceil(remaining / 4.0))
        drive_magnitude = uniform(
            generator,
            base_count,
            float(args.drive_bias_min),
            float(args.drive_bias_max),
        )
        drive_wave = uniform(
            generator, base_count, 0.0, float(args.drive_amplitude_max)
        )
        steer_magnitude = uniform(
            generator, base_count, 0.0, float(args.steer_bias_max)
        )
        steer_wave = uniform(
            generator, base_count, 0.0, float(args.steer_amplitude_max)
        )
        frequency = uniform(
            generator,
            base_count,
            float(args.frequency_min_hz),
            float(args.frequency_max_hz),
        )
        phase = uniform(generator, base_count, 0.0, 2.0 * math.pi)

        variants = []
        for drive_sign, steer_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            variants.append(
                torch.stack(
                    (
                        drive_sign * drive_magnitude,
                        drive_sign * drive_wave,
                        steer_sign * steer_magnitude,
                        steer_sign * steer_wave,
                        frequency,
                        phase,
                    ),
                    dim=1,
                )
            )
        periodic = torch.cat(variants, dim=0)[:remaining]
        start = constant_count
        drive_bias[start:] = periodic[:, 0]
        drive_amplitude[start:] = periodic[:, 1]
        steer_bias[start:] = periodic[:, 2]
        steer_amplitude[start:] = periodic[:, 3]
        frequency_hz[start:] = periodic[:, 4]
        phase_rad[start:] = periodic[:, 5]

    return {
        "drive_bias": drive_bias.to(device),
        "drive_amplitude": drive_amplitude.to(device),
        "steer_bias": steer_bias.to(device),
        "steer_amplitude": steer_amplitude.to(device),
        "frequency_hz": frequency_hz.to(device),
        "phase_rad": phase_rad.to(device),
        "constant_count": constant_count,
    }


def candidate_actions(candidates, time_s):
    phase = 2.0 * math.pi * candidates["frequency_hz"] * float(time_s)
    drive = candidates["drive_bias"] + candidates["drive_amplitude"] * torch.sin(
        phase
    )
    steer = candidates["steer_bias"] + candidates["steer_amplitude"] * torch.sin(
        phase + candidates["phase_rad"]
    )
    return torch.clamp(torch.stack((drive, steer), dim=1), -1.0, 1.0)


def fit_circles(position_samples):
    """Vectorized algebraic circle fit for [time, env, xy] center paths."""
    positions = position_samples.permute(1, 0, 2)
    positions = positions - positions[:, :1, :]
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    design = torch.stack((2.0 * x, 2.0 * y, torch.ones_like(x)), dim=2)
    target = (x.square() + y.square()).unsqueeze(2)
    normal = design.transpose(1, 2).bmm(design)
    rhs = design.transpose(1, 2).bmm(target)
    scale = torch.diagonal(normal, dim1=1, dim2=2).mean(dim=1).clamp_min(1.0)
    eye = torch.eye(3, dtype=normal.dtype, device=normal.device).unsqueeze(0)
    normal = normal + 1.0e-6 * scale[:, None, None] * eye
    solution = torch.linalg.solve(normal, rhs).squeeze(2)
    center_x = solution[:, 0]
    center_y = solution[:, 1]
    radius_squared = center_x.square() + center_y.square() + solution[:, 2]
    radius = torch.sqrt(torch.clamp(radius_squared, min=1.0e-12))
    radial_distance = torch.sqrt(
        (x - center_x[:, None]).square() + (y - center_y[:, None]).square()
    )
    residual = torch.sqrt(
        torch.mean((radial_distance - radius[:, None]).square(), dim=1)
    )
    residual_ratio = residual / radius.clamp_min(1.0e-6)
    return radius, residual_ratio


def run_search(env, env_cfg, candidates, args):
    device = env.device
    count = int(args.num_candidates)
    preload_steps = max(0, int(round(args.preload_seconds / env.dt)))
    settle_steps = max(1, int(round(args.settle_seconds / env.dt)))
    measure_steps = max(2, int(round(args.measure_seconds / env.dt)))
    total_steps = settle_steps + measure_steps

    env.reset()
    env.commands.zero_()

    positions = torch.empty((measure_steps, count, 2), device=device)
    speeds = torch.empty((measure_steps, count), device=device)
    forward_velocities = torch.empty((measure_steps, count), device=device)
    lateral_ratios = torch.empty((measure_steps, count), device=device)
    yaw_rates = torch.empty((measure_steps, count), device=device)
    yaw_valid = torch.empty((measure_steps, count), dtype=torch.bool, device=device)
    tilt_sines = torch.empty((measure_steps, count), device=device)
    target_saturated = torch.empty(
        (measure_steps, count), dtype=torch.bool, device=device
    )
    joint_position_ratios = torch.empty((measure_steps, count), device=device)
    torque_ratios = torch.empty((measure_steps, count), device=device)
    early_termination = torch.zeros(count, dtype=torch.bool, device=device)

    torque_limits = torch.tensor(
        [env_cfg.control.torque_limits_1, env_cfg.control.torque_limits_2],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1.0e-6)
    second_limit = float(env_cfg.control.second_pos_limits)

    with torch.no_grad():
        # Reproduce the physical experiment: first move the lateral joint to
        # its chosen angle while joint 0 is stopped, then start driving.
        preload_actions = torch.zeros((count, 2), device=device)
        preload_actions[:, 1] = torch.clamp(candidates["steer_bias"], -1.0, 1.0)
        for _ in range(preload_steps):
            env.commands.zero_()
            _, _, _, dones, _ = env.step(preload_actions)
            timeout = getattr(env, "time_out_buf", torch.zeros_like(dones))
            early_termination |= dones.bool() & ~timeout.bool()
        reset_window = getattr(env, "_reset_trajectory_window", None)
        if reset_window is not None:
            reset_window(torch.arange(count, device=device, dtype=torch.long))

        for step in range(total_steps):
            env.commands.zero_()
            actions = candidate_actions(candidates, step * env.dt)
            _, _, _, dones, _ = env.step(actions)
            timeout = getattr(env, "time_out_buf", torch.zeros_like(dones))
            early_termination |= dones.bool() & ~timeout.bool()
            if step < settle_steps:
                continue

            index = step - settle_steps
            speed = env.trajectory_speed.detach()
            positions[index] = env.root_states[:, :2].detach()
            speeds[index] = speed
            forward_velocities[index] = env.trajectory_lin_vel[:, 0].detach()
            lateral_ratios[index] = (
                torch.abs(env.trajectory_lin_vel[:, 1]) / speed.clamp_min(0.05)
            )
            yaw_rates[index] = env.trajectory_ang_vel.detach()
            yaw_valid[index] = env.trajectory_ang_vel_valid.detach()
            tilt_sines[index] = torch.linalg.vector_norm(
                env.projected_gravity[:, :2], dim=1
            )
            target_saturated[index] = (
                torch.abs(env.output_actions[:, 1]) >= 0.95 * second_limit
            )
            joint_position_ratios[index] = (
                torch.abs(env.dof_pos[:, 1]) / second_limit
            )
            torque_ratios[index] = torch.max(
                torch.abs(env.torques) / torque_limits.unsqueeze(0), dim=1
            ).values

    valid_float = yaw_valid.float()
    valid_count = valid_float.sum(dim=0).clamp_min(1.0)
    valid_fraction = valid_float.mean(dim=0)
    speed_mean = speeds.mean(dim=0)
    speed_cv = speeds.std(dim=0, unbiased=False) / speed_mean.clamp_min(0.05)
    forward_velocity_mean = forward_velocities.mean(dim=0)
    yaw_rate_mean = (yaw_rates * valid_float).sum(dim=0) / valid_count
    yaw_deviation = (yaw_rates - yaw_rate_mean.unsqueeze(0)) * valid_float
    yaw_rate_std = torch.sqrt(yaw_deviation.square().sum(dim=0) / valid_count)
    yaw_rate_cv = yaw_rate_std / torch.abs(yaw_rate_mean).clamp_min(0.01)
    turn_direction = torch.sign(yaw_rate_mean)
    sign_consistency = (
        ((yaw_rates * turn_direction.unsqueeze(0)) > 0.0).float()
        * valid_float
    ).sum(dim=0) / valid_count
    heading_change_deg = torch.abs(
        (yaw_rates * valid_float).sum(dim=0) * env.dt
    ) * (180.0 / math.pi)
    kappa = yaw_rate_mean / speed_mean.clamp_min(0.05)
    radius_kinematic = speed_mean / torch.abs(yaw_rate_mean).clamp_min(1.0e-6)
    radius_circle, circle_residual_ratio = fit_circles(positions)
    radius_agreement_ratio = torch.abs(radius_circle - radius_kinematic) / torch.maximum(
        radius_circle, radius_kinematic
    ).clamp_min(1.0e-6)
    lateral_ratio_95 = torch.quantile(lateral_ratios, 0.95, dim=0)
    target_saturation_fraction = target_saturated.float().mean(dim=0)
    joint_position_ratio_95 = torch.quantile(joint_position_ratios, 0.95, dim=0)
    torque_ratio_95 = torch.quantile(torque_ratios, 0.95, dim=0)
    tilt_sin_max = tilt_sines.max(dim=0).values
    tilt_deg_max = torch.asin(torch.clamp(tilt_sin_max, 0.0, 1.0)) * (
        180.0 / math.pi
    )
    path_steps = positions[1:] - positions[:-1]
    path_length = torch.linalg.vector_norm(path_steps, dim=2).sum(dim=0)

    # Mechanical reachability intentionally permits commanding a legal joint
    # stop. Target/position margin is reported separately as a practical tier.
    stable_circle = (
        (speed_mean >= float(args.target_speed_min))
        & (speed_mean <= float(args.target_speed_max))
        & (speed_cv <= float(args.max_speed_cv))
        & (valid_fraction >= float(args.min_valid_fraction))
        & (torch.abs(yaw_rate_mean) >= 0.01)
        & (yaw_rate_cv <= float(args.max_yaw_rate_cv))
        & (sign_consistency >= float(args.min_turn_sign_consistency))
        & (heading_change_deg >= float(args.min_heading_change_deg))
        & (circle_residual_ratio <= float(args.max_circle_residual_ratio))
        & (radius_agreement_ratio <= float(args.max_radius_agreement_ratio))
        & (lateral_ratio_95 <= float(args.max_lateral_ratio))
        & (torque_ratio_95 <= float(args.max_torque_ratio))
        & torch.isfinite(radius_circle)
        & ~early_termination
    )

    rows = []
    for index in range(count):
        stable = bool(stable_circle[index].item())
        stable_joint_margin = bool(
            stable
            and target_saturation_fraction[index].item()
            <= float(args.max_target_saturation)
            and joint_position_ratio_95[index].item()
            <= float(args.max_joint_position_ratio)
        )
        row = {
            "candidate_id": index,
            "controller_type": (
                "constant" if index < candidates["constant_count"] else "periodic"
            ),
            "drive_bias": float(candidates["drive_bias"][index].item()),
            "drive_amplitude": float(candidates["drive_amplitude"][index].item()),
            "steer_bias": float(candidates["steer_bias"][index].item()),
            "steer_amplitude": float(candidates["steer_amplitude"][index].item()),
            "frequency_hz": float(candidates["frequency_hz"][index].item()),
            "phase_rad": float(candidates["phase_rad"][index].item()),
            "speed_mean_mps": float(speed_mean[index].item()),
            "forward_velocity_mean_mps": float(forward_velocity_mean[index].item()),
            "speed_cv": float(speed_cv[index].item()),
            "yaw_rate_mean_radps": float(yaw_rate_mean[index].item()),
            "yaw_rate_std_radps": float(yaw_rate_std[index].item()),
            "yaw_rate_cv": float(yaw_rate_cv[index].item()),
            "valid_fraction": float(valid_fraction[index].item()),
            "turn_sign_consistency": float(sign_consistency[index].item()),
            "heading_change_deg": float(heading_change_deg[index].item()),
            "kappa_mean_1_per_m": float(kappa[index].item()),
            "radius_kinematic_m": float(radius_kinematic[index].item()),
            "radius_circle_fit_m": float(radius_circle[index].item()),
            "circle_residual_ratio": float(circle_residual_ratio[index].item()),
            "radius_agreement_ratio": float(radius_agreement_ratio[index].item()),
            "path_length_m": float(path_length[index].item()),
            "lateral_ratio_95": float(lateral_ratio_95[index].item()),
            "target_saturation_fraction": float(
                target_saturation_fraction[index].item()
            ),
            "joint_position_ratio_95": float(joint_position_ratio_95[index].item()),
            "torque_ratio_95": float(torque_ratio_95[index].item()),
            "tilt_deg_max": float(tilt_deg_max[index].item()),
            "early_termination": bool(early_termination[index].item()),
            "stable_circle": stable,
            "stable_joint_margin": stable_joint_margin,
            "stable_tilt_20deg": stable and tilt_deg_max[index].item() <= 20.0,
            "stable_tilt_15deg": stable and tilt_deg_max[index].item() <= 15.0,
            "stable_tilt_10deg": stable and tilt_deg_max[index].item() <= 10.0,
        }
        rows.append(row)
    return rows, preload_steps, settle_steps, measure_steps


def make_summary(rows, args):
    tiers = (
        ("stable_no_tilt_cap", "stable_circle"),
        ("stable_with_joint_margin", "stable_joint_margin"),
        ("stable_tilt_20deg", "stable_tilt_20deg"),
        ("stable_tilt_15deg", "stable_tilt_15deg"),
        ("stable_tilt_10deg", "stable_tilt_10deg"),
    )
    summary = []
    for tier_name, field in tiers:
        selected = [row for row in rows if row[field]]
        best = min(selected, key=lambda row: row["radius_circle_fit_m"]) if selected else None
        summary.append(
            {
                "tier": tier_name,
                "num_stable_candidates": len(selected),
                "min_radius_circle_fit_m": (
                    best["radius_circle_fit_m"] if best else float("nan")
                ),
                "radius_kinematic_m": (
                    best["radius_kinematic_m"] if best else float("nan")
                ),
                "speed_mean_mps": best["speed_mean_mps"] if best else float("nan"),
                "yaw_rate_mean_radps": (
                    best["yaw_rate_mean_radps"] if best else float("nan")
                ),
                "tilt_deg_max": best["tilt_deg_max"] if best else float("nan"),
                "candidate_id": best["candidate_id"] if best else -1,
                "controller_type": best["controller_type"] if best else "none",
                "radius_below_hypothesis": (
                    bool(best["radius_circle_fit_m"] < args.radius_hypothesis_m)
                    if best
                    else False
                ),
            }
        )
    return summary


def write_outputs(output_dir, rows, summary, metadata):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "rotunbot_mechanical_limit_sweep.csv")
    summary_path = os.path.join(output_dir, "rotunbot_mechanical_limit_summary.csv")
    metadata_path = os.path.join(output_dir, "rotunbot_mechanical_limit_metadata.json")
    with open(raw_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return raw_path, summary_path, metadata_path


def print_summary(summary, rows, args):
    print("Stable-circle mechanical-limit estimates near the requested speed:")
    for item in summary:
        if item["candidate_id"] < 0:
            print(f"  {item['tier']}: no stable candidate found")
            continue
        print(
            "  {}: R_fit={:.3f} m, R_vw={:.3f} m, v={:.3f} m/s, "
            "w={:+.3f} rad/s, tilt={:.1f} deg, candidate={}".format(
                item["tier"],
                item["min_radius_circle_fit_m"],
                item["radius_kinematic_m"],
                item["speed_mean_mps"],
                item["yaw_rate_mean_radps"],
                item["tilt_deg_max"],
                item["candidate_id"],
            )
        )

    unrestricted = summary[0]
    if unrestricted["radius_below_hypothesis"]:
        print(
            "RESULT: found a sustained direct-control circle below {:.3f} m; "
            "the nominal simulator is not mechanically limited to that radius.".format(
                args.radius_hypothesis_m
            )
        )
    else:
        print(
            "RESULT: no sustained direct-control circle below {:.3f} m was found "
            "in this finite search.".format(args.radius_hypothesis_m)
        )
        print(
            "This is not proof of impossibility. Increase candidates, repeat seeds, "
            "or enlarge the controller family before claiming a mechanical limit."
        )

    stable_rows = sorted(
        (row for row in rows if row["stable_circle"]),
        key=lambda row: row["radius_circle_fit_m"],
    )[:10]
    if stable_rows:
        print("Ten smallest stable circles:")
        for row in stable_rows:
            print(
                "  id={candidate_id:4d} {controller_type:8s} R={radius_circle_fit_m:.3f} m "
                "v={speed_mean_mps:.3f} w={yaw_rate_mean_radps:+.3f} "
                "tilt={tilt_deg_max:.1f} deg residual={circle_residual_ratio:.3f} "
                "actions=({drive_bias:+.3f},{drive_amplitude:+.3f},"
                "{steer_bias:+.3f},{steer_amplitude:+.3f},"
                "{frequency_hz:.3f}Hz,{phase_rad:.3f}rad)".format(**row)
            )


def print_filter_diagnostics(rows, args):
    """Show which sustained-circle condition rejects the candidate population."""
    checks = (
        (
            "target speed window",
            lambda row: args.target_speed_min
            <= row["speed_mean_mps"]
            <= args.target_speed_max,
        ),
        ("speed CV", lambda row: row["speed_cv"] <= args.max_speed_cv),
        (
            "valid path rate",
            lambda row: row["valid_fraction"] >= args.min_valid_fraction,
        ),
        ("nonzero yaw rate", lambda row: abs(row["yaw_rate_mean_radps"]) >= 0.01),
        ("yaw-rate CV", lambda row: row["yaw_rate_cv"] <= args.max_yaw_rate_cv),
        (
            "turn sign consistency",
            lambda row: row["turn_sign_consistency"]
            >= args.min_turn_sign_consistency,
        ),
        (
            "heading change",
            lambda row: row["heading_change_deg"] >= args.min_heading_change_deg,
        ),
        (
            "circle residual",
            lambda row: row["circle_residual_ratio"]
            <= args.max_circle_residual_ratio,
        ),
        (
            "radius agreement",
            lambda row: row["radius_agreement_ratio"]
            <= args.max_radius_agreement_ratio,
        ),
        (
            "lateral ratio",
            lambda row: row["lateral_ratio_95"] <= args.max_lateral_ratio,
        ),
        ("torque ratio", lambda row: row["torque_ratio_95"] <= args.max_torque_ratio),
        ("no termination", lambda row: not row["early_termination"]),
    )
    print("Individual mechanical-filter pass counts:")
    for label, predicate in checks:
        print(f"  {label}: {sum(predicate(row) for row in rows)}/{len(rows)}")
    cumulative = list(rows)
    print("Cumulative pass counts:")
    for label, predicate in checks:
        cumulative = [row for row in cumulative if predicate(row)]
        print(f"  through {label}: {len(cumulative)}/{len(rows)}")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    total_seconds = (
        float(args.preload_seconds)
        + float(args.settle_seconds)
        + float(args.measure_seconds)
    )
    env, env_cfg = prepare_environment(args, total_seconds)
    candidates = build_candidates(args, env.device)
    try:
        rows, preload_steps, settle_steps, measure_steps = run_search(
            env, env_cfg, candidates, args
        )
        summary = make_summary(rows, args)
        if args.output_dir:
            output_dir = os.path.abspath(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.abspath(
                os.path.join("logs", "rotunbot_mechanical_limit", timestamp)
            )
        metadata = {
            "task": args.task,
            "seed": args.seed,
            "num_candidates": args.num_candidates,
            "num_constant_candidates": candidates["constant_count"],
            "num_periodic_candidates": (
                args.num_candidates - candidates["constant_count"]
            ),
            "preload_steps": preload_steps,
            "settle_steps": settle_steps,
            "measure_steps": measure_steps,
            "dt": float(env.dt),
            "target_speed_range_mps": [
                args.target_speed_min,
                args.target_speed_max,
            ],
            "radius_hypothesis_m": args.radius_hypothesis_m,
            "steer_position_limit_rad": args.steer_position_limit_rad,
            "stability_thresholds": {
                "min_valid_fraction": args.min_valid_fraction,
                "max_speed_cv": args.max_speed_cv,
                "max_yaw_rate_cv": args.max_yaw_rate_cv,
                "min_turn_sign_consistency": args.min_turn_sign_consistency,
                "min_heading_change_deg": args.min_heading_change_deg,
                "max_circle_residual_ratio": args.max_circle_residual_ratio,
                "max_radius_agreement_ratio": args.max_radius_agreement_ratio,
                "max_lateral_ratio": args.max_lateral_ratio,
                "max_target_saturation": args.max_target_saturation,
                "max_joint_position_ratio": args.max_joint_position_ratio,
                "max_torque_ratio": args.max_torque_ratio,
            },
            "interpretation": (
                "A found candidate proves nominal-simulation reachability. "
                "No candidate is not proof of mechanical impossibility."
            ),
        }
        raw_path, summary_path, metadata_path = write_outputs(
            output_dir, rows, summary, metadata
        )
        print(f"Raw candidates: {raw_path}")
        print(f"Summary: {summary_path}")
        print(f"Metadata: {metadata_path}")
        print_filter_diagnostics(rows, args)
        print_summary(summary, rows, args)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
