"""Deterministic fixed-command evaluation for rotunbot_vel_clean."""

import csv
import os

import numpy as np

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

import torch

# Test straight motion and progressively harder curves in both directions.
COMMANDS = [
    (0.0, 0.0),
    (0.20, 0.0),
    (-0.20, 0.0),
    (0.20, 0.05),
    (0.20, -0.05),
    (0.20, 0.10),
    (0.20, -0.10),
    (0.20, 0.15),
    (0.20, -0.15),
    (0.20, 0.20),
    (0.20, -0.20),
    (-0.20, 0.05),
    (-0.20, -0.05),
    (-0.20, 0.10),
    (-0.20, -0.10),
    (-0.20, 0.15),
    (-0.20, -0.15),
    (-0.20, 0.20),
    (-0.20, -0.20),
]


def main():
    args = get_args()
    args.task = "rotunbot_vel_clean"
    if not args.load_run:
        raise ValueError("Please provide --load_run for a rotunbot_vel_clean run.")
    args.load_run = os.path.abspath(args.load_run)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.commands.manual_command_mode = True
    env_cfg.commands.resampling_time = 1.0e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = -1 if args.checkpoint is None else args.checkpoint
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)
    restored_curriculum_stage = getattr(env, "tracking_curriculum_stage", 0)
    print(
        "Restored tracking curriculum stage: {} "
        "(0=large radius, 1=medium radius, 2=small radius)".format(
            restored_curriculum_stage
        )
    )

    rows = []
    try:
        for command in COMMANDS:
            obs, _ = env.reset()
            command_kappa = (
                command[1] / command[0] if abs(command[0]) > 1.0e-6 else 0.0
            )
            env.commands[:, 0] = command[0]
            env.commands[:, 1] = command_kappa
            reset_history = getattr(env, "reset_observation_history", None)
            if reset_history is not None:
                reset_history()
            env.compute_observations()
            obs = env.get_observations()

            samples = []
            total_steps = 250
            unexpected_resets = 0
            for _ in range(total_steps):
                with torch.no_grad():
                    actions = policy(obs)
                obs, _, _, dones, _ = env.step(actions)
                unexpected_resets += int(dones[0].item())
                samples.append(
                    (
                        env.trajectory_lin_vel[0, 0].item(),
                        env.trajectory_lin_vel[0, 1].item(),
                        env.trajectory_ang_vel[0].item(),
                        env.output_actions[0, 0].item(),
                        env.output_actions[0, 1].item(),
                        env.dof_vel[0, 0].item(),
                        env.dof_pos[0, 1].item(),
                        torch.linalg.vector_norm(env.projected_gravity[0, :2]).item(),
                        float(env.trajectory_ang_vel_valid[0].item()),
                        env.lateral_lean[0].item(),
                        env.lateral_lean_trend[0].item(),
                        env.lateral_lean_wobble[0].item(),
                        env.roll_rate_wobble[0].item(),
                    )
                )

            # Discard the first half as transient and score the settled half.
            data = np.asarray(samples[total_steps // 2 :], dtype=np.float64)
            vx = data[:, 0]
            vy = data[:, 1]
            wz = data[:, 2]
            qdot_target = data[:, 3]
            q_target = data[:, 4]
            qdot_actual = data[:, 5]
            q_actual = data[:, 6]
            tilt_sin = data[:, 7]
            wz_valid = data[:, 8]
            lateral_lean = data[:, 9]
            lateral_lean_trend = data[:, 10]
            lateral_lean_wobble = data[:, 11]
            roll_rate_wobble = data[:, 12]
            vx_error = vx - command[0]
            wz_error = wz - command[1]
            working_limit = float(env.cfg.control.second_pos_limits)
            mechanical_limit = float(env.cfg.control.second_mechanical_pos_limit)
            q_actual_peak_abs = float(np.abs(q_actual).max())
            achieved_kappa = (
                float(wz.mean() / vx.mean()) if abs(vx.mean()) > 0.05 else float("nan")
            )
            achieved_radius_m = (
                float(abs(vx.mean() / wz.mean())) if abs(wz.mean()) > 0.01 else float("nan")
            )
            row = {
                "command_vx": command[0],
                "command_wz": command[1],
                "command_kappa": command_kappa,
                "vx_mean": float(vx.mean()),
                "wz_mean": float(wz.mean()),
                "wz_std": float(wz.std()),
                "wz_peak_abs": float(np.abs(wz).max()),
                "achieved_kappa": achieved_kappa,
                "achieved_radius_m": achieved_radius_m,
                "vx_mae": float(np.abs(vx_error).mean()),
                "wz_mae": float(np.abs(wz_error).mean()),
                "vx_rmse": float(np.sqrt(np.mean(vx_error**2))),
                "wz_rmse": float(np.sqrt(np.mean(wz_error**2))),
                "lateral_speed_mae": float(np.abs(vy).mean()),
                "joint0_vel_target_mean": float(qdot_target.mean()),
                "joint0_vel_target_std": float(qdot_target.std()),
                "joint1_pos_target_mean": float(q_target.mean()),
                "joint1_pos_target_std": float(q_target.std()),
                "joint1_pos_target_peak_abs": float(np.abs(q_target).max()),
                "joint1_pos_target_saturation_fraction": float(
                    np.mean(
                        np.abs(q_target)
                        >= 0.95 * env.cfg.control.second_pos_limits
                    )
                ),
                "joint0_vel_target_mae": float(
                    np.abs(qdot_actual - qdot_target).mean()
                ),
                "joint1_pos_target_mae": float(np.abs(q_actual - q_target).mean()),
                "tilt_sin_max": float(tilt_sin.max()),
                "tilt_angle_max_deg": float(
                    np.degrees(np.arcsin(np.clip(tilt_sin.max(), 0.0, 1.0)))
                ),
                "joint2_target_peak_abs_deg": float(
                    np.degrees(np.abs(q_target).max())
                ),
                "joint2_actual_peak_abs_deg": float(np.degrees(q_actual_peak_abs)),
                "joint2_working_limit_exceed_fraction": float(
                    np.mean(np.abs(q_actual) > working_limit)
                ),
                "joint2_min_mechanical_margin_deg": float(
                    np.degrees(mechanical_limit - q_actual_peak_abs)
                ),
                "lean_mean_deg": float(np.degrees(lateral_lean.mean())),
                "lean_trend_mean_deg": float(
                    np.degrees(lateral_lean_trend.mean())
                ),
                "lean_abs_max_deg": float(
                    np.degrees(np.abs(lateral_lean).max())
                ),
                "wobble_rms_deg": float(
                    np.degrees(np.sqrt(np.mean(lateral_lean_wobble**2)))
                ),
                "wobble_p95_deg": float(
                    np.degrees(np.percentile(np.abs(lateral_lean_wobble), 95.0))
                ),
                "wobble_robust_peak_to_peak_deg": float(
                    np.degrees(
                        np.percentile(lateral_lean_wobble, 95.0)
                        - np.percentile(lateral_lean_wobble, 5.0)
                    )
                ),
                "roll_rate_wobble_rms": float(
                    np.sqrt(np.mean(roll_rate_wobble**2))
                ),
                "wz_valid_fraction": float(wz_valid.mean()),
                "unexpected_resets": unexpected_resets,
            }
            rows.append(row)
            print(
                "cmd(vx={command_vx:+.2f}, wz={command_wz:+.2f}, "
                "k={command_kappa:+.2f}) | "
                "mean(vx={vx_mean:+.3f}, wz={wz_mean:+.3f}, "
                "k={achieved_kappa:+.3f}) | "
                "MAE(vx={vx_mae:.3f}, wz={wz_mae:.3f}) | "
                "|vy|={lateral_speed_mae:.3f} | "
                "target(dq={joint0_vel_target_mean:+.3f}, "
                "q={joint1_pos_target_mean:+.3f}) | "
                "std(wz={wz_std:.3f}, dq={joint0_vel_target_std:.3f}, "
                "q={joint1_pos_target_std:.3f}) | "
                "joint MAE(dq={joint0_vel_target_mae:.3f}, "
                "q={joint1_pos_target_mae:.3f}) | "
                "q_peak={joint1_pos_target_peak_abs:.3f}, "
                "q_sat={joint1_pos_target_saturation_fraction:.2f} | "
                "joint2(target={joint2_target_peak_abs_deg:.1f}deg, "
                "actual={joint2_actual_peak_abs_deg:.1f}deg, "
                "margin={joint2_min_mechanical_margin_deg:.1f}deg) | "
                "lean(mean={lean_mean_deg:+.1f}deg, max={lean_abs_max_deg:.1f}deg) | "
                "wobble(rms={wobble_rms_deg:.2f}deg, "
                "p95={wobble_p95_deg:.2f}deg) | "
                "wz_valid={wz_valid_fraction:.2f} | "
                "resets={unexpected_resets}".format(**row)
            )
    finally:
        env.gym.destroy_sim(env.sim)

    os.makedirs(args.load_run, exist_ok=True)
    checkpoint_name = "latest" if args.checkpoint is None else str(args.checkpoint)
    output_path = os.path.join(
        args.load_run,
        "velocity_test_model_{}.csv".format(checkpoint_name),
    )
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Results saved to: {}".format(os.path.abspath(output_path)))

    moving_rows = [row for row in rows if abs(row["command_vx"]) > 0.0]
    turning_rows = [row for row in rows if abs(row["command_wz"]) > 0.0]
    straight_rows = [
        row
        for row in moving_rows
        if abs(row["command_wz"]) <= 1.0e-9
    ]
    low_curve_rows = [
        row
        for row in turning_rows
        if abs(row["command_kappa"]) < 0.50
    ]
    high_curve_rows = [
        row
        for row in turning_rows
        if abs(row["command_kappa"]) >= 0.50
    ]
    medium_curve_rows = [
        row
        for row in turning_rows
        if 0.50 <= abs(row["command_kappa"]) < 0.75
    ]
    extreme_curve_rows = [
        row
        for row in turning_rows
        if abs(row["command_kappa"]) >= 0.75
    ]
    stop_rows = [
        row
        for row in rows
        if abs(row["command_vx"]) <= 1.0e-9
        and abs(row["command_wz"]) <= 1.0e-9
    ]
    mean_vx_rmse = float(np.mean([row["vx_rmse"] for row in moving_rows]))
    mean_wz_rmse = float(np.mean([row["wz_rmse"] for row in turning_rows]))
    correct_direction = sum(
        row["command_wz"] * row["wz_mean"] > 0.0 for row in turning_rows
    )
    active_turns = sum(
        abs(row["wz_mean"]) >= 0.25 * abs(row["command_wz"])
        for row in turning_rows
    )
    extreme_direction_correct = sum(
        row["command_wz"] * row["wz_mean"] > 0.0
        for row in extreme_curve_rows
    )
    print(
        "Summary: vx_rmse={:.3f}, wz_rmse={:.3f}, direction={}/{}, "
        "active_turn={}/{}".format(
            mean_vx_rmse,
            mean_wz_rmse,
            correct_direction,
            len(turning_rows),
            active_turns,
            len(turning_rows),
        )
    )
    print(
        "Breakdown: straight_vx_rmse={:.3f}, low_curve_wz_rmse={:.3f}, "
        "medium_curve_wz_rmse={:.3f}, high_curve_wz_rmse={:.3f}, "
        "extreme_curve_wz_rmse={:.3f}, mean_wz_std={:.3f}, "
        "mean_dq_target_std={:.3f}, mean_q_sat={:.2f}, "
        "mean_wz_valid={:.2f}".format(
            float(np.mean([row["vx_rmse"] for row in straight_rows])),
            float(np.mean([row["wz_rmse"] for row in low_curve_rows])),
            float(np.mean([row["wz_rmse"] for row in medium_curve_rows])),
            float(np.mean([row["wz_rmse"] for row in high_curve_rows])),
            float(np.mean([row["wz_rmse"] for row in extreme_curve_rows])),
            float(np.mean([row["wz_std"] for row in turning_rows])),
            float(
                np.mean(
                    [row["joint0_vel_target_std"] for row in moving_rows]
                )
            ),
            float(
                np.mean(
                    [
                        row["joint1_pos_target_saturation_fraction"]
                        for row in moving_rows
                    ]
                )
            ),
            float(
                np.mean([row["wz_valid_fraction"] for row in turning_rows])
            ),
        )
    )
    print(
        "Safety: extreme_direction={}/{}, max_q_sat={:.2f}, "
        "max_joint2_actual={:.1f}deg, min_mechanical_margin={:.1f}deg, "
        "max_wobble_rms={:.2f}deg, max_wobble_p95={:.2f}deg, "
        "max_absolute_tilt={:.1f}deg, total_resets={}".format(
            extreme_direction_correct,
            len(extreme_curve_rows),
            max(
                row["joint1_pos_target_saturation_fraction"] for row in rows
            ),
            max(row["joint2_actual_peak_abs_deg"] for row in rows),
            min(row["joint2_min_mechanical_margin_deg"] for row in rows),
            max(row["wobble_rms_deg"] for row in rows),
            max(row["wobble_p95_deg"] for row in rows),
            max(row["tilt_angle_max_deg"] for row in rows),
            sum(row["unexpected_resets"] for row in rows),
        )
    )
    stage_kappa_limit = (0.25, 0.50, float("inf"))[
        min(max(int(restored_curriculum_stage), 0), 2)
    ]
    stage_rows = [
        row for row in rows if abs(row["command_kappa"]) <= stage_kappa_limit + 1.0e-9
    ]

    def passes_stage_tracking(row):
        wz_tolerance = max(0.02, 0.25 * abs(row["command_wz"]))
        return (
            row["vx_mae"] <= 0.03
            and row["wz_mae"] <= wz_tolerance
            and row["lateral_speed_mae"] <= 0.05
            and row["unexpected_resets"] == 0
        )

    stage_pass_count = sum(passes_stage_tracking(row) for row in stage_rows)
    print(
        "Curriculum-stage qualification: stage={}, |k|<={}, passed={}/{} "
        "using vx_mae<=0.03, wz_mae<=max(0.02,0.25|wz_cmd|), |vy|<=0.05".format(
            restored_curriculum_stage,
            stage_kappa_limit,
            stage_pass_count,
            len(stage_rows),
        )
    )


if __name__ == "__main__":
    main()
