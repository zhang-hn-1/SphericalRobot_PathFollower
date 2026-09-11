"""Dependency-light regression checks for the clean Rotunbot task logic."""

import ast
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLEAN_ENV = ROOT / "legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel_clean.py"
CLEAN_CFG = ROOT / "legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel_clean_config.py"
IDENTIFY = ROOT / "legged_gym/scripts/identify_rotunbot_curvature.py"
MECHANICAL_LIMIT = ROOT / "legged_gym/scripts/identify_rotunbot_mechanical_limit.py"
SUSTAINED_CIRCLES = ROOT / "legged_gym/scripts/test_rotunbot_sustained_circles.py"
EVALUATE = ROOT / "legged_gym/scripts/test_rotunbot_vel_clean.py"
DWL_RUNNER = ROOT / "legged_gym/dwl/on_policy_runner_dwl.py"


def class_body_by_name(tree, name):
    return next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def literal_assignments(nodes):
    values = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


class RotunbotVelCleanLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_source = CLEAN_ENV.read_text(encoding="utf-8")
        cls.cfg_source = CLEAN_CFG.read_text(encoding="utf-8")
        cls.id_source = IDENTIFY.read_text(encoding="utf-8")
        cls.mechanical_source = MECHANICAL_LIMIT.read_text(encoding="utf-8")
        cls.sustained_source = SUSTAINED_CIRCLES.read_text(encoding="utf-8")
        cls.evaluate_source = EVALUATE.read_text(encoding="utf-8")
        cls.runner_source = DWL_RUNNER.read_text(encoding="utf-8")
        cls.env_tree = ast.parse(cls.env_source)
        cls.cfg_tree = ast.parse(cls.cfg_source)
        cls.id_tree = ast.parse(cls.id_source)

    def test_observation_dimensions_match_concatenation(self):
        env_values = literal_assignments(class_body_by_name(self.cfg_tree, "env"))
        self.assertEqual(env_values["num_single_obs"], 2 + 3 + 3 + 3 + 1 + 2 + 2)
        self.assertEqual(
            env_values["single_num_privileged_obs"],
            2 + 3 + 3 + 3 + 2 + 2 + 2 + 1 + 1 + 1,
        )
        self.assertIn("num_observations = frame_stack * num_single_obs", self.cfg_source)
        self.assertIn(
            "num_privileged_obs = c_frame_stack * single_num_privileged_obs",
            self.cfg_source,
        )

    def test_temporal_policy_matches_history_observation(self):
        env_values = literal_assignments(class_body_by_name(self.cfg_tree, "env"))
        self.assertGreaterEqual(env_values["frame_stack"], 10)
        self.assertGreaterEqual(env_values["short_frame_stack"], 2)
        self.assertIn('runner_class_name = "DWLOnPolicyRunner"', self.cfg_source)
        self.assertIn("self.obs_history.append(obs)", self.env_source)
        self.assertIn("self.reset_observation_history(env_ids)", self.env_source)

    def test_zero_motion_is_not_a_high_reward_at_max_speed(self):
        reward_values = literal_assignments(class_body_by_name(self.cfg_tree, "rewards"))
        sigma = reward_values["tracking_sigma_lin"]
        reward_ratio = math.exp(-(0.25**2) / sigma)
        self.assertLess(reward_ratio, 0.05)

    def test_stop_probability_is_explicit_and_small(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        for name in (
            "trajectory_mode_probabilities_large",
            "trajectory_mode_probabilities_medium",
            "trajectory_mode_probabilities_small",
        ):
            probabilities = command_values[name]
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertLessEqual(probabilities[0], 0.10)

    def test_low_curvature_commands_are_included(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        self.assertLessEqual(command_values["trajectory_min_abs_curvature"], 0.25)
        self.assertGreater(command_values["trajectory_min_abs_curvature"], 0.0)
        self.assertGreaterEqual(command_values["trajectory_curve_min_speed"], 0.15)
        self.assertEqual(
            command_values["trajectory_curvature_anchor_values_large"], [0.25]
        )
        self.assertGreaterEqual(
            command_values["trajectory_mode_probabilities_large"][2], 0.50
        )

    def test_v11_stages_large_medium_and_small_radius_anchors(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        self.assertEqual(command_values["trajectory_speed_anchor"], 0.20)
        self.assertGreaterEqual(
            command_values["trajectory_speed_anchor_probability"], 0.50
        )
        self.assertEqual(
            command_values["trajectory_curvature_anchor_values_large"], [0.25]
        )
        self.assertEqual(
            command_values["trajectory_curvature_anchor_values_medium"],
            [0.25, 0.50],
        )
        self.assertEqual(
            command_values["trajectory_curvature_anchor_values_small"],
            [0.25, 0.50, 0.75, 1.00],
        )
        self.assertGreaterEqual(
            command_values["trajectory_curvature_anchor_probability"], 0.50
        )
        self.assertIn("use_curvature_anchor", self.env_source)
        self.assertIn("use_speed_anchor", self.env_source)

    def test_v11_curriculum_is_performance_gated_not_time_gated(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        self.assertTrue(command_values["tracking_curriculum_enabled"])
        self.assertGreater(command_values["tracking_curriculum_min_updates"], 0)
        self.assertGreater(command_values["tracking_curriculum_pass_rate"], 0.5)
        self.assertIn("def _update_tracking_curriculum", self.env_source)
        self.assertIn("self.tracking_curriculum_ema", self.env_source)

    def test_curriculum_status_is_logged_every_ppo_iteration(self):
        self.assertIn("def get_tracking_curriculum_status", self.env_source)
        self.assertIn("Tracking curriculum stage:", self.runner_source)
        self.assertIn("Tracking pass rate (batch/EMA):", self.runner_source)
        self.assertIn("Curriculum/ema_pass_rate", self.runner_source)
        self.assertIn("Curriculum/batch_pass_rate", self.runner_source)
        self.assertIn("completed_iterations", self.runner_source)

    def test_final_curriculum_stage_continues_measuring_success(self):
        curriculum_method = self.env_source.split(
            "def _update_tracking_curriculum", 1
        )[1].split("def _update_wobble_quantities", 1)[0]
        self.assertNotIn("tracking_curriculum_stage >= 2", curriculum_method)
        self.assertIn("self.tracking_curriculum_stage < 2", curriculum_method)

    def test_turn_reward_is_separate_and_sharp(self):
        reward_values = literal_assignments(class_body_by_name(self.cfg_tree, "rewards"))
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        self.assertLessEqual(reward_values["trajectory_lateral_weight"], 0.10)
        self.assertGreaterEqual(scale_values["tracking_ang_vel"], 4.0)
        self.assertGreater(scale_values["tracking_lin_vel"], 0.0)
        self.assertIn("relative_turn_reward", self.env_source)
        self.assertLessEqual(reward_values["wrong_turn_reward_floor"], -0.50)
        self.assertIn("self.cfg.rewards.wrong_turn_reward_floor", self.env_source)
        self.assertIn("relative_turn_reward = relative_turn_reward * speed_progress", self.env_source)

    def test_straight_turn_penalty_cannot_outweigh_perfect_speed_tracking(self):
        reward_values = literal_assignments(class_body_by_name(self.cfg_tree, "rewards"))
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        max_straight_turn_penalty = (
            reward_values["straight_turn_penalty_weight"]
            * scale_values["tracking_ang_vel"]
        )
        self.assertLess(max_straight_turn_penalty, scale_values["tracking_lin_vel"])

    def test_no_turn_has_zero_curved_command_reward(self):
        desired_wz = 0.20
        measured_wz = 0.0
        relative_reward = 1.0 - abs(desired_wz - measured_wz) / abs(desired_wz)
        self.assertEqual(relative_reward, 0.0)

    def test_wrong_direction_turn_has_bounded_negative_reward(self):
        reward_values = literal_assignments(class_body_by_name(self.cfg_tree, "rewards"))
        desired_wz = 0.20
        measured_wz = -0.20
        raw_reward = 1.0 - abs(desired_wz - measured_wz) / abs(desired_wz)
        bounded_reward = max(
            reward_values["wrong_turn_reward_floor"], min(raw_reward, 1.0)
        )
        self.assertLess(bounded_reward, 0.0)
        self.assertGreaterEqual(
            bounded_reward, reward_values["wrong_turn_reward_floor"]
        )

    def test_stationary_policy_has_zero_tracking_reward(self):
        command_speed = 0.20
        measured_speed = 0.0
        linear_reward = max(
            0.0,
            1.0 - abs(command_speed - measured_speed) / abs(command_speed),
        )
        desired_wz = 0.10
        measured_wz = 0.0
        angular_reward = max(
            0.0,
            1.0 - abs(desired_wz - measured_wz) / abs(desired_wz),
        )
        self.assertEqual(linear_reward, 0.0)
        self.assertEqual(angular_reward, 0.0)
        self.assertIn("straight_penalty", self.env_source)

    def test_command_curriculum_is_valid_or_explicitly_disabled(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        self.assertTrue(command_values["tracking_curriculum_enabled"])
        self.assertIn("self.tracking_curriculum_stage < 2", self.env_source)
        self.assertIn("self.tracking_curriculum_stage += 1", self.env_source)
        self.assertIn("def get_checkpoint_state", self.env_source)
        self.assertIn("def set_checkpoint_state", self.env_source)

    def test_command_channels_are_speed_and_curvature(self):
        self.assertIn("mode == 2, curvature", self.env_source)
        self.assertIn("self.commands[:, 0] * self.commands[:, 1]", self.env_source)

    def test_commands_are_scaled_in_actor_and_critic_observations(self):
        self.assertGreaterEqual(self.env_source.count("self._scaled_commands()"), 2)
        self.assertIn("self.obs_scales.command_speed", self.env_source)
        self.assertIn("self.obs_scales.command_yaw_rate", self.env_source)
        self.assertIn("self._desired_trajectory_ang_vel()", self.env_source)

    def test_path_turning_rate_uses_two_center_displacement_windows(self):
        command_values = literal_assignments(class_body_by_name(self.cfg_tree, "commands"))
        self.assertGreaterEqual(command_values["trajectory_heading_window_s"], 0.25)
        self.assertLessEqual(command_values["trajectory_heading_window_s"], 0.50)
        self.assertIn("recent_displacement = current_center_xy - previous_center_xy", self.env_source)
        self.assertIn("previous_displacement = previous_center_xy - older_center_xy", self.env_source)
        self.assertIn("heading_delta / window_duration", self.env_source)
        self.assertNotIn("heading_delta / self.dt", self.env_source)

    def test_path_window_resets_when_command_changes(self):
        self.assertIn("def _reset_trajectory_window", self.env_source)
        self.assertIn("self._reset_trajectory_window(env_ids)", self.env_source)
        self.assertIn("self._trajectory_window_age[env_ids] = 0", self.env_source)

    def test_clean_reset_zeros_root_velocity(self):
        self.assertIn("self.root_states[env_ids, 7:13] = 0.0", self.env_source)

    def test_substep_derivative_uses_physics_dt(self):
        self.assertIn("self._last_substep_dof_vel", self.env_source)
        self.assertIn(") / self.sim_params.dt", self.env_source)

    def test_v6_finetune_suppresses_untrackable_target_oscillation(self):
        control_values = literal_assignments(class_body_by_name(self.cfg_tree, "control"))
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        self.assertTrue(control_values["set_a_rate_limit"])
        self.assertLessEqual(control_values["rate_limit_1"], 0.08)
        self.assertLessEqual(control_values["rate_limit_2"], 0.02)
        self.assertLessEqual(scale_values["action_rate"], -0.005)
        self.assertLess(scale_values["joint_target_error"], 0.0)
        self.assertIn("def _reward_joint_target_error", self.env_source)

    def test_v6_finetune_penalizes_unstable_stop(self):
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        self.assertLess(scale_values["stop_stability"], 0.0)
        self.assertIn("def _reward_stop_stability", self.env_source)
        self.assertIn("stop_command.float()", self.env_source)

    def test_v10_reserves_five_degrees_beyond_joint2_working_limit(self):
        control_values = literal_assignments(class_body_by_name(self.cfg_tree, "control"))
        working_limit = control_values["second_pos_limits"]
        mechanical_limit = control_values["second_mechanical_pos_limit"]
        self.assertAlmostEqual(working_limit, math.radians(25.0), places=6)
        self.assertAlmostEqual(
            control_values["second_actionScale"], working_limit, places=6
        )
        self.assertAlmostEqual(mechanical_limit, math.radians(30.0), places=4)
        self.assertAlmostEqual(
            mechanical_limit - working_limit, math.radians(5.0), places=4
        )
        self.assertIn("actions[:, 1] * self.cfg.control.second_actionScale", self.env_source)
        self.assertIn("-self.cfg.control.second_pos_limits", self.env_source)

    def test_v10_penalizes_only_actual_joint2_reserved_margin_incursion(self):
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        self.assertLess(scale_values["second_joint_limit_margin"], 0.0)
        self.assertEqual(scale_values["second_joint_target_margin"], 0.0)
        self.assertIn("def _reward_second_joint_limit_margin", self.env_source)
        self.assertIn("mechanical_limit - working_limit", self.env_source)
        self.assertIn("torch.abs(self.dof_pos[:, 1]) - working_limit", self.env_source)

    def test_v13_measures_and_lightly_penalizes_high_frequency_wobble(self):
        reward_values = literal_assignments(class_body_by_name(self.cfg_tree, "rewards"))
        scale_values = literal_assignments(class_body_by_name(self.cfg_tree, "scales"))
        self.assertGreater(reward_values["wobble_filter_time_constant"], 0.0)
        self.assertGreater(reward_values["wobble_angle_tolerance"], 0.0)
        self.assertEqual(scale_values["tilt_margin"], 0.0)
        self.assertEqual(scale_values["ang_vel_xy"], 0.0)
        self.assertLess(scale_values["wobble"], 0.0)
        self.assertIn("def _update_wobble_quantities", self.env_source)
        self.assertIn("self.lateral_lean - self.lateral_lean_trend", self.env_source)
        self.assertIn("def _reward_wobble", self.env_source)
        self.assertIn("self.lateral_lean_wobble", self.env_source)

    def test_v9_resume_uses_fresh_optimizer_at_lower_learning_rate(self):
        algorithm_values = literal_assignments(class_body_by_name(self.cfg_tree, "algorithm"))
        runner_values = literal_assignments(class_body_by_name(self.cfg_tree, "runner"))
        self.assertLessEqual(algorithm_values["learning_rate"], 1.0e-4)
        self.assertFalse(runner_values["load_optimizer"])
        self.assertTrue(runner_values["load_optimizer_when_env_state"])
        self.assertIn("self.cfg.get('load_optimizer', True)", self.runner_source)
        self.assertIn("loaded_dict.get('env_state') is not None", self.runner_source)

    def test_evaluation_groups_curvature_by_csv_field_name(self):
        self.assertIn('row["command_kappa"]', self.evaluate_source)
        self.assertNotIn('row["command_curvature"]', self.evaluate_source)
        self.assertIn("medium_curve_rows", self.evaluate_source)
        self.assertIn("extreme_curve_rows", self.evaluate_source)
        self.assertIn("extreme_direction_correct", self.evaluate_source)

    def test_evaluation_reports_joint_margin_and_wobble_separately(self):
        self.assertIn('"joint2_min_mechanical_margin_deg"', self.evaluate_source)
        self.assertIn('"lean_mean_deg"', self.evaluate_source)
        self.assertIn('"wobble_rms_deg"', self.evaluate_source)
        self.assertIn('"wobble_p95_deg"', self.evaluate_source)
        self.assertIn('"tilt_angle_max_deg"', self.evaluate_source)
        self.assertIn("Restored tracking curriculum stage", self.evaluate_source)
        self.assertIn("Curriculum-stage qualification", self.evaluate_source)

    def test_envelope_monotonic_assumption_is_opt_in(self):
        self.assertIn('"--enforce_monotonic_envelope"', self.id_source)
        self.assertIn("if args.enforce_monotonic_envelope:", self.id_source)
        self.assertIn('candidate = row["kappa_max_raw_1_per_m"]', self.id_source)

    def test_mechanical_limit_search_bypasses_the_learned_policy(self):
        self.assertIn("env.step(actions)", self.mechanical_source)
        self.assertIn("env_cfg.seed = int(args.seed)", self.mechanical_source)
        self.assertNotIn("get_inference_policy", self.mechanical_source)
        self.assertNotIn("make_alg_runner", self.mechanical_source)

    def test_mechanical_limit_search_includes_coordinated_periodic_actions(self):
        self.assertIn("def candidate_actions", self.mechanical_source)
        self.assertIn('candidates["drive_amplitude"]', self.mechanical_source)
        self.assertIn('candidates["steer_amplitude"]', self.mechanical_source)
        self.assertIn('candidates["phase_rad"]', self.mechanical_source)

    def test_mechanical_limit_requires_a_sustained_circle(self):
        self.assertIn("def fit_circles", self.mechanical_source)
        self.assertIn("min_heading_change_deg", self.mechanical_source)
        self.assertIn("turn_sign_consistency", self.mechanical_source)
        self.assertIn("circle_residual_ratio", self.mechanical_source)
        self.assertIn("stable_tilt_10deg", self.mechanical_source)
        self.assertIn("not proof of mechanical impossibility", self.mechanical_source)

    def test_mechanical_limit_reproduces_preloaded_full_steer_experiment(self):
        self.assertIn("--preload_seconds", self.mechanical_source)
        self.assertIn("--steer_position_limit_rad", self.mechanical_source)
        self.assertIn(
            "env_cfg.control.second_pos_limits = float(args.steer_position_limit_rad)",
            self.mechanical_source,
        )
        self.assertIn("preload_actions[:, 1]", self.mechanical_source)
        self.assertIn("stable_joint_margin", self.mechanical_source)
        self.assertIn("print_filter_diagnostics", self.mechanical_source)

    def test_sustained_policy_test_is_a_long_circle_fit_positive_control(self):
        self.assertIn("MEASURE_SECONDS = 20.0", self.sustained_source)
        self.assertIn("def fit_circle", self.sustained_source)
        self.assertIn("heading_change_deg", self.sustained_source)
        self.assertIn("radius_agreement_ratio", self.sustained_source)
        self.assertIn("stable_below_3m", self.sustained_source)
        self.assertIn("MECHANICAL HYPOTHESIS DISPROVED", self.sustained_source)


if __name__ == "__main__":
    unittest.main()
