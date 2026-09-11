"""Configuration for geometric path following with the Rotunbot."""

import os

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel_clean_config import (
    RotunbotVelCleanCfg,
    RotunbotVelCleanCfgPPO,
)


class RotunbotPathCfg(RotunbotVelCleanCfg):
    class env(RotunbotVelCleanCfg.env):
        num_envs = 2048
        num_actions = 2

        # The actor receives 20 frames of 18-D yaw-invariant tracking state plus one current
        # 10 x 4 path preview.  The preview is intentionally not duplicated in
        # every history frame.
        num_single_obs = 19
        frame_stack = 20
        short_frame_stack = 5
        num_path_points = 10
        path_point_dim = 4
        # V5 optionally appends explicit lookahead curvature.
        include_curvature_observation = os.environ.get("PATH_V5_CURVATURE_OBS", "0") == "1"
        num_path_obs = num_path_points * path_point_dim + 3 + int(include_curvature_observation)
        num_observations = frame_stack * num_single_obs + num_path_obs

        c_frame_stack = 1
        single_num_privileged_obs = 28
        num_privileged_obs = single_num_privileged_obs
        episode_length_s = 40.0
        env_spacing = 3.0

    class sim(RotunbotVelCleanCfg.sim):
        dt = 0.02

    class control(RotunbotVelCleanCfg.control):
        # 50 Hz policy.  Actions remain joint-1 velocity target and joint-2
        # position target, followed by the existing servo model.
        decimation = 1
        first_actionScale = 3.0
        first_vel_limits = 3.0

        # Use the Rotunbot_test2 URDF mechanical limit.  A soft reward keeps
        # normal motion away from the hard stop without imposing the old 0.45
        # rad command clip.
        second_actionScale = 0.5236
        second_pos_limits = 0.5236
        second_mechanical_pos_limit = 0.5236

        set_a_rate_limit = True
        rate_limit_1 = 0.08
        rate_limit_2 = 0.02
        target_filter_alpha_1 = 0.50
        target_filter_alpha_2 = 0.40
        use_path_action_prior = os.environ.get("PATH_USE_ACTION_PRIOR", "0") == "1"

    class path:
        sample_spacing = 0.05
        num_samples = 201
        preview_distances = [
            0.20, 0.40, 0.60, 0.80, 1.00,
            1.20, 1.40, 1.60, 1.80, 2.00,
        ]
        preview_position_scale = 0.5
        projection_back_samples = 4
        projection_forward_samples = 30

        # Measured-action residual prior. PPO still selects both joint
        # commands, but starts from a reproducible feasible controller.
        include_curvature_observation = os.environ.get("PATH_V5_CURVATURE_OBS", "0") == "1"
        prior_drive = 0.65
        prior_stop_distance = 1.60
        prior_speed_kp = 0.65
        prior_cross_track_kp = 0.60
        prior_heading_kp = 1.50
        prior_curvature_gain = 0.69
        prior_residual_scale_1 = 0.35
        prior_residual_scale_2 = 0.35

        # V2 curriculum adds one difficulty at a time.  Constant arcs are
        # capped at 120 degrees so a local path never becomes a near-loop.
        length_range_stage0 = [2.0, 4.0]
        length_range_stage1 = [2.5, 5.0]
        length_range_stage2 = [3.0, 6.0]
        length_range_stage3 = [3.0, 7.0]
        length_range_stage4 = [3.0, 7.0]
        length_range_stage5 = [3.0, 8.0]
        length_range_stage6 = [3.0, 8.0]
        maximum_constant_arc_angle = 2.0943951024

        straight_probability_stage0 = 1.00
        straight_probability_stage1 = 0.40
        straight_probability_stage2 = 0.30
        straight_probability_stage3 = 0.25
        straight_probability_stage4 = 0.25
        straight_probability_stage5 = 0.25
        straight_probability_stage6 = 0.20

        curvature_values_stage1 = [0.25]
        curvature_values_stage2 = [0.20, 0.25, 0.3333333333]
        curvature_values_stage3 = [0.20, 0.25, 0.3333333333, 0.40]
        curvature_values_stage4 = [0.20, 0.25, 0.3333333333, 0.40, 0.50]
        curvature_values_stage5 = [0.20, 0.25, 0.3333333333, 0.40, 0.50]
        curvature_values_stage6 = [0.20, 0.25, 0.3333333333, 0.40, 0.50]

        s_curve_probability_stage0 = 0.00
        s_curve_probability_stage1 = 0.00
        s_curve_probability_stage2 = 0.00
        s_curve_probability_stage3 = 0.00
        s_curve_probability_stage4 = 0.00
        s_curve_probability_stage5 = 0.20
        s_curve_probability_stage6 = 0.35
        s_curvature_values_stage5 = [0.20, 0.25, 0.3333333333]
        s_curvature_values_stage6 = [0.20, 0.25, 0.3333333333, 0.40, 0.50]

        initial_offset_stage = 6
        initial_lateral_offset = 0.10
        initial_heading_offset = 0.0872664626

        success_endpoint_distance = 0.20
        success_remaining_length = 0.20
        success_speed = 0.10
        deviation_termination = 1.50
        unstable_gravity_z = -0.30

        curriculum_enabled = not bool(os.environ.get("PATH_TRAIN_TYPE", ""))
        curriculum_window = 8192
        curriculum_min_type_attempts = 512
        curriculum_success_rate = 0.70
        curriculum_max_stage = 6

    class commands(RotunbotVelCleanCfg.commands):
        num_commands = 2
        curriculum = False
        resampling_time = 1000.0

    class asset(RotunbotVelCleanCfg.asset):
        # Keep global angular damping at zero so rolling is not damped.  Model
        # only the missing contact-patch torsional resistance around world z.
        angular_damping = 0.0
        contact_yaw_damping = True
        contact_yaw_damping_body = "link1"
        contact_yaw_damping_viscous = 2.0
        contact_yaw_damping_coulomb = 0.5
        contact_yaw_damping_transition = 0.02
        contact_yaw_damping_max_torque = 2.0
        contact_yaw_damping_force_threshold = 10.0
        contact_yaw_damping_speed_scale = 0.10
        contact_yaw_damping_speed_exponent = 4.0

    class init_state(RotunbotVelCleanCfg.init_state):
        randomize_initial_velocity = False

    class domain_rand(RotunbotVelCleanCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_link_mass = False
        randomize_com = False
        randomize_link_com = False
        randomize_base_inertia = False
        randomize_link_inertia = False
        randomize_motor_strength = False
        randomize_motor_offset = False
        push_robots = False
        add_dof_lag = False
        add_imu_lag = False

    class noise(RotunbotVelCleanCfg.noise):
        add_noise = False
        noise_level = 0.2

    class normalization(RotunbotVelCleanCfg.normalization):
        clip_observations = 100.0
        clip_actions = 1.0

        class obs_scales(RotunbotVelCleanCfg.normalization.obs_scales):
            gravity = 1.0
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 2.0
            dof_vel = 0.5

    class rewards(RotunbotVelCleanCfg.rewards):
        only_positive_rewards = False
        tracking_sigma_cross = 0.09
        tracking_sigma_heading = 0.25
        near_end_distance = 0.50
        soft_dof_pos_limit = 0.90
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        class scales:
            termination = -10.0
            path_progress = 8.0
            path_tracking = -5.0
            path_heading = -1.00
            completion = 20.0
            near_end_speed = -2.0
            reverse_motion = -2.0 if os.environ.get("PATH_USE_ACTION_PRIOR", "0") == "1" else 0.0
            path_deviation = -0.25
            action_rate = -0.003
            target_rate = -0.03
            steering_direction = 0.0 if os.environ.get("PATH_USE_ACTION_PRIOR", "0") == "1" else -0.50
            steering_feedforward = 0.0 if os.environ.get("PATH_USE_ACTION_PRIOR", "0") == "1" else -0.30
            steering_centering = -0.20
            steering_saturation = -0.10
            torques = -1.0e-6
            dof_pos_limits = -0.10
            time = -0.05


class RotunbotPathCfgPPO(RotunbotVelCleanCfgPPO):
    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy(RotunbotVelCleanCfgPPO.policy):
        init_noise_std = float(os.environ.get("PATH_INIT_STD", "0.35"))
        min_noise_std = float(os.environ.get("PATH_MIN_STD", "0.05"))
        max_noise_std = float(os.environ.get("PATH_MAX_STD", "0.50"))
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        kernel_size = [3, 2]
        filter_size = [32, 16]
        stride_size = [1, 1]
        lh_output_dim = 32
        history_frames = RotunbotPathCfg.env.frame_stack
        path_obs_dim = RotunbotPathCfg.env.num_path_obs
        turn_gate_threshold = 0.04
        freeze_history_encoder = os.environ.get("PATH_FREEZE_HISTORY", "1") == "1"
        trainable_experts = os.environ.get("PATH_TRAIN_EXPERT", "all")
        zero_init_actor_output = os.environ.get("PATH_ZERO_INIT_ACTOR", "0") == "1"

    class algorithm(RotunbotVelCleanCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = float(os.environ.get("PATH_PPO_CLIP", "0.1"))
        entropy_coef = float(os.environ.get("PATH_ENTROPY", "2.0e-4"))
        num_learning_epochs = int(os.environ.get("PATH_PPO_EPOCHS", "2"))
        num_mini_batches = 8
        learning_rate = float(os.environ.get("PATH_PPO_LR", "2.0e-5"))
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.0

    class runner(RotunbotVelCleanCfgPPO.runner):
        policy_class_name = "ActorCriticPathDWL"
        algorithm_class_name = "PPODWL"
        experiment_name = "rotunbot_path"
        run_name = os.environ.get("PATH_RUN_NAME", "geometric_path_v4c_specialist")
        resume = False
        load_run = -1
        checkpoint = -1
        num_steps_per_env = 96
        max_iterations = 5000
        save_interval = 25
        load_optimizer = os.environ.get("PATH_LOAD_OPTIMIZER", "1") == "1"
        load_optimizer_when_env_state = False



