from .rotunbot_vel_config import RotunbotVelCfg, RotunbotVelCfgPPO


class RotunbotVelCleanCfg(RotunbotVelCfg):
    """Clean command-conditioned velocity-tracking baseline."""

    class env(RotunbotVelCfg.env):
        num_envs = 2048
        # command(2) + gravity(3) + base lin vel(3) + base ang vel(3)
        # + joint2 position(1) + joint1/joint2 velocity(2) + actions(2)
        num_single_obs = 16
        frame_stack = 20
        short_frame_stack = 5
        c_frame_stack = 3
        single_num_privileged_obs = 20
        num_observations = frame_stack * num_single_obs
        num_privileged_obs = c_frame_stack * single_num_privileged_obs
        episode_length_s = 20

    class sim(RotunbotVelCfg.sim):
        dt = 0.02

    class control(RotunbotVelCfg.control):
        control_type = "R"
        action_scale = 40.0
        # Match the successful point-to-point controller's 50 Hz policy rate.
        decimation = 1
        # A normalized policy action of +/-1 corresponds to the full physical
        # target range below.
        first_actionScale = 3.0
        # joint2 has a physical URDF limit of +/-30 deg.  Keep policy targets
        # inside +/-25 deg so a disturbance has 5 deg of mechanical travel
        # available before the hard stop.
        second_actionScale = 0.4363323129985824  # 25 deg [rad]
        first_vel_limits = 3.0
        second_pos_limits = 0.4363323129985824  # 25 deg working limit [rad]
        second_mechanical_pos_limit = 0.5236  # 30 deg URDF limit [rad]
        # The v5 policy found a high-frequency target oscillation that produced
        # the correct mean yaw rate but could not be followed by the joints.
        # Limit the physical target change at each 50 Hz policy step.
        set_a_rate_limit = True
        rate_limit_1 = 0.08
        rate_limit_2 = 0.02
        # First-order filtering removes one-step target sign changes that can
        # excite the internal-mass/ball mode. The hard rate limits below still
        # bound the physical target acceleration.
        target_filter_alpha_1 = 0.50
        target_filter_alpha_2 = 0.40

    class commands(RotunbotVelCfg.commands):
        num_commands = 2
        # Give the history encoder several seconds under each fixed command.
        resampling_time = 4.0
        manual_command_mode = False
        # Generate commands from signed trajectory speed v and curvature
        # kappa [1/m], then set yaw rate to wz = v * kappa.
        trajectory_speed_range = [-0.25, 0.25]
        trajectory_curvature_range = [-1.0, 1.0]
        # Performance-gated curriculum. Stage 0 first learns stop and straight
        # motion. Stage 1 adds the large 4 m radius, and stage 2 progressively
        # exposes 2 m, 1.33 m and 1 m radii.
        tracking_curriculum_enabled = True
        tracking_curriculum_min_updates = 6400
        tracking_curriculum_check_interval = 100
        tracking_curriculum_ema_alpha = 0.995
        tracking_curriculum_pass_rate = 0.80
        tracking_curriculum_v_tolerance = 0.03
        tracking_curriculum_w_abs_tolerance = 0.02
        tracking_curriculum_w_rel_tolerance = 0.25
        tracking_curriculum_lateral_tolerance = 0.05
        # Stage 0 is motion-dominant: one stop sample for nine moving samples.
        # Repeating the entries is intentional because the command table is
        # sampled uniformly. The old 1/3 stop ratio let the policy collapse to
        # the nearly-stationary solution seen in the V14 evaluation.
        trajectory_command_table_large = (
            (0.00, 0.00),
            (+0.20, 0.00),
            (+0.20, 0.00),
            (+0.20, 0.00),
            (+0.20, 0.00),
            (+0.20, 0.00),
            (-0.20, 0.00),
            (-0.20, 0.00),
            (-0.20, 0.00),
            (-0.20, 0.00),
        )
        # V15 used all four forward/reverse turn combinations as soon as
        # stage 1 opened. The evaluation showed that the policy collapsed to
        # one +25 deg steering target and could not separate turn direction.
        # First teach the two forward turn directions; reverse turns are added
        # together with the tighter radii in stage 2.
        trajectory_command_table_medium = trajectory_command_table_large + (
            (+0.20, +0.25),
            (+0.20, +0.25),
            (+0.20, -0.25),
            (+0.20, -0.25),
            (+0.20, +0.25),
            (+0.20, -0.25),
        )
        trajectory_command_table_small = trajectory_command_table_medium + (
            (-0.20, -0.25),
            (-0.20, +0.25),
            (+0.20, +0.50),
            (+0.20, -0.50),
            (-0.20, -0.50),
            (-0.20, +0.50),
            (+0.20, +0.75),
            (+0.20, -0.75),
            (-0.20, -0.75),
            (-0.20, +0.75),
            (+0.20, +1.00),
            (+0.20, -1.00),
            (-0.20, -1.00),
            (-0.20, +1.00),
        )
        trajectory_command_table_probability_large = 1.00
        trajectory_command_table_probability_medium = 1.00
        trajectory_command_table_probability_small = 1.00
        trajectory_curvature_anchor_values_large = [0.25]
        trajectory_curvature_anchor_values_medium = [0.25, 0.50]
        trajectory_curvature_anchor_values_small = [0.25, 0.50, 0.75, 1.00]
        trajectory_curvature_anchor_probability = 0.90
        trajectory_speed_anchor = 0.20
        trajectory_speed_anchor_probability = 0.85
        # Include the low-curvature region that was absent from v5 training.
        trajectory_min_abs_curvature = 0.20
        trajectory_curve_min_speed = 0.15
        # R = 1 / |kappa|. At v=0.20 m/s, |kappa|=1.0 requests
        # R=1 m and wz=v*kappa=0.20 rad/s.
        trajectory_target_radius_min = 1.0
        trajectory_max_yaw_rate = 0.25
        trajectory_min_speed = 0.05
        # Estimate path turning from two adjacent ball-center displacement
        # windows. This rejects a one-off lateral lean and high-frequency
        # left/right jitter without introducing a position trajectory command.
        trajectory_heading_window_s = 0.40
        # At the beginning of training v is small.  The 0.02 m threshold kept
        # path-yaw invalid too long, while this 0.4 s window still rejects
        # one-step lateral wobble.
        trajectory_min_window_distance = 0.01
        trajectory_max_measured_yaw_rate = 2.0
        trajectory_mode_probabilities_large = [0.10, 0.35, 0.55]
        trajectory_mode_probabilities_medium = [0.08, 0.27, 0.65]
        trajectory_mode_probabilities_small = [0.05, 0.20, 0.75]

        class ranges(RotunbotVelCfg.commands.ranges):
            # Start with a smaller command envelope for the stable baseline.
            # Expand to the original range only after this run tracks reliably.
            lin_vel_x = [-0.25, 0.25]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.25, 0.25]

    class domain_rand(RotunbotVelCfg.domain_rand):
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

    class noise(RotunbotVelCfg.noise):
        # First run is a deterministic controller calibration baseline.
        # Turn this on only after the fixed-command test passes.
        add_noise = False
        noise_level = 0.2

    class normalization(RotunbotVelCfg.normalization):
        class obs_scales(RotunbotVelCfg.normalization.obs_scales):
            # The policy sees the quantities it must track directly: [v, wz].
            # Feasibility is still enforced by sampling kappa and setting
            # wz=v*kappa, so a stop command cannot request in-place yaw.
            command_speed = 4.0
            command_yaw_rate = 4.0
            gravity = 1.0
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 2.0
            dof_vel = 0.5

    class rewards(RotunbotVelCfg.rewards):
        only_positive_rewards = False
        # Separate, sharper error scales keep the turn learning signal alive.
        # Rotunbot may need lateral motion while redirecting its trajectory.
        tracking_sigma_lin = 0.01
        tracking_sigma_ang = 0.005
        linear_command_threshold = 0.02
        turn_command_threshold = 0.015
        stop_speed_tolerance = 0.05
        lateral_speed_tolerance = 0.10
        straight_yaw_rate_tolerance = 0.05
        # A straight-path wobble must not outweigh perfect forward tracking.
        straight_turn_penalty_weight = 0.10
        # Make a confidently wrong turn worse than producing no turn. This is
        # needed for the four signed (v, kappa) combinations at |kappa|=1.
        wrong_turn_reward_floor = -1.00
        trajectory_lateral_weight = 0.10
        # Separate a slowly varying steering lean from left/right wobble.  The
        # time constant is deliberately longer than one 50 Hz action step but
        # shorter than a sustained turn.  The tolerances normalize the reward;
        # they are not hard attitude limits.
        wobble_filter_time_constant = 0.60
        wobble_angle_tolerance = 0.08726646259971647  # 5 deg [rad]
        wobble_roll_rate_tolerance = 0.3490658503988659  # 20 deg/s [rad/s]
        wobble_roll_rate_weight = 0.25
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        class scales(RotunbotVelCfg.rewards.scales):
            # Separate terms prevent one poor channel from erasing the other
            # channel's gradient. Keep a small coupled term as a tie-breaker.
            tracking_lin_vel = 1.5
            tracking_ang_vel = 6.0
            tracking_trajectory = 0.0
            # v5 already discovered turning. During fine-tuning, suppress its
            # untrackable near-full-range target oscillation and unstable stop.
            action_rate = -0.005
            # Penalize the filtered physical target rate as well as the policy
            # action rate, so the policy cannot hide chatter behind the limiter.
            target_rate = -0.05
            joint_target_error = -0.01
            stop_stability = -1.50
            # V15 held the second joint at +25 deg even for stop and straight
            # commands. Center it whenever no curvature is requested, while
            # leaving a curved command free to use the lean needed to turn.
            steering_target_centering = -0.60
            steering_target_direction = -0.20
            # Zero below the 25 deg working range.  It only pushes the actual
            # joint back when a disturbance carries it into the reserved
            # 25--30 deg mechanical margin.
            second_joint_limit_margin = -1.0
            second_joint_target_margin = 0.0
            # Do not penalize the absolute lean needed for a turn.  Penalize
            # only motion around its slowly varying equilibrium instead.
            tilt_margin = 0.0
            # Penalize only high-frequency rocking around the slowly varying
            # steering lean. Absolute lean remains allowed during turns.
            # Absolute lean remains allowed. Only high-pass roll/lean motion
            # is penalized, and this is stronger than the previous light term.
            wobble = -0.30
            torques = -1.0e-6
            ang_vel_xy = 0.0


class RotunbotVelCleanCfgPPO(RotunbotVelCfgPPO):
    # Use the same temporal policy family as the successful point-to-point run.
    runner_class_name = "DWLOnPolicyRunner"

    class policy(RotunbotVelCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        # Give the second joint enough early exploration to discover turning.
        init_noise_std = 0.35
        min_noise_std = 0.15
        max_noise_std = 0.45
        kernel_size = [3, 2]
        filter_size = [16, 8]
        stride_size = [1, 1]
        lh_output_dim = 16
        in_channels = RotunbotVelCleanCfg.env.frame_stack

    class algorithm(RotunbotVelCfgPPO.algorithm):
        # Reduce update aggressiveness and entropy pressure to avoid the
        # policy noise growing until actions saturate at +/-1.
        num_learning_epochs = 3
        num_mini_batches = 8
        learning_rate = 1.0e-4
        # Do not inherit the base adaptive schedule.  With 24 optimizer
        # minibatch updates per PPO iteration, the adaptive 1.5x rule quickly
        # raises the learning rate by orders of magnitude and destabilizes
        # this velocity controller.
        schedule = "fixed"
        desired_kl = None
        entropy_coef = 2.5e-4

    class runner(RotunbotVelCfgPPO.runner):
        policy_class_name = "ActorCriticDWL"
        algorithm_class_name = "PPODWL"
        experiment_name = "rotunbot_vel_clean"
        run_name = "trajectory_v16_turn_direction_centering"
        # V16 changes the command distribution and reward balance. Start with
        # a fresh optimizer and a fresh V16 curriculum state.
        load_optimizer = False
        load_optimizer_when_env_state = True
        load_optimizer_env_state_version = 16
        num_steps_per_env = 64
        max_iterations = 2000
        save_interval = 100
