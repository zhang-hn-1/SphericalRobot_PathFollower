"""Paper-reproduction conditions with the existing 19-D DWL-CNN policy unchanged."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO

from .rotunbot_target_lh_config import RotunbotTargetLHCfg


class RotunbotTargetReproCfg(RotunbotTargetLHCfg):
    """Flat, obstacle-free point-to-point task for code reproduction."""

    class env(RotunbotTargetLHCfg.env):
        # Keep the Graduation training batch/environment setting.  The number
        # of parallel environments does not change the per-episode test rule.
        num_envs = 2048
        num_actions = 2
        num_single_obs = 19
        frame_stack = 20
        short_frame_stack = 5
        c_frame_stack = 3
        num_observations = frame_stack * num_single_obs
        num_privileged_obs = c_frame_stack * RotunbotTargetLHCfg.env.single_num_privileged_obs
        episode_length_s = 60

    class terrain(RotunbotTargetLHCfg.terrain):
        # The requested reproduction is the planar, no-obstacle experiment.
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class control(RotunbotTargetLHCfg.control):
        # Reproduction control period is dt=0.02 s.  LeggedRobot uses
        # cfg.sim.dt * cfg.control.decimation as the policy period.
        decimation = 1
        rate_limit_1 = 0.02
        rate_limit_2 = 0.04
        first_vel_limits = 3.0
        second_pos_limits = 0.45
        torque_limits_1 = 100.0
        torque_limits_2 = 100.0

    class commands(RotunbotTargetLHCfg.commands):
        # Compatibility aliases retained for inherited LH code.  The formal
        # The reproduction success check in rotunbot_target_repro.py reads evaluation.* below.
        random_start_yaw = True
        stop_distance = 0.20
        stop_vel = 0.1

        # Point-to-point curriculum: keep the paper's complete target map
        # fixed and gradually tighten only the training success tolerance.
        target_curriculum = True
        curriculum_success_distance_start = 1.0
        curriculum_success_distance_min = 0.20
        curriculum_success_distance_step = 0.20
        target_curriculum_window = 2048
        target_curriculum_success_rate = 0.80

        class ranges(RotunbotTargetLHCfg.commands.ranges):
            pos_x = [-5.0, 5.0]
            pos_y = [-5.0, 5.0]

    class asset(RotunbotTargetLHCfg.asset):
        # Keep the Graduation robot model; it is mechanically equivalent for
        # this comparison and keeps the trained policy/model pairing intact.
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot_test2.urdf"

    class init_state(RotunbotTargetLHCfg.init_state):
        randomize_initial_velocity = False

    class noise(RotunbotTargetLHCfg.noise):
        # Keep the Graduation training observation-noise setting. play.py
        # disables it during the clean paper-style evaluation.
        add_noise = True

    class rewards(RotunbotTargetLHCfg.rewards):
        debug_print = False
        only_positive_rewards = False
        close_para = 1.0
        # A narrow distance-shaping scale is necessary at the initial
        # [-1, 1] curriculum stage.  sigma=8 makes the distance reward almost
        # constant, so a stationary policy can receive reward without moving.
        tracking_sigma_main = 0.5
        soft_dof_pos_limit = 1.0
        stop_reward_multiplier = 1.0

        class scales(RotunbotTargetLHCfg.rewards.scales):
            # Paper-inspired point-to-point shaping.  The asymmetric
            # close/away pair is disabled because it can make a loop around
            # the target have positive net reward.
            termination = -0.0
            close_to_target = 0.0
            away_to_target = 0.0
            approaching_target = 0.5
            to_target = 1.5
            stop = 20.0
            balance = 0.1
            torques = -1.0e-5
            action_rate = -0.004
            time = -0.5
            overturn = -0.5
            lin_vel_x_limit = -1.0
            ang_vel_z_limit = -0.2

            # Additional small braking term to reduce overspeed inside the
            # formal 0.20 m stopping region.
            near_goal_speed = -0.2

            # Graduation-only terms are disabled in this paper-inspired run.
            ang_vel_xy = 0.0
            lin_vel_z = 0.0
            dof_acc = 0.0
            dof_pos_limits = 0.0


    class evaluation:
        # Evaluation protocol requested for the paper-style comparison.
        # The paper explicitly reports 40 repeated trials per method and
        # environment in the real-robot experiments.
        num_eval_episodes = 40
        target_error_threshold = 0.20
        stop_velocity_threshold = 0.10


class RotunbotTargetReproCfgPPO(LeggedRobotCfgPPO):
    """Checkpoint-compatible PPO/DWL-CNN settings for continued training.

    The paper's exact actor uses a Transformer LH Encoder; this class keeps
    the existing 19-D DWL-CNN so its checkpoints remain loadable.
    """

    seed = 3
    # Keep the existing DWL runner so the 19-D checkpoint remains loadable.
    runner_class_name = "DWLOnPolicyRunner"

    class policy:
        init_noise_std = 0.5
        min_noise_std = 0.2
        max_noise_std = 1.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        kernel_size = [3, 2]
        filter_size = [16, 8]
        stride_size = [1, 1]
        lh_output_dim = 16
        in_channels = RotunbotTargetReproCfg.env.frame_stack

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 16
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCriticDWL"
        algorithm_class_name = "PPODWL"
        num_steps_per_env = 96
        max_iterations = 15000
        save_interval = 50
        experiment_name = "rotunbot_target_repro"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
