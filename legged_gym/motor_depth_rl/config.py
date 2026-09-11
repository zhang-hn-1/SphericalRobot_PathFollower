"""Explicit configuration for the new depth-to-joint RL experiment."""
from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro_config import RotunbotTargetReproCfg


class MotorDepthCfg(RotunbotTargetReproCfg):
    experiment_profile = "motor_depth_rl_v1"

    class env(RotunbotTargetReproCfg.env):
        num_envs = 64
        env_spacing = 32.0
        episode_length_s = 180.0
        # The inherited simulator buffers remain proprioception-only.
        num_observations = 380
        num_privileged_obs = 63

    class control(RotunbotTargetReproCfg.control):
        # Normalized policy actions are scaled exactly once by the original R
        # executor. The original actuator limits/rate limits/PD are unchanged.
        first_actionScale = 3.0
        second_actionScale = 0.45

    class commands(RotunbotTargetReproCfg.commands):
        curriculum = False
        target_curriculum = False
        random_start_yaw = False
        stop_distance = 0.4
        command_yaw = False

    class evaluation(RotunbotTargetReproCfg.evaluation):
        target_error_threshold = 0.4
        stop_velocity_threshold = 0.1

    class rewards(RotunbotTargetReproCfg.rewards):
        # This complete table replaces inherited reward scales. Terms below
        # are applied explicitly in compute_reward; there is no hidden dt
        # multiplication and no positive alive/balance/proximity reward.
        only_positive_rewards = False
        reward_schema = "depth_avoidance_reward_v3"
        excessive_tilt_start_rad = 0.5
        near_goal_region_m = 0.8

        class scales:
            progress_per_m = 2.0
            success_once = 40.0
            collision_once = -10.0
            bounds_once = -10.0
            unstable_once = -10.0
            time_per_s = -0.02
            excessive_tilt_per_s = -0.03
            near_goal_speed_per_s = -0.1
            torques_squared_per_s = -1.0e-7
            target_change_squared_per_step = -0.002

    class task_catalog:
        pool_root = "artifacts/MOTOR_SRU_INTEGRATION_20260906/depth_avoidance_redesign/maps"
        role = "train"
        manifest_path = pool_root + "/data/train.json"
        sample_seed = 60
        max_stage = 0
        resample_on_reset = True
        fixed_task_indices = None
        ordered_first_batch = False

    class navigation:
        robot_radius = 0.4
        pillar_radius = 0.4
        pillar_height = 1.5
        pillar_count = 40
        half_extent = 8.0
        radius_constraint_enabled = False
        pillar_contact_force_threshold = 1.0
        instability_angle = 1.2

    class camera:
        width = 64
        height = 40
        horizontal_fov = 105.0
        near_plane = 0.25
        far_plane = 10.0
        position = (0.42, 0.0, 0.0)
        rotation = (0.0, 0.0, 0.0, 1.0)
        motor_steps_per_frame = 10
        horizontal_flip_to_robot_frame = False
        invalid_depth_value_m = 0.0


def make_depth_config(role="train", num_envs=64, seed=60, fixed_task_indices=None, max_stage=0):
    if role not in ("train", "val", "test", "diagnostic"):
        raise ValueError("Unknown depth RL split")
    if int(num_envs) != num_envs or int(num_envs) <= 0:
        raise ValueError("num_envs must be a positive integer")
    if max_stage is not None and max_stage not in (0, 1, 2):
        raise ValueError("max_stage must be 0, 1, 2, or None for the complete pool")
    cfg = MotorDepthCfg()
    cfg.seed = int(seed)
    cfg.env.num_envs = int(num_envs)
    cfg.task_catalog.role = role
    cfg.task_catalog.manifest_path = cfg.task_catalog.pool_root + "/data/" + role + ".json"
    cfg.task_catalog.sample_seed = int(seed)
    cfg.task_catalog.max_stage = max_stage if role == "train" else None
    cfg.task_catalog.fixed_task_indices = None if fixed_task_indices is None else list(fixed_task_indices)
    cfg.task_catalog.resample_on_reset = role == "train" and fixed_task_indices is None
    cfg.task_catalog.ordered_first_batch = role != "train"
    return cfg


make_motor_config = make_depth_config
