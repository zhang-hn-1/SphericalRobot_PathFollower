"""Isolated direct-motor task; the source point-to-point configuration is unchanged."""
from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro_config import RotunbotTargetReproCfg


class MotorPillarCfg(RotunbotTargetReproCfg):
    experiment_profile = "motor_sru_no_radius_time_v2"

    class env(RotunbotTargetReproCfg.env):
        num_envs = 64
        env_spacing = 32.0
        episode_length_s = 180.0
        # These remain the original proprioceptive buffer dimensions.
        num_observations = 380
        num_privileged_obs = 63

    class commands(RotunbotTargetReproCfg.commands):
        curriculum = False
        target_curriculum = False
        random_start_yaw = False  # the manifest stores tangent + one +/-15 degree draw
        stop_distance = 0.4
        command_yaw = False

    class evaluation(RotunbotTargetReproCfg.evaluation):
        target_error_threshold = 0.4
        stop_velocity_threshold = 0.1

    class rewards(RotunbotTargetReproCfg.rewards):
        class scales(RotunbotTargetReproCfg.rewards.scales):
            # The inherited |w| > .72*|v|+.05 cost is also a radius-like
            # constraint. Disable it only in this unrestricted motor profile.
            ang_vel_z_limit = 0.0

    class task_catalog:
        pool_root = "artifacts/MOTOR_SRU_INTEGRATION_20260906/task_pool"
        role = "train"
        manifest_path = pool_root + "/data/train.json"
        sample_seed = 60
        resample_on_reset = True
        fixed_task_indices = None
        ordered_first_batch = False
        strict_pillar_intersection_max_distance = 0.35

    class navigation:
        robot_radius = 0.4
        pillar_radius = 0.4
        pillar_height = 1.5
        pillar_count = 8
        half_extent = 8.0
        # Historical diagnostic reference only. Actual motion is unconstrained
        # by turning radius; the existing conservative map witnesses stay fixed.
        radius_reference_m = 2.0
        radius_constraint_enabled = False
        radius_speed_threshold = 0.02
        radius_yaw_tolerance = 0.005
        radius_penalty_per_second = 0.0
        collision_penalty_once = 10.0
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
        checkpoint_path = "artifacts/MOTOR_SRU_INTEGRATION_20260906/models/depth_encoder.pt"


def make_motor_config(role="train", num_envs=64, seed=60, fixed_task_indices=None):
    if role not in ("train", "val", "dev20"):
        raise ValueError("Unknown motor task split")
    if int(num_envs) <= 0:
        raise ValueError("num_envs must be positive")
    cfg = MotorPillarCfg()
    cfg.seed = int(seed)
    cfg.env.num_envs = int(num_envs)
    cfg.task_catalog.role = role
    cfg.task_catalog.manifest_path = cfg.task_catalog.pool_root + "/data/" + role + ".json"
    cfg.task_catalog.sample_seed = int(seed)
    cfg.task_catalog.fixed_task_indices = None if fixed_task_indices is None else list(fixed_task_indices)
    cfg.task_catalog.resample_on_reset = role == "train" and fixed_task_indices is None
    cfg.task_catalog.ordered_first_batch = role != "train"
    return cfg
