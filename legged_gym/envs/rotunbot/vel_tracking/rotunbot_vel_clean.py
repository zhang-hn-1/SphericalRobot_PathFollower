from collections import deque
import math

import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from .rotunbot_vel import RotunbotVel
from .rotunbot_vel_clean_config import RotunbotVelCleanCfg


TRACKING_CURRICULUM_DETAIL_NAMES = (
    "speed",
    "turn",
    "lateral",
    "stop",
    "straight",
    "forward_curve",
    "reverse_curve",
    "left",
    "right",
)


class RotunbotVelClean(RotunbotVel):
    """Stable two-action velocity-tracking task for controller calibration."""

    cfg: RotunbotVelCleanCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.data_print = False
        self.output_actions = torch.zeros_like(self.actions)
        self.last_output_actions = torch.zeros_like(self.actions)
        self._target_filter_alpha = torch.tensor(
            [
                self.cfg.control.target_filter_alpha_1,
                self.cfg.control.target_filter_alpha_2,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._target_rate_limits = torch.tensor(
            [self.cfg.control.rate_limit_1, self.cfg.control.rate_limit_2],
            dtype=torch.float32,
            device=self.device,
        )

        self.trajectory_lin_vel = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.trajectory_ang_vel = torch.zeros(self.num_envs, device=self.device)
        self.trajectory_speed = torch.zeros(self.num_envs, device=self.device)
        self.trajectory_ang_vel_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.lateral_lean = torch.atan2(
            self.projected_gravity[:, 1], -self.projected_gravity[:, 2]
        )
        self.lateral_lean_trend = self.lateral_lean.clone()
        self.lateral_lean_wobble = torch.zeros_like(self.lateral_lean)
        self.roll_rate_trend = self.base_ang_vel[:, 0].clone()
        self.roll_rate_wobble = torch.zeros_like(self.roll_rate_trend)
        self.tracking_curriculum_stage = 0
        self.tracking_curriculum_batch_pass_rate = torch.zeros((), device=self.device)
        self.tracking_curriculum_ema = torch.zeros((), device=self.device)
        self.tracking_curriculum_detail_ema = torch.zeros(
            len(TRACKING_CURRICULUM_DETAIL_NAMES), device=self.device
        )
        self.tracking_curriculum_updates = 0
        wobble_tau = max(
            float(self.cfg.rewards.wobble_filter_time_constant), self.dt
        )
        self._wobble_filter_alpha = math.exp(-self.dt / wobble_tau)
        window_s = getattr(
            self.cfg.commands, "trajectory_heading_window_s", 0.4
        )
        self._trajectory_window_steps = max(1, int(round(window_s / self.dt)))
        history_length = 2 * self._trajectory_window_steps + 1
        initial_center_xy = self.root_states[:, :2]
        self._trajectory_position_history = initial_center_xy.unsqueeze(0).repeat(
            history_length, 1, 1
        )
        self._trajectory_position_index = 0
        self._trajectory_window_age = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # The velocity loop is evaluated once per physics substep.  Keep a
        # matching substep state instead of reusing the policy-step history.
        self._last_substep_dof_vel = self.dof_vel.clone()

    def _init_buffers(self):
        """Allocate the temporal observations used by the v4 policy."""
        super()._init_buffers()
        self.num_single_obs = self.cfg.env.num_single_obs
        self.num_short_obs = (
            self.cfg.env.short_frame_stack * self.cfg.env.num_single_obs
        )
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def _scaled_commands(self):
        """Return the directly tracked policy commands [center vx, path wz]."""
        return torch.stack(
            (
                self.commands[:, 0] * self.obs_scales.command_speed,
                self._desired_trajectory_ang_vel()
                * self.obs_scales.command_yaw_rate,
            ),
            dim=1,
        )

    def _desired_trajectory_ang_vel(self):
        """Convert the high-level (v, kappa) command to tangent yaw rate."""
        return torch.clamp(
            self.commands[:, 0] * self.commands[:, 1],
            -self.cfg.commands.trajectory_max_yaw_rate,
            self.cfg.commands.trajectory_max_yaw_rate,
        )

    def _update_trajectory_quantities(self):
        """Compute center velocity and a windowed center-path turning rate.

        The root position/velocity is the ball-center state for this asset.
        Linear velocity is expressed in the horizontally projected base frame.
        Turning rate comes from two adjacent ball-center displacement windows,
        not a one-step velocity-direction derivative.  A bounded lateral lean
        or alternating jitter therefore cannot masquerade as sustained path
        curvature.
        """
        world_lin_vel = self.root_states[:, 7:10]
        ground_lin_vel = world_lin_vel.clone()
        ground_lin_vel[:, 2] = 0.0
        self.trajectory_speed[:] = torch.linalg.vector_norm(
            ground_lin_vel[:, :2], dim=1
        )

        forward_world = quat_apply(self.base_quat, self.forward_vec)
        forward_world[:, 2] = 0.0
        forward_norm = torch.linalg.vector_norm(forward_world[:, :2], dim=1)
        forward_world = forward_world / forward_norm.clamp_min(1.0e-6).unsqueeze(1)
        fallback_forward = torch.zeros_like(forward_world)
        fallback_forward[:, 0] = 1.0
        forward_world = torch.where(
            (forward_norm > 1.0e-6).unsqueeze(1),
            forward_world,
            fallback_forward,
        )
        left_world = torch.stack(
            (
                -forward_world[:, 1],
                forward_world[:, 0],
                torch.zeros_like(forward_world[:, 0]),
            ),
            dim=1,
        )

        self.trajectory_lin_vel[:, 0] = torch.sum(
            ground_lin_vel * forward_world, dim=1
        )
        self.trajectory_lin_vel[:, 1] = torch.sum(
            ground_lin_vel * left_world, dim=1
        )
        self.trajectory_lin_vel[:, 2] = 0.0

        history_index = self._trajectory_position_index
        window_steps = self._trajectory_window_steps
        history_length = self._trajectory_position_history.shape[0]
        previous_index = (history_index - window_steps) % history_length
        older_index = (history_index - 2 * window_steps) % history_length
        current_center_xy = self.root_states[:, :2]

        self._trajectory_position_history[history_index] = current_center_xy
        previous_center_xy = self._trajectory_position_history[previous_index]
        older_center_xy = self._trajectory_position_history[older_index]
        recent_displacement = current_center_xy - previous_center_xy
        previous_displacement = previous_center_xy - older_center_xy

        recent_distance = torch.linalg.vector_norm(recent_displacement, dim=1)
        previous_distance = torch.linalg.vector_norm(previous_displacement, dim=1)
        min_window_distance = getattr(
            self.cfg.commands, "trajectory_min_window_distance", 0.02
        )
        self._trajectory_window_age += 1
        enough_history = self._trajectory_window_age >= 2 * window_steps
        valid_rate = (
            enough_history
            & (recent_distance >= min_window_distance)
            & (previous_distance >= min_window_distance)
        )

        recent_heading = torch.atan2(
            recent_displacement[:, 1], recent_displacement[:, 0]
        )
        previous_heading = torch.atan2(
            previous_displacement[:, 1], previous_displacement[:, 0]
        )
        heading_delta = torch.atan2(
            torch.sin(recent_heading - previous_heading),
            torch.cos(recent_heading - previous_heading),
        )
        window_duration = window_steps * self.dt
        raw_yaw_rate = heading_delta / window_duration
        max_yaw_rate = getattr(
            self.cfg.commands, "trajectory_max_measured_yaw_rate", 2.0
        )
        raw_yaw_rate = torch.clamp(
            raw_yaw_rate, -max_yaw_rate, max_yaw_rate
        )
        self.trajectory_ang_vel[:] = torch.where(
            valid_rate, raw_yaw_rate, torch.zeros_like(raw_yaw_rate)
        )
        self.trajectory_ang_vel_valid[:] = valid_rate
        self._trajectory_position_index = (history_index + 1) % history_length
        self._update_wobble_quantities()
        self._update_tracking_curriculum()

    def _update_tracking_curriculum(self):
        """Unlock tighter radii only after the current v/wz range is tracked."""
        if (
            not getattr(self.cfg.commands, "tracking_curriculum_enabled", False)
            or getattr(self.cfg.commands, "manual_command_mode", False)
            or not hasattr(self, "tracking_curriculum_stage")
        ):
            return

        desired_v = self.commands[:, 0]
        desired_w = self._desired_trajectory_ang_vel()
        stop_command = torch.abs(desired_v) < self.cfg.rewards.linear_command_threshold
        curved_command = torch.abs(desired_w) >= (
            self.cfg.rewards.turn_command_threshold
        )
        # Every command participates in the gate. A moving command whose
        # path yaw rate is not measurable is a failed turn, not an ignored
        # sample. V11 ignored these samples and could advance while nearly
        # standing still.
        valid_sample = torch.ones_like(stop_command, dtype=torch.bool)

        v_error = torch.abs(self.trajectory_lin_vel[:, 0] - desired_v)
        w_error = torch.abs(self.trajectory_ang_vel - desired_w)
        w_tolerance = torch.maximum(
            torch.full_like(
                desired_w,
                self.cfg.commands.tracking_curriculum_w_abs_tolerance,
            ),
            self.cfg.commands.tracking_curriculum_w_rel_tolerance
            * torch.abs(desired_w),
        )
        speed_ok = torch.where(
            stop_command,
            self.trajectory_speed
            <= self.cfg.commands.tracking_curriculum_v_tolerance,
            v_error <= self.cfg.commands.tracking_curriculum_v_tolerance,
        )
        # A straight command does not need a valid path-yaw window. Requiring
        # trajectory_ang_vel_valid for straight motion made the stage-0 gate
        # reject samples that were only supposed to teach forward speed.
        turn_ok = torch.where(
            stop_command | ~curved_command,
            torch.ones_like(speed_ok),
            self.trajectory_ang_vel_valid
            & torch.isfinite(w_error)
            & (w_error <= w_tolerance),
        )
        lateral_ok = (
            torch.abs(self.trajectory_lin_vel[:, 1])
            <= self.cfg.commands.tracking_curriculum_lateral_tolerance
        )
        success = speed_ok & turn_ok & lateral_ok & valid_sample
        batch_pass_rate = success[valid_sample].float().mean()
        self.tracking_curriculum_batch_pass_rate.copy_(batch_pass_rate)

        moving_command = ~stop_command

        def masked_rate(values, mask):
            mask_float = mask.float()
            return torch.sum(values.float() * mask_float) / torch.sum(
                mask_float
            ).clamp_min(1.0)

        detail_batch_rates = torch.stack(
            (
                masked_rate(speed_ok, valid_sample),
                masked_rate(turn_ok, valid_sample),
                masked_rate(lateral_ok, valid_sample),
                masked_rate(success, valid_sample & stop_command),
                masked_rate(
                    success, valid_sample & moving_command & ~curved_command
                ),
                masked_rate(
                    success,
                    valid_sample & (desired_v > 0.0) & curved_command,
                ),
                masked_rate(
                    success,
                    valid_sample & (desired_v < 0.0) & curved_command,
                ),
                masked_rate(
                    success,
                    valid_sample
                    & (desired_w >= self.cfg.rewards.turn_command_threshold),
                ),
                masked_rate(
                    success,
                    valid_sample
                    & (desired_w <= -self.cfg.rewards.turn_command_threshold),
                ),
            )
        )

        alpha = float(self.cfg.commands.tracking_curriculum_ema_alpha)
        if self.tracking_curriculum_updates == 0:
            self.tracking_curriculum_ema.copy_(batch_pass_rate)
            self.tracking_curriculum_detail_ema.copy_(detail_batch_rates)
        else:
            self.tracking_curriculum_ema.mul_(alpha).add_(
                batch_pass_rate,
                alpha=(1.0 - alpha),
            )
            self.tracking_curriculum_detail_ema.mul_(alpha).add_(
                detail_batch_rates,
                alpha=(1.0 - alpha),
            )
        self.tracking_curriculum_updates += 1

        check_interval = self.cfg.commands.tracking_curriculum_check_interval
        should_check = (
            self.tracking_curriculum_updates
            >= self.cfg.commands.tracking_curriculum_min_updates
            and self.tracking_curriculum_updates % check_interval == 0
        )
        ema_pass_rate = (
            float(self.tracking_curriculum_ema.item()) if should_check else 0.0
        )
        if (
            self.tracking_curriculum_stage < 2
            and should_check
            and ema_pass_rate >= self.cfg.commands.tracking_curriculum_pass_rate
        ):
            self.tracking_curriculum_stage += 1
            print(
                "[v/w curriculum] advanced to stage {} after {} updates "
                "with EMA pass rate {:.1%}".format(
                    self.tracking_curriculum_stage,
                    self.tracking_curriculum_updates,
                    ema_pass_rate,
                )
            )
            self.tracking_curriculum_ema.zero_()
            self.tracking_curriculum_detail_ema.zero_()
            self.tracking_curriculum_updates = 0

    def _update_wobble_quantities(self):
        """Separate intentional steering lean from higher-frequency rocking."""
        if not hasattr(self, "lateral_lean_trend"):
            return

        self.lateral_lean[:] = torch.atan2(
            self.projected_gravity[:, 1], -self.projected_gravity[:, 2]
        )
        alpha = self._wobble_filter_alpha
        lean_delta = torch.atan2(
            torch.sin(self.lateral_lean - self.lateral_lean_trend),
            torch.cos(self.lateral_lean - self.lateral_lean_trend),
        )
        self.lateral_lean_trend[:] = self.lateral_lean_trend + (
            1.0 - alpha
        ) * lean_delta
        self.lateral_lean_trend[:] = torch.atan2(
            torch.sin(self.lateral_lean_trend),
            torch.cos(self.lateral_lean_trend),
        )
        self.lateral_lean_wobble[:] = torch.atan2(
            torch.sin(self.lateral_lean - self.lateral_lean_trend),
            torch.cos(self.lateral_lean - self.lateral_lean_trend),
        )

        roll_rate = self.base_ang_vel[:, 0]
        self.roll_rate_trend[:] = (
            alpha * self.roll_rate_trend + (1.0 - alpha) * roll_rate
        )
        self.roll_rate_wobble[:] = roll_rate - self.roll_rate_trend

    def _reset_wobble_state(self, env_ids):
        """Anchor the wobble trend to the current pose after an explicit reset."""
        if not hasattr(self, "lateral_lean_trend") or len(env_ids) == 0:
            return
        lean = torch.atan2(
            self.projected_gravity[env_ids, 1],
            -self.projected_gravity[env_ids, 2],
        )
        self.lateral_lean[env_ids] = lean
        self.lateral_lean_trend[env_ids] = lean
        self.lateral_lean_wobble[env_ids] = 0.0
        self.roll_rate_trend[env_ids] = self.base_ang_vel[env_ids, 0]
        self.roll_rate_wobble[env_ids] = 0.0

    def _reset_trajectory_window(self, env_ids):
        """Re-anchor center-path history after reset or a command change."""
        if not hasattr(self, "_trajectory_position_history") or len(env_ids) == 0:
            return
        center_xy = self.root_states[env_ids, :2]
        self._trajectory_position_history[:, env_ids, :] = center_xy.unsqueeze(0)
        self._trajectory_window_age[env_ids] = 0
        self.trajectory_ang_vel[env_ids] = 0.0
        self.trajectory_ang_vel_valid[env_ids] = False

    def _reset_root_states(self, env_ids):
        """Reset clean calibration trials with zero linear/angular velocity."""
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _process_dof_props(self, props, env_id):
        # Keep the URDF limits available to the base class, then explicitly
        # use effort mode because _compute_torques returns torques.
        LeggedRobot._process_dof_props(self, props, env_id)
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return props

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        noise_vec[2:5] = self.obs_scales.gravity * noise_scales.gravity * noise_level
        noise_vec[5:8] = self.obs_scales.lin_vel * noise_scales.lin_vel * noise_level
        noise_vec[8:11] = self.obs_scales.ang_vel * noise_scales.ang_vel * noise_level
        noise_vec[11] = self.obs_scales.dof_pos * noise_scales.dof_pos * noise_level
        noise_vec[12:14] = self.obs_scales.dof_vel * noise_scales.dof_vel * noise_level
        self.add_noise = cfg.noise.add_noise
        return noise_vec

    def compute_observations(self):
        # Avoid the CPU SciPy quaternion conversion used by the old task.
        # RotunbotVel.post_physics_step keeps these aliases for its history
        # bookkeeping, even though this clean task does not use lag buffers.
        self.lagged_base_lin_vel = self.base_lin_vel
        self.lagged_base_ang_vel = self.base_ang_vel
        self.lagged_dof_pos = self.dof_pos
        self.lagged_dof_vel = self.dof_vel
        obs = torch.cat(
            (
                self._scaled_commands(),
                self.projected_gravity * self.obs_scales.gravity,
                 self.trajectory_lin_vel * self.obs_scales.lin_vel,
                 torch.stack(
                     (
                         self.base_ang_vel[:, 0],
                         self.base_ang_vel[:, 1],
                         self.trajectory_ang_vel,
                     ),
                     dim=1,
                 ) * self.obs_scales.ang_vel,
                self.dof_pos[:, 1:2] * self.obs_scales.dof_pos,
                self.dof_vel[:, :2] * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.privileged_obs_buf is not None:
            privileged_obs = torch.cat(
                (
                    self._scaled_commands(),
                    self.projected_gravity * self.obs_scales.gravity,
                    self.trajectory_lin_vel * self.obs_scales.lin_vel,
                    torch.stack(
                        (
                            self.base_ang_vel[:, 0],
                            self.base_ang_vel[:, 1],
                            self.trajectory_ang_vel,
                        ),
                        dim=1,
                    ) * self.obs_scales.ang_vel,
                    self.dof_pos[:, :2] * self.obs_scales.dof_pos,
                    self.dof_vel[:, :2] * self.obs_scales.dof_vel,
                    self.actions,
                    self.env_frictions,
                    self.body_mass / 10.0,
                    self.total_mass / 10.0,
                ),
                dim=-1,
            )

        if self.add_noise:
            obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * self.noise_scale_vec

        self.obs_history.append(obs)
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )
        if self.privileged_obs_buf is not None:
            self.critic_history.append(privileged_obs)
            self.privileged_obs_buf = torch.cat(
                list(self.critic_history), dim=1
            )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0 or getattr(self.cfg.commands, "manual_command_mode", False):
            return

        speed_range = self.cfg.commands.trajectory_speed_range
        curvature_range = self.cfg.commands.trajectory_curvature_range
        num_commands = len(env_ids)

        # Commands are generated through kappa so v=0 still implies wz=0, but
        # the policy observes the resulting (v, wz) directly.  Radius ranges
        # are unlocked by measured tracking performance, not elapsed time.
        curriculum_stage = getattr(self, "tracking_curriculum_stage", 0)
        if curriculum_stage <= 0:
            mode_probabilities = self.cfg.commands.trajectory_mode_probabilities_large
            curvature_anchor_values = (
                self.cfg.commands.trajectory_curvature_anchor_values_large
            )
            command_table = self.cfg.commands.trajectory_command_table_large
            command_table_probability = (
                self.cfg.commands.trajectory_command_table_probability_large
            )
        elif curriculum_stage == 1:
            mode_probabilities = self.cfg.commands.trajectory_mode_probabilities_medium
            curvature_anchor_values = (
                self.cfg.commands.trajectory_curvature_anchor_values_medium
            )
            command_table = self.cfg.commands.trajectory_command_table_medium
            command_table_probability = (
                self.cfg.commands.trajectory_command_table_probability_medium
            )
        else:
            mode_probabilities = self.cfg.commands.trajectory_mode_probabilities_small
            curvature_anchor_values = (
                self.cfg.commands.trajectory_curvature_anchor_values_small
            )
            command_table = self.cfg.commands.trajectory_command_table_small
            command_table_probability = (
                self.cfg.commands.trajectory_command_table_probability_small
            )

        probabilities = torch.as_tensor(
            mode_probabilities,
            dtype=torch.float32,
            device=self.device,
        )
        probabilities = probabilities / probabilities.sum()
        mode = torch.multinomial(
            probabilities.unsqueeze(0).expand(num_commands, -1),
            num_samples=1,
        ).squeeze(1)

        speed = torch_rand_float(
            speed_range[0], speed_range[1], (num_commands, 1), device=self.device
        ).squeeze(1)
        min_abs_curvature = getattr(
            self.cfg.commands, "trajectory_min_abs_curvature", 0.0
        )
        max_abs_curvature = max(abs(curvature_range[0]), abs(curvature_range[1]))
        curvature_magnitude = torch_rand_float(
            min_abs_curvature,
            max_abs_curvature,
            (num_commands, 1),
            device=self.device,
        ).squeeze(1)
        curvature_anchor_probability = getattr(
            self.cfg.commands, "trajectory_curvature_anchor_probability", 0.0
        )
        if curvature_anchor_values and curvature_anchor_probability > 0.0:
            curvature_anchors = torch.as_tensor(
                curvature_anchor_values,
                dtype=curvature_magnitude.dtype,
                device=self.device,
            )
            anchor_indices = torch.randint(
                curvature_anchors.numel(),
                (num_commands,),
                device=self.device,
            )
            use_curvature_anchor = (mode == 2) & (
                torch_rand_float(
                    0.0, 1.0, (num_commands, 1), device=self.device
                ).squeeze(1)
                < curvature_anchor_probability
            )
            curvature_magnitude = torch.where(
                use_curvature_anchor,
                curvature_anchors[anchor_indices],
                curvature_magnitude,
            )
        curvature_sign = torch.where(
            torch_rand_float(
                -1.0, 1.0, (num_commands, 1), device=self.device
            ).squeeze(1) < 0.0,
            -torch.ones_like(curvature_magnitude),
            torch.ones_like(curvature_magnitude),
        )
        curvature = torch.clamp(
            curvature_sign * curvature_magnitude,
            curvature_range[0],
            curvature_range[1],
        )

        # Non-stop modes should not silently become stop commands.  Move a
        # sampled near-zero speed to the minimum meaningful signed speed.
        min_speed = self.cfg.commands.trajectory_min_speed
        curve_min_speed = getattr(
            self.cfg.commands, "trajectory_curve_min_speed", min_speed
        )
        required_speed = torch.where(
            mode == 2,
            torch.full_like(speed, curve_min_speed),
            torch.full_like(speed, min_speed),
        )
        speed_sign = torch.where(speed < 0.0, -torch.ones_like(speed), torch.ones_like(speed))
        speed = torch.where(
            torch.abs(speed) < required_speed,
            speed_sign * required_speed,
            speed,
        )
        speed_anchor = getattr(self.cfg.commands, "trajectory_speed_anchor", 0.0)
        speed_anchor_probability = getattr(
            self.cfg.commands, "trajectory_speed_anchor_probability", 0.0
        )
        if speed_anchor > 0.0 and speed_anchor_probability > 0.0:
            use_speed_anchor = (mode != 0) & (
                torch_rand_float(
                    0.0, 1.0, (num_commands, 1), device=self.device
                ).squeeze(1)
                < speed_anchor_probability
            )
            speed = torch.where(
                use_speed_anchor,
                speed_sign * speed_anchor,
                speed,
            )
        speed = torch.where(mode == 0, torch.zeros_like(speed), speed)
        curvature = torch.where(
            mode == 1, torch.zeros_like(curvature), curvature
        )
        curvature = torch.where(
            mode == 0, torch.zeros_like(curvature), curvature
        )

        # V12: replace random command imbalance with a balanced table. Each
        # resampling batch receives every command direction as evenly as its
        # size allows, so the curriculum gate and fixed-command evaluator are
        # measuring the same task.
        if command_table and command_table_probability > 0.0:
            table = torch.as_tensor(
                command_table,
                dtype=speed.dtype,
                device=self.device,
            )
            table_size = table.shape[0]
            repeat_count = (num_commands + table_size - 1) // table_size
            table_indices = torch.arange(
                repeat_count * table_size, device=self.device
            ) % table_size
            table_indices = table_indices[
                torch.randperm(table_indices.numel(), device=self.device)
            ][:num_commands]
            table_samples = table[table_indices]
            use_table = (
                torch_rand_float(
                    0.0, 1.0, (num_commands, 1), device=self.device
                ).squeeze(1)
                < command_table_probability
            )
            speed = torch.where(use_table, table_samples[:, 0], speed)
            curvature = torch.where(use_table, table_samples[:, 1], curvature)

        self.commands[env_ids, 0] = torch.where(
            torch.abs(speed) > 0.0, speed, torch.zeros_like(speed)
        )
        self.commands[env_ids, 1] = curvature
        # Measurements spanning two different commands do not describe either
        # command. Start a fresh local path window at every resample.
        self._reset_trajectory_window(env_ids)

    def get_checkpoint_state(self):
        """Persist the performance-gated radius curriculum across resumes."""
        return {
            "checkpoint_state_version": 16,
            "tracking_curriculum_stage": int(self.tracking_curriculum_stage),
            "tracking_curriculum_batch_pass_rate": float(
                self.tracking_curriculum_batch_pass_rate.item()
            ),
            "tracking_curriculum_ema": float(self.tracking_curriculum_ema.item()),
            "tracking_curriculum_detail_ema": (
                self.tracking_curriculum_detail_ema.detach().cpu().tolist()
            ),
            "tracking_curriculum_updates": int(self.tracking_curriculum_updates),
        }

    def get_tracking_curriculum_status(self):
        """Return lightweight values for the PPO console and TensorBoard logs."""
        if not getattr(self.cfg.commands, "tracking_curriculum_enabled", False):
            return None
        stage_names = ("stop/straight", "large radius", "small radius")
        stage = max(0, min(int(self.tracking_curriculum_stage), len(stage_names) - 1))
        detail_values = self.tracking_curriculum_detail_ema.detach().cpu().tolist()
        return {
            "stage": stage,
            "max_stage": len(stage_names) - 1,
            "stage_name": stage_names[stage],
            "batch_pass_rate": float(
                self.tracking_curriculum_batch_pass_rate.item()
            ),
            "ema_pass_rate": float(self.tracking_curriculum_ema.item()),
            "detail_ema": dict(
                zip(TRACKING_CURRICULUM_DETAIL_NAMES, detail_values)
            ),
            "updates": int(self.tracking_curriculum_updates),
            "min_updates": int(self.cfg.commands.tracking_curriculum_min_updates),
            "pass_threshold": float(
                self.cfg.commands.tracking_curriculum_pass_rate
            ),
        }

    def set_checkpoint_state(self, state):
        """Restore curriculum state only for a compatible V16 checkpoint."""
        if not state:
            return
        if int(state.get("checkpoint_state_version", -1)) != 16:
            print(
                "Ignoring pre-V16 environment state; starting staged "
                "curriculum at stage 0"
            )
            return
        self.tracking_curriculum_stage = int(
            state.get("tracking_curriculum_stage", self.tracking_curriculum_stage)
        )
        self.tracking_curriculum_batch_pass_rate.fill_(
            float(
                state.get(
                    "tracking_curriculum_batch_pass_rate",
                    self.tracking_curriculum_batch_pass_rate.item(),
                )
            )
        )
        self.tracking_curriculum_ema.fill_(
            float(state.get("tracking_curriculum_ema", self.tracking_curriculum_ema.item()))
        )
        detail_ema = state.get("tracking_curriculum_detail_ema")
        if detail_ema is not None and len(detail_ema) == len(
            TRACKING_CURRICULUM_DETAIL_NAMES
        ):
            self.tracking_curriculum_detail_ema.copy_(
                torch.as_tensor(
                    detail_ema,
                    dtype=self.tracking_curriculum_detail_ema.dtype,
                    device=self.device,
                )
            )
        self.tracking_curriculum_updates = int(
            state.get("tracking_curriculum_updates", self.tracking_curriculum_updates)
        )

    def _compute_torques(self, actions):
        actions = torch.clamp(actions, -1.0, 1.0)
        qdot_target = actions[:, 0] * self.cfg.control.first_actionScale
        q_target = actions[:, 1] * self.cfg.control.second_actionScale
        qdot_target = torch.clamp(
            qdot_target,
            -self.cfg.control.first_vel_limits,
            self.cfg.control.first_vel_limits,
        )
        q_target = torch.clamp(
            q_target,
            -self.cfg.control.second_pos_limits,
            self.cfg.control.second_pos_limits,
        )

        targets = torch.stack((qdot_target, q_target), dim=-1)
        if getattr(self.cfg.control, "set_a_rate_limit", False):
            targets = self.last_output_actions + self._target_filter_alpha * (
                targets - self.last_output_actions
            )
            targets = self.last_output_actions + torch.clamp(
                targets - self.last_output_actions,
                -self._target_rate_limits,
                self._target_rate_limits,
            )

        self.output_actions[:] = targets
        torques = torch.zeros_like(actions)
        first_axis_accel = (
            self.dof_vel[:, 0] - self._last_substep_dof_vel[:, 0]
        ) / self.sim_params.dt
        torques[:, 0] = 21.17 * (targets[:, 0] - self.dof_vel[:, 0]) - 0.97 * (
            first_axis_accel
        )
        torques[:, 1] = 297.46 * (targets[:, 1] - self.dof_pos[:, 1]) - 149.97 * self.dof_vel[:, 1]
        self._last_substep_dof_vel[:] = self.dof_vel
        torque_limits = torch.tensor(
            [self.cfg.control.torque_limits_1, self.cfg.control.torque_limits_2],
            device=self.device,
        )
        return torch.clamp(torques, -torque_limits, torque_limits)

    def step(self, actions):
        # Keep the action channel itself normalized as well as the physical
        # joint targets. This prevents an outlier policy sample from entering
        # the next observation with the old clip_actions=100 range.
        result = super().step(torch.clamp(actions, -1.0, 1.0))
        self.last_output_actions[:] = self.output_actions
        return result

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.output_actions[env_ids] = 0.0
        self.last_output_actions[env_ids] = 0.0
        self.trajectory_lin_vel[env_ids] = 0.0
        self.trajectory_ang_vel[env_ids] = 0.0
        self.trajectory_speed[env_ids] = 0.0
        self.trajectory_ang_vel_valid[env_ids] = False
        self._last_substep_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.reset_observation_history(env_ids)

    def reset_observation_history(self, env_ids=None):
        """Clear temporal state after an episode reset or manual command set."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for history in self.obs_history:
            history[env_ids] = 0.0
        for history in self.critic_history:
            history[env_ids] = 0.0
        self._reset_trajectory_window(env_ids)
        self._reset_wobble_state(env_ids)

    def _reward_tracking_trajectory(self):
        """Track linear and turning commands as one coupled path objective."""
        vx_error = self.commands[:, 0] - self.trajectory_lin_vel[:, 0]
        lateral_speed = self.trajectory_lin_vel[:, 1]
        wz_error = self._desired_trajectory_ang_vel() - self.trajectory_ang_vel
        lin_cost = (
            vx_error.square()
            + self.cfg.rewards.trajectory_lateral_weight * lateral_speed.square()
        ) / self.cfg.rewards.tracking_sigma_lin
        ang_cost = wz_error.square() / self.cfg.rewards.tracking_sigma_ang
        return torch.exp(-(lin_cost + ang_cost))

    def _reward_tracking_lin_vel(self):
        """Dense progress reward with exactly zero reward for standing still."""
        vx_error = self.commands[:, 0] - self.trajectory_lin_vel[:, 0]
        command_speed = torch.abs(self.commands[:, 0])
        moving_command = command_speed >= self.cfg.rewards.linear_command_threshold

        relative_progress = 1.0 - torch.abs(vx_error) / command_speed.clamp_min(
            self.cfg.rewards.linear_command_threshold
        )
        # A moving command must not receive a neutral/positive score while
        # the robot is nearly stopped.  The old [0, 1] clamp made vx=0.043
        # under a 0.20 m/s command look like useful progress and encouraged a
        # low-speed local optimum.  The centered score is -1 at zero speed,
        # 0 at 50% tracking, and +1 at exact tracking.
        moving_reward = torch.clamp(2.0 * relative_progress - 1.0, -1.0, 1.0)
        stop_penalty = -torch.clamp(
            torch.abs(self.trajectory_lin_vel[:, 0])
            / self.cfg.rewards.stop_speed_tolerance,
            0.0,
            1.0,
        )
        reward = torch.where(moving_command, moving_reward, stop_penalty)

        lateral_penalty = self.cfg.rewards.trajectory_lateral_weight * torch.clamp(
            torch.abs(self.trajectory_lin_vel[:, 1])
            / self.cfg.rewards.lateral_speed_tolerance,
            0.0,
            1.0,
        )
        return reward - lateral_penalty

    def _reward_tracking_ang_vel(self):
        """Reward actual turning improvement over the no-turn solution.

        For a curved command this relative score is exactly zero when the
        robot does not turn and one at perfect tracking. Wrong-direction
        motion receives a bounded negative signal so left and right commands
        cannot collapse to the same policy. Straight/stop commands retain a
        zero stationary baseline and only a mild unwanted-turn penalty.
        """
        desired_wz = self._desired_trajectory_ang_vel()
        wz_error = desired_wz - self.trajectory_ang_vel
        relative_turn_reward = 1.0 - torch.abs(wz_error) / torch.abs(
            desired_wz
        ).clamp_min(self.cfg.rewards.turn_command_threshold)
        relative_turn_reward = torch.clamp(
            relative_turn_reward,
            self.cfg.rewards.wrong_turn_reward_floor,
            1.0,
        )
        speed_progress = torch.clamp(
            torch.sign(self.commands[:, 0]) * self.trajectory_lin_vel[:, 0]
            / torch.abs(self.commands[:, 0]).clamp_min(
                self.cfg.rewards.linear_command_threshold
            ),
            0.0,
            1.0,
        )
        relative_turn_reward = relative_turn_reward * speed_progress

        turn_valid = self.trajectory_ang_vel_valid & torch.isfinite(
            self.trajectory_ang_vel
        )
        # Once the robot is actually moving, an unmeasurable path turn must
        # not be treated as a zero-cost turn.  The speed reward handles the
        # initial acceleration; this term supplies a bounded extra signal for
        # curved commands until a valid center-path yaw rate is established.
        invalid_turn_penalty = -0.25 * speed_progress

        straight_penalty = -self.cfg.rewards.straight_turn_penalty_weight * (
            torch.clamp(
                torch.abs(self.trajectory_ang_vel)
                / self.cfg.rewards.straight_yaw_rate_tolerance,
                0.0,
                1.0,
            )
        )
        curved_command = torch.abs(desired_wz) >= (
            self.cfg.rewards.turn_command_threshold
        )
        curved_reward = torch.where(
            turn_valid,
            relative_turn_reward,
            invalid_turn_penalty,
        )
        return torch.where(curved_command, curved_reward, straight_penalty)

    def _reward_joint_target_error(self):
        """Penalize physical targets that the two actuators cannot follow."""
        velocity_error = (
            self.output_actions[:, 0] - self.dof_vel[:, 0]
        ) / self.cfg.control.first_vel_limits
        position_error = (
            self.output_actions[:, 1] - self.dof_pos[:, 1]
        ) / self.cfg.control.second_pos_limits
        return velocity_error.square() + position_error.square()

    def _reward_steering_target_centering(self):
        """Center the steering joint for stop/straight commands.

        V15 learned a shortcut: it held the second joint at its +25 degree
        working limit for stop, straight, and both turn directions. That
        produced the large unwanted yaw in the fixed-command evaluation. A
        curved command is intentionally exempt here because a sustained lean
        is the mechanism used to turn.
        """
        desired_wz = self._desired_trajectory_ang_vel()
        curved_command = torch.abs(desired_wz) >= (
            self.cfg.rewards.turn_command_threshold
        )
        target_norm = (
            self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        )
        actual_norm = self.dof_pos[:, 1] / self.cfg.control.second_pos_limits
        center_cost = target_norm.square() + 0.25 * actual_norm.square()
        return (~curved_command).float() * center_cost

    def _reward_steering_target_direction(self):
        """Discourage a steering target with the wrong curvature sign.

        The URDF joint axis makes the useful steering sign correspond to the
        signed curvature command (v and wz have already been combined into
        kappa in ``commands[:, 1]``). This term is deliberately weak; the
        measured path-yaw tracking reward remains the authority on magnitude.
        """
        desired_kappa = self.commands[:, 1]
        desired_wz = self._desired_trajectory_ang_vel()
        curved_command = torch.abs(desired_wz) >= (
            self.cfg.rewards.turn_command_threshold
        )
        target_norm = (
            self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        )
        wrong_sign = torch.relu(-desired_kappa * target_norm)
        return curved_command.float() * wrong_sign.square()

    def _reward_target_rate(self):
        """Penalize rapid changes of the filtered physical targets.

        A constant steering lean has zero target-rate cost, so this term does
        not suppress the roll angle needed by a turn. It only suppresses the
        target chatter that excited the earlier velocity policies.
        """
        target_delta = self.output_actions - self.last_output_actions
        normalized_delta = torch.stack(
            (
                target_delta[:, 0] / self.cfg.control.first_vel_limits,
                target_delta[:, 1] / self.cfg.control.second_pos_limits,
            ),
            dim=1,
        )
        return torch.sum(normalized_delta.square(), dim=1)

    def _reward_stop_stability(self):
        """Keep the internal mass and sphere quiet for an exact stop command."""
        stop_command = (torch.abs(self.commands[:, 0]) < 1.0e-6) & (
            torch.abs(self.commands[:, 1]) < 1.0e-6
        )
        second_joint_offset = (
            self.dof_pos[:, 1] / self.cfg.control.second_pos_limits
        ).square()
        target_effort = torch.sum(self.actions.square(), dim=1)
        tilt = torch.sum(self.projected_gravity[:, :2].square(), dim=1)
        stop_speed = torch.square(
            self.trajectory_speed / self.cfg.rewards.stop_speed_tolerance
        ).clamp_max(4.0)
        stop_joint_motion = torch.square(
            self.dof_vel[:, 0] / self.cfg.control.first_vel_limits
        ) + torch.square(self.dof_vel[:, 1] / 5.0)
        stop_body_motion = torch.sum(torch.square(self.base_ang_vel / 2.0), dim=1)
        return stop_command.float() * (
            second_joint_offset
            + target_effort
            + tilt
            + 0.25 * stop_speed
            + 0.10 * stop_joint_motion
            + 0.05 * stop_body_motion
        )

    def _reward_second_joint_limit_margin(self):
        """Return the actual-joint incursion into the reserved 25--30 deg margin."""
        working_limit = float(self.cfg.control.second_pos_limits)
        mechanical_limit = float(self.cfg.control.second_mechanical_pos_limit)
        reserve = max(mechanical_limit - working_limit, 1.0e-6)
        overtravel = torch.relu(torch.abs(self.dof_pos[:, 1]) - working_limit)
        return torch.square(overtravel / reserve)

    def _reward_second_joint_target_margin(self):
        """Guard against target overtravel if another controller bypasses clipping."""
        working_limit = float(self.cfg.control.second_pos_limits)
        mechanical_limit = float(self.cfg.control.second_mechanical_pos_limit)
        reserve = max(mechanical_limit - working_limit, 1.0e-6)
        overtravel = torch.relu(torch.abs(self.output_actions[:, 1]) - working_limit)
        return torch.square(overtravel / reserve)

    def _reward_tilt_margin(self):
        """Legacy reward retained for checkpoint/config compatibility; disabled."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_wobble(self):
        """Penalize rocking around the slow steering lean, not the lean itself."""
        angle_error = (
            self.lateral_lean_wobble / self.cfg.rewards.wobble_angle_tolerance
        )
        roll_rate_error = (
            self.roll_rate_wobble
            / self.cfg.rewards.wobble_roll_rate_tolerance
        )
        return angle_error.square() + (
            self.cfg.rewards.wobble_roll_rate_weight * roll_rate_error.square()
        )
