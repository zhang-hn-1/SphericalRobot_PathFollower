"""Geometric path-following task for the two-joint spherical robot."""

import math
import os
from collections import deque

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel_clean import (
    RotunbotVelClean,
)
from .rotunbot_path_config import RotunbotPathCfg


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class RotunbotPath(RotunbotVelClean):
    """Track an arc-length path while choosing progress speed through joints."""

    cfg: RotunbotPathCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # These fields must exist before the parent creates and resets the first
        # environments.  Per-type statistics prevent easy straight paths from
        # hiding failed left/right/S paths at a curriculum transition.
        self.path_curriculum_stage = int(os.environ.get("PATH_TRAIN_STAGE", "0"))
        self.path_curriculum_attempts = 0
        self.path_curriculum_successes = 0
        self.path_curriculum_last_rate = 0.0
        self.path_curriculum_type_attempts = [0, 0, 0, 0]
        self.path_curriculum_type_successes = [0, 0, 0, 0]
        self.path_curriculum_last_rates = [0.0, 0.0, 0.0, 0.0]
        forced_name = os.environ.get("PATH_TRAIN_TYPE", "").strip()
        forced_types = {"straight": 0, "left_arc": 1, "right_arc": 2, "s_curve": 3}
        if forced_name:
            if forced_name not in forced_types:
                raise ValueError("Unsupported PATH_TRAIN_TYPE: " + forced_name)
            self.forced_path_type = forced_types[forced_name]
        forced_curvature = os.environ.get("PATH_TRAIN_CURVATURE", "").strip()
        if forced_curvature:
            self.forced_curvature = float(forced_curvature)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _get_noise_scale_vec(self, cfg):
        noise = torch.zeros(cfg.env.num_single_obs, device=self.device)
        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        # Path anchor [0:4] and previous actions [17:19] stay exact.
        noise[4:8] = scales.quat * level
        noise[8:11] = scales.lin_vel * level
        noise[11:14] = scales.ang_vel * level
        noise[14] = scales.dof_pos * level
        noise[15:17] = scales.dof_vel * level
        self.add_noise = cfg.noise.add_noise
        return noise

    def _init_buffers(self):
        super()._init_buffers()
        # RotunbotVel bypasses LeggedRobot''s damping initializer.
        self._init_contact_yaw_damping()
        count = self.cfg.path.num_samples
        self.path_xy = torch.zeros(self.num_envs, count, 2, device=self.device)
        self.path_yaw = torch.zeros(self.num_envs, count, device=self.device)
        self.path_curvature = torch.zeros(
            self.num_envs, count, device=self.device
        )
        self.path_last_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.path_index = torch.zeros_like(self.path_last_index)
        self.path_s = torch.zeros(self.num_envs, device=self.device)
        self.last_path_s = torch.zeros_like(self.path_s)
        self.path_length = torch.zeros_like(self.path_s)
        self.path_cross_track = torch.zeros_like(self.path_s)
        self.path_heading_error = torch.zeros_like(self.path_s)
        self.path_endpoint_distance = torch.zeros_like(self.path_s)
        self.path_remaining = torch.zeros_like(self.path_s)
        self.path_type = torch.zeros_like(self.path_last_index)
        # Signed peak curvature the current path was generated from; zero for
        # straight.  Recorded per episode so evaluations can break success down
        # by (path type x curvature) instead of averaging the hardest curvature
        # into its path-type mean.
        self.path_curvature_value = torch.zeros(
            self.num_envs, device=self.device
        )
        self.success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.policy_residual = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        self.path_preview_obs = torch.zeros(
            self.num_envs, self.cfg.env.num_path_obs, device=self.device
        )
        self.terminal_cross_track = torch.zeros_like(self.path_s)
        self.terminal_endpoint_distance = torch.zeros_like(self.path_s)
        self.terminal_speed = torch.zeros_like(self.path_s)
        self.terminal_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.terminal_success = torch.zeros_like(self.success_buf)
        self.terminal_path_type = torch.zeros_like(self.path_type)
        self.terminal_path_length = torch.zeros_like(self.path_length)
        self.terminal_path_curvature = torch.zeros_like(self.path_curvature_value)
        self.terminal_reason = torch.zeros_like(self.path_type)

        # Replace inherited history buffers explicitly so the dimensions are
        # guaranteed to match this task even when parent configs change.
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    device=self.device,
                )
            )
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    device=self.device,
                )
            )
        self.num_single_obs = self.cfg.env.num_single_obs
        self.num_short_obs = (
            self.cfg.env.short_frame_stack * self.cfg.env.num_single_obs
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        yaw = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        half = 0.5 * yaw
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 2] = torch.sin(half)
        quat[:, 3] = torch.cos(half)
        self.root_states[env_ids, 3:7] = quat
        self.root_states[env_ids, 7:13] = 0.0
        ids32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32),
            len(ids32),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) > 0:
            self.commands[env_ids] = 0.0

    def _path_length_range(self):
        return getattr(
            self.cfg.path,
            f"length_range_stage{self.path_curriculum_stage}",
        )

    def _layout_split(self):
        """Which side of the generalisation split this process generates.

        ``train`` (default) reproduces the historical behaviour exactly: the
        per-stage curvature lists.  ``test`` swaps in ``heldout_curvature_values``,
        whose values never occur in any stage list, so a policy evaluated under
        it is measured on unseen curvature.
        """
        return os.environ.get("PATH_LAYOUT_SPLIT", "train").strip().lower()

    def _curvature_values(self):
        if self._layout_split() == "test":
            return self.cfg.path.heldout_curvature_values
        stage = max(1, self.path_curriculum_stage)
        return getattr(self.cfg.path, f"curvature_values_stage{stage}")

    def _s_curvature_values(self):
        if self._layout_split() == "test":
            return self.cfg.path.heldout_s_curvature_values
        stage = max(1, self.path_curriculum_stage)
        return getattr(
            self.cfg.path,
            f"s_curvature_values_stage{stage}",
            self._curvature_values(),
        )

    def _generate_paths(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        ds = float(self.cfg.path.sample_spacing)
        sample_count = int(self.cfg.path.num_samples)

        length_range = self._path_length_range()
        minimum_steps = int(round(length_range[0] / ds))
        maximum_steps = int(round(length_range[1] / ds))
        last_index = torch.randint(
            minimum_steps,
            maximum_steps + 1,
            (n,),
            device=self.device,
        )
        self.path_last_index[env_ids] = last_index
        self.path_length[env_ids] = last_index.float() * ds

        stage = int(self.path_curriculum_stage)
        path_type = torch.zeros(n, dtype=torch.long, device=self.device)
        if stage >= 1:
            straight_probability = float(
                getattr(self.cfg.path, f"straight_probability_stage{stage}")
            )
            s_probability = float(
                getattr(self.cfg.path, f"s_curve_probability_stage{stage}")
            )
            draw = torch.rand(n, device=self.device)
            s_mask = draw >= (1.0 - s_probability)
            arc_mask = (draw >= straight_probability) & ~s_mask
            direction = torch.where(
                torch.rand(n, device=self.device) < 0.5,
                -torch.ones(n, device=self.device),
                torch.ones(n, device=self.device),
            )
            path_type[arc_mask & (direction > 0)] = 1
            path_type[arc_mask & (direction < 0)] = 2
            path_type[s_mask] = 3
        forced_type = getattr(self, "forced_path_type", None)
        if forced_type is not None:
            path_type.fill_(int(forced_type))
        self.path_type[env_ids] = path_type

        curvature = torch.zeros(n, sample_count, device=self.device)
        # Signed peak curvature, 0 for straight; recorded per episode.
        curvature_value = torch.zeros(n, device=self.device)
        if stage >= 1:
            values = torch.tensor(self._curvature_values(), device=self.device)
            magnitude = values[
                torch.randint(0, len(values), (n,), device=self.device)
            ]
            forced_curvature = getattr(self, "forced_curvature", None)
            if forced_curvature is not None:
                magnitude.fill_(abs(float(forced_curvature)))
            sign = torch.where(
                path_type == 2,
                -torch.ones_like(magnitude),
                torch.ones_like(magnitude),
            )
            constant = magnitude * sign
            arc_mask = (path_type == 1) | (path_type == 2)
            curvature[arc_mask] = constant[arc_mask, None]
            curvature_value = torch.where(arc_mask, constant, curvature_value)

            # Long constant-curvature paths can loop behind the robot and make
            # nearest-point projection ambiguous.  NeuPAN supplies local paths,
            # so cap a single arc at 120 degrees and use composite paths for
            # longer manoeuvres.
            max_angle = float(self.cfg.path.maximum_constant_arc_angle)
            max_arc_steps = torch.floor(
                max_angle / torch.clamp(magnitude * ds, min=1.0e-6)
            ).long()
            last_index[arc_mask] = torch.minimum(
                last_index[arc_mask], max_arc_steps[arc_mask]
            )
            self.path_last_index[env_ids] = last_index
            self.path_length[env_ids] = last_index.float() * ds

            s_mask = path_type == 3
            if torch.any(s_mask):
                s_values = torch.tensor(
                    self._s_curvature_values(),
                    device=self.device,
                )
                s_magnitude = s_values[
                    torch.randint(0, len(s_values), (n,), device=self.device)
                ]
                if forced_curvature is not None:
                    s_magnitude.fill_(abs(float(forced_curvature)))
                s_grid = (
                    torch.arange(sample_count, device=self.device).float()[None, :]
                    * ds
                )
                phase = 2.0 * math.pi * s_grid / self.path_length[env_ids, None]
                s_sign = torch.where(
                    torch.rand(n, device=self.device) < 0.5,
                    -torch.ones(n, device=self.device),
                    torch.ones(n, device=self.device),
                )
                s_curve_value = s_magnitude * s_sign
                curvature[s_mask] = (
                    s_curve_value[s_mask, None] * torch.sin(phase[s_mask])
                )
                curvature_value = torch.where(s_mask, s_curve_value, curvature_value)
                variable_probability = float(
                    os.environ.get("PATH_VARIABLE_CURVE_PROBABILITY", "0.0")
                )
                if variable_probability > 0.0:
                    variable_mask = s_mask & (
                        torch.rand(n, device=self.device) < variable_probability
                    )
                    phase1 = 2.0 * math.pi * torch.rand(n, 1, device=self.device)
                    phase2 = 2.0 * math.pi * torch.rand(n, 1, device=self.device)
                    wave = torch.sin(phase + phase1) + 0.35 * torch.sin(
                        2.0 * phase + phase2
                    )
                    wave_peak = torch.clamp(
                        torch.max(torch.abs(wave), dim=1).values, min=1.0e-6
                    )
                    curvature[variable_mask] = (
                        s_magnitude[variable_mask, None]
                        * wave[variable_mask]
                        / wave_peak[variable_mask, None]
                    )

        self.path_curvature_value[env_ids] = curvature_value

        point_index = torch.arange(sample_count, device=self.device)[None, :]
        valid_points = point_index <= last_index[:, None]
        curvature = torch.where(valid_points, curvature, torch.zeros_like(curvature))

        segment_active = (
            torch.arange(sample_count - 1, device=self.device)[None, :]
            < last_index[:, None]
        )
        yaw_local = torch.zeros(n, sample_count, device=self.device)
        yaw_local[:, 1:] = torch.cumsum(
            curvature[:, :-1] * ds * segment_active.float(), dim=1
        )
        midpoint_yaw = 0.5 * (yaw_local[:, :-1] + yaw_local[:, 1:])
        dx = torch.cos(midpoint_yaw) * ds * segment_active.float()
        dy = torch.sin(midpoint_yaw) * ds * segment_active.float()
        local_xy = torch.zeros(n, sample_count, 2, device=self.device)
        local_xy[:, 1:, 0] = torch.cumsum(dx, dim=1)
        local_xy[:, 1:, 1] = torch.cumsum(dy, dim=1)

        q = self.root_states[env_ids, 3:7]
        root_yaw = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
        )
        lateral = torch.zeros(n, device=self.device)
        heading_offset = torch.zeros(n, device=self.device)
        if stage >= int(self.cfg.path.initial_offset_stage):
            lateral_limit = float(self.cfg.path.initial_lateral_offset)
            heading_limit = float(self.cfg.path.initial_heading_offset)
            lateral = torch_rand_float(
                -lateral_limit, lateral_limit, (n, 1), device=self.device
            ).squeeze(1)
            heading_offset = torch_rand_float(
                -heading_limit, heading_limit, (n, 1), device=self.device
            ).squeeze(1)

        start_yaw = root_yaw - heading_offset
        c = torch.cos(start_yaw)
        s = torch.sin(start_yaw)
        start_xy = self.root_states[env_ids, :2].clone()
        start_xy[:, 0] += s * lateral
        start_xy[:, 1] -= c * lateral

        world_xy = torch.empty_like(local_xy)
        world_xy[:, :, 0] = (
            start_xy[:, 0:1]
            + c[:, None] * local_xy[:, :, 0]
            - s[:, None] * local_xy[:, :, 1]
        )
        world_xy[:, :, 1] = (
            start_xy[:, 1:2]
            + s[:, None] * local_xy[:, :, 0]
            + c[:, None] * local_xy[:, :, 1]
        )
        self.path_xy[env_ids] = world_xy
        self.path_yaw[env_ids] = start_yaw[:, None] + yaw_local
        self.path_curvature[env_ids] = curvature
        self.path_index[env_ids] = 0
        self.path_s[env_ids] = 0.0
        self.last_path_s[env_ids] = 0.0

    def _update_path_state(self):
        cfg = self.cfg.path
        offsets = torch.arange(
            -int(cfg.projection_back_samples),
            int(cfg.projection_forward_samples) + 1,
            device=self.device,
        )
        candidates = self.path_index[:, None] + offsets[None, :]
        candidates = torch.maximum(candidates, self.path_index[:, None])
        candidates = torch.minimum(candidates, self.path_last_index[:, None])
        batch = torch.arange(self.num_envs, device=self.device)[:, None]
        candidate_xy = self.path_xy[batch, candidates]
        distance2 = torch.sum(
            (candidate_xy - self.root_states[:, None, :2]).square(), dim=-1
        )
        best = torch.argmin(distance2, dim=1)
        selected = candidates[torch.arange(self.num_envs, device=self.device), best]
        self.path_index = torch.maximum(self.path_index, selected)

        batch1 = torch.arange(self.num_envs, device=self.device)
        nearest_xy = self.path_xy[batch1, self.path_index]
        nearest_yaw = self.path_yaw[batch1, self.path_index]
        delta = self.root_states[:, :2] - nearest_xy
        normal = torch.stack((-torch.sin(nearest_yaw), torch.cos(nearest_yaw)), dim=1)
        self.path_cross_track = torch.sum(delta * normal, dim=1)

        q = self.root_states[:, 3:7]
        base_yaw = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
        )
        self.path_heading_error = wrap_to_pi(base_yaw - nearest_yaw)
        ds = float(cfg.sample_spacing)
        self.path_s = self.path_index.float() * ds
        self.path_remaining = torch.clamp(self.path_length - self.path_s, min=0.0)
        endpoint = self.path_xy[batch1, self.path_last_index]
        self.path_endpoint_distance = torch.linalg.vector_norm(
            self.root_states[:, :2] - endpoint, dim=1
        )

        preview_steps = torch.tensor(
            [round(value / ds) for value in cfg.preview_distances],
            dtype=torch.long,
            device=self.device,
        )
        preview_index = self.path_index[:, None] + preview_steps[None, :]
        preview_index = torch.minimum(preview_index, self.path_last_index[:, None])
        preview_xy = self.path_xy[batch, preview_index]
        preview_yaw = self.path_yaw[batch, preview_index]
        preview_delta = preview_xy - self.root_states[:, None, :2]
        c = torch.cos(base_yaw)[:, None]
        s = torch.sin(base_yaw)[:, None]
        local_x = c * preview_delta[:, :, 0] + s * preview_delta[:, :, 1]
        local_y = -s * preview_delta[:, :, 0] + c * preview_delta[:, :, 1]
        heading_delta = preview_yaw - base_yaw[:, None]
        preview = torch.stack(
            (
                local_x * float(cfg.preview_position_scale),
                local_y * float(cfg.preview_position_scale),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
            ),
            dim=-1,
        )
        goal_delta = endpoint - self.root_states[:, :2]
        goal_local_x = c[:, 0] * goal_delta[:, 0] + s[:, 0] * goal_delta[:, 1]
        goal_local_y = -s[:, 0] * goal_delta[:, 0] + c[:, 0] * goal_delta[:, 1]
        goal_features = torch.stack(
            (
                goal_local_x * float(cfg.preview_position_scale),
                goal_local_y * float(cfg.preview_position_scale),
                self.path_remaining * 0.20,
            ),
            dim=1,
        )
        preview_parts = [preview.reshape(self.num_envs, -1), goal_features]
        if self.cfg.path.include_curvature_observation:
            preview_parts.append((2.0 * self._lookahead_curvature())[:, None])
        self.path_preview_obs = torch.cat(preview_parts, dim=1)

    def _update_trajectory_quantities(self):
        # RotunbotVel calls this optional hook.  Path state is updated in the
        # task callback after root/body state refresh instead.
        return None

    def _post_physics_step_callback(self):
        self._update_path_state()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % int(self.cfg.domain_rand.push_interval) == 0
        ):
            self._push_robots()

    def compute_observations(self):
        self.lagged_base_lin_vel = self.base_lin_vel
        self.lagged_base_ang_vel = self.base_ang_vel
        self.lagged_dof_pos = self.dof_pos
        self.lagged_dof_vel = self.dof_vel

        batch = torch.arange(self.num_envs, device=self.device)
        nearest = self.path_xy[batch, self.path_index]
        q = self.root_states[:, 3:7]
        base_yaw = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
        )
        delta = nearest - self.root_states[:, :2]
        c = torch.cos(base_yaw)
        s = torch.sin(base_yaw)
        nearest_local = torch.stack(
            (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]),
            dim=1,
        ) * float(self.cfg.path.preview_position_scale)
        tangent_delta = -self.path_heading_error
        anchor = torch.cat(
            (
                nearest_local,
                torch.cos(tangent_delta)[:, None],
                torch.sin(tangent_delta)[:, None],
            ),
            dim=1,
        )

        obs_now = torch.cat(
            (
                anchor,
                self.base_quat,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.dof_pos[:, 1:2] * self.obs_scales.dof_pos,
                self.dof_vel[:, :2] * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=1,
        )
        if self.add_noise:
            obs_now = obs_now + (
                2.0 * torch.rand_like(obs_now) - 1.0
            ) * self.noise_scale_vec
        self.obs_history.append(obs_now)
        history = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )
        self.obs_buf = torch.cat((history, self.path_preview_obs), dim=1)

        type_one_hot = torch.nn.functional.one_hot(
            self.path_type, num_classes=4
        ).float()
        current_curvature = self.path_curvature[batch, self.path_index]
        privileged = torch.cat(
            (
                self.path_s[:, None],
                self.path_remaining[:, None],
                self.path_cross_track[:, None],
                torch.sin(self.path_heading_error)[:, None],
                torch.cos(self.path_heading_error)[:, None],
                self.root_states[:, 2:3],
                self.root_states[:, 7:10],
                self.root_states[:, 10:13],
                self.projected_gravity,
                self.dof_pos[:, :2],
                self.dof_vel[:, :2],
                self.actions,
                self.output_actions,
                current_curvature[:, None],
                type_one_hot,
            ),
            dim=1,
        )
        if privileged.shape[1] != self.cfg.env.single_num_privileged_obs:
            raise RuntimeError(
                f"Privileged observation has {privileged.shape[1]} values; "
                f"expected {self.cfg.env.single_num_privileged_obs}"
            )
        self.critic_history.append(privileged)
        self.privileged_obs_buf = torch.cat(list(self.critic_history), dim=1)

    def _analytic_pure_pursuit_prior(self):
        """Measured action map driven by a geometric pure-pursuit curvature."""
        cfg = self.cfg.path
        ds = float(cfg.sample_spacing)
        lookahead_m = float(getattr(cfg, "prior_pp_lookahead", 1.20))
        lookahead_steps = max(1, int(round(lookahead_m / ds)))
        index = torch.minimum(
            self.path_index + lookahead_steps, self.path_last_index
        )
        batch = torch.arange(self.num_envs, device=self.device)
        target = self.path_xy[batch, index]
        delta = target - self.root_states[:, :2]
        q = self.root_states[:, 3:7]
        base_yaw = torch.atan2(
            2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
        )
        c = torch.cos(base_yaw)
        s = torch.sin(base_yaw)
        local_x = c * delta[:, 0] + s * delta[:, 1]
        local_y = -s * delta[:, 0] + c * delta[:, 1]
        distance2 = torch.clamp(local_x.square() + local_y.square(), min=0.04)
        max_curvature = float(getattr(cfg, "prior_pp_max_curvature", 0.50))
        curvature_command = torch.clamp(
            2.0 * local_y / distance2, -max_curvature, max_curvature
        )

        # Identification showed that the curvature authority grows with joint-1
        # drive: |k|max ~= 0.092 + 0.92*|a1|.  Use only the drive required by
        # the local geometric command instead of the peak of a 10 m window.
        minimum_drive = float(getattr(cfg, "prior_pp_min_drive", 0.22))
        drive_margin = float(getattr(cfg, "prior_pp_drive_margin", 0.03))
        drive = torch.clamp(
            (torch.abs(curvature_command) - float(cfg.prior_gain_offset))
            / float(cfg.prior_gain_slope)
            + drive_margin,
            min=minimum_drive,
            max=float(getattr(cfg, "prior_pp_max_drive", 0.48)),
        )
        stop_distance = torch.clamp(
            float(getattr(cfg, "prior_pp_base_stop_distance", 0.90))
            + float(getattr(cfg, "prior_pp_curvature_stop_gain", 3.0))
            * torch.abs(curvature_command),
            max=float(getattr(cfg, "prior_pp_max_stop_distance", 2.40)),
        )
        distance_scale = torch.clamp(self.path_remaining / stop_distance, 0.0, 1.0)
        desired_speed = 1.18 * drive * distance_scale
        forward_speed = self.base_lin_vel[:, 0]
        first = desired_speed / 1.18 + float(cfg.prior_speed_kp) * (
            desired_speed - forward_speed
        )
        curvature_gain = torch.clamp(
            float(cfg.prior_gain_offset) + float(cfg.prior_gain_slope) * torch.abs(first),
            float(cfg.prior_gain_min), float(cfg.prior_gain_max),
        )
        second = -curvature_command / curvature_gain
        return torch.stack((first, second), dim=1).clamp(-1.0, 1.0)

    def _analytic_action_prior(self):
        cfg = self.cfg.path
        if bool(getattr(cfg, "prior_use_pure_pursuit", False)):
            return self._analytic_pure_pursuit_prior()
        peak_curvature = torch.max(torch.abs(self.path_curvature), dim=1).values
        reference_curvature = self._lookahead_curvature()
        schedule_curvature = (
            torch.abs(reference_curvature)
            if bool(getattr(cfg, "prior_use_local_curvature_schedule", False))
            else peak_curvature
        )
        tight = schedule_curvature >= float(cfg.prior_tight_curvature)
        drive = torch.where(
            tight,
            torch.full_like(self.path_length, float(cfg.prior_tight_drive)),
            torch.full_like(self.path_length, float(cfg.prior_normal_drive)),
        )
        stop_distance = torch.where(
            tight,
            torch.full_like(self.path_length, float(cfg.prior_tight_stop_distance)),
            torch.full_like(self.path_length, float(cfg.prior_normal_stop_distance)),
        )
        cross_gain = torch.where(
            tight,
            torch.full_like(self.path_length, float(cfg.prior_tight_cross_track_kp)),
            torch.full_like(self.path_length, float(cfg.prior_normal_cross_track_kp)),
        )
        heading_gain = torch.where(
            tight,
            torch.full_like(self.path_length, float(cfg.prior_tight_heading_kp)),
            torch.full_like(self.path_length, float(cfg.prior_normal_heading_kp)),
        )
        # R2.5 arcs need a slower drive only after 4.6 m; shorter arcs use the
        # R2 profile. S-curves need more preview because curvature changes sign.
        moderate_long = (
            (schedule_curvature >= float(cfg.prior_tight_curvature))
            & (schedule_curvature < float(cfg.prior_r2_curvature))
            & (self.path_length > float(cfg.prior_moderate_long_length))
            & (self.path_type != 3)
        )
        drive = torch.where(moderate_long, torch.full_like(drive, float(cfg.prior_moderate_long_drive)), drive)
        stop_distance = torch.where(moderate_long, torch.full_like(stop_distance, float(cfg.prior_moderate_long_stop_distance)), stop_distance)
        cross_gain = torch.where(moderate_long, torch.full_like(cross_gain, float(cfg.prior_moderate_long_cross_track_kp)), cross_gain)
        heading_gain = torch.where(moderate_long, torch.full_like(heading_gain, float(cfg.prior_moderate_long_heading_kp)), heading_gain)
        s_curve = self.path_type == 3
        tight_s = s_curve & (schedule_curvature >= float(cfg.prior_r2_curvature))
        tight_s_long = tight_s & (self.path_length > float(cfg.prior_s_tight_short_length))
        normal_s_drive = torch.where(
            self.path_length <= float(cfg.prior_s_short_length),
            torch.full_like(drive, float(cfg.prior_s_short_drive)),
            torch.full_like(drive, float(cfg.prior_s_long_drive)),
        )
        tight_s_drive = torch.where(
            tight_s_long,
            torch.full_like(drive, float(cfg.prior_s_tight_long_drive)),
            torch.full_like(drive, float(cfg.prior_s_tight_short_drive)),
        )
        s_drive = torch.where(tight_s, tight_s_drive, normal_s_drive)
        drive = torch.where(s_curve, s_drive, drive)
        normal_s_stop = torch.full_like(stop_distance, float(cfg.prior_s_stop_distance))
        tight_s_stop = torch.full_like(stop_distance, float(cfg.prior_s_tight_stop_distance))
        stop_distance = torch.where(s_curve, torch.where(tight_s, tight_s_stop, normal_s_stop), stop_distance)
        normal_s_cross = torch.full_like(cross_gain, float(cfg.prior_s_cross_track_kp))
        curvature_batch = torch.arange(self.num_envs, device=self.device)
        boundary_activity = (
            torch.abs(self.path_curvature[:, 0])
            + torch.abs(self.path_curvature[curvature_batch, self.path_last_index])
        )
        complex_s = s_curve & (
            boundary_activity >= float(cfg.prior_s_complex_boundary_threshold)
        )
        tight_long_cross = torch.where(
            complex_s,
            torch.full_like(cross_gain, float(cfg.prior_s_tight_long_cross_track_kp)),
            torch.full_like(cross_gain, float(cfg.prior_s_regular_long_cross_track_kp)),
        )
        tight_s_cross = torch.where(
            tight_s_long,
            tight_long_cross,
            torch.full_like(cross_gain, float(cfg.prior_s_tight_short_cross_track_kp)),
        )
        cross_gain = torch.where(s_curve, torch.where(tight_s, tight_s_cross, normal_s_cross), cross_gain)
        normal_s_heading = torch.full_like(heading_gain, float(cfg.prior_s_heading_kp))
        tight_long_heading = torch.where(
            complex_s,
            torch.full_like(heading_gain, float(cfg.prior_s_tight_long_heading_kp)),
            torch.full_like(heading_gain, float(cfg.prior_s_regular_long_heading_kp)),
        )
        tight_s_heading = torch.where(
            tight_s_long,
            tight_long_heading,
            torch.full_like(heading_gain, float(cfg.prior_s_tight_short_heading_kp)),
        )
        heading_gain = torch.where(s_curve, torch.where(tight_s, tight_s_heading, normal_s_heading), heading_gain)
        # Variable-curvature paths near k=0.4 need more steering authority than
        # the low-curvature S profile, while still retaining heading feedback.
        mid_s = (
            s_curve
            & (schedule_curvature >= float(cfg.prior_s_mid_curvature))
            & (schedule_curvature < float(cfg.prior_r2_curvature))
        )
        drive = torch.where(
            mid_s, torch.full_like(drive, float(cfg.prior_s_mid_drive)), drive
        )
        stop_distance = torch.where(
            mid_s,
            torch.full_like(stop_distance, float(cfg.prior_s_mid_stop_distance)),
            stop_distance,
        )
        cross_gain = torch.where(
            mid_s,
            torch.full_like(cross_gain, float(cfg.prior_s_mid_cross_track_kp)),
            cross_gain,
        )
        heading_gain = torch.where(
            mid_s,
            torch.full_like(heading_gain, float(cfg.prior_s_mid_heading_kp)),
            heading_gain,
        )
        # Arc-length progress is monotone.  Euclidean endpoint distance grows
        # again after an overshoot and used to make the robot accelerate away.
        distance_scale = torch.clamp(self.path_remaining / stop_distance, 0.0, 1.0)
        desired_speed = (1.18 * drive) * distance_scale
        forward_speed = self.base_lin_vel[:, 0]
        first = desired_speed / 1.18 + float(cfg.prior_speed_kp) * (
            desired_speed - forward_speed
        )
        curvature_command = (
            reference_curvature
            - cross_gain * self.path_cross_track
            - heading_gain * self.path_heading_error
        )
        if bool(getattr(cfg, "prior_endpoint_pure_pursuit", False)):
            batch = torch.arange(self.num_envs, device=self.device)
            endpoint = self.path_xy[batch, self.path_last_index]
            goal_delta = endpoint - self.root_states[:, :2]
            q = self.root_states[:, 3:7]
            base_yaw = torch.atan2(
                2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
            )
            c = torch.cos(base_yaw)
            s = torch.sin(base_yaw)
            goal_x = c * goal_delta[:, 0] + s * goal_delta[:, 1]
            goal_y = -s * goal_delta[:, 0] + c * goal_delta[:, 1]
            goal_distance2 = torch.clamp(goal_x.square() + goal_y.square(), min=0.04)
            goal_curvature = torch.clamp(
                2.0 * goal_y / goal_distance2,
                -float(cfg.prior_endpoint_max_curvature),
                float(cfg.prior_endpoint_max_curvature),
            )
            blend_distance = float(cfg.prior_endpoint_blend_distance)
            endpoint_blend = torch.clamp(
                1.0 - self.path_remaining / blend_distance, 0.0, 1.0
            )
            curvature_command = (
                (1.0 - endpoint_blend) * curvature_command
                + endpoint_blend * goal_curvature
            )
        curvature_gain = torch.clamp(
            float(cfg.prior_gain_offset) + float(cfg.prior_gain_slope) * torch.abs(first),
            float(cfg.prior_gain_min), float(cfg.prior_gain_max),
        )
        second = -curvature_command / curvature_gain
        return torch.stack((first, second), dim=1).clamp(-1.0, 1.0)

    def step(self, actions):
        self.last_path_s.copy_(self.path_s)
        self.policy_residual.copy_(actions)
        if self.cfg.control.use_path_action_prior:
            prior = self._analytic_action_prior()
            actions = torch.stack((
                prior[:, 0] + float(self.cfg.path.prior_residual_scale_1) * actions[:, 0],
                prior[:, 1] + float(self.cfg.path.prior_residual_scale_2) * actions[:, 1],
            ), dim=1)
        return super().step(actions)

    def _curriculum_required_types(self):
        forced_type = getattr(self, "forced_path_type", None)
        if forced_type is not None:
            return [int(forced_type)]
        return [0] if self.path_curriculum_stage == 0 else (
            [0, 1, 2] if self.path_curriculum_stage < 5 else [0, 1, 2, 3]
        )

    def _update_path_curriculum(self, completed_success, completed_types):
        # Keep statistics live for fixed-path specialist runs.
        if self.common_step_counter <= 0:
            return
        self.path_curriculum_attempts += int(completed_success.numel())
        self.path_curriculum_successes += int(completed_success.sum().item())
        for path_type in range(4):
            mask = completed_types == path_type
            self.path_curriculum_type_attempts[path_type] += int(mask.sum().item())
            self.path_curriculum_type_successes[path_type] += int(
                completed_success[mask].sum().item()
            )
        if self.path_curriculum_attempts < int(self.cfg.path.curriculum_window):
            return

        self.path_curriculum_last_rate = self.path_curriculum_successes / max(
            self.path_curriculum_attempts, 1
        )
        for path_type in range(4):
            attempts = self.path_curriculum_type_attempts[path_type]
            self.path_curriculum_last_rates[path_type] = (
                self.path_curriculum_type_successes[path_type] / max(attempts, 1)
            )
        required = self._curriculum_required_types()
        threshold = float(self.cfg.path.curriculum_success_rate)
        minimum = int(self.cfg.path.curriculum_min_type_attempts)
        ready = all(
            self.path_curriculum_type_attempts[t] >= minimum
            and self.path_curriculum_last_rates[t] >= threshold
            for t in required
        )
        maximum = int(self.cfg.path.curriculum_max_stage)
        rates = ", ".join(
            f"type{t}={self.path_curriculum_last_rates[t]:.1%}"
            for t in required
        )
        if self.cfg.path.curriculum_enabled and ready and self.path_curriculum_stage < maximum:
            self.path_curriculum_stage += 1
            print(
                f"[Path curriculum] advanced to stage {self.path_curriculum_stage}/{maximum}; "
                + rates
            )
        elif self.cfg.path.curriculum_enabled:
            print(
                f"[Path curriculum] held at stage {self.path_curriculum_stage}/{maximum}; "
                + rates
            )
        else:
            print(
                f"[Path statistics] fixed stage {self.path_curriculum_stage}; "
                f"overall={self.path_curriculum_last_rate:.1%}; " + rates
            )
        self.path_curriculum_attempts = 0
        self.path_curriculum_successes = 0
        self.path_curriculum_type_attempts = [0, 0, 0, 0]
        self.path_curriculum_type_successes = [0, 0, 0, 0]

    def reset_idx(self, env_ids):
        completed_success = None
        completed_types = None
        if len(env_ids) > 0:
            completed_success = self.success_buf[env_ids].detach().clone()
            completed_types = self.path_type[env_ids].detach().clone()
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if completed_success is not None:
            self._update_path_curriculum(completed_success, completed_types)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.output_actions[env_ids] = 0.0
        self.last_output_actions[env_ids] = 0.0
        self.policy_residual[env_ids] = 0.0
        self.path_preview_obs[env_ids] = 0.0
        self._generate_paths(env_ids)
        self._update_path_state()
        for history in self.obs_history:
            history[env_ids] = 0.0
        for history in self.critic_history:
            history[env_ids] = 0.0

        if "episode" in self.extras:
            self.extras["episode"]["path_stage"] = float(
                self.path_curriculum_stage
            )
            self.extras["episode"]["path_success_rate"] = float(
                self.path_curriculum_last_rate
            )

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        speed = torch.linalg.vector_norm(self.root_states[:, 7:9], dim=1)
        self.success_buf = (
            (self.path_remaining <= self.cfg.path.success_remaining_length)
            & (self.path_endpoint_distance <= self.cfg.path.success_endpoint_distance)
            & (speed <= self.cfg.path.success_speed)
        )
        deviated = torch.abs(self.path_cross_track) > self.cfg.path.deviation_termination
        unstable = self.projected_gravity[:, 2] > self.cfg.path.unstable_gravity_z
        out_of_bounds = torch.linalg.vector_norm(
            self.root_states[:, :2] - self.env_origins[:, :2], dim=1
        ) > float(getattr(self.cfg.path, "out_of_bounds_radius", 15.0))

        self.terminal_cross_track.copy_(torch.abs(self.path_cross_track))
        self.terminal_endpoint_distance.copy_(self.path_endpoint_distance)
        self.terminal_speed.copy_(speed)
        self.terminal_position.copy_(self.root_states[:, :2])
        self.terminal_success.copy_(self.success_buf)
        self.terminal_path_type.copy_(self.path_type)
        self.terminal_path_length.copy_(self.path_length)
        self.terminal_path_curvature.copy_(self.path_curvature_value)
        self.terminal_reason.zero_()
        self.terminal_reason = torch.where(self.time_out_buf, 1, self.terminal_reason)
        self.terminal_reason = torch.where(deviated, 2, self.terminal_reason)
        self.terminal_reason = torch.where(unstable, 3, self.terminal_reason)
        self.terminal_reason = torch.where(out_of_bounds, 4, self.terminal_reason)
        self.reset_buf = (
            self.time_out_buf | self.success_buf | deviated | unstable | out_of_bounds
        )

    def _reward_path_progress(self):
        progress_rate = (self.path_s - self.last_path_s) / max(self.dt, 1.0e-6)
        # V1 rewarded projection progress even while cutting far across a curve.
        # Gate progress by tracking quality so leaving the path is never the
        # fastest way to collect progress reward.
        cross_quality = torch.exp(-self.path_cross_track.square() / 0.0625)
        heading_quality = torch.exp(
            -(1.0 - torch.cos(self.path_heading_error)) / 0.25
        )
        return torch.clamp(progress_rate, 0.0, 1.0) * cross_quality * heading_quality

    def _reward_path_tracking(self):
        return self.path_cross_track.square()

    def _reward_path_heading(self):
        active = (self.path_remaining > 0.30).float()
        return active * (1.0 - torch.cos(self.path_heading_error))

    def _reward_completion(self):
        return 100.0 * self.success_buf.float()

    def _reward_failure_terminal(self):
        failed = self.reset_buf & ~self.success_buf
        return 100.0 * failed.float()

    def _reward_policy_residual(self):
        return torch.sum(self.policy_residual.square(), dim=1)

    def _reward_reverse_motion(self):
        batch = torch.arange(self.num_envs, device=self.device)
        tangent_yaw = self.path_yaw[batch, self.path_index]
        along_speed = (
            torch.cos(tangent_yaw) * self.root_states[:, 7]
            + torch.sin(tangent_yaw) * self.root_states[:, 8]
        )
        active = (self.path_remaining > 0.30).float()
        return active * torch.relu(-along_speed).square()

    def _reward_near_end_speed(self):
        near = (
            (self.path_remaining <= self.cfg.rewards.near_end_distance)
            & (self.path_endpoint_distance <= self.cfg.rewards.near_end_distance)
        ).float()
        speed = torch.linalg.vector_norm(self.root_states[:, 7:9], dim=1)
        return near * speed.square()

    def _reward_path_deviation(self):
        return self.path_cross_track.square()

    def _lookahead_curvature(self):
        ds = float(self.cfg.path.sample_spacing)
        arc_steps = max(1, int(round(0.40 / ds)))
        peak_curvature = torch.max(torch.abs(self.path_curvature), dim=1).values
        tight_s = (self.path_type == 3) & (
            peak_curvature >= float(self.cfg.path.prior_r2_curvature)
        )
        tight_distance = torch.maximum(
            torch.full_like(self.path_length, float(self.cfg.path.prior_s_tight_min_lookahead)),
            float(self.cfg.path.prior_s_tight_lookahead_fraction) * self.path_length,
        )
        s_distance = torch.where(
            tight_s,
            tight_distance,
            torch.full_like(self.path_length, float(self.cfg.path.prior_s_lookahead)),
        )
        s_steps = torch.clamp(torch.round(s_distance / ds).long(), min=1)
        lookahead = torch.where(
            self.path_type == 3,
            s_steps,
            torch.full_like(self.path_index, arc_steps),
        )
        index = torch.minimum(self.path_index + lookahead, self.path_last_index)
        batch = torch.arange(self.num_envs, device=self.device)
        return self.path_curvature[batch, index]

    def _reward_steering_direction(self):
        desired_curvature = self._lookahead_curvature()
        curved = torch.abs(desired_curvature) >= 0.05
        target = self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        desired_sign = torch.sign(desired_curvature)
        wrong_sign = torch.relu(desired_sign * target)
        return curved.float() * wrong_sign.square()

    def _reward_steering_feedforward(self):
        curvature = self._lookahead_curvature()
        target = self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        nominal = torch.clamp(-2.5 * curvature, -0.90, 0.90)
        valid = (
            (torch.abs(curvature) >= 0.05)
            & (torch.abs(self.path_cross_track) <= 0.25)
            & (torch.abs(self.path_heading_error) <= 0.35)
            & (self.path_remaining > 0.50)
        )
        return valid.float() * (target - nominal).square()

    def _reward_steering_centering(self):
        desired_curvature = self._lookahead_curvature()
        straight = torch.abs(desired_curvature) < 0.05
        target = self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        actual = self.dof_pos[:, 1] / self.cfg.control.second_pos_limits
        return straight.float() * (target.square() + 0.25 * actual.square())

    def _reward_steering_saturation(self):
        target = self.output_actions[:, 1] / self.cfg.control.second_pos_limits
        return torch.relu(torch.abs(target) - 0.90).square()
    def _reward_time(self):
        return torch.ones(self.num_envs, device=self.device)

    def get_checkpoint_state(self):
        return {
            "path_curriculum_stage": int(self.path_curriculum_stage),
            "path_curriculum_attempts": int(self.path_curriculum_attempts),
            "path_curriculum_successes": int(self.path_curriculum_successes),
            "path_curriculum_last_rate": float(self.path_curriculum_last_rate),
            "path_curriculum_type_attempts": list(self.path_curriculum_type_attempts),
            "path_curriculum_type_successes": list(self.path_curriculum_type_successes),
            "path_curriculum_last_rates": list(self.path_curriculum_last_rates),
        }

    def set_checkpoint_state(self, state):
        if not state:
            return
        self.path_curriculum_stage = int(
            state.get("path_curriculum_stage", self.path_curriculum_stage)
        )
        self.path_curriculum_attempts = int(
            state.get("path_curriculum_attempts", self.path_curriculum_attempts)
        )
        self.path_curriculum_successes = int(
            state.get("path_curriculum_successes", self.path_curriculum_successes)
        )
        self.path_curriculum_last_rate = float(
            state.get("path_curriculum_last_rate", self.path_curriculum_last_rate)
        )
        self.path_curriculum_type_attempts = list(
            state.get("path_curriculum_type_attempts", [0, 0, 0, 0])
        )
        self.path_curriculum_type_successes = list(
            state.get("path_curriculum_type_successes", [0, 0, 0, 0])
        )
        self.path_curriculum_last_rates = list(
            state.get("path_curriculum_last_rates", [0.0, 0.0, 0.0, 0.0])
        )


    def get_tracking_curriculum_status(self):
        # Path stages are logged through episode extras.
        return None














