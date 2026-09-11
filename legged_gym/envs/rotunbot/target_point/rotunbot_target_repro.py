"""Paper-reproduction LH point-to-point Rotunbot task on a flat plane."""

import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from .rotunbot_target_lh import RotunbotTargetLH
from .rotunbot_target_repro_config import RotunbotTargetReproCfg


class RotunbotTargetRepro(RotunbotTargetLH):
    """Paper task protocol while retaining the existing 19-D policy input."""

    cfg: RotunbotTargetReproCfg

    def _init_buffers(self):
        super()._init_buffers()

        # Values captured before LeggedRobot automatically resets a finished
        # environment.  play.py uses these to calculate episode metrics.
        self.terminal_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.terminal_speed = torch.zeros(self.num_envs, device=self.device)
        self.terminal_balance_reward = torch.zeros(self.num_envs, device=self.device)
        self.terminal_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.terminal_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Curriculum state: the target distribution remains the complete
        # paper range; only the training success distance is tightened.
        self.training_success_distance = float(
            getattr(
                self.cfg.commands,
                "curriculum_success_distance_start",
                self.cfg.evaluation.target_error_threshold,
            )
        )
        self.target_curriculum_successes = 0
        self.target_curriculum_attempts = 0
        self.target_curriculum_last_success_rate = 0.0

    def _reset_root_states(self, env_ids):
        """Reset the base and explicitly apply the configured initial yaw.

        The inherited LeggedRobot reset does not read
        ``cfg.commands.random_start_yaw`` and always randomizes the initial
        root velocity.  This task needs both settings to be explicit so the
        training/evaluation configuration is not silently ignored.
        """
        super()._reset_root_states(env_ids)

        if len(env_ids) == 0:
            return

        if self.cfg.commands.random_start_yaw:
            yaw = torch_rand_float(
                -math.pi,
                math.pi,
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            half_yaw = 0.5 * yaw
            quat = torch.zeros(len(env_ids), 4, device=self.device)
            # Isaac Gym/scipy use the [x, y, z, w] quaternion convention.
            quat[:, 2] = torch.sin(half_yaw)
            quat[:, 3] = torch.cos(half_yaw)
            self.root_states[env_ids, 3:7] = quat

        if not getattr(self.cfg.init_state, "randomize_initial_velocity", True):
            self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _process_dof_props(self, props, env_id):
        # RotunbotTargetLH overrides the base callback and therefore never
        # creates dof_pos_limits.  The reproduction enables dof_pos_limits reward, so
        # first preserve the URDF limits and then apply the LH drive settings.
        LeggedRobot._process_dof_props(self, props, env_id)
        return super()._process_dof_props(props, env_id)

    def reset_idx(self, env_ids):
        terminal_success = None
        if len(env_ids) > 0:
            terminal_success = self.success_buf[env_ids].detach().clone()

        super().reset_idx(env_ids)

        if terminal_success is not None:
            self._update_target_curriculum(terminal_success)

        if len(env_ids) > 0 and bool(
            getattr(self.cfg.commands, "target_curriculum", False)
        ) and "episode" in self.extras:
            self.extras["episode"]["target_range"] = self._target_sampling_range()
            self.extras["episode"]["target_stage"] = self._target_curriculum_stage()
            self.extras["episode"]["target_success_distance"] = (
                self.training_success_distance
            )
            self.extras["episode"]["target_success_rate"] = (
                self.target_curriculum_last_success_rate
            )

        if len(env_ids) > 0:
            # Do not carry the previous episode's controller target into the
            # first action of the new episode.
            self.output_actions[env_ids] = 0.0
            self.last_output_actions[env_ids] = 0.0

            # The first transition should compare the new target distance with
            # itself, rather than compare it against the reset value zero.
            self.goal_dist[env_ids] = torch.linalg.norm(
                self.commands[env_ids, :2] - self.root_states[env_ids, :2],
                dim=1,
            )
            self.last_goal_dist[env_ids] = self.goal_dist[env_ids]

        # The history must not leak observations across episodes.
        if len(env_ids) > 0 and hasattr(self, "obs_history"):
            for history in self.obs_history:
                history[env_ids] = 0.0
            for history in self.critic_history:
                history[env_ids] = 0.0

    def _get_noise_scale_vec(self, cfg):
        """Noise scales for the 19-D Graduation observation."""
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:2] = 0.0  # target position
        noise_vec[2:4] = noise_scales.pos * noise_level
        noise_vec[4:8] = noise_scales.quat * noise_level
        noise_vec[8:11] = noise_scales.lin_vel * noise_level
        noise_vec[11:14] = noise_scales.ang_vel * noise_level
        noise_vec[14] = noise_scales.dof_pos * noise_level
        noise_vec[15:17] = noise_scales.dof_vel * noise_level
        noise_vec[17:19] = 0.0  # previous actions
        return noise_vec

    def _update_base_euler(self):
        """Update XYZ Euler angles from the GPU quaternion tensor."""
        qx, qy, qz, qw = self.base_quat.unbind(dim=1)
        sin_roll = 2.0 * (qw * qx + qy * qz)
        cos_roll = 1.0 - 2.0 * (qx.square() + qy.square())
        roll = torch.atan2(sin_roll, cos_roll)

        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))

        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy.square() + qz.square())
        yaw = torch.atan2(sin_yaw, cos_yaw)
        self.base_euler_tensor[:] = torch.stack((roll, pitch, yaw), dim=1)

    def compute_observations(self):
        """Build the 19-D per-frame observation used by Graduation."""
        # Keep this conversion on the same device.  The old scipy conversion
        # copied every environment to CPU on every observation step, which
        # caused an unnecessary synchronization and slowed training/testing.
        self._update_base_euler()

        privileged_obs_buf = torch.cat(
            (
                self.commands[:, :2] * self.obs_scales.command,
                self.root_states[:, :3] * self.obs_scales.pos,
                self.base_quat * self.obs_scales.quat,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.commands[:, :2] * self.obs_scales.command,
                self.root_states[:, :2] * self.obs_scales.pos,
                self.base_quat * self.obs_scales.quat,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.dof_pos[:, 1].unsqueeze(1) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            obs_now = obs_buf + (2.0 * torch.rand_like(obs_buf) - 1.0) * self.noise_scale_vec
        else:
            obs_now = obs_buf

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat(
            list(self.critic_history)[: self.cfg.env.c_frame_stack], dim=1
        )

    def _resample_commands(self, env_ids):
        """Sample targets from the fixed full paper distribution."""
        if len(env_ids) == 0:
            return

        super()._resample_commands(env_ids)

        # The paper excludes targets within a 0.5 m radius of the origin.
        for _ in range(20):
            invalid = torch.norm(self.commands[env_ids, :2], dim=1) <= 0.5
            if not torch.any(invalid):
                break
            invalid_ids = env_ids[invalid]
            self.commands[invalid_ids, 0] = torch_rand_float(
                self.command_ranges["pos_x"][0], self.command_ranges["pos_x"][1],
                (len(invalid_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[invalid_ids, 1] = torch_rand_float(
                self.command_ranges["pos_y"][0], self.command_ranges["pos_y"][1],
                (len(invalid_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _target_sampling_range(self):
        """Return the fixed maximum absolute XY target range for logging."""
        return max(
            abs(float(self.command_ranges["pos_x"][0])),
            abs(float(self.command_ranges["pos_x"][1])),
            abs(float(self.command_ranges["pos_y"][0])),
            abs(float(self.command_ranges["pos_y"][1])),
        )

    def _current_success_distance(self):
        """Use curriculum tolerance only during training, formal tolerance in play."""
        if bool(getattr(self.cfg.commands, "target_curriculum", False)):
            return self.training_success_distance
        return float(self.cfg.evaluation.target_error_threshold)

    def _update_target_curriculum(self, terminal_success):
        """Advance target difficulty after a window of completed episodes."""
        if not bool(getattr(self.cfg.commands, "target_curriculum", False)):
            return
        # Do not count the synthetic reset performed before the first rollout.
        if self.common_step_counter <= 0:
            return

        self.target_curriculum_attempts += int(terminal_success.numel())
        self.target_curriculum_successes += int(terminal_success.sum().item())
        window = int(getattr(self.cfg.commands, "target_curriculum_window", 2048))
        if self.target_curriculum_attempts < window:
            return

        success_rate = self.target_curriculum_successes / max(
            self.target_curriculum_attempts, 1
        )
        success_threshold = float(
            getattr(self.cfg.commands, "target_curriculum_success_rate", 0.80)
        )
        minimum_distance = float(
            getattr(self.cfg.commands, "curriculum_success_distance_min", 0.20)
        )
        step = float(
            getattr(self.cfg.commands, "curriculum_success_distance_step", 0.20)
        )
        old_distance = self.training_success_distance
        if success_rate >= success_threshold:
            self.training_success_distance = max(
                minimum_distance, old_distance - step
            )

        self.target_curriculum_last_success_rate = success_rate
        if self.training_success_distance < old_distance:
            print(
                "[Target curriculum] "
                f"stage {self._target_curriculum_stage()} / "
                f"success distance <= {self.training_success_distance:.2f} m, "
                f"target range [-{self._target_sampling_range():.1f}, "
                f"{self._target_sampling_range():.1f}], "
                f"window success rate={success_rate:.2%}"
            )

        self.target_curriculum_successes = 0
        self.target_curriculum_attempts = 0
        if "episode" in self.extras:
            self.extras["episode"]["target_range"] = self._target_sampling_range()
            self.extras["episode"]["target_stage"] = self._target_curriculum_stage()
            self.extras["episode"]["target_success_distance"] = (
                self.training_success_distance
            )
            self.extras["episode"]["target_success_rate"] = success_rate

    def _target_curriculum_stage(self):
        start = float(
            getattr(self.cfg.commands, "curriculum_success_distance_start", 1.0)
        )
        step = max(
            float(
                getattr(self.cfg.commands, "curriculum_success_distance_step", 0.20)
            ),
            1.0e-6,
        )
        return int(round((start - self.training_success_distance) / step)) + 1

    def _post_physics_step_callback(self):
        """Keep one fixed target for the whole point-to-point episode."""
        # post_physics_step has refreshed base_quat immediately before this
        # callback.  Update orientation now so termination/reward calculations
        # use the current physics state rather than the previous observation.
        self._update_base_euler()
        self.goal_dist = torch.linalg.norm(
            self.commands[:, :2] - self.root_states[:, :2], dim=1
        )

        # The parent callback also resamples locomotion commands, which is not
        # appropriate here because the point-to-point target must stay fixed.
        # Preserve only its optional terrain measurement and push behavior.
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % int(self.cfg.domain_rand.push_interval) == 0
        ):
            self._push_robots()

    def get_checkpoint_state(self):
        """Return non-network state needed to resume target curriculum."""
        return {
            "training_success_distance": float(self.training_success_distance),
            "target_curriculum_successes": int(self.target_curriculum_successes),
            "target_curriculum_attempts": int(self.target_curriculum_attempts),
            "target_curriculum_last_success_rate": float(
                self.target_curriculum_last_success_rate
            ),
        }

    def set_checkpoint_state(self, state):
        """Restore target curriculum state when it exists in a checkpoint."""
        if not state:
            return
        self.training_success_distance = float(
            state.get("training_success_distance", self.training_success_distance)
        )
        self.target_curriculum_successes = int(
            state.get("target_curriculum_successes", self.target_curriculum_successes)
        )
        self.target_curriculum_attempts = int(
            state.get("target_curriculum_attempts", self.target_curriculum_attempts)
        )
        self.target_curriculum_last_success_rate = float(
            state.get(
                "target_curriculum_last_success_rate",
                self.target_curriculum_last_success_rate,
            )
        )


    def check_termination(self):
        """Implement the paper's formal success flag."""
        self.reset_buf[:] = self.episode_length_buf > self.max_episode_length
        self.time_out_buf[:] = self.episode_length_buf > self.max_episode_length

        arrived_target = self.goal_dist <= self._current_success_distance()
        stopped = torch.linalg.norm(self.base_lin_vel, dim=1) <= self.cfg.evaluation.stop_velocity_threshold
        self.arrived_target_buf = arrived_target
        self.stop_buf = stopped
        self.success_buf = arrived_target & stopped

        roll_cutoff = torch.abs(self.base_euler_tensor[:, 0]) > 1.2
        pitch_cutoff = torch.abs(self.base_euler_tensor[:, 1]) > 1.2
        x_cutoff = torch.abs(self.base_pos[:, 0]) > 10.0
        y_cutoff = torch.abs(self.base_pos[:, 1]) > 10.0

        # Save terminal values before LeggedRobot.reset_idx() replaces the
        # finished environment with the next episode's initial state.
        self.terminal_goal_dist[:] = self.goal_dist
        self.terminal_speed[:] = torch.linalg.norm(self.base_lin_vel, dim=1)
        # Paper Table II balance term, using the pitch/yaw angular rates
        # (omega_by and omega_bz).  This is cached before reset for evaluation
        # and uses the same definition as the training reward below.
        self.terminal_balance_reward[:] = torch.exp(
            -torch.sum(torch.square(self.base_ang_vel[:, 1:3]), dim=1)
        )
        self.terminal_position[:] = self.root_states[:, :2]
        self.terminal_timeout[:] = self.time_out_buf
        self.terminal_unstable[:] = roll_cutoff | pitch_cutoff
        self.terminal_out_of_bounds[:] = x_cutoff | y_cutoff

        self.reset_buf |= self.success_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= x_cutoff
        self.reset_buf |= y_cutoff

    def _reward_stop(self):
        # Table II uses a raw success reward of 100 with weight 20.
        return 100.0 * self.success_buf.float()

    def _reward_time(self):
        return 1.0

    def _reward_approaching_target(self):
        """Symmetric paper-style approach reward: +1 closer, -1 otherwise."""
        return torch.where(
            self.goal_dist < self.last_goal_dist,
            torch.ones_like(self.goal_dist),
            -torch.ones_like(self.goal_dist),
        )

    def _reward_away_to_target(self):
        goal_change = self.goal_dist - self.last_goal_dist
        return (goal_change > 0).float() * self.cfg.rewards.close_para

    def _reward_balance(self):
        # Penalize pitch/yaw angular motion.  The previous implementation used
        # roll/pitch (x/y), which did not directly discourage in-place yaw spin.
        return torch.exp(-torch.sum(torch.square(self.base_ang_vel[:, 1:3]), dim=1))

    def _reward_near_goal_speed(self):
        near_goal = (self.goal_dist <= self._current_success_distance()).float()
        speed = torch.linalg.norm(self.base_lin_vel, dim=1)
        excess_speed = torch.relu(
            speed - self.cfg.evaluation.stop_velocity_threshold
        )
        return near_goal * torch.square(excess_speed)

    def _reward_to_target(self):
        distance = torch.linalg.norm(
            self.commands[:, :2] - self.root_states[:, :2], dim=1
        )
        sigma = max(float(self.cfg.rewards.tracking_sigma_main), 1.0e-6)
        # Table II: exp[-(d/sigma)^2].
        return torch.exp(-torch.square(distance / sigma))

    def _reward_torques(self):
        return super()._reward_torques()

    def _reward_action_rate(self):
        return super()._reward_action_rate()

    def _reward_overturn(self):
        return ((torch.abs(self.base_euler_tensor[:, 0]) > 1.2) |
                (torch.abs(self.base_euler_tensor[:, 1]) > 1.2)).float()

    def _reward_lin_vel_x_limit(self):
        return (self.base_lin_vel[:, 0] > 1.5).float()

    def _reward_ang_vel_z_limit(self):
        # The paper relates yaw rate to forward speed.  Add a small tolerance
        # so tiny numerical yaw motion at nearly zero forward speed does not
        # dominate the early-stage learning signal.
        yaw_limit = 0.72 * torch.abs(self.base_lin_vel[:, 0]) + 0.05
        return (torch.abs(self.base_ang_vel[:, 2]) > yaw_limit).float()
