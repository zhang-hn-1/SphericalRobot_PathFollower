"""Rotunbot point-navigation environment with procedural maze walls."""

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import torch_rand_float

from legged_gym.maps import (
    build_maze,
    cell_centers_to_world,
    reachable_free_cells,
    wall_cells,
)
from ..target_point.rotunbot_target_obstacle import RotunbotTargetObstacle
from .rotunbot_maze_config import RotunbotMazeCfg


class RotunbotMaze(RotunbotTargetObstacle):
    """A fixed seeded maze shared by all vectorized environment instances."""

    cfg: RotunbotMazeCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.maze_layout = None
        self.mazes = []
        self._goal_rng = np.random.default_rng(int(cfg.maze.seed) + 1)
        self._maze_goal_positions = None
        self._maze_wall_centers = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _compute_torques(self, actions):
        """Map policy actions to the same velocity/steering control as teleop."""
        if self.cfg.control.control_type != "R":
            return super()._compute_torques(actions)

        targets = torch.empty_like(actions)
        targets[:, 0] = torch.clip(
            actions[:, 0] * self.cfg.control.first_actionScale,
            -6.0,
            6.0,
        )
        targets[:, 1] = torch.clip(
            actions[:, 1] * self.cfg.control.second_actionScale,
            -0.5236,
            0.5236,
        )
        self.output_actions = targets

        torques = torch.empty_like(actions)
        torques[:, 0] = self.cfg.control.first_velocity_kp * (
            targets[:, 0] - self.dof_vel[:, 0]
        )
        torques[:, 1] = (
            self.cfg.control.second_position_kp
            * (targets[:, 1] - self.dof_pos[:, 1])
            - self.cfg.control.second_velocity_kd * self.dof_vel[:, 1]
        )
        torques[:, 0] = torch.clip(
            torques[:, 0],
            -self.cfg.control.torque_limits_1,
            self.cfg.control.torque_limits_1,
        )
        torques[:, 1] = torch.clip(
            torques[:, 1],
            -self.cfg.control.torque_limits_2,
            self.cfg.control.torque_limits_2,
        )
        return torques

    def _create_scene_assets(self):
        self.maze_layout = build_maze(
            grid_size=self.cfg.maze.grid_size,
            seed=self.cfg.maze.seed,
            center_clearance_radius=self.cfg.maze.center_clearance_radius,
        )
        # Sharing one layout guarantees the same number and ordering of actors
        # in every vectorized environment, which Isaac Gym tensor views require.
        self.mazes = [self.maze_layout] * self.num_envs

        reachable_cells = reachable_free_cells(self.maze_layout)
        goal_positions = cell_centers_to_world(
            reachable_cells,
            self.maze_layout.shape,
            self.cfg.maze.cell_size,
        )
        minimum_distance = float(self.cfg.maze.min_goal_distance)
        goal_positions = goal_positions[
            np.linalg.norm(goal_positions, axis=1) >= minimum_distance
        ]
        if len(goal_positions) == 0:
            raise ValueError(
                "maze configuration has no reachable goal beyond "
                f"min_goal_distance={minimum_distance}"
            )
        self._maze_goal_positions = goal_positions
        wall_cell_indices = wall_cells(self.maze_layout)
        self._maze_wall_centers = cell_centers_to_world(
            wall_cell_indices,
            self.maze_layout.shape,
            self.cfg.maze.cell_size,
        )

        wall_options = gymapi.AssetOptions()
        wall_options.fix_base_link = True
        wall_asset = self.gym.create_box(
            self.sim,
            float(self.cfg.maze.cell_size),
            float(self.cfg.maze.cell_size),
            float(self.cfg.maze.wall_height),
            wall_options,
        )
        return {
            "wall_asset": wall_asset,
            "wall_cells": wall_cell_indices,
            "wall_color": gymapi.Vec3(*self.cfg.maze.wall_color),
        }

    def _init_buffers(self):
        super()._init_buffers()
        self.maze_wall_centers = torch.as_tensor(
            self._maze_wall_centers,
            dtype=torch.float32,
            device=self.device,
        )
        self.maze_collision_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
        cell_size = float(self.cfg.maze.cell_size)
        wall_height = float(self.cfg.maze.wall_height)
        maze_shape = np.asarray(self.maze_layout.shape, dtype=np.float64)
        origin = self.env_origins[env_id]

        for x, y in scene_assets["wall_cells"]:
            wall_pose = gymapi.Transform()
            wall_pose.p.x = origin[0].item() + (
                (float(x) - maze_shape[0] / 2.0 + 0.5) * cell_size
            )
            wall_pose.p.y = origin[1].item() + (
                (float(y) - maze_shape[1] / 2.0 + 0.5) * cell_size
            )
            wall_pose.p.z = origin[2].item() + wall_height / 2.0
            wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            wall_actor = self.gym.create_actor(
                env_handle,
                scene_assets["wall_asset"],
                wall_pose,
                f"maze_wall_{int(x)}_{int(y)}",
                env_id,
                0,
                0,
            )
            self.gym.set_rigid_body_color(
                env_handle,
                wall_actor,
                0,
                gymapi.MESH_VISUAL,
                scene_assets["wall_color"],
            )

    def _resample_commands(self, env_ids):
        """Sample only collision-free goals connected to the center spawn."""
        count = len(env_ids)
        if count == 0:
            return

        goal_indices = self._goal_rng.integers(
            0, len(self._maze_goal_positions), size=count
        )
        local_goals = torch.as_tensor(
            self._maze_goal_positions[goal_indices],
            dtype=self.commands.dtype,
            device=self.device,
        )
        self.commands[env_ids, :2] = local_goals + self.env_origins[env_ids, :2]

        if self.cfg.commands.command_yaw:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["yaw"][0],
                self.command_ranges["yaw"][1],
                (count, 1),
                device=self.device,
            ).squeeze(1)

    def _maze_collision_mask(self):
        """Detect XY overlap between the spherical shell and any wall box."""
        local_position = self.root_states[:, :2] - self.env_origins[:, :2]
        center_delta = torch.abs(
            local_position[:, None, :] - self.maze_wall_centers[None, :, :]
        )
        outside_delta = torch.clamp(
            center_delta - float(self.cfg.maze.cell_size) / 2.0,
            min=0.0,
        )
        wall_distance = torch.linalg.vector_norm(outside_delta, dim=2)
        nearest_wall_distance = torch.min(wall_distance, dim=1).values
        return nearest_wall_distance <= float(self.cfg.maze.robot_collision_radius)

    def check_termination(self):
        super().check_termination()
        self.maze_collision_buf = self._maze_collision_mask()
        if self.cfg.maze.terminate_on_collision:
            self.reset_buf |= self.maze_collision_buf

    def _reward_collision(self):
        """Penalize contacts with every maze wall, not the legacy two boxes."""
        return self.maze_collision_buf
