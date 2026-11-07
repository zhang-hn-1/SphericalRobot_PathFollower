"""2D action crowd-following benchmark with a moving target and pedestrians."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from perception.research.predictive_follow_env import (
    TARGET_WAYPOINTS, route_progress, point_at_loop_distance)


class CrowdFollow2DEnv:
    def __init__(self, episode_s=30.0, seed=0, num_pedestrians=3,
                 target_speed=0.45, obstacle_num=2):
        self.episode_s = float(episode_s)
        self.rng = np.random.default_rng(seed)
        self.num_pedestrians = int(num_pedestrians)
        self.target_speed = float(target_speed)
        self.obstacle_num = int(obstacle_num)
        self.action_dim = 2
        self.obs_dim = 15 + 4 * self.num_pedestrians
        self.route = np.asarray(TARGET_WAYPOINTS, dtype=np.float64)
        self.obstacles = []
        self.pedestrians = []
        self.reset()

    def _random_scene(self):
        angle = float(self.rng.uniform(-0.25, 0.25))
        width = float(self.rng.uniform(2.0, 2.8))
        height = float(self.rng.uniform(2.0, 2.8))
        center = np.array([self.rng.uniform(-0.4, 0.4),
                           self.rng.uniform(-2.3, -2.0)])
        c, s = math.cos(angle), math.sin(angle)
        local = np.array([[-width / 2, -height / 2],
                          [width / 2, -height / 2],
                          [width / 2, height / 2],
                          [-width / 2, height / 2]])
        self.route = local @ np.array([[c, s], [-s, c]]) + center
        self.obstacles = []
        for _ in range(self.obstacle_num):
            obstacle_center = np.array([
                center[0] + float(self.rng.uniform(-0.8, 0.8)),
                float(self.rng.uniform(-2.7, -1.7))])
            size = (float(self.rng.uniform(0.4, 0.8)),
                    float(self.rng.uniform(0.4, 0.8)))
            self.obstacles.append((obstacle_center, size,
                                   float(self.rng.uniform(-0.5, 0.5))))

    def _obstacle_clearance(self, point):
        best = float("inf")
        for center, size, yaw in self.obstacles:
            delta = np.asarray(point) - center
            c, s = math.cos(yaw), math.sin(yaw)
            local = np.array([c * delta[0] + s * delta[1],
                              -s * delta[0] + c * delta[1]])
            dx = max(abs(local[0]) - size[0] / 2.0, 0.0)
            dy = max(abs(local[1]) - size[1] / 2.0, 0.0)
            best = min(best, math.hypot(dx, dy))
        return best

    def _point_inside(self, point, obstacle):
        center, size, yaw = obstacle
        delta = np.asarray(point) - center
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * delta[0] + s * delta[1],
                          -s * delta[0] + c * delta[1]])
        return (abs(local[0]) <= size[0] / 2.0
                and abs(local[1]) <= size[1] / 2.0)

    def _line_blocked(self, start, end, samples=32):
        for alpha in np.linspace(0.05, 0.95, samples):
            point = np.asarray(start) + alpha * (np.asarray(end)
                                                  - np.asarray(start))
            if any(self._point_inside(point, obstacle)
                   for obstacle in self.obstacles):
                return True
        return False

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._random_scene()
        self.t = 0.0
        self.robot = np.array([self.rng.uniform(-0.4, 0.4), -4.0])
        self.robot_velocity = np.zeros(2)
        self.target_distance = float(self.rng.uniform(
            0.0, sum(np.linalg.norm(self.route[(i + 1) % len(self.route)]
                                    - self.route[i])
                     for i in range(len(self.route)))))
        self.target, self.target_velocity = point_at_loop_distance(
            self.target_distance, self.route)
        self.pedestrians = []
        for _ in range(self.num_pedestrians):
            for _attempt in range(50):
                point = np.array([self.rng.uniform(-2.8, 2.8),
                                  self.rng.uniform(-6.0, -1.0)])
                if np.linalg.norm(point - self.robot) > 2.0:
                    break
            angle = float(self.rng.uniform(-math.pi, math.pi))
            speed = 0.55 * float(self.rng.uniform(0.5, 1.2))
            self.pedestrians.append(dict(
                position=point,
                velocity=np.array([speed * math.cos(angle),
                                   speed * math.sin(angle)])))
        self.visible_frames = 0
        self.steps = 0
        self.static_collisions = 0
        self.pedestrian_collisions = 0
        self.min_ped_distance = float("inf")
        self.distance_sum = 0.0
        self.path_length = 0.0
        self.control_effort = 0.0
        self.route_length = float(sum(
            np.linalg.norm(self.route[(i + 1) % len(self.route)]
                           - self.route[i]) for i in range(len(self.route))))
        return self.observation()

    def observation(self):
        target_delta = self.target - self.robot
        distance = float(np.linalg.norm(target_delta))
        features = [
            float(np.clip(target_delta[0] / 5.0, -1.0, 1.0)),
            float(np.clip(target_delta[1] / 5.0, -1.0, 1.0)),
            float(np.clip(self.target_velocity[0], -1.0, 1.0)),
            float(np.clip(self.target_velocity[1], -1.0, 1.0)),
            float(np.clip(distance / 6.0, 0.0, 1.0)),
            1.0,
            float(np.clip(self._obstacle_clearance(self.robot) / 2.0,
                          0.0, 1.0)),
            float(np.clip(route_progress(self.robot, self.route) / 6.0,
                          0.0, 1.0)),
            float(np.clip(self.robot_velocity[0], -1.0, 1.0)),
            float(np.clip(self.robot_velocity[1], -1.0, 1.0)),
            float(np.clip(self.target[0] / 3.0, -1.0, 1.0)),
            float(np.clip(self.target[1] / 3.0, -1.0, 1.0)),
            float(np.clip((self.robot[0] - self.target[0]) / 2.0,
                          -1.0, 1.0)),
            float(np.clip((self.robot[1] - self.target[1]) / 2.0,
                          -1.0, 1.0)),
            float(np.clip(route_progress(self.target, self.route) / 6.0,
                          0.0, 1.0)),
        ]
        for pedestrian in self.pedestrians:
            delta = pedestrian["position"] - self.robot
            features.extend([
                float(np.clip(delta[0] / 3.0, -1.0, 1.0)),
                float(np.clip(delta[1] / 3.0, -1.0, 1.0)),
                float(np.clip(pedestrian["velocity"][0], -1.0, 1.0)),
                float(np.clip(pedestrian["velocity"][1], -1.0, 1.0)),
            ])
        return np.asarray(features, dtype=np.float32)

    def set_orca_state(self):
        return self.robot.copy(), self.pedestrians

    def orca_action(self, sim_params=(10, 10, 5, 5), radius=0.3):
        import rvo2
        robot, pedestrians = self.set_orca_state()
        sim = rvo2.PyRVOSimulator(0.25, *sim_params, radius, 1.0)
        sim.addAgent(tuple(robot), *sim_params, radius, 1.0,
                     tuple(self.robot_velocity))
        velocity = self.target - robot
        speed = float(np.linalg.norm(velocity))
        preferred = velocity / speed if speed > 1.0 else velocity
        sim.setAgentPrefVelocity(0, tuple(preferred))
        for index, pedestrian in enumerate(pedestrians):
            sim.addAgent(tuple(pedestrian["position"]), *sim_params,
                         radius, 1.0, tuple(pedestrian["velocity"]))
            sim.setAgentPrefVelocity(index + 1, (0.0, 0.0))
        sim.doStep()
        return np.clip(np.asarray(sim.getAgentVelocity(0)), -0.6, 0.6)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1),
                         -1.0, 1.0)
        velocity = action * 0.6
        previous = self.robot.copy()
        candidate = self.robot + velocity * 0.1
        if self._obstacle_clearance(candidate) < 0.35:
            velocity = np.zeros(2)
            self.static_collisions += 1
        self.robot = self.robot + velocity * 0.1
        self.robot_velocity = velocity
        self.path_length += float(np.linalg.norm(self.robot - previous))
        self.control_effort += float(np.linalg.norm(velocity))
        self.t += 0.1
        self.target_distance += self.target_speed * 0.1
        self.target, self.target_velocity = point_at_loop_distance(
            self.target_distance, self.route)
        minimum = float("inf")
        collision = False
        for pedestrian in self.pedestrians:
            pedestrian["position"] += pedestrian["velocity"] * 0.1
            position = pedestrian["position"]
            if position[0] < -3.0 or position[0] > 3.0:
                pedestrian["velocity"][0] *= -1.0
            if position[1] < -6.0 or position[1] > -1.0:
                pedestrian["velocity"][1] *= -1.0
            distance = float(np.linalg.norm(position - self.robot))
            minimum = min(minimum, distance)
            if distance < 0.45:
                collision = True
        self.min_ped_distance = min(self.min_ped_distance, minimum)
        if collision:
            self.pedestrian_collisions += 1
        self.steps += 1
        self.visible_frames += 1
        distance = float(np.linalg.norm(self.target - self.robot))
        self.distance_sum += distance
        reward = (0.10
                  - 0.02 * abs(distance - 2.0)
                  - 0.004 * float(np.linalg.norm(velocity)))
        if collision:
            reward -= 10.0
        if self.static_collisions:
            reward -= 2.0
        done = self.t >= self.episode_s - 1e-9
        info = dict(distance=distance,
                    pedestrian_collision=collision,
                    static_collision=bool(self.static_collisions),
                    min_ped_distance=minimum,
                    route_progress=self.target_distance / self.route_length)
        return self.observation(), reward, done, info
