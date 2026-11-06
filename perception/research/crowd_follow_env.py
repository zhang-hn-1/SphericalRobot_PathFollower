"""Multi-pedestrian 2D benchmark on top of the stage-like follow env.

The target follows the same random route, while K dynamic pedestrians cross
the robot's workspace.  A social policy sees the target features plus a set
of pedestrian features; flat/rule baselines only see the original 15D follow
observation.  The environment is intentionally lightweight so the comparison
can run many randomized episodes.
"""

from __future__ import annotations

import math

import numpy as np

from perception.research.predictive_follow_env import PredictiveFollowEnv


class CrowdFollowEnv(PredictiveFollowEnv):
    def __init__(self, *args, num_pedestrians=3, pedestrian_speed=0.55,
                 **kwargs):
        self.num_pedestrians = int(num_pedestrians)
        self.pedestrian_speed = float(pedestrian_speed)
        super().__init__(*args, **kwargs)
        self.pedestrians = []
        self.pedestrian_collisions = 0
        self.min_ped_distance = float("inf")
        self._base_obs = None

    def _spawn_pedestrians(self):
        self.pedestrians = []
        for _ in range(self.num_pedestrians):
            # Start around the workspace, avoiding the robot start.
            for _attempt in range(40):
                point = np.array([self.rng.uniform(-2.6, 2.6),
                                  self.rng.uniform(-6.0, -1.5)])
                if np.linalg.norm(point - np.array([self.robot_x,
                                                     self.robot_y])) > 2.0:
                    break
            angle = float(self.rng.uniform(-math.pi, math.pi))
            speed = self.pedestrian_speed * float(self.rng.uniform(0.5, 1.2))
            self.pedestrians.append(dict(
                position=point,
                velocity=np.array([speed * math.cos(angle),
                                   speed * math.sin(angle)]),
            ))

    def reset(self, seed=None):
        obs = super().reset(seed)
        self._spawn_pedestrians()
        self.pedestrian_collisions = 0
        self.min_ped_distance = float("inf")
        self._base_obs = np.asarray(obs, dtype=np.float32).copy()
        return self._base_obs

    def _step_pedestrians(self):
        for pedestrian in self.pedestrians:
            pedestrian["position"] += pedestrian["velocity"] * 0.1
            position = pedestrian["position"]
            if position[0] < -3.0 or position[0] > 3.0:
                pedestrian["velocity"][0] *= -1.0
            if position[1] < -6.0 or position[1] > -1.0:
                pedestrian["velocity"][1] *= -1.0

    def pedestrian_features(self):
        features = []
        robot = np.array([self.robot_x, self.robot_y])
        for pedestrian in self.pedestrians:
            delta = pedestrian["position"] - robot
            velocity = pedestrian["velocity"]
            features.extend([
                float(np.clip(delta[0] / 3.0, -1.0, 1.0)),
                float(np.clip(delta[1] / 3.0, -1.0, 1.0)),
                float(np.clip(velocity[0], -1.0, 1.0)),
                float(np.clip(velocity[1], -1.0, 1.0)),
            ])
        return np.asarray(features, dtype=np.float32)

    def scenario_metadata(self):
        """Serializable scenario record for replay datasets."""
        return dict(
            route=[[float(value) for value in point] for point in self.route],
            obstacles=[dict(center=[float(center[0]), float(center[1])],
                            size=[float(size[0]), float(size[1])],
                            yaw=float(yaw))
                       for center, size, yaw in self.obstacles],
            pedestrians=[dict(position=[float(p["position"][0]),
                                        float(p["position"][1])],
                              velocity=[float(p["velocity"][0]),
                                        float(p["velocity"][1])])
                         for p in self.pedestrians],
            target_speed=float(self.target_speed),
            robot_start=[float(self.robot_x), float(self.robot_y)],
        )

    def social_observation(self):
        base = (self._base_obs if self._base_obs is not None
                else super()._observation(self.target, True, 0.0))
        return np.concatenate([base, self.pedestrian_features()])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._base_obs = np.asarray(obs, dtype=np.float32).copy()
        self._step_pedestrians()
        robot = np.array([self.robot_x, self.robot_y])
        minimum = float("inf")
        collision = False
        for pedestrian in self.pedestrians:
            distance = float(np.linalg.norm(
                pedestrian["position"] - robot))
            minimum = min(minimum, distance)
            if distance < 0.45:
                collision = True
        self.min_ped_distance = min(self.min_ped_distance, minimum)
        if collision:
            self.pedestrian_collisions += 1
            reward -= 10.0
            done = True
        info["pedestrian_collision"] = collision
        info["min_ped_distance"] = minimum
        return self._base_obs, reward, done, info
