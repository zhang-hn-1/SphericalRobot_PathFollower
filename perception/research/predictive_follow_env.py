"""Lightweight 2D predictive-follow environment for RL.

The environment keeps the structure that matters for the lost-target policy:

* a target walking a rectangular loop around a refrigerator,
* line-of-sight occlusion by that refrigerator,
* a robot on a safe observation lane that can translate laterally,
* a measurement history, velocity confidence and prediction uncertainty,
* the same clearance rule used by the ReplicaCAD stage demo.

It is deliberately fast (pure NumPy, no renderer, no detector) so PPO can be
trained in minutes.  The observation/action contract mirrors the features that
``perception/sim/stage_follow.py`` can compute from a ``TargetBelief``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from perception.stage_map import StageMap
except Exception:  # keep the 2D env importable from a bare script checkout
    StageMap = None

DT = 0.1
ROBOT_LANE_Y = -4.0
TARGET_WAYPOINTS = np.array([
    [-1.20, -1.10],
    [1.20, -1.10],
    [1.20, -3.20],
    [-1.20, -3.20],
], dtype=np.float64)
TARGET_SPEED = 0.40
FRIDGE_CENTER = np.array([0.0, -2.20])
FRIDGE_SIZE = (0.75, 0.80)


def _wrap(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _point_inside(point, center, size, yaw=0.0):
    delta = np.asarray(point) - center
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([c * delta[0] + s * delta[1],
                      -s * delta[0] + c * delta[1]])
    return abs(local[0]) <= size[0] / 2.0 and abs(local[1]) <= size[1] / 2.0


def _obstacle_clearance(point, obstacle):
    center, size, yaw = obstacle
    delta = np.asarray(point) - np.asarray(center)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([c * delta[0] + s * delta[1],
                      -s * delta[0] + c * delta[1]])
    dx = max(abs(local[0]) - size[0] / 2.0, 0.0)
    dy = max(abs(local[1]) - size[1] / 2.0, 0.0)
    return math.hypot(dx, dy)


def clearance(point, obstacles=None):
    if obstacles is None:
        return _obstacle_clearance(
            point, (FRIDGE_CENTER, FRIDGE_SIZE, 0.0))
    return min(_obstacle_clearance(point, obstacle) for obstacle in obstacles)


def line_blocked(a, b, samples=40, obstacles=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    obstacles = ([(FRIDGE_CENTER, FRIDGE_SIZE, 0.0)]
                 if obstacles is None else obstacles)
    for alpha in np.linspace(0.05, 0.95, samples):
        point = a + alpha * (b - a)
        if any(_point_inside(point, center, size, yaw)
               for center, size, yaw in obstacles):
            return True
    return False


def point_at_loop_distance(distance, route=None):
    points = TARGET_WAYPOINTS if route is None else np.asarray(route)
    segments = [points[(i + 1) % len(points)] - points[i]
                for i in range(len(points))]
    lengths = [float(np.linalg.norm(segment)) for segment in segments]
    total = sum(lengths)
    distance = distance % total
    for index, (segment, length) in enumerate(zip(segments, lengths)):
        if distance <= length:
            unit = segment / length
            return points[index] + unit * distance, unit
        distance -= length
    return points[0], segments[0] / lengths[0]


def loop_length(route=None):
    points = TARGET_WAYPOINTS if route is None else np.asarray(route)
    return float(sum(np.linalg.norm(points[(i + 1) % len(points)] - points[i])
                     for i in range(len(points))))


def route_progress(position, route=None):
    """Arc length of the closest point on the target route."""
    position = np.asarray(position, dtype=np.float64).reshape(2)
    points = TARGET_WAYPOINTS if route is None else np.asarray(route)
    travelled = 0.0
    best_error = float("inf")
    best = 0.0
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        unit = segment / length
        projection = float(np.clip(np.dot(position - start, unit),
                                   0.0, length))
        error = float(np.linalg.norm(position - (start + unit * projection)))
        if error < best_error:
            best_error = error
            best = travelled + projection
        travelled += length
    return best


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class PredictiveFollowEnv:
    """Single-agent 2D environment with a hidden target and a fridge."""

    def __init__(self, episode_s=30.0, seed=0, stage_visibility=False,
                 detection_dropout=0.0, measurement_noise=0.03,
                 privileged_obs=False, randomize=False, route=None,
                 obstacles=None):
        self.episode_s = float(episode_s)
        self.rng = np.random.default_rng(seed)
        self.privileged_obs = bool(privileged_obs)
        self.randomize = bool(randomize)
        self.route = (TARGET_WAYPOINTS if route is None
                      else np.asarray(route, dtype=np.float64))
        self.obstacles = ([(FRIDGE_CENTER, FRIDGE_SIZE, 0.0)]
                          if obstacles is None else obstacles)
        self.detection_dropout = float(np.clip(detection_dropout, 0.0, 0.8))
        self.measurement_noise = float(max(0.0, measurement_noise))
        self.stage_map = None
        if stage_visibility and StageMap is not None:
            self.stage_map = StageMap(
                Path(__file__).resolve().parents[1]
                / "assets/scenes/replica_stage/stage.obj")
        self.predictor = None
        self.geometric_visible = True
        self.obs_dim = 15
        self.action_dim = 1
        self.reset()

    # -- helpers ----------------------------------------------------

    def _random_scene(self):
        """Generate a route and 1-3 occluders for an OOD episode."""
        angle = float(self.rng.uniform(-0.25, 0.25))
        width = float(self.rng.uniform(1.8, 2.8))
        height = float(self.rng.uniform(1.4, 2.1))
        center = np.array([self.rng.uniform(-0.4, 0.4),
                           self.rng.uniform(-2.3, -2.0)])
        c, s = math.cos(angle), math.sin(angle)
        local = np.array([[-width / 2, -height / 2],
                          [width / 2, -height / 2],
                          [width / 2, height / 2],
                          [-width / 2, height / 2]])
        self.route = (local @ np.array([[c, s], [-s, c]])) + center
        self.obstacles = []
        count = int(self.rng.integers(1, 4))
        for _ in range(count):
            # Keep occluders in the route band, not on the robot observation
            # lane at y=-4; otherwise a random episode starts in collision
            # and the policy comparison is invalid.
            occluder_center = np.array([
                center[0] + float(self.rng.uniform(-0.8, 0.8)),
                float(self.rng.uniform(-2.7, -1.7)),
            ])
            occluder_size = (float(self.rng.uniform(0.45, 0.8)),
                             float(self.rng.uniform(0.45, 0.8)))
            self.obstacles.append((occluder_center, occluder_size,
                                   float(self.rng.uniform(-0.6, 0.6))))

    def _target_state(self, distance):
        position, tangent = point_at_loop_distance(distance, self.route)
        velocity = tangent * self.target_speed
        return position, velocity

    def set_predictor(self, predictor):
        """Use a frozen supervised predictor for hidden-target observations."""
        self.predictor = predictor

    def predicted_position(self, age):
        """World point used by observations and the rule baseline."""
        age = max(0.0, float(age))
        if self.predictor is not None:
            displacement = self.predictor.predict_at(
                self.measurement_history, age, self.last_velocity)
            if displacement is not None:
                return self.last_position + displacement
        return self.last_position + self.last_velocity * age

    def _visible(self, target):
        if float(np.linalg.norm(target - np.array([self.robot_x,
                                                   self.robot_y]))) > 12.0:
            return False
        if line_blocked((self.robot_x, self.robot_y), target,
                        obstacles=self.obstacles):
            return False
        if self.stage_map is not None and not self.stage_map.line_of_sight(
                (self.robot_x, self.robot_y), target):
            return False
        return True

    def privileged_observation(self):
        """Teacher view: clean target state, no detection dropout."""
        target = self.target
        velocity = self.target_velocity
        delta = target - np.array([self.robot_x, self.robot_y])
        distance = float(np.linalg.norm(delta))
        probe = np.array([self.robot_x + np.clip(delta[0], -1.0, 1.0) * 0.4,
                          self.robot_y + np.clip(delta[1], -1.0, 1.0) * 0.4])
        return np.array([
            float(np.clip(delta[0] / 5.0, -1.0, 1.0)),
            float(np.clip(delta[1] / 5.0, -1.0, 1.0)),
            float(np.clip(velocity[0] / 1.0, -1.0, 1.0)),
            float(np.clip(velocity[1] / 1.0, -1.0, 1.0)),
            0.0,
            1.0,
            0.0,
            1.0,
            float(np.clip(distance / 6.0, 0.0, 1.0)),
            float(np.clip(clearance((self.robot_x, self.robot_y),
                                     self.obstacles) / 2.0, 0.0, 1.0)),
            float(np.clip(clearance(probe, self.obstacles) / 2.0, 0.0, 1.0)),
            float(np.clip((self.robot_y - ROBOT_LANE_Y) / 1.0, -1.0, 1.0)),
            float(np.clip((target[0] - self.robot_x) / 2.0, -1.0, 1.0)),
            float(np.clip(target[0] / 3.0, -1.0, 1.0)),
            float(np.clip(route_progress(target, self.route) / 6.0,
                          0.0, 1.0)),
        ], dtype=np.float32)

    def _observation(self, target, visible, measurement_age):
        if self.privileged_obs:
            return self.privileged_observation()
        if visible:
            # The policy sees the same measured position/velocity as the
            # world-frame target layer, not the simulator's clean target.
            predicted = self.last_position.copy()
            target_velocity = self.last_velocity.copy()
        else:
            predicted = self.predicted_position(measurement_age)
            target_velocity = self.last_velocity
        delta = predicted - np.array([self.robot_x, self.robot_y])
        distance = float(np.linalg.norm(delta))
        probe = np.array([self.robot_x + np.clip(delta[0], -1.0, 1.0) * 0.4,
                          self.robot_y + np.clip(delta[1], -1.0, 1.0) * 0.4])
        return np.array([
            float(np.clip(delta[0] / 5.0, -1.0, 1.0)),
            float(np.clip(delta[1] / 5.0, -1.0, 1.0)),
            float(np.clip(target_velocity[0] / 1.0, -1.0, 1.0)),
            float(np.clip(target_velocity[1] / 1.0, -1.0, 1.0)),
            float(np.clip(measurement_age / 3.0, 0.0, 1.0)),
            float(self.velocity_confidence),
            float(np.clip(self.prediction_uncertainty / 1.0, 0.0, 1.0)),
            1.0 if visible else 0.0,
            float(np.clip(distance / 6.0, 0.0, 1.0)),
            float(np.clip(clearance((self.robot_x, self.robot_y),
                                     self.obstacles) / 2.0, 0.0, 1.0)),
            float(np.clip(clearance(probe, self.obstacles) / 2.0, 0.0, 1.0)),
            float(np.clip((self.robot_y - ROBOT_LANE_Y) / 1.0, -1.0, 1.0)),
            float(np.clip((predicted[0] - self.robot_x) / 2.0, -1.0, 1.0)),
            float(np.clip(predicted[0] / 3.0, -1.0, 1.0)),
            float(np.clip(route_progress(predicted, self.route) / 6.0,
                          0.0, 1.0)),
        ], dtype=np.float32)

    # -- API --------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0.0
        if self.randomize:
            self._random_scene()
        self.robot_x = float(self.rng.uniform(-0.5, 1.5))
        self.robot_y = ROBOT_LANE_Y
        self.robot_vx = 0.0
        self.target_speed = TARGET_SPEED * float(self.rng.uniform(0.65, 1.35))
        self.target_distance = float(self.rng.uniform(0.0, loop_length(self.route)))
        target, velocity = self._target_state(self.target_distance)
        self.target = target
        self.target_velocity = velocity
        self.last_position = target.copy()
        self.last_velocity = np.zeros(2)
        self.last_measurement_t = self.t
        self.velocity_confidence = 0.7
        self.prediction_uncertainty = 0.15
        self.visible = True
        self.geometric_visible = True
        self.reacquired = False
        self.was_hidden = False
        self.hidden_frames = 0
        self.min_clearance = clearance(
            (self.robot_x, self.robot_y), self.obstacles)
        self.path_length = 0.0
        self.control_effort = 0.0
        self.collisions = 0
        self.visible_frames = 0
        self.distance_sum = 0.0
        self.steps = 0
        self.measurement_history = [(self.t, target.copy())]
        return self._observation(target, True, 0.0)

    def step(self, action):
        action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self.steps += 1
        self.t += DT
        self.target_distance += self.target_speed * DT
        target, target_velocity = self._target_state(self.target_distance)
        self.target = target
        self.target_velocity = target_velocity

        # Holonomic action; the observation lane remains the safe corridor.
        vx = 0.60 * action
        vy = float(np.clip(0.9 * (ROBOT_LANE_Y - self.robot_y), -0.30, 0.30))
        probe = np.array([self.robot_x + vx * 0.4,
                          self.robot_y + vy * 0.4])
        if clearance(probe, self.obstacles) < 0.55:
            vx = vy = 0.0
        previous_x, previous_y = self.robot_x, self.robot_y
        self.robot_x += vx * DT
        self.robot_y += vy * DT
        self.robot_vx = vx
        self.path_length += math.hypot(self.robot_x - previous_x,
                                       self.robot_y - previous_y)
        self.control_effort += abs(vx) + abs(vy)
        current_clearance = clearance((self.robot_x, self.robot_y),
                                      self.obstacles)
        self.min_clearance = min(self.min_clearance, current_clearance)

        geometric_visible = self._visible(target)
        self.geometric_visible = geometric_visible
        visible = (geometric_visible
                   and self.rng.random() >= self.detection_dropout)
        self.visible = visible
        if visible:
            previous_measurement = self.last_position.copy()
            measurement = target + self.rng.normal(
                0.0, self.measurement_noise, 2)
            raw_velocity = (measurement - previous_measurement) / DT
            self.last_velocity = 0.65 * self.last_velocity + 0.35 * raw_velocity
            self.last_position = measurement.copy()
            self.last_measurement_t = self.t
            self.measurement_history.append((self.t, measurement.copy()))
            cutoff = self.t - 3.0
            self.measurement_history = [item for item in self.measurement_history
                                        if item[0] >= cutoff][-64:]
            self.velocity_confidence = self._history_confidence()
            self.visible_frames += 1
            self.hidden_frames = 0
        else:
            self.hidden_frames += 1
            self.was_hidden = True

        measurement_age = max(0.0, self.t - self.last_measurement_t)
        self.prediction_uncertainty = 0.15 + 0.35 * measurement_age
        self.velocity_confidence = float(
            self._history_confidence() * math.exp(-measurement_age / 2.0))
        distance = float(np.linalg.norm(
            target - np.array([self.robot_x, self.robot_y])))
        self.distance_sum += distance

        reward = 0.0
        if visible:
            reward += 0.10
            reward += 0.05 if 1.5 <= distance <= 5.0 else -0.03
            reward -= 0.02 * abs(distance - 2.5)
            if self.was_hidden and not self.reacquired:
                self.reacquired = True
                reward += 1.0
                self.was_hidden = False
        else:
            predicted = self.predicted_position(measurement_age)
            error = abs(self.robot_x - predicted[0])
            reward += 0.08 * max(0.0, 1.0 - error / 3.0)
            reward -= 0.015 * measurement_age
            if self.velocity_confidence < 0.35:
                reward -= 0.02
        reward -= 0.006 * abs(vx)
        collision = current_clearance < 0.42
        if collision:
            self.collisions += 1
            reward -= 10.0
        done = collision or self.t >= self.episode_s - 1e-9
        info = dict(visible=visible, distance=distance, age=measurement_age,
                    velocity_confidence=self.velocity_confidence,
                    clearance=current_clearance, collision=collision,
                    reacquired=self.reacquired)
        return self._observation(target, visible, measurement_age), reward, done, info

    def _history_confidence(self):
        history = self.measurement_history
        if len(history) < 3:
            return 0.2
        times = np.asarray([item[0] for item in history])
        positions = np.asarray([item[1] for item in history])
        span = float(times[-1] - times[0])
        if span <= 1e-6:
            return 0.0
        design = np.column_stack([times - times.mean(), np.ones_like(times)])
        residual = []
        for axis in range(2):
            coefficients = np.linalg.lstsq(
                design, positions[:, axis], rcond=None)[0]
            residual.append(design @ coefficients - positions[:, axis])
        rms = float(np.sqrt(np.mean(np.square(np.column_stack(residual)))))
        speeds = np.diff(positions, axis=0) / np.diff(times)[:, None]
        speed_std = float(np.linalg.norm(
            speeds - speeds.mean(axis=0), axis=1).mean())
        return float(np.clip(
            0.35 * min(1.0, span) + 0.35 * math.exp(-rms / 0.25)
            + 0.30 * math.exp(-speed_std / 0.5), 0.0, 1.0))
