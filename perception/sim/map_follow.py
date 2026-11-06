#!/usr/bin/env python3
"""Closed-loop follow through polygon map obstacles.

The target follows an A* route from the map start to the map goal.  The robot
sees the target through the front-pair camera, tracks it with the world-frame
target layer, and follows the *perceived* target position.  A separate A* route
from the robot to a stand-off point behind the target keeps the ball out of the
map walls; a short pure-pursuit/candidate-curvature search respects the
identified P0 steering envelope.

This is deliberately a perception + local-planning loop, not a full navigation
stack.  The map is known (as it would be on a robot with a map), the target is
not given to the controller: its position and velocity come from the tracker.
When the target is hidden, the controller uses the layer's prediction for up to
1 s and then stops.

Run in the Isaac Gym environment:

    source local_env.sh
    /home/jason/legged_gym/.venv/bin/python perception/sim/map_follow.py \
        --out /home/jason/perception_data/runs/map_follow_v1 \
        --map irregular_clutter --frames 320 --video
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from follow_p0 import (P0Robot, invert_steer, max_curvature,  # noqa: E402
                       wrap_angle)
from l1_front_pair import front_pair_config  # noqa: E402
from map_perception import (add_walls, load_scene, point_inside,  # noqa: E402
                            transform_scene)
from occlusion_perception import DetectionWorker, box_iou, deduplicate  # noqa: E402
from people_scene import PeopleScene, _bbox_from_mask  # noqa: E402
from range_bearing import FrontPairCamera  # noqa: E402
from target_perception import (PersonDetection,  # noqa: E402
                               TargetPerceptionLayer)
from target_tracker import hsv_histogram  # noqa: E402


# ------------------------------------------------------------------ maps


@dataclass(frozen=True)
class FollowMap:
    name: str
    map_file: str
    scene_id: str
    target_speed: float = 0.5
    note: str = ""


FOLLOW_MAPS = {
    "irregular_clutter": FollowMap(
        name="irregular_clutter",
        map_file="rl_test.json",
        scene_id="mb30v4_test_432",
        target_speed=0.5,
        note="map_013_irregular_clutter"),
    "forced_hairpin": FollowMap(
        name="forced_hairpin",
        map_file="rl_validation.json",
        scene_id="mb30v4_test_039",
        target_speed=0.45,
        note="map_004_forced_hairpin"),
    "u_goal": FollowMap(
        name="u_goal",
        map_file="rl_test.json",
        scene_id="mb30v4_test_038",
        target_speed=0.5,
        note="map_012_u_goal_inside"),
}


class HolonomicBase:
    """Simple holonomic base for the map/maze demonstration."""

    def __init__(self, xy=(0.0, 0.0), yaw=0.0):
        self.x, self.y, self.yaw = float(xy[0]), float(xy[1]), float(yaw)
        self.v = 0.0

    def move(self, vx, vy, w, dt):
        self.v = math.hypot(vx, vy)
        self.yaw = math.atan2(math.sin(self.yaw + w * dt),
                              math.cos(self.yaw + w * dt))
        self.x += vx * dt
        self.y += vy * dt


# ------------------------------------------------------------------ geometry


def inflate(obstacle, margin):
    center, (size_x, size_y), yaw = obstacle
    return center, (size_x + 2.0 * margin, size_y + 2.0 * margin), yaw


def map_bounds(goal, obstacles, margin=3.0):
    """Grid bounds that cover the start, goal and every obstacle."""
    xs = [0.0, float(goal[0])]
    ys = [0.0, float(goal[1])]
    for center, (size_x, size_y), _yaw in obstacles:
        radius = max(size_x, size_y) / 2.0 + margin
        xs.extend([center[0] - radius, center[0] + radius])
        ys.extend([center[1] - radius, center[1] + radius])
    return (min(xs) - margin, min(ys) - margin,
            max(xs) + margin, max(ys) + margin)


def clearance(xy, obstacles):
    """Distance from a point to the nearest OBB surface; 0 if inside."""
    best = float("inf")
    for obstacle in obstacles:
        center, (size_x, size_y), yaw = obstacle
        delta = np.asarray(xy) - center
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * delta[0] + s * delta[1],
                          -s * delta[0] + c * delta[1]])
        dx = max(abs(local[0]) - size_x / 2.0, 0.0)
        dy = max(abs(local[1]) - size_y / 2.0, 0.0)
        d = math.hypot(dx, dy)
        if d < best:
            best = d
    return best


def segment_clear(a, b, obstacles, samples=40):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for alpha in np.linspace(0.0, 1.0, samples):
        if any(point_inside(a + alpha * (b - a), obstacle)
               for obstacle in obstacles):
            return False
    return True


def plan_path(start, goal, obstacles, bounds=(-30.0, 30.0),
              resolution=0.25, inflation=0.55):
    """8-connected A* on an inflated OBB occupancy grid."""
    inflated = [inflate(obstacle, inflation) for obstacle in obstacles]
    xmin, ymin, xmax, ymax = bounds
    nx = int(round((xmax - xmin) / resolution)) + 1
    ny = int(round((ymax - ymin) / resolution)) + 1

    def cell(point):
        ix = int(round((point[0] - xmin) / resolution))
        iy = int(round((point[1] - ymin) / resolution))
        return max(0, min(nx - 1, ix)), max(0, min(ny - 1, iy))

    def world(ix, iy):
        return np.array([xmin + ix * resolution, ymin + iy * resolution])

    blocked = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        x = xmin + ix * resolution
        for iy in range(ny):
            y = ymin + iy * resolution
            blocked[ix, iy] = any(point_inside((x, y), obstacle)
                                  for obstacle in inflated)

    def nearest_free(point):
        start_cell = cell(point)
        if not blocked[start_cell]:
            return start_cell
        for radius in range(1, 30):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (start_cell[0] + dx, start_cell[1] + dy)
                    if (0 <= candidate[0] < nx and 0 <= candidate[1] < ny
                            and not blocked[candidate]):
                        return candidate
        return None

    start_cell = nearest_free(start)
    goal_cell = nearest_free(goal)
    if start_cell is None or goal_cell is None:
        return []

    open_heap = [(0.0, start_cell)]
    cost = {start_cell: 0.0}
    parent = {}
    visited = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal_cell:
            break
        visited.add(current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
                    continue
                if blocked[neighbor]:
                    continue
                if dx and dy:
                    if blocked[(current[0] + dx, current[1])] or \
                            blocked[(current[0], current[1] + dy)]:
                        continue
                step = math.hypot(dx, dy) * resolution
                new_cost = cost[current] + step
                if new_cost < cost.get(neighbor, float("inf")):
                    cost[neighbor] = new_cost
                    parent[neighbor] = current
                    heuristic = math.hypot(neighbor[0] - goal_cell[0],
                                           neighbor[1] - goal_cell[1]) * resolution
                    heapq.heappush(open_heap, (new_cost + heuristic, neighbor))

    if goal_cell not in parent and goal_cell != start_cell:
        return []
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(parent[cells[-1]])
    cells.reverse()
    path = [world(ix, iy) for ix, iy in cells]

    # Shortcut collinear / visible segments; the grid path is deliberately
    # conservative around the inflated walls.
    smoothed = [path[0]]
    index = 0
    while index < len(path) - 1:
        furthest = index + 1
        for candidate in range(index + 2, len(path)):
            if segment_clear(path[index], path[candidate], inflated, samples=30):
                furthest = candidate
            else:
                break
        smoothed.append(path[furthest])
        index = furthest
    return smoothed


def path_length(path):
    return float(sum(np.linalg.norm(path[i + 1] - path[i])
                     for i in range(len(path) - 1)))


def point_at_distance(path, distance):
    if not path:
        return None, None
    remaining = max(0.0, float(distance))
    for index in range(len(path) - 1):
        segment = np.linalg.norm(path[index + 1] - path[index])
        if remaining <= segment or index == len(path) - 2:
            if segment <= 1e-9:
                continue
            ratio = min(1.0, remaining / segment)
            point = path[index] + ratio * (path[index + 1] - path[index])
            tangent = (path[index + 1] - path[index]) / segment
            return point, tangent
        remaining -= segment
    tangent = path[-1] - path[-2]
    length = np.linalg.norm(tangent)
    return path[-1], tangent / max(length, 1e-9)


def lookahead_point(path, robot_xy, distance=1.2):
    if not path:
        return None
    distances = [np.linalg.norm(point - robot_xy) for point in path]
    index = int(np.argmin(distances))
    remaining = distance
    for next_index in range(index + 1, len(path)):
        segment = np.linalg.norm(path[next_index] - path[next_index - 1])
        if remaining <= segment:
            ratio = remaining / max(segment, 1e-9)
            return path[next_index - 1] + ratio * (
                path[next_index] - path[next_index - 1])
        remaining -= segment
    return path[-1]


def path_progress(path, point):
    """Arc length of the nearest point on a polyline."""
    point = np.asarray(point, dtype=np.float64).reshape(2)
    best_error = float("inf")
    best = 0.0
    travelled = 0.0
    for index in range(len(path) - 1):
        segment = path[index + 1] - path[index]
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        unit = segment / length
        projection = float(np.clip(np.dot(point - path[index], unit),
                                   0.0, length))
        error = float(np.linalg.norm(point - (path[index] + unit * projection)))
        if error < best_error:
            best_error = error
            best = travelled + projection
        travelled += length
    return best


def choose_bystander(obstacles, target_points):
    candidates = []
    for x in np.arange(-2.0, 8.0, 0.5):
        for y in np.arange(-3.0, 3.0, 0.5):
            point = np.array([x, y])
            if np.linalg.norm(point) < 3.0 or np.linalg.norm(point) > 10.0:
                continue
            if abs(math.atan2(y, x)) > math.radians(32.0):
                continue
            if any(point_inside(point, obstacle) for obstacle in obstacles):
                continue
            if clearance(point, obstacles) < 0.6:
                continue
            separation = min(float(np.linalg.norm(point - target))
                             for target in target_points)
            if separation < 2.0:
                continue
            candidates.append((separation, point))
    if not candidates:
        return np.array([3.0, 0.0])
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# ------------------------------------------------------------------ control


def candidate_curvature(robot, target_point, path, obstacles):
    """Pick a P0-admissible curvature that makes progress without a wall hit."""
    look = lookahead_point(path, (robot.x, robot.y), distance=1.2)
    if look is None:
        look = np.asarray(target_point)
    desired = math.atan2(look[1] - robot.y, look[0] - robot.x) - robot.yaw
    desired = wrap_angle(desired)
    distance = float(np.linalg.norm(np.asarray(target_point)
                                    - np.array([robot.x, robot.y])))
    if distance > 5.0:
        speed = 0.82
    elif distance > 3.0:
        speed = 0.6
    elif distance > 2.0:
        speed = 0.35
    else:
        speed = 0.0
    if speed <= 0.0:
        return 0.0, 0.0, clearance((robot.x, robot.y), obstacles)

    max_kappa = max_curvature(max(speed, 0.3))
    best = (float("inf"), 0.0, 0.0, 0.0)
    for kappa in np.linspace(-max_kappa, max_kappa, 13):
        # Short arc preview using the identified steady-state speed.
        v = 0.397 * 3.0 * speed
        x, y, yaw = robot.x, robot.y, robot.yaw
        min_clearance = float("inf")
        for step in range(1, 11):
            dt = 0.1
            yaw += v * kappa * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            min_clearance = min(min_clearance, clearance((x, y), obstacles))
        heading_error = abs(wrap_angle(desired - math.atan(v * kappa)))
        penalty = 0.0
        if min_clearance < 0.55:
            penalty += 8.0 * (0.55 - min_clearance)
        if min_clearance < 0.35:
            penalty += 20.0
        score = heading_error + 0.08 * abs(kappa) + penalty
        if score < best[0]:
            best = (score, kappa, speed, min_clearance)
    if best[0] == float("inf"):
        return 0.0, 0.0, clearance((robot.x, robot.y), obstacles)
    _, kappa, speed, min_clearance = best
    if min_clearance < 0.35:
        speed = 0.0
    elif min_clearance < 0.5:
        speed *= 0.5
    return speed, kappa, min_clearance


# ------------------------------------------------------------------ episode


def run_map(scenario, args, worker, camera):
    map_path = Path(scenario.map_file)
    if not map_path.is_absolute():
        cwd_candidate = Path.cwd() / map_path
        map_path = (cwd_candidate if cwd_candidate.exists()
                    else Path(args.maps_root) / map_path)
    raw = load_scene(map_path, scenario.scene_id)
    obstacles, goal, _ = transform_scene(raw)
    bounds = map_bounds(goal, obstacles)
    target_path_robot = plan_path((0.0, 0.0), goal, obstacles,
                                  bounds=bounds,
                                  inflation=args.plan_inflation,
                                  resolution=0.25)
    if len(target_path_robot) < 2:
        raise RuntimeError(f"no A* route for {scenario.scene_id}")
    total_length = path_length(target_path_robot)
    target_start_s = min(1.2, 0.08 * total_length)
    target_goal = goal

    cfg = front_pair_config(
        num_envs=1, num_people=2, frames=args.frames, episodes=1,
        seed=args.seed,
        avatar_dir=str(REPO / "perception/assets/rocketbox"),
        avatar_count=4,
        avatar_list="Business_Male_01,Business_Male_02",
        num_walls=0, num_pillars=0, save_seg_every=1, keep_in_fan=False)
    scene = PeopleScene(cfg)
    video = None
    try:
        add_walls(scene, obstacles)
        scene.reset(0, episode_seed=args.seed)
        people = sorted(scene.people[0], key=lambda person: person.person_id)
        target, distractor = people[0], people[1]
        target_points = [point_at_distance(target_path_robot, s)[0]
                         for s in np.linspace(0.0, total_length, 61)]
        bystander = choose_bystander(obstacles, target_points)
        distractor.pos = bystander.copy()
        distractor.vel = np.zeros(2)
        target.pos = point_at_distance(target_path_robot, target_start_s)[0]
        target.vel = np.zeros(2)

        if args.video:
            import cv2
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(
                str(out / f"{scenario.name}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                (camera.width, camera.height))

        first_delta = target_path_robot[1] - target_path_robot[0]
        initial_yaw = math.atan2(first_delta[1], first_delta[0])
        robot = (HolonomicBase((0.0, 0.0), yaw=initial_yaw)
                 if args.holonomic else P0Robot(yaw=initial_yaw))
        layer = None
        path = (list(target_path_robot) if args.follow_target_route
                else list(target_path_robot))
        last_plan_t = -1.0
        rows = []
        last_visible_t = None
        target_center = target.pos.copy()
        last_seen_target_s = target_start_s
        last_seen_t = 0.0
        last_seen_velocity = np.zeros(2)

        for frame in range(args.frames):
            t = round(frame * 0.1, 6)
            target_s = min(total_length,
                           target_start_s + scenario.target_speed * t)
            target_pos, tangent = point_at_distance(target_path_robot, target_s)
            target.pos = target_pos.copy()
            target.vel = np.zeros(2)
            target.yaw = float(math.atan2(tangent[1], tangent[0]))
            distractor.pos = bystander.copy()
            scene._apply_people(0)
            scene.set_cameras(0, roll_deg=0.0, base_yaw=robot.yaw,
                              base_xy=(robot.x, robot.y))
            views = scene.render(0, save_seg=True)
            left = views[0]
            image = np.asarray(left["color"])
            seg = np.asarray(left["seg"])

            target_mask = seg == target.person_id
            target_pixels = int(target_mask.sum())
            target_box, _ = _bbox_from_mask(target_mask, 30)
            distractor_mask = seg == distractor.person_id
            distractor_box, _ = _bbox_from_mask(distractor_mask, 30)
            target_visible = target_pixels >= 30 and target_box is not None

            raw_detections = deduplicate(worker.detect(image))
            detections = []
            for detection in raw_detections:
                bbox = np.asarray(detection["bbox"], dtype=np.float64)
                world = camera.detection_to_world(
                    bbox, robot_xy=(robot.x, robot.y), robot_yaw=robot.yaw,
                    depth=None, side="left")
                if world is None:
                    continue
                detections.append(PersonDetection(
                    position_xy=(world["position_world_xy"]
                                 if world.get("consistent", True) else None),
                    bbox=bbox,
                    appearance=hsv_histogram(image, bbox),
                    score=float(detection.get("conf", 1.0)),
                    covariance_xy=world["covariance_xy"],
                    meta=dict(range_source=world["range_source"])))

            if layer is None:
                if not target_visible:
                    continue
                best = None
                best_iou = 0.0
                for detection in detections:
                    value = box_iou(detection.bbox, target_box)
                    if value > best_iou:
                        best_iou, best = value, detection
                if best is None or best_iou < 0.3:
                    continue
                layer = TargetPerceptionLayer(
                    init_bbox=best.bbox,
                    init_appearance=best.appearance,
                    max_prediction_s=1.0,
                    max_memory_s=5.0,
                    birth_confidence_min=0.25,
                    init_location_priority=True,
                )
            belief = layer.update(detections, t)

            identity_correct = False
            target_error = None
            if belief.position_xy is not None:
                target_error = float(np.linalg.norm(belief.position_xy
                                                    - target_pos))
                distractor_error = float(np.linalg.norm(belief.position_xy
                                                        - bystander))
                identity_correct = target_error <= distractor_error

            if target_visible:
                last_visible_t = t

            # Track the target's progress along the known map route.  If the
            # visual prediction expires behind a maze wall, continue along
            # that route instead of stopping at the last stale point.
            if (belief.position_xy is not None and not belief.lost):
                last_seen_target_s = path_progress(
                    target_path_robot, belief.position_xy)
                last_seen_t = t
                if belief.velocity_xy is not None:
                    last_seen_velocity = np.asarray(
                        belief.velocity_xy, dtype=float)
            lost_age = t - last_seen_t
            target_point = None
            target_tangent = np.zeros(2)
            if not belief.lost and belief.position_xy is not None:
                target_point = np.asarray(belief.position_xy, dtype=float)
            elif last_seen_target_s is not None and lost_age <= 15.0:
                speed_estimate = max(0.18, float(np.linalg.norm(
                    last_seen_velocity)))
                predicted_s = last_seen_target_s + speed_estimate * lost_age
                target_point, target_tangent = point_at_distance(
                    target_path_robot, predicted_s)
            if target_point is None:
                target_point = target_center
                target_tangent = np.zeros(2)

            # Replan the robot's route to a stand-off point behind the target
            # or its map-aware continuation.
            if (not args.follow_target_route and target_point is not None
                    and (t - last_plan_t >= 0.8 or len(path) < 2)):
                target_xy = np.asarray(target_point, dtype=float)
                velocity = (last_seen_velocity if belief.lost
                            else np.asarray(belief.velocity_xy, dtype=float))
                if np.linalg.norm(velocity) > 0.15:
                    direction = velocity / np.linalg.norm(velocity)
                elif np.linalg.norm(target_tangent) > 1e-6:
                    direction = target_tangent
                else:
                    direction = target_xy - np.array([robot.x, robot.y])
                    direction /= max(np.linalg.norm(direction), 1e-6)
                goal_xy = target_xy - 1.5 * direction
                planned = plan_path((robot.x, robot.y), goal_xy, obstacles,
                                    bounds=bounds,
                                    inflation=args.replan_inflation,
                                    resolution=0.25)
                if len(planned) >= 2:
                    path = planned
                last_plan_t = t

            if args.holonomic:
                look = lookahead_point(path, (robot.x, robot.y), 1.2)
                if look is None:
                    look = np.asarray(target_point)
                desired = math.atan2(look[1] - robot.y,
                                     look[0] - robot.x)
                best = None
                for offset in np.radians([0, 15, -15, 30, -30, 45, -45,
                                          60, -60, 90, -90]):
                    heading = desired + float(offset)
                    direction = np.array([math.cos(heading),
                                          math.sin(heading)])
                    probe = np.array([robot.x, robot.y]) + 0.55 * direction
                    probe_clearance = clearance(probe, obstacles)
                    cost = abs(float(offset)) + 20.0 * max(
                        0.0, 0.55 - probe_clearance)
                    if probe_clearance < 0.40:
                        cost += 50.0
                    if best is None or cost < best[0]:
                        best = (cost, heading, probe_clearance)
                _, heading, probe_clearance = best
                if (probe_clearance < 0.45
                        or (belief.lost and lost_age > 15.0)):
                    vx = vy = 0.0
                else:
                    vx = 0.50 * math.cos(heading)
                    vy = 0.50 * math.sin(heading)
                desired_yaw = math.atan2(target_point[1] - robot.y,
                                         target_point[0] - robot.x)
                yaw_error = wrap_angle(desired_yaw - robot.yaw)
                robot.move(vx, vy, np.clip(2.0 * yaw_error, -1.8, 1.8), 0.1)
            else:
                if belief.lost and lost_age > 15.0:
                    # Do not chase an unbounded map prediction forever.
                    speed, kappa = 0.0, 0.0
                else:
                    speed, kappa, _ = candidate_curvature(
                        robot, target_point, path, obstacles)
                a2 = invert_steer(robot.a1_f, kappa) if robot.a1_f > 0.05 else 0.0
                a1 = speed
                for _ in range(5):
                    robot.tick(a1, a2)

            robot_clearance = clearance((robot.x, robot.y), obstacles)
            target_center = target.pos.copy()
            distance = float(np.linalg.norm(
                np.array([robot.x, robot.y]) - target.pos))
            rows.append(dict(
                t=t,
                robot_xy=[round(robot.x, 3), round(robot.y, 3)],
                robot_yaw=round(robot.yaw, 4),
                robot_v=round(robot.v, 3),
                target_xy=[round(float(target.pos[0]), 3),
                           round(float(target.pos[1]), 3)],
                target_visible=target_visible,
                target_pixels=target_pixels,
                detections=len(detections),
                status=belief.status,
                track_id=belief.track_id,
                identity_correct=bool(identity_correct),
                target_error=None if target_error is None else round(target_error, 3),
                distance=round(distance, 3),
                robot_clearance=round(robot_clearance, 3),
                path_progress=round(target_s / max(total_length, 1e-6), 3),
                lost_age=round(lost_age, 3),
            ))

            if video is not None:
                import cv2
                canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for detection in raw_detections:
                    x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
                    colour = (0, 255, 0)
                    if target_box is not None and box_iou(
                            detection["bbox"], target_box) > 0.3:
                        colour = (255, 255, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                if target_box is not None:
                    x1, y1, x2, y2 = [int(v) for v in target_box]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = (f"{scenario.scene_id}  t={t:.1f}s  {belief.status}  "
                        f"d={distance:.1f}m  clr={robot_clearance:.2f}m  "
                        f"id={belief.track_id}")
                cv2.putText(canvas, text, (18, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                video.write(canvas)

            if frame % 10 == 0:
                print(f"{scenario.name} f{frame:03d} t={t:.1f} "
                      f"target=({target.pos[0]:.1f},{target.pos[1]:.1f}) "
                      f"robot=({robot.x:.1f},{robot.y:.1f}) d={distance:.1f} "
                      f"clr={robot_clearance:.2f} {belief.status}",
                      flush=True)
    finally:
        if video is not None:
            video.release()
        scene.close()

    tracked = [row for row in rows if row["identity_correct"]]
    distances = [row["distance"] for row in rows]
    clearances = [row["robot_clearance"] for row in rows]
    identity = len(tracked) / max(1, len(rows))
    collision_frames = sum(1 for value in clearances if value < 0.42)
    return dict(
        scenario=scenario.name,
        map=scenario.scene_id,
        family=scenario.note,
        frames=len(rows),
        target_path_length=round(total_length, 2),
        identity_accuracy=round(identity, 4),
        mean_distance=round(float(np.mean(distances)), 3),
        min_distance=round(float(np.min(distances)), 3),
        max_distance=round(float(np.max(distances)), 3),
        min_clearance=round(float(np.min(clearances)), 3),
        mean_clearance=round(float(np.mean(clearances)), 3),
        collision_frames=int(collision_frames),
        measured_rate=round(sum(1 for row in rows if row["status"] == "measured")
                            / max(1, len(rows)), 4),
        predicted_rate=round(sum(1 for row in rows if row["status"] == "predicted")
                             / max(1, len(rows)), 4),
        lost_rate=round(sum(1 for row in rows if row["status"] == "lost")
                        / max(1, len(rows)), 4),
        final_progress=rows[-1]["path_progress"] if rows else 0.0,
        success=bool(identity >= 0.9 and collision_frames == 0
                     and 1.2 <= float(np.mean(distances)) <= 5.0),
        rows=rows,
    )


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--map", default="all",
                        choices=["all"] + list(FOLLOW_MAPS))
    parser.add_argument("--map-json", default="",
                        help="single map JSON file; overrides --map")
    parser.add_argument("--scene-id", default="",
                        help="scene id inside --map-json")
    parser.add_argument("--map-name", default="",
                        help="label for --map-json")
    parser.add_argument("--plan-inflation", type=float, default=0.52,
                        help="robot clearance used for the initial A* route")
    parser.add_argument("--replan-inflation", type=float, default=0.48,
                        help="robot clearance used for the stand-off replan")
    parser.add_argument("--follow-target-route", action="store_true",
                        help="keep the robot on the target's maze route instead of replanning around it")
    parser.add_argument("--holonomic", action="store_true",
                        help="use a simple holonomic base instead of the P0 model")
    parser.add_argument("--maps-root", default=str(
        REPO / "perception/assets/maps"))
    parser.add_argument("--frames", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--weights", default="/home/jason/perception_data/runs/"
                                             "l1_front_pair_v1/weights/best.pt")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    camera = FrontPairCamera()
    worker = DetectionWorker(args.weights, conf=0.25)
    if args.map_json:
        selected = [FollowMap(
            name=args.map_name or Path(args.map_json).stem,
            map_file=args.map_json,
            scene_id=args.scene_id,
            target_speed=0.45,
            note="converted open map")]
    else:
        selected = ([FOLLOW_MAPS[name] for name in FOLLOW_MAPS]
                    if args.map == "all" else [FOLLOW_MAPS[args.map]])
    results = []
    try:
        for scenario in selected:
            result = run_map(scenario, args, worker, camera)
            rows = result.pop("rows")
            results.append(result)
            (out / f"{scenario.name}.json").write_text(
                json.dumps(result, indent=2))
            (out / f"{scenario.name}_rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
    finally:
        worker.close()

    summary = dict(
        maps_root=args.maps_root,
        scenarios=results,
        success_rate=float(np.mean([r["success"] for r in results])),
        mean_identity=float(np.mean([r["identity_accuracy"] for r in results])),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
