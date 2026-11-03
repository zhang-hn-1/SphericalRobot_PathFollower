#!/usr/bin/env python3
"""Closed-loop follow in a realistic ReplicaCAD apartment.

Scene
-----
The visual shell is Meta's ReplicaCAD ``frl_apartment_stage`` (CC BY-NC 4.0),
converted to OBJ/URDF.  A real ReplicaCAD refrigerator is placed in the living
area as the occluder.  The target walks laterally behind it; the robot follows
using only YOLO + the world-frame target layer.

The target is hidden for about 1.2 s.  The layer predicts for 1 s, then reports
``lost``; the robot keeps the last prediction and stops only after the horizon
expires.  When the target reappears on the other side, the same track is
re-acquired and the robot continues.

    source local_env.sh
    /home/jason/legged_gym/.venv/bin/python perception/sim/stage_follow.py \
        --out /home/jason/perception_data/runs/stage_follow_v1 \
        --frames 180 --video
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

from l1_front_pair import front_pair_config  # noqa: E402
from map_follow import clearance  # noqa: E402
from occlusion_perception import DetectionWorker, box_iou, deduplicate  # noqa: E402
from people_scene import PeopleScene, _bbox_from_mask  # noqa: E402
from range_bearing import FrontPairCamera  # noqa: E402
from target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402
from target_tracker import hsv_histogram  # noqa: E402

from isaacgym import gymapi  # noqa: E402


# ReplicaCAD objects are in metres; the refrigerator is 0.70 x 0.76 x 1.92 m.
FRIDGE_CENTER = np.array([0.0, -2.20])
FRIDGE_SIZE = (0.75, 0.80, 0.0)
FRIDGE_OBB = (FRIDGE_CENTER, FRIDGE_SIZE[:2], 0.0)
ROBOT_START = np.array([1.00, -4.00])
ROBOT_LANE_Y = -4.00
ROBOT_YAW = math.pi / 2.0
TARGET_WAYPOINTS = (
    np.array([-1.20, -1.10]),
    np.array([ 1.20, -1.10]),
    np.array([ 1.20, -3.20]),
    np.array([-1.20, -3.20]),
)
TARGET_START = TARGET_WAYPOINTS[0]
ROBOT_YAW = math.atan2(TARGET_START[1] - ROBOT_START[1],
                        TARGET_START[0] - ROBOT_START[0])
TARGET_SPEED = 0.40
# Stage floor sits at z ~ 0; the OBJ was shifted so its minimum is 0.
STAGE_Z = 0.08


class StageMap:
    """2D wall occupancy / clearance from the ReplicaCAD stage OBJ.

    The visual stage has no collision body in this lightweight loop.  Vertical
    triangles (walls, doors, stairs) are rasterised into a 5 cm grid and a
    distance transform gives the robot a conservative clearance field.
    """

    def __init__(self, obj_path, resolution=0.05, z_min=0.20, z_max=2.60):
        vertices = []
        faces = []
        for line in Path(obj_path).read_text().splitlines():
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1
                           for token in line.split()[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        self.resolution = float(resolution)
        self.minimum = vertices[:, :2].min(axis=0) - 1.0
        maximum = vertices[:, :2].max(axis=0) + 1.0
        self.width = int(round((maximum[0] - self.minimum[0]) / resolution)) + 1
        self.height = int(round((maximum[1] - self.minimum[1]) / resolution)) + 1
        image = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(image)
        for triangle in faces:
            points = vertices[triangle]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            if points[:, 2].max() < z_min or points[:, 2].min() > z_max:
                continue
            if abs(normal[2]) / norm > 0.5:
                continue  # floor/ceiling
            draw.polygon([
                ((point[0] - self.minimum[0]) / self.resolution,
                 self.height - 1 - (point[1] - self.minimum[1]) / self.resolution)
                for point in points[:, :2]
            ], fill=255)
        occupied = np.asarray(image) > 0
        self.distance = distance_transform_edt(~occupied) * self.resolution

    def clearance(self, point):
        ix = int(round((point[0] - self.minimum[0]) / self.resolution))
        iy = self.height - 1 - int(round(
            (point[1] - self.minimum[1]) / self.resolution))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return 0.0
        return float(self.distance[iy, ix])


class SimpleBase:
    """Holonomic stage base: in-place yaw plus forward speed.

    The P0 identified model cannot turn in place, which makes a target hidden
    behind a refrigerator a low-level-control problem.  This stage demo uses a
    simple holonomic base so the *perception* prediction/re-acquisition is the
    thing being tested; P0 remains the model for the path-following controller.
    """

    def __init__(self, xy, yaw):
        self.x, self.y, self.yaw = float(xy[0]), float(xy[1]), float(yaw)
        self.v = 0.0

    def step(self, v, w, dt):
        self.v = float(v)
        self.yaw = math.atan2(math.sin(self.yaw + w * dt),
                              math.cos(self.yaw + w * dt))
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt

    def move(self, vx, vy, w, dt):
        """Holonomic translation with a separately controlled camera yaw."""
        self.v = math.hypot(vx, vy)
        self.yaw = math.atan2(math.sin(self.yaw + w * dt),
                              math.cos(self.yaw + w * dt))
        self.x += float(vx) * dt
        self.y += float(vy) * dt


def stage_control(robot, target_point, clearance_fn):
    """Turn/move toward the target while sliding around the refrigerator."""
    delta = np.asarray(target_point) - np.array([robot.x, robot.y])
    distance = float(np.linalg.norm(delta))
    desired = math.atan2(delta[1], delta[0])
    best = None
    for offset in np.linspace(-1.5, 1.5, 25):
        heading = desired + offset
        probe = np.array([robot.x, robot.y]) + 1.0 * np.array(
            [math.cos(heading), math.sin(heading)])
        probe_clearance = clearance_fn(probe)
        cost = abs(offset) * 0.8
        if probe_clearance < 0.85:
            cost += 8.0 * (0.85 - probe_clearance)
        if probe_clearance < 0.55:
            cost += 30.0
        if best is None or cost < best[0]:
            best = (cost, heading, probe_clearance)
    _, heading, probe_clearance = best
    angle_error = math.atan2(math.sin(heading - robot.yaw),
                             math.cos(heading - robot.yaw))
    w = float(np.clip(2.0 * angle_error, -1.8, 1.8))
    if distance > 2.5:
        v = 0.60
    elif distance > 1.7:
        v = 0.35
    else:
        v = 0.0
    if clearance_fn((robot.x, robot.y)) < 0.55:
        v = 0.0
    return v, w


def look_at(robot, target_point):
    delta = np.asarray(target_point) - np.array([robot.x, robot.y])
    desired = math.atan2(delta[1], delta[0])
    error = math.atan2(math.sin(desired - robot.yaw),
                       math.cos(desired - robot.yaw))
    return float(np.clip(2.0 * error, -1.8, 1.8))


def load_urdf(scene, root: Path, filename: str):
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.use_mesh_materials = True
    return scene.gym.load_asset(scene.sim, str(root), filename, options)


def add_actor(scene, asset, position, name):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(position[0]), float(position[1]), float(position[2]))
    return scene.gym.create_actor(scene.envs[0], asset, pose, name, 0, 0, 0)


def target_loop_distance(position):
    """Arc length from the first waypoint to the closest loop point."""
    points = TARGET_WAYPOINTS
    best_distance = 0.0
    best_error = float("inf")
    travelled = 0.0
    for index in range(len(points)):
        start = points[index]
        end = points[(index + 1) % len(points)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        unit = segment / length
        projection = float(np.clip(np.dot(np.asarray(position) - start, unit),
                                   0.0, length))
        candidate = start + unit * projection
        error = float(np.linalg.norm(np.asarray(position) - candidate))
        if error < best_error:
            best_error = error
            best_distance = travelled + projection
        travelled += length
    return best_distance


def target_at(t: float):
    """Walk a closed rectangular loop around the living area / refrigerator."""
    points = TARGET_WAYPOINTS
    segments = [points[(index + 1) % len(points)] - points[index]
                for index in range(len(points))]
    lengths = [float(np.linalg.norm(segment)) for segment in segments]
    total = float(sum(lengths))
    distance = (TARGET_SPEED * t) % total
    for index, length in enumerate(lengths):
        if distance <= length:
            unit = segments[index] / length
            return points[index] + unit * distance, unit
        distance -= length
    unit = segments[-1] / lengths[-1]
    return points[-1], unit


def run(args):
    camera = FrontPairCamera()
    worker = DetectionWorker(args.weights, conf=0.25)
    policy = None
    if args.policy:
        sys.path.insert(0, str(REPO / "perception/research"))
        from train_predictive_follow import Actor as PolicyActor
        import torch
        policy = PolicyActor(15)
        policy.load_state_dict(torch.load(args.policy, map_location="cpu"))
        policy.eval()
    predictor = None
    if args.predictor:
        from perception.learned_predictor import TrajectoryPredictor
        predictor = TrajectoryPredictor(args.predictor)
    conformal = None
    if args.conformal_json:
        raw = json.loads(Path(args.conformal_json).read_text())
        conformal = raw.get("radii", {}).get(args.conformal_level, {})

    def conformal_radius(age):
        if not conformal:
            return 0.0
        points = sorted((float(key.rstrip("s")), float(value))
                        for key, value in conformal.items())
        if age <= points[0][0]:
            return points[0][1]
        for (low_t, low_r), (high_t, high_r) in zip(points, points[1:]):
            if low_t <= age <= high_t:
                alpha = (age - low_t) / (high_t - low_t)
                return low_r * (1.0 - alpha) + high_r * alpha
        return points[-1][1]
    cfg = front_pair_config(
        num_envs=1, num_people=max(2, min(4, args.num_people)),
        frames=args.frames, episodes=1,
        seed=args.seed,
        avatar_dir=str(REPO / "perception/assets/rocketbox"),
        avatar_count=4,
        avatar_list="Business_Male_01,Business_Male_02,"
                    "Business_Female_01,Business_Female_02",
        num_walls=0, num_pillars=0, save_seg_every=1, keep_in_fan=False)
    scene = PeopleScene(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = None
    try:
        stage_asset = load_urdf(
            scene, REPO / "perception/assets/scenes/replica_stage", "stage.urdf")
        add_actor(scene, stage_asset, (0.0, 0.0, STAGE_Z), "replica_stage")
        fridge_asset = load_urdf(
            scene, REPO / "perception/assets/scenes/replica_refrigerator",
            "refrigerator.urdf")
        add_actor(scene, fridge_asset,
                  (FRIDGE_CENTER[0], FRIDGE_CENTER[1], STAGE_Z), "refrigerator")
        stage_map = StageMap(
            REPO / "perception/assets/scenes/replica_stage/stage.obj")
        scene.reset(0, episode_seed=args.seed)
        people = sorted(scene.people[0], key=lambda person: person.person_id)
        target, bystander = people[0], people[1]
        bystander.pos = np.array([2.30, -3.60])
        bystander.vel = np.zeros(2)
        extra_positions = [np.array([-2.20, -3.75]),
                           np.array([2.70, -1.45])]
        extras = people[2:]
        for person, position in zip(extras, extra_positions):
            person.pos = position.copy()
            person.vel = np.zeros(2)
        target.pos, _ = target_at(0.0)
        target.vel = np.zeros(2)
        robot = SimpleBase(ROBOT_START, ROBOT_YAW)

        def scene_clearance(point):
            return min(stage_map.clearance(point),
                       clearance(point, [FRIDGE_OBB]))

        if args.video:
            import cv2
            video = cv2.VideoWriter(
                str(out / "stage_follow.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                (camera.width, camera.height))

        layer = None
        rows = []
        last_target_point = TARGET_START.copy()
        last_velocity = np.zeros(2)
        lost_since = None
        observe_only = False
        search_direction = 1.0
        search_flip_t = 0.0
        trail = [TARGET_START.copy()]
        last_visible_t = None
        hidden_started = None
        hidden_ended = None
        reacquired_t = None
        id_before_hidden = None
        reacquired_id = None
        path = [np.array([robot.x, ROBOT_LANE_Y]),
                np.array([TARGET_START[0], ROBOT_LANE_Y])]

        for frame in range(args.frames):
            frame_started = time.perf_counter()
            t = round(frame * 0.1, 6)
            target_pos, tangent = target_at(t)
            target.pos = target_pos.copy()
            target.vel = np.zeros(2)
            target.yaw = float(math.atan2(tangent[1], tangent[0]))
            bystander.pos = np.array([2.30, -3.60])
            for person, position in zip(extras, extra_positions):
                person.pos = position.copy()
            scene._apply_people(0)
            scene.set_cameras(0, roll_deg=0.0, base_yaw=robot.yaw,
                              base_xy=(robot.x, robot.y))
            render_started = time.perf_counter()
            views = scene.render(0, save_seg=True, save_depth=True)
            render_ms = (time.perf_counter() - render_started) * 1000.0
            left = views[0]
            image = np.asarray(left["color"])
            if args.image_gain != 1.0 or args.image_noise > 0.0:
                image_rng = np.random.default_rng(args.seed + frame)
                image = image.astype(np.float32) * args.image_gain
                image += image_rng.normal(
                    0.0, args.image_noise * 255.0, image.shape)
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
            seg = np.asarray(left["seg"])

            target_mask = seg == target.person_id
            target_pixels = int(target_mask.sum())
            target_box, _ = _bbox_from_mask(target_mask, 30)
            bystander_mask = seg == bystander.person_id
            bystander_box, _ = _bbox_from_mask(bystander_mask, 30)
            target_visible = target_pixels >= 30 and target_box is not None

            detect_started = time.perf_counter()
            raw_detections = deduplicate(worker.detect(image))
            detect_ms = (time.perf_counter() - detect_started) * 1000.0
            detections = []
            for detection in raw_detections:
                bbox = np.asarray(detection["bbox"], dtype=np.float64)
                world = camera.detection_to_world(
                    bbox, robot_xy=(robot.x, robot.y), robot_yaw=robot.yaw,
                    depth=left["depth"], side="left")
                if world is None:
                    continue
                range_ratio = world.get("range_ratio")
                # A partial upper-body box makes the height prior say "far
                # away"; a valid depth that is closer than that prior is the
                # reliable range in this case.  A depth much farther than the
                # height prior is more likely a wall/occluder, so drop it.
                use_position = bool(world.get("consistent", True))
                if (not use_position and world.get("depth_m") is not None
                        and range_ratio is not None and range_ratio < 0.67):
                    use_position = True
                detections.append(PersonDetection(
                    position_xy=(world["position_world_xy"]
                                 if use_position else None),
                    bbox=bbox,
                    appearance=hsv_histogram(image, bbox),
                    score=float(detection.get("conf", 1.0)),
                    covariance_xy=world["covariance_xy"],
                    meta=dict(range_source=world["range_source"],
                              consistent=bool(world.get("consistent", True)),
                              range_ratio=range_ratio,
                              depth_m=world.get("depth_m"),
                              height_range_m=world.get("height_range_m"))))

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
                    reacquire_cosine_min=0.55,
                    birth_confidence_min=0.25,
                    init_location_priority=True,
                )
            track_started = time.perf_counter()
            belief = layer.update(detections, t)
            track_ms = (time.perf_counter() - track_started) * 1000.0

            identity_correct = False
            target_error = None
            if belief.position_xy is not None:
                target_error = float(np.linalg.norm(
                    belief.position_xy - target_pos))
                bystander_error = float(np.linalg.norm(
                    belief.position_xy - bystander.pos))
                identity_correct = target_error <= bystander_error

            if target_visible:
                if hidden_started is not None and hidden_ended is None:
                    hidden_ended = t
                if (hidden_ended is not None and reacquired_t is None
                        and belief.status == "measured" and identity_correct):
                    reacquired_t = t
                    reacquired_id = belief.track_id
                last_visible_t = t
                if hidden_started is None:
                    id_before_hidden = belief.track_id
            else:
                if hidden_started is None:
                    hidden_started = t
                if (reacquired_t is None and belief.status == "measured"
                        and identity_correct):
                    reacquired_t = t
                    reacquired_id = belief.track_id

            if belief.position_xy is not None and not belief.lost:
                last_target_point = np.asarray(belief.position_xy).copy()
                if belief.velocity_xy is not None:
                    last_velocity = np.asarray(belief.velocity_xy).copy()
                lost_since = None
            elif belief.lost and lost_since is None:
                lost_since = t
            target_point = (belief.position_xy if belief.position_xy is not None
                            else last_target_point)
            # Follow a point 1.5 m behind the target along the target's own
            # trail.  The target route is collision-free, so the robot follows
            # the same corridor instead of cutting through a wall.
            if belief.position_xy is not None and not belief.lost:
                trail_point = np.asarray(belief.position_xy, dtype=np.float64)
                if float(np.linalg.norm(trail_point - trail[-1])) > 0.05:
                    trail.append(trail_point.copy())
                    if len(trail) > 400:
                        trail = trail[-400:]
            follow_point = trail[-1].copy()
            remaining = 1.5
            for index in range(len(trail) - 1, 0, -1):
                segment = trail[index] - trail[index - 1]
                length = float(np.linalg.norm(segment))
                if length <= 1e-9:
                    continue
                if remaining <= length:
                    follow_point = trail[index] - segment / length * remaining
                    break
                remaining -= length
            predictive_move = False
            lost_age = (max(0.0, t - lost_since)
                        if lost_since is not None else 0.0)
            prediction_age = (belief.measurement_age_s
                              if belief.measurement_age_s is not None
                              else lost_age)
            learned_prediction = None
            predictor_started = time.perf_counter()
            if predictor is not None and belief.lost:
                learned_prediction = predictor.predict_belief(
                    belief, prediction_age)
            predictor_ms = (time.perf_counter() - predictor_started) * 1000.0
            prediction_radius = (conformal_radius(prediction_age)
                                 if conformal is not None and belief.lost
                                 else None)
            conformal_scale = 1.0
            if prediction_radius is not None:
                conformal_scale = max(0.25, 1.0 - prediction_radius / 0.50)
            policy_action = None
            policy_ms = 0.0
            if policy is not None:
                predicted_policy = (learned_prediction
                                    if learned_prediction is not None
                                    else belief.position_xy
                                    if belief.position_xy is not None
                                    else last_target_point)
                target_velocity_policy = (belief.velocity_xy
                                          if belief.velocity_xy is not None
                                          else last_velocity)
                delta_policy = (np.asarray(predicted_policy)
                                - np.array([robot.x, robot.y]))
                distance_policy = float(np.linalg.norm(delta_policy))
                probe_policy = np.array([
                    robot.x + np.clip(delta_policy[0], -1.0, 1.0) * 0.4,
                    robot.y + np.clip(delta_policy[1], -1.0, 1.0) * 0.4])
                policy_obs = np.array([
                    float(np.clip(delta_policy[0] / 5.0, -1.0, 1.0)),
                    float(np.clip(delta_policy[1] / 5.0, -1.0, 1.0)),
                    float(np.clip(target_velocity_policy[0] / 1.0, -1.0, 1.0)),
                    float(np.clip(target_velocity_policy[1] / 1.0, -1.0, 1.0)),
                    float(np.clip(lost_age / 3.0, 0.0, 1.0)),
                    float(belief.velocity_confidence),
                    float(np.clip(belief.prediction_uncertainty_m / 1.0, 0.0, 1.0)),
                    1.0 if belief.target_visible else 0.0,
                    float(np.clip(distance_policy / 6.0, 0.0, 1.0)),
                    float(np.clip(clearance((robot.x, robot.y),
                                            [FRIDGE_OBB]) / 2.0, 0.0, 1.0)),
                    float(np.clip(clearance(probe_policy, [FRIDGE_OBB]) / 2.0,
                                  0.0, 1.0)),
                    float(np.clip((robot.y - ROBOT_LANE_Y) / 1.0, -1.0, 1.0)),
                    float(np.clip((predicted_policy[0] - robot.x) / 2.0,
                                  -1.0, 1.0)),
                    float(np.clip(predicted_policy[0] / 3.0, -1.0, 1.0)),
                    float(np.clip(target_loop_distance(predicted_policy) / 6.0,
                                  0.0, 1.0)),
                ], dtype=np.float32)
                policy_started = time.perf_counter()
                with torch.no_grad():
                    policy_action = float(policy(torch.from_numpy(
                        policy_obs).unsqueeze(0)))
                policy_ms = (time.perf_counter() - policy_started) * 1000.0
                if args.residual:
                    base_target = (target_point if belief.target_visible
                                   else learned_prediction
                                   if learned_prediction is not None
                                   else last_target_point)
                    base_vx = float(np.clip(
                        0.8 * (np.asarray(base_target)[0] - robot.x),
                        -0.55, 0.55))
                    policy_action = float(np.clip(
                        base_vx / 0.60
                        + args.residual_scale * policy_action, -1.0, 1.0))
            if policy_action is not None and belief.lost:
                # The supervised predictor is the local target model.  If the
                # RL action points away from its prediction, or if the model
                # has no usable history, fall back to the rule controller.
                if learned_prediction is None:
                    policy_action = None
                else:
                    delta_x = float(learned_prediction[0] - robot.x)
                    if (abs(delta_x) > 0.20
                            and policy_action * delta_x < 0.0):
                        policy_action = None
            if (policy_action is not None and belief.target_visible):
                # The actor was trained while acting at every step, so use it
                # in the visible phase too instead of switching policies.
                vx = float(np.clip(0.60 * policy_action, -0.55, 0.55))
                vy = float(np.clip(0.9 * (ROBOT_LANE_Y - robot.y),
                                   -0.30, 0.30))
                probe = np.array([robot.x + vx * 0.4, robot.y + vy * 0.4])
                if scene_clearance(probe) >= 0.55:
                    robot.move(vx, vy, look_at(robot, target_point), 0.1)
                else:
                    robot.move(0.0, 0.0, look_at(robot, target_point), 0.1)
            elif (policy_action is not None and belief.lost
                    and lost_age <= 3.0):
                vx = float(np.clip(0.60 * policy_action * conformal_scale,
                                   -0.45, 0.45))
                vy = float(np.clip(0.9 * (ROBOT_LANE_Y - robot.y),
                                   -0.25, 0.25))
                probe = np.array([robot.x + vx * 0.4, robot.y + vy * 0.4])
                if scene_clearance(probe) >= 0.55:
                    robot.move(vx, vy, look_at(robot, last_target_point), 0.1)
                    predictive_move = True
                else:
                    robot.move(0.0, 0.0, look_at(robot, last_target_point), 0.1)
            elif (belief.lost and lost_age <= args.predict_follow_s
                    and belief.velocity_confidence >= args.velocity_confidence_min
                    and float(np.linalg.norm(last_velocity)) > 0.05):
                # The target disappeared, but its recent velocity is still
                # coherent.  Continue in that direction, on the safe lane,
                # with a speed that decays over the prediction window.
                predicted = (learned_prediction if learned_prediction is not None
                             else last_target_point
                             + last_velocity * prediction_age)
                decay = (max(0.0, 1.0 - lost_age / args.predict_follow_s)
                         ** 1.5 * conformal_scale)
                vx = float(np.clip(
                    args.predict_speed_scale * 0.8
                    * (predicted[0] - robot.x) * decay, -0.40, 0.40))
                vy = float(np.clip(0.9 * (ROBOT_LANE_Y - robot.y),
                                   -0.25, 0.25))
                probe = np.array([robot.x + vx * 0.4, robot.y + vy * 0.4])
                if scene_clearance(probe) >= 0.55:
                    robot.move(vx, vy, look_at(robot, predicted), 0.1)
                    predictive_move = True
                else:
                    robot.move(0.0, 0.0, look_at(robot, predicted), 0.1)
            elif belief.lost:
                # The prediction is no longer trustworthy; stop and sweep the
                # camera until the target comes back into view.
                if lost_since is not None and t - lost_since > 1.5:
                    search_direction *= -1.0
                    lost_since = t
                robot.move(0.0, 0.0, 0.8 * search_direction, 0.1)
            else:
                # Stay on the clear observation lane; follow the target's
                # lateral x while the camera independently looks at it.
                cautious = 0.70 if belief.status == "predicted" else 1.0
                vx = float(np.clip(0.8 * (target_point[0] - robot.x)
                                   * cautious, -0.55, 0.55))
                vy = float(np.clip(0.9 * (ROBOT_LANE_Y - robot.y),
                                   -0.30, 0.30))
                probe = np.array([robot.x + vx * 0.4, robot.y + vy * 0.4])
                if scene_clearance(probe) < 0.55:
                    vx = vy = 0.0
                robot.move(vx, vy, look_at(robot, target_point), 0.1)

            robot_clearance = scene_clearance((robot.x, robot.y))
            distance = float(np.linalg.norm(
                np.array([robot.x, robot.y]) - target.pos))
            hidden_age = (None if last_visible_t is None or target_visible
                          else t - last_visible_t)
            total_ms = (time.perf_counter() - frame_started) * 1000.0
            rows.append(dict(
                t=t,
                robot_xy=[round(robot.x, 3), round(robot.y, 3)],
                robot_yaw=round(robot.yaw, 4),
                robot_v=round(robot.v, 3),
                target_xy=[round(float(target.pos[0]), 3),
                           round(float(target.pos[1]), 3)],
                target_visible=target_visible,
                target_pixels=target_pixels,
                hidden_age=None if hidden_age is None else round(hidden_age, 3),
                detections=len(detections),
                status=belief.status,
                track_id=belief.track_id,
                identity_correct=bool(identity_correct),
                target_error=None if target_error is None else round(target_error, 3),
                predictive_move=bool(predictive_move),
                policy_action=(None if policy_action is None else
                               round(policy_action, 3)),
                policy_obs=(policy_obs.tolist()
                            if policy is not None else None),
                learned_prediction=(None if learned_prediction is None
                                    else [round(float(learned_prediction[0]), 3),
                                          round(float(learned_prediction[1]), 3)]),
                conformal_radius=(None if prediction_radius is None
                                  else round(float(prediction_radius), 3)),
                distance=round(distance, 3),
                robot_clearance=round(robot_clearance, 3),
                latency_ms=dict(render=round(render_ms, 3),
                                detect=round(detect_ms, 3),
                                track=round(track_ms, 3),
                                predictor=round(predictor_ms, 3),
                                policy=round(policy_ms, 3),
                                total=round(total_ms, 3)),
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
                text = (f"ReplicaCAD stage  t={t:.1f}s  {belief.status}  "
                        f"age={belief.measurement_age_s or 0.0:.1f}s  "
                        f"d={distance:.1f}m  clr={robot_clearance:.2f}m")
                cv2.putText(canvas, text, (18, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                video.write(canvas)

            if frame % 10 == 0:
                print(f"frame {frame:03d} t={t:.1f} "
                      f"target=({target.pos[0]:.2f},{target.pos[1]:.2f}) "
                      f"robot=({robot.x:.2f},{robot.y:.2f}) d={distance:.2f} "
                      f"clr={robot_clearance:.2f} {belief.status}", flush=True)
    finally:
        if video is not None:
            video.release()
        worker.close()
        scene.close()

    visible = [row for row in rows if row["target_visible"]]
    hidden = [row for row in rows if not row["target_visible"]]
    measured_errors = [row["target_error"] for row in rows
                       if row["status"] == "measured"
                       and row["target_error"] is not None]
    hidden_errors = [row["target_error"] for row in hidden
                     if row["target_error"] is not None]
    predicted_hidden = [row for row in hidden if row["status"] == "predicted"]
    lost_hidden = [row for row in hidden if row["status"] == "lost"]
    wrong = [row for row in rows
             if not row["identity_correct"] and row["status"] != "lost"]
    distances = [row["distance"] for row in rows]
    clearances = [row["robot_clearance"] for row in rows]
    latency = {}
    for key in ("render", "detect", "track", "predictor", "policy", "total"):
        values = [row["latency_ms"][key] for row in rows]
        latency[key] = dict(
            mean_ms=round(float(np.mean(values)), 3),
            p95_ms=round(float(np.percentile(values, 95)), 3),
            max_ms=round(float(np.max(values)), 3),
        )
    # Longest continuous hidden interval.
    longest_hidden = 0
    current_hidden = 0
    hidden_intervals = []
    previous_visible = True
    for row in rows:
        if row["target_visible"]:
            if not previous_visible and current_hidden:
                hidden_intervals.append(current_hidden)
            current_hidden = 0
            previous_visible = True
        else:
            current_hidden += 1
            longest_hidden = max(longest_hidden, current_hidden)
            if previous_visible:
                previous_visible = False
    if current_hidden:
        hidden_intervals.append(current_hidden)
    error_at = {}
    for checkpoint in (0.5, 1.0):
        candidates = [row for row in hidden
                      if row["hidden_age"] is not None
                      and abs(row["hidden_age"] - checkpoint) <= 0.051
                      and row["target_error"] is not None
                      and row["status"] != "lost"]
        if candidates:
            error_at[f"{checkpoint:.1f}"] = float(np.mean(
                [row["target_error"] for row in candidates]))
    return dict(
        scene="ReplicaCAD frl_apartment_stage + refrigerator",
        frames=len(rows),
        visible_frames=len(visible),
        hidden_frames=len(hidden),
        identity_accuracy=1.0 - len(wrong) / max(1, len(rows)),
        wrong_frames=len(wrong),
        measured_rate=sum(1 for row in rows if row["status"] == "measured") / max(1, len(rows)),
        predicted_rate=sum(1 for row in rows if row["status"] == "predicted") / max(1, len(rows)),
        lost_rate=sum(1 for row in rows if row["status"] == "lost") / max(1, len(rows)),
        predictive_follow_frames=sum(
            1 for row in rows if row.get("predictive_move")),
        predictive_follow_rate=sum(
            1 for row in rows if row.get("predictive_move")) / max(1, len(rows)),
        learned_prediction_frames=sum(
            1 for row in rows if row.get("learned_prediction") is not None),
        mean_distance=round(float(np.mean(distances)), 3),
        min_distance=round(float(np.min(distances)), 3),
        max_distance=round(float(np.max(distances)), 3),
        min_clearance=round(float(np.min(clearances)), 3),
        collision_frames=sum(1 for value in clearances if value < 0.42),
        latency_ms=latency,
        hidden_prediction_rate=len(predicted_hidden) / max(1, len(hidden)),
        hidden_lost_rate=len(lost_hidden) / max(1, len(hidden)),
        hidden_rmse=(float(np.sqrt(np.mean(np.square(hidden_errors))))
                     if hidden_errors else None),
        prediction_error_at=error_at,
        hidden_start=hidden_started,
        hidden_end=hidden_ended,
        longest_hidden_s=round(longest_hidden * 0.1, 2),
        hidden_intervals_s=[round(count * 0.1, 1) for count in hidden_intervals],
        reacquisition_delay=(None if reacquired_t is None or hidden_ended is None
                             else round(reacquired_t - hidden_ended, 3)),
        same_track=(None if id_before_hidden is None or reacquired_id is None
                    else id_before_hidden == reacquired_id),
        success=bool(1.0 - len(wrong) / max(1, len(rows)) >= 0.9
                     and sum(1 for value in clearances if value < 0.42) == 0
                     and len(hidden) >= 10),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--weights", default="/home/jason/perception_data/runs/"
                                             "l1_front_pair_v1/weights/best.pt")
    parser.add_argument("--search-s", type=float, default=6.0,
                        help="bounded open-loop search after the prediction expires")
    parser.add_argument("--predict-follow-s", type=float, default=2.0,
                        help="continue along the last velocity for this long")
    parser.add_argument("--velocity-confidence-min", type=float, default=0.35,
                        help="minimum velocity confidence for predictive movement")
    parser.add_argument("--predict-speed-scale", type=float, default=0.55,
                        help="maximum speed scale while following a prediction")
    parser.add_argument("--policy", default="",
                        help="optional PPO actor (.pt) for the lateral action")
    parser.add_argument("--predictor", default="",
                        help="optional supervised trajectory predictor (.pt)")
    parser.add_argument("--residual", action="store_true",
                        help="treat --policy as a correction on the rule action")
    parser.add_argument("--residual-scale", type=float, default=0.4,
                        help="maximum residual correction in normalized action units")
    parser.add_argument("--conformal-json", default="",
                        help="split-conformal radii JSON for a safety speed cap")
    parser.add_argument("--conformal-level", default="0.95",
                        help="calibration level key, e.g. 0.90 or 0.95")
    parser.add_argument("--num-people", type=int, default=2,
                        help="scene people count (2-4; extras are static distractors)")
    parser.add_argument("--image-gain", type=float, default=1.0,
                        help="camera brightness gain for domain-shift tests")
    parser.add_argument("--image-noise", type=float, default=0.0,
                        help="camera Gaussian noise std in [0,1] for domain-shift tests")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    result = run(args)
    rows = result.pop("rows")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
