#!/usr/bin/env python3
"""Perception in polygon map scenes: pass behind a wall, disappear, reappear.

The map JSONs are the reusable 30x30 m polygon layouts that already exist for
the navigation experiments (``rl_validation.json`` / ``rl_test.json``).  Each
scene is transformed so the robot sits at the navigation start and looks along
start -> goal; the polygons become vertical wall boxes (oriented bounding
boxes) in Isaac Gym.  A person walks along a hand-picked corridor that is clear
of walls but passes behind one of them.  The detector, appearance cue and
target layer are exactly the same as in the regular perception loop.

Three scenarios are bundled:

* ``irregular_clutter_pass``: the target crosses behind a clutter block and
  reappears on the other side (natural occlusion, no U-turn).
* ``forced_hairpin_uturn``: the target walks behind a wall, turns around and
  comes back (longer occlusion, tests lost/re-acquisition).
* ``u_goal_uturn``: same idea in a different map family.

    source local_env.sh
    /home/jason/legged_gym/.venv/bin/python perception/sim/map_perception.py \
        --out /home/jason/perception_data/runs/map_perception_v1 \
        --scenario all --frames 65 --video
"""

from __future__ import annotations

import argparse
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

from l1_front_pair import front_pair_config  # noqa: E402
from occlusion_perception import (DetectionWorker, deduplicate,  # noqa: E402
                                  box_iou)
from people_scene import PeopleScene, _bbox_from_mask  # noqa: E402
from range_bearing import FrontPairCamera  # noqa: E402
from target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402
from target_tracker import hsv_histogram  # noqa: E402

from isaacgym import gymapi  # noqa: E402


# ------------------------------------------------------------------ maps


@dataclass(frozen=True)
class MapScenario:
    name: str
    map_file: str
    scene_id: str
    x: float
    y0: float
    y1: float
    speed: float = 1.0
    mode: str = "pass"  # pass / uturn
    note: str = ""


SCENARIOS = (
    MapScenario(
        name="irregular_clutter_pass",
        map_file="rl_test.json",
        scene_id="mb30v4_test_432",
        x=6.5, y0=-1.5, y1=1.0, speed=0.4, mode="pass",
        note="map_013_irregular_clutter: pass behind a block",
    ),
    MapScenario(
        name="forced_hairpin_uturn",
        map_file="rl_validation.json",
        scene_id="mb30v4_test_039",
        x=8.5, y0=1.0, y1=3.5, speed=1.0, mode="uturn",
        note="map_004_forced_hairpin: wall, U-turn, reacquire",
    ),
    MapScenario(
        name="u_goal_uturn",
        map_file="rl_test.json",
        scene_id="mb30v4_test_038",
        x=8.5, y0=1.0, y1=3.8, speed=1.0, mode="uturn",
        note="map_012_u_goal_inside: wall, U-turn, reacquire",
    ),
)


def load_scene(path: Path, scene_id: str):
    blob = json.loads(path.read_text())
    # A single-scene file is convenient for converted open datasets.
    if isinstance(blob, dict) and "scenes" not in blob and "obstacles" in blob:
        if not scene_id or blob.get("id") == scene_id:
            return blob
        raise KeyError(f"{scene_id} not in single-scene file {path}")
    scenes = blob["scenes"] if isinstance(blob, dict) else blob
    for scene in scenes:
        if scene["id"] == scene_id:
            return scene
    raise KeyError(f"{scene_id} not in {path}")


def obb_from_polygon(vertices):
    points = np.asarray(vertices, dtype=np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    yaw = math.atan2(axis[1], axis[0])
    if yaw > math.pi / 2.0:
        yaw -= math.pi
    if yaw < -math.pi / 2.0:
        yaw += math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    local = (points - center) @ rotation.T
    size = local.max(axis=0) - local.min(axis=0)
    return center, np.maximum(size, 0.12), yaw


def transform_scene(scene):
    """Put the robot at the map start, looking along start -> goal."""
    start = np.asarray(scene["start"][:2], dtype=np.float64)
    goal = np.asarray(scene["goal"], dtype=np.float64)
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    c, s = math.cos(-yaw), math.sin(-yaw)

    def transform(point):
        delta = np.asarray(point, dtype=np.float64) - start
        return np.array([c * delta[0] - s * delta[1],
                         s * delta[0] + c * delta[1]])

    polygons = [
        [transform(vertex) for vertex in obstacle["vertices"]]
        for obstacle in scene.get("obstacles", [])
        if obstacle.get("shape") == "polygon"
    ]
    return [obb_from_polygon(poly) for poly in polygons], transform(goal), yaw


def add_walls(scene, obstacles, height=1.5):
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    for center, (size_x, size_y), yaw in obstacles:
        asset = scene.gym.create_box(
            scene.sim, float(size_x), float(size_y), float(height), options)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(center[0]), float(center[1]), height / 2.0)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), float(yaw))
        actor = scene.gym.create_actor(
            scene.envs[0], asset, pose, "map_wall", 0, 0, 0)
        scene.gym.set_rigid_body_color(
            scene.envs[0], actor, 0, gymapi.MESH_VISUAL,
            gymapi.Vec3(0.46, 0.46, 0.50))


def point_inside(point, obstacle):
    center, (size_x, size_y), yaw = obstacle
    delta = np.asarray(point) - center
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([c * delta[0] + s * delta[1],
                      -s * delta[0] + c * delta[1]])
    return abs(local[0]) <= size_x / 2.0 and abs(local[1]) <= size_y / 2.0


def line_blocked(point, obstacles, samples=50):
    point = np.asarray(point, dtype=np.float64)
    for alpha in np.linspace(0.05, 0.95, samples):
        sample = point * alpha
        if any(point_inside(sample, obstacle) for obstacle in obstacles):
            return True
    return False


def choose_distractor(scenario, obstacles):
    """Pick a visible, map-clear bystander far from the target's path."""
    span = abs(scenario.y1 - scenario.y0)
    duration = span / max(scenario.speed, 1e-6)
    if scenario.mode == "uturn":
        duration *= 2.0
    target_points = [target_path(scenario, t)
                     for t in np.linspace(0.0, duration, 21)]
    candidates = [
        (3.0, -1.0), (3.0, 1.0), (3.0, -2.0), (3.0, 2.0),
        (4.0, -2.0), (4.0, 2.0), (5.0, -2.5), (5.0, 2.5),
        (scenario.x - 4.0, scenario.y0 - 1.0),
        (scenario.x - 4.0, scenario.y0 + 1.0),
        (scenario.x - 3.0, scenario.y0 - 2.0),
        (scenario.x - 3.0, scenario.y0 + 2.0),
    ]
    best_point = None
    best_score = -1.0
    for x, y in candidates:
        point = np.array([x, y])
        if any(point_inside(point, obstacle) for obstacle in obstacles):
            continue
        if line_blocked(point, obstacles):
            continue
        distance = float(np.linalg.norm(point))
        if distance < 3.0 or distance > 12.0:
            continue
        if abs(math.atan2(y, x)) > math.radians(35.0):
            continue
        separation = min(float(np.linalg.norm(point - target))
                         for target in target_points)
        if separation < 2.0:
            continue
        if separation > best_score:
            best_score, best_point = separation, point
    return best_point if best_point is not None else np.array([3.5, 0.0])


# ------------------------------------------------------------------ episode


def target_path(scenario: MapScenario, t: float):
    span = abs(scenario.y1 - scenario.y0)
    direction = 1.0 if scenario.y1 >= scenario.y0 else -1.0
    travel = scenario.speed * t
    if scenario.mode == "uturn":
        if travel <= span:
            y = scenario.y0 + direction * travel
        else:
            back = travel - span
            y = scenario.y1 - direction * min(back, span)
    else:
        y = scenario.y0 + direction * min(travel, span)
    return np.array([scenario.x, y])


def run_scenario(scenario, args, worker, camera):
    maps_root = Path(args.maps_root)
    raw = load_scene(maps_root / scenario.map_file, scenario.scene_id)
    obstacles, goal, _ = transform_scene(raw)

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
        distractor_xy = choose_distractor(scenario, obstacles)
        distractor.pos = distractor_xy.copy()
        distractor.vel = np.zeros(2)
        target.pos = target_path(scenario, 0.0)
        target.vel = np.zeros(2)

        if args.video:
            import cv2
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(
                str(out / f"{scenario.name}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                (camera.width, camera.height))

        layer = None
        rows = []
        last_visible_t = None
        hidden_started = None
        hidden_ended = None
        reacquired_t = None
        id_before_hidden = None
        reacquired_id = None

        for frame in range(args.frames):
            t = round(frame * 0.1, 6)
            target_pos = target_path(scenario, t)
            target.pos = target_pos.copy()
            target.vel = np.zeros(2)
            distractor.pos = distractor_xy.copy()
            scene._apply_people(0)
            scene.set_cameras(0, roll_deg=0.0, base_yaw=0.0,
                              base_xy=(0.0, 0.0))
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
                    bbox, robot_xy=(0.0, 0.0), robot_yaw=0.0,
                    depth=None, side="left")
                if world is None:
                    continue
                appearance = hsv_histogram(image, bbox)
                detections.append(PersonDetection(
                    position_xy=(world["position_world_xy"]
                                 if world.get("consistent", True) else None),
                    bbox=bbox, appearance=appearance,
                    score=float(detection.get("conf", 1.0)),
                    covariance_xy=world["covariance_xy"],
                    meta=dict(range_source=world["range_source"])))

            detection_target_iou = (max(
                (box_iou(detection.bbox, target_box) for detection in detections),
                default=0.0) if target_box is not None else 0.0)
            target_detected = detection_target_iou >= 0.3

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

            if belief.position_xy is not None:
                target_error = float(np.linalg.norm(
                    belief.position_xy - target_pos))
                distractor_error = float(np.linalg.norm(
                    belief.position_xy - distractor_xy))
                identity_correct = target_error <= distractor_error
            else:
                target_error = None
                identity_correct = False

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

            hidden_age = (None if last_visible_t is None or target_visible
                          else t - last_visible_t)
            rows.append(dict(
                t=t,
                target_visible=target_visible,
                target_pixels=target_pixels,
                hidden_age=None if hidden_age is None else round(hidden_age, 3),
                detections=len(detections),
                detection_target_iou=round(detection_target_iou, 3),
                target_detected=bool(target_detected),
                status=belief.status,
                track_id=belief.track_id,
                measurement_age=(None if belief.measurement_age_s is None
                                 else round(belief.measurement_age_s, 3)),
                identity_correct=bool(identity_correct),
                target_error=(None if target_error is None
                              else round(target_error, 3)),
                target_position=[round(float(v), 3) for v in target_pos],
                distractor_position=[round(float(v), 3) for v in distractor_xy],
                position=(None if belief.position_xy is None else
                          [round(float(v), 3) for v in belief.position_xy]),
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
                        f"age={belief.measurement_age_s or 0.0:.1f}s  "
                        f"id={belief.track_id}  "
                        f"err={target_error if target_error is not None else -1:.2f}m")
                cv2.putText(canvas, text, (18, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                video.write(canvas)

            if frame % 10 == 0:
                print(f"{scenario.name} frame {frame:03d} t={t:.1f} "
                      f"visible={target_visible} status={belief.status} "
                      f"age={belief.measurement_age_s or 0.0:.1f} "
                      f"err={target_error if target_error is not None else -1:.2f}",
                      flush=True)
    finally:
        if video is not None:
            video.release()
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
    wrong = [row for row in rows if row["position"] is not None
             and not row["identity_correct"]]
    detection_gap = [row for row in rows if not row["target_detected"]]
    gap_predicted = [row for row in detection_gap
                     if row["status"] == "predicted"]
    gap_lost = [row for row in detection_gap if row["status"] == "lost"]
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
        scenario=scenario.name,
        map=scenario.scene_id,
        family=scenario.note,
        frames=len(rows),
        visible_frames=len(visible),
        hidden_frames=len(hidden),
        identity_accuracy=1.0 - len(wrong) / max(1, len(rows)),
        wrong_frames=len(wrong),
        measured_rate=sum(1 for row in rows if row["status"] == "measured") / max(1, len(rows)),
        predicted_rate=sum(1 for row in rows if row["status"] == "predicted") / max(1, len(rows)),
        lost_rate=sum(1 for row in rows if row["status"] == "lost") / max(1, len(rows)),
        visible_rmse=(float(np.sqrt(np.mean(np.square(measured_errors))))
                      if measured_errors else None),
        hidden_rmse=(float(np.sqrt(np.mean(np.square(hidden_errors))))
                     if hidden_errors else None),
        hidden_prediction_rate=len(predicted_hidden) / max(1, len(hidden)),
        hidden_lost_rate=len(lost_hidden) / max(1, len(hidden)),
        detection_gap_frames=len(detection_gap),
        detection_gap_prediction_rate=len(gap_predicted) / max(1, len(detection_gap)),
        detection_gap_lost_rate=len(gap_lost) / max(1, len(detection_gap)),
        prediction_error_at=error_at,
        hidden_start=hidden_started,
        hidden_end=hidden_ended,
        reacquisition_delay=(None if reacquired_t is None or hidden_ended is None
                             else round(reacquired_t - hidden_ended, 3)),
        same_track=(None if id_before_hidden is None or reacquired_id is None
                    else id_before_hidden == reacquired_id),
        rows=rows,
    )


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--maps-root", default=str(
        REPO / "perception/assets/maps"))
    parser.add_argument("--scenario", default="all",
                        choices=["all"] + [s.name for s in SCENARIOS])
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--weights", default="/home/jason/perception_data/runs/"
                                             "l1_front_pair_v1/weights/best.pt")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    camera = FrontPairCamera()
    worker = DetectionWorker(args.weights, conf=0.25)
    selected = [s for s in SCENARIOS
                if args.scenario == "all" or s.name == args.scenario]
    results = []
    try:
        for scenario in selected:
            result = run_scenario(scenario, args, worker, camera)
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
        identity_accuracy=float(np.mean(
            [result["identity_accuracy"] for result in results])),
        hidden_prediction_rate=float(np.mean(
            [result["hidden_prediction_rate"] for result in results])),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
