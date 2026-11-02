#!/usr/bin/env python3
"""Rendered Isaac Gym check: follow a target through a 1 s occlusion.

Scene
-----
The robot stands at the origin.  The target walks laterally behind a fixed
opaque panel (a wall segment) and is completely hidden from the camera for
about one second.  A second person stands to the side and stays visible the
whole time, so the perception layer has a distractor to reject.

The detector runs in the separate Ultralytics environment through
``perception/sop/detection_worker.py``, exactly as in the closed-loop follow
script.  The target layer receives only:

* detection boxes and confidences,
* HSV appearance descriptors computed from the image,
* metric ground positions computed from the box height (monocular) or depth
  (``--use-depth``).

Ground-truth masks are used only for scoring: to decide when the target is
really hidden, whether the tracker is on the right person, and how large the
prediction error is.  They never enter the layer.

    source local_env.sh
    /home/jason/legged_gym/.venv/bin/python perception/sim/occlusion_perception.py \
        --out /home/jason/perception_data/runs/occlusion_perception_v1 \
        --episodes 2 --frames 55 --video
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from l1_front_pair import front_pair_config  # noqa: E402
from people_scene import PeopleScene, _bbox_from_mask  # noqa: E402
from range_bearing import FrontPairCamera  # noqa: E402
from target_perception import PersonDetection, TargetPerceptionLayer  # noqa: E402
from target_tracker import hsv_histogram  # noqa: E402

# ------------------------------------------------------------------ detector


class DetectionWorker:
    """One-line JSON detector process over stdin/stdout."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 960):
        self.tmp = Path(tempfile.mkdtemp(prefix="occlusion_perception_"))
        self.frame_id = 0
        self.proc = subprocess.Popen(
            ["/home/jason/.venvs/ultralytics-rice/bin/python",
             str(REPO / "perception/sop/detection_worker.py"),
             "--weights", str(weights),
             "--conf", str(conf),
             "--imgsz", str(imgsz)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ready = self.proc.stdout.readline().strip()
        if not ready:
            raise RuntimeError("detection worker exited before ready")
        payload = json.loads(ready)
        if not payload.get("ready"):
            raise RuntimeError(f"detection worker failed: {payload}")

    def detect(self, image_rgb):
        path = self.tmp / f"f{self.frame_id:06d}.png"
        self.frame_id += 1
        Image.fromarray(np.asarray(image_rgb)).save(path)
        self.proc.stdin.write(json.dumps(dict(image=str(path))) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("detection worker closed the stream")
        payload = json.loads(line)
        if "error" in payload:
            raise RuntimeError(f"detection worker: {payload['error']}")
        return payload.get("detections", [])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


# ------------------------------------------------------------------ geometry


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def deduplicate(detections, iou_threshold=0.5, min_size=8.0):
    """Greedy NMS.  YOLO sometimes emits a clipped duplicate at the frame edge."""
    order = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    kept = []
    for detection in order:
        box = detection["bbox"]
        if (box[2] - box[0]) < min_size or (box[3] - box[1]) < min_size:
            continue
        if all(box_iou(box, other["bbox"]) < iou_threshold for other in kept):
            kept.append(detection)
    return kept


# ------------------------------------------------------------------ episode


def run_episode(scene, worker, camera, args, episode: int):
    scene.reset(0, episode_seed=args.seed + episode)
    people = sorted(scene.people[0], key=lambda p: p.person_id)
    target, distractor = people[0], people[1]
    target_pos = np.array([4.0, -1.0], dtype=np.float64)
    target_vel = np.array([0.0, args.target_speed], dtype=np.float64)
    distractor_pos = np.array([5.0, 2.5], dtype=np.float64)
    hidden_start = float(args.hidden_start)
    hidden_end = hidden_start + float(args.hidden_duration)
    target.pos = target_pos.copy()
    target.vel = target_vel.copy()
    distractor.pos = distractor_pos.copy()
    distractor.vel = np.zeros(2)

    layer = None
    rows = []
    last_visible_t = None
    hidden_started = None
    hidden_ended = None
    target_id_before_hidden = None
    reacquired_t = None
    reacquired_track_id = None
    video = None
    if args.video:
        import cv2
        video_path = Path(args.out) / f"episode{episode:02d}.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                10.0, (camera.width, camera.height))
        if not video.isOpened():
            print("warning: VideoWriter could not be opened; video disabled")
            video = None

    try:
        for frame in range(args.frames):
            t = round(frame * 0.1, 6)
            target_virtual = target_pos + target_vel * t
            hidden = hidden_start <= t < hidden_end
            # A controlled disappearance: the actor is moved out of the scene
            # for the hidden interval, but the scorer keeps the virtual
            # trajectory.  This isolates prediction from partial-occlusion
            # range bias, which the rendered-box-height estimator cannot
            # resolve without a mask.
            target.pos = np.array([100.0, 100.0]) if hidden else target_virtual.copy()
            target.vel = np.zeros(2) if hidden else target_vel.copy()
            distractor.pos = distractor_pos.copy()
            scene._apply_people(0)
            roll = args.roll_amp * math.sin(2.0 * math.pi * t / 1.7)
            scene.set_cameras(0, roll_deg=roll, base_yaw=0.0, base_xy=(0.0, 0.0))
            views = scene.render(0, save_depth=args.use_depth, save_seg=True)
            left = views[0]
            image = np.asarray(left["color"])
            seg = np.asarray(left["seg"])

            target_mask = seg == target.person_id
            target_pixels = int(target_mask.sum())
            target_box, _ = _bbox_from_mask(target_mask, 30)
            distractor_mask = seg == distractor.person_id
            distractor_box, _ = _bbox_from_mask(distractor_mask, 30)
            target_visible = (not hidden) and target_pixels >= 30 \
                and target_box is not None

            detections = deduplicate(worker.detect(image))
            converted = []
            for detection in detections:
                bbox = np.asarray(detection["bbox"], dtype=np.float64)
                world = camera.detection_to_world(
                    bbox,
                    robot_xy=(0.0, 0.0),
                    robot_yaw=0.0,
                    depth=left["depth"] if args.use_depth else None,
                    side="left",
                )
                if world is None:
                    continue
                appearance = hsv_histogram(image, bbox)
                position_xy = (world["position_world_xy"]
                               if world.get("consistent", True) else None)
                converted.append(PersonDetection(
                    position_xy=position_xy,
                    bbox=bbox,
                    appearance=appearance,
                    score=float(detection.get("conf", 1.0)),
                    covariance_xy=world["covariance_xy"],
                    meta=dict(range_source=world["range_source"],
                              range_m=world["range_m"],
                              consistent=bool(world.get("consistent", True))),
                ))

            if layer is None:
                if target_box is None:
                    continue
                layer = TargetPerceptionLayer(
                    init_bbox=np.asarray(target_box, dtype=np.float64),
                    init_appearance=hsv_histogram(image, target_box),
                    max_prediction_s=1.0,
                    max_memory_s=5.0,
                    birth_confidence_min=args.min_conf,
                    accel_std=0.6,
                    init_location_priority=True,
                )
            belief = layer.update(converted, t)

            if belief.position_xy is not None:
                target_error = float(np.linalg.norm(
                    belief.position_xy - target_virtual))
                distractor_error = float(np.linalg.norm(
                    belief.position_xy - distractor.pos))
                identity_correct = target_error <= distractor_error
            else:
                target_error = None
                distractor_error = None
                identity_correct = False

            detection_target_iou = max(
                (box_iou(d["bbox"], target_box) for d in detections),
                default=0.0) if target_box is not None else 0.0
            detection_distractor_iou = max(
                (box_iou(d["bbox"], distractor_box) for d in detections),
                default=0.0) if distractor_box is not None else 0.0

            if target_visible:
                if hidden_started is not None and hidden_ended is None:
                    hidden_ended = t
                if (hidden_ended is not None and reacquired_t is None
                        and belief.status == "measured" and identity_correct):
                    reacquired_t = t
                    reacquired_track_id = belief.track_id
                last_visible_t = t
                if hidden_started is None:
                    target_id_before_hidden = belief.track_id
            else:
                if hidden_started is None:
                    hidden_started = t
                if reacquired_t is None and belief.status == "measured" \
                        and identity_correct:
                    reacquired_t = t
                    reacquired_track_id = belief.track_id

            hidden_age = None
            if not target_visible and last_visible_t is not None:
                hidden_age = t - last_visible_t

            rows.append(dict(
                t=t,
                target_visible=target_visible,
                target_pixels=target_pixels,
                hidden_age=None if hidden_age is None else round(hidden_age, 3),
                detections=len(detections),
                detection_target_iou=round(detection_target_iou, 3),
                detection_distractor_iou=round(detection_distractor_iou, 3),
                status=belief.status,
                track_id=belief.track_id,
                measurement_age=(
                    None if belief.measurement_age_s is None
                    else round(belief.measurement_age_s, 3)),
                identity_correct=bool(identity_correct),
                target_error=None if target_error is None else round(target_error, 3),
                position=None if belief.position_xy is None else [
                    round(float(v), 3) for v in belief.position_xy],
                target_position=[round(float(v), 3) for v in target_virtual],
                distractor_position=[round(float(v), 3) for v in distractor.pos],
                range_source=converted[0].meta.get("range_source")
                if converted else None,
                detection_detail=[
                    dict(
                        bbox=[round(float(v), 1) for v in item.bbox],
                        conf=round(float(item.score), 3),
                        position=(None if item.position_xy is None else
                                  [round(float(v), 3) for v in item.position_xy]),
                    )
                    for item in converted
                ],
            ))

            if video is not None:
                import cv2
                canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for detection in detections:
                    x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
                    colour = (0, 255, 0)
                    if target_box is not None and box_iou(
                            detection["bbox"], target_box) > 0.3:
                        colour = (255, 255, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(canvas, f"{float(detection['conf']):.2f}",
                                (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                if target_box is not None:
                    x1, y1, x2, y2 = [int(v) for v in target_box]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(canvas, "GT target", (x1, max(20, y1 - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
                text = (f"t={t:.1f}s  {belief.status}  "
                        f"age={belief.measurement_age_s or 0.0:.1f}s  "
                        f"id={belief.track_id}  "
                        f"err={target_error if target_error is not None else -1:.2f}m")
                cv2.putText(canvas, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                video.write(canvas)

            if frame % 10 == 0:
                print(f"ep{episode} frame {frame:03d} t={t:.1f} "
                      f"visible={target_visible} status={belief.status} "
                      f"age={belief.measurement_age_s or 0.0:.1f} "
                      f"err={target_error if target_error is not None else -1:.2f}",
                      flush=True)
    finally:
        if video is not None:
            video.release()

    visible_rows = [r for r in rows if r["target_visible"]]
    hidden_rows = [r for r in rows if not r["target_visible"]]
    errors = [r["target_error"] for r in rows
              if r["status"] == "measured" and r["target_error"] is not None]
    hidden_errors = [r["target_error"] for r in hidden_rows
                     if r["target_error"] is not None]
    predicted_hidden = [r for r in hidden_rows if r["status"] == "predicted"]
    lost_hidden = [r for r in hidden_rows if r["status"] == "lost"]
    wrong = [r for r in rows if r["position"] is not None
             and not r["identity_correct"]]
    error_at = {}
    for checkpoint in (0.5, 1.0):
        candidates = [r for r in hidden_rows
                      if r["hidden_age"] is not None
                      and abs(r["hidden_age"] - checkpoint) < 0.051
                      and r["target_error"] is not None]
        if candidates:
            error_at[f"{checkpoint:.1f}"] = float(np.mean(
                [r["target_error"] for r in candidates]))
    return dict(
        episode=episode,
        frames=len(rows),
        visible_frames=len(visible_rows),
        hidden_frames=len(hidden_rows),
        measured_rate=sum(1 for r in rows if r["status"] == "measured") / max(1, len(rows)),
        predicted_rate=sum(1 for r in rows if r["status"] == "predicted") / max(1, len(rows)),
        lost_rate=sum(1 for r in rows if r["status"] == "lost") / max(1, len(rows)),
        identity_accuracy=1.0 - len(wrong) / max(1, len(rows)),
        wrong_frames=len(wrong),
        visible_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        hidden_rmse=float(np.sqrt(np.mean(np.square(hidden_errors))))
        if hidden_errors else None,
        hidden_prediction_rate=len(predicted_hidden) / max(1, len(hidden_rows)),
        hidden_lost_rate=len(lost_hidden) / max(1, len(hidden_rows)),
        prediction_error_at=error_at,
        hidden_start=hidden_started,
        hidden_end=hidden_ended,
        reacquired_t=reacquired_t,
        reacquisition_delay=(None if reacquired_t is None or hidden_ended is None
                             else round(reacquired_t - hidden_ended, 3)),
        same_track=(None if target_id_before_hidden is None
                    or reacquired_track_id is None
                    else target_id_before_hidden == reacquired_track_id),
        rows=rows,
    )


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="/home/jason/perception_data/runs/"
                                             "l1_front_pair_v1/weights/best.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--target-speed", type=float, default=0.6)
    parser.add_argument("--hidden-start", type=float, default=2.0,
                        help="time the target is removed from the scene")
    parser.add_argument("--hidden-duration", type=float, default=1.0,
                        help="duration of the controlled disappearance")
    parser.add_argument("--roll-amp", type=float, default=0.0)
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--use-depth", action="store_true",
                        help="use the camera depth when sampling the box; "
                             "default is the monocular box-height prior")
    parser.add_argument("--avatar-list", default="Business_Male_01,Business_Male_02",
                        help="comma-separated Rocketbox avatars; the default "
                             "pair is deliberately similar in appearance")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = front_pair_config(
        num_envs=1, num_people=2, frames=args.frames, episodes=args.episodes,
        seed=args.seed,
        avatar_dir=str(REPO / "perception/assets/rocketbox"),
        avatar_list=args.avatar_list,
        num_walls=0, num_pillars=0, save_seg_every=1, keep_in_fan=False)
    scene = PeopleScene(cfg)
    camera = FrontPairCamera()
    worker = DetectionWorker(args.weights, conf=args.min_conf)
    results = []
    try:
        for episode in range(args.episodes):
            result = run_episode(scene, worker, camera, args, episode)
            rows = result.pop("rows")
            results.append(result)
            (out / f"episode{episode:02d}.json").write_text(
                json.dumps(result, indent=2))
            (out / f"episode{episode:02d}_rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
    finally:
        worker.close()
        scene.close()

    summary = dict(
        weights=args.weights,
        episodes=results,
        identity_accuracy=float(np.mean(
            [r["identity_accuracy"] for r in results])),
        hidden_prediction_rate=float(np.mean(
            [r["hidden_prediction_rate"] for r in results])),
        prediction_error_at={
            key: float(np.mean([r["prediction_error_at"][key]
                                for r in results
                                if key in r["prediction_error_at"]]))
            for key in ("0.5", "1.0")
            if any(key in r["prediction_error_at"] for r in results)
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
