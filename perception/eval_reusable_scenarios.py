#!/usr/bin/env python3
"""Run the world-frame target layer over reusable, already-generated datasets.

Why this exists
---------------
Rendering new Isaac Gym scenes is expensive.  The repository already contains
several useful sequences:

* ``l1_crossing_v2/{biz035,gap035,gap060,gap100}``
    two-person head-on crossings at 5 m, with a similar-appearance variant and
    three depth gaps.  Raw YOLO detections are cached per frame, so this tests
    detector + tracker, not just the tracker.
* ``sim_v4_avatar_dev`` / ``sim_v4_avatar_test``
    640x480 multi-person scenes with 24 Rocketbox avatars, random walls and
    pillars, and an explicit target id.  The target is invisible in many
    frames, which is exactly the prediction/re-acquisition case.
* ``l1_front_pair_v1`` / ``l1_appearance_v1``
    768x1024 front-pair scenes with four people, random avatars and clutter.
    No target label exists, so person 1 is used as the operator's click
    (``--target-id``).

This is a *tracker* benchmark when it uses GT boxes, and a detector+tracker
benchmark for the crossing variants when it uses the cached YOLO boxes.  Metric
inputs are visible boxes, image crops for HSV, and a monocular range from the
box height.  Ground-truth identities and world positions are used only for
scoring and for the initial operator click.

Run with the Ultralytics venv because it has NumPy 2.x (the crossing caches
were written with NumPy 2.x) and OpenCV:

    /home/jason/.venvs/ultralytics-rice/bin/python \
        perception/eval_reusable_scenarios.py \
        --out artifacts/perception/target_perception_20261102/reusable_scenarios.json

Use ``--groups crossing`` for only the four crossing variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from perception.range_bearing import FrontPairCamera  # noqa: E402
from perception.target_perception import (PersonDetection,  # noqa: E402
                                          TargetPerceptionLayer)
from perception.target_tracker import hsv_histogram  # noqa: E402

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


DATA_ROOT = Path("/home/jason/perception_data")


# ------------------------------------------------------------------ specs


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    camera: str = "front"          # "front" (768x1024) or "legacy" (640x480)
    target_id: int | None = 1      # used when the GT has no target_id
    detections: str = "gt"         # "gt" or "cached"
    position_source: str = "camera"  # "camera" or "gt" (sim_v4 world_xy)
    cache_name: str = "bytetrack"
    max_groups: int = 0            # 0 = all
    note: str = ""


@dataclass
class GroupResult:
    dataset: str
    group: str
    status: str = "ok"
    frames: int = 0
    measured: int = 0
    predicted: int = 0
    lost: int = 0
    correct: int = 0
    wrong: int = 0
    visible_frames: int = 0
    hidden_frames: int = 0
    visible_errors: list = field(default_factory=list)
    hidden_errors: list = field(default_factory=list)
    predicted_hidden_errors: list = field(default_factory=list)
    prediction_at_05: list = field(default_factory=list)
    prediction_at_10: list = field(default_factory=list)
    reacquisition_delays: list = field(default_factory=list)
    id_switches: int = 0
    wrong_events: int = 0
    init_frame: int | None = None
    detection_source: str = "gt"

    def summary(self):
        scored = self.correct + self.wrong
        visible_rmse = (float(np.sqrt(np.mean(np.square(self.visible_errors))))
                        if self.visible_errors else None)
        hidden_rmse = (float(np.sqrt(np.mean(np.square(self.hidden_errors))))
                       if self.hidden_errors else None)
        return dict(
            status=self.status,
            frames=self.frames,
            visible_frames=self.visible_frames,
            hidden_frames=self.hidden_frames,
            measured_rate=self.measured / max(1, self.frames),
            predicted_rate=self.predicted / max(1, self.frames),
            lost_rate=self.lost / max(1, self.frames),
            identity_accuracy=self.correct / max(1, scored),
            wrong_rate=self.wrong / max(1, scored),
            visible_position_rmse=visible_rmse,
            hidden_position_rmse=hidden_rmse,
            prediction_error_0_5s=(float(np.mean(self.prediction_at_05))
                                   if self.prediction_at_05 else None),
            prediction_error_1_0s=(float(np.mean(self.prediction_at_10))
                                   if self.prediction_at_10 else None),
            mean_reacquisition_s=(float(np.mean(self.reacquisition_delays))
                                  if self.reacquisition_delays else None),
            reacquisition_events=len(self.reacquisition_delays),
            id_switches=self.id_switches,
            wrong_events=self.wrong_events,
            init_frame=self.init_frame,
            detection_source=self.detection_source,
        )


# ------------------------------------------------------------------ data


def _target_person(record, default_target_id):
    target_id = record.get("target_id", default_target_id)
    if target_id is None:
        return None
    for person in record.get("persons", []):
        pid = person.get("id", person.get("person_id"))
        if pid == target_id:
            return person
    return None


def _visible_box(person):
    if person is None:
        return None
    box = person.get("visible_box")
    if box is None:
        box = person.get("bbox")
    if box is None:
        return None
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _position(person):
    if person is None:
        return None
    value = person.get("pos")
    if value is None:
        value = person.get("world_xy")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array[:2]).all():
        return None
    return array[:2]


def _person_id(person):
    return person.get("id", person.get("person_id"))


def _image_path(spec, record):
    if record.get("image"):
        return spec.root / "images" / record["image"]
    stem = record.get("stem")
    if stem:
        return spec.root / "images" / f"{stem}_left.png"
    return None


def _load_image(path):
    if path is None or not Path(path).exists():
        return None
    if Image is not None:
        return np.asarray(Image.open(path).convert("RGB"))
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _load_groups(spec: DatasetSpec):
    path = spec.root / "gt" / "tracks.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open() if line.strip()]
    eyes = {row.get("eye") for row in rows}
    if "left" in eyes:
        rows = [row for row in rows if row.get("eye") == "left"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("env"), row.get("episode"))].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["frame"])
    items = sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    if spec.max_groups:
        items = items[:spec.max_groups]
    return dict(items)


def _load_cached_detections(spec: DatasetSpec, group_key):
    episode = group_key[1]
    path = spec.root / "det_cache" / f"{spec.cache_name}_left_ep{episode:02d}.npz"
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=True)
        return blob["boxes"], blob["scores"]
    except Exception:
        return None


def _camera(spec: DatasetSpec):
    if spec.camera == "legacy":
        # The 640x480 people_scene dataset in sim_v4_avatar_*.
        return FrontPairCamera(
            width=640, height=480, horizontal_fov_deg=57.4,
            toe_out_deg=0.0, cam_forward=0.20, cam_half_baseline=0.30,
            cam_up=0.05, ball_radius=0.40, default_person_height=1.70)
    return FrontPairCamera()


def _world_from_box(camera, box, person_height=1.70):
    result = camera.detection_to_base(
        box, depth=None, side="left", person_height=person_height)
    if result is None:
        return None, None
    return result["position_base_xy"], result["covariance_xy"]


def _deduplicate(detections, iou_threshold=0.5):
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / area if area > 0 else 0.0
    order = sorted(detections, key=lambda item: item.score, reverse=True)
    kept = []
    for detection in order:
        if all(iou(detection.bbox, other.bbox) < iou_threshold for other in kept):
            kept.append(detection)
    return kept


# ------------------------------------------------------------------ one group


def _box_stable(previous, current, ratio=0.25):
    if previous is None or current is None:
        return False
    previous_w = previous[2] - previous[0]
    previous_h = previous[3] - previous[1]
    current_w = current[2] - current[0]
    current_h = current[3] - current[1]
    if min(previous_w, previous_h, current_w, current_h) <= 0:
        return False
    return (abs(current_w / previous_w - 1.0) <= ratio
            and abs(current_h / previous_h - 1.0) <= ratio)


def run_group(spec: DatasetSpec, group_key, records, camera, args):
    result = GroupResult(dataset=spec.name, group=str(group_key))
    cached = (_load_cached_detections(spec, group_key)
              if spec.detections == "cached" else None)
    if spec.detections == "cached" and cached is None:
        result.status = "no_detection_cache"
        return result
    result.detection_source = spec.detections

    layer = None
    last_visible_t = None
    last_visible_position = None
    hidden_started = None
    previous_track_id = None
    wrong_streak = 0
    last_target_visible = False
    previous_target_box = None
    dt = 0.1

    for record in records:
        t = float(record.get("t", record["frame"] * dt))
        target = _target_person(record, spec.target_id)
        target_box = _visible_box(target)
        target_position = _position(target)
        target_visible = target_box is not None and target_position is not None

        image = _load_image(_image_path(spec, record))
        detections = []
        if cached is not None:
            frame = int(record["frame"])
            boxes = cached[0]
            scores = cached[1]
            if frame < len(boxes):
                frame_boxes = boxes[frame]
                frame_scores = scores[frame]
                if isinstance(frame_boxes, np.ndarray) and frame_boxes.size:
                    frame_boxes = np.asarray(frame_boxes, dtype=np.float64).reshape(-1, 4)
                else:
                    frame_boxes = np.zeros((0, 4))
                for box, score in zip(frame_boxes, np.asarray(frame_scores).reshape(-1)):
                    box = np.asarray(box, dtype=np.float64)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    appearance = hsv_histogram(image, box) if image is not None else None
                    if spec.position_source == "gt_match":
                        # Diagnostic only: keep the detector box/appearance but
                        # borrow the matched GT person's metric position.  This
                        # separates tracker identity from monocular-range bias.
                        matched_person = None
                        matched_iou = 0.3
                        for person in record.get("persons", []):
                            gt_box = _visible_box(person)
                            if gt_box is None:
                                continue
                            ix1, iy1 = max(box[0], gt_box[0]), max(box[1], gt_box[1])
                            ix2, iy2 = min(box[2], gt_box[2]), min(box[3], gt_box[3])
                            if ix2 <= ix1 or iy2 <= iy1:
                                continue
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            union = ((box[2] - box[0]) * (box[3] - box[1])
                                     + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                                     - inter)
                            value = inter / union if union > 0 else 0.0
                            if value > matched_iou:
                                matched_iou, matched_person = value, person
                        position = _position(matched_person)
                        covariance = None
                    elif spec.position_source == "gt":
                        continue  # cached boxes have no GT identity assignment
                    else:
                        position, covariance = _world_from_box(camera, box)
                    if position is None:
                        continue
                    detections.append(PersonDetection(
                        position_xy=position, bbox=box, appearance=appearance,
                        score=float(score), covariance_xy=covariance))
        else:
            for person in record.get("persons", []):
                box = _visible_box(person)
                position = _position(person)
                if box is None or position is None:
                    continue
                appearance = hsv_histogram(image, box) if image is not None else None
                if spec.position_source == "gt":
                    base_position, covariance = position, None
                else:
                    base_position, covariance = _world_from_box(camera, box)
                if base_position is None:
                    continue
                detections.append(PersonDetection(
                    position_xy=base_position, bbox=box, appearance=appearance,
                    score=1.0, covariance_xy=covariance))
        # Deduplicate only detector output.  GT boxes are one per person even
        # when two people overlap almost completely; removing one would erase
        # the person the operator selected.
        if cached is not None:
            detections = _deduplicate(detections)

        if layer is None:
            if not target_visible:
                continue
            # Do not initialise on a sliver: a heavily occluded first box
            # makes a poor appearance anchor and the track never recovers.
            if ((target_box[2] - target_box[0]) < args.init_min_width
                    or (target_box[3] - target_box[1]) < args.init_min_height):
                previous_target_box = target_box
                continue
            area = ((target_box[2] - target_box[0])
                    * (target_box[3] - target_box[1]))
            if previous_target_box is None:
                if area < args.init_min_area:
                    previous_target_box = target_box
                    continue
            elif not _box_stable(previous_target_box, target_box,
                                 args.init_stable_ratio):
                previous_target_box = target_box
                continue
            init_detection = None
            best_iou = 0.0
            for detection in detections:
                box = detection.bbox
                ix1, iy1 = max(box[0], target_box[0]), max(box[1], target_box[1])
                ix2, iy2 = min(box[2], target_box[2]), min(box[3], target_box[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = ((box[2] - box[0]) * (box[3] - box[1])
                             + (target_box[2] - target_box[0])
                             * (target_box[3] - target_box[1]) - inter)
                    value = inter / union if union > 0 else 0.0
                    if value > best_iou:
                        best_iou, init_detection = value, detection
            if init_detection is None or best_iou < args.init_iou:
                previous_target_box = target_box
                continue
            layer = TargetPerceptionLayer(
                init_bbox=init_detection.bbox,
                init_appearance=init_detection.appearance,
                max_prediction_s=1.0,
                max_memory_s=5.0,
                birth_confidence_min=0.25,
                target_position_gate_m=1.2,
                init_location_priority=True,
            )
            result.init_frame = int(record["frame"])

        if target_visible:
            previous_target_box = target_box
        belief = layer.update(detections, t)
        result.frames += 1
        if belief.status == "measured":
            result.measured += 1
        elif belief.status == "predicted":
            result.predicted += 1
        else:
            result.lost += 1

        if target_visible:
            result.visible_frames += 1
            last_visible_t = t
            last_visible_position = target_position.copy()
            if hidden_started is not None and belief.status == "measured":
                delay = t - hidden_started
                if delay >= 0.4:
                    result.reacquisition_delays.append(delay)
                hidden_started = None
        else:
            result.hidden_frames += 1
            if last_target_visible and hidden_started is None:
                hidden_started = t

        if belief.position_xy is not None:
            # Which GT person is the output actually following?
            people = []
            for person in record.get("persons", []):
                position = _position(person)
                if position is not None:
                    people.append((_person_id(person), position))
            if people:
                target_distance = (float(np.linalg.norm(
                    belief.position_xy - target_position))
                    if target_position is not None else float("inf"))
                best_id, best_distance = None, float("inf")
                for person_id, position in people:
                    distance = float(np.linalg.norm(belief.position_xy - position))
                    if distance < best_distance:
                        best_id, best_distance = person_id, distance
                target_id = _person_id(target) if target is not None else None
                correct = (target_id is not None and best_id == target_id
                           and target_distance <= best_distance + 0.1)
                if correct:
                    result.correct += 1
                else:
                    result.wrong += 1
                    wrong_streak += 1
                    if wrong_streak == 3:
                        result.wrong_events += 1
                if correct:
                    wrong_streak = 0
                if target_visible and target_position is not None:
                    result.visible_errors.append(target_distance)
                if not target_visible and target_position is not None:
                    result.hidden_errors.append(target_distance)
                    if belief.status == "predicted":
                        result.predicted_hidden_errors.append(target_distance)
                    if last_visible_t is not None:
                        age = t - last_visible_t
                        if abs(age - 0.5) < 0.051 and belief.status != "lost":
                            result.prediction_at_05.append(target_distance)
                        if abs(age - 1.0) < 0.051 and belief.status != "lost":
                            result.prediction_at_10.append(target_distance)

        if (target_visible and belief.track_id is not None
                and previous_track_id is not None
                and belief.track_id != previous_track_id):
            result.id_switches += 1
        if target_visible and belief.track_id is not None:
            previous_track_id = belief.track_id
        last_target_visible = target_visible

    if layer is None:
        result.status = "target_never_initialized"
    return result


# ------------------------------------------------------------------ report


def aggregate(results):
    total = sum(r.frames for r in results)
    correct = sum(r.correct for r in results)
    wrong = sum(r.wrong for r in results)
    def pooled(key):
        values = [value for r in results for value in getattr(r, key)]
        return float(np.sqrt(np.mean(np.square(values)))) if values else None
    def mean_pool(key):
        values = [value for r in results for value in getattr(r, key)]
        return float(np.mean(values)) if values else None
    return dict(
        groups=len(results),
        frames=total,
        visible_frames=sum(r.visible_frames for r in results),
        hidden_frames=sum(r.hidden_frames for r in results),
        identity_accuracy=correct / max(1, correct + wrong),
        wrong_rate=wrong / max(1, correct + wrong),
        measured_rate=sum(r.measured for r in results) / max(1, total),
        predicted_rate=sum(r.predicted for r in results) / max(1, total),
        lost_rate=sum(r.lost for r in results) / max(1, total),
        visible_position_rmse=pooled("visible_errors"),
        hidden_position_rmse=pooled("hidden_errors"),
        prediction_error_0_5s=mean_pool("prediction_at_05"),
        prediction_error_1_0s=mean_pool("prediction_at_10"),
        mean_reacquisition_s=mean_pool("reacquisition_delays"),
        reacquisition_events=sum(len(r.reacquisition_delays) for r in results),
        id_switches=sum(r.id_switches for r in results),
        wrong_events=sum(r.wrong_events for r in results),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", default="all",
                        choices=["all", "crossing", "avatars", "front_pair"])
    parser.add_argument("--target-id", type=int, default=1,
                        help="target for datasets without target_id")
    parser.add_argument("--init-iou", type=float, default=0.3)
    parser.add_argument("--init-min-width", type=float, default=20.0,
                        help="wait for this many pixels of target box width")
    parser.add_argument("--init-min-height", type=float, default=50.0,
                        help="wait for this many pixels of target box height")
    parser.add_argument("--init-stable-ratio", type=float, default=0.25,
                        help="require the target box to change by at most "
                             "this fraction between two consecutive frames")
    parser.add_argument("--init-min-area", type=float, default=2000.0,
                        help="first visible target box must be this large "
                             "before a stabilising second look is required")
    parser.add_argument("--max-groups", type=int, default=0,
                        help="per dataset; 0 = all")
    parser.add_argument("--cache-name", default="bytetrack",
                        help="crossing det_cache prefix")
    parser.add_argument("--crossing-position", default="camera",
                        choices=["camera", "gt_match"],
                        help="camera: monocular range from cached box; "
                             "gt_match: diagnostic GT position for the matched "
                             "person, isolating tracker from range bias")
    args = parser.parse_args()

    specs = []
    if args.groups in ("all", "crossing"):
        for variant in ("gap035", "gap060", "gap100", "biz035"):
            suffix = "_gtpos" if args.crossing_position == "gt_match" else ""
            specs.append(DatasetSpec(
                name=f"crossing/{variant}{suffix}",
                root=DATA_ROOT / "l1_crossing_v2" / variant,
                camera="front", target_id=1, detections="cached",
                position_source=args.crossing_position,
                cache_name=args.cache_name, max_groups=args.max_groups,
                note="two people crossing; cached YOLO detections"))
    if args.groups in ("all", "avatars"):
        for split in ("dev", "test"):
            specs.append(DatasetSpec(
                name=f"avatars/{split}",
                root=DATA_ROOT / f"sim_v4_avatar_{split}",
                camera="legacy", target_id=None, detections="gt",
                position_source="gt",
                max_groups=args.max_groups,
                note="24 Rocketbox avatars, random walls/pillars"))
    if args.groups in ("all", "front_pair"):
        specs.append(DatasetSpec(
            name="front_pair/clutter",
            root=DATA_ROOT / "l1_front_pair_v1",
            camera="front", target_id=args.target_id, detections="gt",
            position_source="gt", max_groups=args.max_groups,
            note="768x1024 front pair, four people, clutter"))
        specs.append(DatasetSpec(
            name="front_pair/appearance",
            root=DATA_ROOT / "l1_appearance_v1",
            camera="front", target_id=args.target_id, detections="gt",
            position_source="gt", max_groups=args.max_groups,
            note="avatar/appearance split, four people"))

    all_results = {}
    print(f"{'dataset':24s} {'ep':>3s} {'frames':>6s} {'id_acc':>7s} "
          f"{'wrong':>7s} {'lost':>7s} {'pred1s':>7s} {'reacq':>7s}")
    for spec in specs:
        groups = _load_groups(spec)
        camera = _camera(spec)
        results = []
        all_group_results = []
        for group_key, records in groups.items():
            result = run_group(spec, group_key, records, camera, args)
            all_group_results.append(result)
            if result.status == "ok":
                results.append(result)
        summary = aggregate(results)
        all_results[spec.name] = dict(
            note=spec.note,
            camera=spec.camera,
            position_source=spec.position_source,
            summary=summary,
            groups={r.group: r.summary() for r in all_group_results},
            skipped=sum(1 for r in all_group_results if r.status != "ok"),
        )
        def fmt(value, digits=2):
            return "--" if value is None else f"{value:.{digits}f}"
        print(f"{spec.name:24s} {summary['groups']:3d} {summary['frames']:6d} "
              f"{fmt(summary['identity_accuracy']):>7s} "
              f"{fmt(summary['wrong_rate']):>7s} "
              f"{fmt(summary['lost_rate']):>7s} "
              f"{fmt(summary['prediction_error_1_0s']):>7s} "
              f"{fmt(summary['mean_reacquisition_s']):>7s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
