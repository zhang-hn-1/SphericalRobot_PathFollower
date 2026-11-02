"""Single-target tracking on top of multi-person detections.

The scene may contain several people; only one of them is the target.  The
caller supplies the target once (an initial box, and optionally its appearance
histogram), and this module keeps that person's identity across frames.

Design
------
* Every person is tracked, not just the target.  Tracking all detections makes
  occlusions and crossings much easier to survive than a single-template
  tracker.
* Association combines IoU with an appearance descriptor.  For the proxy sim,
  the descriptor is an HSV histogram from the box; for real footage swap in a
  ReID embedding through the same interface.
* If the target track dies, the target is re-identified among the surviving
  tracks by appearance similarity.  Only if that fails is the target reported
  as lost.

Nothing here depends on Ros/Isaac Gym, so it runs with the system Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal environments
    linear_sum_assignment = None


# ------------------------------------------------------------------ boxes


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])


def xywh_to_xyxy(state):
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 0 else float(np.dot(a, b) / denom)


# ------------------------------------------------------------- appearance


def hsv_histogram(image_rgb, bbox, bins=(8, 8), center=0.6, min_saturation=25):
    """A cheap colour descriptor; replace with a ReID embedding for real people.

    Only the central part of the box is used (background at the edges is the
    main source of confusion), and low-saturation pixels (grey floor/walls) are
    ignored so the descriptor focuses on clothing colour.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * center / 2.0)
    half_h = max(1.0, (y2 - y1) * center / 2.0)
    x1, x2 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y1, y2 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] >= min_saturation).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    hist = cv2.calcHist(
        [hsv], [0, 1], mask,
        [bins[0], bins[1]],
        [0, 180, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


# ---------------------------------------------------------------- kalman


class KalmanBox:
    """Constant-velocity filter on (cx, cy, w, h)."""

    def __init__(self, bbox):
        z = xyxy_to_xywh(bbox)
        self.x = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)
        # Position noise dominates because boxes jump when a person is partly
        # occluded; velocity noise is deliberately small.
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.05])
        self.R = np.diag([2.0, 2.0, 2.0, 2.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return xywh_to_xyxy(self.x)

    def update(self, bbox):
        z = xyxy_to_xywh(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


# ----------------------------------------------------------------- track


@dataclass
class Detection:
    bbox: np.ndarray
    score: float = 1.0
    appearance: Optional[np.ndarray] = None


@dataclass
class Track:
    track_id: int
    filter: KalmanBox
    appearance: Optional[np.ndarray] = None
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    last_bbox: Optional[np.ndarray] = None
    predicted_bbox: Optional[np.ndarray] = None
    last_score: float = 1.0


@dataclass
class TargetState:
    bbox: Optional[np.ndarray]
    track_id: Optional[int]
    score: float = 0.0
    lost: bool = True
    reidentified: bool = False


class TargetTracker:
    def __init__(
        self,
        init_box,
        init_appearance=None,
        iou_weight: float = 0.6,
        appearance_weight: float = 0.4,
        center_weight: float = 0.0,
        iou_gate: float = 0.05,
        appearance_gate: float = 0.45,
        max_age: int = 20,
        min_hits: int = 2,
        reid_threshold: float = 0.55,
        appearance_smoothing: float = 0.8,
        max_area_ratio: float = 2.2,
        min_area_ratio: float = 0.40,
        max_center_ratio: float = 1.20,
        target_appearance_gate: float = 0.22,
        init_iou_threshold: float = 0.30,
        template_update_min_sim: float = 0.50,
        template_update_min_iou: float = 0.30,
        continuous_reid: bool = False,
        coast_frames: int = 0,
        init_fallback_frames: int = 0,
        init_location_priority: bool = False,
    ):
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)
        self.center_weight = float(center_weight)
        self.iou_gate = float(iou_gate)
        self.appearance_gate = float(appearance_gate)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.reid_threshold = float(reid_threshold)
        self.appearance_smoothing = float(appearance_smoothing)
        self.max_area_ratio = float(max_area_ratio)
        self.min_area_ratio = float(min_area_ratio)
        self.max_center_ratio = float(max_center_ratio)
        self.target_appearance_gate = float(target_appearance_gate)
        self.init_iou_threshold = float(init_iou_threshold)
        self.template_update_min_sim = float(template_update_min_sim)
        # Try to re-identify while the target track is merely unmatched,
        # not only after it has been deleted.  Off by default so recorded
        # results stay reproducible; the closed-loop search experiments turn
        # it on.  See _reidentify_other().
        self.continuous_reid = bool(continuous_reid)
        # How many missed frames the target may coast on its Kalman
        # prediction before the tracker reports it as lost.  0 preserves the
        # original behaviour (any miss -> lost).
        self.coast_frames = int(coast_frames)
        # Fall back to location-based initialisation (the operator pointed at
        # a person) after this many frames without an appearance-gated lock.
        # 0 preserves the original behaviour.
        self.init_fallback_frames = int(init_fallback_frames)
        # Operator-click semantics: among detections overlapping the init box,
        # pick by appearance.  Off by default so recorded results stay
        # reproducible.
        self.init_location_priority = bool(init_location_priority)
        self._init_frames = 0
        self.template_update_min_iou = float(template_update_min_iou)

        self.tracks: list[Track] = []
        self.next_id = 1
        self.target_id: Optional[int] = None
        self.target_template = (
            None if init_appearance is None else np.asarray(init_appearance, dtype=np.float32)
        )
        self.init_box = np.asarray(init_box, dtype=np.float64)
        self._initialized = False
        self._frames_since_target = 0

    # -- association ------------------------------------------------

    def _match(self, detections: Sequence[Detection]):
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(self.tracks):
            # Associate against the Kalman prediction, not the last observed
            # box: when a person is briefly missed, the prediction is a much
            # better reference than a stale detection.
            reference = (
                track.predicted_bbox
                if track.predicted_bbox is not None
                else track.last_bbox
            )
            tx1, ty1, tx2, ty2 = reference
            track_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            track_center = np.array([(tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0])
            for j, det in enumerate(detections):
                iou = box_iou(reference, det.bbox)
                sim = cosine(track.appearance, det.appearance)
                if iou < self.iou_gate and sim < self.appearance_gate:
                    cost[i, j] = 1e6
                    continue

                # The target track gets stricter gates.  A merged box around
                # the target and an occluder is a different object; following
                # it silently is worse than reporting the target as lost.
                if track.track_id == self.target_id and self.target_template is not None:
                    dx1, dy1, dx2, dy2 = det.bbox
                    area_ratio = max(1.0, (dx2 - dx1) * (dy2 - dy1)) / track_area
                    det_center = np.array([(dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0])
                    center_ratio = float(np.linalg.norm(det_center - track_center)) / np.sqrt(track_area)
                    if area_ratio > self.max_area_ratio or area_ratio < self.min_area_ratio:
                        cost[i, j] = 1e6
                        continue
                    if center_ratio > self.max_center_ratio and iou < 0.3:
                        cost[i, j] = 1e6
                        continue
                    if sim < self.target_appearance_gate and iou < 0.5:
                        cost[i, j] = 1e6
                        continue

                cost[i, j] = 1.0 - (
                    self.iou_weight * iou + self.appearance_weight * sim
                ) / max(1e-6, self.iou_weight + self.appearance_weight)

                if self.center_weight > 0.0:
                    det_center = np.array(
                        [(det.bbox[0] + det.bbox[2]) / 2.0,
                         (det.bbox[1] + det.bbox[3]) / 2.0]
                    )
                    center_dist = float(np.linalg.norm(det_center - track_center))
                    center_sim = math.exp(
                        -(center_dist / (0.5 * math.sqrt(track_area) + 1e-6)) ** 2
                    )
                    total = (
                        self.iou_weight + self.appearance_weight + self.center_weight
                    )
                    cost[i, j] = 1.0 - (
                        self.iou_weight * iou
                        + self.appearance_weight * sim
                        + self.center_weight * center_sim
                    ) / total

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
        else:  # greedy fallback
            rows, cols, used = [], [], set()
            for i, j in sorted(
                ((i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])),
                key=lambda pair: cost[pair[0], pair[1]],
            ):
                if i not in used and j not in used and cost[i, j] < 1e6:
                    rows.append(i)
                    cols.append(j)
                    used.add(i)
        pairs = [
            (int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e6
        ]
        matched_tracks = {i for i, _ in pairs}
        matched_dets = {j for _, j in pairs}
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        return pairs, unmatched_dets, unmatched_tracks

    # -- update -----------------------------------------------------

    def update(self, detections: Sequence[Detection]) -> TargetState:
        for track in self.tracks:
            track.predicted_bbox = track.filter.predict()

        pairs, unmatched_dets, unmatched_tracks = self._match(detections)

        for track_index, det_index in pairs:
            track = self.tracks[track_index]
            det = detections[det_index]
            track.filter.update(det.bbox)
            track.last_bbox = np.asarray(det.bbox, dtype=np.float64)
            track.last_score = float(det.score)
            track.hits += 1
            track.time_since_update = 0
            if det.appearance is not None:
                if track.appearance is None:
                    track.appearance = det.appearance
                else:
                    a = self.appearance_smoothing
                    track.appearance = a * track.appearance + (1 - a) * det.appearance
                    norm = np.linalg.norm(track.appearance)
                    if norm > 0:
                        track.appearance /= norm

                # Only refresh the target template on a confident, well-localised
                # match.  Updating on a merged/loose box would drift the identity.
                if track.track_id == self.target_id and self.target_template is not None:
                    reference = (
                        track.predicted_bbox
                        if track.predicted_bbox is not None
                        else track.last_bbox
                    )
                    sim = cosine(self.target_template, det.appearance)
                    if (
                        sim >= self.template_update_min_sim
                        and box_iou(reference, det.bbox) >= self.template_update_min_iou
                    ):
                        updated = 0.9 * self.target_template + 0.1 * det.appearance
                        norm = np.linalg.norm(updated)
                        if norm > 0:
                            self.target_template = updated / norm
            if track.hits >= self.min_hits:
                track.confirmed = True

        for det_index in unmatched_dets:
            det = detections[det_index]
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    filter=KalmanBox(det.bbox),
                    appearance=det.appearance,
                    last_bbox=np.asarray(det.bbox, dtype=np.float64),
                    last_score=float(det.score),
                    confirmed=self.min_hits <= 1,
                )
            )
            self.next_id += 1

        survivors = []
        for track_index in range(len(self.tracks)):
            track = self.tracks[track_index]
            if track_index in unmatched_tracks:
                track.time_since_update += 1
            if track.time_since_update <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        if not self._initialized:
            self._init_frames += 1
            self._select_initial_target()
            if (not self._initialized and self.init_fallback_frames
                    and self._init_frames >= self.init_fallback_frames
                    and self.target_template is not None):
                # The operator's click outranks the appearance gate: lock the
                # best-overlapping confirmed track by location.
                candidates = [t for t in self.tracks
                              if t.last_bbox is not None
                              and box_iou(self.init_box, t.last_bbox)
                              >= self.init_iou_threshold]
                if candidates:
                    best = max(candidates,
                               key=lambda t: box_iou(self.init_box, t.last_bbox))
                    self.target_id = best.track_id
                    self.target_template = best.appearance
                    self._initialized = True
        elif self.target_id is not None:
            target_track = next(
                (t for t in self.tracks if t.track_id == self.target_id), None)
            if target_track is None:
                self.target_id = None
                self._reidentify()
            elif self.continuous_reid and target_track.time_since_update > 0:
                # The target track missed this frame.  A detection of the
                # same person may already have formed a new track; recover it
                # now instead of waiting max_age frames for the old track to
                # die, which is what E11 exposed (detector saw the target in
                # 85/92 frames, tracker accepted 6).
                self._reidentify_other(target_track)
        elif self._initialized and self.continuous_reid:
            # The target track was lost and the first re-identification
            # attempt found no candidate.  Keep trying as new tracks appear:
            # a single attempt at the moment the old track dies is why the
            # E11 failure never recovered even though the target reappeared
            # with a perfect appearance match.
            self._reidentify()

        target = self._target_state()
        if target is None:
            self._frames_since_target += 1
        else:
            self._frames_since_target = 0
        return target or TargetState(bbox=None, track_id=None, lost=True)

    # -- target selection -------------------------------------------

    def _select_initial_target(self):
        candidates = [t for t in self.tracks if t.last_bbox is not None]
        if not candidates:
            return
        if self.target_template is not None:
            if self.init_location_priority:
                candidates = [t for t in candidates
                              if box_iou(self.init_box, t.last_bbox)
                              >= self.init_iou_threshold]
                if not candidates:
                    return
            best = max(candidates, key=lambda t: cosine(t.appearance, self.target_template))
            if cosine(best.appearance, self.target_template) < self.appearance_gate:
                return
        else:
            best = max(candidates, key=lambda t: box_iou(self.init_box, t.last_bbox))
            # Do not lock onto a wrong person just because something was
            # detected: wait for a detection that actually overlaps the click.
            if box_iou(self.init_box, best.last_bbox) < self.init_iou_threshold:
                return
        self.target_id = best.track_id
        self.target_template = best.appearance
        self._initialized = True

    def _reidentify_other(self, current):
        candidates = [
            t for t in self.tracks
            if t.confirmed and t.last_bbox is not None
            and t.track_id != current.track_id
        ]
        if not candidates or self.target_template is None:
            return False
        best = max(candidates,
                   key=lambda t: cosine(t.appearance, self.target_template))
        if cosine(best.appearance, self.target_template) >= self.reid_threshold:
            self.target_id = best.track_id
            return True
        return False

    def _reidentify(self):
        candidates = [t for t in self.tracks if t.confirmed and t.last_bbox is not None]
        if not candidates:
            return
        best = max(candidates, key=lambda t: cosine(t.appearance, self.target_template))
        if cosine(best.appearance, self.target_template) >= self.reid_threshold:
            self.target_id = best.track_id
            return True
        return False

    def _target_state(self):
        if self.target_id is None:
            return None
        for track in self.tracks:
            if track.track_id == self.target_id:
                if track.time_since_update > self.coast_frames:
                    return TargetState(
                        bbox=(
                            track.predicted_bbox
                            if track.predicted_bbox is not None
                            else track.last_bbox
                        ),
                        track_id=track.track_id,
                        score=track.last_score,
                        lost=True,
                    )
                if track.time_since_update > 0:
                    return TargetState(
                        bbox=(
                            track.predicted_bbox
                            if track.predicted_bbox is not None
                            else track.last_bbox
                        ),
                        track_id=track.track_id,
                        score=track.last_score,
                        lost=not track.confirmed,
                    )
                return TargetState(
                    bbox=track.last_bbox,
                    track_id=track.track_id,
                    score=track.last_score,
                    lost=not track.confirmed,
                )
        return None
