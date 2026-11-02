"""Target-person perception: identity-locked tracking with short-horizon prediction.

The layer consumes one detection set per frame.  Every detection may carry an
image box, a metric ground position, an appearance descriptor and a position
covariance.  All people are tracked; exactly one track is the operator's
target.  The target identity is anchored at selection time and never updated,
so a distractor cannot slowly replace it by being matched for many frames.

Two behaviours matter for following a person:

* **Distractor rejection.**  Association uses appearance *and* motion.  If the
  target's best match is not clearly better than a second candidate, the
  frame is treated as ambiguous and the target coasts instead of switching.
  This is deliberately conservative: reporting a short prediction is better
  than following the wrong person.
* **Short-dropout prediction.**  A constant-velocity Kalman filter runs in the
  ground frame.  When the target is not detected, the filter continues for
  ``max_prediction_s`` (default 1 s, the requested horizon) and the output
  status is ``predicted``.  Beyond that horizon the status is ``lost`` but the
  track is retained for ``max_memory_s`` so a later detection can re-acquire
  the same identity.

The module is pure NumPy (SciPy's Hungarian solver is used when available), so
it can be unit-tested without Isaac Gym, torch or a camera.

Frame conventions
-----------------
Positions are metric world coordinates ``(x, y)`` with +x forward and +y left.
The caller converts image boxes and depth to that frame (see
``perception.range_bearing``).  This keeps the tracker independent of a
particular camera rig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:  # scipy is present in the perception environments; keep a fallback.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised only in minimal installs
    linear_sum_assignment = None


# ------------------------------------------------------------------ helpers


def _unit(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return None
    return value / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _as_xy(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array[:2]).all():
        return None
    return array[:2].copy()


def _as_cov(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 2) or not np.isfinite(array).all():
        return None
    # Make the covariance symmetric positive semidefinite enough for a KF.
    array = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _box_area(box) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


# ------------------------------------------------------------------ data


@dataclass
class PersonDetection:
    """One detection, in the frame the tracker works in.

    ``bbox`` is optional (used for operator selection and diagnostics).
    ``position_xy`` is required for a new track; a detection without a metric
    position can still refresh the appearance of an existing track.
    """

    position_xy: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    appearance: Optional[np.ndarray] = None
    score: float = 1.0
    covariance_xy: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@dataclass
class TargetBelief:
    """The target state produced by one update.

    ``status`` is one of:

    ``measured``
        A detection was associated with the target in this frame.
    ``predicted``
        No detection this frame, but the dropout is within
        ``max_prediction_s``; ``position_xy`` is the motion prediction.
    ``lost``
        The dropout exceeds ``max_prediction_s`` (or no target was selected).
        ``position_xy`` is still the last prediction, for diagnostics only.

    ``lost`` is false for both ``measured`` and ``predicted``; it is true only
    when the caller should stop following.  Use ``status`` when the
    measured/predicted distinction matters and ``lost`` when a simple yes/no
    is enough.
    """

    position_xy: Optional[np.ndarray] = None
    velocity_xy: Optional[np.ndarray] = None
    covariance_xy: Optional[np.ndarray] = None
    status: str = "lost"
    lost: bool = True
    track_id: Optional[int] = None
    bbox: Optional[np.ndarray] = None
    measurement_age_s: Optional[float] = None
    identity_score: float = 0.0
    reidentified: bool = False
    target_visible: bool = False
    predicted_trajectory: list = field(default_factory=list)
    predicted_horizon_s: float = 0.0
    velocity_confidence: float = 0.0
    prediction_uncertainty_m: float = 0.0
    trajectory_history: list = field(default_factory=list)
    history_span_s: float = 0.0

    @property
    def predicted(self) -> bool:
        return self.status == "predicted"


# ------------------------------------------------------------------ filter


class _ConstantVelocityKalman:
    """Small constant-velocity Kalman filter in the ground plane."""

    def __init__(self, position_xy, covariance_xy=None,
                 position_var: float = 0.25, velocity_var: float = 1.0):
        position = _as_xy(position_xy)
        if position is None:
            raise ValueError("a new track needs a finite position")
        self.x = np.array([position[0], position[1], 0.0, 0.0],
                          dtype=np.float64)
        self.P = np.diag([position_var, position_var,
                          velocity_var, velocity_var]).astype(np.float64)
        covariance = _as_cov(covariance_xy)
        if covariance is not None:
            self.P[:2, :2] = covariance

    def predict(self, dt: float, accel_std: float = 0.6) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return
        F = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # White-noise acceleration model; each axis gets [dt^2/2, dt] * sigma_a.
        G = np.array([[0.5 * dt * dt, 0.0],
                      [0.0, 0.5 * dt * dt],
                      [dt, 0.0],
                      [0.0, dt]], dtype=np.float64)
        q = max(0.0, float(accel_std)) ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ np.eye(2) @ G.T * q
        self.P += np.eye(4) * 1e-9

    def update(self, position_xy, covariance_xy=None,
               default_std: float = 0.25) -> None:
        position = _as_xy(position_xy)
        if position is None:
            return
        R = _as_cov(covariance_xy)
        if R is None:
            variance = max(1e-6, float(default_std) ** 2)
            R = np.diag([variance, variance])
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        innovation = position - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        # Joseph form keeps the covariance symmetric and positive semidefinite
        # even when a measurement is much more precise than the prediction.
        I = np.eye(4) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T

    def state_at(self, dt: float, accel_std: float = 0.6):
        """Return ``(position, velocity, covariance)`` after an extra ``dt``."""
        copy = _ConstantVelocityKalman.__new__(_ConstantVelocityKalman)
        copy.x = self.x.copy()
        copy.P = self.P.copy()
        copy.predict(dt, accel_std)
        return copy.x[:2].copy(), copy.x[2:4].copy(), copy.P[:2, :2].copy()


# ------------------------------------------------------------------ track


@dataclass
class _Track:
    track_id: int
    anchor: Optional[np.ndarray]
    filter: _ConstantVelocityKalman
    last_measure_t: float
    last_seen_t: float
    appearance: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    score: float = 1.0
    hits: int = 1
    confirmed: bool = False
    identity_score: float = 0.0
    history: list = field(default_factory=list)

    def observe(self, detection: PersonDetection, t: float,
                appearance_update_cosine: float) -> None:
        position = _as_xy(detection.position_xy)
        if position is not None:
            self.filter.update(position, detection.covariance_xy)
            self.last_measure_t = t
            # Only a metric detection may refresh the box.  A position-less
            # detection (e.g. depth/height disagreement at an occlusion edge)
            # keeps its appearance cue but must not move the size reference.
            if detection.bbox is not None:
                self.bbox = np.asarray(detection.bbox, dtype=np.float64).copy()
            self.score = float(detection.score)
            self.history.append((float(t), position.copy()))
            cutoff = float(t) - 3.0
            self.history = [item for item in self.history
                            if item[0] >= cutoff][-64:]
        self.last_seen_t = t
        self.hits += 1
        self.confirmed = True
        similarity = _cosine(self.anchor, detection.appearance)
        if similarity is not None:
            self.identity_score = float(similarity)
            if self.appearance is None:
                self.appearance = _unit(detection.appearance)
            elif similarity >= appearance_update_cosine:
                # The anchor never changes.  The running feature is only a
                # short-term appearance hint and is updated slowly.
                merged = 0.95 * self.appearance + 0.05 * _unit(detection.appearance)
                self.appearance = _unit(merged)


# ------------------------------------------------------------------ layer


class TargetPerceptionLayer:
    """Multi-person tracker with one operator-selected target.

    Parameters
    ----------
    init_bbox:
        Image box clicked by the operator in the first usable frame.  Used to
        select the target by overlap; the tracker never sees GT identities.
    init_position_xy:
        Alternative selection when the caller already has a metric click
        point instead of a box.
    init_appearance:
        Optional appearance descriptor taken at selection time.  When given,
        selection requires a minimum similarity, which prevents locking onto a
        bystander that merely overlaps the click.
    max_prediction_s:
        How long the target may coast on the Kalman prediction.  The requested
        default is 1.0 s.
    max_memory_s:
        How long a track (including the target) is retained without a
        measurement for re-identification.  This is intentionally longer than
        the prediction horizon: a target can be invisible for 1 s and still
        return, and a longer hole should not erase its identity.
    """

    def __init__(
        self,
        init_bbox=None,
        init_position_xy=None,
        init_appearance=None,
        *,
        max_prediction_s: float = 1.0,
        max_memory_s: float = 5.0,
        cosine_min: float = 0.55,
        reacquire_cosine_min: float = 0.72,
        target_cosine_min: float = 0.45,
        appearance_update_cosine_min: float = 0.85,
        ambiguity_margin: float = 0.06,
        position_gate_m: float = 1.0,
        target_position_gate_m: float = 0.9,
        target_close_gap_s: float = 0.3,
        target_close_position_gate_m: float = 0.6,
        target_close_appearance_weight: float = 0.25,
        target_close_motion_weight: float = 1.0,
        max_speed_mps: float = 2.0,
        accel_std: float = 0.6,
        measurement_std: float = 0.25,
        birth_confidence_min: float = 0.55,
        min_hits: int = 1,
        init_iou_threshold: float = 0.30,
        init_fallback_frames: int = 10,
        init_location_priority: bool = False,
        defer_ambiguous_reacquisition_birth: bool = True,
        max_box_area_ratio: float = 1.8,
        min_box_area_ratio: float = 0.45,
        default_dt: float = 0.1,
    ):
        self.init_bbox = None if init_bbox is None else np.asarray(init_bbox, dtype=np.float64)
        self.init_position_xy = _as_xy(init_position_xy)
        self.init_appearance = _unit(init_appearance)

        self.max_prediction_s = float(max_prediction_s)
        self.max_memory_s = float(max_memory_s)
        self.cosine_min = float(cosine_min)
        self.reacquire_cosine_min = float(reacquire_cosine_min)
        self.target_cosine_min = float(target_cosine_min)
        self.appearance_update_cosine_min = float(appearance_update_cosine_min)
        self.ambiguity_margin = float(ambiguity_margin)
        self.position_gate_m = float(position_gate_m)
        self.target_position_gate_m = float(target_position_gate_m)
        self.target_close_gap_s = float(target_close_gap_s)
        self.target_close_position_gate_m = float(target_close_position_gate_m)
        self.target_close_appearance_weight = float(target_close_appearance_weight)
        self.target_close_motion_weight = float(target_close_motion_weight)
        self.max_speed_mps = float(max_speed_mps)
        self.accel_std = float(accel_std)
        self.measurement_std = float(measurement_std)
        self.birth_confidence_min = float(birth_confidence_min)
        self.min_hits = int(min_hits)
        self.init_iou_threshold = float(init_iou_threshold)
        self.init_fallback_frames = int(init_fallback_frames)
        self.init_location_priority = bool(init_location_priority)
        self.defer_ambiguous_reacquisition_birth = bool(defer_ambiguous_reacquisition_birth)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.default_dt = float(default_dt)

        self.tracks: list[_Track] = []
        self.next_id = 1
        self.target_id: Optional[int] = None
        self.target_anchor = self.init_appearance
        self._initialized = False
        self._frames = 0
        self._last_t: Optional[float] = None
        self._last_belief = TargetBelief()
        self._reidentified_this_frame = False

    # -- public -----------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.target_id = None
        self.target_anchor = self.init_appearance
        self._initialized = False
        self._frames = 0
        self._last_t = None
        self._last_belief = TargetBelief()
        self._reidentified_this_frame = False

    def update(self, detections: Sequence[PersonDetection],
               t: Optional[float] = None) -> TargetBelief:
        """Advance one frame and return the target belief."""
        detections = list(detections)
        if t is None:
            t = (self._last_t if self._last_t is not None else 0.0) + self.default_dt
        t = float(t)
        dt = self.default_dt if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self._frames += 1
        self._reidentified_this_frame = False

        for track in self.tracks:
            track.filter.predict(dt, self.accel_std)
        # Expire tracks before association.  Otherwise a track older than the
        # memory window could still match a new detection and be revived,
        # which would make ``max_memory_s`` meaningless.
        self._prune(t)

        pairs, unmatched_dets, unmatched_tracks = self._associate(detections, t)
        matched_tracks = set()
        matched_dets = set()
        blocked_dets = set()
        for track_index, det_index in pairs:
            track = self.tracks[track_index]
            detection = detections[det_index]
            if track.track_id == self.target_id and self._ambiguous_target(
                    track_index, det_index, detections, t):
                # Do not force a best guess at a crossing / merge.  The target
                # track will coast and the same decision is retried next frame.
                blocked_dets.add(det_index)
                continue
            track.observe(detection, t, self.appearance_update_cosine_min)
            if track.identity_score > 0.0:
                if track.track_id == self.target_id:
                    self.target_anchor = (
                        self.target_anchor if self.target_anchor is not None
                        else track.anchor
                    )
            matched_tracks.add(track_index)
            matched_dets.add(det_index)

        # A detection that did not match any track can start a new track.  A
        # weak or ambiguous detection is allowed to maintain an existing track
        # but not to create a competing anchor.
        for det_index, detection in enumerate(detections):
            if det_index in matched_dets or det_index in blocked_dets:
                continue
            if float(detection.score) < self.birth_confidence_min:
                continue
            if _as_xy(detection.position_xy) is None:
                continue
            if self._defer_birth(detection, t):
                continue
            self.tracks.append(_Track(
                track_id=self.next_id,
                anchor=_unit(detection.appearance),
                filter=_ConstantVelocityKalman(
                    detection.position_xy, detection.covariance_xy,
                    position_var=max(1e-6, self.measurement_std ** 2),
                    velocity_var=1.0,
                ),
                last_measure_t=t,
                last_seen_t=t,
                appearance=_unit(detection.appearance),
                bbox=None if detection.bbox is None else np.asarray(
                    detection.bbox, dtype=np.float64).copy(),
                score=float(detection.score),
                hits=1,
                confirmed=self.min_hits <= 1,
                identity_score=1.0 if detection.appearance is not None else 0.0,
                history=[(t, np.asarray(detection.position_xy,
                                       dtype=np.float64).reshape(2).copy())],
            ))
            self.next_id += 1

        self._prune(t)

        if not self._initialized:
            self._select_initial_target()
            if (not self._initialized and self.init_fallback_frames
                    and self._frames >= self.init_fallback_frames):
                # The operator's click outranks the appearance gate after a
                # short grace period.  This is the same fallback used by the
                # image-space tracker: keep trying to verify by appearance,
                # but do not spend the whole episode unlocked because the
                # first crops were low quality.
                self.init_location_priority = True
                self._select_initial_target(force_location=True)
        elif self.target_id is not None and not any(
                track.track_id == self.target_id for track in self.tracks):
            self.target_id = None
            self._reidentify(t)
        elif self._initialized and self.target_id is None:
            self._reidentify(t)

        belief = self._belief(t)
        self._last_belief = belief
        return belief

    def position_at(self, horizon_s: float) -> Optional[np.ndarray]:
        """Predict the target position ``horizon_s`` seconds from now."""
        horizon = max(0.0, float(horizon_s))
        track = self._target_track()
        if track is None:
            return None
        position, _, _ = track.filter.state_at(horizon, self.accel_std)
        return position

    def predicted_trajectory(self, horizon_s: Optional[float] = None,
                             step_s: float = 0.1) -> list:
        horizon = self.max_prediction_s if horizon_s is None else float(horizon_s)
        track = self._target_track()
        if track is None:
            return []
        out = []
        steps = max(1, int(round(max(0.0, horizon) / max(step_s, 1e-3))))
        for index in range(steps + 1):
            position, _, covariance = track.filter.state_at(
                index * step_s, self.accel_std)
            out.append(dict(
                horizon_s=round(index * step_s, 3),
                position_xy=position,
                covariance_xy=covariance,
            ))
        return out

    # -- association ------------------------------------------------

    def _associate(self, detections: Sequence[PersonDetection], t: float):
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        cost = np.full((len(self.tracks), len(detections)), 1e6, dtype=np.float64)
        for i, track in enumerate(self.tracks):
            gap = max(0.0, t - track.last_measure_t)
            predicted = track.filter.x[:2]
            covariance = track.filter.P[:2, :2]
            uncertainty = math.sqrt(max(0.0, float(np.trace(covariance))))
            for j, detection in enumerate(detections):
                similarity = _cosine(track.anchor, detection.appearance)
                threshold = (
                    self.reacquire_cosine_min
                    if gap > self.max_prediction_s else self.cosine_min
                )
                position = _as_xy(detection.position_xy)
                # When two people overlap, the target crop is contaminated
                # and its appearance can look worse than the occluder's.
                # If the target was seen very recently and this detection is
                # at the predicted position, motion continuity wins.
                close_target = bool(
                    track.track_id == self.target_id
                    and gap <= self.target_close_gap_s
                    and position is not None
                    and float(np.linalg.norm(predicted - position))
                    <= self.target_close_position_gate_m
                )
                if close_target and similarity is not None                         and similarity < self.target_cosine_min:
                    # Do not steal a detection that belongs to a different,
                    # better-appearance track.  This is the crossing case:
                    # the target is hidden, the distractor is the only box
                    # near the prediction, and its appearance clearly belongs
                    # to the distractor track.
                    for other in self.tracks:
                        if other.track_id == self.target_id or other.anchor is None:
                            continue
                        other_similarity = _cosine(other.anchor,
                                                   detection.appearance)
                        if (other_similarity is None
                                or other_similarity <= similarity + 0.05):
                            continue
                        other_gap = max(0.0, t - other.last_measure_t)
                        other_gate = (
                            self.position_gate_m
                            + self.max_speed_mps * min(other_gap, 3.0)
                            + 1.5 * math.sqrt(max(0.0, float(
                                np.trace(other.filter.P[:2, :2]))))
                        )
                        if float(np.linalg.norm(
                                other.filter.x[:2] - position)) <= other_gate:
                            close_target = False
                            break
                if (similarity is not None and similarity < threshold
                        and not close_target):
                    continue

                distance = 0.0
                gate = 0.0
                if position is not None:
                    distance = float(np.linalg.norm(predicted - position))
                    gate = (
                        self.position_gate_m
                        + self.max_speed_mps * min(gap, 3.0)
                        + 1.5 * uncertainty
                    )
                    if track.track_id == self.target_id:
                        # The target has a tighter motion gate while it is
                        # still within the prediction horizon.  This is what
                        # stops a nearby distractor from stealing the track.
                        gate = min(
                            gate,
                            self.target_position_gate_m
                            + self.max_speed_mps * min(gap, 3.0)
                            + 1.5 * uncertainty,
                        )
                    if gap <= self.max_memory_s and distance > gate:
                        continue

                if similarity is None:
                    # No appearance: use motion only.  A track that has never
                    # had an appearance can still be followed, but it cannot
                    # be re-identified by appearance later.
                    if position is None or gate <= 0.0:
                        continue
                    cost[i, j] = min(1.0, distance / gate)
                else:
                    motion_term = 0.0 if gate <= 0.0 else min(1.0, distance / gate)
                    if (track.track_id == self.target_id
                            and gap <= self.target_close_gap_s):
                        # While the target is continuously visible, motion
                        # continuity is more trustworthy than a possibly
                        # contaminated crop (two people overlapping).  Once
                        # the track has missed frames, appearance takes over.
                        cost[i, j] = (
                            self.target_close_appearance_weight
                            * (1.0 - float(similarity))
                            + self.target_close_motion_weight * motion_term
                        )
                    else:
                        cost[i, j] = (1.0 - float(similarity)) + 0.30 * motion_term

                # Never let a low-confidence box own the target.  Weak boxes
                # may still update non-target tracks (they are useful for
                # keeping an occluder from becoming a ghost track).
                if (track.track_id == self.target_id
                        and float(detection.score) < self.birth_confidence_min * 0.8):
                    cost[i, j] = 1e6

                if (track.track_id == self.target_id
                        and similarity is not None
                        and not close_target
                        and gap <= self.max_prediction_s
                        and similarity < self.target_cosine_min
                        and distance > self.target_position_gate_m):
                    cost[i, j] = 1e6

                # A merged box (target + occluder) or a thin sliver at an
                # occlusion boundary has the wrong area even when its
                # appearance still looks like the target.  Reject it and let
                # the motion model coast; a proper box usually follows one
                # frame later.
                if (track.track_id == self.target_id
                        and track.bbox is not None and detection.bbox is not None
                        and similarity is not None
                        and similarity < self.appearance_update_cosine_min):
                    track_area = _box_area(track.bbox)
                    det_area = _box_area(detection.bbox)
                    if track_area > 0.0 and det_area > 0.0:
                        ratio = det_area / track_area
                        if (ratio > self.max_box_area_ratio
                                or ratio < self.min_box_area_ratio):
                            cost[i, j] = 1e6

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
        else:  # pragma: no cover - only for environments without SciPy
            rows, cols, used = [], [], set()
            for i, j in sorted(
                    ((i, j) for i in range(cost.shape[0])
                     for j in range(cost.shape[1])),
                    key=lambda pair: cost[pair]):
                if i not in used and j not in used and cost[i, j] < 1e6:
                    rows.append(i)
                    cols.append(j)
                    used.add(i)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols)
                 if cost[i, j] < 1e6]
        matched_tracks = {i for i, _ in pairs}
        matched_dets = {j for _, j in pairs}
        return (pairs,
                [j for j in range(len(detections)) if j not in matched_dets],
                [i for i in range(len(self.tracks)) if i not in matched_tracks])

    def _ambiguous_target(self, track_index: int, det_index: int,
                          detections: Sequence[PersonDetection], t: float) -> bool:
        """Reject a target association when a second person is nearly as good.

        The comparison is only meaningful when both detections exist.  A
        distractor that is a clear second choice is ignored; a distractor that
        is within ``ambiguity_margin`` makes the frame ambiguous.
        """
        if self.ambiguity_margin <= 0.0:
            return False
        track = self.tracks[track_index]
        best = self._pair_cost(track, detections[det_index], t)
        second = 1e6
        for index, detection in enumerate(detections):
            if index == det_index:
                continue
            value = self._pair_cost(track, detection, t)
            if value < second:
                second = value
        return second < 1e6 and (second - best) < self.ambiguity_margin

    def _pair_cost(self, track: _Track, detection: PersonDetection,
                   t: float) -> float:
        similarity = _cosine(track.anchor, detection.appearance)
        position = _as_xy(detection.position_xy)
        gap = max(0.0, t - track.last_measure_t)
        close_target = bool(
            track.track_id == self.target_id
            and gap <= self.target_close_gap_s
            and position is not None
            and float(np.linalg.norm(track.filter.x[:2] - position))
            <= self.target_close_position_gate_m
        )
        if similarity is None:
            if position is None:
                return 1e6
            gate = self.position_gate_m + self.max_speed_mps * min(gap, 3.0)
            distance = float(np.linalg.norm(track.filter.x[:2] - position))
            return min(1.0, distance / max(gate, 1e-6))
        threshold = (
            self.reacquire_cosine_min
            if gap > self.max_prediction_s else self.cosine_min
        )
        if similarity < threshold and not close_target:
            return 1e6
        if (track.track_id == self.target_id
                and track.bbox is not None and detection.bbox is not None
                and similarity < self.appearance_update_cosine_min):
            track_area = _box_area(track.bbox)
            det_area = _box_area(detection.bbox)
            if track_area > 0.0 and det_area > 0.0:
                ratio = det_area / track_area
                if (ratio > self.max_box_area_ratio
                        or ratio < self.min_box_area_ratio):
                    return 1e6
        gate = self.position_gate_m + self.max_speed_mps * min(gap, 3.0)
        distance = 0.0
        if position is not None:
            distance = float(np.linalg.norm(track.filter.x[:2] - position))
            if distance > gate:
                return 1e6
        motion_term = min(1.0, distance / max(gate, 1e-6))
        if track.track_id == self.target_id and gap <= self.target_close_gap_s:
            return (self.target_close_appearance_weight * (1.0 - float(similarity))
                    + self.target_close_motion_weight * motion_term)
        return (1.0 - float(similarity)) + 0.30 * motion_term

    def _defer_birth(self, detection: PersonDetection, t: float) -> bool:
        if not self.defer_ambiguous_reacquisition_birth:
            return False
        position = _as_xy(detection.position_xy)
        if position is None:
            return False
        for track in self.tracks:
            gap = t - track.last_measure_t
            if not (self.max_prediction_s < gap <= self.max_memory_s):
                continue
            similarity = _cosine(track.anchor, detection.appearance)
            if similarity is None or similarity < self.cosine_min:
                continue
            distance = float(np.linalg.norm(track.filter.x[:2] - position))
            gate = self.position_gate_m + self.max_speed_mps * min(gap, 3.0)
            if distance <= gate:
                # This is probably the occluded person returning, not a new
                # person.  Wait for the association to pass before creating a
                # duplicate anchor.
                return True
        return False

    # -- target selection / re-identification -----------------------

    def _select_initial_target(self, force_location: bool = False) -> None:
        candidates = [track for track in self.tracks if track.confirmed]
        if not candidates:
            return
        if self.init_bbox is None and self.init_position_xy is None:
            # No click: select the oldest confirmed track.  This is useful for
            # a single-person demo, but an operator click is the normal path.
            if self.init_appearance is None:
                return
            best = max(candidates, key=lambda track: track.hits)
        elif self.init_bbox is not None:
            overlapping = [
                track for track in candidates
                if track.bbox is not None
                and _box_iou(self.init_bbox, track.bbox) >= self.init_iou_threshold
            ]
            if not overlapping:
                return
            if (self.init_appearance is not None and not force_location
                    and not self.init_location_priority):
                best = max(
                    overlapping,
                    key=lambda track: _cosine(track.anchor, self.init_appearance) or -1.0,
                )
                if (_cosine(best.anchor, self.init_appearance) or 0.0) < self.cosine_min:
                    return
            else:
                # Operator-click semantics: the box the operator pointed at
                # wins; appearance is used for later re-identification.
                best = max(overlapping,
                           key=lambda track: _box_iou(self.init_bbox, track.bbox))
        else:
            if self.init_appearance is not None:
                best = max(
                    candidates,
                    key=lambda track: _cosine(track.anchor, self.init_appearance) or -1.0,
                )
                if (_cosine(best.anchor, self.init_appearance) or 0.0) < self.cosine_min:
                    return
            else:
                best = min(
                    candidates,
                    key=lambda track: float(np.linalg.norm(
                        track.filter.x[:2] - self.init_position_xy)),
                )
                if float(np.linalg.norm(best.filter.x[:2]
                                        - self.init_position_xy)) > self.position_gate_m + 1.0:
                    return

        self.target_id = best.track_id
        self.target_anchor = (
            best.anchor if best.anchor is not None else self.init_appearance
        )
        self._initialized = True

    def _reidentify(self, t: float) -> None:
        if self.target_anchor is None:
            return
        candidates = [
            track for track in self.tracks
            if track.confirmed and track.track_id != self.target_id
        ]
        if not candidates:
            return
        scored = []
        old = self._target_track()
        for track in candidates:
            similarity = _cosine(track.anchor, self.target_anchor)
            if similarity is None or similarity < self.reacquire_cosine_min:
                continue
            plausible = True
            if old is not None:
                gap = max(0.0, t - old.last_measure_t)
                gate = self.position_gate_m + self.max_speed_mps * min(gap, 3.0)
                distance = float(np.linalg.norm(
                    old.filter.x[:2] - track.filter.x[:2]))
                plausible = distance <= gate
            if plausible:
                scored.append((float(similarity), track))
        if not scored:
            return
        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.ambiguity_margin:
            # Two candidates look like the target.  Abstaining is safer than
            # picking one at random; the next frames can break the tie.
            return
        selected = scored[0][1]
        self.target_id = selected.track_id
        self.target_anchor = selected.anchor
        self._initialized = True
        self._reidentified_this_frame = True

    # -- lifecycle --------------------------------------------------

    def _prune(self, t: float) -> None:
        # The target is *not* kept forever.  After ``max_memory_s`` without a
        # measurement the track is dropped, the target id is cleared, and the
        # next frames attempt appearance re-identification among the surviving
        # tracks.  Keeping it forever would make re-identification dead code.
        self.tracks = [
            track for track in self.tracks
            if (t - track.last_measure_t) <= self.max_memory_s
        ]

    def _target_track(self) -> Optional[_Track]:
        if self.target_id is None:
            return None
        for track in self.tracks:
            if track.track_id == self.target_id:
                return track
        return None

    def _velocity_confidence(self, track: _Track, t: float) -> float:
        """How much the recent track supports a constant-velocity extrapolation."""
        history = track.history
        if len(history) < 3:
            return 0.0
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        positions = np.asarray([item[1] for item in history], dtype=np.float64)
        span = float(times[-1] - times[0])
        if span <= 1e-6:
            return 0.0
        # Least-squares constant-velocity fit, per axis.
        centered = times - times.mean()
        design = np.column_stack([centered, np.ones_like(centered)])
        fitted = []
        for axis in range(2):
            coefficients = np.linalg.lstsq(
                design, positions[:, axis], rcond=None)[0]
            fitted.append(design @ coefficients)
        residual = float(np.sqrt(np.mean(
            (np.column_stack(fitted) - positions) ** 2)))
        deltas = np.diff(times)
        valid = deltas > 1e-6
        if np.any(valid):
            speeds = np.diff(positions, axis=0)[valid] / deltas[valid, None]
            speed_std = float(np.linalg.norm(
                speeds - speeds.mean(axis=0), axis=1).mean())
        else:
            speed_std = 0.0
        span_score = float(np.clip(span / 1.0, 0.0, 1.0))
        residual_score = float(np.exp(-residual / 0.25))
        stability_score = float(np.exp(-speed_std / 0.50))
        age = max(0.0, t - track.last_measure_t)
        recency_score = float(np.exp(-age / 2.0))
        return float(np.clip(
            0.30 * span_score + 0.30 * residual_score
            + 0.20 * stability_score + 0.20 * recency_score,
            0.0, 1.0))

    def _belief(self, t: float) -> TargetBelief:
        track = self._target_track()
        if track is None:
            return TargetBelief(
                position_xy=None, velocity_xy=None, covariance_xy=None,
                status="lost", lost=True, track_id=None,
                measurement_age_s=None, identity_score=0.0,
                predicted_trajectory=[], predicted_horizon_s=0.0,
                velocity_confidence=0.0, prediction_uncertainty_m=0.0,
                trajectory_history=[], history_span_s=0.0,
            )

        age = max(0.0, t - track.last_measure_t)
        position = track.filter.x[:2].copy()
        velocity = track.filter.x[2:4].copy()
        covariance = track.filter.P[:2, :2].copy()
        visible = age <= 1e-9
        if visible:
            status, lost = "measured", False
        elif age <= self.max_prediction_s + 1e-9:
            status, lost = "predicted", False
        else:
            status, lost = "lost", True

        velocity_confidence = self._velocity_confidence(track, t)
        eigenvalues = np.linalg.eigvalsh(covariance)
        prediction_uncertainty = math.sqrt(max(
            0.0, float(np.max(eigenvalues))))
        trajectory_history = [
            dict(t=float(history_t), position_xy=history_position.copy())
            for history_t, history_position in track.history
        ]
        history_span = (float(track.history[-1][0] - track.history[0][0])
                        if len(track.history) >= 2 else 0.0)
        trajectory = []
        if not lost:
            steps = int(round(self.max_prediction_s / 0.1))
            for index in range(steps + 1):
                horizon = index * 0.1
                future_position, _, future_covariance = track.filter.state_at(
                    horizon, self.accel_std)
                trajectory.append(dict(
                    horizon_s=round(horizon, 3),
                    position_xy=future_position,
                    covariance_xy=future_covariance,
                    confidence=float(velocity_confidence * math.exp(
                        -horizon / 1.5)),
                ))

        return TargetBelief(
            position_xy=position,
            velocity_xy=velocity,
            covariance_xy=covariance,
            status=status,
            lost=lost,
            track_id=track.track_id,
            bbox=None if track.bbox is None else track.bbox.copy(),
            measurement_age_s=age,
            identity_score=float(track.identity_score),
            reidentified=bool(self._reidentified_this_frame),
            target_visible=visible,
            predicted_trajectory=trajectory,
            predicted_horizon_s=float(self.max_prediction_s),
            velocity_confidence=velocity_confidence,
            prediction_uncertainty_m=prediction_uncertainty,
            trajectory_history=trajectory_history,
            history_span_s=history_span,
        )
