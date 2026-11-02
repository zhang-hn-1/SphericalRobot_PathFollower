"""Turn image detections into metric ground points for the target layer.

The detector sees an upright image.  The front pair used by the 20-robot
simulation has one camera on each side of the ball, toed out by six degrees;
the image horizontal axis maps to the robot's left-right axis.  This module
keeps that geometry in one place so the tracker can work in metres and never
has to know about pixels.

Two range sources are supported:

* **depth** (preferred when the sensor provides it): sample the lower part of
  the detection box and take a robust depth quantile.  Isaac Gym returns
  negative planar depth, so the absolute value is used.
* **bbox height** (fallback): ``range = person_height * fx / box_height``.
  This is the standard monocular person-range prior and is surprisingly hard
  to beat for an upright, unoccluded person.

The output is a ground-plane point in the robot's ``base_link`` frame
(+x forward, +y left), plus a diagonal position covariance.  The caller can
then rotate it into the world frame with :func:`base_to_world`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _clip_box(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _depth_sample(depth, box, width, height, quantile=25.0):
    """Robust depth in the lower-central part of a person box.

    The very bottom row often hits the floor, and the box corners hit the
    background, so the sample is restricted to the lower half and the central
    half-width, then a low quantile is used to keep the person's surface.
    """
    if depth is None:
        return None
    values = np.asarray(depth)
    if values.ndim != 2:
        return None
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    box_width = x2 - x1
    box_height = y2 - y1
    u0 = int(round(x1 + 0.25 * box_width))
    u1 = int(round(x2 - 0.25 * box_width))
    v0 = int(round(y1 + 0.55 * box_height))
    v1 = int(round(y2))
    u0, u1 = max(0, u0), min(values.shape[1] - 1, u1)
    v0, v1 = max(0, v0), min(values.shape[0] - 1, v1)
    if u1 <= u0 or v1 <= v0:
        return None
    patch = np.abs(values[v0:v1 + 1, u0:u1 + 1].astype(np.float64))
    valid = patch[np.isfinite(patch) & (patch > 0.2) & (patch < 30.0)]
    if valid.size < 4:
        return None
    return float(np.percentile(valid, quantile))


@dataclass(frozen=True)
class FrontPairCamera:
    """Geometry of the 20-robot front stereo pair (one eye at a time)."""

    width: int = 768
    height: int = 1024
    horizontal_fov_deg: float = 78.3
    toe_out_deg: float = 6.0
    cam_forward: float = 0.21565
    cam_half_baseline: float = 0.31821
    cam_up: float = 0.097
    ball_radius: float = 0.40
    default_person_height: float = 1.70

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(
            math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def cx(self) -> float:
        return self.width / 2.0 - 0.5

    @property
    def cy(self) -> float:
        return self.height / 2.0 - 0.5

    def detection_to_base(
        self,
        bbox,
        depth: Optional[np.ndarray] = None,
        side: str = "left",
        person_height: Optional[float] = None,
        depth_quantile: float = 25.0,
    ) -> Optional[dict]:
        """Convert one upright-image box to a ground point in ``base_link``.

        Returns
        -------
        dict or None
            ``position_base_xy``, ``range_m``, ``bearing_rad``,
            ``range_source``, ``depth_m``, ``height_range_m``,
            ``covariance_xy``.
        """
        clipped = _clip_box(bbox, self.width, self.height)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        u = (x1 + x2) / 2.0
        bearing = math.atan2(self.cx - u, self.focal_px)

        height = self.default_person_height if person_height is None else float(person_height)
        height_range = height * self.focal_px / height_px
        depth_value = _depth_sample(depth, (x1, y1, x2, y2),
                                    self.width, self.height,
                                    quantile=depth_quantile)
        if depth_value is not None and not (0.4 <= depth_value <= 25.0):
            depth_value = None

        forward = height_range
        source = "bbox_height"
        consistent = True
        range_ratio = None
        if depth_value is not None:
            # A depth sample can hit the floor, a background object, or an
            # occluder, and a partial box makes the height prior too large.
            # When the two independent estimates disagree, the metric position
            # is not trustworthy; the caller should keep the appearance and
            # let the motion model coast instead of updating with a jump.
            range_ratio = depth_value / max(height_range, 1e-6)
            if 0.67 <= range_ratio <= 1.50:
                forward = 0.75 * depth_value + 0.25 * height_range
                source = "depth_fused"
            else:
                forward = depth_value
                source = "inconsistent"
                consistent = False

        if not math.isfinite(forward) or forward <= 0.0:
            return None

        toe = math.radians(self.toe_out_deg) * (1.0 if side == "left" else -1.0)
        cam_y = self.cam_half_baseline * (1.0 if side == "left" else -1.0)
        # Optical axis and the direction 90 degrees to its left, both in the
        # base frame.  The camera is toed out, so this is not just (1, 0).
        axis = np.array([math.cos(toe), math.sin(toe)], dtype=np.float64)
        left_dir = np.array([-math.sin(toe), math.cos(toe)], dtype=np.float64)
        lateral_left = forward * math.tan(bearing)
        position = np.array([
            self.cam_forward + forward * axis[0] + lateral_left * left_dir[0],
            cam_y + forward * axis[1] + lateral_left * left_dir[1],
        ])
        range_m = float(np.linalg.norm(position - np.array([self.cam_forward, cam_y])))
        if source == "bbox_height":
            sigma = 0.15 + 0.07 * range_m
        else:
            sigma = 0.07 + 0.035 * range_m
        covariance = np.diag([sigma * sigma, sigma * sigma])

        return dict(
            position_base_xy=position,
            range_m=range_m,
            bearing_rad=float(bearing),
            range_source=source,
            consistent=bool(consistent),
            range_ratio=None if range_ratio is None else float(range_ratio),
            depth_m=depth_value,
            height_range_m=float(height_range),
            covariance_xy=covariance,
        )

    def detection_to_world(
        self,
        bbox,
        robot_xy,
        robot_yaw: float,
        depth: Optional[np.ndarray] = None,
        side: str = "left",
        person_height: Optional[float] = None,
    ) -> Optional[dict]:
        base = self.detection_to_base(
            bbox, depth=depth, side=side, person_height=person_height)
        if base is None:
            return None
        position = base_to_world(
            base["position_base_xy"], robot_xy=robot_xy, robot_yaw=robot_yaw)
        base["position_world_xy"] = position
        base["robot_xy"] = np.asarray(robot_xy, dtype=np.float64).copy()
        base["robot_yaw"] = float(robot_yaw)
        return base


def base_to_world(position_base_xy, robot_xy=(0.0, 0.0), robot_yaw: float = 0.0):
    """Rotate a base-frame ground point by the robot yaw and translate it."""
    point = np.asarray(position_base_xy, dtype=np.float64).reshape(2)
    origin = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    c, s = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
    return np.array([
        origin[0] + c * point[0] - s * point[1],
        origin[1] + s * point[0] + c * point[1],
    ])


def world_to_base(position_world_xy, robot_xy=(0.0, 0.0), robot_yaw: float = 0.0):
    point = np.asarray(position_world_xy, dtype=np.float64).reshape(2)
    origin = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    delta = point - origin
    c, s = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
    return np.array([c * delta[0] + s * delta[1],
                     -s * delta[0] + c * delta[1]])
