"""Rotunbot camera rig geometry, derived from the frozen evidence.

Source of truth
---------------
* ``robot/research_review_20261009/camera_evidence/camera_rig.py``  -- rig id and offsets
* ``robot/research_review_20261009/camera_evidence/mask_identity.json`` -- self-occlusion
* ``ball_wm/benchmark/isaac_runtime.py``                            -- focal length, apertures
* ``ball_wm/scripts/collect_data_dubins.py``                        -- optical<->USD convention

Key facts
---------
* The cameras are children of ``base_link`` (the *inner* frame), not of the rolling
  shell (``link1``).  The shell spins about Y; the cameras do not move with it.
  So there is no rolling-shutter-style image rotation to undo.
* The only image attitude perturbation is the pendulum roll from ``joint2``
  (revolute, +/-0.5236 rad).  Its geometric effect on the body is about
  +/-10 degrees, not +/-30.
* The rig is *lying down*: the optical axis points along base +X (forward),
  but image right maps to base +Z (up) and image down maps to base -Y (right).
  The image is the scene rotated by 90 degrees.

All numbers below are metres, radians, degrees, or pixels as noted.
"""

from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------ rig

CAMERA_RIG_ID = "ball_stereo_front020_up005_v2"

#: Camera optical centre offsets in the base_link frame (x forward, y left, z up).
CAMERA_FORWARD_M = 0.20
CAMERA_HALF_BASELINE_M = 0.30
CAMERA_UP_M = 0.05

#: USD euler angles (degrees) applied to the camera prims.
CAMERA_ROTATE_YZX_DEG = (0.0, -90.0, 0.0)

#: base_link origin height above the ground (ball radius + chassis height).
BASE_HEIGHT_M = 0.45
BALL_RADIUS_M = 0.40

#: Roll (degrees) that maps an Isaac Gym camera onto the frozen rig pose.
#: Isaac Gym cameras look along +X with image right = +Y and image down = -Z.
#: Rolling by -90 degrees about the optical axis gives image right = base +Z
#: and image down = base -Y, which is the frozen rig.
#:
#: Verified empirically in Isaac Gym with a vertical pillar: at 0 degrees the
#: pillar stands upright in the image; at -90 degrees it lies down, which is
#: how a person appears in the real rig.
ISAAC_GYM_RIG_ROLL_DEG = -90.0

# ------------------------------------------------------------------ optics

IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
FOCAL_LENGTH_MM = 16.45
HORIZONTAL_APERTURE_MM = 18.0
VERTICAL_APERTURE_MM = HORIZONTAL_APERTURE_MM * IMAGE_HEIGHT_PX / IMAGE_WIDTH_PX

#: Focal length in pixels.  Isaac Camera uses a square pixel model.
FOCAL_LENGTH_PX = IMAGE_WIDTH_PX * FOCAL_LENGTH_MM / HORIZONTAL_APERTURE_MM

HORIZONTAL_FOV_DEG = 2.0 * math.degrees(
    math.atan(0.5 * HORIZONTAL_APERTURE_MM / FOCAL_LENGTH_MM)
)
VERTICAL_FOV_DEG = 2.0 * math.degrees(
    math.atan(0.5 * VERTICAL_APERTURE_MM / FOCAL_LENGTH_MM)
)

# ------------------------------------------------------------------ frames


def _rotation_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


#: USD camera frame -> base_link frame (the USD camera looks along its own +Z).
R_CAM_USD_FROM_BASE = (
    _rotation_z(CAMERA_ROTATE_YZX_DEG[2])
    @ _rotation_y(CAMERA_ROTATE_YZX_DEG[1])
    @ _rotation_x(CAMERA_ROTATE_YZX_DEG[0])
)

#: optical (OpenCV: x right, y down, z forward) -> USD camera frame.
CAMERA_OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0])

#: optical frame -> base_link frame.
R_BASE_FROM_OPTICAL = R_CAM_USD_FROM_BASE @ CAMERA_OPTICAL_TO_USD

#: Unit vectors of the optical frame expressed in base_link.
OPTICAL_X_IN_BASE = R_BASE_FROM_OPTICAL @ np.array([1.0, 0.0, 0.0])  # image right
OPTICAL_Y_IN_BASE = R_BASE_FROM_OPTICAL @ np.array([0.0, 1.0, 0.0])  # image down
OPTICAL_Z_IN_BASE = R_BASE_FROM_OPTICAL @ np.array([0.0, 0.0, 1.0])  # image forward

#: The image is the scene rotated by this many degrees relative to a camera
#: that is mounted "upright" (image right = base +Y, image down = base −Z).
IMAGE_ROTATION_DEG = 90.0


def camera_position_base_link(y_offset):
    """Optical-centre position in the base_link frame."""
    return np.array([CAMERA_FORWARD_M, float(y_offset), CAMERA_UP_M])


def left_camera_position():
    return camera_position_base_link(+CAMERA_HALF_BASELINE_M)


def right_camera_position():
    return camera_position_base_link(-CAMERA_HALF_BASELINE_M)


# ------------------------------------------------------------------ projection


def project_point_to_image(point_base, camera_position_base, width=IMAGE_WIDTH_PX,
                           height=IMAGE_HEIGHT_PX, focal_px=FOCAL_LENGTH_PX):
    """Project a base_link point into the image of one camera.

    Returns ``(u, v, depth)`` with ``depth`` the range along the optical axis
    (base +X), or ``None`` when the point is behind the camera.
    """
    offset = np.asarray(point_base, dtype=np.float64) - np.asarray(
        camera_position_base, dtype=np.float64
    )
    # Optical frame coordinates: x right, y down, z forward.
    optical = R_BASE_FROM_OPTICAL.T @ offset
    forward = float(optical[2])
    if forward <= 1e-6:
        return None
    u = width / 2.0 + focal_px * float(optical[0]) / forward
    v = height / 2.0 + focal_px * float(optical[1]) / forward
    return float(u), float(v), forward


def vertical_disparity_to_range(disparity_px, baseline_m=CAMERA_HALF_BASELINE_M * 2.0,
                                focal_px=FOCAL_LENGTH_PX):
    """Range from a *vertical* disparity.

    The rig's baseline is horizontal in the base frame, but the cameras lie
    down, so the stereo correspondence runs along the image's vertical axis.
    ``disparity_px`` is signed in the OpenCV convention: a point closer to the
    left camera appears lower (larger v) in the left image.
    """
    if abs(disparity_px) < 1e-6:
        return float("inf")
    return float(focal_px * baseline_m / disparity_px)


# ------------------------------------------------------------------ calibration


def camera_intrinsics(width=IMAGE_WIDTH_PX, height=IMAGE_HEIGHT_PX,
                       focal_px=FOCAL_LENGTH_PX):
    """Isaac-style pinhole intrinsics matrix (column-vector convention)."""
    return np.array(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def optical_to_base_matrix(y_offset):
    """4x4 transform from the optical frame to base_link for one camera."""
    matrix = np.eye(4)
    matrix[:3, :3] = R_BASE_FROM_OPTICAL
    matrix[:3, 3] = camera_position_base_link(y_offset)
    return matrix


def isaac_gym_camera_roll_deg():
    """Roll to apply to an Isaac Gym camera so it matches the frozen rig."""
    return ISAAC_GYM_RIG_ROLL_DEG


def rig_metadata():
    return {
        "rig_id": CAMERA_RIG_ID,
        "parent_frame": "base_link",
        "units": "m",
        "left_position_base_link_m": list(left_camera_position()),
        "right_position_base_link_m": list(right_camera_position()),
        "rotate_yzx_deg": list(CAMERA_ROTATE_YZX_DEG),
        "baseline_m": CAMERA_HALF_BASELINE_M * 2.0,
        "focal_length_mm": FOCAL_LENGTH_MM,
        "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM,
        "vertical_aperture_mm": VERTICAL_APERTURE_MM,
        "focal_length_px": FOCAL_LENGTH_PX,
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "vertical_fov_deg": VERTICAL_FOV_DEG,
        "image_width_px": IMAGE_WIDTH_PX,
        "image_height_px": IMAGE_HEIGHT_PX,
        "base_height_m": BASE_HEIGHT_M,
        "ball_radius_m": BALL_RADIUS_M,
        "image_right_is_base": "+Z (up)",
        "image_down_is_base": "-Y (right)",
        "optical_forward_is_base": "+X (forward)",
        "stereo_disparity_axis": "vertical (image y)",
        "isaac_gym_rig_roll_deg": ISAAC_GYM_RIG_ROLL_DEG,
        "self_occlusion_visible_fraction": 0.4624,
        "self_occlusion_near_body_fraction": 0.381,
    }


# ------------------------------------------------------------------ env pose


def base_to_env(point_base, base_xy=(0.0, 0.0), base_yaw=0.0):
    """Map a base_link point into the environment frame.

    ``base_yaw`` rotates about +Z (positive is a left turn, since +Y is left).
    This is the only extra degree of freedom the active-search experiments
    add, and it is kept here rather than in the simulator so the projection
    checks can run without Isaac Gym.
    """
    p = np.asarray(point_base, dtype=np.float64)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    return np.array([
        float(base_xy[0]) + c * p[0] - s * p[1],
        float(base_xy[1]) + s * p[0] + c * p[1],
        p[2],
    ])


def camera_pose_in_env(y_offset, base_xy=(0.0, 0.0), base_yaw=0.0,
                       lookahead=10.0):
    """(position, look-at target) for one eye, in the environment frame."""
    position_base = camera_position_base_link(y_offset)
    position = base_to_env(position_base, base_xy, base_yaw)
    target = base_to_env(position_base + np.array([lookahead, 0.0, 0.0]),
                         base_xy, base_yaw)
    return position, target
