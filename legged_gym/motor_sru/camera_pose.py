"""Rigid base_link camera mounting, independent of simulator body-pose caches."""
import numpy as np


def mounted_camera_world_poses(root_states, mount_position, mount_quaternion):
    """Return [N, xyz+xyzw] from the full root pose and fixed camera transform.

    All three root attitude axes are preserved, including chassis roll. This
    only composes rigid transforms; it does not alter depth pixels or level the
    camera. Root states and the result use the simulator's world coordinates.
    """
    roots = np.asarray(root_states, dtype=np.float64)
    q = roots[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    mount_q = np.asarray(mount_quaternion, dtype=np.float64)
    mount_q = mount_q / np.linalg.norm(mount_q)
    offset = np.broadcast_to(np.asarray(mount_position, dtype=np.float64), q[:, :3].shape)
    uv = np.cross(q[:, :3], offset)
    position = roots[:, :3] + offset + 2.0 * (q[:, 3:4] * uv + np.cross(q[:, :3], uv))
    xyz = (q[:, 3:4] * mount_q[:3] + mount_q[3] * q[:, :3]
           + np.cross(q[:, :3], mount_q[:3]))
    w = q[:, 3] * mount_q[3] - np.sum(q[:, :3] * mount_q[:3], axis=1)
    result = np.concatenate((position, xyz, w[:, None]), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite root or mount transform for real depth camera")
    return result
