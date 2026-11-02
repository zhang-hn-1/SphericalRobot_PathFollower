"""Standalone 2D stage wall map used by the fast predictive-follow env.

The ReplicaCAD stage OBJ is rasterised exactly as in ``sim/stage_follow.py``;
this module has no Isaac Gym dependency so the RL environment can use the same
visibility geometry on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


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
        self.occupied = occupied
        self.distance = distance_transform_edt(~occupied) * self.resolution

    def clearance(self, point):
        ix = int(round((point[0] - self.minimum[0]) / self.resolution))
        iy = self.height - 1 - int(round(
            (point[1] - self.minimum[1]) / self.resolution))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return 0.0
        return float(self.distance[iy, ix])

    def line_of_sight(self, start, end, samples=48):
        """Boolean 2D visibility through the stage wall occupancy grid."""
        a = np.asarray(start, dtype=np.float64).reshape(2)
        b = np.asarray(end, dtype=np.float64).reshape(2)
        for alpha in np.linspace(0.02, 0.98, samples):
            point = a + alpha * (b - a)
            ix = int(round((point[0] - self.minimum[0]) / self.resolution))
            iy = self.height - 1 - int(round(
                (point[1] - self.minimum[1]) / self.resolution))
            if not (0 <= ix < self.width and 0 <= iy < self.height):
                return False
            if self.occupied[iy, ix]:
                return False
        return True
