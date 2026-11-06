#!/usr/bin/env python3
"""Compose the maze follow video: front camera + top-down maze + telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def transform(points, start, yaw):
    points = np.asarray(points, dtype=np.float64)
    delta = points - start
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.stack([
        c * delta[:, 0] - s * delta[:, 1],
        s * delta[:, 0] + c * delta[:, 1],
    ], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    blob = json.loads(Path(args.map_json).read_text())
    if "scenes" in blob:
        scene = next((item for item in blob["scenes"]
                      if not args.scene_id or item["id"] == args.scene_id),
                     blob["scenes"][0])
    else:
        scene = blob
    start = np.asarray(scene["start"][:2], dtype=np.float64)
    goal = np.asarray(scene["goal"], dtype=np.float64)
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    goal_t = transform(goal[None, :], start, yaw)[0]
    walls = [transform(np.asarray(ob["vertices"]), start, yaw)
             for ob in scene.get("obstacles", [])
             if ob.get("shape") == "polygon"]
    all_points = [goal_t] + [point for wall in walls for point in wall]
    all_points = np.asarray(all_points)
    low = all_points.min(axis=0) - 1.2
    high = all_points.max(axis=0) + 1.2
    span = np.maximum(high - low, 1e-6)

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()
            if line.strip()]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width, height = 768, 1024
    panel_h = height // 2
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (width, height))

    def map_point(point):
        rel = (np.asarray(point[:2], dtype=np.float64) - low) / span
        return (int(rel[0] * (width - 1)),
                int((1.0 - rel[1]) * (panel_h - 1)))

    target_trail, robot_trail = [], []
    for row in rows:
        ok, frame = cap.read()
        if not ok:
            break
        target_trail.append(row["target_xy"])
        robot_trail.append(row["robot_xy"])
        camera = cv2.resize(frame, (width, panel_h),
                            interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:panel_h] = camera
        map_panel = np.full((panel_h, width, 3), 235, dtype=np.uint8)
        for wall in walls:
            pts = np.asarray([map_point(point) for point in wall],
                             dtype=np.int32)
            cv2.fillPoly(map_panel, [pts], (150, 150, 155))
        if len(target_trail) > 1:
            cv2.polylines(
                map_panel,
                [np.asarray([map_point(p) for p in target_trail],
                            dtype=np.int32)],
                False, (0, 180, 0), 2)
        if len(robot_trail) > 1:
            cv2.polylines(
                map_panel,
                [np.asarray([map_point(p) for p in robot_trail],
                            dtype=np.int32)],
                False, (255, 150, 0), 2)
        cv2.circle(map_panel, map_point(row["target_xy"]), 6, (0, 0, 255), -1)
        cv2.circle(map_panel, map_point(row["robot_xy"]), 6, (255, 150, 0), -1)
        cv2.putText(map_panel, "top-down maze", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        text = [
            f"t={row['t']:4.1f}s  status={row['status']}",
            f"d={row['distance']:.2f}m  clr={row['robot_clearance']:.2f}m",
            f"progress={row['path_progress']*100:.0f}%  id={row['track_id']}",
            f"identity={bool(row['identity_correct'])}",
        ]
        for index, line in enumerate(text):
            cv2.putText(map_panel, line, (10, 48 + 22 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        canvas[panel_h:] = map_panel
        cv2.putText(canvas, "front camera", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        out.write(canvas)
    cap.release()
    out.release()
    print(json.dumps(dict(video=args.out, frames=len(target_trail)), indent=2))


if __name__ == "__main__":
    main()
