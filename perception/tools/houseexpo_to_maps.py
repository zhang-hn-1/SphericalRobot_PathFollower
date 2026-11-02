#!/usr/bin/env python3
"""Convert HouseExpo floorplans (MIT) into the local map JSON schema.

HouseExpo stores each apartment as a wall polyline ``verts`` plus room-axis
bounding boxes.  This tool turns every polyline segment into a thin wall
polygon, picks two room centres as start/goal, validates an A* route through the
resulting walls, and writes the same single-scene JSON format used by
``perception/sim/map_follow.py``.

HouseExpo: https://github.com/TeaganLi/HouseExpo (MIT)
The repository here contains a subset of floorplans selected from the public
archive; see ``perception/assets/maps/houseexpo/SOURCE.md``.

    python3 perception/tools/houseexpo_to_maps.py \
        --raw /tmp/he_partial/json \
        --out perception/assets/maps/houseexpo \
        --limit 12
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import shutil
from pathlib import Path

import numpy as np


WALL_THICKNESS = 0.10
WALL_HEIGHT = 1.5
# The P0 ball has radius ~0.4 m.  HouseExpo doors are often ~0.8 m, so the
# planning inflation must be smaller than the open-area inflation.
PLAN_CLEARANCE = 0.34
RESOLUTION = 0.2


def distance_point_segment(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip((point - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def wall_segments(verts, max_length=4.0, min_length=0.12):
    segments = []
    for a, b in zip(verts[:-1], verts[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        length = float(np.linalg.norm(b - a))
        if not (min_length <= length <= max_length):
            continue
        segments.append((a, b))
    return segments


def segment_polygon(a, b, thickness=WALL_THICKNESS):
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    normal = np.array([-direction[1], direction[0]]) / length * (thickness / 2.0)
    return np.array([a + normal, b + normal, b - normal, a - normal])


def clearance(point, segments):
    return min(distance_point_segment(point, a, b) for a, b in segments)


def point_free(point, segments, margin=0.5):
    return clearance(point, segments) >= margin


def room_centres(floorplan):
    centres = []
    for category, boxes in floorplan.get("room_category", {}).items():
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            centres.append((category, np.array([(x1 + x2) / 2.0,
                                                (y1 + y2) / 2.0])))
    return centres


def astar(start, goal, segments, bounds, clearance_margin=PLAN_CLEARANCE):
    xmin, ymin, xmax, ymax = bounds
    nx = int(round((xmax - xmin) / RESOLUTION)) + 1
    ny = int(round((ymax - ymin) / RESOLUTION)) + 1

    def cell(point):
        return (int(round((point[0] - xmin) / RESOLUTION)),
                int(round((point[1] - ymin) / RESOLUTION)))

    def world(ix, iy):
        return np.array([xmin + ix * RESOLUTION, ymin + iy * RESOLUTION])

    blocked = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        x = xmin + ix * RESOLUTION
        for iy in range(ny):
            y = ymin + iy * RESOLUTION
            blocked[ix, iy] = clearance((x, y), segments) < clearance_margin

    def nearest_free(point):
        start_cell = cell(point)
        for radius in range(0, 25):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (start_cell[0] + dx, start_cell[1] + dy)
                    if (0 <= candidate[0] < nx and 0 <= candidate[1] < ny
                            and not blocked[candidate]):
                        return candidate
        return None

    start_cell = nearest_free(start)
    goal_cell = nearest_free(goal)
    if start_cell is None or goal_cell is None:
        return []
    open_heap = [(0.0, start_cell)]
    cost = {start_cell: 0.0}
    parent = {}
    visited = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal_cell:
            break
        visited.add(current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbour = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbour[0] < nx and 0 <= neighbour[1] < ny):
                    continue
                if blocked[neighbour]:
                    continue
                if dx and dy:
                    if blocked[(current[0] + dx, current[1])] or \
                            blocked[(current[0], current[1] + dy)]:
                        continue
                new_cost = cost[current] + math.hypot(dx, dy) * RESOLUTION
                if new_cost < cost.get(neighbour, float("inf")):
                    cost[neighbour] = new_cost
                    parent[neighbour] = current
                    heuristic = math.hypot(neighbour[0] - goal_cell[0],
                                           neighbour[1] - goal_cell[1]) * RESOLUTION
                    heapq.heappush(open_heap, (new_cost + heuristic, neighbour))
    if goal_cell not in parent and goal_cell != start_cell:
        return []
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(parent[cells[-1]])
    cells.reverse()
    return [world(ix, iy) for ix, iy in cells]


def path_length(path):
    return float(sum(np.linalg.norm(path[i + 1] - path[i])
                     for i in range(len(path) - 1)))


def choose_room_pair(floorplan, segments, bounds, validate=False):
    centres = [(category, point) for category, point in room_centres(floorplan)
               if point_free(point, segments, 0.55)]
    pairs = []
    for i, (category_a, a) in enumerate(centres):
        for j, (category_b, b) in enumerate(centres):
            if j <= i or category_a == category_b:
                continue
            distance = float(np.linalg.norm(a - b))
            if 5.0 <= distance <= 13.0:
                pairs.append((distance, category_a, a, category_b, b))
    pairs.sort(key=lambda item: item[0], reverse=True)
    if not validate:
        if not pairs:
            return None
        distance, category_a, a, category_b, b = pairs[0]
        return distance + 0.2 * distance, a, b, category_a, category_b, distance
    best = None
    for distance, category_a, a, category_b, b in pairs[:8]:
        path = astar(a, b, segments, bounds)
        if len(path) < 2:
            continue
        length = path_length(path)
        if length < 5.0:
            continue
        score = length + 0.2 * distance
        if best is None or score > best[0]:
            best = (score, a, b, category_a, category_b, length)
    return best


def convert_floorplan(path, out_dir, validate=False, scale=1.35):
    floorplan = json.loads(path.read_text())
    verts = floorplan.get("verts", [])
    if len(verts) < 20:
        return None
    segments = wall_segments(verts)
    if len(segments) < 12:
        return None
    bbox = floorplan.get("bbox", {})
    if "min" not in bbox:
        return None
    xmin, ymin = [float(v) for v in bbox["min"]]
    xmax, ymax = [float(v) for v in bbox["max"]]
    margin = 0.6
    bounds = (xmin - margin, ymin - margin, xmax + margin, ymax + margin)
    selection = choose_room_pair(floorplan, segments, bounds, validate=validate)
    if selection is None:
        return None
    _, start, goal, category_a, category_b, length = selection
    # HouseExpo doors are often 0.8 m; the ball is 0.8 m in diameter.  Scale the
    # apartment so the same topology is navigable with a small safety margin.
    segments = [(a * scale, b * scale) for a, b in segments]
    start = np.asarray(start) * scale
    goal = np.asarray(goal) * scale
    bounds = tuple(v * scale for v in bounds)
    length *= scale
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    obstacles = []
    for a, b in segments:
        polygon = segment_polygon(a, b)
        if polygon is None:
            continue
        obstacles.append(dict(
            height=WALL_HEIGHT,
            shape="polygon",
            vertices=[[round(float(x), 4), round(float(y), 4)]
                      for x, y in polygon],
        ))
    scene = dict(
        id=f"houseexpo_{floorplan['id'][:12]}",
        family=f"houseexpo_{category_a}_to_{category_b}",
        bounds=[round(bounds[0], 3), round(bounds[1], 3),
                round(bounds[2], 3), round(bounds[3], 3)],
        start=[round(float(start[0]), 4), round(float(start[1]), 4),
               round(float(yaw), 4)],
        goal=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
        ground_z=0.0,
        obstacles=obstacles,
        source=dict(dataset="HouseExpo", license="MIT",
                    floorplan_id=floorplan["id"],
                    scale=scale,
                    route_length_m=round(length, 3),
                    room_num=int(floorplan.get("room_num", 0))),
    )
    (out_dir / f"{scene['id']}.json").write_text(
        json.dumps(dict(role="houseexpo", scenes=[scene]), indent=2))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-scan", type=int, default=120,
                        help="only scan this many raw floorplans")
    parser.add_argument("--validate", action="store_true",
                        help="run an A* route check per floorplan (slow)")
    parser.add_argument("--scale", type=float, default=1.35,
                        help="scale floorplans (doors 0.8 m -> ~1.08 m)")
    args = parser.parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    candidates = []
    for path in sorted(raw.glob("*.json"))[:args.max_scan]:
        try:
            scene = convert_floorplan(path, out)
        except Exception:
            continue
        if scene is None:
            continue
        candidates.append((scene["source"]["route_length_m"],
                           scene["source"]["room_num"], path, scene))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = candidates[:args.limit]
    # Remove any previous generated maps/raw files so reruns are clean.
    for old in out.glob("houseexpo_*.json"):
        old.unlink()
    for old in (out / "raw").glob("*.json"):
        old.unlink()
    index = []
    for length, rooms, path, scene in selected:
        shutil.copy2(path, out / "raw" / path.name)
        (out / f"{scene['id']}.json").write_text(
            json.dumps(dict(role="houseexpo", scenes=[scene]), indent=2))
        index.append(dict(id=scene["id"], floorplan=path.name,
                          route_length_m=round(length, 3),
                          room_num=rooms))
    (out / "index.json").write_text(json.dumps(dict(
        dataset="HouseExpo", license="MIT",
        source="https://github.com/TeaganLi/HouseExpo",
        selected=index), indent=2))
    print(json.dumps(dict(converted=len(candidates), selected=len(selected),
                          out=str(out)), indent=2))
    for item in index:
        print(item)


if __name__ == "__main__":
    main()
