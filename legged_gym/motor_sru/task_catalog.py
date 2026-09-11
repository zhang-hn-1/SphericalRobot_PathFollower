"""Immutable task catalog and no-replacement sampler, isolated from velocity control.

Geometry hashing and sampling order preserve the prior pillar catalog convention.
Additional admission checks require a real pillar to cross the zero-width chord.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_hash(task):
    geometry = dict(posts=sorted([[round(float(x), 6), round(float(y), 6)] for x, y in task["posts"]]),
                    radius_m=float(task["post_radius_m"]), height_m=float(task["post_height_m"]),
                    map_bounds=[float(v) for v in task["map_bounds"]])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def direct_line_metrics(task):
    start, goal, posts = (np.asarray(task[k], dtype=np.float64) for k in ("start", "goal", "posts"))
    direction = goal - start
    squared_length = float(direction.dot(direction))
    if squared_length <= 0 or not np.isfinite(np.concatenate((start, goal, posts.ravel()))).all():
        raise ValueError("Task geometry must be finite with distinct endpoints")
    fractions = (posts - start).dot(direction) / squared_length
    distances = np.linalg.norm(posts - start - np.clip(fractions, 0., 1.)[:, None] * direction, axis=1)
    return fractions, distances


def validate_task(task, strict_distance=0.35):
    if task["geometry_sha256"] != geometry_hash(task):
        raise ValueError("Geometry hash mismatch: " + task["task_id"])
    if len(task["posts"]) != 8 or task["map_bounds"] != [-8., 8., -8., 8.]:
        raise ValueError("Expected eight pillars in a 16 m square")
    if abs(task["post_radius_m"] - .4) > 1e-9 or abs(task["post_height_m"] - 1.5) > 1e-9:
        raise ValueError("Pillar dimensions mismatch")
    delta = float(task["delta"])
    angle_error = float(task["yaw"] - task["theta_ref"] - delta)
    if abs(delta) > np.pi / 12 + 1e-8 or abs(np.arctan2(np.sin(angle_error), np.cos(angle_error))) > 1e-7:
        raise ValueError("Initial yaw must preserve the stored +/-15 degree tangent offset")
    fractions, distances = direct_line_metrics(task)
    physical_block = np.any((fractions >= .2) & (fractions <= .8) & (distances <= strict_distance + 1e-8))
    category = task["category"]
    if category == "required_detour":
        if not physical_block:
            raise ValueError("Detour task lacks a central pillar intersecting the straight line: " + task["task_id"])
    elif category == "clear_direct":
        if distances.min() - .8 < .6 - 1e-7:
            raise ValueError("Direct task lacks the required swept-body clearance")
    else:
        raise ValueError("Unknown task category")
    radius = task["requested_min_turn_radius_m"]
    # JSON null denotes the infinite radius of an exactly straight witness.
    if (radius is not None and float(radius) < 2. - 1e-7) or float(task["duration_s"]) > 180.:
        raise ValueError("Nominal witness violates radius or time admission")
    if float(task["requested_path_min_clearance_m"]) < 0 or float(task["requested_boundary_clearance_m"]) < 0:
        raise ValueError("Nominal witness collides or leaves bounds")
    return dict(pillar_intersects_line=bool(np.any(distances <= .4)),
                swept_body_line_collision=bool(np.any(distances <= .8)),
                minimum_center_to_line_m=float(distances.min()))


class PillarTaskCatalog:
    def __init__(self, manifest_path, project_root, expected_role=None, strict_distance=.35):
        path = Path(manifest_path)
        if not path.is_absolute():
            path = Path(project_root) / path
        self.path = path.resolve()
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        self.role = self.manifest["role"]
        if expected_role and self.role != expected_role:
            raise ValueError("Manifest role mismatch")
        if self.manifest.get("status") != "COMPLETE":
            raise ValueError("Incomplete task manifest")
        self.tasks = self.manifest["tasks"]
        if not self.tasks or len({t["task_id"] for t in self.tasks}) != len(self.tasks):
            raise ValueError("Empty catalog or duplicate task IDs")
        self.admission = [validate_task(t, strict_distance) for t in self.tasks]
        required = sum(t["category"] == "required_detour" for t in self.tasks)
        if required * 5 != len(self.tasks) * 4:
            raise ValueError("Catalog must contain exactly 80% strict detours and 20% direct tasks")
        self.manifest_sha256 = sha256(path)


def validate_pool(project_root, pool_root, strict_distance=.35):
    root = Path(project_root) / pool_root
    generation_path = root / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if generation.get("status") != "COMPLETE":
        raise ValueError("Refusing unfinished motor task generation")
    identity = {"generation_sha256": sha256(generation_path), "roles": {}}
    catalogs, geometries = {}, set()
    for role in ("train", "val", "dev20"):
        catalog = PillarTaskCatalog(root / "data" / (role + ".json"), project_root, role, strict_distance)
        record = generation["roles"][role]
        if catalog.manifest_sha256 != record["sha256"] or len(catalog.tasks) != int(record["count"]):
            raise ValueError("Task manifest SHA/count differs from completed generation")
        hashes = {t["geometry_sha256"] for t in catalog.tasks}
        if len(hashes) != len(catalog.tasks) or hashes & geometries:
            raise ValueError("Duplicate geometry within/across data splits")
        geometries |= hashes
        identity["roles"][role] = dict(sha256=catalog.manifest_sha256, count=len(catalog.tasks))
        catalogs[role] = catalog
    return catalogs, identity


class TaskSampler:
    def __init__(self, count, seed, ordered_first=False):
        self.rng = np.random.default_rng(int(seed))
        self.order = np.arange(int(count), dtype=np.int64)
        if not ordered_first:
            self.rng.shuffle(self.order)
        self.cursor = 0

    def draw(self, count):
        selected = []
        while len(selected) < count:
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(count - len(selected), len(self.order) - self.cursor)
            selected.extend(self.order[self.cursor:self.cursor + take].tolist())
            self.cursor += take
        return np.asarray(selected, dtype=np.int64)

    def state_dict(self):
        return dict(order=self.order.tolist(), cursor=self.cursor, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        order = np.asarray(state["order"], dtype=np.int64)
        cursor = int(state["cursor"])
        if sorted(order.tolist()) != list(range(len(self.order))) or not 0 <= cursor <= len(order):
            raise ValueError("Invalid task sampler permutation/cursor")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        self.order, self.cursor, self.rng = order, cursor, rng
