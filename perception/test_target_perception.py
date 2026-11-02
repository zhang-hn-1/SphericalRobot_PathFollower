#!/usr/bin/env python3
"""Unit checks for the world-frame target layer and range-bearing helpers.

Run with the system Python (NumPy only):

    python3 -m unittest perception.test_target_perception -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from perception.range_bearing import FrontPairCamera, base_to_world, world_to_base
from perception.target_perception import (PersonDetection, TargetPerceptionLayer)


def detection(x, y, appearance=None, bbox=None, score=0.9, covariance=None):
    return PersonDetection(
        position_xy=np.asarray([x, y], dtype=np.float64),
        bbox=None if bbox is None else np.asarray(bbox, dtype=np.float64),
        appearance=None if appearance is None else np.asarray(appearance, dtype=np.float64),
        score=float(score),
        covariance_xy=covariance,
    )


class TargetLayerTests(unittest.TestCase):
    def test_lock_and_crossing_keeps_distinct_target(self):
        layer = TargetPerceptionLayer(
            init_bbox=[95, 100, 105, 200],
            init_appearance=[1.0, 0.0, 0.0],
            max_prediction_s=1.0,
        )
        target_app = np.array([1.0, 0.0, 0.0])
        distractor_app = np.array([0.0, 1.0, 0.0])
        wrong = 0
        ids = set()
        for step in range(60):
            t = step * 0.1
            tx, ty = 5.0, -2.0 + 0.4 * t
            dx, dy = 5.0, 2.0 - 0.4 * t
            belief = layer.update([
                detection(tx, ty, target_app, bbox=[90, 100, 110, 200]),
                detection(dx, dy, distractor_app, bbox=[90, 100, 110, 200]),
            ], t)
            self.assertIsNotNone(belief.position_xy)
            ids.add(belief.track_id)
            target_error = np.linalg.norm(belief.position_xy - [tx, ty])
            distractor_error = np.linalg.norm(belief.position_xy - [dx, dy])
            if distractor_error < target_error:
                wrong += 1
        self.assertEqual(wrong, 0)
        self.assertEqual(len(ids), 1)
        self.assertEqual(layer.target_id, 1)

    def test_identical_appearance_crossing_uses_motion_continuity(self):
        appearance = np.array([1.0, 0.0, 0.0])
        first_box = [50, 100, 70, 200]
        layer = TargetPerceptionLayer(
            init_bbox=first_box, init_appearance=appearance,
            max_prediction_s=1.0, ambiguity_margin=0.06,
        )
        wrong = 0
        for step in range(60):
            t = step * 0.1
            tx, ty = 5.0, -2.0 + 0.4 * t
            dx, dy = 5.0, 2.0 - 0.4 * t
            target_box = [90 + 20 * ty, 100, 110 + 20 * ty, 200]
            distractor_box = [90 + 20 * dy, 100, 110 + 20 * dy, 200]
            belief = layer.update([
                detection(tx, ty, appearance, target_box),
                detection(dx, dy, appearance, distractor_box),
            ], t)
            self.assertIsNotNone(belief.position_xy)
            target_error = np.linalg.norm(belief.position_xy - [tx, ty])
            distractor_error = np.linalg.norm(belief.position_xy - [dx, dy])
            if distractor_error < target_error:
                wrong += 1
        self.assertEqual(wrong, 0)

    def test_one_second_dropout_is_predicted_then_lost(self):
        appearance = np.array([1.0, 0.0, 0.0])
        layer = TargetPerceptionLayer(
            init_bbox=[90, 100, 110, 200],
            init_appearance=appearance,
            max_prediction_s=1.0,
            max_memory_s=5.0,
            accel_std=0.6,
        )
        # 1.9 s of measurements, then the target is fully occluded.
        for step in range(20):
            t = step * 0.1
            belief = layer.update([
                detection(5.0, 0.5 * t, appearance, [90, 100, 110, 200]),
                detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
            ], t)
        self.assertEqual(belief.status, "measured")
        initial_covariance = float(np.trace(belief.covariance_xy))

        # Ages 0.1 ... 1.0 s: predicted, position still close to the true
        # constant-velocity trajectory.
        for step in range(20, 30):
            t = step * 0.1
            belief = layer.update([
                detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
            ], t)
            self.assertEqual(belief.status, "predicted", (t, belief.measurement_age_s))
            self.assertFalse(belief.lost)
            self.assertAlmostEqual(belief.measurement_age_s, t - 1.9, places=6)
            true_position = np.array([5.0, 0.5 * t])
            self.assertLess(np.linalg.norm(belief.position_xy - true_position), 0.08)
            self.assertGreater(float(np.trace(belief.covariance_xy)), initial_covariance)

        # Age 1.1 s: outside the supported prediction horizon.
        t = 3.0
        belief = layer.update([
            detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
        ], t)
        self.assertEqual(belief.status, "lost")
        self.assertTrue(belief.lost)
        self.assertLess(abs(belief.measurement_age_s - 1.1), 1e-6)
        self.assertLess(np.linalg.norm(
            belief.position_xy - np.array([5.0, 0.5 * t])), 0.10)

        # The target returns: the same track is re-acquired, not a new one.
        t = 3.1
        belief = layer.update([
            detection(5.0, 0.5 * t, appearance, [90, 100, 110, 200]),
            detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
        ], t)
        self.assertEqual(belief.status, "measured")
        self.assertEqual(belief.track_id, 1)
        self.assertAlmostEqual(belief.measurement_age_s, 0.0)

    def test_reidentification_after_track_pruned(self):
        appearance = np.array([1.0, 0.0, 0.0])
        layer = TargetPerceptionLayer(
            init_bbox=[90, 100, 110, 200],
            init_appearance=appearance,
            max_prediction_s=1.0,
            max_memory_s=2.0,
        )
        for step in range(20):
            t = step * 0.1
            layer.update([
                detection(5.0, 0.4 * t, appearance, [90, 100, 110, 200]),
                detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
            ], t)
        # Stay hidden long enough that the target track is pruned.
        for step in range(20, 46):
            t = step * 0.1
            layer.update([
                detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
            ], t)
        self.assertIsNone(layer.target_id)

        # Reappear with the same appearance but a different track id.
        t = 4.6
        belief = layer.update([
            detection(6.0, 3.0, appearance, [150, 100, 170, 200]),
            detection(4.0, 2.0, [0.0, 1.0, 0.0], [100, 100, 120, 200]),
        ], t)
        self.assertEqual(belief.status, "measured")
        self.assertTrue(belief.reidentified)
        self.assertIsNotNone(layer.target_id)

    def test_selection_waits_for_a_click_overlap(self):
        layer = TargetPerceptionLayer(
            init_bbox=[90, 100, 110, 200],
            init_appearance=np.array([1.0, 0.0, 0.0]),
            max_prediction_s=1.0,
        )
        belief = layer.update([
            detection(5.0, 0.0, [1.0, 0.0, 0.0], [300, 100, 320, 200]),
        ], 0.0)
        self.assertEqual(belief.status, "lost")
        self.assertIsNone(layer.target_id)

        belief = layer.update([
            detection(5.0, 0.0, [1.0, 0.0, 0.0], [90, 100, 110, 200]),
        ], 0.1)
        self.assertEqual(belief.status, "measured")
        self.assertIsNotNone(layer.target_id)

    def test_range_bearing_height_round_trip(self):
        camera = FrontPairCamera()
        # A person 5 m ahead and 1 m to the left, at the robot's +y side.
        point = np.array([5.0, 1.0])
        camera_position = np.array([camera.cam_forward, camera.cam_half_baseline])
        toe = math.radians(camera.toe_out_deg)
        axis = np.array([math.cos(toe), math.sin(toe)])
        left_dir = np.array([-math.sin(toe), math.cos(toe)])
        delta = point - camera_position
        forward = float(delta @ axis)
        lateral_left = float(delta @ left_dir)
        bearing = math.atan2(lateral_left, forward)
        u = camera.cx - camera.focal_px * math.tan(bearing)
        height_px = camera.default_person_height * camera.focal_px / forward
        bbox = [u - 10.0, 200.0, u + 10.0, 200.0 + height_px]
        result = camera.detection_to_base(bbox, depth=None, side="left")
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result["position_base_xy"] - point), 0.03)
        self.assertAlmostEqual(result["range_m"],
                               float(np.linalg.norm(delta)), places=2)

    def test_depth_sample_prefers_the_person(self):
        camera = FrontPairCamera()
        depth = np.full((camera.height, camera.width), 12.0, dtype=np.float32)
        # Person box in the middle; the lower-central patch is at 4.8 m.
        bbox = [300.0, 300.0, 420.0, 800.0]
        depth[560:760, 340:380] = -4.8
        result = camera.detection_to_base(
            bbox, depth=depth, side="left", person_height=1.7)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["depth_m"])
        self.assertLess(abs(result["depth_m"] - 4.8), 0.2)

    def test_base_world_round_trip(self):
        point = np.array([3.0, -1.0])
        world = base_to_world(point, robot_xy=(2.0, 4.0), robot_yaw=0.7)
        back = world_to_base(world, robot_xy=(2.0, 4.0), robot_yaw=0.7)
        self.assertLess(np.linalg.norm(back - point), 1e-9)


if __name__ == "__main__":
    unittest.main()
