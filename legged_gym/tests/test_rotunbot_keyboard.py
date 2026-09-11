"""Unit tests for the simulator-independent Rotunbot keyboard mapping."""

import unittest

import numpy as np

from legged_gym.teleop import RotunbotKeyboardController


class RotunbotKeyboardControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = RotunbotKeyboardController(
            forward_speed=1.0,
            steering_position=0.2,
        )

    def test_wasd_mapping_uses_policy_action_scales(self):
        self.controller.handle_event("teleop_forward", 1.0)
        np.testing.assert_allclose(
            self.controller.normalized_action(1.0, 0.5), [1.0, 0.0]
        )
        self.controller.handle_event("teleop_left", 1.0)
        np.testing.assert_allclose(
            self.controller.normalized_action(1.0, 0.5), [1.0, -0.4]
        )
        self.controller.handle_event("teleop_right", 1.0)
        np.testing.assert_allclose(
            self.controller.normalized_action(1.0, 0.5), [1.0, 0.4]
        )

    def test_stop_and_reset(self):
        self.controller.handle_event("teleop_reverse", 1.0)
        self.controller.handle_event("teleop_left", 1.0)
        self.controller.handle_event("teleop_stop", 1.0)
        np.testing.assert_allclose(
            self.controller.normalized_action(1.0, 0.5), [0.0, 0.0]
        )
        self.controller.handle_event("teleop_reset", 1.0)
        self.assertTrue(self.controller.consume_reset_request())
        self.assertFalse(self.controller.consume_reset_request())

    def test_release_and_unknown_events_do_not_change_state(self):
        self.assertFalse(self.controller.handle_event("teleop_forward", 0.0))
        self.assertFalse(self.controller.handle_event("unknown", 1.0))
        np.testing.assert_allclose(
            self.controller.normalized_action(1.0, 0.5), [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()
