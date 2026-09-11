"""Stateful WASD command mapping for the two-axis Rotunbot controller."""

import numpy as np


class RotunbotKeyboardController:
    """Convert semantic viewer events into normalized two-axis actions.

    Commands are latched because Isaac Gym keyboard events are press events.
    W/S set longitudinal speed and center steering; A/D change steering while
    preserving speed.  Space stops both axes.
    """

    def __init__(self, forward_speed=1.0, steering_position=0.2):
        self.forward_speed = float(forward_speed)
        self.steering_position = float(steering_position)
        if self.forward_speed <= 0.0:
            raise ValueError("forward_speed must be positive")
        if self.steering_position <= 0.0:
            raise ValueError("steering_position must be positive")

        self.desired_speed = 0.0
        self.desired_steering = 0.0
        self._reset_requested = False

    def handle_event(self, action, value):
        """Apply one semantic key event and report whether state changed."""
        if value <= 0.0:
            return False

        if action == "teleop_forward":
            self.desired_speed = self.forward_speed
            self.desired_steering = 0.0
        elif action == "teleop_reverse":
            self.desired_speed = -self.forward_speed
            self.desired_steering = 0.0
        elif action == "teleop_left":
            self.desired_steering = -self.steering_position
        elif action == "teleop_right":
            self.desired_steering = self.steering_position
        elif action == "teleop_stop":
            self.stop()
        elif action == "teleop_reset":
            self.stop()
            self._reset_requested = True
        else:
            return False
        return True

    def stop(self):
        self.desired_speed = 0.0
        self.desired_steering = 0.0

    def consume_reset_request(self):
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def normalized_action(self, first_action_scale, second_action_scale):
        first_action_scale = float(first_action_scale)
        second_action_scale = float(second_action_scale)
        if first_action_scale == 0.0 or second_action_scale == 0.0:
            raise ValueError("action scales must be non-zero")
        return np.asarray(
            [
                self.desired_speed / first_action_scale,
                self.desired_steering / second_action_scale,
            ],
            dtype=np.float32,
        )

    def status(self):
        return (
            f"speed_target={self.desired_speed:+.2f} rad/s, "
            f"steering_target={self.desired_steering:+.2f} rad"
        )
