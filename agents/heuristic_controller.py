from __future__ import annotations

from typing import Tuple

from .sensors import SensorReadings


# Tunable constants for the heuristic controller
TURN_GAIN: float = 1.5
FORWARD_BASE: float = 0.8

WALL_FRONT_THRESHOLD: float = 0.2
WALL_SIDE_THRESHOLD: float = 0.15
WALL_AVOID_TURN_GAIN: float = 0.9

FOOD_DISTANCE_SLOWDOWN_START: float = 0.2
MIN_FORWARD_NEAR_FOOD: float = 0.3


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_heuristic_action(readings: SensorReadings) -> Tuple[float, float]:
    """Map sensor readings to (forward, turn) using simple rules.

    The policy:
      - turns toward the food direction,
      - slows and turns away when a wall is close in front,
      - keeps moving forward when the path is clear.
    """
    forward = FORWARD_BASE
    turn = 0.0

    # Turn toward the food: food_angle is in [-1, 1]
    turn += TURN_GAIN * readings.food_angle

    # Wall avoidance: if something is close in front, slow down and steer away
    if readings.wall_front < WALL_FRONT_THRESHOLD:
        # Reduce forward proportionally to how close the wall is
        scale = readings.wall_front / WALL_FRONT_THRESHOLD
        forward *= _clamp(scale, 0.0, 1.0)

        # Turn away from the closer side wall
        if readings.wall_left > readings.wall_right:
            # More room on the left, so turn left
            turn -= WALL_AVOID_TURN_GAIN
        else:
            # More room on the right, so turn right
            turn += WALL_AVOID_TURN_GAIN

    # Extra slowdown when hugging side walls
    if readings.wall_left < WALL_SIDE_THRESHOLD or readings.wall_right < WALL_SIDE_THRESHOLD:
        forward *= 0.5

    # Gently slow down when very close to the food to avoid overshooting
    if readings.food_distance < FOOD_DISTANCE_SLOWDOWN_START:
        factor = readings.food_distance / FOOD_DISTANCE_SLOWDOWN_START
        forward *= _clamp(factor, MIN_FORWARD_NEAR_FOOD, 1.0)

    # Clamp outputs to the expected range for Robot.apply_manual_input
    forward = _clamp(forward, -1.0, 1.0)
    turn = _clamp(turn, -1.0, 1.0)

    return forward, turn

