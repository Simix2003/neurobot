from __future__ import annotations

import math
from dataclasses import dataclass

from config import WINDOW_HEIGHT, WINDOW_WIDTH
from world.food import Food
from .robot import Robot


@dataclass
class SensorReadings:
    food_distance: float
    food_angle: float
    wall_front: float
    wall_left: float
    wall_right: float


MAX_SENSOR_DISTANCE: float = math.hypot(WINDOW_WIDTH, WINDOW_HEIGHT)


def compute_sensor_readings(robot: Robot, food: Food) -> SensorReadings:
    """Compute normalized food and wall sensors for the current world state."""
    food_distance, food_angle = _food_sensors(robot, food)
    wall_front = _wall_distance(robot, 0.0)
    wall_left = _wall_distance(robot, math.pi / 2.0)
    wall_right = _wall_distance(robot, -math.pi / 2.0)

    return SensorReadings(
        food_distance=food_distance,
        food_angle=food_angle,
        wall_front=wall_front,
        wall_left=wall_left,
        wall_right=wall_right,
    )


def _food_sensors(robot: Robot, food: Food) -> tuple[float, float]:
    dx = food.x - robot.x
    dy = food.y - robot.y

    distance = math.hypot(dx, dy)
    norm_distance = min(distance / MAX_SENSOR_DISTANCE, 1.0)

    # Angle from robot heading to food direction
    absolute_angle = math.atan2(dy, dx)
    relative_angle = absolute_angle - robot.angle
    # Wrap to [-pi, pi]
    relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
    # Normalize to [-1, 1]
    norm_angle = max(-1.0, min(1.0, relative_angle / math.pi))

    return norm_distance, norm_angle


def _wall_distance(robot: Robot, direction_offset: float) -> float:
    """Compute normalized distance to the nearest wall in a given direction."""
    angle = robot.angle + direction_offset
    dx = math.cos(angle)
    dy = math.sin(angle)

    # Avoid division by zero; tiny epsilon
    eps = 1e-6
    dx = dx if abs(dx) > eps else eps
    dy = dy if abs(dy) > eps else eps

    t_candidates: list[float] = []

    # Intersections with vertical walls x = 0 and x = WINDOW_WIDTH
    for x_wall in (0.0, float(WINDOW_WIDTH)):
        t = (x_wall - robot.x) / dx
        if t > 0:
            y = robot.y + t * dy
            if 0.0 <= y <= WINDOW_HEIGHT:
                t_candidates.append(t)

    # Intersections with horizontal walls y = 0 and y = WINDOW_HEIGHT
    for y_wall in (0.0, float(WINDOW_HEIGHT)):
        t = (y_wall - robot.y) / dy
        if t > 0:
            x = robot.x + t * dx
            if 0.0 <= x <= WINDOW_WIDTH:
                t_candidates.append(t)

    if not t_candidates:
        return 1.0

    distance = min(t_candidates)
    norm_distance = min(distance / MAX_SENSOR_DISTANCE, 1.0)
    return norm_distance

