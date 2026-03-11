from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

from agents import Brain, Robot
from agents.sensors import SensorReadings, compute_sensor_readings
from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WORLD_BORDER_COLOR,
    WORLD_BORDER_THICKNESS,
)
from world.food import Food, spawn_food_random


@dataclass
class Environment:
    """Minimal world container for Step 1A.

    For now it only represents the rectangular world area and exposes
    update/draw hooks that will host entities in later steps.
    """

    width: int = WINDOW_WIDTH
    height: int = WINDOW_HEIGHT
    robot: Robot = field(default_factory=Robot)
    food: Food = field(default_factory=spawn_food_random)
    score: int = 0

    brain: Brain = field(default_factory=Brain)
    control_mode: str = "brain"  # "brain" or "manual"

    sensors: SensorReadings | None = None

    _pending_forward: float = 0.0
    _pending_turn: float = 0.0

    def handle_manual_controls(self, forward: float, turn: float) -> None:
        """Store current frame's normalized control inputs."""
        self._pending_forward = max(-1.0, min(1.0, forward))
        self._pending_turn = max(-1.0, min(1.0, turn))

    def update(self, dt: float) -> None:
        """Advance world simulation by dt seconds."""
        if dt <= 0.0:
            return

        if self.control_mode == "brain":
            # Compute sensors and query brain for movement
            self.sensors = compute_sensor_readings(self.robot, self.food)
            inputs = np.array(
                [
                    self.sensors.food_distance,
                    self.sensors.food_angle,
                    self.sensors.wall_front,
                    self.sensors.wall_left,
                    self.sensors.wall_right,
                ],
                dtype=float,
            )
            forward, turn = self.brain.forward(inputs)
            self.robot.apply_manual_input(forward, turn, dt)
        else:
            # Manual mode (Step 1B behavior)
            self.robot.apply_manual_input(self._pending_forward, self._pending_turn, dt)

        # Reset manual controls so we do not accumulate stale input
        self._pending_forward = 0.0
        self._pending_turn = 0.0

        # Collision check between robot and food
        self._check_robot_food_collision()

    def _check_robot_food_collision(self) -> None:
        dx = self.robot.x - self.food.x
        dy = self.robot.y - self.food.y
        min_distance = self.robot.radius + self.food.radius
        if dx * dx + dy * dy <= min_distance * min_distance:
            self.score += 1
            self.food = spawn_food_random()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw world-level visuals (currently just the border)."""
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(
            surface,
            WORLD_BORDER_COLOR,
            rect,
            WORLD_BORDER_THICKNESS,
        )
        self.food.draw(surface)
        self.robot.draw(surface)

