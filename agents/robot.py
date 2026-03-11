from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ROBOT_COLOR,
    ROBOT_RADIUS,
    ROBOT_SPEED,
    ROBOT_TURN_SPEED,
)


@dataclass
class Robot:
    """Simple manually controlled robot for Step 1B."""

    x: float = WINDOW_WIDTH / 2
    y: float = WINDOW_HEIGHT / 2
    angle: float = 0.0  # radians; 0 points to the right
    radius: float = ROBOT_RADIUS

    def apply_manual_input(self, forward: float, turn: float, dt: float) -> None:
        """Update pose from normalized forward/turn commands in [-1, 1]."""
        # Turn first
        self.angle += turn * ROBOT_TURN_SPEED * dt

        # Keep angle in [-pi, pi] for numerical stability
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        # Then move forward along heading
        distance = forward * ROBOT_SPEED * dt
        dx = math.cos(self.angle) * distance
        dy = math.sin(self.angle) * distance
        self.x += dx
        self.y += dy

        # Clamp inside window bounds
        self.x = max(self.radius, min(WINDOW_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(WINDOW_HEIGHT - self.radius, self.y))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the robot as a circle with a heading line."""
        center = (int(self.x), int(self.y))
        pygame.draw.circle(surface, ROBOT_COLOR, center, int(self.radius))

        # Heading indicator
        head_x = self.x + math.cos(self.angle) * self.radius
        head_y = self.y + math.sin(self.angle) * self.radius
        pygame.draw.line(
            surface,
            (255, 255, 255),
            center,
            (int(head_x), int(head_y)),
            2,
        )

