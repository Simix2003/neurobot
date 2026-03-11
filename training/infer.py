from __future__ import annotations

import argparse
import os

import numpy as np
import pygame
import torch

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS
from world.environment import Environment
from render.renderer import Renderer
from agents.sensors import compute_sensor_readings
from training.policy import PolicyNetwork
from training.io import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained NeuroBot policy in the simulator.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved policy checkpoint (.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy = PolicyNetwork()
    load_policy(policy, args.checkpoint, map_location="cpu")
    policy.eval()

    pygame.init()
    pygame.display.set_caption("NeuroBot - Trained Policy Inference")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)
    env.control_mode = "manual"  # actions will come from the trained policy
    renderer = Renderer(screen, env)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Build observation from current sensors
        sensors = compute_sensor_readings(env.robot, env.food)
        env.sensors = sensors

        obs_np = np.asarray(
            [
                sensors.food_distance,
                sensors.food_angle,
                sensors.wall_front,
                sensors.wall_left,
                sensors.wall_right,
            ],
            dtype=np.float32,
        )
        obs_t = torch.from_numpy(obs_np)

        with torch.no_grad():
            action_t, _ = policy.act(obs_t, deterministic=True)

        action_np = action_t.cpu().numpy()
        forward = float(np.clip(action_np[0], -1.0, 1.0))
        turn = float(np.clip(action_np[1], -1.0, 1.0))

        env.handle_manual_controls(forward, turn)
        env.update(dt)
        renderer.render_frame()

    pygame.quit()


if __name__ == "__main__":
    main()

