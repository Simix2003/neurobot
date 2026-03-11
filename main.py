import pygame

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS
from world.environment import Environment
from render.renderer import Renderer


def main() -> None:
    pygame.init()
    pygame.display.set_caption("NeuroBot - Simulation Sandbox (Step 1A)")

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)
    renderer = Renderer(screen, env)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Manual keyboard controls (Step 1B)
        keys = pygame.key.get_pressed()
        forward = 0.0
        turn = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            forward += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            forward -= 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            turn -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            turn += 1.0

        env.handle_manual_controls(forward, turn)
        env.update(dt)
        renderer.render_frame()

    pygame.quit()


if __name__ == "__main__":
    main()

