"""
main.py
───────
Human-playable entry point.

Controls:
  Player 1 — Arrow keys
  Player 2 — WASD
  Players 3 & 4 — GreedyBot

Run:  python main.py
"""

import sys
import pygame

from config import SCREEN_W, SCREEN_H, HUD_H, FPS, NUM_PLAYERS
from game.game_state import GameState
from rendering.renderer import Renderer
from input.keyboard_handler import KeyboardHandler
from bots.greedy_bot import GreedyBot


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H + HUD_H))
    pygame.display.set_caption("Paper.io")
    clock = pygame.time.Clock()

    # Input sources — one per player slot
    # Slots 0 & 1: human keyboards; slots 2+ : bots
    human_handlers = [KeyboardHandler(0), KeyboardHandler(1)]
    bots = [GreedyBot() for _ in range(max(0, NUM_PLAYERS - 2))]

    renderer = Renderer(num_players=NUM_PLAYERS)
    game_state = GameState(num_players=NUM_PLAYERS)
    obs_list, _, _, _ = game_state.reset()

    running = True
    while running:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs_list, _, _, _ = game_state.reset()   # R to restart

        if game_state.game_over:
            _show_game_over(screen, game_state)
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # ── Collect actions ────────────────────────────────────────────────────
        actions = []
        for i in range(NUM_PLAYERS):
            if i < len(human_handlers):
                actions.append(human_handlers[i].get_action())
            else:
                bot_idx = i - len(human_handlers)
                actions.append(bots[bot_idx].act(obs_list[i]))

        # ── Step ──────────────────────────────────────────────────────────────
        obs_list, rewards, dones, info = game_state.step(actions)

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill((20, 20, 28))
        renderer.draw(screen, game_state)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def _show_game_over(screen, game_state):
    font = pygame.font.SysFont("monospace", 28, bold=True)
    alive = [p for p in game_state.players if p.alive]
    if alive:
        msg = f"Player {alive[0].player_id} wins!  (R to restart)"
    else:
        msg = "Draw!  (R to restart)"
    surf = font.render(msg, True, (255, 255, 255))
    screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2,
                        SCREEN_H // 2 + HUD_H - surf.get_height() // 2))


if __name__ == "__main__":
    main()
