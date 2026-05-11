"""
rendering/renderer.py
─────────────────────
Reads GameState and draws to a pygame Surface.
Never imports from game/ except to read — no game logic here.

Rendering strategy:
  1. Build a (H, W, 3) uint8 color array from grid.cells in numpy.
  2. Blit it via pygame.surfarray.blit_array for O(1) draw calls.
  3. Draw player heads and HUD on top with regular pygame calls.
"""

import numpy as np
import pygame
from typing import List

from config import (
    GRID_H, GRID_W, CELL_SIZE, SCREEN_W, SCREEN_H,
    HUD_H, NUM_PLAYERS, PLAYER_COLORS,
)


# Pre-build color lookup: index → (R, G, B)
# index 0         → empty (dark background)
# index +player_id → territory color (full)
# index -player_id → trail color (dimmed)
def _build_color_table(num_players: int) -> dict:
    table = {0: (30, 30, 40)}          # empty cells
    for i, color in enumerate(PLAYER_COLORS[:num_players]):
        pid = i + 1
        r, g, b = color
        table[pid]  = (r, g, b)
        table[-pid] = (r // 2, g // 2, b // 2)    # trail = darker version
    return table


class Renderer:
    def __init__(self, num_players: int = NUM_PLAYERS):
        self.color_table = _build_color_table(num_players)

        # Offscreen surface sized to the grid (no HUD here)
        self.grid_surface = pygame.Surface((SCREEN_W, SCREEN_H))

        # Font for HUD
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 16, bold=True)

    # ── Main draw entry point ─────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface, game_state):
        """
        Call once per frame.
          screen     : the main pygame display surface
          game_state : GameState instance (read-only)
        """
        self._draw_grid(game_state.grid)
        screen.blit(self.grid_surface, (0, HUD_H))
        self._draw_player_heads(screen, game_state.players)
        self._draw_hud(screen, game_state.players)

    # ── Grid (fast surfarray path) ────────────────────────────────────────────

    def _draw_grid(self, grid):
        """
        Convert grid.cells (H×W int8) → (W×H×3) uint8 color array,
        then blit in one call. surfarray expects (W, H) column-major order.
        """
        cells = grid.cells                               # (H, W)
        color_arr = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

        for cell_val, rgb in self.color_table.items():
            mask = cells == cell_val
            color_arr[mask] = rgb

        # Scale: each cell → CELL_SIZE × CELL_SIZE pixels
        # Use np.repeat for a fast integer zoom
        scaled = np.repeat(np.repeat(color_arr, CELL_SIZE, axis=0), CELL_SIZE, axis=1)

        # surfarray wants (W, H, 3) — transpose axes 0 and 1
        pygame.surfarray.blit_array(self.grid_surface, scaled.transpose(1, 0, 2))

    # ── Player heads ─────────────────────────────────────────────────────────

    def _draw_player_heads(self, screen: pygame.Surface, players):
        for p in players:
            if not p.alive:
                continue
            color = PLAYER_COLORS[p.player_id - 1]
            px = p.col * CELL_SIZE + CELL_SIZE // 2
            py = p.row * CELL_SIZE + CELL_SIZE // 2 + HUD_H
            pygame.draw.circle(screen, color, (px, py), CELL_SIZE // 2 + 1)
            # white dot in center for clarity at small cell sizes
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, screen: pygame.Surface, players):
        pygame.draw.rect(screen, (20, 20, 28), (0, 0, SCREEN_W, HUD_H))
        slot_w = SCREEN_W // max(len(players), 1)

        for i, p in enumerate(players):
            color = PLAYER_COLORS[p.player_id - 1] if p.alive else (80, 80, 80)
            label = f"P{p.player_id}: {p.score}"
            surf = self.font.render(label, True, color)
            screen.blit(surf, (i * slot_w + 8, HUD_H // 2 - surf.get_height() // 2))
