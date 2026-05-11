"""
game/grid.py
────────────
The raw game board — a single numpy int8 array.

Cell encoding (matches config.py):
  0        → empty
  +player_id → that player's territory
  -player_id → that player's active trail
"""

import numpy as np
from config import GRID_H, GRID_W, EMPTY


class GameGrid:
    def __init__(self):
        # shape: (H, W), dtype int8
        self.cells = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    # ── Basic accessors ───────────────────────────────────────────────────────

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < GRID_H and 0 <= c < GRID_W

    def get(self, r: int, c: int) -> int:
        return int(self.cells[r, c])

    def set(self, r: int, c: int, value: int):
        self.cells[r, c] = value

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_territory_of(self, r: int, c: int, player_id: int) -> bool:
        return self.cells[r, c] == player_id

    def is_trail_of(self, r: int, c: int, player_id: int) -> bool:
        return self.cells[r, c] == -player_id

    def is_any_trail(self, r: int, c: int) -> bool:
        return self.cells[r, c] < 0

    def is_empty(self, r: int, c: int) -> bool:
        return self.cells[r, c] == EMPTY

    # ── Territory ops ─────────────────────────────────────────────────────────

    def place_start_territory(self, player_id: int, center_r: int, center_c: int, radius: int):
        """Fill a square of territory around a starting position."""
        r0 = max(0, center_r - radius)
        r1 = min(GRID_H, center_r + radius + 1)
        c0 = max(0, center_c - radius)
        c1 = min(GRID_W, center_c + radius + 1)
        self.cells[r0:r1, c0:c1] = player_id

    def clear_trail(self, player_id: int):
        """Remove all trail cells belonging to this player."""
        self.cells[self.cells == -player_id] = EMPTY

    def convert_trail_to_territory(self, player_id: int):
        """Turn all trail cells into territory (called after flood fill)."""
        self.cells[self.cells == -player_id] = player_id

    def count_territory(self, player_id: int) -> int:
        return int(np.sum(self.cells == player_id))

    # ── Copy for RL rollouts ──────────────────────────────────────────────────

    def copy(self) -> "GameGrid":
        g = GameGrid()
        g.cells = self.cells.copy()
        return g

    def reset(self):
        self.cells[:] = EMPTY
