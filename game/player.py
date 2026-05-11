"""
game/player.py
──────────────
Pure data — no pygame, no grid writes.
GameState owns movement and collision; Player just holds state.
"""

from dataclasses import dataclass, field
from typing import Tuple
from config import DIRECTION_MAP, ACTION_RIGHT


@dataclass
class Player:
    player_id: int                       # 1-indexed (matches grid encoding)
    pos: list = field(default_factory=lambda: [0, 0])   # [row, col]
    direction: Tuple[int, int] = (0, 1)  # (dr, dc) — starts moving right
    alive: bool = True
    in_territory: bool = True            # False while trailing outside home
    score: int = 0                       # tiles owned (updated after flood fill)

    # ── Direction ─────────────────────────────────────────────────────────────

    def apply_action(self, action: int):
        """
        Convert action int → direction vector.
        Ignores 180° reversal (can't go directly backwards).
        """
        new_dir = DIRECTION_MAP[action]
        dr, dc = self.direction
        # block direct reversal
        if new_dir != (-dr, -dc):
            self.direction = new_dir

    def next_pos(self) -> Tuple[int, int]:
        """Returns (row, col) of the cell this player will step into next tick."""
        dr, dc = self.direction
        return self.pos[0] + dr, self.pos[1] + dc

    # ── Convenience ───────────────────────────────────────────────────────────

    def kill(self):
        self.alive = False

    @property
    def row(self) -> int:
        return self.pos[0]

    @property
    def col(self) -> int:
        return self.pos[1]

    def __repr__(self):
        status = "alive" if self.alive else "dead"
        return (f"Player(id={self.player_id}, pos={self.pos}, "
                f"dir={self.direction}, {status}, score={self.score})")
