"""
game/collision.py
─────────────────
Stateless collision checks. All methods are pure functions of the grid + players.
GameState calls these; nothing here mutates state.
"""

from config import GRID_H, GRID_W
from game.grid import GameGrid
from game.player import Player
from typing import List


def check_wall(r: int, c: int) -> bool:
    """True if (r, c) is outside the grid bounds."""
    return not (0 <= r < GRID_H and 0 <= c < GRID_W)


def check_own_trail(grid: GameGrid, r: int, c: int, player_id: int) -> bool:
    """True if the player steps onto their own trail — always fatal."""
    return grid.is_trail_of(r, c, player_id)


def check_enemy_trail(grid: GameGrid, r: int, c: int, player_id: int) -> bool:
    """True if the player steps onto any enemy trail cell."""
    v = grid.get(r, c)
    return v < 0 and v != -player_id


def get_trail_owner(grid: GameGrid, r: int, c: int) -> int:
    """Return the player_id whose trail occupies (r, c), or 0 if none."""
    v = grid.get(r, c)
    return -v if v < 0 else 0


def check_head_collision(players: List[Player]) -> List[int]:
    """
    Detect simultaneous head-on collisions between alive players.
    Returns list of player_ids that die from head collision this tick.
    Both players die if they swap cells or land on the same cell.
    """
    dead = []
    positions = {}
    for p in players:
        if not p.alive:
            continue
        key = (p.row, p.col)
        if key in positions:
            # two players on same cell — both die
            dead.append(p.player_id)
            dead.append(positions[key])
        else:
            positions[key] = p.player_id
    return dead


class CollisionSystem:
    """
    Evaluates all collision rules for a tick.
    Returns a dict of player_id → list of kill reasons (for reward shaping).
    """

    def evaluate(
        self,
        grid: GameGrid,
        players: List[Player],
        next_positions: dict,   # {player_id: (r, c)}
    ) -> dict:
        """
        next_positions: proposed new positions for all alive players this tick.
        Returns: {player_id: [reason_str, ...]}  — reasons non-empty means death.

        Reasons: "wall", "own_trail", "enemy_trail", "head_collision"
        """
        kill_map = {p.player_id: [] for p in players}

        # 1. Wall + trail collisions
        for p in players:
            if not p.alive:
                continue
            next_pos = next_positions[p.player_id]

            # player blocked by wall this tick
            if next_pos is None:
                continue

            r, c = next_pos

            # if check_wall(r, c):
            #     kill_map[p.player_id].append("wall")
            #     continue                         # no further checks needed

            if check_own_trail(grid, r, c, p.player_id):
                kill_map[p.player_id].append("own_trail")

            if check_enemy_trail(grid, r, c, p.player_id):
                owner = get_trail_owner(grid, r, c)

                if owner and owner != p.player_id:
                    kill_map[owner].append("enemy_trail")

        # 2. Head-on collisions (after filtering already-dead)
        alive_next = {
            pid: pos
            for pid, pos in next_positions.items()
            if pos is not None and not kill_map[pid]
        }
        head_dead = _find_head_collisions(alive_next)
        for pid in head_dead:
            kill_map[pid].append("head_collision")

        return kill_map


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_head_collisions(next_positions: dict) -> List[int]:
    """Both players die if they land on the same cell."""
    dead = []
    seen = {}
    for pid, pos in next_positions.items():
        if pos in seen:
            dead.append(pid)
            dead.append(seen[pos])
        else:
            seen[pos] = pid
    return dead
