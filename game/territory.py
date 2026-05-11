"""
game/territory.py
─────────────────
Territory claiming via BFS flood fill.

Algorithm:
  When a player returns home (steps onto their own territory while trailing),
  flood-fill from ALL border cells of the grid. Any cell NOT reachable from
  the border that is not the player's own territory gets claimed.
  The player's trail is then converted to territory.

This correctly handles concave shapes and avoids claiming cells that are
"outside" the enclosed region.
"""

from collections import deque
from config import GRID_H, GRID_W
from game.grid import GameGrid


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def claim_territory(grid: GameGrid, player_id: int) -> int:
    """
    Call this when player_id has just returned to their own territory.

    1. BFS from the grid border — marks every cell reachable from outside.
    2. Any un-reached cell that isn't already this player's territory → claim it.
    3. Convert the player's trail to territory.

    Returns: number of new tiles claimed (for reward calculation).
    """
    reachable = _flood_from_border(grid, player_id)

    new_tiles = 0
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) not in reachable:
                current = grid.get(r, c)

                # don't overwrite enemy territory
                if current <= 0:
                    grid.set(r, c, player_id)
                    new_tiles += 1

    # trail becomes territory
    grid.convert_trail_to_territory(player_id)
    return new_tiles


def _flood_from_border(grid: GameGrid, player_id: int) -> set:
    """
    BFS starting from every border cell that is not this player's trail.
    Trail cells act as walls — the flood cannot cross them,
    which is exactly what "closes off" the enclosed area.
    """
    visited = set()
    queue = deque()

    # Seed from all 4 edges
    for r in range(GRID_H):
        for c in [0, GRID_W - 1]:
            _try_seed(grid, player_id, r, c, visited, queue)
    for c in range(GRID_W):
        for r in [0, GRID_H - 1]:
            _try_seed(grid, player_id, r, c, visited, queue)

    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                _try_seed(grid, player_id, nr, nc, visited, queue)

    return visited


def _try_seed(grid, player_id, r, c, visited, queue):
    if (r, c) in visited:
        return

    current = grid.get(r, c)

    # Own territory and own trail are walls
    if abs(current) == player_id:
        return

    visited.add((r, c))
    queue.append((r, c))
