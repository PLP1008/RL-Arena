"""
game/game_state.py
──────────────────
The single source of truth for game logic.

KEY DESIGN RULE: zero pygame imports here.
This module can be instantiated in a headless RL training loop
with no display, no clock, no surface — just pure logic.

Public interface:
    gs = GameState()
    obs, rewards, dones, info = gs.step(actions)   # actions: List[int]
    gs.reset()
"""

import numpy as np
from typing import List, Tuple, Dict

from config import (
    GRID_H, GRID_W, NUM_PLAYERS, START_TERRITORY_RADIUS,
    DIRECTION_MAP,
    REWARD_TILE_CAPTURED, REWARD_KILL_ENEMY, REWARD_DEATH, REWARD_STEP,
    ACTION_RIGHT,
)
from game.grid import GameGrid
from game.player import Player
from game.collision import CollisionSystem
from game.territory import claim_territory


class GameState:
    def __init__(self, num_players: int = NUM_PLAYERS):
        self.num_players = num_players
        self.grid = GameGrid()
        self.players: List[Player] = []
        self.collision_system = CollisionSystem()
        self.tick = 0
        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> Tuple:
        """Reset to a fresh game. Returns initial (obs, rewards, dones, info)."""
        self._reset_state()
        return self._build_step_result(
            rewards={p.player_id: 0.0 for p in self.players},
            kill_map={p.player_id: [] for p in self.players},
        )

    def step(self, actions: List[int]) -> Tuple:
        """
        Advance the game by one tick.

        actions : list of action ints, one per player (dead players ignored).
                  Indexes match player order (actions[0] → player 1, etc.)

        Returns : (obs, rewards, dones, info)
          obs     : List[np.ndarray]  — one (2R+1, 2R+1, 4) array per player
          rewards : List[float]       — one scalar per player
          dones   : List[bool]        — True if that player is dead
          info    : dict              — {"scores": [...], "tick": int}
        """
        self.tick += 1
        rewards = {p.player_id: REWARD_STEP for p in self.players}

        # 1. Apply actions → update directions
        self._apply_actions(actions)

        # 2. Compute next positions (don't move yet)
        next_positions = {}

        for p in self.players:
            if not p.alive:
                continue

            r, c = p.next_pos()

            # If next move hits wall,
            # stay in place until player changes direction
            if not self.grid.in_bounds(r, c):
                next_positions[p.player_id] = None
                continue
            

            next_positions[p.player_id] = (r, c)

        # 3. Collision detection
        kill_map = self.collision_system.evaluate(self.grid, self.players, next_positions)

        # 4. Kill enemies whose trails were crossed (reward the cutter)
        self._apply_kills(kill_map, rewards)

        # 5. Move surviving players + write trails
        for p in self.players:
            if not p.alive:
                continue
            if kill_map[p.player_id]:
                p.kill()

                # Remove all territory + trails
                self.grid.cells[np.abs(self.grid.cells) == p.player_id] = 0

                # Refresh scores
                for player in self.players:
                    player.score = self.grid.count_territory(player.player_id)

                rewards[p.player_id] += REWARD_DEATH
                continue

            next_pos = next_positions[p.player_id]

            # blocked by wall → stay still
            if next_pos is None:
                continue

            r, c = next_pos
            p.pos = [r, c]

            was_in_territory = p.in_territory
            now_in_territory = self.grid.is_territory_of(r, c, p.player_id)

            if not now_in_territory:
                # player left home — start/continue trailing
                self.grid.set(r, c, -p.player_id)
                p.in_territory = False
            else:
                p.in_territory = True
                if not was_in_territory:
                    # just returned home — flood fill!
                    new_tiles = claim_territory(self.grid, p.player_id)
                    p.score = self.grid.count_territory(p.player_id)
                    rewards[p.player_id] += new_tiles * REWARD_TILE_CAPTURED

        return self._build_step_result(rewards, kill_map)

    # ── Observation building ──────────────────────────────────────────────────

    def get_obs(self, player_id: int) -> np.ndarray:
        """
        Local crop centered on the player, shape (2R+1, 2R+1, N_OBS_CHANNELS).

        Channel 0: cells that are THIS player's territory  (1 or 0)
        Channel 1: cells that are THIS player's trail      (1 or 0)
        Channel 2: cells that are ANY enemy's territory    (1 or 0)
        Channel 3: cells that are ANY enemy's trail        (1 or 0)

        Out-of-bounds cells are treated as walls (all channels 0).
        """
        from env.obs_builder import build_local_obs
        p = self._get_player(player_id)
        return build_local_obs(self.grid.cells, player_id, p.row, p.col)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_state(self):
        self.grid.reset()
        self.tick = 0
        self.players = []

        # Space out starting positions in a grid pattern
        starts = _compute_start_positions(self.num_players)
        for i in range(self.num_players):
            pid = i + 1
            r, c = starts[i]
            p = Player(player_id=pid, pos=[r, c])
            self.grid.place_start_territory(pid, r, c, START_TERRITORY_RADIUS)
            p.score = self.grid.count_territory(pid)
            self.players.append(p)

    def _apply_actions(self, actions: List[int]):
        for i, p in enumerate(self.players):
            if p.alive and i < len(actions):
                p.apply_action(actions[i])

    def _apply_kills(self, kill_map: Dict, rewards: Dict):
        """
        If player B steps onto player A's trail, B dies and A gets kill reward.
        The collision system already tagged B with reason "enemy_trail";
        we just need to look up which player owns the trail cell that B
        walked into, and credit that player.
        """
        from game.collision import get_trail_owner

        for p in self.players:
            if not p.alive:
                continue
            for reason in kill_map[p.player_id]:
                if reason == "enemy_trail":
                    # p is the one who died by stepping on someone's trail
                    next_pos = p.next_pos()

                    # ignore wall-blocked movement
                    if not self.grid.in_bounds(*next_pos):
                        continue

                    r, c = next_pos
                    trail_owner_id = get_trail_owner(self.grid, r, c)
                    if trail_owner_id and trail_owner_id != p.player_id:
                        rewards[trail_owner_id] = rewards.get(trail_owner_id, 0.0) + REWARD_KILL_ENEMY

    def _build_step_result(self, rewards: Dict, kill_map: Dict) -> Tuple:
        obs     = [self.get_obs(p.player_id) for p in self.players]
        rew_list = [rewards.get(p.player_id, 0.0) for p in self.players]
        dones   = [not p.alive for p in self.players]
        info    = {
            "scores": [p.score for p in self.players],
            "tick": self.tick,
            "alive": [p.alive for p in self.players],
        }
        return obs, rew_list, dones, info

    def _get_player(self, player_id: int) -> Player:
        return next(p for p in self.players if p.player_id == player_id)

    @property
    def all_dead(self) -> bool:
        return all(not p.alive for p in self.players)

    @property
    def game_over(self) -> bool:
        alive = [p for p in self.players if p.alive]
        return len(alive) <= 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_start_positions(n: int):
    """Place n players symmetrically on the grid."""
    offsets = {
        1: [(GRID_H // 2,     GRID_W // 2)],
        2: [(GRID_H // 4,     GRID_W // 2),     (3 * GRID_H // 4, GRID_W // 2)],
        3: [(GRID_H // 4,     GRID_W // 4),     (GRID_H // 4,     3 * GRID_W // 4),
            (3 * GRID_H // 4, GRID_W // 2)],
        4: [(GRID_H // 4,     GRID_W // 4),     (GRID_H // 4,     3 * GRID_W // 4),
            (3 * GRID_H // 4, GRID_W // 4),     (3 * GRID_H // 4, 3 * GRID_W // 4)],
    }
    return offsets.get(n, offsets[4][:n])
