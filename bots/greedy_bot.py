"""
bots/greedy_bot.py
──────────────────
Heuristic bot: prefers actions that move toward unclaimed territory.
Avoids stepping on own trail (suicidal). Falls back to random on ties.

Useful as a stronger baseline than RandomBot when testing RL agents.
"""

import random
import numpy as np
from config import (
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    N_ACTIONS, DIRECTION_MAP,
)


class GreedyBot:
    """
    Reads the local obs array (2R+1, 2R+1, 4) and picks the action
    that moves toward the most empty/enemy territory while avoiding
    immediate death from own trail.

    Channel layout (matches obs_builder):
        0 = my territory
        1 = my trail       ← avoid stepping here
        2 = enemy territory
        3 = enemy trail    ← avoid stepping here
    """

    def __init__(self):
        self._last_action = ACTION_RIGHT

    def act(self, obs: np.ndarray) -> int:
        if obs is None:
            return self._last_action

        center = obs.shape[0] // 2   # radius index

        scores = {}
        for action, (dr, dc) in DIRECTION_MAP.items():
            nr = center + dr
            nc = center + dc

            # out of local view → treat as wall (bad)
            if not (0 <= nr < obs.shape[0] and 0 <= nc < obs.shape[1]):
                scores[action] = -999
                continue

            my_trail    = obs[nr, nc, 1]
            enemy_trail = obs[nr, nc, 3]
            my_territory = obs[nr, nc, 0]

            if my_trail > 0 or enemy_trail > 0:
                scores[action] = -10      # likely death
            elif my_territory > 0:
                scores[action] = 0        # safe but no gain
            else:
                scores[action] = 1        # unclaimed — go here

        best_score = max(scores.values())
        best_actions = [a for a, s in scores.items() if s == best_score]
        chosen = random.choice(best_actions)
        self._last_action = chosen
        return chosen
