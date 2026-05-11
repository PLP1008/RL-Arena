"""
bots/random_bot.py
──────────────────
Simplest possible bot: picks a random action every tick.
Useful as a sanity-check opponent during development.
"""

import random
from config import N_ACTIONS


class RandomBot:
    """Same interface as an RL agent: act(obs) → int."""

    def act(self, obs=None) -> int:
        return random.randint(0, N_ACTIONS - 1)
