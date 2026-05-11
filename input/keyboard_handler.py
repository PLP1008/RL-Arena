"""
input/keyboard_handler.py
──────────────────────────
Translates pygame key events → action int (same interface as an RL agent).
Player 1 uses arrow keys; Player 2 uses WASD.
"""

import pygame
from config import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT


# Key bindings per human player slot
_BINDINGS = [
    # Player 1: arrow keys
    {
        pygame.K_UP:    ACTION_UP,
        pygame.K_DOWN:  ACTION_DOWN,
        pygame.K_LEFT:  ACTION_LEFT,
        pygame.K_RIGHT: ACTION_RIGHT,
    },
    # Player 2: WASD
    {
        pygame.K_w: ACTION_UP,
        pygame.K_s: ACTION_DOWN,
        pygame.K_a: ACTION_LEFT,
        pygame.K_d: ACTION_RIGHT,
    },
]


class KeyboardHandler:
    """
    Reads the current keyboard state and returns an action int.
    Call get_action() once per tick — it never blocks.
    """

    def __init__(self, player_slot: int = 0):
        """
        player_slot : 0 = arrow keys (P1), 1 = WASD (P2)
        """
        self.bindings = _BINDINGS[player_slot % len(_BINDINGS)]
        self._last_action = ACTION_RIGHT   # sensible default

    def get_action(self) -> int:
        """
        Returns the action int for the currently held key.
        If no directional key is pressed, repeats the last action
        (player keeps moving in the same direction).
        """
        keys = pygame.key.get_pressed()
        for key, action in self.bindings.items():
            if keys[key]:
                self._last_action = action
                return action
        return self._last_action
