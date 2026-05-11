"""
env/paper_io_env.py
────────────────────
Thin Gymnasium wrapper around GameState.

Designed for single-agent training (player 0 is the RL agent;
all other slots are filled by bots passed at construction time).

Usage:
    from env.paper_io_env import PaperIoEnv
    from bots.greedy_bot import GreedyBot

    env = PaperIoEnv(opponent_bots=[GreedyBot(), GreedyBot(), GreedyBot()])
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Drop-in compatible with Stable-Baselines3 / CleanRL.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional

from config import N_ACTIONS, OBS_RADIUS, N_OBS_CHANNELS, NUM_PLAYERS
from game.game_state import GameState
from bots.random_bot import RandomBot


class PaperIoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        num_players: int = NUM_PLAYERS,
        opponent_bots: Optional[List] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 2000,
    ):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Opponents for all player slots except 0
        if opponent_bots is None:
            opponent_bots = [RandomBot() for _ in range(num_players - 1)]
        self.opponent_bots = opponent_bots

        # ── Observation & action spaces ───────────────────────────────────────
        obs_size = 2 * OBS_RADIUS + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size, obs_size, N_OBS_CHANNELS),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # ── Internal state ────────────────────────────────────────────────────
        self._state: Optional[GameState] = None
        self._last_obs: Optional[List] = None
        self._step_count = 0
        self._renderer = None     # lazy-init only if render_mode=="human"

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = GameState(num_players=self.num_players)
        obs_list, _, _, info = self._state.reset()
        self._last_obs = obs_list
        self._step_count = 0
        return obs_list[0], info

    def step(self, action: int):
        assert self._state is not None, "Call reset() before step()."

        # Build action list: agent action + bot actions
        bot_actions = [
            bot.act(self._last_obs[i + 1])
            for i, bot in enumerate(self.opponent_bots)
        ]
        actions = [int(action)] + bot_actions

        obs_list, rewards, dones, info = self._state.step(actions)
        self._last_obs = obs_list
        self._step_count += 1

        agent_obs    = obs_list[0]
        agent_reward = rewards[0]
        terminated   = dones[0] or self._state.game_over
        truncated    = self._step_count >= self.max_steps

        return agent_obs, agent_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        if self._renderer is not None:
            import pygame
            pygame.quit()

    # ── Render helpers ────────────────────────────────────────────────────────

    def _render_rgb(self) -> np.ndarray:
        """Return (H, W, 3) RGB array — useful for video recording."""
        import pygame
        surface = self._get_surface()
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2)

    def _render_human(self):
        import pygame
        from rendering.renderer import Renderer
        from config import SCREEN_W, SCREEN_H, HUD_H

        if self._renderer is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H + HUD_H))
            pygame.display.set_caption("Paper.io — RL env")
            self._renderer = Renderer(num_players=self.num_players)
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._renderer.draw(self._screen, self._state)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _get_surface(self):
        import pygame
        from rendering.renderer import Renderer
        from config import SCREEN_W, SCREEN_H, HUD_H

        surface = pygame.Surface((SCREEN_W, SCREEN_H + HUD_H))
        renderer = Renderer(num_players=self.num_players)
        renderer.draw(surface, self._state)
        return surface
