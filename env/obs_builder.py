"""
env/obs_builder.py
──────────────────
Standalone observation encoder — converts the raw grid into the
observation tensor that RL agents consume.

Factored out of game_state.py so you can swap observation representations
(local crop vs. full grid, different channel layouts, etc.) without
touching game logic.

Current encoding: local crop (Option A)
  Shape:    (2*OBS_RADIUS+1, 2*OBS_RADIUS+1, N_OBS_CHANNELS)
  Channels:
    0 = my territory  (binary)
    1 = my trail       (binary)
    2 = enemy territory (binary)
    3 = enemy trail    (binary)
  Out-of-bounds → all zeros (treated as wall).
"""

import numpy as np
from config import OBS_RADIUS, N_OBS_CHANNELS, GRID_H, GRID_W


def build_local_obs(grid_cells: np.ndarray, player_id: int,
                    player_row: int, player_col: int) -> np.ndarray:
    """
    Extract a local observation crop centered on the player.

    Parameters
    ----------
    grid_cells : np.ndarray, shape (H, W), dtype int8
        The raw grid (positive = territory, negative = trail, 0 = empty).
    player_id : int
        1-indexed player id.
    player_row, player_col : int
        Current position of the player.

    Returns
    -------
    np.ndarray, shape (2R+1, 2R+1, 4), dtype float32
    """
    R = OBS_RADIUS
    size = 2 * R + 1
    obs = np.zeros((size, size, N_OBS_CHANNELS), dtype=np.float32)

    # Compute the slice of the grid visible to this player
    # and the corresponding slice in the obs array
    grid_r0 = player_row - R
    grid_r1 = player_row + R + 1
    grid_c0 = player_col - R
    grid_c1 = player_col + R + 1

    # Clamp to grid bounds
    obs_r0 = max(0, -grid_r0)
    obs_c0 = max(0, -grid_c0)
    src_r0 = max(0, grid_r0)
    src_c0 = max(0, grid_c0)
    src_r1 = min(GRID_H, grid_r1)
    src_c1 = min(GRID_W, grid_c1)
    obs_r1 = obs_r0 + (src_r1 - src_r0)
    obs_c1 = obs_c0 + (src_c1 - src_c0)

    # Extract the visible patch in one go
    patch = grid_cells[src_r0:src_r1, src_c0:src_c1]

    obs[obs_r0:obs_r1, obs_c0:obs_c1, 0] = (patch == player_id).astype(np.float32)
    obs[obs_r0:obs_r1, obs_c0:obs_c1, 1] = (patch == -player_id).astype(np.float32)
    obs[obs_r0:obs_r1, obs_c0:obs_c1, 2] = ((patch > 0) & (patch != player_id)).astype(np.float32)
    obs[obs_r0:obs_r1, obs_c0:obs_c1, 3] = ((patch < 0) & (patch != -player_id)).astype(np.float32)

    return obs


def build_global_obs(grid_cells: np.ndarray, player_id: int) -> np.ndarray:
    """
    Full-grid observation (Option B). Larger input but easier to reason about.

    Shape: (GRID_H, GRID_W, N_OBS_CHANNELS), dtype float32
    """
    obs = np.zeros((GRID_H, GRID_W, N_OBS_CHANNELS), dtype=np.float32)
    obs[:, :, 0] = (grid_cells == player_id).astype(np.float32)
    obs[:, :, 1] = (grid_cells == -player_id).astype(np.float32)
    obs[:, :, 2] = ((grid_cells > 0) & (grid_cells != player_id)).astype(np.float32)
    obs[:, :, 3] = ((grid_cells < 0) & (grid_cells != -player_id)).astype(np.float32)
    return obs
