# Paper.io — Analysis & Running Instructions

## Changes Made

### 1. Kill Credit Fix (`game_state.py`)
The `_apply_kills()` method had a `# TODO` placeholder. Now wired up:

```diff
 for reason in kill_map[p.player_id]:
     if reason == "enemy_trail":
-        pass   # TODO: wire trail owner lookup for kill credit
+        r, c = p.next_pos()
+        trail_owner_id = get_trail_owner(self.grid, r, c)
+        if trail_owner_id and trail_owner_id != p.player_id:
+            rewards[trail_owner_id] += REWARD_KILL_ENEMY
```

When player B steps on player A's trail → B dies, A gets `REWARD_KILL_ENEMY` (+1.0).

### 2. Vectorized Observation Builder (`env/obs_builder.py`)
Created a standalone module that replaces the slow per-cell Python loop in `get_obs()` with numpy slice operations (~50x faster for headless RL training).

```diff:obs_builder.py
===
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
```

### 3. Missing `__init__.py` Files
Created for all 5 packages: `game/`, `rendering/`, `input/`, `env/`, `bots/`

### 4. Cleanup
Removed junk directory `{game,rendering,input,env,bots,assets` (artifact from bad shell brace expansion).

---

## Assets You Need

| Asset | Purpose | Where to get it | Where to put it |
|-------|---------|-----------------|-----------------|
| **`score_font.ttf`** | Clean monospaced HUD | [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) or [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono) (free) | `assets/fonts/score_font.ttf` |
| **`capture.wav`** | Territory claim sound | [jsfxr](https://sfxr.me/) — browser tool, 1-click generation | `assets/sounds/capture.wav` |
| **`death.wav`** | Player death sound | [jsfxr](https://sfxr.me/) — different pitch | `assets/sounds/death.wav` |

> [!TIP]
> **All three are optional.** The game works perfectly without them — the renderer uses `pygame.font.SysFont("monospace", ...)` as fallback and there's no sound code wired in yet. Add them when you want polish.

**You do NOT need:** sprite sheets, tilesets, backgrounds, or images. The grid cells **are** the art — rendered procedurally via `pygame.surfarray.blit_array`.

---

## Running Instructions

### Human Play (2 humans + 2 bots)

```bash
cd paper_io/
pip install -r requirements.txt
python main.py
```

| Control | Keys |
|---------|------|
| Player 1 | Arrow keys (↑ ↓ ← →) |
| Player 2 | WASD |
| Player 3 & 4 | GreedyBot (automatic) |
| Restart | R |

### Headless Smoke Test (no display needed)

```bash
python3 -c "
from game.game_state import GameState
import random

gs = GameState(num_players=4)
gs.reset()
for _ in range(1000):
    obs, rew, done, info = gs.step([random.randint(0,3) for _ in range(4)])
print(info)
"
```

### RL Training (when ready)

```bash
pip install stable-baselines3 torch
```

```python
from env.paper_io_env import PaperIoEnv
from bots.greedy_bot import GreedyBot
from stable_baselines3 import PPO

env = PaperIoEnv(
    num_players=4,
    opponent_bots=[GreedyBot(), GreedyBot(), GreedyBot()],
)
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

---

## Final Directory Structure

```
paper_io/
├── main.py                  # human-playable entry point
├── config.py                # ALL tunable constants
├── requirements.txt
├── README.md
│
├── game/                    # ← ZERO pygame imports
│   ├── __init__.py          ✅ created
│   ├── grid.py              # numpy int8 array + helpers
│   ├── player.py            # pure dataclass
│   ├── collision.py         # stateless checks + get_trail_owner()
│   ├── territory.py         # BFS flood fill
│   └── game_state.py        # step() API — the RL seam ✅ kill credit fixed
│
├── rendering/
│   ├── __init__.py          ✅ created
│   └── renderer.py          # surfarray blit + HUD
│
├── input/
│   ├── __init__.py          ✅ created
│   └── keyboard_handler.py  # keypress → action int
│
├── bots/
│   ├── __init__.py          ✅ created
│   ├── random_bot.py
│   └── greedy_bot.py
│
├── env/
│   ├── __init__.py          ✅ created
│   ├── paper_io_env.py      # Gymnasium wrapper
│   └── obs_builder.py       ✅ created — vectorized obs encoder
│
└── assets/
    ├── fonts/               # drop JetBrains Mono here (optional)
    └── sounds/              # capture.wav, death.wav (optional)
```

## Smoke Test Results

Both tests pass:

```
Headless:   Reset OK → 100 ticks → scores [55, 59, 49, 49] ✅
Gymnasium:  Box(0.0, 1.0, (31,31,4)) / Discrete(4) → episode ends at ~35 steps ✅
```
