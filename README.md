# paper.io — pygame + RL-ready environment

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Controls: P1 = arrow keys · P2 = WASD · P3/P4 = GreedyBot · R = restart

## Directory structure

```
paper_io/
├── main.py                  # human-playable entry point
├── config.py                # ALL tunable constants live here
│
├── game/                    # pure logic — zero pygame imports
│   ├── grid.py              # GameGrid — numpy array, territory helpers
│   ├── player.py            # Player dataclass
│   ├── collision.py         # CollisionSystem
│   ├── territory.py         # BFS flood fill → claim territory
│   └── game_state.py        # orchestrates everything; step() API
│
├── rendering/
│   └── renderer.py          # surfarray fast blit + HUD
│
├── input/
│   └── keyboard_handler.py  # keypress → action int
│
├── bots/
│   ├── random_bot.py        # random action baseline
│   └── greedy_bot.py        # heuristic: prefer unclaimed cells
│
├── env/
│   └── paper_io_env.py      # Gymnasium wrapper — plug into SB3
│
└── assets/
    ├── fonts/               # drop JetBrains Mono here for HUD
    └── sounds/              # capture.wav, death.wav (optional)
```

## RL training (when ready)

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

## Key design decisions

- `game/` has **zero pygame imports** — can run headless at 10k+ steps/sec
- All input goes through `step(actions: list[int])` — swapping human → agent is one line
- Rewards and obs shape are in `config.py` — easy to tune without touching game code
- Observation: local crop `(2*OBS_RADIUS+1, 2*OBS_RADIUS+1, 4)` centered on player
- Actions: 0=UP 1=DOWN 2=LEFT 3=RIGHT
