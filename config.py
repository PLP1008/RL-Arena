# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_W          = 60
GRID_H          = 60
CELL_SIZE       = 10          # pixels per cell (rendering only)

# ── Game ──────────────────────────────────────────────────────────────────────
NUM_PLAYERS     = 4
FPS             = 15          # human-playable speed; set to 0 for RL headless

# ── Players ───────────────────────────────────────────────────────────────────
# (R, G, B) — one per player slot
PLAYER_COLORS   = [
    (66,  133, 244),   # P1 blue
    (234,  67,  53),   # P2 red
    (52,  168,  83),   # P3 green
    (251, 188,   4),   # P4 yellow
]
START_TERRITORY_RADIUS = 3    # each player starts owning a small square

# ── Actions ───────────────────────────────────────────────────────────────────
# 0=UP  1=DOWN  2=LEFT  3=RIGHT
ACTION_UP       = 0
ACTION_DOWN     = 1
ACTION_LEFT     = 2
ACTION_RIGHT    = 3
N_ACTIONS       = 4

# direction vectors indexed by action int
DIRECTION_MAP   = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}

# ── Grid cell values ──────────────────────────────────────────────────────────
# territory of player i  →  +i   (i = 1..NUM_PLAYERS)
# trail of player i      →  -i
EMPTY           = 0

# ── Observation ───────────────────────────────────────────────────────────────
OBS_RADIUS      = 15          # local crop: (2*R+1) × (2*R+1)
# channels: 0=my_territory  1=my_trail  2=enemy_territory  3=enemy_trail
N_OBS_CHANNELS  = 4

# ── Rewards ───────────────────────────────────────────────────────────────────
REWARD_TILE_CAPTURED  =  0.01
REWARD_KILL_ENEMY     =  1.0
REWARD_DEATH          = -1.0
REWARD_STEP           = -0.001   # small time penalty to discourage idling

# ── Rendering ─────────────────────────────────────────────────────────────────
SCREEN_W        = GRID_W * CELL_SIZE
SCREEN_H        = GRID_H * CELL_SIZE
HUD_H           = 40           # pixels reserved for score bar at top
TRAIL_ALPHA     = 180          # 0-255; trails slightly transparent
