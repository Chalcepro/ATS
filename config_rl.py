"""ATS reinforcement-learning configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
INVENTORY_SLOTS = 24  # 24 slots, each stores (item_id, count) in state

# ---------------------------------------------------------------------------
# Action space — context-masked (dynamic action window)
# ---------------------------------------------------------------------------
# Movement
ACT_MOVE_FORWARD = 0
ACT_MOVE_BACKWARD = 1
ACT_MOVE_LEFT = 2
ACT_MOVE_RIGHT = 3
ACT_SPRINT_ON = 4
ACT_SPRINT_OFF = 5
ACT_JUMP = 6

# Combat / interaction
ACT_ATTACK = 7
ACT_PICK_UP = 8
ACT_INTERACT = 9  # NPC, door, chest, pond, etc.

# Inventory management  (slot select = 10..33)
ACT_SELECT_SLOT_BASE = 10  # 10 + slot_index (0..23) → actions 10..33
ACT_USE_ITEM = 34
ACT_OPEN_INVENTORY = 35
ACT_CLOSE_INVENTORY = 36

# Crafting
ACT_OPEN_CRAFTING = 37
ACT_ADD_TO_CRAFTING = 38  # adds selected-slot item to crafting queue
ACT_CLOSE_CRAFTING = 39

# Utility
ACT_SLEEP = 40
ACT_WAIT = 41

ACTION_SIZE = 42  # total action indices 0..41

# ---------------------------------------------------------------------------
# State vector sizing
# ---------------------------------------------------------------------------
# Vitals (5) + position/Z (5) + adjacent tiles (4) + hostile (4) + object (2)
#   + inventory 24*(item_id, count) = 48
#   + status effects (6) + environment (7) + memory features (3)
#   + action mask (42)
#   = 5+5+4+4+2+48+6+7+3+42 = 126
STATE_SIZE = 126

# ---------------------------------------------------------------------------
# Growing mind / policy network
# ---------------------------------------------------------------------------
MIND_HIDDEN_SIZE = 256                 # starting hidden width (increased from 128)
MIND_GROWTH_CHECK_EVERY = 500          # ticks between growth checks
MIND_GROWTH_THRESHOLD = 0.02           # reward plateau delta that triggers growth
MIND_MAX_HIDDEN_SIZE = 1024            # eventual upper bound (not a hard ceiling)

# ---------------------------------------------------------------------------
# Training hyperparameters (PPO)
# ---------------------------------------------------------------------------
EPISODES = 20
MAX_TICKS = 9000
LEARNING_RATE = 1e-4
GAMMA = 0.99
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01                   # encourages active exploration vs premature collapse

# Continual Learning
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
CONTINUAL_BUFFER_SIZE = 256            # rolling transition buffer
CONTINUAL_UPDATE_EVERY = 64            # ticks between online PPO updates
CONTINUAL_MINI_EPOCHS = 4              # PPO epochs per online update

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SPEED_MULTIPLIER = 9.0
GUI_RENDER_EVERY = 1

DAY_LENGTH_TICKS = 900
PERCEPTION_RADIUS = 7
NIGHT_PERCEPTION_RADIUS = 4
TORCH_PERCEPTION_RADIUS = 6

STARTING_ROOM_SIZE = 3

MODEL_PATH = CHECKPOINT_DIR / "model_rl.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_best.pt"

# ---------------------------------------------------------------------------
# Logging / TensorBoard
# ---------------------------------------------------------------------------
LOG_DIR = ROOT / "logs"
LOG_INTERVAL = 10                      # print & log every N episodes

# ---------------------------------------------------------------------------
# Virus integration
# ---------------------------------------------------------------------------
VIRUS_ROOT = ROOT.parent / "Virus"
VIRUS_PROFILE = "ats"
VIRUS_EXPORT_PATH = VIRUS_ROOT / "data_ats" / "ats_runtime_corpus.txt"
