"""ATS reinforcement-learning configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
INVENTORY_SLOTS = 7  # 7 slots, each stores (item_id, count) in state

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

# Inventory management  (slot select = 10..16)
ACT_SELECT_SLOT_BASE = 10  # 10 + slot_index (0..6) → actions 10..16
ACT_USE_ITEM = 17
ACT_OPEN_INVENTORY = 18
ACT_CLOSE_INVENTORY = 19

# Crafting
ACT_OPEN_CRAFTING = 20
ACT_ADD_TO_CRAFTING = 21  # adds selected-slot item to crafting queue
ACT_CLOSE_CRAFTING = 22

# Utility
ACT_SLEEP = 23
ACT_WAIT = 24

ACTION_SIZE = 25  # total action indices 0..24

# ---------------------------------------------------------------------------
# State vector sizing & Embeddings
# ---------------------------------------------------------------------------
ITEM_EMBED_DIM    = 32
ENTITY_EMBED_DIM  = 32
ITEM_VOCAB_SIZE   = 100
ENTITY_VOCAB_SIZE = 40

# Vitals (5) + position/Z (5) + adjacent tiles (4) + hostile (4) + object (2)
#   + inventory 7*(item_id, count) = 14
#   + status effects (8) + environment (9) + memory features (3)
#   + action mask (25)
#   = 5+5+4+4+2+14+8+9+3+25 = 79
STATE_SIZE = 79

# Hazard & Ocean constants
OCEAN_SHARK_TICKS = 3     # ticks before lethal shark arrives
LAVA_DAMAGE_PER_TICK = 5  # HP lost per tick on TILE_LAVA (not safe ash path)
ACID_POISON_TICKS = 10    # POISONED duration from TILE_ACID

# ---------------------------------------------------------------------------
# Growing mind / policy network
# ---------------------------------------------------------------------------
MIND_HIDDEN_SIZE = 256                 # starting hidden width (increased from 128)
MIND_GROWTH_ENABLED = False            # OFF until the fixed-width policy demonstrably learns
MIND_GROWTH_CHECK_EVERY = 500          # ticks between growth checks
MIND_GROWTH_THRESHOLD = 0.02           # reward plateau delta that triggers growth
MIND_MAX_HIDDEN_SIZE = 1024            # eventual upper bound (not a hard ceiling)

# ---------------------------------------------------------------------------
# Training hyperparameters (PPO)
# ---------------------------------------------------------------------------
EPISODES = 20
MAX_TICKS = 9000
LEARNING_RATE = 3e-4                   # PPO default; 1e-4 was very slow for a from-scratch policy
GAMMA = 0.99
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.005                  # exploration bonus; 0.01 was strong enough to fight
                                      # convergence on small action spaces (see issues log #3)

# Continual Learning
# ---------------------------------------------------------------------------
# A PPO update now consumes a full fresh rollout and then clears the buffer,
# so BUFFER_SIZE and UPDATE_EVERY are kept equal — every transition is trained
# on exactly once (times MINI_EPOCHS), never stale.
BATCH_SIZE = 32                        # minimum transitions before an update is allowed
CONTINUAL_BUFFER_SIZE = 256            # rollout length
CONTINUAL_UPDATE_EVERY = 256           # ticks between online PPO updates (== rollout length)
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
