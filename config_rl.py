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
ITEM_EMBED_DIM    = 64
ENTITY_EMBED_DIM  = 64
ITEM_VOCAB_SIZE   = 100
ENTITY_VOCAB_SIZE = 40
# 16, not 64: there are thirteen tile types. A 64-wide table for thirteen
# things is 51 columns of noise for the optimiser to push around, and tiles
# are the one vocabulary here that genuinely is small.
TILE_EMBED_DIM    = 16
TILE_VOCAB_SIZE   = 16     # world.py defines TILE_EMPTY..TILE_SPORE = 0..12

# Egocentric perception patch (added 2026-09-05).  Before this the agent saw
# only the 4 adjacent tiles plus a *distance* to the nearest object/hostile
# with NO direction — so "walk toward the food" was not a learnable function
# of the observation, and a straight line (which maximises new-tile
# exploration reward) was close to optimal.  See updates/2026-09-05b ... .
PATCH_RADIUS = 2                              # 2 -> 5x5 window centred on the agent
PATCH_SIZE   = PATCH_RADIUS * 2 + 1
PATCH_TILES  = PATCH_SIZE * PATCH_SIZE        # 25

# Vitals (5) + position/Z (5) + adjacent tiles (4) + hostile (4) + object (2)
#   + inventory 7*(item_id, count) = 14
#   + status effects (8) + environment (9) + memory features (3)   -> 54
#   + direction unit-vectors (4) + egocentric patch (25)
#   + action mask (25)
#   = 54+4+25+25 = 108
# NOTE: the new blocks are inserted *before* the action mask on purpose —
# indices 0..53 keep their historical meaning (mind/need_detector.py and
# ExperienceMemory's state[:20] slice depend on them) and the mask stays at
# the tail (sanity_env.py writes it as s[-ACTION_SIZE:]).
IDX_ENT_TYPE   = 15        # nearest entity type, RAW int -> entity_embed
IDX_OBJ_ID     = 19        # nearest object id, NORMALISED (id/ITEM_VOCAB_SIZE)
IDX_INV_START  = 20        # 7 slots x (norm_item_id, norm_count)
IDX_DIR_START  = 54        # obj_dx, obj_dy, hostile_dx, hostile_dy
IDX_PATCH_START = IDX_DIR_START + 4
IDX_MASK_START  = IDX_PATCH_START + PATCH_TILES
STATE_SIZE = IDX_MASK_START + ACTION_SIZE     # 108

# Hazard & Ocean constants
OCEAN_SHARK_TICKS = 3     # ticks before lethal shark arrives
LAVA_DAMAGE_PER_TICK = 5  # HP lost per tick on TILE_LAVA (not safe ash path)
ACID_POISON_TICKS = 10    # POISONED duration from TILE_ACID

# ---------------------------------------------------------------------------
# Growing mind / policy network
# ---------------------------------------------------------------------------
# The recalled-solution logit boost in mind/solution_loop.py.
#
# OFF, and this is the empirical evaluation the roadmap asked for. Measured
# 2026-09-19 through SolutionLoop - the path the GUI actually uses - two
# seeds, 700 nursery episodes each:
#
#   hint ON    detour 10.04   success 87%
#   hint off   detour  8.57   success 98%
#
# Eleven points of success rate, the same direction on both seeds. Detour is
# noise between the arms, so this is not the whole story, but the hint is
# not paying for itself.
#
# The cause is the need detector, not the boost. On a nursery state it
# returns [0, 0, 0, 1.0] - one need pinned at maximum - so the recall fires
# on EVERY tick rather than occasionally, and the same remembered action gets
# +3.0 on its logit over and over. Three logits is roughly twenty times the
# probability, which is not a hint, it is a decision.
#
# Turn it back on when the need detector reports something that varies.
MIND_HINT_ENABLED = False

MIND_HIDDEN_SIZE = 256                 # starting hidden width (increased from 128)
# Still OFF, and this is why nothing has ever appeared to change when it was
# switched on and off: RLPolicy.expand() exists and works, but this flag has
# gated it out of every run so far. The observation that the dynamic sizing
# does nothing is correct - it has never run.
MIND_GROWTH_ENABLED = False
MIND_GROWTH_CHECK_EVERY = 500          # ticks between growth checks
MIND_GROWTH_THRESHOLD = 0.02           # reward plateau delta that triggers growth
MIND_MAX_HIDDEN_SIZE = 1024            # eventual upper bound (not a hard ceiling)

# ---------------------------------------------------------------------------
# Training hyperparameters (PPO)
# ---------------------------------------------------------------------------
EPISODES = 700
MAX_TICKS = 4400
# 3e-4 (0.0003), not 1e-4. Measured 2026-09-18 on the fixed nursery, 3600
# episodes, same seed and same everything else:
#
#   1e-4   41,159 ticks   detour 2.90   entropy 0.5569   commits ~ep 2500
#   3e-4   33,341 ticks   detour 2.26   entropy 0.5723   commits ~ep 1400
#
# Both converge; 3e-4 does it in roughly half the episodes and ends better.
# Note the tick counts: the same episode count cost FEWER environment steps
# at the higher rate, because a policy that navigates finishes episodes
# sooner. Faster learning is cheaper here, not more expensive.
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95                      # advantage smoothing; raw n-step MC advantage was
                                      # too high-variance to give a stable policy gradient
                                      # (reward climbed then collapsed non-monotonically —
                                      # see updates/2026-09-04 first-long-run.md)
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.002                  # the floor, and safe to start there: the
                                       # adaptive controller below raises it 8% at a
                                       # time whenever entropy falls under target.
                                       # 0.01 was strong enough to fight
                                      # convergence on small action spaces (see issues log #3)

# Entropy-collapse guard (2026-09-05). A *flat* coefficient multiplies
# whatever the entropy gradient happens to be — once entropy has already
# decayed near zero, that gradient is tiny too, so a fixed coefficient
# can't pull a collapsed policy back out; it ratchets tighter instead.
# Confirmed directly (not guessed) over a 2000-episode real run: entropy
# decayed 0.0162 -> 0.0001 within a single episode with actor loss pinned
# at 0.0000 the whole time. See updates/2026-09-04e ... .
# Fix: adapt the coefficient itself — ramp it up while measured entropy is
# below ENTROPY_TARGET (so the push gets *stronger* exactly when the policy
# is at risk of collapsing, instead of staying flat), relax it back toward
# the floor once entropy is healthy again (so a well-converged policy isn't
# permanently forced to over-explore).
ENTROPY_TARGET = 0.5                   # nats; max possible for 25 actions is ln(25)=3.22
# Under a narrow action mask the 0.5-nat target above is meaningless: with
# four moves available the ceiling is ln(4)=1.39, a healthy policy sits near
# 1.3, and the guard never engages until the policy is already most of the way
# to deterministic. For masks this narrow the target becomes a fraction of the
# achievable ceiling instead. The full 25-action world is deliberately left on
# the flat 0.5 - changing that is a separate decision with its own evidence.
ENTROPY_NARROW_MASK = 10       # masks at or below this width use the fraction
ENTROPY_TARGET_FRAC = 0.45     # ...of ln(available actions)
ENTROPY_COEFF_MAX = 0.10               # ceiling = 20x ENTROPY_COEFF
ENTROPY_ADAPT_UP = 1.08                # multiplicative step when entropy < target
ENTROPY_ADAPT_DOWN = 0.98              # multiplicative step when entropy >= target
# NOTE: the floor is `ENTROPY_COEFF` itself (not a separate baked constant),
# on purpose — debug_gui.py's Hyperparameter Studio lets the user live-tune
# ENTROPY_COEFF at runtime; a separate ENTROPY_COEFF_MIN captured once at
# import time would silently stop tracking that control.

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

DAY_LENGTH_TICKS = 4000
# Where an episode starts in the day. 0.25 is dawn and 0.75 is dusk, so 0.30
# opens just after sunrise. Zero means midnight, which is what this was by
# accident for as long as the counter started at zero.
DAY_START_FRACTION = 0.30
PERCEPTION_RADIUS = 7
NIGHT_PERCEPTION_RADIUS = 4
TORCH_PERCEPTION_RADIUS = 6

STARTING_ROOM_SIZE = 3

MODEL_PATH = CHECKPOINT_DIR / "model_rl.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_best.pt"

# Bump this whenever the reward function or the return/advantage computation
# changes in a way that makes an old checkpoint's *critic* calibration invalid
# (even though its weights still load fine, structurally).  A checkpoint
# stamped with a different version is treated as incompatible for resume —
# see _load_policy() in main.py.  History: 1 = pre-2026-09-04 (broken
# return-normalisation, moved to checkpoints/_legacy_pre_2026-09-04_fix/);
# 2 = fixed return/critic handling + on-policy rollouts + GAE(lambda);
# 3 = STATE_SIZE 79->108 (directional perception + egocentric patch) and the
#     reworked embedding front-end — old checkpoints are structurally
#     incompatible (fc1/embed_proj shapes differ), not merely miscalibrated.
REWARD_SCHEME_VERSION = 3

# ---------------------------------------------------------------------------
# Logging / TensorBoard
# ---------------------------------------------------------------------------
# Path tracing. The per-tick CSV is one row per step - 700 episodes of a
# 200-step rung is 140k rows, which is a few megabytes and worth it while the
# question is "what is it actually doing". Turn it off for a long unattended
# run; the per-episode summary and the map are written either way.
TRACE_PER_TICK = True

LOG_DIR = ROOT / "logs"
LOG_INTERVAL = 10                      # print & log every N episodes

# ---------------------------------------------------------------------------
# Virus integration
# ---------------------------------------------------------------------------
VIRUS_ROOT = ROOT.parent / "Virus"
VIRUS_PROFILE = "ats"
VIRUS_EXPORT_PATH = VIRUS_ROOT / "data_ats" / "ats_runtime_corpus.txt"
