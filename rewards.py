"""Reward table — single source of truth for all reward signals.

All reward values are tunable constants at the top of this file.
No reward logic lives anywhere else.

Reward rules can be toggled on/off via the ``rules`` dict — the GUI
exposes keyboard shortcuts so the user can flip them at runtime.
"""

from item_ids import REGISTRY


# ---------------------------------------------------------------------------
# Reward constants (all tunable)
# ---------------------------------------------------------------------------

# Pickups
R_PICKUP_DISCOVERY = 1.5    # first time picking up any item type
R_PICKUP_CONSUMABLE = 1.0   # consumable already discovered
R_PICKUP_DEFAULT = 0.5      # any other item

# Kills — indexed by entity ID
_KILL_REWARDS = {
    "E001": 2.0,   # Weak Slime
    "E002": 3.0,   # Zombie
    "E003": 3.5,   # Skeleton
    "E004": 4.0,   # Creeper
    "E005": 2.5,   # Spider
    "E006": 3.0,   # Husk
    "E007": 2.5,   # Fire Imp
    "E008": 3.0,   # Cave Spider
    "E009": 10.0,  # Golem (boss tier progression validator)
    "E010": 3.5,   # Witch
    "E011": 2.5,   # Wolf
    "E012": 4.0,   # Yeti
    "E013": 0.5,   # Bat (passive)
    "E015": 1.0,   # Sheep (passive)
    "E016": 1.0,   # Pig (passive)
    "E017": 1.0,   # Chicken (passive)
    "E020": 5.0,   # Void Echo (rare)
    "E023": 8.0,   # Lava Golem
    "E025": 6.0,   # Cave Bat Queen
}
R_KILL_DEFAULT = 2.0

# Exploration & Island Progression
R_ISLAND_DISCOVERY   = 50.0  # first time discovering a new island
R_TILE_EXPLORE       = 0.2   # per-tile exploration (first visit only)
R_NEW_BIOME          = 3.0
R_RUINS_FOUND        = 2.0
R_VILLAGE_FOUND      = 3.0
R_CAPABILITY_UNLOCKED = 5.0
R_GATE_UNLOCKED      = 10.0
R_BOSS_DEFEATED      = 25.0
R_CHECKPOINT_REACHED = 5.0

R_KILL_TIER_SCALE = [1.0, 1.5, 2.0, 3.0, 5.0]  # multiplier by island tier (0..4)

# Crafting
R_CRAFT_DISCOVERY = 1.5     # first time crafting any item type

# Progression & Vitals
R_LEVEL_UP = 1.0
R_SLEEP = 2.0

# Penalties
R_WAIT = -0.2               # wait penalty
R_WALL_HIT = -0.3           # wall hit
R_DAMAGE_PER_HIT = -1.0     # damage penalty
R_HUNGER_DRAIN = -0.5       # per starvation tick
R_LEG_INJURY = -2.0
R_DEATH = -10.0             # Normal death penalty
R_OCEAN_DEATH = -500.0      # Falling into ocean / shark kill penalty
R_STANDING_STILL = -0.05    # scales up per tick while standing still

# Standing still threshold — ticks before penalty kicks in
STANDING_STILL_THRESHOLD = 5

# Survival bonus
R_SURVIVE_EPISODE = 10.0


# ---------------------------------------------------------------------------
# Default toggleable rules
# ---------------------------------------------------------------------------
def default_reward_rules() -> dict[str, bool]:
    """Return a fresh dict of toggleable reward rules (all on by default)."""
    return {
        "standing_still":  True,  # penalise not moving for too long
        "damage_penalty":  True,  # penalise taking damage
        "wait_penalty":    True,  # penalise choosing the wait action
        "hunger_penalty":  True,  # penalise starvation drain
        "wall_hit":        True,  # penalise walking into walls
        "death_penalty":   True,  # big penalty on death
        "pickup_reward":   True,  # reward for picking up items
        "kill_reward":     True,  # reward for killing entities
        "explore_reward":  True,  # reward for discovering biomes / ruins
        "progression":     True,  # reward for capabilities, gates, bosses
    }


class RewardEngine:
    """Core reward management, dynamic peak-relative classification, and progression efficiency."""

    # High watermark and low trough tracking shared across episodes in session
    best_score_ever: float | None = None
    worst_score_ever: float | None = None

    def __init__(self):
        self.total = 0.0
        self.tick_reward = 0.0
        self.progression_points = 0.0
        self.damage_taken = 0.0
        self.damage_dealt = 0.0
        self.deaths = 0

        self.discovered_items: set[str] = set()
        self.discovered_islands: set[int] = set()
        self.discovered_tiles: set[tuple[int, int]] = set()
        self.discovered_biomes: set[int] = set()
        self.biome_visits: dict[int, int] = {}
        self.zone_visits: dict[str, int] = {}
        self.crafted_items: set[str] = set()
        self.capabilities_unlocked: set[str] = set()
        self.gates_unlocked: set[str] = set()
        self.checkpoints_reached: set[str] = set()

        self.rules: dict[str, bool] = default_reward_rules()
        # Standing-still tracker
        self._still_ticks = 0
        self._last_pos: tuple[int, int] | None = None

    @classmethod
    def reset_history(cls):
        """Reset historical peak/trough watermarks."""
        cls.best_score_ever = None
        cls.worst_score_ever = None

    @classmethod
    def classify_score(cls, score: float) -> str:
        """Classify score relative to personal historical watermark:
        
        - GOOD: Strictly exceeds all-time best score (> best_score_ever).
        - MID: Non-negative score that matches/stays below previous peak (0 <= score <= best_score_ever).
        - BAD: Negative score that reaches a new low or worsens (score <= worst_score_ever < 0).
        - FAIR: Negative score that is an improvement over previous negative low (worst_score_ever < score < 0).
        """
        if score >= 0:
            if cls.best_score_ever is None or score > cls.best_score_ever:
                cls.best_score_ever = score
                return "GOOD"
            else:
                return "MID"
        else:
            # score < 0
            if cls.worst_score_ever is None or score <= cls.worst_score_ever:
                cls.worst_score_ever = score
                return "BAD"
            else:
                return "FAIR"

    def reset_tick(self):
        self.tick_reward = 0.0

    def add(self, value: float, reason: str = "") -> float:
        self.tick_reward += value
        self.total += value
        return value

    def compute_progression_efficiency(self, ticks: int = 1) -> float:
        """Compute Progression Efficiency = meaningful_progression / cost."""
        cost = self.damage_taken + (self.deaths * 10.0) + (max(ticks, 1) * 0.01) + 1.0
        progression = self.progression_points + max(0.0, self.total * 0.1)
        return float(progression / cost)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def on_new_tile(self, x: int, y: int) -> float:
        """Per-tile exploration reward (first visit only)."""
        if not self.rules.get("explore_reward", True):
            return 0.0
        pos = (x, y)
        if pos not in self.discovered_tiles:
            self.discovered_tiles.add(pos)
            self.progression_points += 0.02
            return self.add(R_TILE_EXPLORE, "tile_explore")
        return 0.0

    def on_island_discovery(self, island_id: int) -> float:
        """First arrival on a new island (excluding starting home island 0 and ocean -1)."""
        if not self.rules.get("explore_reward", True):
            return 0.0
        if island_id <= 0:
            return 0.0
        if island_id not in self.discovered_islands:
            self.discovered_islands.add(island_id)
            self.progression_points += 5.0
            return self.add(R_ISLAND_DISCOVERY, f"island_discover_{island_id}")
        return 0.0

    def on_ocean_death(self) -> float:
        """Killed by ocean / shark."""
        self.deaths += 1
        if not self.rules.get("death_penalty", True):
            return 0.0
        return self.add(R_OCEAN_DEATH, "ocean_shark_death")

    def on_pickup(self, item_id: str) -> float:
        if not self.rules.get("pickup_reward", True):
            return 0.0
        item = REGISTRY.get_item(item_id)
        if item_id not in self.discovered_items:
            self.discovered_items.add(item_id)
            self.progression_points += 0.5
            return self.add(R_PICKUP_DISCOVERY, "pickup_discovery")
        if item and item.get("type") == "consumable":
            return self.add(R_PICKUP_CONSUMABLE, "pickup_consumable")
        return self.add(R_PICKUP_DEFAULT, "pickup_default")

    def on_kill(self, entity_id: str, island_tier: int = 0) -> float:
        if not self.rules.get("kill_reward", True):
            return 0.0
        base_reward = _KILL_REWARDS.get(entity_id, R_KILL_DEFAULT)
        tier_mult = R_KILL_TIER_SCALE[min(max(island_tier, 0), len(R_KILL_TIER_SCALE) - 1)]
        reward = base_reward * tier_mult
        self.progression_points += 1.0 * tier_mult
        return self.add(reward, f"kill_tier{island_tier}")

    def on_craft(self, item_id: str) -> float:
        if item_id not in self.crafted_items:
            self.crafted_items.add(item_id)
            self.progression_points += 1.5
            return self.add(R_CRAFT_DISCOVERY, "craft_discovery")
        return 0.0

    def on_capability_unlocked(self, capability: str) -> float:
        """Reward discovering an emergent capability in the Minecraft-style chain."""
        if not self.rules.get("progression", True):
            return 0.0
        if capability not in self.capabilities_unlocked:
            self.capabilities_unlocked.add(capability)
            self.progression_points += 3.0
            return self.add(R_CAPABILITY_UNLOCKED, f"capability_{capability}")
        return 0.0

    def on_gate_unlocked(self, gate_id: str) -> float:
        """Reward unlocking a Dark Souls-style spatial progression gate."""
        if not self.rules.get("progression", True):
            return 0.0
        if gate_id not in self.gates_unlocked:
            self.gates_unlocked.add(gate_id)
            self.progression_points += 5.0
            return self.add(R_GATE_UNLOCKED, f"gate_{gate_id}")
        return 0.0

    def on_boss_defeated(self, boss_id: str) -> float:
        """Reward defeating a major progression validator boss."""
        if not self.rules.get("progression", True):
            return 0.0
        self.progression_points += 10.0
        return self.add(R_BOSS_DEFEATED, f"boss_{boss_id}")

    def on_checkpoint_reached(self, checkpoint_id: str) -> float:
        if not self.rules.get("progression", True):
            return 0.0
        if checkpoint_id not in self.checkpoints_reached:
            self.checkpoints_reached.add(checkpoint_id)
            self.progression_points += 2.0
            return self.add(R_CHECKPOINT_REACHED, f"checkpoint_{checkpoint_id}")
        return 0.0

    def on_level_up(self) -> float:
        self.progression_points += 0.5
        return self.add(R_LEVEL_UP, "level_up")

    def on_sleep(self) -> float:
        return self.add(R_SLEEP, "sleep")

    def on_biome_enter(self, biome_id: int) -> float:
        """Biome discovery with diminishing returns to prevent wandering exploits."""
        if not self.rules.get("explore_reward", True):
            return 0.0
        visits = self.biome_visits.get(biome_id, 0)
        self.biome_visits[biome_id] = visits + 1

        if visits == 0:
            self.discovered_biomes.add(biome_id)
            self.progression_points += 1.0
            return self.add(R_NEW_BIOME * 1.0, "new_biome_first")
        elif visits == 1:
            return self.add(R_NEW_BIOME * 0.4, "new_biome_second")
        elif visits == 2:
            return self.add(R_NEW_BIOME * 0.1, "new_biome_third")
        return 0.0  # Diminishing returns cap at 0 for subsequent re-entries

    def on_ruins_found(self) -> float:
        if not self.rules.get("explore_reward", True):
            return 0.0
        self.progression_points += 1.0
        return self.add(R_RUINS_FOUND, "ruins_found")

    def on_village_found(self) -> float:
        if not self.rules.get("explore_reward", True):
            return 0.0
        self.progression_points += 2.0
        return self.add(R_VILLAGE_FOUND, "village_found")

    def on_wait(self) -> float:
        if not self.rules.get("wait_penalty", True):
            return 0.0
        return self.add(R_WAIT, "wait")

    def on_damage(self, amount: float = 1.0) -> float:
        self.damage_taken += amount
        if not self.rules.get("damage_penalty", True):
            return 0.0
        return self.add(R_DAMAGE_PER_HIT, "damage")

    def on_wall_hit(self) -> float:
        if not self.rules.get("wall_hit", True):
            return 0.0
        return self.add(R_WALL_HIT, "wall_hit")

    def on_leg_injury(self) -> float:
        return self.add(R_LEG_INJURY, "leg_injury")

    def on_death(self) -> float:
        self.deaths += 1
        if not self.rules.get("death_penalty", True):
            return 0.0
        return self.add(R_DEATH, "death")

    def on_survive_bonus(self) -> float:
        self.progression_points += 5.0
        return self.add(R_SURVIVE_EPISODE, "survive_bonus")

    # ------------------------------------------------------------------
    # Standing-still detection — call once per tick from env
    # ------------------------------------------------------------------
    def update_standing_still(self, x: int, y: int):
        """Track agent position. If it hasn't moved for STANDING_STILL_THRESHOLD
        ticks, apply a penalty each additional tick."""
        if not self.rules.get("standing_still", True):
            self._still_ticks = 0
            self._last_pos = (x, y)
            return

        if self._last_pos == (x, y):
            self._still_ticks += 1
        else:
            self._still_ticks = 0
            self._last_pos = (x, y)

        if self._still_ticks >= STANDING_STILL_THRESHOLD:
            multiplier = min(self._still_ticks - STANDING_STILL_THRESHOLD + 1, 4)
            self.add(R_STANDING_STILL * multiplier, "standing_still")


