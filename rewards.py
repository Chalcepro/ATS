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
    "E009": 6.0,   # Golem (boss tier)
    "E010": 3.5,   # Witch
    "E011": 2.5,   # Wolf
    "E012": 4.0,   # Yeti
    "E013": 0.5,   # Bat (passive)
    "E015": 1.0,   # Sheep (passive)
    "E016": 1.0,   # Pig (passive)
    "E017": 1.0,   # Chicken (passive)
    "E020": 5.0,   # Void Echo (rare)
}
R_KILL_DEFAULT = 2.0

# Exploration
R_NEW_BIOME = 2.0
R_RUINS_FOUND = 1.5
R_VILLAGE_FOUND = 3.0

# Crafting
R_CRAFT_DISCOVERY = 1.0     # first time crafting any item type

# Progression
R_LEVEL_UP = 1.0
R_SLEEP = 2.0

# Penalties
R_WAIT = -0.2               # harsher wait penalty
R_WALL_HIT = -0.3           # harsher wall hit
R_DAMAGE_PER_HIT = -1.0     # doubled — damage should hurt
R_HUNGER_DRAIN = -0.5       # per starvation tick
R_LEG_INJURY = -2.0
R_DEATH = -10.0             # Death penalty (lowered from -50 so per-tick rewards aren't drowned out)
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
    }


class RewardEngine:
    def __init__(self):
        self.total = 0.0
        self.tick_reward = 0.0
        self.discovered_items: set[str] = set()
        self.discovered_biomes: set[int] = set()
        self.crafted_items: set[str] = set()
        self.rules: dict[str, bool] = default_reward_rules()
        # Standing-still tracker
        self._still_ticks = 0
        self._last_pos: tuple[int, int] | None = None

    def reset_tick(self):
        self.tick_reward = 0.0

    def add(self, value: float, reason: str = "") -> float:
        self.tick_reward += value
        self.total += value
        return value

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def on_pickup(self, item_id: str) -> float:
        if not self.rules.get("pickup_reward", True):
            return 0.0
        item = REGISTRY.get_item(item_id)
        if item_id not in self.discovered_items:
            self.discovered_items.add(item_id)
            return self.add(R_PICKUP_DISCOVERY)
        if item and item.get("type") == "consumable":
            return self.add(R_PICKUP_CONSUMABLE)
        return self.add(R_PICKUP_DEFAULT)

    def on_kill(self, entity_id: str) -> float:
        if not self.rules.get("kill_reward", True):
            return 0.0
        reward = _KILL_REWARDS.get(entity_id, R_KILL_DEFAULT)
        return self.add(reward)

    def on_craft(self, item_id: str) -> float:
        if item_id not in self.crafted_items:
            self.crafted_items.add(item_id)
        return self.add(R_CRAFT_DISCOVERY)

    def on_level_up(self) -> float:
        return self.add(R_LEVEL_UP)

    def on_sleep(self) -> float:
        return self.add(R_SLEEP)

    def on_biome_enter(self, biome_id: int) -> float:
        if biome_id in self.discovered_biomes:
            return 0.0
        if not self.rules.get("explore_reward", True):
            return 0.0
        self.discovered_biomes.add(biome_id)
        return self.add(R_NEW_BIOME)

    def on_ruins_found(self) -> float:
        if not self.rules.get("explore_reward", True):
            return 0.0
        return self.add(R_RUINS_FOUND)

    def on_village_found(self) -> float:
        if not self.rules.get("explore_reward", True):
            return 0.0
        return self.add(R_VILLAGE_FOUND)

    def on_wait(self) -> float:
        if not self.rules.get("wait_penalty", True):
            return 0.0
        return self.add(R_WAIT)

    def on_damage(self) -> float:
        if not self.rules.get("damage_penalty", True):
            return 0.0
        return self.add(R_DAMAGE_PER_HIT)

    def on_wall_hit(self) -> float:
        if not self.rules.get("wall_hit", True):
            return 0.0
        return self.add(R_WALL_HIT)

    def on_leg_injury(self) -> float:
        return self.add(R_LEG_INJURY)

    def on_death(self) -> float:
        if not self.rules.get("death_penalty", True):
            return 0.0
        return self.add(R_DEATH)

    def on_survive_bonus(self) -> float:
        return self.add(R_SURVIVE_EPISODE)

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
            # Penalty scales linearly up to a max cap (-0.20 per tick)
            multiplier = min(self._still_ticks - STANDING_STILL_THRESHOLD + 1, 4)
            self.add(R_STANDING_STILL * multiplier, "standing_still")

