"""Reward table — single source of truth for all reward signals.

All reward values are tunable constants at the top of this file.
No reward logic lives anywhere else.

Reward rules can be toggled on/off via the ``rules`` dict — the GUI
exposes keyboard shortcuts so the user can flip them at runtime.
"""

import config_rl
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
# Swinging at empty air.
#
# Without this, ATTACK is a free WAIT that occasionally pays, and the agent
# finds that. Measured on the island with a brain that had cleared the whole
# curriculum: it chose attack in 85.8% of ticks, stood more or less still,
# and survived by not going anywhere - because nothing it was doing cost
# anything and one action in a hundred landed a hit.
#
# Smaller than R_WAIT on purpose. Swinging at nothing is a wasted tick, not a
# worse one than deliberately idling, and the curriculum uses the same -0.05
# for the same reason.
R_ATTACK_MISS = -0.05
R_WALL_HIT = -0.05          # wall hit. Was -0.3, which at 30 ticks/s is -9 a
                            # second for being pressed against a rock
R_DAMAGE_PER_HIT = -1.0     # damage penalty
R_HUNGER_DRAIN = -0.5       # per starvation tick
R_LEG_INJURY = -2.0
R_DEATH = -10.0             # Normal death penalty
# Ocean/shark penalty. Was -500.0 — but ats_env applies it on every shark
# trigger, including *survived* strikes, and a single -500 tick under gamma=0.99
# dominates the whole discounted return and (after advantage normalisation)
# flattens every other transition in the same rollout to ~zero. Kept clearly
# worse than a normal death, but on the same order of magnitude.
R_OCEAN_DEATH = -15.0       # Falling into ocean / shark strike penalty
R_STANDING_STILL = -0.05    # scales up per tick while standing still

# Standing still threshold — ticks before penalty kicks in
STANDING_STILL_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Staying alive
# ---------------------------------------------------------------------------
#
# THE BUG THIS FIXES. Until now, being alive cost money and dying was a
# one-off -10. Pressed against a wall the agent lost 0.3 (wall hit) + 0.2
# (standing still) every tick, so after roughly twenty ticks of being stuck,
# death was the cheaper option. The episode history bears it out: 724 of 726
# episodes ended in death, and the one episode that survived to max ticks
# scored -3281 against a survival bonus of +10.
#
# An agent that learns "end the episode early" under those numbers has learned
# correctly. The reward function was the bug, and no amount of tuning the
# learning rate or the entropy was ever going to find it.
#
# Two changes make living the better option and keep it that way:
#
#   1. A wage for being alive. Small, per tick, unconditional. It is what
#      makes the integral of a long episode positive instead of negative.
#   2. A budget for the nuisance penalties. Walking into a wall and standing
#      still are mistakes worth flagging, but they must not be able to
#      out-earn everything else simply by being repeatable. Once an episode
#      has spent its budget they stop accruing, so they can shape behaviour
#      early without dominating the return.
#
# Hunger, damage and death are deliberately *not* budgeted: those are real
# stakes and they are supposed to hurt.
R_ALIVE = 0.02              # per tick, just for still being here
NUISANCE_BUDGET = 25.0      # most an episode can lose to wall-hits + standing still

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
        # How much of this episode's nuisance budget has been spent
        self._nuisance_spent = 0.0
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

    def add_nuisance(self, value: float, reason: str = "") -> float:
        """A small, repeatable penalty - drawn from a per-episode budget.

        Anything charged every tick has to be capped, or its total is decided
        by how long the episode ran rather than by how badly the agent played.
        That is the arithmetic that made dying the rational move.
        """
        room = NUISANCE_BUDGET - self._nuisance_spent
        if room <= 0.0:
            return 0.0
        value = max(value, -room)
        self._nuisance_spent += -value
        return self.add(value, reason)

    def tick_alive(self) -> float:
        """The wage for existing. Called once per tick while the agent lives."""
        return self.add(R_ALIVE, "alive")

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
        return self.add_nuisance(R_WALL_HIT, "wall_hit")

    def on_leg_injury(self) -> float:
        return self.add(R_LEG_INJURY, "leg_injury")

    def on_death(self) -> float:
        self.deaths += 1
        if not self.rules.get("death_penalty", True):
            return 0.0
        return self.add(R_DEATH, "death")

    def on_survive_bonus(self, ticks: int | None = None) -> float:
        """Paid for reaching the end of an episode, scaled by how long it was.

        A flat bonus is a cliff. Under it, surviving a one-day episode and a
        ten-day episode both pay exactly +10, so once the cap is reached
        there is nothing left to want - and with an earned, growing cap
        (survival.py) the agent would be paid the same for clearing the
        easiest rung as the hardest.

        Scaled by days, it is worth more to reach the end of a longer
        episode, which is the thing being asked for. R_ALIVE already pays
        per tick and gives the gradient WITHIN an episode; this is what
        makes a longer episode worth wanting in the first place.

        `ticks` omitted keeps the old flat behaviour, so nothing that has
        not been updated changes its arithmetic.
        """
        self.progression_points += 5.0
        if ticks is None:
            return self.add(R_SURVIVE_EPISODE, "survive_bonus")
        days = max(1.0, float(ticks) / float(config_rl.DAY_LENGTH_TICKS))
        return self.add(R_SURVIVE_EPISODE * days, "survive_bonus")

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
            self.add_nuisance(R_STANDING_STILL * multiplier, "standing_still")




# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def audit(max_ticks: int = 9000, verbose: bool = True) -> bool:
    """Is staying alive worth more than dying? Check, do not assume.

    The full world ran 725 episodes against a reward function where the answer
    was no. Nothing in the training loop could have told you - the loss curves
    were fine, the entropy was fine, the agent was simply optimising what it
    had been asked to optimise.

    So this is the question asked directly: an agent that plays badly for a
    whole episode - stuck on walls, standing still, starving - must still come
    out ahead of one that dies on tick one.
    """
    worst_tick = R_WALL_HIT + (R_STANDING_STILL * 4)
    nuisance = max(worst_tick * max_ticks, -NUISANCE_BUDGET)
    starving = (R_HUNGER_DRAIN * max_ticks) / 30.0
    wage = R_ALIVE * max_ticks

    survive_badly = wage + nuisance + starving + R_SURVIVE_EPISODE
    die_at_once = R_DEATH

    ok = survive_badly > die_at_once
    if verbose:
        print("reward audit over %d ticks" % max_ticks)
        print("   wage for being alive      %+9.1f" % wage)
        print("   nuisance (budget %-5.1f)   %+9.1f" % (NUISANCE_BUDGET, nuisance))
        print("   starving the whole time   %+9.1f" % starving)
        print("   survival bonus            %+9.1f" % R_SURVIVE_EPISODE)
        print("   --------------------------------")
        print("   played badly, survived    %+9.1f" % survive_badly)
        print("   died immediately          %+9.1f" % die_at_once)
        print("   %s" % ("ok - living beats dying"
                         if ok else "BROKEN - the agent is paid to die"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if audit() else 1)
