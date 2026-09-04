"""Entity definitions and AI state machines.

Each entity follows its spec from entities.json.  Behavior flags in the
JSON data dict drive which AI branches are active — no hardcoded per-ID
branching where avoidable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from item_ids import REGISTRY


@dataclass
class Entity:
    entity_id: str
    x: int
    y: int
    hp: int
    state: int = 0              # 0=idle, 1=telegraph, 2=attacking
    telegraph_ticks: int = 0
    alive: bool = True
    data: dict = field(default_factory=dict)

    # Void proximity tracker
    _void_proximity_ticks: int = 0
    # Wander timer
    _wander_ticks: int = 0
    # Creeper explosion flag
    exploded: bool = False

    @classmethod
    def from_id(cls, entity_id: str, x: int, y: int) -> "Entity":
        spec = REGISTRY.get_entity(entity_id) or {}
        return cls(
            entity_id=entity_id,
            x=x,
            y=y,
            hp=int(spec.get("hp", 1)),
            data=spec,
        )

    # ------------------------------------------------------------------
    # Item drop
    # ------------------------------------------------------------------
    def drop_item_id(self) -> str:
        """Return loot item ID on death, or '' if no drop."""
        drop = self.data.get("drop")
        return drop if drop else ""

    # ------------------------------------------------------------------
    # Main AI tick
    # ------------------------------------------------------------------
    def tick_ai(self, agent_x: int, agent_y: int, world=None) -> int:
        """Advance entity AI.  Returns damage dealt to agent this tick (0 if none)."""
        if not self.alive:
            return 0

        dist = abs(self.x - agent_x) + abs(self.y - agent_y)
        entity_type = self.data.get("type", "hostile")

        # ---- Passive entities ----------------------------------------
        if entity_type == "passive":
            self._wander(world)
            return 0

        # ---- Neutral entities (villager, guardian) -------------------
        if entity_type == "neutral":
            self._wander(world)
            return 0

        # ---- Special: Ocean Shark ------------------------------------
        if entity_type == "ocean" or self.data.get("ocean_spawn"):
            if dist <= 1:
                return 9999
            # Move towards agent aggressively
            self._wander(world, flee_from=(self.x + (self.x - agent_x), self.y + (self.y - agent_y)))
            return 0

        # ---- Special: Void -------------------------------------------
        if self.data.get("unkillable"):
            return self._tick_void(agent_x, agent_y, dist)

        # ---- Special: Void Echo (no telegraph, instant damage) -------
        if self.entity_id == "E020" and dist <= 1:
            self.state = 2
            return int(self.data.get("damage", 5))

        # ---- Hostile AI ----------------------------------------------
        telegraph_dur = int(self.data.get("telegraph", 2))

        if dist > 2:
            # Out of range — wander
            self.state = 0
            self.telegraph_ticks = 0
            self._wander(world)
            return 0

        # In range — attack cycle
        if self.state == 0:
            self.state = 1
            self.telegraph_ticks = 0
            return 0

        if self.state == 1:
            self.telegraph_ticks += 1
            if self.telegraph_ticks >= telegraph_dur:
                self.state = 2
                # Creeper: explode instead of normal attack
                if self.data.get("explodes") and dist <= 1:
                    self.exploded = True
                    self.alive = False
                    return int(self.data.get("damage", 15))
            return 0

        # state == 2: deal damage
        damage = int(self.data.get("damage", 1))
        self.state = 1
        self.telegraph_ticks = 0
        return damage

    # ------------------------------------------------------------------
    # Void proximity logic
    # ------------------------------------------------------------------
    def _tick_void(self, agent_x: int, agent_y: int, dist: int) -> int:
        prox_range = int(self.data.get("proximity_range", 3))
        prox_ticks = int(self.data.get("proximity_ticks", 5))

        if dist <= prox_range:
            self._void_proximity_ticks += 1
            if self._void_proximity_ticks >= prox_ticks:
                # Instant kill — signal via large damage value
                return 9999
        else:
            self._void_proximity_ticks = max(0, self._void_proximity_ticks - 1)
        return 0

    # ------------------------------------------------------------------
    # Wander / flee movement
    # ------------------------------------------------------------------
    def _wander(self, world, flee_from: tuple[int, int] | None = None):
        """Move randomly (wander) or away from a point (flee)."""
        self._wander_ticks += 1
        wander_interval = 5  # tunable: move every N ticks
        if self._wander_ticks < wander_interval:
            return

        self._wander_ticks = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        if flee_from is not None:
            fx, fy = flee_from
            # Sort directions: prefer those that increase distance from flee_from
            directions.sort(
                key=lambda d: -(abs(self.x + d[0] - fx) + abs(self.y + d[1] - fy))
            )

        random.shuffle(directions[:2])  # some randomness even when fleeing

        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if world is None or _passable(world, nx, ny):
                self.x, self.y = nx, ny
                return

    def flee(self, from_x: int, from_y: int, world=None):
        """Flee away from a point (passive entity behavior when attacked)."""
        self._wander(world, flee_from=(from_x, from_y))

    # ------------------------------------------------------------------
    # Take damage & split mechanics
    # ------------------------------------------------------------------
    def take_damage(self, amount: int, attacker_pos: tuple[int, int] | None = None, world=None):
        if self.data.get("unkillable"):
            return  # Shark/Void cannot be killed
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
            # Check for split behavior (e.g. Magma Cube E026)
            if self.data.get("splits_on_death") and world is not None:
                world.spawn_entity("E027", self.x + 1, self.y)
                world.spawn_entity("E027", self.x - 1, self.y)
        elif self.data.get("flees") and attacker_pos:
            # Passive entities flee when hit
            self._wander(world, flee_from=attacker_pos)


def _passable(world, x: int, y: int) -> bool:
    """Simple passability check for entity movement."""
    try:
        tile = world._tile(x, y)
        return tile.tile_type not in (1, 7)  # not a wall or gate
    except Exception:
        return True

