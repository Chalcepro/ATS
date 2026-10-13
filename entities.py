"""Creatures in the world and their AI.

Each creature follows its CreatureSpec (creature_spec.py, from
assets/entities.json). Its hostility kind picks the AI branch; nothing here
asks which creature it is by ID.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from creature_spec import CreatureSpec
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
    spec: CreatureSpec = None

    # Void proximity tracker
    _void_proximity_ticks: int = 0
    # Wander timer
    _wander_ticks: int = 0
    # Creeper explosion flag
    exploded: bool = False
    # A bridge guard holds its post: it never wanders off the bridge, only
    # fights what comes within reach. Set by World._lay_bridges.
    guard: bool = False

    @classmethod
    def from_id(cls, entity_id: str, x: int, y: int) -> "Entity":
        spec = REGISTRY.get_entity(entity_id)
        if spec is None:
            raise KeyError(f"no creature {entity_id} in entities.json")
        return cls(entity_id=entity_id, x=x, y=y, hp=spec.hp, spec=spec)

    # ------------------------------------------------------------------
    # Item drop
    # ------------------------------------------------------------------
    def drop_item_id(self) -> str:
        """Return loot item ID on death, or '' if no drop."""
        return self.spec.drop

    # ------------------------------------------------------------------
    # Main AI tick
    # ------------------------------------------------------------------
    def tick_ai(self, agent_x: int, agent_y: int, world=None) -> int:
        """Advance entity AI.  Returns damage dealt to agent this tick (0 if none)."""
        if not self.alive:
            return 0

        dist = abs(self.x - agent_x) + abs(self.y - agent_y)
        kind = self.spec.hostility

        # ---- Leaves the agent alone (fleeing happens in take_damage) --
        if kind in ("ignores", "flees_when_hit"):
            self._wander(world)
            return 0

        # ---- Hunts: closes in, bites (the shark) ---------------------
        if kind == "hunts":
            if dist <= 1 and _shark_can_reach(world, agent_x, agent_y):
                return 9999
            # Move towards agent aggressively
            self._wander(world, flee_from=(self.x + (self.x - agent_x), self.y + (self.y - agent_y)))
            return 0

        # ---- Lingering death: stay near too long and die (the Void) --
        if kind == "lingering_death":
            return self._tick_void(agent_x, agent_y, dist)

        # ---- Strikes instantly when adjacent, else defends its space -
        if kind == "strikes_instantly" and dist <= 1:
            self.state = 2
            return self.spec.damage

        # ---- Defends its space: attacks what comes within reach ------
        telegraph_dur = self.spec.telegraph

        if dist > self.spec.hostility_args["space"]:
            # Out of range — wander (a guard stays on its post)
            self.state = 0
            self.telegraph_ticks = 0
            if not self.guard:
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
                if kind == "explodes" and dist <= 1:
                    self.exploded = True
                    self.alive = False
                    return self.spec.damage
            return 0

        # state == 2: deal damage
        damage = self.spec.damage
        self.state = 1
        self.telegraph_ticks = 0
        return damage

    # ------------------------------------------------------------------
    # Void proximity logic
    # ------------------------------------------------------------------
    def _tick_void(self, agent_x: int, agent_y: int, dist: int) -> int:
        prox_range = self.spec.hostility_args["range"]
        prox_ticks = self.spec.hostility_args["ticks"]

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
        if self._wander_ticks < self.spec.move_every:
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

        sea_only = self.spec.has("ocean") and _sharks_stay_in_water()
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if world is None or _passable(world, nx, ny, sea_only):
                self.x, self.y = nx, ny
                return

    def flee(self, from_x: int, from_y: int, world=None):
        """Flee away from a point (passive entity behavior when attacked)."""
        self._wander(world, flee_from=(from_x, from_y))

    # ------------------------------------------------------------------
    # Take damage & split mechanics
    # ------------------------------------------------------------------
    def take_damage(self, amount: int, attacker_pos: tuple[int, int] | None = None, world=None):
        if self.spec.has("unkillable"):
            return
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False
            # What it splits into (Magma Cube). world.attack_at doesn't pass
            # `world`, so in the real game this never fires yet.
            if world is not None:
                for sid, dx, dy in self.spec.spawn_on_death:
                    world.spawn_entity(sid, self.x + dx, self.y + dy)
        elif self.spec.hostility == "flees_when_hit" and attacker_pos:
            # Passive entities flee when hit
            self._wander(world, flee_from=attacker_pos)


def _sharks_stay_in_water() -> bool:
    try:
        import config_rl
        return bool(getattr(config_rl, "SHARKS_STAY_IN_WATER", True))
    except Exception:
        return True


def _shark_can_reach(world, agent_x: int, agent_y: int) -> bool:
    """With SHARKS_BITE_ONLY_AT_SEA a shark bites only an agent that is in
    the ocean itself - not one on the beach or a bridge beside it."""
    try:
        import config_rl
        if not getattr(config_rl, "SHARKS_BITE_ONLY_AT_SEA", True):
            return True
        return world._tile(agent_x, agent_y).tile_type == 8   # TILE_OCEAN
    except Exception:
        return True


def _passable(world, x: int, y: int, sea_only: bool = False) -> bool:
    """Simple passability check for entity movement. `sea_only` is a shark:
    open ocean and nothing else - not the beach, not a bridge."""
    try:
        tile = world._tile(x, y)
        if sea_only:
            return tile.tile_type == 8           # TILE_OCEAN
        return tile.tile_type not in (1, 7)  # not a wall or gate
    except Exception:
        return True

