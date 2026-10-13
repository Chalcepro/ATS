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

REST_AFTER = 300    # ticks awake before a rest
REST_TICKS = 20     # how long a rest lasts
FLEE_TICKS = 6      # how long a hit creature keeps running
HERD_GAP   = 3      # a herd animal further than this from its kind walks back


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
    # Its own life (W2): needs, what it's doing, and why
    hunger: int = 0
    tired: int = 0
    job: str = "wander"         # wander/graze/hunt/eat/flee/rest/fight/chase
    rest_left: int = 0
    flee_left: int = 0
    flee_from: tuple = None
    angry: bool = False         # the agent hit it; chases until far away
    # A charging shark (W2b): ticks left to wait, and the spot it locked onto
    wait_left: int = -1
    charge_to: tuple = None
    _bite_ticks: int = 0
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
        """Advance one tick.  Returns damage dealt to agent this tick (0 if none).

        Its own life - eating, fleeing, resting - is decided from the creatures
        the world says are near it (World.creatures_near).
        """
        if not self.alive:
            return 0

        dist = abs(self.x - agent_x) + abs(self.y - agent_y)
        kind = self.spec.hostility
        if self.spec.hunger_max:
            self.hunger += 1
        self.tired += 1

        # ---- Charges: surfaces, waits, locks on, charges (the shark) --
        if kind == "charges":
            return self._tick_charge(agent_x, agent_y, world)

        # ---- Hunts: closes in, bites ---------------------------------
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

        # ---- Fights the agent: within its space, or adjacent while it
        #      holds a grudge (the agent hit it first) ----------------------
        if self.angry and dist > self.spec.grudge:
            self.angry = False                  # the agent got far enough away
        space = self.spec.hostility_args.get("space", -1)
        if dist <= space or (self.angry and dist <= 1):
            self.job = "fight"
            return self._attack_cycle(kind, dist)
        self.state = 0
        self.telegraph_ticks = 0

        if self.angry:
            self.job = "chase"
            self._step(world, (agent_x, agent_y), away=False, every=self._run_every(3))
        elif not self.guard:                    # a guard holds its post
            self._live(world)
        return 0

    def _attack_cycle(self, kind: str, dist: int) -> int:
        """Telegraph, then strike (the Creeper explodes instead)."""
        if self.state == 0:
            self.state = 1
            self.telegraph_ticks = 0
            return 0

        if self.state == 1:
            self.telegraph_ticks += 1
            if self.telegraph_ticks >= self.spec.telegraph:
                self.state = 2
                # Creeper: explode instead of normal attack
                if kind == "explodes" and dist <= 1:
                    self.exploded = True
                    self.alive = False
                    return self.spec.damage
            return 0

        # state == 2: deal damage
        self.state = 1
        self.telegraph_ticks = 0
        return self.spec.damage

    # ------------------------------------------------------------------
    # Its own life: flee > eat > rest > wander, most urgent first
    # ------------------------------------------------------------------
    def _live(self, world) -> None:
        spec = self.spec
        others = (world.creatures_near(self.x, self.y, spec.sight)
                  if spec.sight and world is not None else ())
        if self.rest_left:                      # finishing a rest
            self.rest_left -= 1
            self.job = "rest"
            return

        threat = None
        if self.flee_left:
            self.flee_left -= 1
            threat = self.flee_from
        elif spec.fears:
            near = self._nearest(others, spec.fears)
            if near:
                threat = (near.x, near.y)
        if threat:
            self.job = "flee"
            self._step(world, threat, away=True, every=self._run_every(2))
            return

        if spec.hunger_max and self.hunger >= spec.hunger_max // 2:
            if "grass" in spec.diet:
                if _land(world, self.x, self.y):
                    self.job = "graze"
                    self.hunger = max(0, self.hunger - 20)
                    return
            else:
                prey = self._nearest(others, spec.diet)
                if prey:
                    self.job = "hunt"
                    if abs(prey.x - self.x) + abs(prey.y - self.y) <= 1:
                        self._bite(prey, world)
                    else:
                        self._step(world, (prey.x, prey.y), away=False, every=self._run_every(3))
                    return

        if self.tired >= REST_AFTER:
            self.tired = 0
            self.rest_left = REST_TICKS
            self.job = "rest"
            return

        self.job = "wander"
        if spec.herd:
            mate = self._nearest(others, None)
            if mate and abs(mate.x - self.x) + abs(mate.y - self.y) > HERD_GAP:
                self._step(world, (mate.x, mate.y), away=False, every=spec.move_every)
                return
        self._wander(world)

    def _bite(self, prey: "Entity", world) -> None:
        """One bite per telegraph+1 ticks; eating the kill ends its hunger."""
        self._bite_ticks += 1
        if self._bite_ticks <= self.spec.telegraph:
            return
        self._bite_ticks = 0
        prey.take_damage(self.spec.damage, (self.x, self.y), world, by_agent=False)
        if not prey.alive:
            self.hunger = 0
            self.job = "eat"

    def _nearest(self, others, tags) -> "Entity | None":
        """Nearest living creature in sight with any of `tags` (None = its
        own kind). Never itself, never its own kind as food."""
        best, best_d = None, self.spec.sight + 1
        for o in others:
            if o is self or not o.alive or o.spec.has("unkillable"):
                continue
            if tags is None:
                if o.entity_id != self.entity_id:
                    continue
            elif o.entity_id == self.entity_id or not (o.spec.tags & tags):
                continue
            d = abs(o.x - self.x) + abs(o.y - self.y)
            if d < best_d:
                best, best_d = o, d
        return best

    def _run_every(self, factor: int) -> int:
        """Running (chasing, fleeing) moves `factor` times faster than walking."""
        return max(1, self.spec.move_every // factor)

    # ------------------------------------------------------------------
    # Charging shark (W2b)
    # ------------------------------------------------------------------
    def _tick_charge(self, agent_x: int, agent_y: int, world) -> int:
        """Wait `wait` ticks in plain sight, lock onto the agent's spot, then
        charge it one tile per tick in a straight line. Contact kills. Moving
        after the lock dodges; moving before it just changes the aim. Missed
        or blocked, it waits again. The agent on land: it sinks and is gone."""
        if world is not None and not _at_sea(world, agent_x, agent_y):
            self.alive = False
            return 0
        if (self.x, self.y) == (agent_x, agent_y):
            return 9999
        if self.charge_to is None:
            if self.wait_left < 0:
                self.wait_left = self.spec.hostility_args["wait"]
            self.job, self.state = "wait", 1      # state 1 = the brain's "telegraph"
            self.wait_left -= 1
            if self.wait_left <= 0:
                self.charge_to = (agent_x, agent_y)
                self.job, self.state = "charge", 2
            return 0

        tx, ty = self.charge_to
        dx, dy = tx - self.x, ty - self.y
        step = ((dx > 0) - (dx < 0), 0) if abs(dx) >= abs(dy) else (0, (dy > 0) - (dy < 0))
        nx, ny = self.x + step[0], self.y + step[1]
        if world is not None and not _passable(world, nx, ny, sea_only=True):
            self._wait_again()
            return 0
        self.x, self.y = nx, ny
        if (nx, ny) == (agent_x, agent_y):
            return 9999
        if (nx, ny) == self.charge_to:
            self._wait_again()
        return 0

    def _wait_again(self):
        self.charge_to = None
        self.wait_left = -1
        self.job, self.state = "wait", 1

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
    def _step(self, world, target: tuple[int, int], away: bool, every: int):
        """Move toward or away from `target`, once every `every` ticks."""
        tx, ty = target
        if not away:                       # toward = away from the mirror point
            tx, ty = 2 * self.x - tx, 2 * self.y - ty
        self._wander(world, flee_from=(tx, ty), every=every)

    def _wander(self, world, flee_from: tuple[int, int] | None = None, every: int = 0):
        """Move randomly (wander) or away from a point (flee)."""
        self._wander_ticks += 1
        if self._wander_ticks < (every or self.spec.move_every):
            return

        self._wander_ticks = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        if flee_from is not None:
            fx, fy = flee_from
            # Sort directions: prefer those that increase distance from flee_from
            directions.sort(
                key=lambda d: -(abs(self.x + d[0] - fx) + abs(self.y + d[1] - fy))
            )

            best = directions[:2]
            random.shuffle(best)            # some randomness even when fleeing
            directions[:2] = best
        else:
            random.shuffle(directions)

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
    def take_damage(self, amount: int, attacker_pos: tuple[int, int] | None = None,
                    world=None, by_agent: bool = True):
        if self.spec.has("unkillable"):
            return
        self.hp -= amount
        if by_agent and self.spec.grudge:
            self.angry = True
        if self.hp <= 0:
            self.alive = False
            # What it splits into (Magma Cube). world.attack_at doesn't pass
            # `world`, so in the real game this never fires yet.
            if world is not None:
                for sid, dx, dy in self.spec.spawn_on_death:
                    world.spawn_entity(sid, self.x + dx, self.y + dy)
        elif self.spec.hostility == "flees_when_hit" and attacker_pos:
            # Passive creatures run from whoever hit them, for a while
            self.flee_from = attacker_pos
            self.flee_left = FLEE_TICKS
            self.rest_left = 0


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
            return tile is not None and tile.tile_type == 8   # TILE_OCEAN
        # Land creatures stay on land: no sea, no wall, no gate
        return tile is not None and tile.tile_type not in (1, 7, 8)
    except Exception:
        return True


def _at_sea(world, x: int, y: int) -> bool:
    tile = world._tile(x, y)
    return tile is not None and tile.tile_type == 8       # TILE_OCEAN


def _land(world, x: int, y: int) -> bool:
    """Open ground a grazer can eat from (W3 replaces this with real food)."""
    tile = world._tile(x, y) if world is not None else None
    return tile is not None and tile.tile_type == 0    # TILE_EMPTY

