"""Make a curriculum rung look like the full environment, for the GUI.

The menu has always been able to *pick* a rung. Nothing read the choice:
``run_gui`` built ``ATSEnvironment()`` every time, so pressing LAUNCH: NURSERY
dropped the agent on the island. ``selected_track`` was never read outside the
GUI, and ``active_stage`` / ``active_mask`` were set to None in ``__init__``
and never assigned anywhere, despite a comment describing exactly what a rung
was supposed to put in them.

The two environments have nothing in common at the surface. ``ATSEnvironment``
carries a world, an agent with an inventory, a day/night cycle and a reward
engine; ``CurriculumEnv`` is a grid, a position and a health number. The GUI
reads all of the former.

So rather than teach the GUI about stages, or grow CurriculumEnv into
something it should not be, this presents one as the other. The stage stays a
stage — a small room with a short action list, which is the whole point of a
first rung — and the panel finds what it expects to draw.

Anything the stage genuinely does not have is reported as absent rather than
faked: no inventory, no night, no progression points. The GUI already greys
out what ``active_mask`` says is unavailable, so absent reads as "not part of
this rung" instead of as a broken panel.
"""

from __future__ import annotations

import config_rl
from curriculum import CurriculumEnv, T_HAZARD, T_WALL


# --- the pieces the panel reads ------------------------------------------

class _Tile:
    """What world._tile returns: explored, maybe an object, a type number."""
    __slots__ = ("explored", "object_id", "tile_type")

    def __init__(self, tile_type: int, object_id=None):
        self.explored = True          # a room is small enough to be all known
        self.object_id = object_id
        self.tile_type = tile_type


class _Entity:
    """A hostile, in the shape the 7x7 view expects."""
    __slots__ = ("alive", "x", "y", "entity_id", "data")

    def __init__(self, x, y, entity_id=0):
        self.alive = True
        self.x, self.y = x, y
        self.entity_id = entity_id
        self.data = {"type": "hostile"}


class _World:
    """The stage grid, answering the two questions the panel asks of a world."""

    def __init__(self, env: CurriculumEnv):
        self._env = env

    @property
    def entities(self):
        e = self._env
        return [_Entity(x, y) for (x, y) in getattr(e, "hostiles", [])]

    def get_zone_info(self, x, y):
        """Biomes belong to the island. A room is one room."""
        return {"name": self._env.stage.name, "tier": 0, "biome": "room"}

    def nearest_entity(self, x, y):
        """The closest hostile, or nothing — the shape the readout expects."""
        best, best_d = None, None
        for (hx, hy) in getattr(self._env, "hostiles", []):
            d = abs(hx - x) + abs(hy - y)
            if best_d is None or d < best_d:
                best, best_d = _Entity(hx, hy), d
        return best, (best_d if best_d is not None else 0)

    def _tile(self, x, y):
        e = self._env
        n = e.stage.grid
        if not (0 <= x < n and 0 <= y < n):
            return None                      # outside the room: drawn as unknown

        raw = e.grid[y][x]
        if raw == T_WALL:
            return _Tile(1)
        if raw == T_HAZARD:
            return _Tile(6)                  # the panel already draws 6 as lava

        # Goals and pickups are the only objects a rung has. They are shown
        # through object_id so they read as things rather than as floor.
        if (x, y) in getattr(e, "goals", []):
            return _Tile(0, object_id=_GOAL_ID)
        if (x, y) in getattr(e, "good", []):
            return _Tile(0, object_id=_GOOD_ID)
        if (x, y) in getattr(e, "trash", []):
            return _Tile(0, object_id=_TRASH_ID)
        # A weapon on the floor. Without this the combat rungs look like the
        # agent is walking over bare ground and then killing things faster
        # for no visible reason.
        if (x, y) in getattr(e, "swords", []):
            return _Tile(0, object_id=_SWORD_ID)
        return _Tile(0)


# Item ids only used for their names in the 7x7 view. Picked so the mapping in
# _draw_sim falls through to its raw[:5] branch and prints something readable.
_GOAL_ID, _GOOD_ID, _TRASH_ID = -101, -102, -103
# A real registry id, not a sentinel: item_name gives "Stone Sword" and
# the panel names it without the adapter carrying its own table.
_SWORD_ID = "0013"


class _Agent:
    """The agent surface: position, facing, health, and an empty inventory."""

    # Which way the last move went. This was pinned to "N", so the panel read
    # FACING: NORTH and drew [AI^] for every step the agent ever took, whatever
    # it actually did — and a uniform policy looked like a policy stuck on one
    # action. A readout that lies is worse than no readout.
    _FACING = {0: "N", 1: "S", 2: "W", 3: "E"}

    def __init__(self, env: CurriculumEnv):
        self._env = env
        self.facing_dir = "N"
        self.selected_slot = 0
        self.capabilities = set()
        self.event_log = []
        # Shown greyed by the panel, because active_mask says pick-up is off in
        # the rungs that cannot pick anything up.
        self._inventory = [{"id": None, "count": 0}
                           for _ in range(config_rl.INVENTORY_SLOTS)]

    @property
    def inventory(self):
        """Slot 0 holds the sword once it has been picked up.

        A property rather than a plain list because the panel reads this on
        every frame and the agent arms itself mid-episode. As a list set once
        in __init__, an agent that had armed itself looked identical to one
        that had not - which on a combat rung is the single most important
        thing on the screen.

        0013 is the real registry's Stone Sword, so the panel names it
        without the adapter carrying its own table.
        """
        armed = bool(getattr(self._env, "armed", False))
        self._inventory[0]["id"] = "0013" if armed else None
        self._inventory[0]["count"] = 1 if armed else 0
        return self._inventory

    @property
    def x(self): return self._env.ax

    @property
    def y(self): return self._env.ay

    @property
    def health(self): return self._env.health

    # Same signature as the real Agent, which takes the world and the clock
    # it is about to consult. A rung's mask depends on neither — it is fixed by
    # what the stage has taught — but the caller does not know that.
    def get_action_mask(self, world=None, day_night=None):
        return self._env.action_mask()

    def get_facing_tile_info(self, world=None):
        """What is under the square ahead. The room has no facing, so this
        answers for the square the agent is standing on instead of inventing a
        direction it does not have."""
        e = self._env
        if (e.ax, e.ay) in getattr(e, "goals", []):
            return "GOAL"
        if (e.ax, e.ay) in getattr(e, "hazards", []):
            return "HAZARD"
        if (e.ax, e.ay) in getattr(e, "swords", []):
            return "SWORD"
        return "floor"

    # Numbers the island tracks and a rung does not. Reported at their resting
    # values rather than omitted, so the readout keeps its shape and says
    # plainly that nothing here moves them.
    hunger = 100
    stamina = 100
    xp = 0
    level = 1
    ocean_ticks = 0
    water_speed_mult = 1.0
    last_action_name = "-"


class _Clock:
    """A rung has no night: permanent daylight, so nothing is hidden."""
    is_night = False
    time_of_day = 0.5


class _Rewards:
    """Enough of the reward engine for the readouts, and no more.

    A rung has no progression system to score — that belongs to the island —
    so efficiency is reported as "not applicable here" rather than as 0.000.
    Zero is a measurement. This is the absence of one, and a panel that cannot
    tell them apart sends you hunting a flatline that was never a number.
    """

    def __init__(self):
        self.total = 0.0
        self.progression_points = 0

    def compute_progression_efficiency(self, tick):
        return None


# --- the adapter ----------------------------------------------------------

class StageSession:
    """One curriculum rung, wearing the environment's face."""

    def __init__(self, stage, seed: int | None = None):
        self.stage = stage
        self._env = CurriculumEnv(stage, seed=seed)
        self.world = _World(self._env)
        self.agent = _Agent(self._env)
        self.day_night = _Clock()
        self.rewards = _Rewards()
        self.tick = 0
        self.episode = 0
        self.done = False

    # The stage's own action mask, so the panel can grey what this rung has
    # not taught rather than showing twenty-five live buttons.
    def action_mask(self):
        return self._env.action_mask()

    def reset(self):
        self.tick = 0
        self.done = False
        self.rewards.total = 0.0
        self.agent.event_log.append((f"{self.stage.name}: {self.stage.grid}x{self.stage.grid} room", 0))
        return self._env.reset()

    # The island's step takes the panel's live toggles. A rung answers to its
    # stage instead: its action mask is what the stage has taught, and its
    # reward rules are the ones the ladder was designed and sanity-checked
    # against. Accepting the arguments and ignoring them keeps the caller
    # unchanged and keeps a rung honest — a toggle that silently rewrote the
    # curriculum's rewards would make best_case_return a lie.
    def step(self, action, disabled_actions=None, reward_rules=None):
        face = _Agent._FACING.get(int(action))
        if face:
            self.agent.facing_dir = face
        state, reward, done, info = self._env.step(action)
        self.tick = self._env.steps
        self.rewards.total += reward
        self.done = done
        if done:
            self.episode += 1
            self.agent.event_log.append(
                ("cleared" if info.get("success") else
                 "died" if info.get("dead") else "out of steps", 0))
        # The island reports needs; a rung has none, so the panel is handed a
        # flat set rather than whatever the last island run left behind.
        info.setdefault("needs", [1.0, 1.0, 1.0, 1.0])
        return state, reward, done, info


def stage_by_name(name: str):
    """The Stage object a menu rung refers to, or None for the full world."""
    from curriculum import default_ladder
    for s in default_ladder():
        if s.name == name:
            return s
    return None
