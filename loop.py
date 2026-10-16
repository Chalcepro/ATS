"""W4: the Loop - dungeon rooms you can't leave until the goal is met.

Chalce, 2026-10-16 (docs/W4-LOOP-FLOW.md). Each room is a stone chamber
built for one challenge, with walls and a doorway in each one. Until the goal
is met, walking out of a doorway walks you into the same room again: the room
repeats in every direction, so the view through a doorway shows the room
itself. Once the goal is met the doors open: the next doorway leads out, and
the route channel shows the way to it. A lever room's way out is the lever.

Dying in the world lands the agent here; LOOP_GOALS_BEFORE_DROP rooms later
it is back in the world beside the nearest real version of the last room.
Rooms are data (ROOMS below): layout, kit, the body it arrives in, the goal,
and how to find "the same thing" in the real world.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from entities import Entity
from world import (World, Tile, TILE_EMPTY, TILE_WALL, TILE_OBJECT,
                   TILE_OCEAN, TILE_STONE, ISLAND_DARK_FOREST)

# Every room is SIZE x SIZE with a doorway in the middle of each wall, so
# walking out the east door is walking in at the west door.
SIZE = 15
MID = SIZE // 2
DOORS = {(MID, 0), (MID, SIZE - 1), (0, MID), (SIZE - 1, MID)}
_TILES = {"#": TILE_WALL, ".": TILE_STONE, "~": TILE_OCEAN}
_OBJECTS = {"a": "0001", "l": "0088"}     # apple, lever
WOLVES = ("E011", "E021")


@dataclass
class Room:
    name: str
    layout: list[str]
    goal: Callable                        # (env) -> True once met
    find: Callable                        # (world, near) -> (x, y) in the real world
    kit: list = field(default_factory=list)      # [(slot, item_id, count)]
    selected: int = 0                     # the slot it arrives holding
    health: int = 100
    hunger: int = 100
    creatures: dict = field(default_factory=dict)  # layout char -> entity id
    sends_back: Callable | None = None    # (env) -> True: back to the start, no death
    death_kinds: tuple = ()               # causes of death this room answers
    goal_is_exit: bool = False            # the lever: meeting it IS leaving

    def start(self):
        for y, row in enumerate(self.layout):
            if "A" in row:
                return row.index("A"), y
        raise ValueError("room %s has no start" % self.name)


class LoopWorld(World):
    """A room that repeats in every direction. Reads wrap, so the agent's
    view through a doorway shows the room again; positions are wrapped back
    into the room by the env after each tick (wrap_all)."""

    def __init__(self, real: World, room: Room):
        # Not World.__init__: that builds six islands. Only what the agent,
        # the creatures and the env read is set up here.
        self.seed = real.seed
        self.room = room
        self.doors_open = False
        self.tiles: dict = {}
        self.entities: list = []
        self._start_counts, self._grid, self._reach = {}, {}, {}
        self._trees0, self._size0, self.villages, self._regrow = {}, {}, [], []
        self._life_ticks = 0
        self._materialised = set()
        self.door_unlocked = True
        self.gates = {}
        self.swim_pace = 1
        self.island_registry = real.island_registry
        self.starting_island = real.starting_island
        for y, row in enumerate(room.layout):
            assert len(row) == SIZE, (room.name, y, len(row))
            for x, ch in enumerate(row):
                t = Tile(tile_type=_TILES.get(ch, TILE_STONE),
                         island_id=-1 if ch == "~" else 0)
                t.object_id = _OBJECTS.get(ch, "")
                self.tiles[(x, y)] = t
                if ch in room.creatures:
                    self.entities.append(Entity.from_id(room.creatures[ch], x, y))
        self.agent_spawn = room.start()

    @staticmethod
    def wrap(x, y):
        return x % SIZE, y % SIZE

    def wrap_all(self, agent) -> None:
        agent.x, agent.y = self.wrap(agent.x, agent.y)
        for e in self.entities:
            e.x, e.y = self.wrap(e.x, e.y)

    def _tile(self, x, y):
        return self.tiles[self.wrap(x, y)]

    def entity_at(self, x, y):
        return super().entity_at(*self.wrap(x, y))

    def tile_type_at(self, x, y):
        return super().tile_type_at(*self.wrap(x, y))

    def in_bounds_passable(self, x, y, door_unlocked=True):
        return super().in_bounds_passable(*self.wrap(x, y), door_unlocked)

    def pickup_at(self, x, y):
        t = self._tile(x, y)
        if t.object_id == "0088":            # a lever is pulled, not carried
            return ""
        return super().pickup_at(*self.wrap(x, y))

    def get_island_id_at(self, x, y):
        return self._tile(x, y).island_id

    def get_island_tier_at(self, x, y):
        return 0

    def get_zone_info(self, x, y):
        return {"id": -2, "tier": 0, "name": "the Loop", "safe": False, "danger": 0.0}

    def is_safe_zone(self, x, y):
        return False

    def materialise(self, island_id):
        return False

    def bridge_target(self, x, y):
        return -1

    def tick_life(self, ax=0, ay=0):
        self._life_ticks += 1           # nothing is born or regrows in a room
        return []

    def lever_pulled(self):
        return any(t.flags == "pulled" for t in self.tiles.values())

    def route(self, ax, ay):
        """Cells on a shortest walk to the way out, for the route channel
        (the same teaching aid as the primary maze rooms). The way out is the
        lever until it's pulled, or a doorway once the doors are open. Empty
        otherwise: the route never solves the room's own challenge."""
        if self.doors_open:
            goals = DOORS
        else:
            goals = {p for p, t in self.tiles.items() if t.object_id == "0088"}
            if not goals:
                return set()
            # a lever is reached by standing next to it
            goals = {(x + dx, y + dy) for x, y in goals
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if 0 <= x + dx < SIZE and 0 <= y + dy < SIZE
                     and self.tiles[(x + dx, y + dy)].tile_type != TILE_WALL
                     and not self.tiles[(x + dx, y + dy)].object_id}
        start = self.wrap(ax, ay)
        prev, q, end = {start: None}, deque([start]), None
        if start in goals:
            end = start
        while q and end is None:
            cur = q.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                c = (cur[0] + dx, cur[1] + dy)
                if c in prev or c not in self.tiles or self.tiles[c].tile_type in (TILE_WALL, TILE_OCEAN):
                    continue
                prev[c] = cur
                if c in goals:
                    end = c
                    break
                q.append(c)
        out, cur = set(), end
        while cur is not None:
            out.add(cur)
            cur = prev[cur]
        # in the agent's own coordinates (it is always inside the room)
        return {(ax - start[0] + x, ay - start[1] + y) for x, y in out}


# ---------------------------------------------------------------------------
# Finding "the same thing" in the real world, near where the agent was
# ---------------------------------------------------------------------------
def _land(world, x, y):
    t = world.tiles.get((x, y))
    return t is not None and t.tile_type in (TILE_EMPTY, TILE_OBJECT) \
        and world.entity_at(x, y) is None


def _beside(world, tx, ty, dist=4):
    """A free land tile about `dist` from (tx, ty), nearest ring first."""
    for r in range(dist, dist + 6):
        for p in [(tx + dx, ty + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                  if max(abs(dx), abs(dy)) == r]:
            if _land(world, *p):
                return p
    return None


def _nearest(points, near):
    return min(points, key=lambda p: abs(p[0] - near[0]) + abs(p[1] - near[1]), default=None)


def find_wolf(world, near):
    wolves = [(e.x, e.y) for e in world.entities if e.alive and e.entity_id in WOLVES]
    if not wolves and world.materialise(ISLAND_DARK_FOREST):   # wolves live there
        wolves = [(e.x, e.y) for e in world.entities if e.alive and e.entity_id in WOLVES]
    w = _nearest(wolves, near)
    return _beside(world, *w) if w else None


def find_coast(world, near):
    """In the sea, two tiles off the nearest beach: the room's own start."""
    sea = [p for p, t in world.tiles.items() if t.tile_type == TILE_OCEAN
           and any(_land(world, p[0] + dx, p[1] + dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2))]
    return _nearest(sea, near)


def find_food(world, near):
    f = _nearest([p for p, t in world.tiles.items() if t.object_id == "0001"], near)
    return _beside(world, *f) if f else None


def find_start(world, near):
    """The lever has no twin in the world: back where it died, on land."""
    return _beside(world, *near, dist=0)


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
def _no_wolf(env):
    return not any(e.alive for e in env.world.entities if e.entity_id in WOLVES)


def _on_land_a_while(env):
    """Shore: out of the water and on stone for LOOP_SHORE_TICKS in a row."""
    a = env.agent
    land = env.world._tile(a.x, a.y).tile_type != TILE_OCEAN
    env.loop_on_land = env.loop_on_land + 1 if land else 0
    return env.loop_on_land >= env.loop_shore_ticks


def _back_in_the_water(env):
    """Shore (Chalce): made it out, then went back into the sea -> start over."""
    a = env.agent
    in_sea = env.world._tile(a.x, a.y).tile_type == TILE_OCEAN
    env.loop_was_on_land = getattr(env, "loop_was_on_land", False) or not in_sea
    return in_sea and env.loop_was_on_land


def _ate(env):
    return env.agent.hunger > env.loop_arrival_hunger


def _pulled(env):
    return env.world.lever_pulled()


# ---------------------------------------------------------------------------
# The rooms
# ---------------------------------------------------------------------------
ROOMS = [
    # A full-health wolf (8 HP); the sword does 5, two hits. It arrives at
    # 20 HP holding an apple, the sword in slot 1. Bare hands need eight hits
    # and five bites kill it: equip the sword or lose.
    Room("wolf", [
        "#######.#######",
        "#.............#",
        "#.............#",
        "#.............#",
        "#......W......#",
        "#.............#",
        "#.............#",
        "...............",
        "#......A......#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#######.#######",
    ], goal=_no_wolf, find=find_wolf,
        kit=[(0, "0001", 1), (1, "0002", 1)], selected=0, health=20,
        creatures={"W": "E011"}, death_kinds=("wolf",)),

    # Chalce's shore room: it starts in a pool of sea. Out onto the stone and
    # stay there a few seconds; back into the water and it starts over.
    Room("shore", [
        "#######.#######",
        "#.............#",
        "#.............#",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "...~~~~A~~~~...",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "#..~~~~~~~~~..#",
        "#.............#",
        "#.............#",
        "#######.#######",
    ], goal=_on_land_a_while, find=find_coast, sends_back=_back_in_the_water,
        death_kinds=("drowning", "shark")),

    # Hungry, and the apples are round the corner.
    Room("hungry", [
        "#######.#######",
        "#.............#",
        "#..a.......a..#",
        "#.............#",
        "#####.....#####",
        "#.............#",
        "#.............#",
        "...............",
        "#......A......#",
        "#.............#",
        "#####.....#####",
        "#.............#",
        "#..a.......a..#",
        "#.............#",
        "#######.#######",
    ], goal=_ate, find=find_food, hunger=10, death_kinds=("starvation",)),

    # The way out is a lever (interact, E). The route channel shows the path.
    Room("lever", [
        "#######.#######",
        "#.....#.......#",
        "#.....#.......#",
        "#.....#...l...#",
        "#.....#.......#",
        "#.....###.#####",
        "#.............#",
        "...............",
        "#......A......#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#.............#",
        "#######.#######",
    ], goal=_pulled, find=find_start, goal_is_exit=True),
]
BY_NAME = {r.name: r for r in ROOMS}


def pick_room(cause: str | None, rng: random.Random, avoid: str | None = None) -> Room:
    """Half the time the room about what killed it (drowned -> shore),
    otherwise any room. Never `avoid`, the one it was just in."""
    pool = [r for r in ROOMS if r.name != avoid] or ROOMS
    low = (cause or "").lower()
    matched = [r for r in pool if any(k in low for k in r.death_kinds)]
    if matched and rng.random() < 0.5:
        return rng.choice(matched)
    return rng.choice(pool)
