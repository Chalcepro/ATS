"""W4: the Loop - dungeon rooms you can't leave until the goal is met.

Chalce, 2026-10-16 (docs/W4-LOOP-FLOW.md). Each room is a stone chamber
built for one challenge, with walls and a doorway in each one. Until the goal
is met, walking out of a doorway walks you into the same room again: the room
repeats in every direction, so the view through a doorway shows the room
itself. Once the goal is met the doors open and a doorway leads out. A lever
room's way out is the lever.

Rooms are GENERATED, not drawn: each entry rolls a kind (fight, food, lever,
shore, lava, chase) and then the details - the layout's inner walls and pillars,
which creature and how many, which weapon in which slot, which food and where,
whether the lever is in plain sight or tucked away, the pool's shape. So it
can't learn four rooms by heart. A retry after dying is the SAME room
(the Room object is kept); a new entry is a new one.

Dying in the world lands the agent here; LOOP_GOALS_BEFORE_DROP rooms later
it is back in the world beside the nearest real version of the last room.
"""
from __future__ import annotations

import dataclasses
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from entities import Entity
from item_ids import REGISTRY
from world import (World, Tile, TILE_EMPTY, TILE_WALL, TILE_OBJECT, TILE_OCEAN,
                   TILE_STONE, TILE_LAVA, TILE_ASH)

SIZE = 15
MID = SIZE // 2
DOORS = {(MID, 0), (MID, SIZE - 1), (0, MID), (SIZE - 1, MID)}
LEVER = "0088"
_TILES = {"#": TILE_WALL, ".": TILE_STONE, "~": TILE_OCEAN, "L": TILE_LAVA, ":": TILE_ASH}
_BLOCKED = (TILE_WALL, TILE_OCEAN, TILE_LAVA)

# Who can be in a fight room: every ordinary hostile. Not the bosses, the
# Void, sharks, or the Creeper (its blast is a different lesson).
FIGHTERS = ["E001", "E002", "E003", "E005", "E006", "E007", "E008", "E010",
            "E011", "E012", "E021", "E023", "E024", "E025", "E026"]
PACKS = {"E011": 3, "E021": 3, "E001": 3, "E005": 2, "E008": 2, "E027": 2}
WEAPONS = ["0034", "0002", "0033", "0013"]          # club 3, sword 5, dagger 6, stone sword 8
FOODS = ["0001", "0023", "0031", "0032"]            # apple, berry bush, raw meat, bread
CLUTTER = ["0008", "0010", "0012", "0003", "0024"]  # stick, wood, stone, book, fallen log
TREASURE = ["0006", "0003"]                         # key, book: what the lava room guards
CHASE_SAFE = 8          # chase room: this far from the chaser (around the loop) and it loses you
CHASE_HP_TIMES = 6      # a chaser can't be fought off bare-handed, only outrun


@dataclass
class Room:
    kind: str
    layout: list[str]
    start: tuple
    goal: Callable                        # (env) -> True once met
    find: Callable                        # (world, near) -> (x, y) in the real world
    objects: dict = field(default_factory=dict)     # (x, y) -> item id
    creatures: list = field(default_factory=list)   # [(entity id, x, y)]
    kit: list = field(default_factory=list)         # [(slot, item id, count)]
    selected: int = 0                     # the slot it arrives holding
    health: int = 100
    hunger: int = 100
    sends_back: Callable | None = None    # (env) -> True: back to the start, no death
    goal_is_exit: bool = False            # the lever: meeting it IS leaving
    route_hint: bool = False              # route channel shows the way to the lever
    stamina: int = 100
    chase: bool = False                   # its creatures hunt it, faster than it walks
    about: str = ""                       # what it is, for the log

    @property
    def name(self):
        return self.about or self.kind


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
            assert len(row) == SIZE, (room.kind, y, len(row))
            for x, ch in enumerate(row):
                self.tiles[(x, y)] = Tile(tile_type=_TILES[ch],
                                          island_id=-1 if ch == "~" else 0)
        for p, item_id in room.objects.items():
            self.tiles[p].object_id = item_id
        for eid, x, y in room.creatures:
            e = Entity.from_id(eid, x, y)
            if room.chase:              # hunts it the whole time, and can't be beaten
                e.spec = dataclasses.replace(e.spec, hp=e.spec.hp * CHASE_HP_TIMES, grudge=99)
                e.hp, e.angry = e.spec.hp, True
            self.entities.append(e)
        self.agent_spawn = room.start

    @staticmethod
    def wrap(x, y):
        return x % SIZE, y % SIZE

    @staticmethod
    def near_image(fx, fy, x, y):
        """(x, y) moved by whole rooms to the copy nearest (fx, fy): the room
        repeats, so a creature goes for the agent the short way round."""
        return (x + round((fx - x) / SIZE) * SIZE, y + round((fy - y) / SIZE) * SIZE)

    @classmethod
    def loop_dist(cls, a, b):
        x, y = cls.near_image(a[0], a[1], b[0], b[1])
        return abs(x - a[0]) + abs(y - a[1])

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
        if self._tile(x, y).object_id == LEVER:   # a lever is pulled, not carried
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
        if self.room.chase and self._life_ticks % 2:
            # A chaser's extra stride every other tick: ~1.5 tiles a tick, so
            # walking away loses and a sprint (2) gets away while stamina lasts.
            for e in self.entities:
                tx, ty = self.near_image(e.x, e.y, ax, ay)
                if e.alive and abs(tx - e.x) + abs(ty - e.y) > e.spec.hostility_args.get("space", 1):
                    e._step(self, (tx, ty), away=False, every=1)
        return []

    def lever_pulled(self):
        return any(t.flags == "pulled" for t in self.tiles.values())

    def route(self, ax, ay):
        """Cells on a shortest walk to the way out, for the route channel
        (the same teaching aid as the primary maze rooms): a doorway once the
        doors are open, or a tucked-away lever. Empty otherwise - the route
        never solves the room's own challenge."""
        if self.doors_open:
            goals = set(DOORS)
        elif self.room.route_hint:
            goals = {n for p, t in self.tiles.items() if t.object_id == LEVER
                     for n in _around(*p) if n in self.tiles
                     and self.tiles[n].tile_type not in _BLOCKED and not self.tiles[n].object_id}
        else:
            return set()
        start = self.wrap(ax, ay)
        path = _path(lambda c: c in self.tiles and self.tiles[c].tile_type not in _BLOCKED,
                     start, goals)
        # in the agent's own coordinates (it is always inside the room)
        return {(ax - start[0] + x, ay - start[1] + y) for x, y in path}


def _around(x, y):
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _path(ok, start, goals):
    """Shortest 4-way path from start to any goal, as a set of cells."""
    prev, q = {start: None}, deque([start])
    end = start if start in goals else None
    while q and end is None:
        cur = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            c = (cur[0] + dx, cur[1] + dy)
            if c in prev or not ok(c):
                continue
            prev[c] = cur
            if c in goals:
                end = c
                break
            q.append(c)
    out = set()
    while end is not None:
        out.add(end)
        end = prev[end]
    return out


# ---------------------------------------------------------------------------
# Layout: walls, doorways, and some random inner walls and pillars
# ---------------------------------------------------------------------------
def _shell():
    g = [["#"] * SIZE for _ in range(SIZE)]
    for y in range(1, SIZE - 1):
        for x in range(1, SIZE - 1):
            g[y][x] = "."
    for x, y in DOORS:
        g[y][x] = "."
    return g


def _keep_open(g):
    """The doorways and the tile inside each, and the centre cross to them."""
    for x, y in DOORS:
        g[y][x] = "."
        ix, iy = min(max(x, 1), SIZE - 2), min(max(y, 1), SIZE - 2)
        g[iy][ix] = "."


def _decorate(g, rng):
    """Inner walls with gaps, and pillars. Rolled per room."""
    for _ in range(rng.choice([0, 1, 1, 2])):          # partition walls
        horiz, at = rng.random() < 0.5, rng.randint(3, SIZE - 4)
        if at == MID:
            at += rng.choice([-2, 2])
        gaps = set(rng.sample(range(1, SIZE - 1), rng.randint(1, 3)))
        for i in range(1, SIZE - 1):
            if i not in gaps:
                if horiz:
                    g[at][i] = "#"
                else:
                    g[i][at] = "#"
    for _ in range(rng.randint(0, 6)):                  # pillars
        x, y = rng.randint(2, SIZE - 3), rng.randint(2, SIZE - 3)
        g[y][x] = "#"
        if rng.random() < 0.3:
            g[y][min(x + 1, SIZE - 2)] = "#"
    _keep_open(g)


def _walkable(g, c):
    x, y = c
    return 0 <= x < SIZE and 0 <= y < SIZE and g[y][x] in ".:"


def _reach(g, start):
    seen, q = {start}, deque([start])
    while q:
        cx, cy = q.popleft()
        for c in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if c not in seen and _walkable(g, c):
                seen.add(c)
                q.append(c)
    return seen


def _bad(reach):
    """Too cramped, or a doorway walled off: roll the room again."""
    return len(reach) < 40 or not DOORS <= reach


def _floor(g, reach, taken, rng, far_from=None, near=None):
    """A free reachable floor tile, optionally far from / near a point."""
    cells = [c for c in reach if c not in taken and c not in DOORS and g[c[1]][c[0]] == "."]
    if far_from:
        cells.sort(key=lambda c: -(abs(c[0] - far_from[0]) + abs(c[1] - far_from[1])))
        cells = cells[:max(3, len(cells) // 4)]
    if near:
        cells = [c for c in cells if max(abs(c[0] - near[0]), abs(c[1] - near[1])) <= 3] or cells
    c = rng.choice(cells)
    taken.add(c)
    return c


def _rows(g):
    return ["".join(r) for r in g]


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


def find_creature(eid):
    def find(world, near):
        found = [(e.x, e.y) for e in world.entities if e.alive and e.entity_id == eid]
        spec = REGISTRY.get_entity(eid)
        if not found and spec and world.materialise(spec.island):   # where they live
            found = [(e.x, e.y) for e in world.entities if e.alive and e.entity_id == eid]
        c = _nearest(found, near)
        return _beside(world, *c) if c else None
    return find


def find_item(item_id):
    def find(world, near):
        c = _nearest([p for p, t in world.tiles.items() if t.object_id == item_id], near)
        return _beside(world, *c) if c else None
    return find


def find_coast(world, near):
    """In the sea, two tiles off the nearest beach: the room's own start."""
    sea = [p for p, t in world.tiles.items() if t.tile_type == TILE_OCEAN
           and any(_land(world, p[0] + dx, p[1] + dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2))]
    return _nearest(sea, near)


def find_lava(world, near):
    c = _nearest([p for p, t in world.tiles.items() if t.tile_type == TILE_LAVA], near)
    return _beside(world, *c, dist=3) if c else None


def find_start(world, near):
    """The lever has no twin in the world: back where it died, on land."""
    return _beside(world, *near, dist=0)


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
def _all_dead(env):
    return not any(e.alive for e in env.world.entities)


def _on_land_a_while(env):
    """Shore: out of the water, on stone for LOOP_SHORE_TICKS in a row."""
    a = env.agent
    land = env.world._tile(a.x, a.y).tile_type != TILE_OCEAN
    env.loop_on_land = env.loop_on_land + 1 if land else 0
    return env.loop_on_land >= env.loop_shore_ticks


def _back_in_the_water(env):
    """Shore (Chalce): made it out, then went back into the sea -> start over."""
    a = env.agent
    in_sea = env.world._tile(a.x, a.y).tile_type == TILE_OCEAN
    env.loop_was_on_land = env.loop_was_on_land or not in_sea
    return in_sea and env.loop_was_on_land


def _ate(env):
    return env.agent.hunger > env.loop_arrival_hunger


def _pulled(env):
    return env.world.lever_pulled()


def _got_away(env):
    """Chase: CHASE_SAFE from every chaser (the short way round the loop).
    It loses the trail and is gone."""
    a, w = env.agent, env.world
    hunters = [e for e in w.entities if e.alive]
    if all(w.loop_dist((e.x, e.y), (a.x, a.y)) >= CHASE_SAFE for e in hunters):
        for e in hunters:
            e.alive = False
        return True
    return False


def _holds(item_id):
    def goal(env):
        return any(s["id"] == item_id and s["count"] > 0 for s in env.agent.inventory)
    return goal


# ---------------------------------------------------------------------------
# The rooms
# ---------------------------------------------------------------------------
def _kit_with(rng, weapon=None):
    """What it arrives carrying. Slot 0 (the one in hand) is never the
    weapon: picking the right slot is half of what a fight room teaches."""
    slots = list(range(1, 7))
    rng.shuffle(slots)
    kit = [(0, rng.choice(CLUTTER[:4] + ["0001"]), 1)]
    if weapon:
        kit.append((slots.pop(), weapon, 1))
    for _ in range(rng.randint(0, 2)):                 # other things it carries
        kit.append((slots.pop(), rng.choice(CLUTTER[:4] + ["0001"]), rng.randint(1, 3)))
    return kit


def fight_room(rng, eid=None):
    eid = eid or rng.choice(FIGHTERS)
    spec = REGISTRY.get_entity(eid)
    n = rng.randint(1, PACKS.get(eid, 1))
    # The weakest weapon that kills each one in about two or three hits.
    weapon = next((w for w in WEAPONS
                   if math.ceil(spec.hp / REGISTRY.get_item(w)["damage"]) <= 3), WEAPONS[-1])
    dmg = REGISTRY.get_item(weapon)["damage"]
    hits_armed = n * math.ceil(spec.hp / dmg)
    # Enough health to win holding the weapon, not enough to brawl through
    # with bare hands (1 damage a hit) when that takes clearly longer.
    health = max(15, min(100, spec.damage * (hits_armed + 3)))
    g = _shell()
    _decorate(g, rng)
    start = (MID, MID + 1)
    g[start[1]][start[0]] = "."
    reach = _reach(g, start)
    if _bad(reach):
        return fight_room(rng, eid)
    taken = {start}
    creatures = []
    for _ in range(n):
        x, y = _floor(g, reach, taken, rng, far_from=start if rng.random() < 0.5 else None)
        creatures.append((eid, x, y))
    return Room("fight", _rows(g), start, _all_dead, find_creature(eid),
                creatures=creatures, kit=_kit_with(rng, weapon), health=health,
                about="fight %s x%d" % (spec.name, n))


def food_room(rng):
    """Hungry AND weak: starving bleeds 1 HP per 30 ticks, so at full health
    a failed food room took ~3,000 ticks to end (seen in the first Loop run)."""
    food = rng.choice(FOODS)
    g = _shell()
    _decorate(g, rng)
    start = (rng.randint(2, SIZE - 3), rng.randint(2, SIZE - 3))
    g[start[1]][start[0]] = "."
    reach = _reach(g, start)
    if _bad(reach):
        return food_room(rng)
    taken = {start}
    objects = {}
    for _ in range(rng.randint(1, 3)):                  # the food, away from it
        objects[_floor(g, reach, taken, rng, far_from=start)] = food
    for _ in range(rng.randint(2, 6)):                  # things that aren't food
        objects[_floor(g, reach, taken, rng)] = rng.choice(CLUTTER)
    return Room("food", _rows(g), start, _ate, find_item("0001" if food == "0023" else food),
                objects=objects, kit=_kit_with(rng) if rng.random() < 0.5 else [],
                hunger=rng.randint(5, 25), health=rng.randint(25, 40),
                about="food %s" % REGISTRY.get_item(food)["name"])


def lever_room(rng):
    """In plain sight, or tucked in a walled corner nook with one gap - and
    then the route channel shows the way in."""
    g = _shell()
    _decorate(g, rng)
    start = (MID, MID + 1)
    where = rng.choice(["at hand", "in sight", "hidden"])
    hidden = where == "hidden"
    if hidden:
        left, top = rng.random() < 0.5, rng.random() < 0.5
        xs = range(1, 4) if left else range(SIZE - 4, SIZE - 1)
        ys = range(1, 4) if top else range(SIZE - 4, SIZE - 1)
        wx = 4 if left else SIZE - 5
        wy = 4 if top else SIZE - 5
        for y in ys:
            for x in xs:
                g[y][x] = "."
        wall = [(wx, y) for y in ys] + [(x, wy) for x in xs] + [(wx, wy)]
        for x, y in wall:
            g[y][x] = "#"
        gx, gy = rng.choice(wall[:-1])                  # one way in, not the corner
        g[gy][gx] = "."
        lever = (1 if left else SIZE - 2, 1 if top else SIZE - 2)
    g[start[1]][start[0]] = "."
    reach = _reach(g, start)
    if _bad(reach):
        return lever_room(rng)                          # walled itself in: roll again
    if where == "at hand":      # right beside it: pressing E once is the whole lesson
        lever = rng.choice([c for c in _around(*start) if c in reach and c not in DOORS])
    elif not hidden:
        lever = _floor(g, reach, {start}, rng)
    elif not any(n in reach for n in _around(*lever)):
        return lever_room(rng)
    taken = {start, lever}
    objects = {lever: LEVER}
    for _ in range(rng.randint(0, 4)):
        objects[_floor(g, reach, taken, rng)] = rng.choice(CLUTTER)
    return Room("lever", _rows(g), start, _pulled, find_start, objects=objects,
                goal_is_exit=True, route_hint=hidden,
                about="lever %s" % where)


def shore_room(rng):
    """It starts in a pool of sea; out onto the stone and stay there."""
    g = _shell()
    w, h = rng.randint(5, 9), rng.randint(5, 9)
    x0, y0 = rng.randint(2, SIZE - 2 - w), rng.randint(2, SIZE - 2 - h)
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            g[y][x] = "~"
    _keep_open(g)
    start = (x0 + w // 2, y0 + h // 2)
    return Room("shore", _rows(g), start, _on_land_a_while, find_coast,
                sends_back=_back_in_the_water, about="shore %dx%d pool" % (w, h))


def lava_room(rng):
    """Something worth having behind a band of lava, with a safe ash path."""
    g = _shell()
    horiz = rng.random() < 0.5
    at = rng.choice([4, 5, 9, 10])
    width = rng.randint(1, 2)
    paths = set(rng.sample(range(2, SIZE - 2), rng.randint(1, 2)))
    for k in range(width):
        for i in range(1, SIZE - 1):
            ch = ":" if i in paths else "L"
            if horiz:
                g[at + k][i] = ch
            else:
                g[i][at + k] = ch
    _keep_open(g)
    side_far = at < MID                                 # the treasure is across the band
    start = (MID, MID + 1) if horiz else (MID + 1, MID)
    lo, hi = (1, at - 1) if side_far else (at + width, SIZE - 2)
    treasure = rng.choice(TREASURE)
    x, y = rng.randint(1, SIZE - 2), rng.randint(lo, hi)
    spot = (x, y) if horiz else (y, x)
    g[spot[1]][spot[0]] = "."
    return Room("lava", _rows(g), start, _holds(treasure), find_lava,
                objects={spot: treasure}, about="lava, %s across"
                % REGISTRY.get_item(treasure)["name"])


def chase_room(rng, eid=None):
    """Chalce's Dark Souls ghosts: something hunts it, faster than it walks
    and slower than it sprints, and too tough to fight with what it has. Get
    far enough away and it loses you - if there's stamina left to run."""
    eid = eid or rng.choice(FIGHTERS)
    g = _shell()
    _decorate(g, rng)
    start = (MID, MID + 1)
    g[start[1]][start[0]] = "."
    reach = _reach(g, start)
    if _bad(reach):
        return chase_room(rng, eid)
    near = [c for c in reach if c not in DOORS and 3 <= LoopWorld.loop_dist(c, start) <= 5
            and g[c[1]][c[0]] == "."]
    if not near:
        return chase_room(rng, eid)
    x, y = rng.choice(near)
    return Room("chase", _rows(g), start, _got_away, find_creature(eid),
                creatures=[(eid, x, y)], kit=_kit_with(rng) if rng.random() < 0.5 else [],
                health=rng.randint(30, 60), stamina=rng.randint(40, 100), chase=True,
                about="chase %s" % REGISTRY.get_entity(eid).name)


KINDS = {"fight": fight_room, "food": food_room, "lever": lever_room,
         "shore": shore_room, "lava": lava_room, "chase": chase_room}


def pick_room(cause: str | None, rng: random.Random, avoid: str | None = None) -> Room:
    """Half the time a room about what killed it (drowned -> shore, a wolf ->
    a wolf fight, starved -> food, lava -> lava); otherwise any kind. Never
    the kind it was just in."""
    low = (cause or "").lower()
    if rng.random() < 0.5 and low:
        named = [e for e in FIGHTERS if REGISTRY.get_entity(e).name.lower() in low]
        kinds = [k for k in ("fight", "chase") if k != avoid]
        if named and kinds:             # fight it, or get away from it
            return KINDS[rng.choice(kinds)](rng, rng.choice(named))
        for kind, words in (("shore", ("drown", "shark")), ("food", ("starv",)),
                            ("lava", ("lava", "burn"))):
            if kind != avoid and any(w in low for w in words):
                return KINDS[kind](rng)
    return KINDS[rng.choice([k for k in KINDS if k != avoid] or list(KINDS))](rng)
