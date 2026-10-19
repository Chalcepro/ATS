"""Tile world, island-ocean architecture, and entity placement.

World topology is now driven by island_gen.py. All tiles outside island
land masks are TILE_OCEAN. Ocean has sharks (spawn after OCEAN_SHARK_TICKS).
Each island type has native entities, resources, and hazard tile types.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import config_rl
from entities import Entity
from island_gen import (
    IslandRegistry, IslandDef, build_island_registry,
    ISLAND_GRASSLAND, ISLAND_DARK_FOREST, ISLAND_VOLCANIC,
    ISLAND_MUSHROOM, ISLAND_TUNDRA, ISLAND_BOSS,
)
from item_ids import REGISTRY

# ---------------------------------------------------------------------------
# Tile type constants
# ---------------------------------------------------------------------------
TILE_EMPTY   = 0   # Open walkable ground
TILE_WALL    = 1   # Impassable rock/cliff
TILE_HOSTILE = 2   # Entity occupying tile (render override)
TILE_OBJECT  = 3   # Harvestable / item on ground
TILE_STONE   = 4   # Dungeon stone floor (W4 Loop rooms). Was TILE_NPC, which
                   # nothing ever drew, so its brain row is unused - no migration.
TILE_WATER   = 5   # Island-internal lake/pond (swimmable, no sharks)
TILE_LAVA    = 6   # Volcanic floor hazard (5 HP/tick burn)
TILE_GATE    = 7   # Progression gate (locked)
TILE_OCEAN   = 8   # Open ocean — sharks spawn after 3 ticks
TILE_ACID    = 9   # Mushroom island acid pool (applies POISONED)
TILE_ICE     = 10  # Tundra ice (passable, 20% slip = leg injury)
TILE_ASH     = 11  # Volcanic safe ash path (no burn damage)
TILE_SPORE   = 12  # Mushroom spore cloud zone (applies BLINDED)
TILE_BRIDGE  = 13  # One tile wide over open ocean; the only way to another island
TILE_SHARK   = 14  # A shark at the surface: what the brain sees of it (W2b)
TILE_HOUSE   = 15  # An empty house (W3): a place people will live in W5
TILE_SWITCH  = 16  # A lever (W4 Loop rooms): operated with interact, not carried


def is_switch(object_id) -> bool:
    item = REGISTRY.get_item(object_id) if object_id else None
    return bool(item) and item.get("type") == "switch"


@dataclass
class Tile:
    tile_type: int  = TILE_EMPTY
    z: int          = 0
    object_id: str  = ""
    explored: bool  = False
    island_id: int  = -1   # which island this tile belongs to (-1 = ocean)
    flags: str      = ""
    drop_item: str  = ""


# ---------------------------------------------------------------------------
# Native entity pools per island
# ---------------------------------------------------------------------------
GRID         = 8      # cell size for creatures_near

# ---- W3: where things grow (2026-10-13, Chalce's rules) ----------------------
# Trees grow in groves from their own smooth noise. Everything else grows in
# PATCHES: a patch centre where its noise says the land suits it, patches of
# one kind at least PATCH_GAP apart, each a bunch or a sparse few, then thinned
# so a bunch is usually 1-2 left and never 0. So walking a straight line
# doesn't keep finding apples: after one, you have to look for the next.
TREES        = {"0021", "0022"}
PATCH_GAP    = {"food": 14, "other": 9}   # min tiles between patches of a kind
FERTILE      = 0.55     # noise above this: the land suits it
BUNCH_CHANCE = 0.4      # else sparse (1-2)
KEEP_CHANCE  = 0.4      # the thinning pass ("heat box"): each item kept with this
GROVE_LEVEL  = 0.66     # tree noise above this is a grove...
GROVE_FILL   = 0.45     # ...and this share of a grove's tiles hold a tree
# Villages: no house, no village. A first house is rare; each house makes the
# next one more likely. Houses are empty for now; man-made things (bread...)
# are only found around them.
VILLAGE_CELL   = 40     # one chance of a first house per 40x40 of land
FIRST_HOUSE    = 0.35
NEXT_HOUSE     = (0.35, 0.10, 0.85)   # chance of house n+1 = min(c, a + b*n)
MAX_HOUSES     = 10
MAN_MADE       = [("0032", 0.5), ("0035", 0.2), ("0005", 0.15), ("0003", 0.15)]
MAN_MADE_PER_HOUSE = 0.7
REGROW_EVERY   = 25     # ticks between regrowth checks

# ---- W3b: the land changes during an episode (2026-10-13, Chalce) ------------
# Every SPREAD_EVERY ticks each built island makes one small change, never
# where the agent can see. Groves spread and die back; lava and acid move
# around inside their island without growing, burning what they cover; the
# volcanic island grows new land into the sea and the tundra an ice shelf,
# each up to a reach set by the island's size. A new world each episode
# starts it all over.
SPREAD_EVERY = 500
SPREAD_STEP  = 4                # tiles in one change
SPREAD_GAP   = 3                # grown land keeps this much sea to other land / bridges
GROVE_RANGE  = (0.75, 1.25)     # trees stay within this share of the start
LAND_REACH   = {ISLAND_VOLCANIC: (TILE_EMPTY, 3, 8),   # tile, reach on a small..big island
                ISLAND_TUNDRA:   (TILE_ICE, 3, 20)}
MOVING_HAZARD = {ISLAND_VOLCANIC: TILE_LAVA, ISLAND_MUSHROOM: TILE_ACID}
BIRTH_EVERY  = 25     # ticks between birth checks
BIRTH_CHANCE = 0.25   # per missing creature per check: ~1 birth per 100 ticks

_ISLAND_ENTITY_POOLS = {
    ISLAND_GRASSLAND:   ["E001", "E015", "E016", "E017", "E018"],
    ISLAND_DARK_FOREST: ["E004", "E005", "E008", "E011", "E013", "E028", "E015"],
    ISLAND_VOLCANIC:    ["E007", "E022", "E026"],
    ISLAND_MUSHROOM:    ["E010", "E023", "E024"],
    ISLAND_TUNDRA:      ["E012", "E021", "E025", "E016"],
    ISLAND_BOSS:        ["E009", "E014", "E020"],
}

# Objects native to each island type
# Each entry: (object_id, spawn_chance_threshold)
_ISLAND_OBJECT_POOLS = {
    ISLAND_GRASSLAND: [
        ("0021", 0.09),   # Oak Tree
        ("0023", 0.16),   # Berry Bush
        ("0024", 0.22),   # Fallen Log
        ("0008", 0.27),   # Stick
        ("0012", 0.31),   # Stone
        ("0028", 0.35),   # Mushroom
        ("0029", 0.39),   # Flower
        ("0001", 0.43),   # Apple
        ("0025", 0.47),   # Boulder
    ],
    ISLAND_DARK_FOREST: [
        ("0022", 0.10),   # Pine Tree
        ("0024", 0.18),   # Fallen Log
        ("0028", 0.26),   # Mushroom (dark)
        ("0008", 0.31),   # Stick
        ("0012", 0.36),   # Stone
        ("0044", 0.39),   # Dark Mushroom
        ("0025", 0.43),   # Boulder
    ],
    ISLAND_VOLCANIC: [
        ("0082", 0.10),   # Lava Rock -> Magma Stone
        ("0083", 0.18),   # Obsidian Node -> Obsidian Shard
        ("0054", 0.26),   # Coal Ore Rich -> Coal
        ("0086", 0.31),   # Fire Bloom Bush -> Fire Blossom
        ("0025", 0.35),   # Boulder
        ("0052", 0.39),   # Volcanic Ash
    ],
    ISLAND_MUSHROOM: [
        ("0080", 0.12),   # Mushroom Grove -> Spore
        ("0085", 0.20),   # Root Herb -> Antidote Root
        ("0087", 0.27),   # Glowshroom Cluster -> Glowshroom
        ("0059", 0.32),   # Mycelium Cap (raw)
        ("0025", 0.36),   # Boulder
        ("0028", 0.40),   # Mushroom
    ],
    ISLAND_TUNDRA: [
        ("0070", 0.10),   # Frost Ore node -> Ice Crystal
        ("0084", 0.18),   # Ice Stalagmite -> Ice Crystal
        ("0022", 0.25),   # Pine Tree
        ("0024", 0.31),   # Fallen Log
        ("0025", 0.36),   # Boulder
        ("0012", 0.40),   # Stone
    ],
    ISLAND_BOSS: [
        ("0025", 0.10),   # Boulder
        ("0020", 0.18),   # Void Shard (old)
        ("0055", 0.22),   # Void Shard Rare
        ("0012", 0.28),   # Stone
    ],
}



def _hash01(x: int, y: int, salt: int) -> float:
    """A fixed pseudo-random number in [0, 1) for a lattice point."""
    h = (x * 374761393 + y * 668265263 + salt * 0x9E3779B9) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(1 << 24)


def _noise(x: int, y: int, salt: int, scale: int) -> float:
    """Smooth value noise in [0, 1): neighbours get similar values, so things
    clump into areas instead of being sprinkled evenly."""
    gx, gy = x // scale, y // scale
    fx, fy = (x % scale) / scale, (y % scale) / scale
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash01(gx, gy, salt), _hash01(gx + 1, gy, salt)
    c, d = _hash01(gx, gy + 1, salt), _hash01(gx + 1, gy + 1, salt)
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy


def _is_food(obj_id: str) -> bool:
    """Food = you eat it, or harvesting it gives you something you eat."""
    info = REGISTRY.get_item(obj_id) or {}
    if info.get("type") == "consumable":
        return True
    drop = REGISTRY.get_item(info.get("drop", "")) or {}
    return drop.get("type") == "consumable"

class World:
    """Island-ocean tile world. All generation driven by IslandRegistry."""

    def __init__(self, seed: int = 1337):
        self.seed = seed
        self.tiles: dict[tuple[int, int], Tile] = {}
        self.entities: list[Entity] = []
        # How many of each creature each island started with; births refill
        # toward it, so an island never empties out (W2).
        self._start_counts: dict[int, dict[str, int]] = {}
        self._life_ticks = 0
        self._grid: dict[tuple[int, int], list] = {}
        self.villages: list[list] = []          # W3: each village's houses
        self._regrow: list[tuple] = []          # (due tick, x, y, object id)
        self._reach: dict = {}                  # W3b: grown tile -> tiles past the first coast
        self._trees0: dict = {}                 # W3b: island -> trees at the first change
        self._size0: dict = {}                  # W3b: island -> radius of its first land
        self.door_unlocked = True
        self.agent_spawn = (0, 0)

        # Build island registry (generates all 6 islands)
        self.island_registry: IslandRegistry = build_island_registry(
            seed, bridges=getattr(config_rl, "WORLD_BRIDGES", True))
        self._materialised: set[int] = set()
        self.starting_island: IslandDef = self.island_registry.get_starting_island()
        self.agent_spawn = self.starting_island.spawn_pos

        # Pre-generate all tiles for the starting island only
        self._generate_island_tiles(self.starting_island)
        self._materialised.add(self.starting_island.island_id)

        # Ensure spawn tile is clear
        spawn_tile = self._tile(*self.agent_spawn)
        spawn_tile.tile_type = TILE_EMPTY
        spawn_tile.object_id = ""

        # Guarantee starter items around spawn
        self._place_starter_surroundings()

        # Spawn entities for starting island
        self._populate_island_entities(self.starting_island)

        # Bridges and the guard standing on each. The far islands are only
        # built for real when the agent first steps onto a bridge to them -
        # every island's creatures ticking every step would slow training to
        # a crawl for land the agent may never reach.
        self._lay_bridges()

    # ------------------------------------------------------------------
    # Island tile generation
    # ------------------------------------------------------------------
    def _generate_island_tiles(self, idef: IslandDef) -> None:
        """Materialise all tiles for a given island definition."""
        cx, cy = idef.center

        for (tx, ty) in idef.land_tiles:
            tile_type = TILE_EMPTY

            if (tx, ty) in idef.internal_hazard:
                if idef.island_id == ISLAND_VOLCANIC:
                    tile_type = TILE_LAVA
                elif idef.island_id == ISLAND_MUSHROOM:
                    tile_type = TILE_ACID
                elif idef.island_id == ISLAND_TUNDRA:
                    tile_type = TILE_ICE
                else:
                    tile_type = TILE_LAVA

            elif (tx, ty) in idef.internal_water:
                tile_type = TILE_WATER

            elif (tx, ty) in idef.safe_path:
                tile_type = TILE_ASH

            elif (tx, ty) in idef.spore_zones:
                tile_type = TILE_SPORE

            elif (tx, ty) in idef.obstacle_tiles:
                tile_type = TILE_WALL

            t = Tile(tile_type=tile_type, island_id=idef.island_id)
            self.tiles[(tx, ty)] = t

            # Place objects on empty tiles only (not on hazard/wall/water)
            if tile_type == TILE_EMPTY and not (tx == cx and ty == cy) \
                    and not getattr(config_rl, "W3_PATCHES", True):
                dist = abs(tx - cx) + abs(ty - cy)
                if dist > 3:  # don't clutter spawn center
                    self._populate_tile_objects(t, tx, ty, idef)

        if getattr(config_rl, "W3_PATCHES", True):
            self._place_island_objects(idef)

    def _populate_tile_objects(self, tile: Tile, tx: int, ty: int, idef: IslandDef) -> None:
        """Deterministically place island-native resources on tile."""
        h = (tx * 374761393 + ty * 668265263 + self.seed * 0x9E3779B9) & 0xFFFFFFFF
        h ^= (h >> 13)
        h ^= (h << 7) & 0xFFFFFFFF
        h ^= (h >> 17)
        rand_val = (h % 10000) / 10000.0

        pool = _ISLAND_OBJECT_POOLS.get(idef.island_id, [])
        cumulative = 0.0
        for (obj_id, threshold) in pool:
            if rand_val < threshold:
                tile.object_id  = obj_id
                tile.tile_type  = TILE_OBJECT
                break

    # ------------------------------------------------------------------
    # W3: groves, patches, villages
    # ------------------------------------------------------------------
    def _free(self, p, cx, cy) -> bool:
        t = self.tiles.get(p)
        return (t is not None and t.tile_type == TILE_EMPTY and not t.object_id
                and abs(p[0] - cx) + abs(p[1] - cy) > 3)

    def _put(self, p, obj_id) -> None:
        t = self.tiles[p]
        t.object_id = obj_id
        t.tile_type = TILE_OBJECT

    def _place_island_objects(self, idef: IslandDef) -> None:
        cx, cy = idef.center
        land = sorted(idef.land_tiles)
        rng = random.Random(self.seed * 7919 + idef.island_id)
        pool = _ISLAND_OBJECT_POOLS.get(idef.island_id, [])
        self._place_villages(idef, land, rng)

        for k, (obj_id, _) in enumerate(pool):
            salt = self.seed * 131 + idef.island_id * 17 + k
            if obj_id in TREES:
                for p in land:
                    if (_noise(p[0], p[1], salt, 9) > GROVE_LEVEL and self._free(p, cx, cy)
                            and rng.random() < GROVE_FILL):
                        self._put(p, obj_id)
                continue
            gap = PATCH_GAP["food" if _is_food(obj_id) else "other"]
            centres: dict = {}                      # gap cell -> centres in it
            for p in land:
                # one candidate per gap x gap cell, picked by hash
                if _hash01(p[0] // gap, p[1] // gap, salt) * gap * gap >= 1 + \
                        (p[0] % gap) * gap + (p[1] % gap):
                    continue
                if _noise(p[0], p[1], salt + 1, 12) < FERTILE or not self._free(p, cx, cy):
                    continue
                cell = (p[0] // gap, p[1] // gap)
                if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < gap
                       for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for q in centres.get((cell[0] + i, cell[1] + j), ())):
                    continue
                centres.setdefault(cell, []).append(p)
                n = rng.randint(3, 6) if rng.random() < BUNCH_CHANCE else rng.randint(1, 2)
                spots = [(p[0] + dx, p[1] + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
                rng.shuffle(spots)
                spots = [p] + [s for s in spots if s != p]
                placed = [s for s in spots if self._free(s, cx, cy)][:n]
                kept = [s for s in placed if rng.random() < KEEP_CHANCE] or placed[:1]
                for s in kept:
                    self._put(s, obj_id)

    def _place_villages(self, idef: IslandDef, land: list, rng) -> None:
        cx, cy = idef.center
        cells: dict = {}
        for p in land:
            cells.setdefault((p[0] // VILLAGE_CELL, p[1] // VILLAGE_CELL), []).append(p)
        for _, tiles in sorted(cells.items()):
            if len(tiles) < 200 or rng.random() >= FIRST_HOUSE:
                continue
            houses = []
            spot = rng.choice(tiles)
            while spot is not None:
                if self._house_fits(spot, cx, cy):
                    houses.append(spot)
                    for dx in (0, 1):
                        for dy in (0, 1):
                            t = self.tiles[(spot[0] + dx, spot[1] + dy)]
                            t.tile_type, t.object_id = TILE_HOUSE, ""
                a, b, c = NEXT_HOUSE
                if not houses or len(houses) >= MAX_HOUSES                         or rng.random() >= min(c, a + b * len(houses)):
                    break
                base = rng.choice(houses)
                spot = (base[0] + rng.choice((-5, -4, 4, 5)), base[1] + rng.randint(-4, 4))
            for hx, hy in houses:
                if rng.random() < MAN_MADE_PER_HOUSE:
                    ids, weights = zip(*MAN_MADE)
                    item = rng.choices(ids, weights)[0]
                    near = [(hx + dx, hy + dy) for dx in range(-2, 4) for dy in range(-2, 4)]
                    rng.shuffle(near)
                    for s in near:
                        if self._free(s, cx, cy):
                            self._put(s, item)
                            break
            if houses:
                self.villages.append(houses)

    def _house_fits(self, spot, cx, cy) -> bool:
        if abs(spot[0] - cx) + abs(spot[1] - cy) < 8:
            return False
        return all(self._free((spot[0] + dx, spot[1] + dy), cx, cy)
                   for dx in range(-1, 3) for dy in range(-1, 3))

    def _populate_island_entities(self, idef: IslandDef) -> None:
        """Spawn entities strictly native to this island type."""
        pool = _ISLAND_ENTITY_POOLS.get(idef.island_id, ["E001"])
        cx, cy = idef.center
        land_list = list(idef.land_tiles)
        if not land_list:
            return

        # Spawn 1 entity per ~40 land tiles, minimum 3
        n_entities = max(3, len(land_list) // 40)
        rng = random.Random(self.seed + idef.island_id * 999983)

        for i in range(n_entities):
            tx, ty = rng.choice(land_list)
            dist = abs(tx - cx) + abs(ty - cy)
            if dist < 8:
                continue  # no hostile spawn in safe zone
            tile = self.tiles.get((tx, ty))
            if not tile or tile.tile_type in (TILE_WALL, TILE_LAVA, TILE_ACID, TILE_WATER):
                continue
            ent_id = pool[i % len(pool)]
            self.entities.append(Entity.from_id(ent_id, tx, ty))
            counts = self._start_counts.setdefault(idef.island_id, {})
            counts[ent_id] = counts.get(ent_id, 0) + 1

    def index_creatures(self) -> None:
        """Sort living creatures into GRID x GRID cells, once per tick, so a
        creature looking around checks nearby cells, not every creature."""
        grid: dict[tuple[int, int], list] = {}
        for e in self.entities:
            if e.alive:
                grid.setdefault((e.x // GRID, e.y // GRID), []).append(e)
        self._grid = grid

    def creatures_near(self, x: int, y: int, r: int) -> list:
        """Creatures in the cells within r of (x, y), as of the last index
        (they may have moved a tile since; callers check distance)."""
        out = []
        for cx in range((x - r) // GRID, (x + r) // GRID + 1):
            for cy in range((y - r) // GRID, (y + r) // GRID + 1):
                out.extend(self._grid.get((cx, cy), ()))
        return out

    def took(self, x: int, y: int, obj_id: str) -> None:
        """Something was taken from (x, y). If it grows back (items.json
        "regrow"), it will - at the same spot."""
        info = REGISTRY.get_item(obj_id) or {}
        if info.get("regrow"):
            self._regrow.append((self._life_ticks + int(info["regrow"]), x, y, obj_id))

    def _tick_regrow(self, ax: int, ay: int) -> None:
        """Due things grow back, but never where the agent can see: it never
        watches food appear in front of it."""
        r = config_rl.PATCH_RADIUS
        left = []
        for due, x, y, obj_id in self._regrow:
            t = self.tiles.get((x, y))
            if due > self._life_ticks or max(abs(x - ax), abs(y - ay)) <= r:
                left.append((due, x, y, obj_id))
            elif t is not None and t.tile_type == TILE_EMPTY and not t.object_id \
                    and self.entity_at(x, y) is None:
                t.object_id, t.tile_type = obj_id, TILE_OBJECT
            else:
                left.append((due, x, y, obj_id))   # something's on it: try later
        self._regrow = left

    # ------------------------------------------------------------------
    # W3b: the land changes
    # ------------------------------------------------------------------
    def _tick_spread(self, ax: int, ay: int) -> None:
        """One small change on every built island, out of the agent's sight."""
        r = config_rl.PATCH_RADIUS
        hidden = lambda p: max(abs(p[0] - ax), abs(p[1] - ay)) > r
        for iid in sorted(self._materialised):
            idef = self.island_registry.get_island(iid)
            land = list(idef.land_tiles)
            self._spread_groves(idef, land, hidden)
            if iid in MOVING_HAZARD:
                self._shift_hazard(idef, MOVING_HAZARD[iid], hidden)
            if iid in LAND_REACH:
                self._grow_coast(idef, land, hidden)

    def _spread_groves(self, idef: IslandDef, land: list, hidden) -> None:
        cx, cy = idef.center
        trees = [p for p in land if self.tiles[p].object_id in TREES]
        start = self._trees0.setdefault(idef.island_id, len(trees))
        if not trees:
            return
        for _ in range(SPREAD_STEP):
            t = random.choice(trees)
            if len(trees) < start * GROVE_RANGE[1] and (
                    len(trees) <= start * GROVE_RANGE[0] or random.random() < 0.5):
                p = (t[0] + random.randint(-2, 2), t[1] + random.randint(-2, 2))
                if self._free(p, cx, cy) and hidden(p) and self.entity_at(*p) is None:
                    self._put(p, self.tiles[t].object_id)
                    trees.append(p)
            elif hidden(t):
                tile = self.tiles[t]
                tile.object_id, tile.tile_type = "", TILE_EMPTY
                trees.remove(t)

    def _shift_hazard(self, idef: IslandDef, hazard: int, hidden) -> None:
        """Lava / acid moves a tile at a time: the same amount, somewhere new."""
        cx, cy = idef.center
        pool = [p for p in idef.internal_hazard if self.tiles[p].tile_type == hazard]
        for _ in range(SPREAD_STEP * 5):           # tries, most picks are interior
            if not pool:
                return
            h = random.choice(pool)
            d = random.choice(((0, 1), (0, -1), (1, 0), (-1, 0)))
            n = (h[0] + d[0], h[1] + d[1])
            t = self.tiles.get(n)
            if (t is None or t.tile_type not in (TILE_EMPTY, TILE_OBJECT)
                    or n in idef.safe_path or abs(n[0] - cx) + abs(n[1] - cy) <= 3
                    or not (hidden(h) and hidden(n))):
                continue
            self.tiles[h].tile_type = TILE_EMPTY
            idef.internal_hazard.discard(h)
            pool.remove(h)
            self.cover(n, hazard)
            if self.tiles[n].tile_type == hazard:    # else a key lies there: flows round it
                idef.internal_hazard.add(n)
                pool.append(n)

    def cover(self, p, hazard: int) -> None:
        """Lava or acid flows onto p: any common thing there is gone, and any
        creature, alive or dead. Keys and gate items survive (fireproof)."""
        t = self.tiles[p]
        info = REGISTRY.get_item(t.object_id) or {}
        if t.object_id and not (info.get("fireproof") or info.get("gate_key")):
            t.object_id = ""
            self._regrow = [g for g in self._regrow if (g[1], g[2]) != p]
        for e in self.entities:
            if e.alive and (e.x, e.y) == p and not e.guard:
                e.alive = False
        t.tile_type = TILE_OBJECT if t.object_id else hazard

    def _grow_coast(self, idef: IslandDef, land: list, hidden) -> None:
        """New land (volcanic) or an ice shelf (tundra) grows into the sea, up
        to a reach that's bigger for a bigger island."""
        tile_type, small, big = LAND_REACH[idef.island_id]
        # size from the land it started with (idef.radius isn't scaled by
        # LAND_SCALE in the bridged layout): ~55 tiles across .. ~95
        r0 = self._size0.setdefault(idef.island_id, (len(land) / 3.1416) ** 0.5)
        reach = small + (big - small) * min(1.0, max(0.0, (r0 - 55) / 40))
        reg = self.island_registry
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        for p in random.sample(land, min(len(land), 400)):
            if self._reach.get(p, 0) >= reach:
                continue
            sea = [(p[0] + dx, p[1] + dy) for dx, dy in dirs if self._open_sea(
                (p[0] + dx, p[1] + dy), idef.island_id, hidden)]
            if sea:
                break
        else:
            return
        grown, at = 0, p
        while grown < SPREAD_STEP and sea:
            n = random.choice(sea)
            self.tiles[n] = Tile(tile_type=tile_type, island_id=idef.island_id)
            idef.land_tiles.add(n)
            reg._tile_map[n] = idef.island_id
            self._reach[n] = self._reach.get(at, 0) + 1
            grown += 1
            # keep going sideways along the coast, not out in a spike
            at = n if self._reach[n] < reach else at
            sea = [(at[0] + dx, at[1] + dy) for dx, dy in dirs if self._open_sea(
                (at[0] + dx, at[1] + dy), idef.island_id, hidden)]
            sea = [q for q in sea if self._reach.get(at, 0) + 1 <= reach]

    def _open_sea(self, p, island_id: int, hidden) -> bool:
        """Sea that may become land: no creature on it, out of sight, and
        SPREAD_GAP of sea kept to any other island and to bridges."""
        t = self.tiles.get(p)
        if p in self.island_registry._tile_map or (t and t.tile_type != TILE_OCEAN):
            return False
        if not hidden(p) or self.entity_at(*p) is not None:
            return False
        g = SPREAD_GAP
        for x in range(p[0] - g, p[0] + g + 1):
            for y in range(p[1] - g, p[1] + g + 1):
                owner = self.island_registry._tile_map.get((x, y))
                if owner is not None and owner != island_id:
                    return False
                q = self.tiles.get((x, y))
                if q is not None and q.tile_type == TILE_BRIDGE:
                    return False
        return True

    def tick_life(self, ax: int = 1 << 30, ay: int = 1 << 30) -> list:
        """Births, every BIRTH_EVERY ticks: a creature below its island's
        starting count may be born beside an adult of its kind (or, if none
        is left, somewhere on its island away from the centre). The dead are
        cleared out at the same time. Returns the newborns. Regrowth too."""
        self._life_ticks += 1
        if self._regrow and self._life_ticks % REGROW_EVERY == 0:
            self._tick_regrow(ax, ay)
        if self._life_ticks % SPREAD_EVERY == 0 and getattr(config_rl, "W3_SPREAD", True):
            self._tick_spread(ax, ay)
        if self._life_ticks % BIRTH_EVERY:
            return []
        self.entities = [e for e in self.entities if e.alive]
        alive: dict[tuple[int, str], list] = {}
        for e in self.entities:
            if not e.guard:
                t = self.tiles.get((e.x, e.y))
                alive.setdefault((t.island_id if t else -1, e.entity_id), []).append(e)
        born = []
        for island_id, counts in self._start_counts.items():
            for eid, want in counts.items():
                adults = alive.get((island_id, eid), [])
                if len(adults) >= want or random.random() >= BIRTH_CHANCE:
                    continue
                pos = self._birth_spot(island_id, adults)
                if pos:
                    born.append(Entity.from_id(eid, *pos))
        self.entities.extend(born)
        return born

    def _birth_spot(self, island_id: int, adults: list):
        bad = (TILE_WALL, TILE_LAVA, TILE_ACID, TILE_WATER)   # as at the first spawn
        if adults:
            mum = random.choice(adults)
            spots = [(mum.x + dx, mum.y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
        else:
            idef = self.island_registry.get_island(island_id)
            cx, cy = idef.center
            spots = [p for p in random.sample(list(idef.land_tiles), min(20, len(idef.land_tiles)))
                     if abs(p[0] - cx) + abs(p[1] - cy) >= 8]
        for p in spots:
            t = self.tiles.get(p)
            if t and t.tile_type not in bad and self.entity_at(*p) is None:
                return p
        return None

    def _lay_bridges(self) -> None:
        reg = self.island_registry
        for b in getattr(reg, "bridges", []):
            for (x, y) in b["tiles"]:
                self.tiles[(x, y)] = Tile(tile_type=TILE_BRIDGE, island_id=-1,
                                          flags="bridge:%d" % b["to"])
            post = b.get("guard")
            if post is None:
                continue
            guard_id = self._guard_for(b["to"])
            if guard_id:
                g = Entity.from_id(guard_id, *post)
                g.guard = True
                self.entities.append(g)

    def _guard_for(self, island_id: int) -> str:
        """A fighter from the island the bridge leads to - the first hostile
        in its native pool, so a harder island has a harder guard."""
        for eid in _ISLAND_ENTITY_POOLS.get(island_id, []):
            spec = REGISTRY.get_entity(eid)
            if spec and spec.has("hostile") and not spec.has("unkillable"):
                return eid
        return "E001"

    def materialise(self, island_id: int) -> bool:
        """Build an island for real - its tiles, resources and creatures -
        the first time the agent heads for it. False if it already was."""
        if island_id in self._materialised:
            return False
        idef = self.island_registry.get_island(island_id)
        if idef is None:
            return False
        self._generate_island_tiles(idef)
        self._populate_island_entities(idef)
        self._materialised.add(island_id)
        return True

    def bridge_target(self, x: int, y: int) -> int:
        """The island the bridge at (x, y) leads to, or -1 if not a bridge."""
        t = self.tiles.get((x, y))
        if t is None or t.tile_type != TILE_BRIDGE or not t.flags.startswith("bridge:"):
            return -1
        return int(t.flags.split(":", 1)[1])

    def _place_starter_surroundings(self) -> None:
        """Place immediate items around spawn point."""
        cx, cy = self.agent_spawn
        starters = [
            (cx+1, cy,   "0001"),   # Apple
            (cx-1, cy,   "0023"),   # Berry Bush
            (cx,   cy+1, "0008"),   # Stick
            (cx,   cy-1, "0010"),   # Wood
            (cx+1, cy+1, "0021"),   # Oak Tree
            (cx-1, cy-1, "0012"),   # Stone
            (cx+2, cy,   "0024"),   # Fallen Log
            (cx-2, cy+1, "0023"),   # Berry Bush
            (cx+1, cy+2, "0009"),   # Coal
        ]
        for sx, sy, obj_id in starters:
            t = self.tiles.get((sx, sy))
            if t and t.tile_type not in (TILE_WALL, TILE_LAVA, TILE_ACID):
                t.object_id = obj_id
                t.tile_type = TILE_OBJECT

    # ------------------------------------------------------------------
    # Tile access (lazy ocean for ungenerated tiles)
    # ------------------------------------------------------------------
    def _tile(self, x: int, y: int) -> Tile:
        if (x, y) not in self.tiles:
            # All ungenerated tiles are ocean
            island_id = self.island_registry.get_island_at(x, y)
            if island_id is not None:
                # This tile belongs to an island not yet materialized
                # Generate it on demand (for future islands)
                t = Tile(tile_type=TILE_EMPTY, island_id=island_id)
            else:
                t = Tile(tile_type=TILE_OCEAN, island_id=-1)
            self.tiles[(x, y)] = t
        return self.tiles[(x, y)]

    # ------------------------------------------------------------------
    # Island / zone info
    # ------------------------------------------------------------------
    def is_ocean(self, x: int, y: int) -> bool:
        tile = self._tile(x, y)
        return tile.tile_type == TILE_OCEAN

    def is_safe_zone(self, x: int, y: int) -> bool:
        """Safe haven: within 6 tiles of the starting island spawn."""
        cx, cy = self.starting_island.spawn_pos
        return abs(x - cx) + abs(y - cy) <= 6

    def get_zone_info(self, x: int, y: int) -> dict:
        island_id = self.island_registry.get_island_at(x, y)
        if island_id is None:
            return {"id": -1, "tier": -1, "name": "Ocean", "safe": False, "danger": 1.0}
        idef = self.island_registry.get_island(island_id)
        return {
            "id": island_id,
            "tier": idef.tier,
            "name": idef.name,
            "safe": self.is_safe_zone(x, y),
            "danger": idef.tier / 4.0,
        }

    def get_island_id_at(self, x: int, y: int) -> int:
        return self.island_registry.get_island_at(x, y) if not self.is_ocean(x, y) else -1

    def get_island_tier_at(self, x: int, y: int) -> int:
        island_id = self.island_registry.get_island_at(x, y)
        if island_id is None:
            return -1
        idef = self.island_registry.get_island(island_id)
        return idef.tier if idef else -1

    # ------------------------------------------------------------------
    # Passability
    # ------------------------------------------------------------------
    def in_bounds_passable(self, x: int, y: int, door_unlocked: bool = True) -> bool:
        tile = self._tile(x, y)
        if not tile:
            return False
        # Walls, lava, and gates are impassable movement
        if tile.tile_type in (TILE_WALL, TILE_GATE, TILE_HOUSE):
            return False
        # Lava — passable but very damaging (handled in env tick)
        # Ocean — passable but shark spawns (handled in env tick)
        # Ice / acid / spore — passable but apply status effects
        # Block movement through alive hostile entities
        for ent in self.entities:
            if ent.alive and ent.x == x and ent.y == y:
                if ent.spec.has("hostile"):
                    return False
        return True

    # ------------------------------------------------------------------
    # Gate system (kept for Boss Sanctum gate)
    # ------------------------------------------------------------------
    def try_unlock_gate(self, x: int, y: int, capabilities: set, inventory_item_ids: set) -> Optional[str]:
        tile = self._tile(x, y)
        if not tile or tile.tile_type != TILE_GATE:
            return None
        for gate_id, gdata in self.gates.items() if hasattr(self, "gates") else []:
            if gdata["pos"] == (x, y) and not gdata["unlocked"]:
                req_cap  = gdata.get("req_cap")
                req_item = gdata.get("req_item")
                if (req_cap and req_cap in capabilities) or (req_item and req_item in inventory_item_ids):
                    gdata["unlocked"] = True
                    tile.tile_type = TILE_EMPTY
                    tile.flags = f"GATE_UNLOCKED:{gate_id}"
                    return gate_id
        return None

    # ------------------------------------------------------------------
    # Shark spawn (called from ats_env after ocean_ticks threshold)
    # ------------------------------------------------------------------
    def spawn_shark(self, x: int, y: int) -> None:
        self.entities.append(Entity.from_id("E_SHARK", x, y))

    def surface_shark(self, ax: int, ay: int, r: int):
        """One shark for a swimmer: surfaces on open sea inside the agent's
        view, as far out as the view allows, so it is seen before it moves.
        Returns the new shark, or None if one is already out (or no open
        sea is in view - then it tries again next tick)."""
        for e in self.entities:
            if e.alive and e.spec.has("ocean"):
                return None
        for ring in range(r, 0, -1):
            spots = [(ax + dx, ay + dy)
                     for dx in range(-ring, ring + 1) for dy in range(-ring, ring + 1)
                     if max(abs(dx), abs(dy)) == ring]
            random.shuffle(spots)
            for x, y in spots:
                t = self._tile(x, y)
                if t and t.tile_type == TILE_OCEAN and self.entity_at(x, y) is None:
                    shark = Entity.from_id("E_SHARK", x, y)
                    self.entities.append(shark)
                    return shark
        return None

    # ------------------------------------------------------------------
    # Drop spawn
    # ------------------------------------------------------------------
    def spawn_drop(self, x: int, y: int, item_id: str) -> None:
        if not item_id:
            return
        candidates = [(x, y)] + [(x+dx, y+dy) for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]]
        for cx, cy in candidates:
            tile = self._tile(cx, cy)
            if tile and tile.tile_type not in (TILE_WALL, TILE_LAVA, TILE_ACID) and not tile.object_id:
                tile.object_id = item_id
                tile.tile_type = TILE_OBJECT
                return
        tile = self._tile(x, y)
        if tile:
            tile.object_id = item_id
            tile.tile_type = TILE_OBJECT

    # ------------------------------------------------------------------
    # Exploration fog
    # ------------------------------------------------------------------
    def mark_explored(self, x: int, y: int, radius: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    t = self._tile(x + dx, y + dy)
                    if t:
                        t.explored = True

    # ------------------------------------------------------------------
    # Entity queries
    # ------------------------------------------------------------------
    def nearest_entity(self, x: int, y: int):
        best, best_dist = None, 999
        for ent in self.entities:
            if not ent.alive:
                continue
            d = abs(ent.x - x) + abs(ent.y - y)
            if d < best_dist:
                best, best_dist = ent, d
        return best, best_dist

    def nearest_entity_chebyshev(self, x: int, y: int):
        best, best_dist = None, 999
        for ent in self.entities:
            if not ent.alive:
                continue
            d = max(abs(ent.x - x), abs(ent.y - y))
            if d < best_dist:
                best, best_dist = ent, d
        return best, best_dist

    def entity_at(self, x: int, y: int) -> Optional[Entity]:
        for ent in self.entities:
            if ent.alive and ent.x == x and ent.y == y:
                return ent
        return None

    # ------------------------------------------------------------------
    # Object queries
    # ------------------------------------------------------------------
    def nearest_object(self, x: int, y: int):
        """Return ``(object_id, manhattan_dist, pos)``.

        ``pos`` is the (x, y) of the nearest object or None — the agent's
        state vector needs the *direction* to it, not just the distance.
        """
        best_id, best_dist, best_pos = "", 999, None
        for (tx, ty), tile in self.tiles.items():
            if tile.object_id:
                d = abs(tx - x) + abs(ty - y)
                if d < best_dist:
                    best_id, best_dist, best_pos = tile.object_id, d, (tx, ty)
        return best_id, best_dist, best_pos

    # ------------------------------------------------------------------
    # Height map helpers (stub — elevation not tile-based in island arch)
    # ------------------------------------------------------------------
    def get_adjacent_z(self, x: int, y: int) -> dict:
        return {"ahead": 0, "behind": 0, "left": 0, "right": 0}

    # ------------------------------------------------------------------
    # Tile type at position
    # ------------------------------------------------------------------
    def tile_type_at(self, x: int, y: int) -> int:
        for e in self.entities:
            if e.alive and e.x == x and e.y == y:
                return TILE_SHARK if e.spec.has("ocean") else TILE_HOSTILE
        tile = self._tile(x, y)
        if not tile:
            return TILE_OCEAN
        if tile.object_id:
            if getattr(config_rl, "LEVERS_ARE_LEVERS", False) and is_switch(tile.object_id):
                return TILE_SWITCH
            return TILE_OBJECT
        return tile.tile_type

    # ------------------------------------------------------------------
    # Pick up
    # ------------------------------------------------------------------
    def pickup_at(self, x: int, y: int) -> str:
        tile = self._tile(x, y)
        if not tile or not tile.object_id:
            return ""
        item    = tile.object_id
        self.took(x, y, item)
        reg_item = REGISTRY.get_item(item)
        drop_yield = item
        if reg_item and reg_item.get("drop"):
            drop_yield = reg_item["drop"]
        tile.object_id = ""
        if tile.tile_type == TILE_OBJECT:
            tile.tile_type = TILE_EMPTY
        return drop_yield

    # ------------------------------------------------------------------
    # Attack / Harvest
    # ------------------------------------------------------------------
    def attack_at(self, x: int, y: int, damage: int,
                  attacker_pos=None, facing_dx: int = 0, facing_dy: int = -1):
        fx, fy = x + facing_dx, y + facing_dy
        ent_facing = self.entity_at(fx, fy)
        if ent_facing and ent_facing.alive and not ent_facing.spec.has("ocean"):
            ent_facing.take_damage(damage, attacker_pos=attacker_pos)
            if not ent_facing.alive:
                return ent_facing.entity_id, ent_facing.drop_item_id(), False
            return "", "", False

        for ent in self.entities:
            if ent.alive and max(abs(ent.x - x), abs(ent.y - y)) <= 1:
                if ent.spec.has("ocean"):
                    continue
                ent.take_damage(damage, attacker_pos=attacker_pos)
                if not ent.alive:
                    return ent.entity_id, ent.drop_item_id(), False
                return "", "", False

        for tx, ty in [(fx, fy), (x, y)]:
            tile = self._tile(tx, ty)
            if tile and tile.object_id:
                obj_id    = tile.object_id
                item_info = REGISTRY.get_item(obj_id)
                if item_info and item_info.get("type") == "world_object":
                    self.took(tx, ty, obj_id)
                    drop = item_info.get("drop", obj_id)
                    tile.object_id = ""
                    tile.tile_type = TILE_EMPTY
                    return obj_id, drop, True

        return "", "", False

    def spawn_entity(self, entity_id: str, x: int, y: int) -> None:
        self.entities.append(Entity.from_id(entity_id, x, y))

    def unlock_door_if_ready(self, has_sword: bool, slime_dead: bool) -> None:
        self.door_unlocked = True
