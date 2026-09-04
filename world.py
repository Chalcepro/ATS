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
TILE_NPC     = 4   # NPC entity
TILE_WATER   = 5   # Island-internal lake/pond (swimmable, no sharks)
TILE_LAVA    = 6   # Volcanic floor hazard (5 HP/tick burn)
TILE_GATE    = 7   # Progression gate (locked)
TILE_OCEAN   = 8   # Open ocean — sharks spawn after 3 ticks
TILE_ACID    = 9   # Mushroom island acid pool (applies POISONED)
TILE_ICE     = 10  # Tundra ice (passable, 20% slip = leg injury)
TILE_ASH     = 11  # Volcanic safe ash path (no burn damage)
TILE_SPORE   = 12  # Mushroom spore cloud zone (applies BLINDED)


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
_ISLAND_ENTITY_POOLS = {
    ISLAND_GRASSLAND:   ["E001", "E015", "E016", "E017", "E018"],
    ISLAND_DARK_FOREST: ["E004", "E005", "E008", "E011", "E013", "E028"],
    ISLAND_VOLCANIC:    ["E007", "E022", "E026"],
    ISLAND_MUSHROOM:    ["E010", "E023", "E024"],
    ISLAND_TUNDRA:      ["E012", "E021", "E025"],
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


class World:
    """Island-ocean tile world. All generation driven by IslandRegistry."""

    def __init__(self, seed: int = 1337):
        self.seed = seed
        self.tiles: dict[tuple[int, int], Tile] = {}
        self.entities: list[Entity] = []
        self.door_unlocked = True
        self.agent_spawn = (0, 0)

        # Build island registry (generates all 6 islands)
        self.island_registry: IslandRegistry = build_island_registry(seed)
        self.starting_island: IslandDef = self.island_registry.get_starting_island()
        self.agent_spawn = self.starting_island.spawn_pos

        # Pre-generate all tiles for the starting island only
        self._generate_island_tiles(self.starting_island)

        # Ensure spawn tile is clear
        spawn_tile = self._tile(*self.agent_spawn)
        spawn_tile.tile_type = TILE_EMPTY
        spawn_tile.object_id = ""

        # Guarantee starter items around spawn
        self._place_starter_surroundings()

        # Spawn entities for starting island
        self._populate_island_entities(self.starting_island)

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
            if tile_type == TILE_EMPTY and not (tx == cx and ty == cy):
                dist = abs(tx - cx) + abs(ty - cy)
                if dist > 3:  # don't clutter spawn center
                    self._populate_tile_objects(t, tx, ty, idef)

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
        if tile.tile_type in (TILE_WALL, TILE_GATE):
            return False
        # Lava — passable but very damaging (handled in env tick)
        # Ocean — passable but shark spawns (handled in env tick)
        # Ice / acid / spore — passable but apply status effects
        # Block movement through alive hostile entities
        for ent in self.entities:
            if ent.alive and ent.x == x and ent.y == y:
                if ent.data.get("type") == "hostile":
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
        best_id, best_dist = "", 999
        for (tx, ty), tile in self.tiles.items():
            if tile.object_id:
                d = abs(tx - x) + abs(ty - y)
                if d < best_dist:
                    best_id, best_dist = tile.object_id, d
        return best_id, best_dist

    # ------------------------------------------------------------------
    # Height map helpers (stub — elevation not tile-based in island arch)
    # ------------------------------------------------------------------
    def get_adjacent_z(self, x: int, y: int) -> dict:
        return {"ahead": 0, "behind": 0, "left": 0, "right": 0}

    # ------------------------------------------------------------------
    # Tile type at position
    # ------------------------------------------------------------------
    def tile_type_at(self, x: int, y: int) -> int:
        if any(e.alive and e.x == x and e.y == y for e in self.entities):
            return TILE_HOSTILE
        tile = self._tile(x, y)
        if not tile:
            return TILE_OCEAN
        if tile.object_id:
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
        if ent_facing and ent_facing.alive and ent_facing.data.get("type") != "ocean":
            ent_facing.take_damage(damage, attacker_pos=attacker_pos)
            if not ent_facing.alive:
                return ent_facing.entity_id, ent_facing.drop_item_id(), False
            return "", "", False

        for ent in self.entities:
            if ent.alive and max(abs(ent.x - x), abs(ent.y - y)) <= 1:
                if ent.data.get("type") == "ocean":
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
                    drop = item_info.get("drop", obj_id)
                    tile.object_id = ""
                    tile.tile_type = TILE_EMPTY
                    return obj_id, drop, True

        return "", "", False

    def spawn_entity(self, entity_id: str, x: int, y: int) -> None:
        self.entities.append(Entity.from_id(entity_id, x, y))

    def unlock_door_if_ready(self, has_sword: bool, slime_dead: bool) -> None:
        self.door_unlocked = True
