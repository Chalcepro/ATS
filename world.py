"""Tile world, procedural chunk generation, natural objects, and entity placement.

Interactivity guarantee: every tile the agent can reach either
  (a) has an item/resource object they can PICK_UP or HARVEST,
  (b) has an entity they can ATTACK or INTERACT with, or
  (c) is open terrain they can MOVE through.
Hundreds of trees, bushes, logs, rocks, and wildlife populate the open world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import biome as biome_mod
import config_rl
from entities import Entity
from item_ids import REGISTRY

# Chunk configuration – each chunk is a square of tiles generated lazily.
CHUNK_SIZE = 16  # 16x16 tiles per chunk

TILE_EMPTY = 0
TILE_WALL = 1
TILE_HOSTILE = 2
TILE_OBJECT = 3
TILE_NPC = 4
TILE_WATER = 5
TILE_LAVA = 6


@dataclass
class Tile:
    tile_type: int = TILE_EMPTY
    z: int = 0
    object_id: str = ""
    explored: bool = False
    biome: int = 0
    flags: str = ""
    drop_item: str = ""  # item placed here by entity death / chest / harvest


class World:
    def __init__(self, seed: int = 1337):
        self.seed = seed
        self.tiles: dict[tuple[int, int], Tile] = {}
        self.entities: list[Entity] = []
        self.generated_chunks: set[tuple[int, int]] = set()
        self.door_unlocked = True
        self.agent_spawn = (0, 0)

        # Generate initial 3x3 chunk area around spawn (48x48 tiles = 2304 tiles)
        for c_dy in range(-1, 2):
            for c_dx in range(-1, 2):
                self._ensure_chunk_generated(c_dx * CHUNK_SIZE, c_dy * CHUNK_SIZE)

        # Ensure spawn tile (0, 0) is clear and safe
        spawn_tile = self._tile(0, 0)
        spawn_tile.tile_type = TILE_EMPTY
        spawn_tile.object_id = ""
        spawn_tile.z = 0

        # Place a few immediate starter harvestables and items nearby for immediate exploration
        self._ensure_starter_surroundings()

    # ------------------------------------------------------------------
    # Tile access & Procedural Generation
    # ------------------------------------------------------------------
    def _tile(self, x: int, y: int) -> Tile:
        """Return a Tile, generating its chunk on demand if it does not exist."""
        self._ensure_chunk_generated(x, y)
        return self.tiles.get((x, y))

    def _ensure_chunk_generated(self, x: int, y: int) -> None:
        """Generate all tiles and procedural objects/entities for the chunk containing (x, y)."""
        chunk_x = (x // CHUNK_SIZE) * CHUNK_SIZE
        chunk_y = (y // CHUNK_SIZE) * CHUNK_SIZE
        chunk_coord = (chunk_x, chunk_y)

        if chunk_coord in self.generated_chunks:
            return

        self.generated_chunks.add(chunk_coord)

        # Generate 16x16 tiles in chunk
        for dy in range(CHUNK_SIZE):
            for dx in range(CHUNK_SIZE):
                tx = chunk_x + dx
                ty = chunk_y + dy

                t_biome = biome_mod.biome_at(tx, ty, seed=self.seed)
                t_elevation = biome_mod.get_elevation(tx, ty, seed=self.seed)

                # Determine base tile type
                tile_type = TILE_EMPTY
                flags = ""

                # Mountain wall barriers at very high elevation
                if t_elevation > 880:
                    tile_type = TILE_WALL
                # Water bodies at low elevation
                elif t_elevation < -420 and t_biome != 2:
                    tile_type = TILE_WATER
                # Lava pools in volcanic biome
                elif t_biome == 2 and t_elevation < -250:
                    tile_type = TILE_LAVA

                tile = Tile(
                    tile_type=tile_type,
                    z=t_elevation,
                    biome=t_biome,
                    flags=flags,
                )

                # Procedural object placement on empty ground tiles
                if tile_type == TILE_EMPTY and not (tx == 0 and ty == 0):
                    self._populate_tile_objects(tile, tx, ty, t_biome)

                self.tiles[(tx, ty)] = tile

        # Spawn entities for this chunk
        self._populate_chunk_entities(chunk_x, chunk_y)

    def _populate_tile_objects(self, tile: Tile, tx: int, ty: int, biome_id: int):
        """Deterministically place natural objects (trees, bushes, logs, rocks, ores) on tile."""
        # 2D coordinate pseudo-random hash
        h = (tx * 374761393 + ty * 668265263 + self.seed * 0x9E3779B9) & 0xFFFFFFFF
        h ^= (h >> 13)
        h ^= (h << 7) & 0xFFFFFFFF
        h ^= (h >> 17)
        rand_val = (h % 10000) / 10000.0

        # Density table by biome
        if biome_id == 0:  # Forest: Lush trees, berry bushes, logs, sticks, mushrooms, flowers
            if rand_val < 0.09:
                tile.object_id = "0021"  # Oak Tree
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.16:
                tile.object_id = "0023"  # Berry Bush
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.22:
                tile.object_id = "0024"  # Fallen Log
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.27:
                tile.object_id = "0008"  # Stick
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.31:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.35:
                tile.object_id = "0028"  # Mushroom
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.39:
                tile.object_id = "0029"  # Flower
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.42:
                tile.object_id = "0009"  # Coal
                tile.tile_type = TILE_OBJECT

        elif biome_id == 3:  # Plains: Open grasslands, scattered bushes, flowers, sticks, stones
            if rand_val < 0.04:
                tile.object_id = "0021"  # Oak Tree
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.13:
                tile.object_id = "0023"  # Berry Bush
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.22:
                tile.object_id = "0029"  # Flower
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.28:
                tile.object_id = "0008"  # Stick
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.33:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.36:
                tile.object_id = "0001"  # Apple
                tile.tile_type = TILE_OBJECT

        elif biome_id == 1:  # Desert: Cacti, sandstone boulders, dead shrubs, stones
            if rand_val < 0.08:
                tile.object_id = "0027"  # Cactus
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.15:
                tile.object_id = "0025"  # Boulder
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.22:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.26:
                tile.object_id = "0008"  # Stick (dead shrub)
                tile.tile_type = TILE_OBJECT

        elif biome_id == 2:  # Volcanic: Obsidian boulders, coal ore, ash debris, void shards
            if rand_val < 0.09:
                tile.object_id = "0026"  # Coal Ore
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.18:
                tile.object_id = "0025"  # Boulder
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.25:
                tile.object_id = "0009"  # Coal
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.29:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.31:
                tile.object_id = "0020"  # Void Shard
                tile.tile_type = TILE_OBJECT

        elif biome_id == 5:  # Tundra: Pine trees, fallen logs, boulders, stones
            if rand_val < 0.10:
                tile.object_id = "0022"  # Pine Tree
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.17:
                tile.object_id = "0024"  # Fallen Log
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.23:
                tile.object_id = "0025"  # Boulder
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.28:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.32:
                tile.object_id = "0008"  # Stick
                tile.tile_type = TILE_OBJECT

        elif biome_id == 6:  # Swamp: Swamp trees, mushrooms, murky flora, fallen logs
            if rand_val < 0.08:
                tile.object_id = "0021"  # Oak Tree
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.17:
                tile.object_id = "0028"  # Mushroom
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.23:
                tile.object_id = "0024"  # Fallen Log
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.28:
                tile.object_id = "0023"  # Berry Bush
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.32:
                tile.object_id = "0008"  # Stick
                tile.tile_type = TILE_OBJECT

        elif biome_id == 4:  # Cave: Stalagmites, coal ore, stone, mushrooms, void shards
            if rand_val < 0.11:
                tile.object_id = "0030"  # Stalagmite
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.19:
                tile.object_id = "0026"  # Coal Ore
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.27:
                tile.object_id = "0012"  # Stone
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.33:
                tile.object_id = "0028"  # Mushroom
                tile.tile_type = TILE_OBJECT
            elif rand_val < 0.36:
                tile.object_id = "0020"  # Void Shard
                tile.tile_type = TILE_OBJECT

    def _populate_chunk_entities(self, chunk_x: int, chunk_y: int):
        """Spawn 1-3 natural entities per chunk appropriate to the biome."""
        center_x = chunk_x + CHUNK_SIZE // 2
        center_y = chunk_y + CHUNK_SIZE // 2
        biome_id = biome_mod.biome_at(center_x, center_y, seed=self.seed)

        # Entity roster by biome
        entity_pools = {
            0: ["E001", "E015", "E005", "E002"],           # Forest: Slime, Sheep, Spider, Zombie
            3: ["E015", "E016", "E017", "E018", "E001"],   # Plains: Sheep, Pig, Chicken, Villager, Slime
            1: ["E021", "E006", "E003"],                   # Desert: Scorpion, Husk, Skeleton
            2: ["E007", "E023"],                           # Volcanic: Fire Imp, Lava Golem
            5: ["E011", "E022", "E012"],                   # Tundra: Wolf, Ice Wolf, Yeti
            6: ["E024", "E010", "E001"],                   # Swamp: Toad, Witch, Slime
            4: ["E013", "E008", "E025", "E009"],           # Cave: Bat, Cave Spider, Queen, Golem
        }

        pool = entity_pools.get(biome_id, ["E001", "E015"])

        # Determine count (1 to 3 per chunk)
        h = (chunk_x * 961748941 + chunk_y * 48271) & 0xFFFFFFFF
        num_spawns = 1 + (h % 3)

        for i in range(num_spawns):
            ent_id = pool[(h + i * 7) % len(pool)]
            # Pick a spot in chunk
            offset_x = 2 + ((h >> (i * 3 + 2)) % (CHUNK_SIZE - 4))
            offset_y = 2 + ((h >> (i * 3 + 5)) % (CHUNK_SIZE - 4))
            ex = chunk_x + offset_x
            ey = chunk_y + offset_y

            # Avoid spawning right on the agent spawn (0, 0)
            if abs(ex) <= 1 and abs(ey) <= 1:
                ex += 3
                ey += 3

            tile = self.tiles.get((ex, ey))
            if tile and tile.tile_type not in (TILE_WALL, TILE_LAVA, TILE_WATER):
                self.entities.append(Entity.from_id(ent_id, ex, ey))

    def _ensure_starter_surroundings(self):
        """Place pleasant initial items/bushes/trees right around spawn."""
        starters = [
            (1, 0, "0001"),    # Apple
            (-1, 0, "0023"),   # Berry Bush
            (0, 1, "0008"),    # Stick
            (0, -1, "0010"),   # Wood
            (1, 1, "0021"),    # Oak Tree
            (-1, -1, "0012"),  # Stone
            (2, 0, "0024"),    # Fallen Log
            (-2, 1, "0023"),   # Berry Bush
            (1, 2, "0009"),    # Coal
        ]
        for sx, sy, obj_id in starters:
            t = self.tiles.get((sx, sy))
            if t and t.tile_type != TILE_WALL:
                t.object_id = obj_id
                t.tile_type = TILE_OBJECT

    def _place_object(self, x: int, y: int, item_id: str):
        tile = self._tile(x, y)
        tile.object_id = item_id
        tile.tile_type = TILE_OBJECT

    # ------------------------------------------------------------------
    # Drop spawn: place a loot item on the map when entity dies or object harvested
    # ------------------------------------------------------------------
    def spawn_drop(self, x: int, y: int, item_id: str):
        """Place *item_id* at (x,y). If occupied, scatter to adjacent tile."""
        if not item_id:
            return
        candidates = [(x, y)] + [(x + dx, y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
        for cx, cy in candidates:
            tile = self._tile(cx, cy)
            if tile and tile.tile_type not in (TILE_WALL, TILE_LAVA) and not tile.object_id:
                tile.object_id = item_id
                tile.tile_type = TILE_OBJECT
                return
        # Fallback: force-place on coordinate
        tile = self._tile(x, y)
        if tile:
            tile.object_id = item_id
            tile.tile_type = TILE_OBJECT

    # ------------------------------------------------------------------
    # Door / progression
    # ------------------------------------------------------------------
    def unlock_door_if_ready(self, has_sword: bool, slime_dead: bool):
        self.door_unlocked = True

    # ------------------------------------------------------------------
    # Passability
    # ------------------------------------------------------------------
    def in_bounds_passable(self, x: int, y: int, door_unlocked: bool = True) -> bool:
        tile = self._tile(x, y)
        if not tile:
            return False
        if tile.tile_type == TILE_WALL:
            return False
        if tile.tile_type == TILE_LAVA:
            return False
        # Block movement through alive hostile entities
        for ent in self.entities:
            if ent.alive and ent.x == x and ent.y == y:
                ent_type = ent.data.get("type", "hostile")
                if ent_type == "hostile":
                    return False
        return True

    # ------------------------------------------------------------------
    # Height map helpers
    # ------------------------------------------------------------------
    def get_adjacent_z(self, x: int, y: int) -> dict:
        t_ahead = self._tile(x, y - 1)
        t_behind = self._tile(x, y + 1)
        t_left = self._tile(x - 1, y)
        t_right = self._tile(x + 1, y)
        return {
            "ahead":  t_ahead.z if t_ahead else 0,
            "behind": t_behind.z if t_behind else 0,
            "left":   t_left.z if t_left else 0,
            "right":  t_right.z if t_right else 0,
        }

    # ------------------------------------------------------------------
    # Exploration fog
    # ------------------------------------------------------------------
    def mark_explored(self, x: int, y: int, radius: int):
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
        """Like nearest_entity but returns Chebyshev (max-of-abs) distance."""
        best, best_dist = None, 999
        for ent in self.entities:
            if not ent.alive:
                continue
            d = max(abs(ent.x - x), abs(ent.y - y))
            if d < best_dist:
                best, best_dist = ent, d
        return best, best_dist

    def entity_at(self, x: int, y: int) -> Entity | None:
        for ent in self.entities:
            if ent.alive and ent.x == x and ent.y == y:
                return ent
        return None

    # ------------------------------------------------------------------
    # Object queries
    # ------------------------------------------------------------------
    def nearest_object(self, x: int, y: int):
        best_id, best_dist = "", 999
        # Check active loaded tiles within perception radius
        for (tx, ty), tile in self.tiles.items():
            if tile.object_id:
                d = abs(tx - x) + abs(ty - y)
                if d < best_dist:
                    best_id, best_dist = tile.object_id, d
        return best_id, best_dist

    # ------------------------------------------------------------------
    # Tile type at position (entity presence overrides tile)
    # ------------------------------------------------------------------
    def tile_type_at(self, x: int, y: int) -> int:
        if any(e.alive and e.x == x and e.y == y for e in self.entities):
            return TILE_HOSTILE
        tile = self._tile(x, y)
        if not tile:
            return TILE_EMPTY
        if tile.object_id:
            return TILE_OBJECT
        return tile.tile_type

    # ------------------------------------------------------------------
    # Pick up item at position
    # ------------------------------------------------------------------
    def pickup_at(self, x: int, y: int) -> str:
        tile = self._tile(x, y)
        if not tile or not tile.object_id:
            return ""
        item = tile.object_id
        
        # Check if it's a world object that yields drops on harvest
        reg_item = REGISTRY.get_item(item)
        drop_yield = item
        if reg_item and reg_item.get("drop"):
            drop_yield = reg_item.get("drop")

        tile.object_id = ""
        if tile.tile_type == TILE_OBJECT:
            tile.tile_type = TILE_EMPTY
        return drop_yield

    # ------------------------------------------------------------------
    # Attack / Harvest — returns (killed_target_id, drop_item_id, is_world_object)
    # ------------------------------------------------------------------
    def attack_at(self, x: int, y: int, damage: int, attacker_pos: tuple[int, int] | None = None, facing_dx: int = 0, facing_dy: int = -1) -> tuple[str, str, bool]:
        """Attack entity or harvest world object in front or within 3x3 square.

        Returns (killed_id, drop_id, is_world_object).
        """
        # 1. First priority: entity directly in front or within 3x3
        # Check directly facing entity first
        fx, fy = x + facing_dx, y + facing_dy
        ent_facing = self.entity_at(fx, fy)
        if ent_facing and ent_facing.alive:
            ent_facing.take_damage(damage, attacker_pos=attacker_pos)
            if not ent_facing.alive:
                drop = ent_facing.drop_item_id()
                return ent_facing.entity_id, drop, False
            return "", "", False

        # Any entity within 3x3
        for ent in self.entities:
            if ent.alive and max(abs(ent.x - x), abs(ent.y - y)) <= 1:
                ent.take_damage(damage, attacker_pos=attacker_pos)
                if not ent.alive:
                    drop = ent.drop_item_id()
                    return ent.entity_id, drop, False
                return "", "", False

        # 2. Second priority: harvest world object directly in front or on current tile
        target_coords = [(fx, fy), (x, y)]
        for tx, ty in target_coords:
            tile = self._tile(tx, ty)
            if tile and tile.object_id:
                obj_id = tile.object_id
                item_info = REGISTRY.get_item(obj_id)
                if item_info and item_info.get("type") == "world_object":
                    drop = item_info.get("drop", obj_id)
                    tile.object_id = ""
                    tile.tile_type = TILE_EMPTY
                    return obj_id, drop, True

        return "", "", False

    # ------------------------------------------------------------------
    # Spawn a new entity
    # ------------------------------------------------------------------
    def spawn_entity(self, entity_id: str, x: int, y: int):
        self.entities.append(Entity.from_id(entity_id, x, y))
