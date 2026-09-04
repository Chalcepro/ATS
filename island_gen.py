"""Island generation engine for ATS.

Replaces the flat biome noise-field world with a proper island-ocean architecture.
Each island has an organic, amoeba-like non-uniform shape generated via multi-octave
noise with a radial falloff mask. Islands are surrounded by TILE_OCEAN (shark territory).
Internal water bodies are native per island type.
6 island types: Grassland, Dark Forest, Volcanic, Mushroom, Tundra, Boss Sanctum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Island type constants
# ---------------------------------------------------------------------------
ISLAND_GRASSLAND   = 0
ISLAND_DARK_FOREST = 1
ISLAND_VOLCANIC    = 2
ISLAND_MUSHROOM    = 3
ISLAND_TUNDRA      = 4
ISLAND_BOSS        = 5

ISLAND_NAMES: Dict[int, str] = {
    ISLAND_GRASSLAND:   "Lush Grassland",
    ISLAND_DARK_FOREST: "Dark Forest",
    ISLAND_VOLCANIC:    "Volcanic Wastes",
    ISLAND_MUSHROOM:    "Poison Mushroom Isle",
    ISLAND_TUNDRA:      "Frozen Tundra",
    ISLAND_BOSS:        "Boss Sanctum",
}

ISLAND_TIERS: Dict[int, int] = {
    ISLAND_GRASSLAND:   0,
    ISLAND_DARK_FOREST: 1,
    ISLAND_VOLCANIC:    2,
    ISLAND_MUSHROOM:    2,
    ISLAND_TUNDRA:      3,
    ISLAND_BOSS:        4,
}

# World-space centers for each island (tile coordinates — min 100m ocean gap)
ISLAND_CENTERS: Dict[int, Tuple[int, int]] = {
    ISLAND_GRASSLAND:   (0,    0),
    ISLAND_DARK_FOREST: (220, -140),
    ISLAND_VOLCANIC:    (-200, 200),
    ISLAND_MUSHROOM:    (260,  190),
    ISLAND_TUNDRA:      (-240, -180),
    ISLAND_BOSS:        (420,  320),
}

# ---------------------------------------------------------------------------
# Island Definition
# ---------------------------------------------------------------------------
@dataclass
class IslandDef:
    island_id: int
    center: Tuple[int, int]
    land_tiles:      Set[Tuple[int, int]] = field(default_factory=set)
    internal_water:  Set[Tuple[int, int]] = field(default_factory=set)
    internal_hazard: Set[Tuple[int, int]] = field(default_factory=set)
    safe_path:       Set[Tuple[int, int]] = field(default_factory=set)
    spore_zones:     Set[Tuple[int, int]] = field(default_factory=set)
    obstacle_tiles:  Set[Tuple[int, int]] = field(default_factory=set)
    radius: int = 30

    @property
    def name(self) -> str:
        return ISLAND_NAMES[self.island_id]

    @property
    def tier(self) -> int:
        return ISLAND_TIERS[self.island_id]

    @property
    def spawn_pos(self) -> Tuple[int, int]:
        return self.center

    def contains(self, x: int, y: int) -> bool:
        return (x, y) in self.land_tiles


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------
def _hash2d(x: int, y: int, seed: int) -> float:
    n = (x * 374761393 + y * 668265263 + seed * 0x9E3779B9) & 0xFFFFFFFF
    n = ((n >> 13) ^ n) * 1274126177 & 0xFFFFFFFF
    n ^= (n >> 16)
    return (n % 1000000) / 1000000.0


def _smooth_noise(fx: float, fy: float, seed: int) -> float:
    x0, y0 = int(math.floor(fx)), int(math.floor(fy))
    x1, y1 = x0 + 1, y0 + 1
    sx = fx - x0
    sy = fy - y0
    sx = sx * sx * (3.0 - 2.0 * sx)
    sy = sy * sy * (3.0 - 2.0 * sy)
    v00 = _hash2d(x0, y0, seed)
    v10 = _hash2d(x1, y0, seed)
    v01 = _hash2d(x0, y1, seed)
    v11 = _hash2d(x1, y1, seed)
    return (v00*(1-sx) + v10*sx)*(1-sy) + (v01*(1-sx) + v11*sx)*sy


def _fbm(fx: float, fy: float, seed: int, octaves: int = 4) -> float:
    value, amplitude, frequency, max_val = 0.0, 1.0, 1.0, 0.0
    for i in range(octaves):
        value    += _smooth_noise(fx * frequency, fy * frequency, seed + i * 997) * amplitude
        max_val  += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return value / max_val


# ---------------------------------------------------------------------------
# Island shape
# ---------------------------------------------------------------------------
def generate_island_shape(
    center: Tuple[int, int],
    seed: int,
    island_id: int,
) -> Tuple[Set[Tuple[int, int]], int]:
    """Organic amoeba-shaped island. Returns (land_tiles, radius)."""
    cx, cy = center
    r_noise = _hash2d(island_id * 17, seed, 0xABCDEF)
    radius  = int(30 + r_noise * 30)      # 30..60 tiles

    land_tiles: Set[Tuple[int, int]] = set()
    for dy in range(-radius - 5, radius + 6):
        for dx in range(-radius - 5, radius + 6):
            tx, ty = cx + dx, cy + dy
            dist = math.sqrt(dx*dx + dy*dy) / radius
            coast    = _fbm(tx * 0.12, ty * 0.12, seed + island_id * 1337, octaves=4)
            interior = _fbm(tx * 0.04, ty * 0.04, seed + island_id * 2749, octaves=3)
            eff_dist = dist - (coast - 0.5) * 0.45 - (interior - 0.5) * 0.15
            if eff_dist < 0.82:
                land_tiles.add((tx, ty))
    return land_tiles, radius


# ---------------------------------------------------------------------------
# Internal water / hazards
# ---------------------------------------------------------------------------
def generate_internal_water(
    land_tiles: Set[Tuple[int, int]],
    island_id: int,
    center: Tuple[int, int],
    seed: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Returns (water_tiles, hazard_tiles, safe_path, spore_zones)."""
    water:   Set[Tuple[int, int]] = set()
    hazard:  Set[Tuple[int, int]] = set()
    safe:    Set[Tuple[int, int]] = set()
    spore:   Set[Tuple[int, int]] = set()
    cx, cy  = center

    for (tx, ty) in land_tiles:
        dist = math.sqrt((tx-cx)**2 + (ty-cy)**2)
        if dist < 8:
            continue  # safe zone — no hazards

        hn = _smooth_noise(tx * 0.08, ty * 0.08, seed + island_id * 7919)

        if island_id == ISLAND_GRASSLAND:
            river = _smooth_noise(tx * 0.05, ty * 0.05, seed + 1111)
            if river > 0.72:
                water.add((tx, ty))

        elif island_id == ISLAND_DARK_FOREST:
            if hn > 0.88:
                water.add((tx, ty))

        elif island_id == ISLAND_VOLCANIC:
            lava = _smooth_noise(tx * 0.07, ty * 0.07, seed + 3333)
            if lava > 0.64:
                hazard.add((tx, ty))
            elif lava > 0.58:
                safe.add((tx, ty))

        elif island_id == ISLAND_MUSHROOM:
            acid  = _smooth_noise(tx * 0.09, ty * 0.09, seed + 5555)
            sp    = _smooth_noise(tx * 0.06, ty * 0.06, seed + 6666)
            if acid > 0.80:
                hazard.add((tx, ty))
            elif sp > 0.76:
                spore.add((tx, ty))

        elif island_id == ISLAND_TUNDRA:
            ice = _smooth_noise(tx * 0.10, ty * 0.10, seed + 7777)
            if ice > 0.84:
                hazard.add((tx, ty))
            elif hn > 0.90:
                water.add((tx, ty))

    return water, hazard, safe, spore


# ---------------------------------------------------------------------------
# Obstacle clusters
# ---------------------------------------------------------------------------
def generate_obstacle_clusters(
    land_tiles: Set[Tuple[int, int]],
    island_id: int,
    center: Tuple[int, int],
    seed: int,
    occupied: Set[Tuple[int, int]],
) -> Set[Tuple[int, int]]:
    """Mandatory obstacle walls that break straight-line traversal."""
    obstacles: Set[Tuple[int, int]] = set()
    cx, cy = center
    n_clusters = {
        ISLAND_GRASSLAND: 25, ISLAND_DARK_FOREST: 30, ISLAND_VOLCANIC: 15,
        ISLAND_MUSHROOM: 20,  ISLAND_TUNDRA: 28,      ISLAND_BOSS: 20,
    }.get(island_id, 20)
    min_sz, max_sz = {
        ISLAND_GRASSLAND: (3,5), ISLAND_DARK_FOREST: (3,6), ISLAND_VOLCANIC: (3,5),
        ISLAND_MUSHROOM: (4,8),  ISLAND_TUNDRA: (3,6),      ISLAND_BOSS: (4,7),
    }.get(island_id, (3,5))

    land_list = list(land_tiles)
    if not land_list:
        return obstacles

    for c_idx in range(n_clusters):
        h  = _hash2d(c_idx * 31 + island_id, seed, 0xDEADBEEF)
        li = int(h * len(land_list)) % len(land_list)
        ox, oy = land_list[li]
        if math.sqrt((ox-cx)**2 + (oy-cy)**2) < 10 or (ox,oy) in occupied:
            continue
        h2 = _hash2d(c_idx * 53, seed + 9999, 0xCAFEBABE)
        cluster_size = min_sz + int(h2 * (max_sz - min_sz + 1))
        cluster: Set[Tuple[int,int]] = {(ox,oy)}
        frontier = [(ox,oy)]
        dirs = [(0,1),(0,-1),(1,0),(-1,0)]
        for _ in range(cluster_size - 1):
            if not frontier:
                break
            fi = int(_hash2d(len(cluster), seed + c_idx, 7777) * len(frontier)) % len(frontier)
            fx_c, fy_c = frontier[fi]
            di = int(_hash2d(fx_c, fy_c, seed + c_idx * 3) * 4) % 4
            ndx, ndy = dirs[di]
            nx, ny = fx_c + ndx, fy_c + ndy
            if (nx,ny) in land_tiles and (nx,ny) not in occupied and (nx,ny) not in cluster:
                cluster.add((nx,ny))
                frontier.append((nx,ny))
        obstacles.update(cluster)
    return obstacles


# ---------------------------------------------------------------------------
# Island Registry
# ---------------------------------------------------------------------------
class IslandRegistry:
    def __init__(self, world_seed: int):
        self.world_seed = world_seed
        self.islands: List[IslandDef] = []
        self._tile_map: Dict[Tuple[int,int], int] = {}

    def build(self) -> None:
        for island_id, center in ISLAND_CENTERS.items():
            idef = self._build_island(island_id, center)
            self.islands.append(idef)
            for tile in idef.land_tiles:
                self._tile_map[tile] = island_id

    def _build_island(self, island_id: int, center: Tuple[int,int]) -> IslandDef:
        seed = self.world_seed + island_id * 1000003
        land, radius = generate_island_shape(center, seed, island_id)
        water, hazard, safe_path, spore = generate_internal_water(land, island_id, center, seed)
        occupied = hazard | water
        obstacles = generate_obstacle_clusters(land, island_id, center, seed, occupied)
        obstacles -= hazard | water | safe_path
        return IslandDef(
            island_id=island_id, center=center, land_tiles=land,
            internal_water=water, internal_hazard=hazard,
            safe_path=safe_path, spore_zones=spore,
            obstacle_tiles=obstacles, radius=radius,
        )

    def get_island_at(self, x: int, y: int) -> Optional[int]:
        return self._tile_map.get((x, y))

    def is_ocean(self, x: int, y: int) -> bool:
        return (x, y) not in self._tile_map

    def get_island(self, island_id: int) -> Optional[IslandDef]:
        for idef in self.islands:
            if idef.island_id == island_id:
                return idef
        return None

    def get_starting_island(self) -> IslandDef:
        return self.get_island(ISLAND_GRASSLAND)


def build_island_registry(world_seed: int) -> "IslandRegistry":
    reg = IslandRegistry(world_seed)
    reg.build()
    return reg
