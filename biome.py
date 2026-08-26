"""Biome definitions, smooth terrain elevation, and blend helpers."""

import math

BIOMES = {
    0: "Forest",
    1: "Desert",
    2: "Volcanic",
    3: "Plains",
    4: "Cave",
    5: "Tundra",
    6: "Swamp",
}


def _hash2d(x: int, y: int, seed: int) -> float:
    """Return deterministic pseudo-random float in [0, 1) for lattice coordinate."""
    n = (x * 374761393 + y * 668265263 + seed * 0x9E3779B9) & 0xFFFFFFFF
    n = ((n >> 13) ^ n) * 1274126177 & 0xFFFFFFFF
    n ^= (n >> 16)
    return (n % 1000000) / 1000000.0


def _smooth_noise(x: float, y: float, seed: int) -> float:
    """Bilinear smoothed value noise across 2D plane."""
    x0 = int(math.floor(x))
    x1 = x0 + 1
    y0 = int(math.floor(y))
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0

    # Smoothstep interpolation
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)

    v00 = _hash2d(x0, y0, seed)
    v10 = _hash2d(x1, y0, seed)
    v01 = _hash2d(x0, y1, seed)
    v11 = _hash2d(x1, y1, seed)

    top = v00 * (1.0 - sx) + v10 * sx
    bot = v01 * (1.0 - sx) + v11 * sx
    return top * (1.0 - sy) + bot * sy


def biome_at(x: int, y: int, seed: int = 12345) -> int:
    """Return a biome index based on smooth macro zones and pseudo-noise blending."""
    # Determine base biome zones
    if abs(x) + abs(y) < 25:
        base = 0  # Forest core around spawn
    elif x > 35:
        base = 1  # Desert
    elif y > 35:
        base = 2  # Volcanic
    elif x < -35:
        base = 5  # Tundra
    elif y < -35:
        base = 6  # Swamp
    else:
        base = 3  # Plains

    # Smooth blending at borders
    noise_val = _smooth_noise(x * 0.08, y * 0.08, seed)
    if noise_val < 0.12:
        base = (base - 1) % len(BIOMES)
    elif noise_val > 0.88:
        base = (base + 1) % len(BIOMES)
    return base


def get_elevation(x: int, y: int, seed: int = 98765) -> int:
    """Return a smooth, natural integer elevation (Z) for tile (x, y).

    Smooth multi-frequency noise produces gentle rolling terrain (Z = 0..4)
    with gradual slopes so normal walking does not cause accidental fall damage.
    High mountains (Z = 6..10) appear naturally at macro peaks.
    """
    # Spawn area (0, 0) is always flat ground at Z = 0
    if abs(x) <= 2 and abs(y) <= 2:
        return 0

    # Multi-octave smooth terrain elevation
    octave1 = _smooth_noise(x * 0.04, y * 0.04, seed)         # Low frequency rolling hills
    octave2 = _smooth_noise(x * 0.12, y * 0.12, seed + 1013)   # Medium frequency details

    combined = octave1 * 0.75 + octave2 * 0.25  # In range [0, 1]

    # Map to gentle integer elevation (0 to 6)
    z = int(combined * 6.0)
    return z


def blend_weight(distance_to_border: float) -> float:
    distance = max(0.0, min(15.0, distance_to_border))
    return distance / 15.0
