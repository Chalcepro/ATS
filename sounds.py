"""Sounds in the island (roadmap H2): things make noise, and the agent hears
them through the same ear as Chalce's microphone (ear.py, senses' 34 prosody
numbers), plus two numbers for where the sound came from, like two ears.

What makes a sound, read from what each creature is already doing (the
creatures themselves are not changed):
  growl   a hostile that is chasing, winding up a strike, angry, or about to
          step into its fighting range - danger, often before it is in sight
  splash  a shark in the water nearby
  chirp   a harmless animal, now and then - so "a sound" alone never means
          "run"; the brain has to learn which sounds do

Only the nearest, loudest sound in a tick is heard. The waveforms are made
here with plain numpy (no files, no model), run once through the ear and
cached, so hearing costs nothing per tick."""
import random

import numpy as np

import config_rl
from ear import hear, RATE, PROSODY_DIM

HEAR_RADIUS = config_rl.HEAR_RADIUS
HEAR_DIM = PROSODY_DIM + 2                               # 34 ear numbers + direction (dx, dy)
assert HEAR_DIM == config_rl.HEAR_DIM
CHIRP_CHANCE = 0.08                                      # a harmless animal in earshot chirps this often
DANGER_KINDS = {"strikes_instantly", "explodes", "hunts"}
VARIANTS = 4                                             # different takes of each sound, so no two are identical

_cache = {}
# Its own dice: drawing from the shared `random` stream would shift every
# creature's moves, so merely hearing would change the world.
_rng = random.Random(0x5EA)


def _wave(kind, rng):
    """About half a second of one sound at full loudness."""
    n = RATE // 2
    t = np.arange(n) / RATE
    if kind == "growl":                  # low, rough, pulsing: 70-110 Hz with noise
        f = rng.uniform(70, 110) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(5, 9) * t))
        ph = 2 * np.pi * np.cumsum(f) / RATE
        x = sum(np.sin(k * ph) / k for k in range(1, 8)) + 0.25 * rng.standard_normal(n)
        x *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(10, 16) * t).clip(0)
    elif kind == "splash":               # a broadband burst that dies away, no pitch
        x = rng.standard_normal(n) * np.exp(-t * rng.uniform(5, 9))
        x = np.convolve(x, np.ones(3) / 3, "same")
    elif kind == "chirp":                # short, high, rising notes
        x = np.zeros(n)
        for start in (0.0, 0.18):
            m = (t >= start) & (t < start + 0.1)
            f = np.linspace(260, 390, m.sum()) * rng.uniform(0.9, 1.05)
            x[m] = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.hanning(m.sum())
    else:
        raise KeyError(kind)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def heard(kind, dist):
    """The 34 ear numbers for one sound at a distance in tiles (cached)."""
    d = min(int(dist), HEAR_RADIUS)
    key = (kind, d, _rng.randrange(VARIANTS))
    v = _cache.get(key)
    if v is None:
        rng = np.random.default_rng(hash(key) & 0xFFFFFFFF)
        loud = 0.5 / (1.0 + 0.6 * d)                     # quieter with distance
        v = _cache[key] = hear(_wave(kind, rng) * loud)
    return v


def growls(e, dist):
    """A hostile that is chasing, winding up, angry or about to step into its space."""
    spec = e.spec
    if not (spec.has("hostile") or spec.hostility in DANGER_KINDS):
        return False
    space = spec.hostility_args.get("space", -1)
    return bool(e.angry or e.state == 1 or e.job in ("chase", "fight") or dist <= space + 3
                or spec.hostility == "hunts")


def danger_near(world, x, y):
    """Is something dangerous growling or splashing in earshot? (No dice: safe for rewards.)"""
    for e in world.entities:
        if not e.alive:
            continue
        d = abs(e.x - x) + abs(e.y - y)
        if 0 < d <= HEAR_RADIUS and (e.spec.has("ocean") or growls(e, d)):
            return True
    return False


def sound_of(e, agent_x, agent_y):
    """What this creature sounds like right now, or None if it's quiet."""
    spec = e.spec
    dist = abs(e.x - agent_x) + abs(e.y - agent_y)
    if spec.has("ocean"):
        return "splash"
    if spec.has("hostile") or spec.hostility in DANGER_KINDS:
        return "growl" if growls(e, dist) else None
    if spec.hostility in ("ignores", "flees_when_hit") and _rng.random() < CHIRP_CHANCE:
        return "chirp"
    return None


def hearing(world, agent_x, agent_y):
    """HEAR_DIM floats for this tick: the loudest thing in earshot, or silence."""
    best = None
    for e in world.entities:            # not creatures_near: a shark that just surfaced isn't indexed yet
        if not e.alive:
            continue
        dist = abs(e.x - agent_x) + abs(e.y - agent_y)
        if dist > HEAR_RADIUS or dist == 0:
            continue
        kind = sound_of(e, agent_x, agent_y)
        if kind and (best is None or dist < best[0]):
            best = (dist, kind, e.x - agent_x, e.y - agent_y)
    out = np.zeros(HEAR_DIM, np.float32)
    if best:
        dist, kind, dx, dy = best
        out[:PROSODY_DIM] = heard(kind, dist)
        n = max(abs(dx), abs(dy))
        out[PROSODY_DIM:] = (dx / n, dy / n)
    return out
