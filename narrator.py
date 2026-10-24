"""The narrator (Chalce's idea, 2026-10-24): a few words about the situation,
every tick, the way you'd narrate to a child - "hungry", "growl, left",
"food, near" - so the brain can slowly tie words to what is happening.

The words are worked out ONLY from the state vector the brain already gets
(its vitals, its 7x7 view, its compass to the nearest thing, its ears). So
they can never tell it something its senses don't: no narrator-shaped sixth
sense to lean on instead of looking and listening. The same function runs on
the island, on the rungs and on the desktop.

They go into the brain as WORD_N on/off inputs (which words are being said),
before the action mask. Whether it understands them is tested the other way
round: describe.py checks whether its own memory state can say the words with
the words hidden (roadmap H6)."""
import config_rl

WORDS = ["HUNGRY", "FED", "HURT", "FOOD", "DANGER", "GROWL", "SAFE",
         "NEAR", "FAR", "LEFT", "RIGHT", "UP", "DOWN"]
WORD_N = len(WORDS)
assert WORD_N == config_rl.WORD_N
W = {w: i for i, w in enumerate(WORDS)}

TILE_HOSTILE, TILE_OBJECT, TILE_SHARK = 2, 3, 14       # world.py's ids, as the patch carries them


def _direction(dx, dy):
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _food(obj_norm):
    from world import _is_food
    i = round(obj_norm * config_rl.ITEM_ID_SCALE)
    return i > 0 and _is_food("%04d" % i)


def words_from_state(s):
    """The words for a state vector (only its first IDX_WORD_START entries are read)."""
    out = set()
    health, hunger = s[0], s[1]
    if hunger < 0.4:
        out.add("HUNGRY")
    elif hunger >= config_rl.STILL_FED_ABOVE / 100.0:
        out.add("FED")
    if health < 0.5:
        out.add("HURT")

    # what's in view: the 7x7 patch, row-major, centred on it
    r = config_rl.PATCH_RADIUS
    p0 = config_rl.IDX_PATCH_START
    danger_at, food_seen = None, False
    for i in range(config_rl.PATCH_TILES):
        t = int(s[p0 + i])
        dx, dy = i % (2 * r + 1) - r, i // (2 * r + 1) - r
        if t in (TILE_HOSTILE, TILE_SHARK):
            if danger_at is None or abs(dx) + abs(dy) < abs(danger_at[0]) + abs(danger_at[1]):
                danger_at = (dx, dy)
        elif t == TILE_OBJECT:
            food_seen = True

    d = config_rl.IDX_DIR_START
    focus = None                                  # what the direction words are about
    if danger_at:
        out.add("DANGER")
        focus = (danger_at, abs(danger_at[0]) + abs(danger_at[1]) <= 2)
    elif config_rl.HEARING and _growl(s):
        out.update(("DANGER", "GROWL"))
        h = config_rl.IDX_HEAR_START + config_rl.HEAR_DIM - 2
        focus = ((s[h], s[h + 1]), False)        # heard, not seen: far
    else:
        out.add("SAFE")

    if food_seen and _food(s[config_rl.IDX_OBJ_ID]):
        out.add("FOOD")
        if focus is None:
            focus = ((s[d], s[d + 1]), s[18] < 0.35)

    if focus:
        (dx, dy), near = focus
        way = _direction(dx, dy)
        if way:
            out.add(way)
        out.add("NEAR" if near else "FAR")
    return [w for w in WORDS if w in out]


def _growl(s):
    """A growl in its ears: loud enough, low and pitched (senses' prosody numbers)."""
    from ear import FIELD
    h = config_rl.IDX_HEAR_START
    pitch = s[h + FIELD["pitch_median_norm"]] * 400.0
    return s[h + FIELD["rms_mean"]] > 0.005 and 60.0 < pitch < 140.0


def as_inputs(words):
    v = [0.0] * WORD_N
    for w in words:
        v[W[w]] = 1.0
    return v
