"""W3 proof: things grow where they belong, far apart, and grow back unseen.

    py -3 tests/test_w3_land.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import random

import config_rl
from world import World, TILE_HOUSE, TILE_EMPTY, TILE_OBJECT, VILLAGE_CELL
from item_ids import REGISTRY


def worlds(n=3):
    for seed in range(1, n + 1):
        w = World(seed=seed)
        for i in range(1, 6):
            w.materialise(i)
        yield w


def objects(w):
    return {p: t.object_id for p, t in w.tiles.items() if t.object_id}


def test_no_straight_line_harvest():
    """Walking 40 tiles in a line, you don't keep finding the same food."""
    for w in worlds():
        objs = objects(w)
        land = [p for p, t in w.tiles.items() if t.island_id >= 0]
        rng = random.Random(0)
        for _ in range(400):
            x, y = rng.choice(land)
            dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            line = [objs.get((x + dx * k, y + dy * k)) for k in range(40)]
            for food in ("0001", "0023", "0032"):
                assert line.count(food) <= 2, f"{line.count(food)}x {food} on one line"


def test_food_comes_in_patches_far_apart():
    """No 9x9 area holds more apples than one bunch (6), outside the
    hand-placed starter food by the spawn."""
    for w in worlds():
        sx, sy = w.agent_spawn
        apples = [p for p, o in objects(w).items() if o == "0001"
                  and abs(p[0] - sx) + abs(p[1] - sy) > 6]
        for p in apples:
            n = sum(1 for q in apples if abs(p[0] - q[0]) <= 4 and abs(p[1] - q[1]) <= 4)
            assert n <= 6, f"{n} apples in one 9x9 area at {p}"


def test_villages_grow_from_a_first_house():
    for w in worlds():
        assert w.villages, "no village anywhere"
        for houses in w.villages:
            assert 1 <= len(houses) <= 10
            for hx, hy in houses:
                assert w.tiles[(hx, hy)].tile_type == TILE_HOUSE
            if len(houses) > 1:              # houses of a village are near each other
                for h in houses[1:]:
                    assert min(abs(h[0] - g[0]) + abs(h[1] - g[1])
                               for g in houses if g != h) <= 10


def test_bread_only_near_houses():
    for w in worlds():
        houses = [h for v in w.villages for h in v]
        breads = [p for p, o in objects(w).items() if o == "0032"]
        assert breads, "no bread anywhere"
        for p in breads:
            assert any(abs(p[0] - h[0]) <= 3 and abs(p[1] - h[1]) <= 3 for h in houses), \
                f"bread at {p} with no house near"


def test_houses_block_movement():
    w = next(worlds(1))
    hx, hy = w.villages[0][0]
    assert not w.in_bounds_passable(hx, hy)


def test_regrows_only_out_of_sight():
    w = World(seed=1)
    p = next(q for q, o in objects(w).items() if o == "0001")
    assert w.pickup_at(*p)
    due = REGISTRY.get_item("0001")["regrow"]
    for _ in range(due + 100):                       # the agent stands right there
        w.tick_life(*p)
    assert not w.tiles[p].object_id, "it grew back in front of the agent"
    for _ in range(50):                              # the agent walks away
        w.tick_life(p[0] + 50, p[1])
    assert w.tiles[p].object_id == "0001", "it never grew back"


def test_regrows_at_its_own_spot_not_before_due():
    w = World(seed=1)
    p = next(q for q, o in objects(w).items() if o == "0001")
    w.pickup_at(*p)
    for _ in range(REGISTRY.get_item("0001")["regrow"] - 30):
        w.tick_life(p[0] + 50, p[1])
    assert not w.tiles[p].object_id, "grew back too early"


def test_old_scatter_behind_the_flag():
    config_rl.W3_PATCHES = False
    try:
        w = World(seed=1)
        assert not w.villages and len(objects(w)) > 2000, "flag off should be the old scatter"
    finally:
        config_rl.W3_PATCHES = True


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, f in tests:
        try:
            f()
            print("ok  ", name)
        except AssertionError as e:
            bad += 1
            print("FAIL", name, "-", e)
    print(f"{len(tests) - bad}/{len(tests)} passed")
    raise SystemExit(bad)
