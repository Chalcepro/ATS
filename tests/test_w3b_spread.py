"""W3b proof: the land changes during an episode, out of sight, within limits.

    py -3 tests/test_w3b_spread.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import random

import config_rl
from entities import Entity
from island_gen import ISLAND_VOLCANIC, ISLAND_MUSHROOM, ISLAND_TUNDRA
from world import (World, TILE_EMPTY, TILE_OCEAN, TILE_LAVA, TILE_ACID, TILE_ICE,
                   TILE_OBJECT, TILE_BRIDGE, TREES, SPREAD_GAP, GROVE_RANGE)

FAR = 1 << 30          # an agent nowhere near anything


def world(seed=1):
    random.seed(seed)
    w = World(seed=seed)
    for i in range(1, 6):
        w.materialise(i)
    return w


def changes(w, n, ax=FAR, ay=FAR):
    for _ in range(n):
        w._tick_spread(ax, ay)


def count(w, iid, tile_type):
    land = w.island_registry.get_island(iid).land_tiles
    return sum(1 for p in land if w.tiles[p].tile_type == tile_type)


def test_lava_and_acid_move_but_do_not_grow():
    w = world()
    for iid, hz in ((ISLAND_VOLCANIC, TILE_LAVA), (ISLAND_MUSHROOM, TILE_ACID)):
        before = {p for p in w.island_registry.get_island(iid).land_tiles
                  if w.tiles[p].tile_type == hz}
        changes(w, 30)
        after = {p for p in w.island_registry.get_island(iid).land_tiles
                 if w.tiles[p].tile_type == hz}
        assert before, f"island {iid} has no hazard to move"
        assert len(after) <= len(before), f"{hz}: {len(before)} -> {len(after)} grew"
        assert len(after) >= len(before) - 2, f"{hz}: {len(before)} -> {len(after)} shrank"
        assert after != before, f"{hz} never moved"


def test_volcanic_land_and_tundra_ice_grow_within_reach():
    w = world()
    land0 = {i: len(w.island_registry.get_island(i).land_tiles)
             for i in (ISLAND_VOLCANIC, ISLAND_TUNDRA)}
    ice0 = count(w, ISLAND_TUNDRA, TILE_ICE)
    changes(w, 300)
    for iid in land0:
        grown = len(w.island_registry.get_island(iid).land_tiles) - land0[iid]
        assert grown > 0, f"island {iid} never grew"
    assert count(w, ISLAND_TUNDRA, TILE_ICE) > ice0, "no ice shelf"
    assert max(w._reach.values()) <= 20, "grew past the biggest reach"
    for p, k in w._reach.items():
        iid = w.tiles[p].island_id
        if iid == ISLAND_VOLCANIC:
            assert k <= 8, f"volcanic land {k} tiles out"


def test_grown_land_keeps_the_sea_gap_and_bridges():
    w = world()
    changes(w, 300)
    reg = w.island_registry
    for p in w._reach:
        own = w.tiles[p].island_id
        for x in range(p[0] - SPREAD_GAP, p[0] + SPREAD_GAP + 1):
            for y in range(p[1] - SPREAD_GAP, p[1] + SPREAD_GAP + 1):
                assert reg._tile_map.get((x, y)) in (None, own), f"{p} touches another island"
                t = w.tiles.get((x, y))
                assert not (t and t.tile_type == TILE_BRIDGE), f"{p} grew onto a bridge"


def test_nothing_changes_where_the_agent_can_see():
    w = world()
    vol = w.island_registry.get_island(ISLAND_VOLCANIC)
    ax, ay = next(iter(p for p in vol.internal_hazard if w.tiles[p].tile_type == TILE_LAVA))
    r = config_rl.PATCH_RADIUS
    box = [(ax + dx, ay + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    seen = {p: (w._tile(*p).tile_type, w._tile(*p).object_id) for p in box}
    changes(w, 200, ax, ay)
    assert all((w.tiles[p].tile_type, w.tiles[p].object_id) == seen[p] for p in box), \
        "the land changed in front of the agent"


def test_lava_burns_common_things_not_keys():
    w = world()
    vol = w.island_registry.get_island(ISLAND_VOLCANIC)
    spots = [p for p in vol.land_tiles if w.tiles[p].tile_type == TILE_EMPTY][:3]
    w._put(spots[0], "0001")                  # apple
    w._put(spots[1], "0006")                  # key
    wolf = Entity.from_id("E011", *spots[2])
    w.entities.append(wolf)
    for p in spots:
        w.cover(p, TILE_LAVA)
    assert w.tiles[spots[0]].object_id == "" and w.tiles[spots[0]].tile_type == TILE_LAVA, "apple survived"
    assert w.tiles[spots[1]].object_id == "0006", "key burned"
    assert not wolf.alive, "creature survived the lava"


def test_groves_stay_in_range():
    w = world()
    changes(w, 200)
    for iid, start in w._trees0.items():
        land = w.island_registry.get_island(iid).land_tiles
        n = sum(1 for p in land if w.tiles[p].object_id in TREES)
        assert start * GROVE_RANGE[0] - 4 <= n <= start * GROVE_RANGE[1] + 4, \
            f"island {iid}: {start} trees -> {n}"


def test_switch_off_means_fixed_land():
    config_rl.W3_SPREAD = False
    try:
        w = world()
        before = {p: (t.tile_type, t.object_id) for p, t in w.tiles.items()}
        for _ in range(1500):
            w.tick_life()
        assert all(w.tiles[p].tile_type == v[0] for p, v in before.items()
                   if w.tiles[p].tile_type != TILE_OBJECT and v[0] != TILE_OBJECT)
        assert not w._reach and not w._trees0, "spread ran with W3_SPREAD off"
    finally:
        config_rl.W3_SPREAD = True


def test_runs_inside_tick_life():
    w = world()
    for _ in range(1000):
        w.tick_life()
    assert w._trees0, "tick_life never made a change"


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
