"""The bridged world: islands close but never touching, one bridge per pair,
a guard on each, sharks kept at sea. Plain Python - no torch needed.

    python test_bridges.py
"""
import time

import config_rl
import island_gen as ig
import world as W
from entities import Entity, _passable

FAIL = []


def check(name, got, ok):
    print("  %-60s %-22s %s" % (name, got, "ok" if ok else "FAILED"))
    if not ok:
        FAIL.append(name)


def coast_gap(a, b):
    """True minimum distance between two land sets (Chebyshev-free: real
    Euclidean on the closest coastal tiles)."""
    def coast(t):
        return [p for p in t if any((p[0] + dx, p[1] + dy) not in t
                                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    ca, cb = coast(a), coast(b)
    # grid bucket b's coast so this is not 500 x 500 every time
    buckets = {}
    for x, y in cb:
        buckets.setdefault((x // 16, y // 16), []).append((x, y))
    best = 1e9
    for x, y in ca:
        bx, by = x // 16, y // 16
        for i in range(bx - 1, bx + 2):
            for j in range(by - 1, by + 2):
                for qx, qy in buckets.get((i, j), ()):
                    d = ((x - qx) ** 2 + (y - qy) ** 2) ** 0.5
                    if d < best:
                        best = d
    return best


for seed in (1, 7, 1337, 258598698, 388373227):
    print("\n=== seed %d ===" % seed)
    t0 = time.time()
    w = W.World(seed=seed)
    build = time.time() - t0
    reg = w.island_registry
    check("world builds in reasonable time", "%.2fs" % build, build < 8.0)
    land = {i.island_id: i.land_tiles for i in reg.islands}
    check("six islands", len(land), len(land) == 6)

    gaps = []
    ids = sorted(land)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            g = coast_gap(land[ids[a]], land[ids[b]])
            gaps.append(g)
    check("every coast is >= 8 tiles from every other", round(min(gaps), 1), min(gaps) >= ig.COAST_GAP - 0.01)
    check("...and the islands are close (nearest gap < 20)", round(min(gaps), 1), min(gaps) < 20)

    check("five bridges", len(reg.bridges), len(reg.bridges) == 5)
    for br in reg.bridges:
        tiles = br["tiles"]
        tag = "%d->%d" % (br["from"], br["to"])
        steps_ok = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(tiles, tiles[1:]))
        check("bridge %s: one tile wide, 4-connected, %d long" % (tag, len(tiles)), steps_ok, steps_ok and len(tiles) >= ig.COAST_GAP)
        first, last = tiles[0], tiles[-1]
        near = lambda t, iid: any(reg.get_island_at(t[0] + dx, t[1] + dy) == iid
                                  for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        check("bridge %s: starts at its island, ends at the other" % tag,
              (near(first, br["from"]), near(last, br["to"])), near(first, br["from"]) and near(last, br["to"]))
        mid = tiles[len(tiles) // 2]
        sides = [(mid[0] + dx, mid[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if (mid[0] + dx, mid[1] + dy) not in tiles]
        check("bridge %s: sea on both sides of the middle" % tag,
              [w._tile(*p).tile_type for p in sides], all(w._tile(*p).tile_type == W.TILE_OCEAN for p in sides))
        g = [e for e in w.entities if e.guard and (e.x, e.y) == br["guard"]]
        check("bridge %s: a guard stands on it" % tag, g[0].entity_id if g else None,
              bool(g) and g[0].data.get("type", "hostile") == "hostile")
        check("bridge %s: the guard blocks the way" % tag, w.in_bounds_passable(*br["guard"]),
              not w.in_bounds_passable(*br["guard"]))
        check("bridge %s: tiles say where they lead" % tag, w.bridge_target(*tiles[0]),
              w.bridge_target(*tiles[0]) == br["to"])

    # the guard holds its post
    g = next(e for e in w.entities if e.guard)
    post = (g.x, g.y)
    for _ in range(50):
        g.tick_ai(post[0] + 20, post[1] + 20, w)
    check("a guard does not wander off its post", (g.x, g.y), (g.x, g.y) == post)

    # far islands are built only when headed for
    far = reg.bridges[0]["to"]
    before = len(w.entities)
    check("the far island is not built yet", far in w._materialised, far not in w._materialised)
    built = w.materialise(far)
    check("stepping on its bridge builds it", built, built and far in w._materialised)
    check("...with its creatures", len(w.entities) - before, len(w.entities) > before)
    check("...only once", w.materialise(far), not w.materialise(far))

print("\n=== sharks stay at sea ===")
w = W.World(seed=1337)
br = w.island_registry.bridges[0]
sea = next((br["tiles"][3][0] + dx, br["tiles"][3][1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
           if w._tile(br["tiles"][3][0] + dx, br["tiles"][3][1] + dy).tile_type == W.TILE_OCEAN)
shark = Entity.from_id("E_SHARK", *sea)
w.entities.append(shark)
check("a shark may not step onto a bridge", _passable(w, *br["tiles"][3], True), not _passable(w, *br["tiles"][3], True))
check("...or onto land", _passable(w, 0, 0, True), not _passable(w, 0, 0, True))
spawn = (0, 0)
for _ in range(400):                       # chase the agent standing at the spawn
    shark.tick_ai(spawn[0], spawn[1], w)
check("after 400 ticks of chasing, it is still in the sea", w._tile(shark.x, shark.y).tile_type,
      w._tile(shark.x, shark.y).tile_type == W.TILE_OCEAN)

print("\n=== the old layout is one switch away ===")
config_rl.WORLD_BRIDGES = False
w = W.World(seed=1337)
check("no bridges", len(w.island_registry.bridges), len(w.island_registry.bridges) == 0)
check("islands back at their old places", w.island_registry.get_island(ig.ISLAND_BOSS).center,
      w.island_registry.get_island(ig.ISLAND_BOSS).center == ig.ISLAND_CENTERS[ig.ISLAND_BOSS])
config_rl.WORLD_BRIDGES = True

print("\n" + "=" * 72)
if FAIL:
    print("FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
    raise SystemExit(1)
print("all checks passed")
