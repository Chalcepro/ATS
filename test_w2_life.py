"""W2 proof: creatures live their own lives, decided by tags in the data.

    py -3 test_w2_life.py

Each scene puts a few creatures on open ground of a real world and checks one
behaviour: hunting, fleeing, grazing, the grudge, staying on land, births.
"""
import random

from entities import Entity
from world import World, Tile, TILE_EMPTY

AGENT_FAR = (10_000, 10_000)          # an agent nowhere near the scene


def open_patch(world, size=9):
    """Clear a size x size patch of open ground at the spawn; its top-left."""
    x0, y0 = world.agent_spawn
    isl = world.starting_island.island_id
    for i in range(size):
        for j in range(size):
            world.tiles[(x0 + i, y0 + j)] = Tile(tile_type=TILE_EMPTY, island_id=isl)
    return x0, y0


def scene(*placed):
    """A fresh world holding only the given creatures: (id, dx, dy)."""
    random.seed(3)
    w = World(seed=7)
    w.entities = []
    x0, y0 = open_patch(w)
    for eid, dx, dy in placed:
        w.spawn_entity(eid, x0 + dx, y0 + dy)
    return w, x0, y0


def run(w, ticks, agent=AGENT_FAR):
    dmg = 0
    for _ in range(ticks):
        w.index_creatures()
        for e in list(w.entities):
            dmg += e.tick_ai(agent[0], agent[1], w)
    return dmg


def dist(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


def test_hungry_wolf_catches_rabbit():
    w, *_ = scene(("E011", 0, 0), ("E016", 3, 0))
    wolf, rabbit = w.entities
    wolf.hunger = wolf.spec.hunger_max          # starving
    run(w, 200)
    assert not rabbit.alive, "the wolf never caught the rabbit"
    assert wolf.hunger < wolf.spec.hunger_max // 2, "eating didn't feed it"


def test_fed_wolf_leaves_prey_alone():
    w, *_ = scene(("E011", 0, 0), ("E016", 3, 0))
    wolf, rabbit = w.entities
    run(w, 100)
    assert rabbit.alive and wolf.job != "hunt"


def test_deer_flees_wolf():
    w, *_ = scene(("E011", 3, 3), ("E015", 4, 3))
    wolf, deer = w.entities
    before = dist(wolf, deer)
    for _ in range(4):
        w.index_creatures()
        deer.tick_ai(*AGENT_FAR, w)
    assert deer.job == "flee" and dist(wolf, deer) > before


def test_wolf_ignores_agent_until_hit():
    w, x0, y0 = scene(("E011", 0, 0))
    wolf = w.entities[0]
    wolf.hunger = wolf.spec.hunger_max          # even starving
    agent = (x0 + 5, y0)                        # outside its space of 2
    run(w, 30, agent)
    assert wolf.job != "chase", "a wolf chased the agent unprovoked"

    wolf.x, wolf.y = x0, y0
    wolf.take_damage(1, attacker_pos=agent)     # the agent hits it
    start = abs(wolf.x - agent[0]) + abs(wolf.y - agent[1])
    run(w, 10, agent)
    now = abs(wolf.x - agent[0]) + abs(wolf.y - agent[1])
    assert wolf.angry and now < start, "a hit wolf didn't come after the agent"

    run(w, 1, (x0 + 100, y0))                   # the agent is far away now
    assert not wolf.angry, "the grudge didn't end at a distance"


def test_grazer_grazes_when_hungry():
    w, *_ = scene(("E017", 3, 3))
    sheep = w.entities[0]
    sheep.hunger = sheep.spec.hunger_max
    run(w, 5)
    assert sheep.job == "graze" and sheep.hunger < sheep.spec.hunger_max


def test_herd_comes_back_together():
    w, *_ = scene(("E015", 0, 0), ("E015", 6, 6))
    a, b = w.entities
    start = dist(a, b)
    run(w, 60)
    assert dist(a, b) < start


def test_creatures_stay_on_land():
    random.seed(5)
    w = World(seed=7)
    for _ in range(3000):
        run(w, 1)
    off = [e for e in w.entities if e.alive and not e.spec.has("ocean")
           and w.tiles.get((e.x, e.y)) is None]
    assert not off, f"{len(off)} land creatures in the sea"


def test_island_refills_gradually():
    random.seed(9)
    w = World(seed=7)
    start = sum(w._start_counts[w.starting_island.island_id].values())
    for e in w.entities:
        e.alive = False                         # everything dies at once
    born_at = []
    for t in range(12000):
        if w.tick_life():
            born_at.append(t)
    alive = sum(e.alive for e in w.entities)
    assert alive == start, f"refilled to {alive}, wanted {start}"
    assert born_at[-1] - born_at[0] > 100, "everything was born at once"


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
