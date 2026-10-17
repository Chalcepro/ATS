"""W1 proof: moving creatures into data changes nothing.

    py -3 tests/test_w1_fingerprint.py --record   # once, BEFORE the change
    py -3 tests/test_w1_fingerprint.py            # after: must match tick for tick

It passed at commit db30fb9 (W1). W2 changes creature behaviour on purpose,
so from W2 on it reports DIFFERENT; it stays as the record of the W1 proof
and the tool for the next "no behaviour change" step (re-record first).

Two parts, both hashed every tick:
  worlds    - 5 fixed seeds x 2000 ticks of the real env, random actions
              from a fixed generator. Covers the state vector, rewards,
              deaths, and every creature the agent's island has.
  creatures - every creature in entities.json, spawned two tiles from a
              fake agent that walks in, stands, hits it, walks away. Covers
              the creatures random walking never meets (bosses, Void, sharks,
              Creeper, Magma Cube split).
"""
from __future__ import annotations

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import hashlib
import json
import random
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "w1_fingerprint.json"
SEEDS = [11, 222, 3333, 44444, 555555]
TICKS = 2000


def _h(*parts) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:12]


def _ents(world):
    return tuple((e.entity_id, e.x, e.y, e.hp, e.state, e.telegraph_ticks,
                  e.alive, e.exploded) for e in world.entities)


def worlds() -> dict:
    from ats_env import ATSEnvironment
    import config_rl
    out = {}
    for seed in SEEDS:
        random.seed(seed)
        acts = random.Random(seed * 7 + 1)
        env = ATSEnvironment(world_seed=seed, vary_world=False, use_loop=False)
        s = env.reset()
        ticks = []
        for _ in range(TICKS):
            s, r, done, _info = env.step(acts.randrange(config_rl.ACTION_SIZE))
            a = env.agent
            ticks.append(_h(tuple(round(v, 6) for v in s), round(r, 6), done,
                            a.x, a.y, a.health, env.death_cause, _ents(env.world)))
            if done:
                s = env.reset()
        out[str(seed)] = ticks
    return out


def creatures() -> dict:
    from item_ids import REGISTRY
    from world import World
    world = World(seed=99)
    ox, oy = world.agent_spawn
    out = {}
    for eid in sorted(REGISTRY.entities):
        random.seed(eid)
        world.entities = []
        world.spawn_entity(eid, ox, oy)
        ent = world.entities[-1]
        ticks = []
        # walk in from 6 tiles, stand next to it, hit it, walk off
        path = [(ox - 6 + min(i, 5), oy) for i in range(30)] + \
               [(ox - 1, oy)] * 20 + [(ox - 1 - i // 3, oy) for i in range(30)]
        for t, (ax, ay) in enumerate(path):
            dmg = ent.tick_ai(ax, ay, world)
            if t in (35, 40, 45):
                ent.take_damage(3, attacker_pos=(ax, ay), world=world)
            ticks.append(_h(dmg, _ents(world)))
        out[eid] = ticks
    return out


def main() -> int:
    got = {"worlds": worlds(), "creatures": creatures()}
    if "--record" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got), encoding="utf-8")
        print("recorded", GOLDEN)
        return 0
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    bad = 0
    for part in ("worlds", "creatures"):
        for key, ticks in want[part].items():
            mine = got[part].get(key, [])
            diff = next((i for i, (a, b) in enumerate(zip(ticks, mine)) if a != b),
                        None if len(ticks) == len(mine) else min(len(ticks), len(mine)))
            print("  %-10s %-8s %s" % (part, key, "ok" if diff is None
                                         else "DIFFERS from tick %d" % diff))
            bad += diff is not None
    print("SAME" if not bad else "%d DIFFER" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
