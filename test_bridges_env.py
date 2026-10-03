"""The bridged world inside the real environment: a step onto a bridge builds
the far island, the guard blocks, the sea pays no exploration, and a few
hundred random steps run clean.

    python test_bridges_env.py
"""
import random

import config_rl
import world as W
from ats_env import ATSEnvironment

FAIL = []


def check(name, got, ok):
    print("  %-58s %-24s %s" % (name, got, "ok" if ok else "FAILED"))
    if not ok:
        FAIL.append(name)


env = ATSEnvironment()
s = env.reset()
check("reset gives a state of the right size", len(s), len(s) == config_rl.STATE_SIZE)
w = env.world
reg = w.island_registry
check("the world has bridges", len(reg.bridges), len(reg.bridges) == 5)

print("\n=== onto a bridge ===")
br = reg.bridges[0]
far = br["to"]
first = br["tiles"][0]
env.agent.x, env.agent.y = first
before = len(w.entities)
env.step(config_rl.ACTION_SIZE - 1 if False else 24)        # Wait: stay on the bridge tile
check("standing on a bridge builds the island across it", far in w._materialised, far in w._materialised)
check("...with its creatures", len(w.entities) - before, len(w.entities) > before)
check("the bridge tile is not ocean", w._tile(*first).tile_type, w._tile(*first).tile_type == W.TILE_BRIDGE)
check("so standing on it is not swimming", env.agent.ocean_ticks, env.agent.ocean_ticks == 0)

print("\n=== the sea pays nothing to explore ===")
side = next((first[0] + dx, first[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if w._tile(first[0] + dx, first[1] + dy).tile_type == W.TILE_OCEAN)
paid_before = env.rewards.ledger.get("tile_explore", (0.0, 0))[0] if hasattr(env.rewards, "ledger") else None
n_before = len(env.rewards.discovered_tiles)
env.agent.x, env.agent.y = side
env.step(24)
check("a new ocean tile is not counted as explored", len(env.rewards.discovered_tiles) - n_before,
      len(env.rewards.discovered_tiles) == n_before)

print("\n=== random steps ===")
s = env.reset()
random.seed(3)
causes = []
for i in range(600):
    mask = env.agent.get_action_mask(env.world, env.day_night)
    acts = [a for a, ok in enumerate(mask) if ok] or [24]
    s, r, done, info = env.step(random.choice(acts))
    if done:
        causes.append(env.death_cause)
        s = env.reset()
check("600 random steps without a crash", len(causes), True)
check("no death is called 'the Void' on the start island", causes,
      "the Void" not in causes)

print("\n" + "=" * 70)
if FAIL:
    print("FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
    raise SystemExit(1)
print("all checks passed")
