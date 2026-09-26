"""The combat tier: items, weapons and things that fight back.

    py -3 test_combat.py

What these check is not "combat exists". The tier this replaces already had
`hostiles`, `can_attack` and `can_pick` on five rungs and none of it was a
fight: the hostile never moved, died to one blow, there was no weapon, and
items on the floor were not in the observation at all. Every claim below is
therefore about a mechanism that can be broken while the rung still runs and
still reports a number.
"""
from __future__ import annotations

import sys

import config_rl
import curriculum as C

FAIL = []


def check(name, got, ok):
    print("  %-56s %-20s %s" % (name, got, "ok" if ok else "FAILED"))
    if not ok:
        FAIL.append(name)


def room(**kw):
    base = dict(walls=False, hazards=0, goals=1, respawn=False, damage=True,
                can_pick=True, can_attack=True, max_steps=200, target=1,
                hostile_hp=3, hostile_damage=8)
    base.update(kw)
    return C.Stage("test", base.pop("grid", 9), **base)


def patch_of(env):
    s = env._state()
    p = config_rl.IDX_PATCH_START
    return s[p:p + config_rl.PATCH_TILES]


print("\n=== the stage carries combat, and keeps it when it grows ===")
def rung(name):
    return next(s for s in C.combat_tier() if s.name == name)


st = rung("warden")
check("warden has hostiles", st.hostiles, st.hostiles >= 4)
check("warden has swords", st.swords, st.swords > 0)
check("hostiles take more than one blow", st.hostile_hp, st.hostile_hp > 1)
g = st.grown()
check("grown keeps the sword", g.swords, g.swords == st.swords)
check("grown keeps hostile hp", g.hostile_hp, g.hostile_hp == st.hostile_hp)
check("grown keeps the chase", g.chase, g.chase == st.chase)
check("grown keeps guarding", g.guards, g.guards is True)
# The rung whose lesson is the encounter rate must not dilute as it grows.
check("grown adds hostiles with the room", "%d -> %d" % (st.hostiles, g.hostiles),
      g.hostiles > st.hostiles)

print("\n=== a rung that does not ask for combat does not get any ===")
low = C.default_ladder()[0]
check("nursery has no swords", low.swords, low.swords == 0)
check("nursery hostiles cannot chase", low.chase, low.chase == 0)
check("nursery hostile dies in one blow", low.hostile_hp, low.hostile_hp == 1)
env = C.CurriculumEnv(low, seed=1)
check("nursery places no swords", len(env.swords), len(env.swords) == 0)

print("\n=== items and hostiles are actually visible ===")
# Before this, `good` and `trash` were placed, rewarded on pick-up, and
# absent from the observation - so every can_pick rung rewarded a lottery.
st = room(grid=9, swords=1, hostiles=1, chase=0)
env = C.CurriculumEnv(st, seed=3)
env.ax, env.ay = 4, 4
env.swords = [(4, 3)]
env.hostiles = [(5, 4)]
env.hostile_hp = [3]
env._observe()
check("a sword reads as TILE_OBJECT", env._view(4, 3), env._view(4, 3) == C.T_OBJECT)
check("a hostile reads as TILE_HOSTILE", env._view(5, 4), env._view(5, 4) == C.T_HOSTILE)
check("the sword is in the patch", C.T_OBJECT in patch_of(env),
      C.T_OBJECT in patch_of(env))
check("the hostile is in the patch", C.T_HOSTILE in patch_of(env),
      C.T_HOSTILE in patch_of(env))
# Terrain must stay terrain, or the maze validator and hazard placer stop
# being able to see the cell as free floor.
check("the grid itself is untouched", env.grid[3][4], env.grid[3][4] == C.T_FLOOR)

print("\n=== memory remembers items, and does not remember ghosts ===")
check("the sword is remembered", env.seen.get((4, 3)), env.seen.get((4, 3)) == C.T_OBJECT)
check("the hostile is NOT remembered", env.seen.get((5, 4)),
      env.seen.get((5, 4)) == C.T_FLOOR)

print("\n=== picking a sword up ===")
env = C.CurriculumEnv(st, seed=3)
env.ax, env.ay = 4, 4
env.swords = [(4, 4)]
env.hostiles = []
env.hostile_hp = []
env._observe()
check("starts unarmed", env.armed, env.armed is False)
_, r, _, info = env.step(config_rl.ACT_PICK_UP)
check("picking it up pays", "%+.2f" % r, r > 0)
check("now armed", info["armed"], info["armed"] is True)
check("the sword is gone from the floor", len(env.swords), len(env.swords) == 0)
check("and gone from memory too", env.seen.get((4, 4)),
      env.seen.get((4, 4)) != C.T_OBJECT)
inv = env._state()[config_rl.IDX_INV_START]
check("and it is in the inventory slot", "%.2f" % inv,
      abs(inv - C.STONE_SWORD / config_rl.ITEM_ID_SCALE) < 1e-6)

print("\n=== the sword is why the sword is worth having ===")


def swings_to_kill(armed):
    e = C.CurriculumEnv(room(grid=9, swords=0, hostiles=1, chase=0), seed=5)
    e.ax, e.ay = 4, 4
    e.hostiles = [(4, 3)]
    e.hostile_hp = [3]
    e.armed = armed
    e.health = 10 ** 6          # measuring the weapon, not the survival
    n = 0
    while e.hostiles and n < 10:
        e.step(config_rl.ACT_ATTACK)
        n += 1
    return n


check("bare-handed takes three blows", swings_to_kill(False),
      swings_to_kill(False) == 3)
check("armed takes one", swings_to_kill(True), swings_to_kill(True) == 1)

print("\n=== swinging at nothing is not free ===")
env = C.CurriculumEnv(room(grid=9, swords=0, hostiles=0, chase=0), seed=7)
env.hostiles, env.hostile_hp = [], []
_, r, _, _ = env.step(config_rl.ACT_ATTACK)
check("a swing at air costs", "%+.3f" % r, r < 0)

print("\n=== hostiles move, and only when the stage says so ===")
st = room(grid=11, swords=0, hostiles=1, chase=2, aggro=20)
env = C.CurriculumEnv(st, seed=11)
env.ax, env.ay = 2, 2
env.hostiles = [(8, 2)]
env.hostile_hp = [3]
start = env.hostiles[0]
for _ in range(6):
    env.step(config_rl.ACT_WAIT)
check("a chaser closes in", "%s -> %s" % (start, env.hostiles[0]),
      abs(env.hostiles[0][0] - 2) < abs(start[0] - 2))

st = room(grid=11, swords=0, hostiles=1, chase=0)
env = C.CurriculumEnv(st, seed=11)
env.ax, env.ay = 2, 2
env.hostiles = [(8, 2)]
env.hostile_hp = [3]
for _ in range(6):
    env.step(config_rl.ACT_WAIT)
check("a static hostile stays put", env.hostiles[0], env.hostiles[0] == (8, 2))

print("\n=== outside its aggro radius it does not home in ===")
st = room(grid=15, swords=0, hostiles=1, chase=1, aggro=3)
env = C.CurriculumEnv(st, seed=13)
env.ax, env.ay = 1, 1
env.hostiles = [(12, 12)]
env.hostile_hp = [3]
before = abs(12 - 1) + abs(12 - 1)
for _ in range(8):
    env.step(config_rl.ACT_WAIT)
after = abs(env.hostiles[0][0] - 1) + abs(env.hostiles[0][1] - 1)
check("it wanders rather than beelines", "%d -> %d" % (before, after),
      after > before - 8)

print("\n=== a blow is a turn, not an aura ===")
# It used to damage every tick while moving every `chase` ticks, so a
# pursuer hit twice as often as it could act.
st = room(grid=9, swords=0, hostiles=1, chase=2, aggro=0)
env = C.CurriculumEnv(st, seed=17)
env.ax, env.ay = 4, 4
env.hostiles = [(4, 3)]
env.hostile_hp = [3]
env.health = 100
env.steps = 0
for _ in range(4):
    env.step(config_rl.ACT_WAIT)
lost = 100 - env.health
check("four ticks beside it costs two blows, not four", lost,
      lost == 2 * st.bite)

print("\n=== guards stand between you and the goal ===")
st = rung("armed")
on_route = 0
for k in range(40):
    e = C.CurriculumEnv(st, seed=500 + k)
    path = set(e._direct_path())
    if any(h in path for h in e.hostiles):
        on_route += 1
check("most rooms put a hostile on the route", "%d of 40" % on_route,
      on_route >= 30)

print("\n=== hunting pays worse than going home ===")
# Success is reaching the goal. If clearing the room paid better, the
# critic would be handed a second objective to confuse with distance -
# which is exactly what the trail rungs did to the bearing.
check("a kill is worth less than the goal", "%.2f < %.2f" % (C.R_KILL, C.R_GOAL),
      C.R_KILL < C.R_GOAL)

print("\n=== the tier is winnable by a competent agent ===")
for st in C.combat_tier():
    v = C.best_case_return(st, trials=12)
    check("%s scores positive" % st.name, "%+.2f" % v, v > 0)

print("\n=== and it is on the ladder ===")
names = [s.name for s in C.default_ladder()]
for want in ("armed", "hunted", "warden"):
    check("ladder has %s" % want, want in names, want in names)
check("combat can be switched off",
      [s.name for s in C.default_ladder(with_combat=False)].count("warden"),
      "warden" not in [s.name for s in C.default_ladder(with_combat=False)])

print("\n" + "=" * 78)
if FAIL:
    print("FAILED (%d): %s" % (len(FAIL), ", ".join(FAIL)))
    sys.exit(1)
print("all checks passed")
