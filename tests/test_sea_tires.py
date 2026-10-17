"""Swimming tires you (config_rl.SEA_TIRES, 2026-10-15).

    py -3 tests/test_sea_tires.py

Each tick at sea costs stamina and stamina does not refill there; with none
left you drown. On land stamina comes back. A short dip is survivable, a long
swim is not.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import config_rl
from agent import Agent
from rewards import RewardEngine
from world import World

SEA = (5000, 5000)                     # far out at sea, no island near


def setup(at):
    w = World(seed=7)
    a = Agent(w)
    a.x, a.y = at
    return a, RewardEngine()


def test_stamina_drains_at_sea_and_refills_on_land():
    a, r = setup(SEA)
    for _ in range(30):
        a.tick_stats(r)
    assert a.stamina == 100 - 30 * config_rl.SWIM_STAMINA_COST, a.stamina
    a.x, a.y = a.world.agent_spawn
    before = a.stamina
    for _ in range(10):
        a.tick_stats(r)
    assert a.stamina > before


def test_short_dip_survives_long_swim_drowns():
    a, r = setup(SEA)
    for _ in range(90):                 # a dip: stamina left, no harm
        a.tick_stats(r)
    assert a.health == 100 and a.stamina > 0
    for _ in range(300):                # stays out: drowns
        a.tick_stats(r)
        a.hunger = 100                  # keep starvation out of it
        if a.health <= 0:
            break
    assert a.health <= 0 and a.last_damage_cause == "drowning", (a.health, a.last_damage_cause)


def test_flag_off_is_the_free_sea():
    old = config_rl.SEA_TIRES
    config_rl.SEA_TIRES = False
    try:
        a, r = setup(SEA)
        for _ in range(300):
            a.tick_stats(r)
            a.hunger = 100
        assert a.stamina == 100 and a.health == 100
    finally:
        config_rl.SEA_TIRES = old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
