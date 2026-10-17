"""Looking (interact), pointless toggles, sprint strides, farmable payouts.

    py -3 tests/test_look_and_stride.py

Measured 2026-10-17 on the trained brain in Loop rooms: ~40% of its actions
were sprint on/off and inventory open/close, it never pressed interact, and
sprinting drained stamina for no speed. See config_rl INSPECT,
POINTLESS_TOGGLE_COSTS, SPRINT_DOUBLE_STEP, FARMABLE_PAYOUTS.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import config_rl
import loop
from ats_env import ATSEnvironment

UP, DOWN, LEFT, RIGHT = (config_rl.ACT_MOVE_FORWARD, config_rl.ACT_MOVE_BACKWARD,
                         config_rl.ACT_MOVE_LEFT, config_rl.ACT_MOVE_RIGHT)
WAIT, INTERACT = config_rl.ACT_WAIT, config_rl.ACT_INTERACT
ON, OFF = config_rl.ACT_SPRINT_ON, config_rl.ACT_SPRINT_OFF
OPEN, CLOSE = config_rl.ACT_OPEN_INVENTORY, config_rl.ACT_CLOSE_INVENTORY

ENV = ATSEnvironment(world_seed=11, vary_world=False, use_loop=True)


def bare_room(rows=None, objects=None, creatures=(), start=(loop.MID, loop.MID)):
    """A plain stone room with nothing to do: a fixed place to test the body in."""
    g = loop._shell()
    for (x, y), ch in (rows or {}).items():
        g[y][x] = ch
    room = loop.Room("test", loop._rows(g), start, lambda env: False, loop.find_start,
                     objects=objects or {}, creatures=list(creatures))
    ENV.reset()
    ENV.max_ticks = 10 ** 6
    ENV.agent.health = 0
    ENV.step(WAIT)                              # into the Loop
    ENV._enter_room(room)
    while ENV.bearings:
        ENV.step(WAIT)
    ENV.rewards.tally.clear()
    return ENV


def paid(env, key):
    return env.rewards.tally.get(key, [0.0, 0])


def test_interact_is_always_there_and_looking_says_what_it_was():
    env = bare_room(objects={(loop.MID, loop.MID - 1): "0001"})    # an apple, faced (north)
    assert env.agent.get_action_mask(env.world, env.day_night)[INTERACT] == 1
    env.agent.facing = (0, -1)
    s, _, _, _ = env.step(INTERACT)
    assert s[config_rl.IDX_INSPECT_KIND] == 3 / config_rl.INSPECT_KINDS, "an apple is food"
    assert s[config_rl.IDX_INSPECT_FRESH] > 0.5
    assert paid(env, "inspect_new")[1] == 1
    env.agent.facing = (0, 1)                                       # bare floor south ...
    s, _, _, _ = env.step(INTERACT)
    assert s[config_rl.IDX_INSPECT_KIND] == 3 / config_rl.INSPECT_KINDS, \
        "... but a thing beside it beats bare ground"
    env.world._tile(loop.MID, loop.MID - 1).object_id = ""
    s, _, _, _ = env.step(INTERACT)
    assert s[config_rl.IDX_INSPECT_KIND] == 11 / config_rl.INSPECT_KINDS, "stone floor"


def test_first_look_at_a_kind_pays_once_and_staring_costs_a_wait():
    env = bare_room()
    env.agent.facing = (0, -1)
    env.step(INTERACT)
    env.step(INTERACT)                          # the same floor, straight away
    assert paid(env, "inspect_new") == [config_rl.INSPECT_NEW_REWARD, 1]
    assert paid(env, "look_again")[1] == 1
    env.step(LEFT)
    env.agent.facing = (0, -1)
    env.step(INTERACT)                          # floor again, after doing something: free
    assert paid(env, "inspect_new")[1] == 1 and paid(env, "look_again")[1] == 1


def test_a_creature_beside_it_is_what_it_looks_at():
    env = bare_room(creatures=[("E011", loop.MID + 1, loop.MID)])
    env.agent.facing = (0, -1)                  # facing away from the wolf
    s, _, _, _ = env.step(INTERACT)
    assert s[config_rl.IDX_INSPECT_KIND] == 1 / config_rl.INSPECT_KINDS, "hostile"


def test_a_toggle_straight_back_costs_a_wait_and_a_real_one_is_free():
    env = bare_room()
    env.step(ON)
    env.step(OFF)                               # on then off: nothing happened
    assert paid(env, "pointless_toggle")[1] == 1
    env.step(OPEN)
    env.step(WAIT)
    env.step(CLOSE)                             # a wait between isn't doing anything
    assert paid(env, "pointless_toggle")[1] == 2
    env.step(ON)
    env.step(LEFT)
    env.step(OFF)                               # it ran: fine
    assert paid(env, "pointless_toggle")[1] == 2


def test_a_sprint_goes_two_tiles_and_costs_stamina_once():
    env = bare_room()
    a = env.agent
    x0, st = a.x, a.stamina
    env.step(LEFT)
    assert a.x == x0 - 1, "walking: one tile"
    env.step(ON)
    x1 = a.x
    env.step(LEFT)
    assert a.x == x1 - 2, "sprinting: two"
    assert a.stamina == st - 2 + 0, a.stamina   # one stride's cost (no regen while sprinting)


def test_a_stride_stops_at_water_and_walls():
    env = bare_room(rows={(loop.MID - 2, loop.MID): "~", (loop.MID, loop.MID - 2): "#"})
    a = env.agent
    env.step(ON)
    env.step(LEFT)
    assert a.x == loop.MID - 1, "stops at the water's edge"
    a.x, a.y = loop.MID, loop.MID
    env.step(UP)
    assert a.y == loop.MID - 1, "stops at the wall"
    assert paid(env, "wall_hit")[1] == 0, "a short second stride isn't a wall hit"


def test_out_of_stamina_a_sprint_is_a_walk():
    env = bare_room()
    a = env.agent
    env.step(ON)
    a.stamina = 1
    x = a.x
    env.step(LEFT)
    assert a.x == x - 1


def test_a_kind_it_already_has_pays_nothing_the_first_does():
    env = bare_room(objects={(loop.MID, loop.MID - 1): "0010", (loop.MID, loop.MID + 1): "0010"})
    env.step(config_rl.ACT_PICK_UP)
    env.step(config_rl.ACT_PICK_UP)
    assert paid(env, "pickup_discovery")[1] == 1
    assert paid(env, "pickup_default")[1] == 0, "the second Wood pays nothing"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
