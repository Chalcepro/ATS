"""W4: the Loop (loop.py, docs/W4-LOOP-FLOW.md).

    py -3 test_loop.py
"""
import config_rl
import loop
from ats_env import ATSEnvironment

UP, DOWN, LEFT, RIGHT = (config_rl.ACT_MOVE_FORWARD, config_rl.ACT_MOVE_BACKWARD,
                         config_rl.ACT_MOVE_LEFT, config_rl.ACT_MOVE_RIGHT)
WAIT, ATTACK, INTERACT = config_rl.ACT_WAIT, config_rl.ACT_ATTACK, config_rl.ACT_INTERACT
USE1 = config_rl.ACT_USE_SLOT_BASE + 1

ENV = ATSEnvironment(world_seed=11, vary_world=False, use_loop=True)


def fresh():
    ENV.reset()
    ENV.max_ticks = 10 ** 6
    return ENV


def die(env):
    env.agent.health = 0
    env.agent.last_damage_cause = "drowning"
    env.step(WAIT)


def settle(env):
    while env.bearings:
        env.step(WAIT)


def into(env, name):
    env._enter_room(loop.BY_NAME[name])
    settle(env)


def test_world_death_lands_in_the_loop():
    env = fresh()
    die(env)
    assert env.in_loop and not env.done and env.deaths == 1
    assert env.death_causes == ["drowning"]
    s = env._state()
    assert s[config_rl.IDX_LOOP_SIGNAL] == 1.0


def test_bearings_hold_everything_still():
    env = fresh()
    die(env)
    env._enter_room(loop.BY_NAME["wolf"])
    pos, hp = (env.agent.x, env.agent.y), env.agent.health
    for _ in range(config_rl.LOOP_BEARINGS_TICKS):
        env.step(UP)
    assert (env.agent.x, env.agent.y) == pos and env.agent.health == hp
    env.step(DOWN)
    assert (env.agent.x, env.agent.y) != pos, "moves once the bearings are over"


def test_a_closed_door_loops_back_into_the_room():
    env = fresh()
    die(env)
    into(env, "hungry")
    a = env.agent
    a.x, a.y = loop.SIZE - 2, loop.MID       # in the doorway row
    env.step(RIGHT)
    assert a.x == loop.SIZE - 1
    env.step(RIGHT)                         # out of the east door ...
    assert env.in_loop and a.x == 0 and a.y == loop.MID   # ... in at the west door
    assert env.loop_room.name == "hungry"


def test_held_weapon_only_and_the_wolf_room():
    env = fresh()
    die(env)
    into(env, "wolf")
    a = env.agent
    assert a.held_damage() == 1, "arrives holding the apple"
    assert env._state()[config_rl.IDX_HELD] == 0.0
    env.step(USE1)                           # take the sword in hand
    assert a.held_damage() == 5
    assert env._state()[config_rl.IDX_HELD] == 5 / 20
    wolf = env.world.entities[0]
    a.x, a.y = wolf.x, wolf.y + 1            # just south of it, facing north
    a.facing = (0, -1)
    a.health = 100
    for _ in range(3):
        env.step(ATTACK)
        a.x, a.y = wolf.x, wolf.y + 1
    assert not wolf.alive and env.world.doors_open


def test_open_doors_lead_on_and_three_rooms_drop_it_by_a_wolf():
    env = fresh()
    die(env)
    env.loop_goals = config_rl.LOOP_GOALS_BEFORE_DROP - 1
    into(env, "wolf")
    env.world.entities[0].alive = False
    env.step(WAIT)                           # goal met -> doors open
    assert env.world.doors_open
    a = env.agent
    a.x, a.y = loop.SIZE - 1, loop.MID
    env.step(RIGHT)                          # through the open east door
    assert not env.in_loop and env.world is env.real_world
    wolves = [(e.x, e.y) for e in env.world.entities if e.alive and e.entity_id in loop.WOLVES]
    near = min(abs(x - a.x) + abs(y - a.y) for x, y in wolves)
    assert near <= 12, near
    assert env.bearings == config_rl.LOOP_BEARINGS_TICKS and a.health == 100


def test_fewer_goals_go_to_another_room():
    env = fresh()
    die(env)
    env.loop_goals = 0
    into(env, "lever")
    lever = next(p for p, t in env.world.tiles.items() if t.object_id == "0088")
    assert env._state()[config_rl.IDX_MEM_ROUTE_START:config_rl.IDX_MASK_START].count(1.0) > 0
    env.agent.x, env.agent.y = lever[0], lever[1] + 1
    assert env.agent.get_action_mask(env.world, env.day_night)[INTERACT] == 1
    env.step(INTERACT)
    assert env.in_loop and env.loop_room.name != "lever" and env.loop_goals == 1


def test_shore_back_into_the_water_starts_over():
    env = fresh()
    die(env)
    into(env, "shore")
    a = env.agent
    a.x, a.y = 1, 1                          # out on the stone
    env.step(WAIT)
    a.x, a.y = 5, 5                          # and back into the pool
    env.step(WAIT)
    assert (a.x, a.y) == loop.BY_NAME["shore"].start() and env.loop_room_deaths == 1


def test_shore_goal_after_staying_on_land():
    env = fresh()
    die(env)
    into(env, "shore")
    a = env.agent
    a.x, a.y = 1, 1
    for _ in range(config_rl.LOOP_SHORE_TICKS):
        env.step(WAIT)
    assert env.world.doors_open


def test_five_deaths_in_a_room_swap_it():
    env = fresh()
    die(env)
    into(env, "wolf")
    for _ in range(config_rl.LOOP_ROOM_DEATHS_TO_SWAP - 1):
        die(env)
        settle(env)
        assert env.loop_room.name == "wolf"
    die(env)
    assert env.loop_room.name != "wolf" and env.loop_room_deaths == 0


def test_world_route_channel_is_empty():
    env = fresh()
    s = env._state()
    assert not any(s[config_rl.IDX_MEM_ROUTE_START:config_rl.IDX_MASK_START])
    assert s[config_rl.IDX_LOOP_SIGNAL] == 0.0


def test_loop_off_death_ends_the_episode():
    env = ATSEnvironment(world_seed=11, vary_world=False, use_loop=False)
    env.reset()
    die(env)
    assert env.done and not env.in_loop


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
