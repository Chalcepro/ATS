"""W4: the Loop (loop.py, docs/W4-LOOP-FLOW.md).

    py -3 test_loop.py
"""
import random
from collections import Counter

import config_rl
import loop
from ats_env import ATSEnvironment

UP, DOWN, LEFT, RIGHT = (config_rl.ACT_MOVE_FORWARD, config_rl.ACT_MOVE_BACKWARD,
                         config_rl.ACT_MOVE_LEFT, config_rl.ACT_MOVE_RIGHT)
WAIT, ATTACK, INTERACT = config_rl.ACT_WAIT, config_rl.ACT_ATTACK, config_rl.ACT_INTERACT
USE = config_rl.ACT_USE_SLOT_BASE

ENV = ATSEnvironment(world_seed=11, vary_world=False, use_loop=True)


def fresh():
    ENV.reset()
    ENV.max_ticks = 10 ** 6
    return ENV


def die(env, cause="drowning"):
    env.agent.health = 0
    env.agent.last_damage_cause = cause
    env.step(WAIT)


def settle(env):
    while env.bearings:
        env.step(WAIT)


def into(env, room):
    env._enter_room(room)
    settle(env)
    return room


def tiles_of(room, ch):
    return [(x, y) for y, row in enumerate(room.layout) for x, c in enumerate(row) if c == ch]


def test_world_death_lands_in_the_loop():
    env = fresh()
    die(env)
    assert env.in_loop and not env.done and env.deaths == 1
    assert env.death_causes == ["drowning"]
    assert env._state()[config_rl.IDX_LOOP_SIGNAL] == 1.0


def test_bearings_hold_everything_still():
    env = fresh()
    die(env)
    env._enter_room(loop.fight_room(random.Random(1), "E011"))
    pos, hp = (env.agent.x, env.agent.y), env.agent.health
    for _ in range(config_rl.LOOP_BEARINGS_TICKS):
        env.step(UP)
    assert (env.agent.x, env.agent.y) == pos and env.agent.health == hp
    env.step(DOWN)
    assert (env.agent.x, env.agent.y) != pos, "moves once the bearings are over"


def test_a_closed_door_loops_back_into_the_room():
    env = fresh()
    die(env)
    room = into(env, loop.food_room(random.Random(2)))
    a = env.agent
    a.x, a.y = loop.SIZE - 2, loop.MID
    env.step(RIGHT)
    assert a.x == loop.SIZE - 1
    env.step(RIGHT)                         # out of the east door ...
    assert env.in_loop and (a.x, a.y) == (0, loop.MID)   # ... in at the west door
    assert env.loop_room is room


def test_held_weapon_only_and_a_fight():
    env = fresh()
    die(env)
    room = into(env, loop.fight_room(random.Random(3), "E011"))
    a = env.agent
    slot, weapon, _ = next(k for k in room.kit if k[0] != 0 and k[1] in loop.WEAPONS)
    dmg = loop.REGISTRY.get_item(weapon)["damage"]
    assert a.held_damage() == 1, "arrives holding something that isn't the weapon"
    assert env._state()[config_rl.IDX_HELD] == 0.0
    env.step(USE + slot)                     # take the weapon in hand
    assert a.held_damage() == dmg and env._state()[config_rl.IDX_HELD] == dmg / 20
    a.health = 100
    for wolf in list(env.world.entities):
        for _ in range(4):
            if not wolf.alive:
                break
            a.x, a.y, a.facing = wolf.x, wolf.y + 1, (0, -1)
            env.step(ATTACK)
        assert not wolf.alive
    assert env.world.doors_open


def test_open_doors_lead_on_and_three_rooms_drop_it_by_the_same_creature():
    env = fresh()
    die(env)
    env.loop_goals = config_rl.LOOP_GOALS_BEFORE_DROP - 1
    into(env, loop.fight_room(random.Random(4), "E011"))
    for e in env.world.entities:
        e.alive = False
    env.step(WAIT)                           # goal met -> doors open
    assert env.world.doors_open
    a = env.agent
    a.x, a.y = loop.SIZE - 1, loop.MID
    env.step(RIGHT)                          # through the open east door
    assert not env.in_loop and env.world is env.real_world
    wolves = [(e.x, e.y) for e in env.world.entities if e.alive and e.entity_id == "E011"]
    near = min(abs(x - a.x) + abs(y - a.y) for x, y in wolves)
    assert near <= 12, near
    assert env.bearings == config_rl.LOOP_BEARINGS_TICKS and a.health == 100


def test_a_lever_is_the_way_out_and_a_hidden_one_has_a_route():
    env = fresh()
    die(env)
    rng = random.Random(5)
    room = loop.lever_room(rng)
    while not room.route_hint:
        room = loop.lever_room(rng)
    env.loop_goals = 0
    into(env, room)
    route = env._state()[config_rl.IDX_MEM_ROUTE_START:config_rl.IDX_MASK_START]
    assert route.count(1.0) > 0, "a hidden lever gets the route"
    lever = next(p for p, i in room.objects.items() if i == loop.LEVER)
    env.agent.x, env.agent.y = next(n for n in loop._around(*lever)
                                    if room.layout[n[1]][n[0]] == "."
                                    and n not in room.objects)
    assert env.agent.get_action_mask(env.world, env.day_night)[INTERACT] == 1
    env.step(INTERACT)
    assert env.in_loop and env.loop_room is not room and env.loop_goals == 1


def test_shore_back_into_the_water_starts_over_and_staying_out_wins():
    env = fresh()
    die(env)
    room = into(env, loop.shore_room(random.Random(6)))
    a = env.agent
    land, sea = tiles_of(room, ".")[0], tiles_of(room, "~")[0]
    a.x, a.y = land
    env.step(WAIT)
    a.x, a.y = sea                           # back into the pool
    env.step(WAIT)
    assert (a.x, a.y) == room.start and env.loop_room_deaths == 1
    settle(env)
    a.x, a.y = land
    for _ in range(config_rl.LOOP_SHORE_TICKS):
        env.step(WAIT)
    assert env.world.doors_open


def test_food_room_eat_to_open():
    env = fresh()
    die(env)
    room = into(env, loop.food_room(random.Random(7)))
    a = env.agent
    a.inventory[6] = {"id": "0001", "count": 1}
    env.step(USE + 6)
    assert env.world.doors_open


def test_lava_room_fetch_across():
    env = fresh()
    die(env)
    room = into(env, loop.lava_room(random.Random(8)))
    (spot, item), = room.objects.items()
    env.agent.x, env.agent.y = spot
    env.step(config_rl.ACT_PICK_UP)
    assert env.world.doors_open


def test_five_deaths_in_a_room_swap_it():
    env = fresh()
    die(env)
    room = into(env, loop.fight_room(random.Random(9), "E011"))
    for _ in range(config_rl.LOOP_ROOM_DEATHS_TO_SWAP - 1):
        die(env, "Wolf")
        settle(env)
        assert env.loop_room is room, "a retry is the same room"
    die(env, "Wolf")
    assert env.loop_room.kind != "fight" and env.loop_room_deaths == 0


def test_out_of_time_is_a_failed_try_and_five_swap_the_room():
    env = fresh()
    die(env)
    room = into(env, loop.lever_room(random.Random(12)))
    for _ in range(config_rl.LOOP_ROOM_TIMEOUT + 1):
        env.step(WAIT)
    assert env.loop_room is room and env.loop_room_deaths == 1
    assert env.rewards.tally["loop_timeout"] == [config_rl.LOOP_TIMEOUT_PENALTY, 1]
    for _ in range(config_rl.LOOP_ROOM_DEATHS_TO_SWAP - 1):
        settle(env)
        for _ in range(config_rl.LOOP_ROOM_TIMEOUT + 1):
            env.step(WAIT)
    assert env.loop_room.kind != "lever"


def test_won_room_but_no_door_found_leaves_anyway():
    env = fresh()
    die(env)
    env.loop_goals = 0
    room = into(env, loop.lava_room(random.Random(13)))   # standing still is safe here
    env.world.doors_open = True
    env.loop_room_ticks = 0
    for _ in range(config_rl.LOOP_ROOM_TIMEOUT + 1):
        env.step(WAIT)
    assert env.loop_room is not room and env.loop_goals == 1


def test_each_room_kind_tallies_how_its_tries_ended():
    env = fresh()
    die(env)
    into(env, loop.lever_room(random.Random(12)))
    key = env.loop_room.name
    for _ in range(config_rl.LOOP_ROOM_TIMEOUT + 1):
        env.step(WAIT)                       # out of time
    settle(env)
    die(env, "Wolf")                         # died
    settle(env)
    env.world.lever_pulled = lambda: True    # won
    env.step(WAIT)
    assert env.room_stats[key] == [1, 1, 1, 0], env.room_stats


def test_rooms_vary():
    rng = random.Random(10)
    rooms = [loop.pick_room(None, rng) for _ in range(200)]
    assert len(Counter(r.kind for r in rooms)) == 5
    assert len({r.about for r in rooms}) > 25
    assert len({tuple(r.layout) for r in rooms}) > 150, "layouts are rolled, not fixed"


def test_cause_picks_the_matching_room_about_half_the_time():
    rng = random.Random(11)
    kinds = Counter(loop.pick_room("Wolf", rng).about for _ in range(400))
    wolf = sum(n for k, n in kinds.items() if k.startswith("fight Wolf"))
    assert 150 < wolf < 260, wolf


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
