"""W2b proof: a shark you can see coming.

    py -3 test_w2b_shark.py

It surfaces inside the agent's view, waits, locks onto the agent's spot and
charges one tile per tick in a straight line. Contact kills. Moving after the
lock dodges it; moving before only changes its aim. On land it gives up.
"""
import random

import config_rl
from ats_env import ATSEnvironment
from world import World, TILE_SHARK

SEA = (5000, 5000)                     # far out at sea, no island near
R = config_rl.PATCH_RADIUS


def fresh():
    random.seed(1)
    w = World(seed=7)
    w.entities = []
    return w


def tick(shark, agent, w):
    return shark.tick_ai(agent[0], agent[1], w)


def test_surfaces_in_view_and_is_seen():
    w = fresh()
    shark = w.surface_shark(*SEA, R)
    assert shark and max(abs(shark.x - SEA[0]), abs(shark.y - SEA[1])) == R
    assert w.tile_type_at(shark.x, shark.y) == TILE_SHARK
    assert shark.spec.brain_id != 0, "the brain can't tell a shark from nothing"
    assert w.surface_shark(*SEA, R) is None, "a second shark for one swimmer"


def test_waits_then_charges_and_kills_a_still_agent():
    w = fresh()
    shark = w.surface_shark(*SEA, R)
    start = (shark.x, shark.y)
    wait = shark.spec.hostility_args["wait"]
    for _ in range(wait - 1):
        assert tick(shark, SEA, w) == 0
        assert (shark.x, shark.y) == start and shark.job == "wait"
    tick(shark, SEA, w)
    assert shark.job == "charge" and shark.charge_to == SEA
    hit, steps = 0, 0
    while not hit and steps < 3 * R:
        before = (shark.x, shark.y)
        hit = tick(shark, SEA, w)
        steps += 1
        assert abs(shark.x - before[0]) + abs(shark.y - before[1]) <= 1, "faster than 1 tile/tick"
    assert hit >= 9999, "a still agent wasn't caught"


def test_moving_after_the_lock_dodges():
    w = fresh()
    shark = w.surface_shark(*SEA, R)
    agent = SEA
    while shark.job != "charge":
        tick(shark, agent, w)
    dx, dy = shark.charge_to[0] - shark.x, shark.charge_to[1] - shark.y
    side = (agent[0] + (0 if abs(dx) >= abs(dy) else 1), agent[1] + (1 if abs(dx) >= abs(dy) else 0))
    agent = side                                     # one step off the line
    for _ in range(3 * R):
        assert tick(shark, agent, w) == 0, "dodged, but still bitten"
        if shark.job == "wait":
            break
    assert shark.job == "wait", "a missed shark should wait again"
    while shark.job != "charge":
        tick(shark, agent, w)
    assert shark.charge_to == agent, "it should re-lock on the new spot"


def test_moving_before_the_lock_only_changes_the_aim():
    w = fresh()
    shark = w.surface_shark(*SEA, R)
    tick(shark, SEA, w)
    moved = (SEA[0] + 1, SEA[1])
    while shark.job != "charge":
        tick(shark, moved, w)
    assert shark.charge_to == moved


def test_gives_up_when_agent_reaches_land():
    w = fresh()
    land = w.agent_spawn
    shark = w.surface_shark(*SEA, R)
    tick(shark, land, w)
    assert not shark.alive


def test_env_one_visible_shark_after_grace():
    """Through the real step(): wait at sea; one shark, seen, then it kills."""
    random.seed(2)
    env = ATSEnvironment(world_seed=7, vary_world=False)
    env.reset()
    env.world.entities = []
    env.agent.x, env.agent.y = SEA
    WAIT = 24                                        # the 'wait' action
    seen_at, most, cause = None, 0, None
    for t in range(60):
        _, _, done, _ = env.step(WAIT)
        sharks = [e for e in env.world.entities if e.alive and e.spec.has("ocean")]
        most = max(most, len(sharks))
        if sharks and seen_at is None:
            seen_at = t
            ent, _ = env.world.nearest_entity(*SEA)
            assert ent.spec.brain_id == 29
        if done:
            cause = env.death_cause
            break
    assert seen_at is not None, "no shark came"
    assert most == 1, f"{most} sharks at once"
    assert cause == "a shark", f"died of {cause!r}"
    assert t - seen_at >= 4, "killed before it could be seen and dodged"


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
