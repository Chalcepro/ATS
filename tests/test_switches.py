"""A lever is a lever: its own tile, no pick-up, and the `switches` rungs.

    python tests/test_switches.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import random
import tempfile
from pathlib import Path

import torch

import brain
import config_rl
import curriculum as C
import loop
from ats_env import ATSEnvironment
from model_rl import RLPolicy
from world import TILE_OBJECT, TILE_SWITCH

INTERACT, PICK_UP, WAIT = config_rl.ACT_INTERACT, config_rl.ACT_PICK_UP, config_rl.ACT_WAIT


def _lever_at_hand(env):
    env.reset()
    env.max_ticks = 10 ** 6
    env.agent.health = 0
    env.agent.last_damage_cause = "drowning"
    env.step(WAIT)
    rng = random.Random(14)
    room = loop.lever_room(rng)
    while room.about != "lever at hand":
        room = loop.lever_room(rng)
    env._enter_room(room)
    return next(p for p, i in room.objects.items() if i == loop.LEVER)


def test_the_world_draws_a_lever_as_a_lever_and_offers_no_pick_up():
    env = ATSEnvironment(world_seed=11, vary_world=False, use_loop=True)
    lever = _lever_at_hand(env)
    assert env.world.tile_type_at(*lever) == TILE_SWITCH
    mask = env.agent.get_action_mask(env.world, env.day_night)
    assert mask[INTERACT] == 1 and mask[PICK_UP] == 0
    s = env._state()
    patch = s[config_rl.IDX_PATCH_START:config_rl.IDX_PATCH_START + config_rl.PATCH_TILES]
    assert float(TILE_SWITCH) in patch
    known = s[config_rl.IDX_MEM_KNOWN_START:config_rl.IDX_MEM_VISIT_START]
    assert max(known) <= 1.0


def test_the_flag_off_is_the_old_rendering():
    config_rl.LEVERS_ARE_LEVERS = False
    try:
        env = ATSEnvironment(world_seed=11, vary_world=False, use_loop=True)
        lever = _lever_at_hand(env)
        assert env.world.tile_type_at(*lever) == TILE_OBJECT
        assert env.agent.get_action_mask(env.world, env.day_night)[PICK_UP] == 1
    finally:
        config_rl.LEVERS_ARE_LEVERS = True


def _rung(name):
    return next(s for s in C.default_ladder() if s.name == name)


def test_switches_sit_after_avoid_and_before_hazards():
    names = [s.name for s in C.default_ladder()]
    assert names.index("avoid") < names.index("switches") < names.index("switches7") < names.index("hazards")


def test_both_wirings_happen_and_the_goal_is_shut_until_the_pull():
    kinds = set()
    for seed in range(40):
        env = C.CurriculumEnv(_rung("switches"), seed=seed)
        assert env.lever and not env.pulled
        if env.cage:
            kinds.add("cage")
            assert env._view(*env.cage) == C.T_GATE and env.goals == [env.cage]
        else:
            kinds.add("lamp")
            assert env.goals == [] and len(env.dark) == 1
        assert env._view(*env.lever) == C.T_SWITCH
        assert env.action_mask()[INTERACT] == 1
    assert kinds == {"cage", "lamp"}


def test_the_lever_is_what_the_object_slots_describe_until_pulled():
    env = C.CurriculumEnv(_rung("switches"), seed=3)
    s = env._state()
    assert abs(s[config_rl.IDX_OBJ_ID] - C.LEVER_ID / config_rl.ITEM_ID_SCALE) < 1e-9


def _walk_beside_lever(env):
    stage = env.stage
    while max(abs(env.lever[0] - env.ax), abs(env.lever[1] - env.ay)) > 1:
        env.step(C.oracle_step(env, stage))


def test_pressing_away_from_the_lever_costs_and_does_nothing():
    for seed in range(60):
        env = C.CurriculumEnv(_rung("switches7"), seed=seed)
        if max(abs(env.lever[0] - env.ax), abs(env.lever[1] - env.ay)) > 1:
            _, r, done, info = env.step(INTERACT)
            assert not info["pulled"] and not done
            assert abs(r - (env.stage.step_cost + C.R_GRAB)) < 1e-9
            return
    raise AssertionError("no room started away from its lever")


def test_the_pull_opens_the_cage_or_lights_the_goal():
    for seed in range(20):
        env = C.CurriculumEnv(_rung("switches"), seed=seed)
        _walk_beside_lever(env)
        _, r, done, info = env.step(INTERACT)
        assert info["pulled"] and not done and r >= C.R_PULL - 0.01
        assert env.goals and not env.dark
        if env.cage:
            assert env._view(*env.cage) == C.T_OBJECT
        _, r, _, info = env.step(INTERACT)        # a second pull does nothing
        assert r < 0


def test_the_cage_cannot_be_walked_into():
    seed = 0
    env = C.CurriculumEnv(_rung("switches"), seed=seed)
    while not env.cage:
        seed += 1
        env = C.CurriculumEnv(_rung("switches"), seed=seed)
    for a, (dx, dy) in C.MOVES.items():
        beside = (env.cage[0] - dx, env.cage[1] - dy)
        if env._tile(*beside) == C.T_FLOOR:
            env.ax, env.ay = beside
            env.step(a)
            assert (env.ax, env.ay) == beside
            return
    raise AssertionError("cage with no floor beside it")


def test_a_competent_player_clears_both_rungs_and_scores_positive():
    for name in ("switches", "switches7"):
        stage, wins = _rung(name), 0
        for seed in range(100):
            env = C.CurriculumEnv(stage, seed=seed)
            done, info = False, {}
            while not done:
                _, _, done, info = env.step(C.oracle_step(env, stage))
            wins += info["success"]
        assert wins == 100, (name, wins)
        assert C.best_case_return(stage) > 1.0, name


def test_an_old_brain_grows_the_lever_row_and_answers_the_same():
    torch.manual_seed(0)
    old = RLPolicy(hidden_size=32, tile_vocab=16)
    path = Path(tempfile.mkdtemp()) / "brain.pt"
    brain.save(old, progress={"passed": ["nursery 5x5"]}, path=path)
    new = RLPolicy(hidden_size=32)
    assert new.tile_embed.num_embeddings == 17
    assert brain.is_loadable(new, path)
    assert brain.load(new, path=path)["passed"] == ["nursery 5x5"]
    s = torch.rand(3, config_rl.STATE_SIZE)
    ps = config_rl.IDX_PATCH_START
    s[:, ps:ps + config_rl.PATCH_TILES] = torch.randint(0, 16, (3, config_rl.PATCH_TILES)).float()
    with torch.no_grad():
        a, va, _ = old(s)
        b, vb, _ = new(s)
    assert torch.equal(a, b) and torch.equal(va, vb)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
