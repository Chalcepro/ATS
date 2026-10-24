"""The H2 gate: does the brain react to a growl it can hear but not see?

    py -3.12 tools/listening_gate.py            (40 rooms each way, a few minutes)
    py -3.12 tools/listening_gate.py --rooms 10

The same rooms (same seeds) are played twice: once with its ears, once with
its ears silenced. On every tick where the hostile is out of view and growling,
it scores whether that tick's move took it further from the hostile. The
silenced run is "chance": what this brain does anyway, without the sound.

Gate: with ears, it moves away from an unseen growl clearly more often than
with ears silenced, and it gets bitten less. Before any training on the
listening rung the two should match - zero weights on the ears."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import argparse

import torch

import brain, config_rl
from curriculum import CurriculumEnv, default_ladder
from model_rl import RLPolicy
from mind.solution_loop import SolutionLoop

MARGIN = 0.10           # "clearly": at least 10 points more often away with ears
MIN_HEARD = 100         # fewer unseen growls than this is too little to judge


def play(policy, stage, seeds, ears=True):
    away = heard = bites = 0
    for seed in seeds:
        # the same dice both ways: an untrained ear (zero weights) then plays
        # both runs identically, so any difference is the ears and not luck
        torch.manual_seed(seed)
        env = CurriculumEnv(stage, seed=seed)
        state = env.reset()
        sl = SolutionLoop(policy=policy)
        done = False
        while not done:
            hx = env.hostiles[0] if env.hostiles else None
            out_of_view_growl = hx and not env.in_view(*hx) and getattr(env, "last_sound", None) == "growl"
            if not ears:
                state = list(state)
                h0, h1 = config_rl.IDX_HEAR_START, config_rl.IDX_HEAR_START + config_rl.HEAR_DIM
                state[h0:h1] = [0.0] * config_rl.HEAR_DIM
                if config_rl.WORDS_ON:      # and the narrator can't say GROWL about a sound it didn't hear
                    import narrator
                    state[config_rl.IDX_WORD_START:config_rl.IDX_MASK_START] = narrator.as_inputs(
                        narrator.words_from_state(state))
            d0 = abs(hx[0] - env.ax) + abs(hx[1] - env.ay) if hx else 0
            action = sl.choose_action(state, env.action_mask())
            hits0 = env.hits_taken
            state, _, done, _ = env.step(action)
            bites += env.hits_taken - hits0
            if out_of_view_growl:
                heard += 1
                away += (abs(hx[0] - env.ax) + abs(hx[1] - env.ay)) > d0
    return away / max(heard, 1), heard, bites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=60)
    args = ap.parse_args()
    stage = next(s for s in default_ladder() if s.name == "listening")
    shape = brain.saved_shape()
    policy = RLPolicy(hidden_size=int(shape.get("hidden_size", config_rl.MIND_HIDDEN_SIZE)))
    brain.load(policy, None)
    policy.eval()
    torch.set_grad_enabled(False)
    seeds = list(range(1000, 1000 + args.rooms))
    with_ears = play(policy, stage, seeds, ears=True)
    silenced = play(policy, stage, seeds, ears=False)
    print(f"\n  unseen growls heard        ears {with_ears[1]:5d}   silenced {silenced[1]:5d}")
    print(f"  moved away from the growl  ears {with_ears[0]:5.1%}   silenced {silenced[0]:5.1%}")
    print(f"  bites taken                ears {with_ears[2]:5d}   silenced {silenced[2]:5d}")
    if min(with_ears[1], silenced[1]) < MIN_HEARD:
        print(f"\n  TOO FEW unseen growls to judge (need {MIN_HEARD}); run with more --rooms")
        return False
    ok = with_ears[0] >= silenced[0] + MARGIN and with_ears[2] <= silenced[2]
    print("\n  PASS: it reacts to what it hears" if ok else
          "\n  NOT YET: no clear difference with its ears on (expected before training on 'listening')")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
