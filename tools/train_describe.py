"""Train and score the describe-it head (describe.py).

    py -3.12 tools/train_describe.py                 ~40 episodes, a few minutes
    py -3.12 tools/train_describe.py --episodes 80

The frozen brain plays rung rooms and island stretches as usual (words and all).
Beside it, a second copy of its memory runs on the same states with the word
inputs zeroed - that memory is what the head reads, so it can't just copy the
words back. Episodes are split: the head trains on 3/4 and is scored on the
rest, which it never saw.

Score per word: balanced accuracy (the mean of "said it when it fit" and "kept
quiet when it didn't"), so a word that is almost always off can't score high by
never being said. Chance is 50%. Gate: 75% averaged over the words that came up
often enough to judge.

What a pass means: the brain's memory already carries what the words name, so
they are learnable labels for things it tracks. It is not yet proof that it
understands words - that needs training with them (roadmap H6)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import argparse, random

import torch
from torch import nn

import brain, config_rl, narrator
from describe import DescribeHead, PATH
from model_rl import RLPolicy
from curriculum import CurriculumEnv, default_ladder
from ats_env import ATSEnvironment
from mind.solution_loop import SolutionLoop

GATE = 0.75
MIN_EACH = 15           # a word must be on and off at least this often in the held-out set
W0, W1 = config_rl.IDX_WORD_START, config_rl.IDX_MASK_START


def blind(state):
    s = list(state)
    s[W0:W1] = [0.0] * config_rl.WORD_N
    return s


def episode(policy, env, ticks, rng):
    """(hidden-with-words-hidden, words) for every tick of one episode."""
    sl = SolutionLoop(policy=policy)
    state = env.reset()
    if isinstance(env, ATSEnvironment):          # vary hunger so HUNGRY / FED both come up
        env.agent.hunger = rng.choice((15, 35, 50, 75, 95))
        state = env._state()
    hx = policy.initial_hidden(1)
    out = []
    for _ in range(ticks):
        with torch.no_grad():
            _, _, hx = policy(torch.tensor([blind(state)]), hx=hx)
        words = state[W0:W1]
        out.append((hx.reshape(-1).clone(), torch.tensor(words)))
        mask = env.action_mask() if hasattr(env, "action_mask") else env.agent.get_action_mask(env.world, env.day_night)
        state, _, done, _ = env.step(sl.choose_action(state, mask))
        if done:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    shape = brain.saved_shape()
    policy = RLPolicy(hidden_size=int(shape.get("hidden_size", config_rl.MIND_HIDDEN_SIZE)))
    brain.load(policy, None)
    policy.eval()

    ladder = {s.name: s for s in default_ladder()}
    kinds = ["listening", "listening", "satchel7", "avoid", "island"]
    eps = []
    for i in range(args.episodes):
        k = kinds[i % len(kinds)]
        env = (ATSEnvironment(vary_world=False, world_seed=100 + i, use_loop=False) if k == "island"
               else CurriculumEnv(ladder[k], seed=100 + i))
        eps.append(episode(policy, env, 300, rng))
        print(f"\r  collecting {i + 1}/{args.episodes} episodes", end="")
    print()
    rng.shuffle(eps)
    cut = len(eps) * 3 // 4
    pack = lambda E: (torch.stack([h for e in E for h, _ in e]), torch.stack([w for e in E for _, w in e]))
    Xtr, Ytr = pack(eps[:cut])
    Xte, Yte = pack(eps[cut:])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    head = DescribeHead(policy.hidden_size)
    # words that are rarely on still count: weight the "on" cases up per word
    pos = Ytr.mean(0).clamp(1e-3, 1 - 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(400):
        opt.zero_grad()
        loss = lossf(head(Xtr), Ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = torch.sigmoid(head(Xte)) >= 0.5
    scores = []
    print(f"\n  {len(Xtr)} training ticks, {len(Xte)} held-out ticks\n")
    print("  word      on/off held out   said-when-fit  quiet-when-not  balanced")
    for j, w in enumerate(narrator.WORDS):
        on, off = Yte[:, j] == 1, Yte[:, j] == 0
        if on.sum() < MIN_EACH or off.sum() < MIN_EACH:
            print(f"  {w:8s}  {int(on.sum()):5d}/{int(off.sum()):<5d}    (too rare to judge)")
            continue
        tpr = pred[on, j].float().mean().item()
        tnr = (~pred[off, j]).float().mean().item()
        scores.append((tpr + tnr) / 2)
        print(f"  {w:8s}  {int(on.sum()):5d}/{int(off.sum()):<5d}    {tpr:12.0%}  {tnr:14.0%}  {scores[-1]:8.0%}")
    mean = sum(scores) / max(len(scores), 1)
    # fold the normalisation into the layer, so the saved head reads raw memory states
    with torch.no_grad():
        head.out.weight /= sd
        head.out.bias -= (head.out.weight * mu).sum(1)
    torch.save(head.state_dict(), PATH)
    print(f"\n  average over {len(scores)} words: {mean:.0%}  (chance 50%, gate {GATE:.0%})")
    print("  PASS: its memory carries what the words name - they are learnable labels for it"
          if mean >= GATE else "  NOT YET")
    print(f"  saved {PATH}")
    return mean >= GATE


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
