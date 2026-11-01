"""Coal moves Rewrite on the CoalOS desktop by copying how you move your player.

Every step you take with the arrows comes here with what your player could see
(where the nearest apple is, where Rewrite is) and which way you went. Coal's
small desk area learns "in this spot, go this way" from those. When you let
Coal drive, it looks from Rewrite's spot and picks a way itself.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MOVES = ["stay", "left", "right", "up", "down"]
DESK_IN = 7
KEEP = 4000                      # newest examples kept in the brain
LEARN_EVERY = 16                 # a few moments of learning after this many new steps


def obs(me, other, apples) -> list[float]:
    """Where things are from `me`, in tiles, squashed to about -1..1."""
    sq = lambda v: max(-1.0, min(1.0, v / 8.0))
    if apples:
        ax, ay = min(apples, key=lambda a: abs(a[0] - me[0]) + abs(a[1] - me[1]))
        a = [sq(ax - me[0]), sq(ay - me[1]), 1.0]
    else:
        a = [0.0, 0.0, 0.0]
    return a + [sq(other[0] - me[0]), sq(other[1] - me[1]), 1.0 if apples and (ax, ay) == tuple(me) else 0.0, 1.0]


def seen(brain) -> list:
    return brain.keep.setdefault("desk_seen", [])


def see(brain, state: dict, move: str) -> bool:
    """One of your steps. Returns True when it is time for a little learning."""
    if move not in MOVES:
        return False
    s = seen(brain)
    s.append((obs(state["me"], state["other"], state.get("apples", [])), MOVES.index(move)))
    del s[:-KEEP]
    return len(s) % LEARN_EVERY == 0


def learn(brain, steps: int = 60) -> float:
    s = seen(brain)
    if not s:
        return 0.0
    x = torch.tensor([o for o, _ in s], dtype=torch.float32)
    y = torch.tensor([m for _, m in s])
    opt = torch.optim.Adam(brain.mover.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.randint(0, len(s), (min(128, len(s)),))
        loss = F.cross_entropy(brain.mover(x[i]), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


@torch.no_grad()
def choose(brain, state: dict) -> dict:
    p = F.softmax(brain.mover(torch.tensor([obs(state["me"], state["other"], state.get("apples", []))])), -1)[0]
    i = int(torch.multinomial(p, 1))                 # mostly its best guess, sometimes another way
    return {"move": MOVES[i], "sure": round(float(p[i]), 2), "learned_from": len(seen(brain))}
