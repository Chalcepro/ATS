"""Coal types (Chalce, 10-31): its own sentence, word by word, instead of picking one taught answer.

The reply head can only hand back an answer it was given, from 256 slots. Typing has its own
area in the brain (like the ears): it reads the class question and what it has typed so far,
and picks the next word from its word list, or the end. Each word it types goes back in, so a
sentence is built one word at a time, the way a child starts to talk: first it repeats what it
was taught, then it starts to mix the pieces.

Only the typing area and its head learn, so typing can not move the household, the class
replies or the ears. Words get slots as they are met (brain.keep["type_words"]); slot 0 is the end.

Run:  python -m household.typist teach      (teaches from every class question so far)
      python -m household.typist "Someone says: I'm hungry. What do you say back?"
"""
from __future__ import annotations

import re
import sys
import zlib

import torch
import torch.nn.functional as F

TYPE_HASH = 256            # the typed-so-far words, hashed (like the class question)
TYPE_POS = 16              # which word of the sentence this is
TYPE_WORDS = 1024          # slots for words it can type; slot 0 is the end
LONGEST = 24


def typed_names() -> list[str]:
    return ([f"typed.last{i}" for i in range(TYPE_HASH)] + [f"typed.so_far{i}" for i in range(TYPE_HASH)]
            + [f"typed.pos{i}" for i in range(TYPE_POS)])


TYPED_SIZE = len(typed_names())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[?!.,]", text.lower())


def typed_obs(so_far: list[str]) -> list[float]:
    last, bag, pos = [0.0] * TYPE_HASH, [0.0] * TYPE_HASH, [0.0] * TYPE_POS
    for w in so_far:
        bag[zlib.crc32(w.encode()) % TYPE_HASH] = 1.0
    if so_far:
        last[zlib.crc32(so_far[-1].encode()) % TYPE_HASH] = 1.0
    pos[min(len(so_far), TYPE_POS - 1)] = 1.0
    return last + bag + pos


def words(brain) -> list[str]:
    return brain.keep.setdefault("type_words", ["<end>"])


def slot(brain, w: str) -> int:
    ws = words(brain)
    if w not in ws:
        if len(ws) >= TYPE_WORDS:
            raise RuntimeError("Coal's typing slots are full; the type head needs widening")
        ws.append(w)
    return ws.index(w)


def steps(brain, pairs) -> tuple[torch.Tensor, torch.Tensor]:
    """Every word of every answer as one lesson: (question + typed so far) -> next word."""
    from household.live import class_obs
    obs, want = [], []
    for q, answer in pairs:
        cq, toks = class_obs(q), tokens(answer)[:LONGEST - 1]
        for k, w in enumerate(toks + ["<end>"]):
            obs.append(cq + typed_obs(toks[:k]))
            want.append(slot(brain, w))
    return torch.tensor(obs, dtype=torch.float32), torch.tensor(want)


def moving(brain):
    return [*brain.typer.parameters(), *brain.typed.parameters()]


def teach(brain, pairs, epochs: int = 300) -> float:
    obs, want = steps(brain, pairs)
    was = [p.requires_grad for p in brain.parameters()]
    for p in brain.parameters():
        p.requires_grad_(False)
    for p in moving(brain):
        p.requires_grad_(True)
    opt = torch.optim.Adam(moving(brain), lr=3e-3)
    brain.train()
    for _ in range(epochs):
        for b in torch.randperm(len(obs)).split(256):
            loss = F.cross_entropy(brain(obs[b])["type"][:, :len(words(brain))], want[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    brain.eval()
    for p, r in zip(brain.parameters(), was):
        p.requires_grad_(r)
    return float(loss.detach())


@torch.no_grad()
def write(brain, question: str) -> str:
    """Coal's own sentence: the likeliest next word each time, until it types the end."""
    from household.live import class_obs
    cq, out, ws = class_obs(question), [], words(brain)
    for _ in range(LONGEST):
        p = brain(torch.tensor([cq + typed_obs(out)], dtype=torch.float32))["type"][0, :len(ws)]
        for w in out[-2:]:                           # no stutter: not the same word twice running
            p[ws.index(w)] = -1e9
        w = ws[int(p.argmax())]
        if w == "<end>":
            break
        out.append(w)
    return re.sub(r" ([?!.,])", r"\1", " ".join(out))


def main() -> None:
    import os
    from household.class_bank import BANK
    from household.live import Answers
    from household.model import load, save
    from household.train import SAVE
    brain, rounds = load(SAVE)
    if sys.argv[1:2] != ["teach"]:
        print(write(brain, " ".join(sys.argv[1:])))
        return
    import shutil
    shutil.copy(SAVE, SAVE.replace(".pt", "_before_typing.pt"))
    answers = Answers(os.devnull, brain.keep.get("answers", {}))
    pairs = [(q, answers.texts[i]) for q, i in answers.pairs]
    held = [(asks[-1], ans) for _, asks, ans in BANK]
    probe = torch.rand(8, 202)
    with torch.no_grad():
        house, reply = brain(probe)["act"], brain(probe)["reply"]
    loss = teach(brain, pairs)
    with torch.no_grad():
        moved = float(max((brain(probe)["act"] - house).abs().max(), (brain(probe)["reply"] - reply).abs().max()))
    exact = sum(write(brain, q) == re.sub(r" ([?!.,])", r"\1", " ".join(tokens(a))) for q, a in held) / len(held)
    taught = sum(write(brain, q) == re.sub(r" ([?!.,])", r"\1", " ".join(tokens(a))) for q, a in pairs[-200:]) / min(200, len(pairs))
    print(f"{len(pairs)} questions, {len(words(brain))} words it can type, loss {loss:.3f}, household/replies moved {moved:.1e}")
    print(f"types the taught answer exactly: {taught:.0%};  on held-back ways of asking: {exact:.0%}")
    for q, a in held[:: max(1, len(held) // 10)]:
        print(f"  {q!r}\n     Coal types: {write(brain, q)!r}   (teacher: {a!r})")
    for q in ["Someone says: I'm so hungry and tired. What do you say back?", "Someone says: my laptop died. What do you say back?",
              "Someone says: I finished my homework! What do you say back?"]:
        print(f"  NEW {q!r}\n     Coal types: {write(brain, q)!r}")
    save(brain, SAVE, rounds)
    print("saved", SAVE)


if __name__ == "__main__":
    main()
