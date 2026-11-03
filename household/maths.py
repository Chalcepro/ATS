"""Coal's maths section (Chalce, 11-02): numbers as values, worked the way children are taught to.

Numbers are not words here. A number is its digits, units first, and one small learned cell works
along the columns like column sums on paper: it reads one column (a digit from each number, and what
it carried over from the column before) and writes the answer digit, passing on what it carries.
The same cell is used for every column, so if it truly learned the method it works on numbers longer
than any it practised on - that is the test: practise on up to 3 digits, test on 5 and 6.

Foundations, in teaching order (each a stage of lessons):
  1 count      - what comes after / before a number (the number line)
  2 compare    - which is bigger
  3 place      - tens and units: how many tens in 47
  4 add        - with carrying
  5 subtract   - with borrowing (bigger minus smaller first)
  6 times      - a number times one digit (tables first), then longer: the sum of shifted rows
Later (not built): sharing (division), parts of a whole (fractions, decimals, percent), shape and
measure, unknowns (the start of algebra), and powers of ten for the very big and the very small.

Words meet numbers through number words ("seven" is 7) - the word layer already puts them in its
number section. Grounding in things (three apples) comes from the CoalOS playground.

    py -3.12 -m household.maths show        (print example lessons of every stage; nothing is trained)
"""
from __future__ import annotations

import random
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as Fn

OPS = ["add", "subtract", "times", "compare", "next", "before"]
HID = 64
WIDTH_TRAIN = 3            # digits it practises on
WIDTH_TEST = (5, 6)        # digits it is tested on: longer than it ever saw

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
OP_WORDS = {"plus": "add", "add": "add", "and": "add", "+": "add", "minus": "subtract", "take": "subtract",
            "subtract": "subtract", "-": "subtract", "times": "times", "multiplied": "times", "*": "times", "x": "times",
            "bigger": "compare", "larger": "compare", "after": "next", "before": "before"}


def digits(n: int, width: int) -> list[int]:
    return [(n // 10 ** i) % 10 for i in range(width)]


def number(ds: list[int]) -> int:
    return sum(d * 10 ** i for i, d in enumerate(ds))


class ColumnCell(nn.Module):
    """One column of a written sum: (digit a, digit b, the operation, what came from the last column) ->
    (the digit to write, what to pass on). Also says, after the last column, which number was bigger."""

    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(10 + 10 + len(OPS), HID)
        self.write = nn.Linear(HID, 10)
        self.verdict = nn.Linear(HID, 3)          # a bigger / same / b bigger

    def forward(self, a: torch.Tensor, b: torch.Tensor, op: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """a, b: (batch, columns) digits, units first; op: (batch,) operation index.
        Returns the written digits' scores (batch, columns, 10) and the verdict scores (batch, 3)."""
        B, C = a.shape
        h = a.new_zeros(B, HID, dtype=torch.float32)
        o = Fn.one_hot(op, len(OPS)).float()
        out = []
        for c in range(C):
            x = torch.cat([Fn.one_hot(a[:, c], 10).float(), Fn.one_hot(b[:, c], 10).float(), o], 1)
            h = self.cell(x, h)
            out.append(self.write(h))
        return torch.stack(out, 1), self.verdict(h)


# ---- working a sum, the way it is done on paper ------------------------------------------------
@torch.no_grad()
def column(cell: ColumnCell, a: int, b: int, op: str, width: int) -> list[int]:
    out, _ = cell(torch.tensor([digits(a, width)]), torch.tensor([digits(b, width)]), torch.tensor([OPS.index(op)]))
    return out[0].argmax(1).tolist()


@torch.no_grad()
def work(cell: ColumnCell, a: int, b: int, op: str) -> int | str:
    """The answer, worked by the cell. Long multiplication is rows of (number x one digit), shifted and added -
    every step done by the cell itself, never by Python's own arithmetic."""
    width = max(len(str(a)), len(str(b))) + 2
    if op in ("add", "next", "before"):
        b = 1 if op != "add" else b
        op = "add" if op == "next" else ("subtract" if op == "before" else op)
    if op == "compare":
        _, v = cell(torch.tensor([digits(a, width)]), torch.tensor([digits(b, width)]), torch.tensor([OPS.index("compare")]))
        return ["the first", "they are the same", "the second"][int(v.argmax())]
    if op == "subtract" and a < b:                    # bigger minus smaller, then it is below zero
        return -work(cell, b, a, "subtract")
    if op in ("add", "subtract"):
        return number(column(cell, a, b, op, width))
    total, width = 0, len(str(a)) + len(str(b)) + 1   # times
    for i, d in enumerate(digits(b, len(str(b)))):
        row = number(column(cell, a, d, "times", width)) * 10 ** i     # the shift is writing a 0 on the right
        total = number(column(cell, total, row, "add", width))
    return total


# ---- lessons -----------------------------------------------------------------------------------
def answer(a: int, b: int, op: str) -> int:
    return {"add": a + b, "subtract": a - b, "times": a * b, "next": a + 1, "before": a - 1}[op]


def batch(r: random.Random, n: int, width: int, ops: list[str]):
    """n column lessons up to `width` digits: (a digits, b digits, op, want digits, want verdict)."""
    A, Bs, O, W, V = [], [], [], [], []
    cols = width + 2
    for _ in range(n):
        op = r.choice(ops)
        k = r.randint(1, width)
        a = r.randint(0, 10 ** k - 1)
        b = r.randint(0, 9) if op == "times" else r.randint(0, 10 ** r.randint(1, width) - 1)
        if op == "subtract" and b > a:
            a, b = b, a
        want = 0 if op == "compare" else a + b if op == "add" else a - b if op == "subtract" else a * b
        A.append(digits(a, cols)); Bs.append(digits(b, cols)); O.append(OPS.index(op))
        W.append(digits(want, cols)); V.append(0 if a > b else 1 if a == b else 2)
    t = lambda x: torch.tensor(x)
    return t(A), t(Bs), t(O), t(W), t(V)


def words_lessons() -> list[tuple[str, str]]:
    """Number words and sums in words, for the writer: "what is three plus four?" -> "seven, 7"."""
    names = list(WORDS) + list(TENS)
    out = [(f"what number is {w}?", f"{w} is {WORDS.get(w, TENS.get(w))}") for w in names]
    out += [(f"how do you write {n} in words?", f"{w}") for w, n in list(WORDS.items()) + list(TENS.items())]
    return out


def read_number(text: str) -> int | None:
    """Digits, or number words up to ninety-nine ("forty two", "forty-two")."""
    t = text.lower().replace("-", " ").split()
    if len(t) == 1 and t[0].isdigit():
        return int(t[0])
    if len(t) == 1 and t[0] in WORDS:
        return WORDS[t[0]]
    if t and t[0] in TENS and (len(t) == 1 or (len(t) == 2 and t[1] in WORDS and WORDS[t[1]] < 10)):
        return TENS[t[0]] + (WORDS[t[1]] if len(t) == 2 else 0)
    return None


def read_sum(question: str) -> tuple[int, str, int] | None:
    """'what is 3 times 4?' / 'what is seven plus five' / '12 * 7' -> (3, 'times', 4). None if it is not a sum.
    Which questions are sums will later be the writer's call (it learns to hand them over); this is the reading."""
    q = re.sub(r"([+*x-])", r" \1 ", question.lower().replace("?", " "))
    toks = q.split()
    for i, w in enumerate(toks):
        if w in OP_WORDS and OP_WORDS[w] in ("add", "subtract", "times"):
            for la in (2, 1):
                for lb in (2, 1):
                    a, b = read_number(" ".join(toks[max(0, i - la):i])), read_number(" ".join(toks[i + 1:i + 1 + lb + (w == "multiplied")][(w == "multiplied"):]))
                    if a is not None and b is not None:
                        return a, OP_WORDS[w], b
    return None


def show() -> None:
    r = random.Random(0)
    for op in ["add", "subtract", "times", "compare"]:
        A, B, O, W, V = batch(r, 3, WIDTH_TRAIN, [op])
        for i in range(3):
            a, b = number(A[i].tolist()), number(B[i].tolist())
            print(f"  {op:<9} {a} , {b} -> " + (["first", "same", "second"][int(V[i])] + " bigger" if op == "compare" else str(number(W[i].tolist()))))
    for q in ["what is 3 times 4?", "what is seven plus five?", "12 * 7", "what is forty two minus nine", "what is a dog?"]:
        print(f"  read {q!r} -> {read_sum(q)}")
    for q, a in words_lessons()[:3]:
        print(f"  {q} -> {a}")
    cell = ColumnCell()                               # untrained: its answers are noise until the maths class runs
    print(f"  untrained cell: 27 + 15 -> {work(cell, 27, 15, 'add')}  ({sum(p.numel() for p in cell.parameters())} numbers)")


if __name__ == "__main__":
    show() if sys.argv[1:2] == ["show"] else print(__doc__)
