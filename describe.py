"""Describe it (roadmap H6, first step): can the brain SAY the narrator's words?

Hearing the words is not understanding them - flowing in, they can be used as
hints. Understanding shows the other way round: with the words hidden from it,
does its own memory state (the GRU hidden) know which words fit the moment?

This is a small head, Linear(hidden -> WORD_N), trained on the frozen brain's
memory states with the word inputs zeroed, and scored on episodes it never saw.
It lives in its own file (checkpoints/describe.pt): brain.pt is never touched.

    py -3.12 tools/train_describe.py           collect, train, score (a few minutes)"""
import os

import torch
from torch import nn

import narrator

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "checkpoints", "describe.pt")


class DescribeHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.out = nn.Linear(hidden, narrator.WORD_N)

    def forward(self, h):
        return self.out(h)

    def say(self, hidden, threshold=0.5):
        """The words it would use for this memory state."""
        with torch.no_grad():
            p = torch.sigmoid(self(torch.tensor(hidden, dtype=torch.float32)))
        return [w for w, q in zip(narrator.WORDS, p.tolist()) if q >= threshold]

    @classmethod
    def load(cls, hidden):
        head = cls(hidden)
        if os.path.exists(PATH):
            saved = torch.load(PATH, map_location="cpu")
            if saved["out.weight"].shape[1] == hidden:
                head.load_state_dict(saved)
                head.trained = True
                return head.eval()
        head.trained = False
        return head.eval()
