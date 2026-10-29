"""The one model for the household: a body of shared layers with heads.

  act        - what to do (copied from the judge's examples, never scored)
  target     - for whom
  condition  - each person: healthy / neutral / hurt
  foresight  - each person: will they be hurt soon if nobody helps
  meaning    - a small vector the word box turns into words (household/words.py)

Untrained it is a blank; household/exam.py says how blank.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from household.world import ACTIONS, MAX_PEOPLE, OBS_SIZE

MEANING = 16


class HouseholdBrain(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.act = nn.Linear(hidden, len(ACTIONS))
        self.target = nn.Linear(hidden, MAX_PEOPLE)
        self.condition = nn.Linear(hidden, MAX_PEOPLE * 3)
        self.foresight = nn.Linear(hidden, MAX_PEOPLE)
        self.meaning = nn.Linear(hidden, MEANING)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.body(obs)
        return {"act": self.act(h), "target": self.target(h),
                "condition": self.condition(h).view(-1, MAX_PEOPLE, 3),
                "foresight": self.foresight(h), "meaning": torch.tanh(self.meaning(h))}

    @torch.no_grad()
    def choose(self, obs: list[float]) -> tuple[str, int]:
        out = self(torch.tensor([obs], dtype=torch.float32))
        return ACTIONS[int(out["act"].argmax())], int(out["target"].argmax())
