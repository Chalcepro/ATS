"""The one model for the household: a body of shared layers with heads.

  act        - what to do (copied from the judge's examples, never scored)
  target     - for whom
  condition  - each person: healthy / neutral / hurt
  foresight  - each person: will they be hurt soon if nobody helps
  words      - each person: which words fit them (what it would say)
  meaning    - a small vector the word box turns into words (household/words.py)

Untrained it is a blank; household/exam.py says how blank.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from household.words import VOCAB
from household.desk_lessons import DESK_ACTS, DESK_SIZE, desk_names
from household.page_lessons import PAGE_SIZE, PARTS, page_names
from household.live import CLASS_SIZE, MAX_ANSWERS, class_names
from household.ears import EAR_CUES, EAR_SIZE, EAR_WORDS, ENDS, ear_names
from household.typist import TYPED_SIZE, TYPE_WORDS, typed_names
from household.world import ACTIONS, MAX_PEOPLE, MEM_SIZE, input_names

IN_SIZE = MEM_SIZE + DESK_SIZE + PAGE_SIZE + CLASS_SIZE + EAR_SIZE + TYPED_SIZE   # household, desktop, a web page, a class question, the ears, what it typed: one brain
CLASS_AT = MEM_SIZE + DESK_SIZE + PAGE_SIZE
EAR_AT = CLASS_AT + CLASS_SIZE


def all_inputs() -> list[str]:
    return input_names() + desk_names() + page_names() + class_names() + ear_names() + typed_names()

MEANING = 16


class HouseholdBrain(nn.Module):
    def __init__(self, hidden: int = 256):      # 128 until round 13: too small once refusal, rooms and warnings joined
        super().__init__()
        self.body = nn.Sequential(nn.Linear(IN_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.act = nn.Linear(hidden, len(ACTIONS))
        self.target = nn.Linear(hidden, MAX_PEOPLE)
        self.condition = nn.Linear(hidden, MAX_PEOPLE * 3)
        self.foresight = nn.Linear(hidden, MAX_PEOPLE)
        self.words = nn.Linear(hidden, MAX_PEOPLE * len(VOCAB))
        self.meaning = nn.Linear(hidden, MEANING)
        self.desk = nn.Linear(hidden, len(DESK_ACTS))   # what to do with this file
        self.reply = nn.Linear(hidden, MAX_ANSWERS)    # which taught answer fits this question (live.py)
        # the hearing area: its own two layers on the sound, so learning to hear never moves the rest
        self.ear = nn.Sequential(nn.Linear(EAR_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.hear = nn.Linear(hidden, EAR_WORDS)       # the word in this slice of sound (ears.py)
        self.hear_end = nn.Linear(hidden, len(ENDS))   # speaking, a pause, a full stop, a question
        self.hear_cue = nn.Linear(hidden, EAR_CUES)    # which sound cue, if it is not speech
        # the typing area: the question and what it typed so far -> the next word (typist.py)
        self.typer = nn.Sequential(nn.Linear(CLASS_SIZE + TYPED_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.typed = nn.Linear(hidden, TYPE_WORDS)
        self.page = nn.ModuleDict({p: nn.Linear(hidden, len(cs)) for p, cs in PARTS.items()})   # a page piece, part by part

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        z = lambda k: obs.new_zeros(obs.shape[0], k)
        at = lambda k: torch.cat([z(k), obs, z(IN_SIZE - k - obs.shape[1])], 1)   # one part alone: the rest is blank
        n = obs.shape[1]
        if n == DESK_SIZE:                          # a file alone
            obs = at(MEM_SIZE)
        elif n == PAGE_SIZE:                        # a page piece alone
            obs = at(MEM_SIZE + DESK_SIZE)
        elif n == CLASS_SIZE:                       # a class question alone
            obs = at(CLASS_AT)
        elif n == EAR_SIZE:                         # a slice of sound alone
            obs = at(EAR_AT)
        elif n == CLASS_SIZE + TYPED_SIZE:          # a class question and what it typed so far
            obs = torch.cat([z(CLASS_AT), obs[:, :CLASS_SIZE], z(EAR_SIZE), obs[:, CLASS_SIZE:]], 1)
        elif n < IN_SIZE:                           # the household alone (or with a file)
            obs = at(0)
        h = self.body(obs)
        e = self.ear(obs[:, EAR_AT:EAR_AT + EAR_SIZE])
        t = self.typer(torch.cat([obs[:, CLASS_AT:EAR_AT], obs[:, -TYPED_SIZE:]], 1))
        return {"act": self.act(h), "target": self.target(h),
                "condition": self.condition(h).view(-1, MAX_PEOPLE, 3),
                "foresight": self.foresight(h), "words": self.words(h).view(-1, MAX_PEOPLE, len(VOCAB)),
                "meaning": torch.tanh(self.meaning(h)), "desk": self.desk(h), "page": {p: head(h) for p, head in self.page.items()}, "reply": self.reply(h),
                "hear": self.hear(e), "hear_end": self.hear_end(e), "hear_cue": self.hear_cue(e), "type": self.typed(t)}

    @torch.no_grad()
    def choose(self, obs: list[float]) -> tuple[str, int]:
        out = self(torch.tensor([obs], dtype=torch.float32))
        return ACTIONS[int(out["act"].argmax())], int(out["target"].argmax())

    @torch.no_grad()
    def file_act(self, desk_obs: list[float]) -> int:
        return int(self(torch.tensor([desk_obs], dtype=torch.float32))["desk"].argmax())

    @torch.no_grad()
    def build_choice(self, page_obs: list[float]) -> dict[str, str]:
        out = self(torch.tensor([page_obs], dtype=torch.float32))["page"]
        return {p: PARTS[p][int(out[p].argmax())] for p in PARTS}

    @torch.no_grad()
    def say(self, obs: list[float], people: int) -> list[list[str]]:
        """What it would say about each person: the words it is sure of."""
        out = self(torch.tensor([obs], dtype=torch.float32))["words"][0]
        return [[VOCAB[j] for j in range(len(VOCAB)) if out[i, j] > 0] for i in range(people)]


# ---- one brain, kept for life ------------------------------------------------------------
def save(brain: HouseholdBrain, path: str, rounds: int) -> None:
    """The weights plus the names of what each input and output means, so a
    later, bigger world can take this brain over instead of starting again.
    brain.keep rides along (taught answers, word library): one file is all of Coal."""
    torch.save({"state": brain.state_dict(), "inputs": all_inputs(), "actions": list(ACTIONS),
                "vocab": list(VOCAB), "hidden": brain.act.in_features, "rounds": rounds,
                "keep": getattr(brain, "keep", {})}, path)


def load(path: str, old_inputs: list[str] | None = None) -> tuple[HouseholdBrain, int]:
    """Load a brain into today's world. Inputs and outputs that still exist keep
    their learned connections (matched by name); new ones start at zero, so on
    day one the brain behaves exactly as before and only learns to use them.
    old_inputs is for a brain saved before names were kept: the inputs it had."""
    ck = torch.load(path)
    if "state" not in ck:                           # an old bare state_dict
        ck = {"state": ck, "inputs": old_inputs or input_names(), "actions": list(ACTIONS),
              "vocab": list(VOCAB), "hidden": ck["act.weight"].shape[1], "rounds": 15}
    brain = HouseholdBrain(ck["hidden"])
    new = brain.state_dict()
    old = ck["state"]
    for k, v in old.items():
        if k == "body.0.weight":                    # inputs, by name
            pos = {name: j for j, name in enumerate(ck["inputs"])}
            w = torch.zeros_like(new[k])
            for j, name in enumerate(all_inputs()):
                if name in pos:
                    w[:, j] = v[:, pos[name]]
            new[k] = w
        elif k.startswith("act."):                  # actions, by name
            keep = [i for i, a in enumerate(ACTIONS) if a in ck["actions"]]
            w = new[k].clone() * 0 if k.endswith("weight") else torch.full_like(new[k], -5.0)
            for i in keep:
                w[i] = v[ck["actions"].index(ACTIONS[i])]
            new[k] = w
        elif k.startswith("words."):                # words it can say, by name
            per = len(ck["vocab"])
            w = torch.zeros_like(new[k])
            for p in range(MAX_PEOPLE):
                for i, word in enumerate(VOCAB):
                    if word in ck["vocab"]:
                        w[p * len(VOCAB) + i] = v[p * per + ck["vocab"].index(word)]
            new[k] = w
        elif k in new and new[k].shape == v.shape:
            new[k] = v
    brain.load_state_dict(new)
    brain.keep = ck.get("keep", {})                 # what it needs to speak: answers, word library
    return brain, ck.get("rounds", 0)
