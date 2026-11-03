"""Coal's record: everything it was taught, everything said to it, everything it said and did (Chalce, 11-02).

Kept inside the brain file (brain.keep["memory"]), so copying the one file copies all of Coal.
Nothing fades by age: a memory is found by what it means, not by how old it is. Words become
word-layer vectors, rare words count most, and the closest events come back for the writer to read.
The last few things said are always in view too (what was just asked, what it just answered).

An event: {"when": "2026-11-02 21:14", "kind": "lesson" | "heard" | "said" | "did", "text": ..., "answer": ...}
Text only, ~200 bytes each: a thousand events a day is about 70 MB a year. Pictures and video
stay on a drive; an event can point at them.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as Fn

from household.typist import tokens
from household.writer import content


class Record:
    def __init__(self, wr, events: list[dict] | None = None):
        self.wr = wr                                    # the writer: its word layer gives the vectors
        self.events = list(events or [])
        self._df: dict[str, int] = {}
        self._mat: torch.Tensor | None = None
        self._built = 0

    def add(self, kind: str, text: str, answer: str = "") -> dict:
        e = {"when": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "text": text, "answer": answer}
        self.events.append(e)
        return e

    def recent(self, n: int = 3) -> list[str]:
        return [self.show(e) for e in self.events if e["kind"] in ("heard", "said", "did")][-n:]

    @staticmethod
    def show(e: dict) -> str:
        who = {"heard": "you said", "said": "i said", "did": "i did"}.get(e["kind"])
        return f"{who} : {e['text']}" if who else f"{e['text']} : {e['answer']}"

    # ---- finding by meaning ---------------------------------------------------------------
    def _vec(self, words: list[str]) -> torch.Tensor:
        n = max(1, len(self.events))
        if not words:
            return torch.zeros(self.wr.vec("a").shape[0])
        w = torch.tensor([math.log((n + 1) / (self._df.get(x, 0) + 1)) + 0.1 for x in words])
        v = torch.stack([self.wr.vec(x) for x in words])
        return Fn.normalize((w.unsqueeze(1) * v).sum(0), dim=0)

    def _index(self) -> None:
        """Only events added since last time are indexed: a chat line costs one row, not the whole record."""
        if self._built == len(self.events) and self._mat is not None:
            return
        new = range(self._built if self._mat is not None else 0, len(self.events))
        if self._mat is None:
            self._df, self._has = {}, {}
        for i in new:
            for x in set(content(self.events[i]["text"])):
                self._df[x] = self._df.get(x, 0) + 1
                self._has.setdefault(x, []).append(i)
        rows = torch.stack([self._vec(content(self.events[i]["text"])) for i in new]) if len(new) else None
        if rows is not None:
            self._mat = rows if self._mat is None else torch.cat([self._mat, rows])
        self._built = len(self.events)

    def recall(self, question: str, k: int = 3, skip: str | None = None) -> list[str]:
        """The k events closest in meaning to the question (skip: leave out this exact text)."""
        self._index()
        if self._mat is None:
            return []
        words = content(question)
        q = self._vec(words)
        sims = self._mat @ q
        # meaning alone misses exact words (the vectors still lean on spelling): shared words count too, rare ones most
        n = len(self.events)
        idf = {w: math.log((n + 1) / (self._df.get(w, 0) + 1)) for w in set(words)}
        total = sum(idf.values()) or 1.0
        cands = set(sims.topk(min(n, k + 30)).indices.tolist())
        for w in sorted(idf, key=lambda w: -idf[w])[:3]:
            cands.update(self._has.get(w, [])[:200])
        def score(i: int) -> float:
            shared = set(content(self.events[i]["text"])) & set(words)
            return float(sims[i]) + sum(idf[w] for w in shared) / total
        out = []
        for i in sorted(cands, key=score, reverse=True):
            e = self.events[i]
            if e["text"] != skip and score(i) > 0.3:
                out.append(self.show(e))
            if len(out) == k:
                break
        return out
