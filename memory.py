"""Cross-episode memory with confidence decay."""

import math


class EpisodeMemory:
    def __init__(self):
        self.entries = []

    def add(self, kind, x, y, value, confidence=1.0):
        self.entries.append(
            {
                "kind": kind,
                "x": x,
                "y": y,
                "value": value,
                "confidence": confidence,
            }
        )

    def decay(self, rate=0.05):
        for entry in self.entries:
            entry["confidence"] *= (1.0 - rate)
        self.entries = [e for e in self.entries if e["confidence"] >= 0.1]

    def top_relevant(self, x, y, limit=3):
        ranked = sorted(
            self.entries,
            key=lambda e: e["confidence"] - 0.01 * (abs(e["x"] - x) + abs(e["y"] - y)),
            reverse=True,
        )
        return ranked[:limit]

    def as_state_features(self, x, y):
        top = self.top_relevant(x, y, limit=3)
        while len(top) < 3:
            top.append({"confidence": 0.0, "value": 0})
        return [float(top[i]["confidence"]) for i in range(3)]
