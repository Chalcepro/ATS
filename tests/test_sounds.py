"""H2 sounds: a hostile closing in growls (often before it's in sight), a shark
splashes, harmless animals only chirp sometimes, and silence is all zeros."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config_rl, sounds
from ear import FIELD
from entities import Entity, REGISTRY

class W:                                    # just enough world for hearing()
    def __init__(self, ents): self.entities = ents

def first(pred):
    return next(i for i, s in REGISTRY.entities.items() if pred(s))

hostile = first(lambda s: s.has("hostile") and s.hostility == "defends_space")
shark = first(lambda s: s.has("ocean"))
calm = first(lambda s: s.hostility == "ignores")

assert not sounds.hearing(W([]), 0, 0).any(), "silence is zeros"
e = Entity.from_id(hostile, 6, 0); e.angry = True             # out of sight (patch radius 3), in earshot
v = sounds.hearing(W([e]), 0, 0)
assert v[FIELD["rms_mean"]] > 0 and tuple(v[-2:]) == (1.0, 0.0), "a growl, from the right"
assert 70 < v[FIELD["pitch_median_norm"]] * 400 < 120, "growls are low"
far = Entity.from_id(hostile, config_rl.HEAR_RADIUS + 2, 0); far.angry = True
assert not sounds.hearing(W([far]), 0, 0).any(), "out of earshot"
s = Entity.from_id(shark, 0, -4)
v2 = sounds.hearing(W([s]), 0, 0)
assert v2.any() and tuple(v2[-2:]) == (0.0, -1.0), "a splash, from above"
c = Entity.from_id(calm, 2, 0)
heard = sum(sounds.hearing(W([c]), 0, 0).any() for _ in range(500))
assert 10 < heard < 100, heard                                   # chirps now and then
near = Entity.from_id(hostile, 5, 0); near.angry = True
print("ok")
