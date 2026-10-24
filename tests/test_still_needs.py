"""Scheme 6: standing still costs only with a reason."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_rl
from rewards import RewardEngine, still_need
from entities import Entity, REGISTRY

class A:
    def __init__(self, hunger): self.x, self.y, self.hunger = 0, 0, hunger
class W:
    def __init__(self, ents=(), obj=("", 999, None)): self.entities, self._obj = list(ents), obj
    def nearest_object(self, x, y): return self._obj

assert still_need(A(100), W()) == 0, "fed and safe: resting is free"
assert 0 < still_need(A(30), W()) < 1, "hungry: costs, more the hungrier"
assert still_need(A(0), W()) == 1
hostile = next(i for i, s in REGISTRY.entities.items() if s.has("hostile") and s.hostility == "defends_space")
e = Entity.from_id(hostile, 5, 0); e.angry = True
assert still_need(A(100), W([e])) == 1, "danger in earshot: costs even when fed"
assert still_need(A(30), W(obj=("0001", 2, (2, 0)))) > still_need(A(30), W()), "food in sight while hungry costs more"
assert still_need(A(30), W(obj=("0012", 2, (2, 0)))) == still_need(A(30), W()), "a stone is not food"

def cost(need, ticks=20):
    r = RewardEngine()
    for _ in range(ticks): r.update_standing_still(3, 3, need)
    return r._nuisance_spent
c0, c1 = cost(0.0), cost(1.0)
print("20 still ticks cost: fed+safe", c0, "| hungry/danger", c1)
assert c0 == 0 and c1 != 0
print("ok")
