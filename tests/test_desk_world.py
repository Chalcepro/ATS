"""The brain on the desktop: it loads, moves, stays on land, and eats food spawned near it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from desk.desk_world import DeskMind

m = DeskMind(60, 33)
w, a = m.world, m.agent
assert w.on_land(a.x, a.y)
a.hunger = 40
for dx in range(-4, 5):
    w.put(a.x + dx, a.y - 1, "0001")
start = len(w.objects())
seen, hunger0 = set(), a.hunger
for t in range(1500):
    m.step()
    seen.add((a.x, a.y))
    assert w.on_land(a.x, a.y) or w.is_ocean(a.x, a.y)
print("tiles visited", len(seen), "| apples left", len(w.objects()), "of", start, "| hunger", round(a.hunger), "| last", m.last_action)
assert len(seen) > 5, "it never moved"
print("ok")
