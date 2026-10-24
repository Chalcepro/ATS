"""Desk food grows where food grows: oak trees drop apples beside them (up to 2),
a bush the brain eats grows back, a bush Chalce removes doesn't."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "desk"))
import desk_world as D

w = D.DeskWorld(20, 10)
w.put(5, 5, D.TREE)
for _ in range(D.TREE_DROP_EVERY * 4):
    w.grow()
apples = [o for o in w.objects() if o[2] == D.APPLE]
assert len(apples) == D.TREE_MAX_APPLES, apples
assert all(max(abs(x - 5), abs(y - 5)) == 1 for x, y, _ in apples), "beside the tree"

w.put(10, 5, D.BUSH)
assert w.pickup_at(10, 5) and not w.tiles[(10, 5)].object_id, "eaten"
for _ in range(D.BUSH_REGROW - 1):
    w.grow()
assert not w.tiles[(10, 5)].object_id, "not back yet"
w.grow()
assert w.tiles[(10, 5)].object_id == D.BUSH, "grew back"

w.put(12, 5, D.BUSH)
w.remove(12, 5)                              # Chalce deleted it
for _ in range(D.BUSH_REGROW + 5):
    w.grow()
assert not w.tiles[(12, 5)].object_id, "removed bushes stay gone"
print("ok")
