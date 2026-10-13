"""A shark at sea must not reach onto the beach.

Since W2b the shark charges (entities.py _tick_charge) instead of biting
whatever is one tile away, so SHARKS_BITE_ONLY_AT_SEA no longer applies to
it: an agent on land makes it give up, and at sea it only kills on contact.
"""
from entities import Entity


class _Tile:
    def __init__(self, t): self.tile_type = t


class _World:
    # x < 5 is sand (2), x >= 5 is ocean (8)
    def _tile(self, x, y): return _Tile(8 if x >= 5 else 2)


def _tick(agent_x):
    shark = Entity.from_id("E_SHARK", 5, 0)
    return shark, shark.tick_ai(agent_x, 0, _World())


shark, dmg = _tick(4)
assert dmg == 0 and not shark.alive, "shark reached onto the beach"
shark, dmg = _tick(6)
assert dmg == 0 and shark.alive, "shark bit from a tile away without charging"
print("ok: sharks never reach the beach, and never bite from a tile away")
