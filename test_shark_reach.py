"""A shark at sea must not reach onto the beach (SHARKS_BITE_ONLY_AT_SEA)."""
import config_rl
from entities import Entity


class _Tile:
    def __init__(self, t): self.tile_type = t


class _World:
    # x < 5 is sand (2), x >= 5 is ocean (8)
    def _tile(self, x, y): return _Tile(8 if x >= 5 else 2)


def _bite(agent_x):
    shark = Entity.from_id("E_SHARK", 5, 0)
    return shark.tick_ai(agent_x, 0, _World())


config_rl.SHARKS_BITE_ONLY_AT_SEA = True
assert _bite(4) == 0, "shark bit an agent standing on the beach"
assert _bite(6) == 9999, "shark failed to bite an agent in the ocean"
config_rl.SHARKS_BITE_ONLY_AT_SEA = False
assert _bite(4) == 9999, "old one-tile reach no longer works with the flag off"
print("ok: sharks bite at sea only, flag off restores the old reach")
