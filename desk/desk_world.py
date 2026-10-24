"""The desktop as an island: Chalce's screen cut into tiles, every tile land,
water past a hidden strip on each side. The brain gets the inputs it was
trained on, so what it learned on the island (find food, eat, explore) shows up
on the desktop with nothing retrained.

It is a real World with the islands wiped, and it runs through the real
ATSEnvironment.step, so hunger, eating and movement are the island's own rules.
The brain is a frozen copy (Sleep plan: nothing is learned while awake)."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import brain, config_rl
from ats_env import ATSEnvironment
from model_rl import RLPolicy
from mind.solution_loop import SolutionLoop
from world import World, Tile, TILE_EMPTY, TILE_OCEAN, TILE_OBJECT

TILE_PX = 32        # one island tile is this many screen pixels
HIDDEN = 3          # tiles of walkable land past each side edge (off screen)

# what the Shift+A menu can spawn: label -> item id from the island's registry
SPAWNABLE = {"Apple": "0001", "Berry bush": "0023", "Stick": "0008", "Wood": "0010",
             "Stone": "0012", "Oak tree": "0021", "Fallen log": "0024", "Coal": "0009"}


class DeskWorld(World):
    def __init__(self, cols, rows):
        super().__init__(seed=1)            # registry, items, bookkeeping; the islands get wiped below
        self.cols, self.rows = cols, rows
        self.tiles.clear()
        self.entities.clear()
        self._grid.clear()
        self.villages.clear()
        self._regrow.clear()
        self.agent_spawn = (cols // 2, rows - 2)

    def on_land(self, x, y):
        return -HIDDEN <= x < self.cols + HIDDEN and 0 <= y < self.rows

    def _tile(self, x, y):
        t = self.tiles.get((x, y))
        if t is None:
            t = Tile(tile_type=TILE_EMPTY, island_id=0) if self.on_land(x, y) else Tile(tile_type=TILE_OCEAN)
            self.tiles[(x, y)] = t
        return t

    def get_island_id_at(self, x, y):
        return 0 if self.on_land(x, y) else -1

    def get_island_tier_at(self, x, y):
        return 0

    def materialise(self, island_id):
        return False

    def tick_life(self, ax=1 << 30, ay=1 << 30):
        return []           # nothing grows back on its own; Chalce spawns things

    def put(self, x, y, item_id):
        if not self.on_land(x, y):
            return False
        t = self._tile(x, y)
        t.object_id, t.tile_type = item_id, TILE_OBJECT
        return True

    def remove(self, x, y):
        t = self.tiles.get((x, y))
        if t and t.object_id:
            t.object_id, t.tile_type = "", TILE_EMPTY
            return True
        return False

    def objects(self):
        return [(x, y, t.object_id) for (x, y), t in self.tiles.items() if t.object_id]


class DeskMind:
    """The trained brain living in a DeskWorld. step() is one brain tick."""

    def __init__(self, cols, rows, path=None):
        shape = brain.saved_shape()
        self.policy = RLPolicy(hidden_size=int(shape.get("hidden_size", config_rl.MIND_HIDDEN_SIZE)))
        brain.load(self.policy, None, path=path)
        self.policy.eval()
        torch.set_grad_enabled(False)
        self.sl = SolutionLoop(policy=self.policy)
        self.env = ATSEnvironment(max_ticks=10 ** 9, vary_world=False, use_loop=False)
        self.cols, self.rows = cols, rows
        self.reset()

    def reset(self):
        self.env.reset()
        w = DeskWorld(self.cols, self.rows)
        env = self.env
        env.world = w
        env.agent.world = w
        env.agent.x, env.agent.y = w.agent_spawn
        env.memory.__init__()
        env.rewards.discovered_tiles = {w.agent_spawn}
        self.state = env._state()
        self.sl.reset_memory()
        self.last_action = "wait"
        self.done = False

    @property
    def world(self):
        return self.env.world

    @property
    def agent(self):
        return self.env.agent

    def step(self):
        if self.done:               # it died: a new life, same desk
            objs = self.world.objects()
            self.reset()
            for x, y, i in objs:
                self.world.put(x, y, i)
        env = self.env
        mask = env.agent.get_action_mask(env.world, env.day_night)
        action = self.sl.choose_action(self.state, mask)
        self.state, _, self.done, info = env.step(action)
        from agent import ACTION_NAMES
        self.last_action = ACTION_NAMES.get(action, str(action))
        return info
