"""ATS simulation environment — main simulation loop.

Wires together:
  Agent  ←→  World  ←→  Rewards  ←→  Mind (NeedDetector + SolutionLoop)

Interactivity guarantee: action mask is re-computed every tick and fed
into the agent's state vector, so the model only ever sees valid actions.
"""

from __future__ import annotations

import random

import config_rl
from agent import Agent
from day_night import DayNight
from events import EventSystem
from memory import EpisodeMemory
from mind.need_detector import NeedDetector
from rewards import RewardEngine
from world import World


class ATSEnvironment:
    def __init__(self):
        self.world = World()
        self.agent = Agent(self.world)
        self.day_night = DayNight()
        self.rewards = RewardEngine()
        self.memory = EpisodeMemory()
        self.events = EventSystem()
        self.need_detector = NeedDetector()
        self.tick = 0
        self.done = False
        self.episode = 0
        self._killed_this_tick: list[str] = []
        self.world.mark_explored(
            self.agent.x,
            self.agent.y,
            self.day_night.perception_radius(self.agent.torch_active),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> list[float]:
        self.episode += 1
        self.world = World()
        self.agent = Agent(self.world)
        self.day_night = DayNight()
        self.rewards = RewardEngine()
        self.events = EventSystem()
        self.tick = 0
        self.done = False
        self._killed_this_tick = []
        self.world.mark_explored(
            self.agent.x,
            self.agent.y,
            self.day_night.perception_radius(self.agent.torch_active),
        )
        return self._state()

    # ------------------------------------------------------------------
    # State vector
    # ------------------------------------------------------------------
    def _state(self) -> list[float]:
        mem = self.memory.as_state_features(self.agent.x, self.agent.y)
        return self.agent.build_state(self.world, self.day_night, mem)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action_idx: int, disabled_actions: set = None, reward_rules: dict = None) -> tuple[list[float], float, bool, dict]:
        if disabled_actions is None:
            disabled_actions = set()
        if reward_rules is not None:
            self.rewards.rules.update(reward_rules)
        if self.done:
            return self._state(), 0.0, True, {}

        self.rewards.reset_tick()
        self._killed_this_tick = []

        # --- Agent action --------------------------------------------------
        failed_action_flag = 0
        if action_idx in disabled_actions:
            self.rewards.add(-1.0)
            failed_action_flag = action_idx
            self.agent.last_failed_action = action_idx
            from agent import ACTION_NAMES
            self.agent.last_action_name = "BLOCKED_" + ACTION_NAMES.get(action_idx, str(action_idx))
        else:
            self.agent.last_failed_action = 0
            self.agent.apply_action(action_idx, self.rewards, self.day_night)
        
        self.agent.crafting.tick()

        # --- Standing-still penalty ----------------------------------------
        self.rewards.update_standing_still(self.agent.x, self.agent.y)

        # Propagate agent attack / harvest results
        self._process_pending_drops()

        # --- Entity AI ticks -----------------------------------------------
        for ent in list(self.world.entities):
            if not ent.alive:
                continue
            dmg = ent.tick_ai(self.agent.x, self.agent.y, self.world)

            # Void proximity kill
            if dmg >= 9999:
                self.agent.health = 0
                self.rewards.on_death()
                self.agent.log_event("Consumed by the Void!")
                self.done = True
                break

            # Creeper AoE — check radius
            if ent.exploded:
                aoe = int(ent.data.get("aoe_radius", 2))
                dist = abs(ent.x - self.agent.x) + abs(ent.y - self.agent.y)
                if dist <= aoe:
                    self.agent.health -= dmg
                    self.rewards.on_damage()
                    self.agent.log_event(f"Caught in Creeper Explosion! (-{dmg} HP)")
            elif dmg > 0:
                self.agent.health -= dmg
                self.rewards.on_damage()
                from item_ids import entity_name
                self.agent.log_event(f"Attacked by {entity_name(ent.entity_id)} (-{dmg} HP)")
                # Poison / burn status effects from entity
                if ent.data.get("poison"):
                    self.agent.status["poisoned"] = max(self.agent.status.get("poisoned", 0), 5)
                if ent.data.get("burn"):
                    self.agent.status["burning"] = max(self.agent.status.get("burning", 0), 3)
                if ent.data.get("slow"):
                    self.agent.status["slowed"] = max(self.agent.status.get("slowed", 0), 4)
                if ent.data.get("blindness"):
                    self.agent.status["blinded"] = max(self.agent.status.get("blinded", 0), 3)

        # --- World / progression -------------------------------------------
        self.world.mark_explored(
            self.agent.x,
            self.agent.y,
            self.day_night.perception_radius(self.agent.torch_active),
        )
        self.day_night.advance()

        # Agent stat tick (hunger drain, status ticks, leg recovery, etc.)
        prev_level = self.agent.level
        self.agent.tick_stats(self.rewards)
        if self.agent.level > prev_level:
            self.rewards.on_level_up()

        self.events.tick(self.world)
        self.tick += 1

        # Biome reward
        cur_t = self.world._tile(self.agent.x, self.agent.y)
        biome_id = cur_t.biome if cur_t else 0
        if biome_id not in self.rewards.discovered_biomes:
            from biome import BIOMES
            self.agent.log_event(f"Discovered new Biome: {BIOMES.get(biome_id, 'Unknown')}!")
        self.rewards.on_biome_enter(biome_id)

        # Death / timeout
        if self.agent.health <= 0:
            self.rewards.on_death()
            self.agent.log_event("Agent fell in battle / perished.")
            self.done = True
            self.memory.decay()
        elif self.tick >= config_rl.MAX_TICKS:
            self.done = True
            self.rewards.on_survive_bonus()
            self.agent.log_event("Day/Night survival complete!")
            self.memory.decay()

        # Compute needs for logging / mind integration
        state_vec = self._state()
        needs = self.need_detector.detect(state_vec)

        info = {
            "tick_reward": self.rewards.tick_reward,
            "total_reward": self.rewards.total,
            "action": action_idx,
            "needs": needs,
            "action_name": self.agent.last_action_name,
            "failed_action_flag": failed_action_flag,
        }
        return state_vec, self.rewards.tick_reward, self.done, info

    # ------------------------------------------------------------------
    # Internal: flush pending drops queued by agent.apply_action
    # ------------------------------------------------------------------
    def _process_pending_drops(self):
        """Agent.apply_action may set agent._pending_kills list."""
        pending = getattr(self.agent, "_pending_kills", [])
        self.agent._pending_kills = []
        for killed_id, drop_id, kill_x, kill_y in pending:
            self.rewards.on_kill(killed_id)
            if killed_id == "E001":
                self.agent.slime_killed = True
            if drop_id:
                self.world.spawn_drop(kill_x, kill_y, drop_id)
            self.agent.xp += 3
            self.agent._check_level_up(self.rewards)

    # ------------------------------------------------------------------
    # Random action for exploration / baseline
    # ------------------------------------------------------------------
    def random_action(self) -> int:
        mask = self.agent.get_action_mask(self.world, self.day_night)
        valid = [i for i, m in enumerate(mask) if m]
        return random.choice(valid) if valid else 0
