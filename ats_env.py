"""ATS simulation environment — main simulation loop.

Wires together:
  Agent  ←→  World  ←→  Rewards  ←→  Mind (NeedDetector + SolutionLoop)

Interactivity guarantee: action mask is re-computed every tick and fed
into the agent's state vector, so the model only ever sees valid actions.
"""

from __future__ import annotations

import random

import config_rl
import loop
from agent import Agent
from day_night import DayNight
from events import EventSystem
from memory import EpisodeMemory
from mind.need_detector import NeedDetector
from rewards import RewardEngine
from world import World


class ATSEnvironment:
    def __init__(self, max_ticks: int | None = None,
                 world_seed: int | None = None, vary_world: bool = True,
                 use_loop: bool | None = None):
        """
        world_seed
            The seed for the first episode. None means one is drawn at random,
            so two runs are not secretly the same run.
        vary_world
            A NEW world every episode, which is the default and the point.

            What it replaces: World() takes `seed: int = 1337` and reset()
            passed nothing, so every episode was byte-identical - the same
            terrain, the same 161 entities in the same places, the same spawn.
            26,433 episodes on one map. Nothing in that setup ever asks the
            agent to generalise; it can succeed by memorising, and a model
            that has memorised one island is worth nothing on a second one.

            Pass False to pin the world, which is what evaluation and
            debugging want: comparing two brains on different terrain measures
            the terrain.
        """
        # The cap is a parameter, not a constant read at the point of use.
        # It is earned now (survival.py): short episodes while the agent is
        # bad, because that is when episodes-per-hour matters most, and long
        # ones once it can use them. A fixed MAX_TICKS cannot express that,
        # and raising the constant to ten days would only make each failure
        # take ten times longer to watch.
        self.max_ticks = int(max_ticks or config_rl.MAX_TICKS)
        # W4: death lands it in the Loop instead of ending the episode.
        self.use_loop = getattr(config_rl, "LOOP", False) if use_loop is None else use_loop
        # Its own generator, seeded once. A per-episode seed drawn from this
        # gives variety AND reproducibility: the run seed plus the episode
        # number names the world exactly, which is what makes a recorded death
        # something you can go back to.
        self.vary_world = bool(vary_world)
        self.run_seed = int(world_seed) if world_seed is not None \
            else random.randrange(1, 2 ** 31 - 1)
        self._seeds = random.Random(self.run_seed)
        self.world_seed = self.run_seed
        self.world = World(seed=self.world_seed)
        self.agent = Agent(self.world)
        self.day_night = DayNight()
        self.rewards = RewardEngine()
        self.rewards.discovered_islands.add(self.world.starting_island.island_id)
        self.rewards.discovered_tiles.add((self.agent.x, self.agent.y))
        self.memory = EpisodeMemory()
        self.events = EventSystem()
        self.need_detector = NeedDetector()
        self.tick = 0
        self.done = False
        self.death_cause = None
        self.episode = 0
        self._killed_this_tick: list[str] = []
        self.world.mark_explored(
            self.agent.x,
            self.agent.y,
            self.day_night.perception_radius(self.agent.torch_active),
        )
        self._loop_reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> list[float]:
        self.episode += 1
        # A different island every episode unless pinned. This line used to be
        # World() with no argument, which meant seed 1337 forever.
        if self.vary_world:
            self.world_seed = self._seeds.randrange(1, 2 ** 31 - 1)
        self.world = World(seed=self.world_seed)
        self.agent = Agent(self.world)
        self.day_night = DayNight()
        self.rewards = RewardEngine()
        self.rewards.discovered_islands.add(self.world.starting_island.island_id)
        self.rewards.discovered_tiles.add((self.agent.x, self.agent.y))
        self.events = EventSystem()
        self.tick = 0
        self.done = False
        self.death_cause = None
        self._killed_this_tick = []
        self.world.mark_explored(
            self.agent.x,
            self.agent.y,
            self.day_night.perception_radius(self.agent.torch_active),
        )
        self._loop_reset()
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
        if self.bearings > 0:
            return self._bearings_tick(action_idx)

        # --- Agent action --------------------------------------------------
        failed_action_flag = 0
        if action_idx in disabled_actions:
            self.rewards.add(-1.0, "disabled_action")
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
        self.world.index_creatures()
        # How many ticks one step takes the agent where it is (agent.py's
        # water drag): a charging shark moves at this pace, so a swimmer
        # slowed to 1/3 in deep sea can still dodge it (2026-10-13).
        a, here = self.agent, self.world._tile(self.agent.x, self.agent.y).tile_type
        self.world.swim_pace = (3 if here == 8 and a.sea_blocks_crossed >= 3
                                else 2 if here in (5, 8) else 1)
        for ent in list(self.world.entities):
            if not ent.alive:
                continue
            dmg = ent.tick_ai(self.agent.x, self.agent.y, self.world)

            # Instant kill (the Void, or a shark). Named after what it was:
            # every one of these used to say "the Void", which hid that
            # sharks were the ones doing it.
            if dmg >= 9999:
                killer = ent.spec.killer_name
                self.agent.health = 0
                self.agent.last_damage_cause = killer
                self.agent.log_event("Consumed by the Void!" if killer == "the Void"
                                     else "Killed by %s!" % killer)
                if not self.use_loop:       # the Loop: the death check below handles it
                    self.death_cause = killer
                    self.rewards.on_death()
                    self.done = True
                break

            # Creeper AoE — check radius
            if ent.exploded:
                aoe = ent.spec.hostility_args["radius"]
                dist = abs(ent.x - self.agent.x) + abs(ent.y - self.agent.y)
                if dist <= aoe:
                    self.agent.health -= dmg
                    self.agent.last_damage_cause = "a Creeper explosion"
                    self.rewards.on_damage(dmg)
                    self.agent.log_event(f"Caught in Creeper Explosion! (-{dmg} HP)")
            elif dmg > 0:
                self.agent.health -= dmg
                self.rewards.on_damage(dmg)
                from item_ids import entity_name
                self.agent.last_damage_cause = entity_name(ent.entity_id)
                self.agent.log_event(f"Attacked by {entity_name(ent.entity_id)} (-{dmg} HP)")
                # Poison / burn status effects from entity
                for status, ticks in ent.spec.on_hit.items():
                    self.agent.status[status] = max(self.agent.status.get(status, 0), ticks)

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

        if not self.in_loop:
            self.events.tick(self.world)
        self.world.tick_life(self.agent.x, self.agent.y)
        self.tick += 1

        # Stepping onto a bridge builds the island it leads to, if it is not
        # built yet - its land, resources and creatures.
        dest = self.world.bridge_target(self.agent.x, self.agent.y) \
            if hasattr(self.world, "bridge_target") else -1
        if dest >= 0 and self.world.materialise(dest):
            self.agent.log_event("A bridge - something lies across it")

        # Per-tile exploration reward (first visit only). The sea pays only
        # if EXPLORE_PAYS_OCEAN says so - see config_rl.
        here = self.world._tile(self.agent.x, self.agent.y)
        if self.in_loop:
            pass                                # a room repeats: nothing new to find
        elif getattr(config_rl, "EXPLORE_PAYS_OCEAN", True) or here.tile_type != 8:
            self.rewards.on_new_tile(self.agent.x, self.agent.y)

        # Island discovery (only for genuinely new islands outside the starting spawn island 0)
        cur_t = self.world._tile(self.agent.x, self.agent.y)
        island_id = cur_t.island_id if cur_t else -1
        if island_id > 0 and island_id not in self.rewards.discovered_islands:
            self.rewards.discovered_islands.add(island_id)
            idef = self.world.island_registry.get_island(island_id)
            name = idef.name if idef else f"Island {island_id}"
            self.agent.log_event(f"★ Discovered New Island: {name}! (+50)")
            self.rewards.on_island_discovery(island_id)

        # Sea danger. Two modes, behind config_rl.SEA_SURVIVABLE:
        #   True  (new): once past the grace ticks, a shark deals HEAVY damage
        #                every tick, but it is ESCAPABLE - retreat to land and
        #                you live. This gives the agent a gradient to learn a
        #                boundary from, instead of a sudden teleport/one-shot it
        #                can never learn against.
        #   False (old): the original teleport-to-spawn / drown behaviour.
        sea_over = self.agent.ocean_ticks >= config_rl.OCEAN_SHARK_TICKS
        charges = getattr(config_rl, "SHARK_CHARGES", True)
        if sea_over and charges:
            # W2b: one shark surfaces where the agent can see it, waits,
            # then charges (entities.py _tick_charge). Contact kills.
            if self.world.surface_shark(self.agent.x, self.agent.y,
                                        config_rl.PATCH_RADIUS):
                self.agent.log_event("A fin breaks the surface!")
        elif sea_over and getattr(config_rl, "SEA_SURVIVABLE", True):
            self.agent.health -= config_rl.SEA_DAMAGE_PER_TICK
            self.agent.last_damage_cause = "a shark"
            self.world.spawn_shark(self.agent.x, self.agent.y)
            self.rewards.on_ocean_death()       # per-bite penalty, not a cliff
            self.agent.log_event(
                "A shark tears in! -%d HP - get to land!"
                % config_rl.SEA_DAMAGE_PER_TICK)
            # Death is handled by the health<=0 check below, in this same tick.

        if sea_over and not charges and not getattr(config_rl, "SEA_SURVIVABLE", True):
            self.rewards.on_ocean_death()  # -500 punishment
            self.world.spawn_shark(self.agent.x, self.agent.y)
            if self.agent.health >= 20:
                # Survived shark strike: respawn on safe land with low health and continue run
                self.agent.health = 15
                self.agent.x, self.agent.y = self.world.agent_spawn
                self.agent.z = 0
                self.agent.ocean_ticks = 0
                self.agent.sea_blocks_crossed = 0
                self.agent._water_move_counter = 0
                self.agent.log_event("Shark strike in open ocean! Escaped with low HP (-500 rew, HP:15, Respawned on land)")
            else:
                # Health is below 20: Harsh stop — terminate episode with BAD result
                self.agent.health = 0
                self.agent.last_damage_cause = "drowning"
                self.death_cause = "drowning"
                self.agent.log_event("Harsh Stop: Drowned in ocean with critical HP < 20 (-500 rew, Perished)")
                self.done = True
                self.memory.decay()

        if self.in_loop and self.agent.health > 0:
            self._loop_check()

        # Death / timeout
        if self.done:
            pass
        elif self.agent.health <= 0 and self.use_loop:
            self.death_cause = self.agent.last_damage_cause or "unknown causes"
            self.agent.log_event("Died of %s." % self.death_cause)
            self._died_into_loop()
        elif self.agent.health <= 0:
            self.rewards.on_death()
            # Say what actually killed it. This line read "Agent fell in
            # battle / perished." for EVERY death at zero health - starvation,
            # lava, poison, a fall, the lot - and a death message that names
            # the wrong cause is worse than none, because it is believed.
            # The trained brain's deaths were pure starvation and this called
            # them battles for 26,000 episodes.
            self.death_cause = self.agent.last_damage_cause or "unknown causes"
            self.agent.log_event("Died of %s." % self.death_cause)
            self.done = True
            self.memory.decay()
        elif self.tick >= self.max_ticks:
            self.done = True
            # Paid for the length actually survived, not a flat amount. With
            # an earned cap the easiest rung and the hardest would otherwise
            # pay identically - see rewards.on_survive_bonus.
            if not self.in_loop:                # paid for the life it is on
                self.rewards.on_survive_bonus(self.tick - self.life_start)
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
            "progression_efficiency": self.rewards.compute_progression_efficiency(self.tick),
            "progression_points": self.rewards.progression_points,
            "capabilities": list(self.agent.capabilities),
            "damage_taken": self.rewards.damage_taken,
            "damage_dealt": self.rewards.damage_dealt,
            "island_id": island_id,
            "ocean_ticks": self.agent.ocean_ticks,
            "in_loop": self.in_loop,
            "deaths": self.deaths,
            "rooms_beaten": self.rooms_beaten,
        }
        return state_vec, self.rewards.tick_reward, self.done, info

    # ------------------------------------------------------------------
    # W4: the Loop (loop.py, docs/W4-LOOP-FLOW.md)
    # ------------------------------------------------------------------
    def _loop_reset(self):
        self.real_world = self.world
        self.in_loop = False
        self.loop_room = None
        self.loop_goals = 0             # rooms beaten since it last died in the world
        self.loop_room_deaths = 0       # failures in the room it is in
        self.bearings = 0
        self.deaths = 0
        self.rooms_beaten = 0
        self.room_stats = {}            # room tally -> [won, out of time, died, sent back]
        self.death_causes: list[str] = []
        self.world_pos = self.world.agent_spawn
        self.life_start = 0
        self.loop_on_land = 0
        self.loop_was_on_land = False
        self.loop_arrival_hunger = 0
        self.loop_shore_ticks = getattr(config_rl, "LOOP_SHORE_TICKS", 20)
        self._loop_rng = random.Random(self.world_seed)
        self.agent.in_loop = False

    def _bearings_tick(self, action_idx):
        """A moment to get its bearings after arriving: the world holds
        still, nothing hurts, and its action does nothing."""
        self.bearings -= 1
        self.tick += 1
        if self.tick >= self.max_ticks:
            self.done = True
            if not self.in_loop:
                self.rewards.on_survive_bonus(self.tick - self.life_start)
            self.memory.decay()
        info = {"tick_reward": self.rewards.tick_reward, "total_reward": self.rewards.total,
                "action": action_idx, "action_name": "bearings", "failed_action_flag": 0,
                "in_loop": self.in_loop, "deaths": self.deaths,
                "rooms_beaten": self.rooms_beaten}
        return self._state(), self.rewards.tick_reward, self.done, info

    def _place(self, world, pos, health=100, hunger=100):
        """Put the agent into `world` at `pos`, in a fresh body."""
        a = self.agent
        self.world = a.world = world
        a.in_loop = self.in_loop = world is not self.real_world
        a.x, a.y = pos
        a.z = 0
        a.health, a.hunger, a.stamina = health, hunger, 100
        for k in a.status:
            a.status[k] = 0
        a.leg_injured = 0
        a.ocean_ticks = a.sea_blocks_crossed = a._water_move_counter = 0
        a._hunger_counter = 0
        a.sprint = a.inventory_open = False
        a.seen, a.visit = {}, {}            # a new place: its remembered map starts empty
        self.bearings = getattr(config_rl, "LOOP_BEARINGS_TICKS", 20)

    def _enter_room(self, room, again=False):
        if not again:
            self.loop_room_deaths = 0
        self.loop_room = room
        self._place(loop.LoopWorld(self.real_world, room), room.start,
                    room.health, room.hunger)
        a = self.agent
        a.inventory = [{"id": "", "count": 0} for _ in range(config_rl.INVENTORY_SLOTS)]
        for slot, item_id, n in room.kit:
            a.inventory[slot] = {"id": item_id, "count": n}
        a.selected_slot = room.selected
        a.has_sword = any(s["id"] in ("0002", "0013", "0033", "0034") for s in a.inventory)
        self.loop_on_land = 0
        self.loop_was_on_land = False
        self.loop_room_ticks = 0        # ticks in this try (or since the doors opened)
        self.loop_arrival_hunger = a.hunger
        a.log_event("The Loop: the %s room" % room.name)

    def _tally_room(self, outcome):
        """Count how a try in the current room ended (0 won, 1 out of time, 2 died,
        3 sent back), keyed by kind; lever rooms split hidden / in sight."""
        room = self.loop_room
        key = room.name if room.kind == "lever" else room.kind
        self.room_stats.setdefault(key, [0, 0, 0, 0])[outcome] += 1

    def _room_failed(self):
        """Died in the room, sent back to its start, or out of time: the same room again,
        or a different one after LOOP_ROOM_DEATHS_TO_SWAP failures in it."""
        self.loop_room_deaths += 1
        if self.loop_room_deaths >= getattr(config_rl, "LOOP_ROOM_DEATHS_TO_SWAP", 5):
            self._enter_room(loop.pick_room(self.death_cause, self._loop_rng,
                                            avoid=self.loop_room.kind))
        else:
            self._enter_room(self.loop_room, again=True)

    def _died_into_loop(self):
        in_world = not self.in_loop
        self.rewards.on_loop_death(in_world, getattr(config_rl, "LOOP_DEATH_PENALTY", -20.0))
        self.deaths += 1
        self.death_causes.append(self.death_cause)
        self.memory.decay()
        if in_world:
            self.world_pos = (self.agent.x, self.agent.y)
            self.loop_goals = 0
            self._enter_room(loop.pick_room(self.death_cause, self._loop_rng))
        else:
            self._tally_room(2)
            self._room_failed()

    def _loop_check(self):
        """Once a tick in a room: out through an open door, sent back, or goal met."""
        room, a = self.loop_room, self.agent
        outside = not (0 <= a.x < loop.SIZE and 0 <= a.y < loop.SIZE)
        if self.world.doors_open and outside:
            self._leave_room()
            return
        self.world.wrap_all(a)
        self.loop_room_ticks += 1
        late = self.loop_room_ticks > getattr(config_rl, "LOOP_ROOM_TIMEOUT", 600)
        if self.world.doors_open:
            if late:                    # it won the room; don't let finding a door eat the episode
                self._leave_room()
            return
        if late:
            self.rewards.add(getattr(config_rl, "LOOP_TIMEOUT_PENALTY", -5.0), "loop_timeout")
            a.log_event("Out of time in the %s room" % room.name)
            self._tally_room(1)
            self._room_failed()
            return
        if room.sends_back and room.sends_back(self):
            a.log_event("Back to the start")
            self._tally_room(3)
            self._room_failed()
            return
        if room.goal(self):
            self.rewards.add(getattr(config_rl, "LOOP_GOAL_REWARD", 10.0), "loop_goal")
            self.rooms_beaten += 1
            self._tally_room(0)
            if room.goal_is_exit:
                self._leave_room()
            else:
                self.world.doors_open = True
                self.loop_room_ticks = 0
                a.log_event("The doors open")

    def _leave_room(self):
        self.loop_goals += 1
        room = self.loop_room
        if self.loop_goals < getattr(config_rl, "LOOP_GOALS_BEFORE_DROP", 3):
            self._enter_room(loop.pick_room(None, self._loop_rng, avoid=room.kind))
            return
        spot = room.find(self.real_world, self.world_pos) or self.real_world.agent_spawn
        self._place(self.real_world, spot)
        self.loop_room = None
        self.life_start = self.tick
        self.agent.log_event("Back in the world, by the %s" % room.name)


    # ------------------------------------------------------------------
    # Internal: flush pending drops queued by agent.apply_action
    # ------------------------------------------------------------------
    def _process_pending_drops(self):
        """Agent.apply_action may set agent._pending_kills list."""
        pending = getattr(self.agent, "_pending_kills", [])
        self.agent._pending_kills = []
        cur_tier = self.world.get_island_tier_at(self.agent.x, self.agent.y)
        for killed_id, drop_id, kill_x, kill_y in pending:
            self.rewards.on_kill(killed_id, island_tier=max(0, cur_tier))
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
