"""Agent stats, 7-slot inventory, action mask, and state vector generation."""

import time
import random
import config_rl
from crafting import CraftingSystem
from item_ids import REGISTRY, item_name, entity_name


# ---------------------------------------------------------------------------
# Action names (human-readable, for GUI only — model sees indices)
# ---------------------------------------------------------------------------
ACTION_NAMES = {
    config_rl.ACT_MOVE_FORWARD: "move_fwd",
    config_rl.ACT_MOVE_BACKWARD: "move_back",
    config_rl.ACT_MOVE_LEFT: "move_left",
    config_rl.ACT_MOVE_RIGHT: "move_right",
    config_rl.ACT_SPRINT_ON: "sprint_on",
    config_rl.ACT_SPRINT_OFF: "sprint_off",
    config_rl.ACT_JUMP: "jump",
    config_rl.ACT_ATTACK: "attack",
    config_rl.ACT_PICK_UP: "pick_up",
    config_rl.ACT_INTERACT: "interact",
    config_rl.ACT_USE_ITEM: "use_item",
    config_rl.ACT_OPEN_INVENTORY: "open_inv",
    config_rl.ACT_CLOSE_INVENTORY: "close_inv",
    config_rl.ACT_OPEN_CRAFTING: "open_craft",
    config_rl.ACT_ADD_TO_CRAFTING: "add_craft",
    config_rl.ACT_CLOSE_CRAFTING: "close_craft",
    config_rl.ACT_SLEEP: "sleep",
    config_rl.ACT_WAIT: "wait",
}

for _i in range(config_rl.INVENTORY_SLOTS):
    ACTION_NAMES[config_rl.ACT_SELECT_SLOT_BASE + _i] = f"sel_slot_{_i}"


class Agent:
    """RL agent with 7-slot inventory, context-masked actions, facing tracking, and stat tracking."""

    def __init__(self, world):
        self.world = world
        self.x, self.y = world.agent_spawn
        self.z = 0
        self.health = 100
        self.hunger = 100
        self.stamina = 100
        self.xp = 0
        self.level = 1
        self.leg_injured = 0

        # Facing direction: (dx, dy) and label ("N", "S", "W", "E")
        self.facing = (0, -1)  # Default North
        self.facing_dir = "N"

        # 24-slot inventory: each slot = {"id": str, "count": int}
        self.inventory = [{"id": "", "count": 0} for _ in range(config_rl.INVENTORY_SLOTS)]
        self.selected_slot = 0
        self.inventory_open = False

        # Assign starter gear (low class)
        self._grant_starter_gear()

        self.sprint = False
        self.has_sword = True if self.inventory[0]["id"] in ("0002", "0013", "0033", "0034") else False
        self.slime_killed = False

        self.status = {
            "poisoned": 0,
            "burning": 0,
            "slowed": 0,
            "blinded": 0,
            "speed_boost": 0,
            "cold": 0,
            "acid_burn": 0,
        }
        self.ocean_ticks = 0
        self.sea_blocks_crossed = 0
        self._water_move_counter = 0

        self.crafting = CraftingSystem()
        self.torch_active = False
        self.recent_actions = []
        self.event_log: list[tuple[str, float]] = []  # (event_text, timestamp)
        self._hunger_counter = 0
        self._xp_for_next_level = 10  # doubles each level
        self._pending_kills: list[tuple[str, str, int, int]] = []  # (entity_id, drop_id, x, y)
        self.last_action_name = "none"
        self.last_failed_action = 0

        # Minecraft-style emergent capability set
        self.capabilities: set[str] = set()
        if self.has_sword:
            self.capabilities.add("CAP_BASIC_WEAPON")

    @property
    def water_speed_mult(self) -> float:
        """Water speed penalty: 0.5x in H2O/sea entrance, 0.33x after crossing 3 sea blocks."""
        cur_t = self.world._tile(self.x, self.y)
        if cur_t and cur_t.tile_type == 8:  # TILE_OCEAN
            return 0.33 if self.sea_blocks_crossed >= 3 else 0.50
        elif cur_t and cur_t.tile_type == 5:  # TILE_WATER
            return 0.50
        return 1.0

    def _grant_starter_gear(self):
        """Give the agent only a basic sword — no free food or resources.
        The agent must forage for food/materials naturally."""
        starters = [
            ("0002", 1),  # Sword Basic (slot 0 only)
        ]
        for slot_idx, (item_id, count) in enumerate(starters):
            if slot_idx < len(self.inventory):
                self.inventory[slot_idx] = {"id": item_id, "count": count}

    def get_inventory_item_ids(self) -> set[str]:
        """Return a set of all item IDs present in inventory."""
        return {s["id"] for s in self.inventory if s["id"] and s["count"] > 0}

    def unlock_capability(self, cap_name: str, rewards):
        """Unlock an emergent capability and notify reward engine."""
        if cap_name not in self.capabilities:
            self.capabilities.add(cap_name)
            rewards.on_capability_unlocked(cap_name)
            self.log_event(f"Unlocked Capability: {cap_name}!")

    def log_event(self, text: str):
        """Append an event to the human-readable event stream (max 30 kept)."""
        self.event_log.append((text, time.time()))
        if len(self.event_log) > 30:
            self.event_log.pop(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _norm(self, value, max_value):
        return max(0.0, min(1.0, value / float(max_value)))


    def _weapon_damage(self):
        """Return best weapon damage from inventory."""
        best = 1  # bare hands
        for slot in self.inventory:
            item = REGISTRY.get_item(slot["id"])
            if item and item.get("type") == "weapon":
                best = max(best, int(item.get("damage", 5)))
        return best

    def _add_recent(self, action_idx):
        self.recent_actions.append(action_idx)
        self.recent_actions = self.recent_actions[-5:]

    def _add_to_inventory(self, item_id, count=1):
        """Add an item to inventory (stacking if same type already present)."""
        if not item_id:
            return False
        # Try to stack first
        for slot in self.inventory:
            if slot["id"] == item_id and slot["count"] > 0:
                slot["count"] += count
                return True
        # Otherwise find first empty slot
        for slot in self.inventory:
            if not slot["id"] or slot["count"] <= 0:
                slot["id"] = item_id
                slot["count"] = count
                return True
        return False  # inventory full

    def _consume_from_slot(self, slot_idx):
        """Consume one unit from a slot. Returns the item_id consumed, or ''."""
        slot = self.inventory[slot_idx]
        if not slot["id"] or slot["count"] <= 0:
            return ""
        item_id = slot["id"]
        slot["count"] -= 1
        if slot["count"] <= 0:
            slot["id"] = ""
            slot["count"] = 0
        return item_id

    def _check_level_up(self, rewards):
        """Check and apply level-ups, notifying reward engine."""
        while self.xp >= self._xp_for_next_level:
            self.xp -= self._xp_for_next_level
            self.level += 1
            self._xp_for_next_level *= 2
            rewards.on_level_up()
            self.log_event(f"Level Up! Reached Level {self.level}")

    def get_facing_tile_info(self, world) -> str:
        """Return a concise description of what is in front of the agent."""
        fx = self.x + self.facing[0]
        fy = self.y + self.facing[1]
        ent = world.entity_at(fx, fy)
        if ent and ent.alive:
            return f"[{entity_name(ent.entity_id)}] (HP:{ent.hp})"
        tile = world._tile(fx, fy)
        if not tile:
            return "Void"
        if not tile.explored:
            return "Unexplored (Fog)"
        if tile.object_id:
            return f"[{item_name(tile.object_id)}]"
        if tile.tile_type == 1:
            return "[Mountain Wall]"
        if tile.tile_type == 5:
            return "[Water Pond]"
        if tile.tile_type == 6:
            return "[Lava Pool]"
        return "Open Ground"

    # ------------------------------------------------------------------
    # Action mask — dynamic action window (model-side only)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Action mask — dynamic action window (model-side only)
    # ------------------------------------------------------------------
    def get_action_mask(self, world, day_night):
        """Return list[int] of length ACTION_SIZE — 1 = valid, 0 = masked."""
        mask = [0] * config_rl.ACTION_SIZE

        # Movement — available if tile passable or if it is a progression gate the agent can unlock
        inv_items = self.get_inventory_item_ids()
        for act, (dx, dy) in [
            (config_rl.ACT_MOVE_FORWARD, (0, -1)),
            (config_rl.ACT_MOVE_BACKWARD, (0, 1)),
            (config_rl.ACT_MOVE_LEFT, (-1, 0)),
            (config_rl.ACT_MOVE_RIGHT, (1, 0)),
        ]:
            nx, ny = self.x + dx, self.y + dy
            can_pass = world.in_bounds_passable(nx, ny, world.door_unlocked)
            if not can_pass:
                g_tile = world._tile(nx, ny)
                if g_tile and g_tile.tile_type == 7:  # TILE_GATE
                    for g_id, g_info in world.gates.items():
                        if g_info["pos"] == (nx, ny):
                            if (g_info.get("req_cap") in self.capabilities) or (g_info.get("req_item") in inv_items):
                                can_pass = True
                                break
            if can_pass:
                mask[act] = 1

        # Sprint on/off
        if not self.sprint and self.stamina > 10 and not self.leg_injured:
            mask[config_rl.ACT_SPRINT_ON] = 1
        if self.sprint:
            mask[config_rl.ACT_SPRINT_OFF] = 1

        # Jump — need stamina
        if self.stamina >= 5:
            mask[config_rl.ACT_JUMP] = 1

        # Attack / Harvest — any entity within 3x3 OR harvestable object facing / near
        ent_cheb, ent_cheb_dist = world.nearest_entity_chebyshev(self.x, self.y)
        if ent_cheb and ent_cheb_dist <= 1:
            mask[config_rl.ACT_ATTACK] = 1
        else:
            # Check if harvestable object is facing or adjacent
            fx, fy = self.x + self.facing[0], self.y + self.facing[1]
            f_tile = world._tile(fx, fy)
            c_tile = world._tile(self.x, self.y)
            if (f_tile and f_tile.object_id) or (c_tile and c_tile.object_id):
                mask[config_rl.ACT_ATTACK] = 1

        # Pick up — object on current or any of 8 neighbors (3×3 square)
        _neighbors_3x3 = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
        for dx, dy in _neighbors_3x3:
            adj = world._tile(self.x + dx, self.y + dy)
            if adj and adj.object_id:
                mask[config_rl.ACT_PICK_UP] = 1
                break

        # Interact — NPC, chest, pond, etc. within 3×3 (Chebyshev ≤ 1)
        if ent_cheb and ent_cheb_dist <= 1:
            ent_data = ent_cheb.data
            if ent_data.get("type") in ("neutral", "passive"):
                mask[config_rl.ACT_INTERACT] = 1

        # Inventory management
        if not self.inventory_open:
            mask[config_rl.ACT_OPEN_INVENTORY] = 1
        else:
            mask[config_rl.ACT_CLOSE_INVENTORY] = 1
            # Only enable slot selection for occupied slots — empty slots are pointless to select
            for i in range(config_rl.INVENTORY_SLOTS):
                if self.inventory[i]["id"] and self.inventory[i]["count"] > 0:
                    mask[config_rl.ACT_SELECT_SLOT_BASE + i] = 1
            # Use item (if selected slot has something)
            if self.inventory[self.selected_slot]["id"] and self.inventory[self.selected_slot]["count"] > 0:
                mask[config_rl.ACT_USE_ITEM] = 1

        # Crafting
        if not self.crafting.open:
            mask[config_rl.ACT_OPEN_CRAFTING] = 1
        else:
            mask[config_rl.ACT_CLOSE_CRAFTING] = 1
            if self.inventory[self.selected_slot]["id"]:
                mask[config_rl.ACT_ADD_TO_CRAFTING] = 1

        # Sleep — need bed, must be night
        if day_night.is_night:
            has_bed = any(
                REGISTRY.get_item(s["id"]) and REGISTRY.get_item(s["id"]).get("sleep")
                for s in self.inventory if s["id"]
            )
            if has_bed:
                mask[config_rl.ACT_SLEEP] = 1

        # Wait — always valid
        mask[config_rl.ACT_WAIT] = 1

        return mask

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------
    def apply_action(self, action_idx, rewards, day_night):
        self._add_recent(action_idx)
        self.last_action_name = ACTION_NAMES.get(action_idx, str(action_idx))

        # --- Wait ---
        if action_idx == config_rl.ACT_WAIT:
            rewards.on_wait()
            return

        # --- Sprint on/off ---
        if action_idx == config_rl.ACT_SPRINT_ON:
            if self.stamina > 10 and not self.leg_injured:
                self.sprint = True
            return
        if action_idx == config_rl.ACT_SPRINT_OFF:
            self.sprint = False
            return

        # --- Inventory open/close ---
        if action_idx == config_rl.ACT_OPEN_INVENTORY:
            self.inventory_open = True
            return
        if action_idx == config_rl.ACT_CLOSE_INVENTORY:
            self.inventory_open = False
            return

        # --- Slot selection ---
        if config_rl.ACT_SELECT_SLOT_BASE <= action_idx < config_rl.ACT_SELECT_SLOT_BASE + config_rl.INVENTORY_SLOTS:
            slot_idx = action_idx - config_rl.ACT_SELECT_SLOT_BASE
            s_item = self.inventory[slot_idx]
            if s_item["id"] and s_item["count"] > 0:
                self.selected_slot = slot_idx
                self.last_failed_action = 0
                self.log_event(f"Selected slot {slot_idx}: {item_name(s_item['id'])} x{s_item['count']}")
            else:
                self.last_failed_action = action_idx
                self.log_event(f"Slot {slot_idx} is EMPTY [Disabled]")
                rewards.add(-1.0)
            return

        # --- Use item ---
        if action_idx == config_rl.ACT_USE_ITEM:
            s_item = self.inventory[self.selected_slot]
            if s_item["id"] and s_item["count"] > 0:
                self._use_item(rewards)
            else:
                self.last_failed_action = action_idx
                self.log_event(f"Cannot use: Slot {self.selected_slot} is EMPTY [Disabled]")
                rewards.add(-1.0)
            return

        # --- Crafting ---
        if action_idx == config_rl.ACT_OPEN_CRAFTING:
            self.crafting.open_menu()
            return
        if action_idx == config_rl.ACT_CLOSE_CRAFTING:
            self.crafting.open = False
            self.crafting.selected = []
            return
        if action_idx == config_rl.ACT_ADD_TO_CRAFTING:
            self._add_to_crafting(rewards)
            return

        # --- Sleep ---
        if action_idx == config_rl.ACT_SLEEP and day_night.is_night:
            has_bed = any(
                REGISTRY.get_item(s["id"]) and REGISTRY.get_item(s["id"]).get("sleep")
                for s in self.inventory if s["id"]
            )
            if has_bed:
                self.health = min(100, self.health + 30)
                day_night.tick = int(day_night.length * 0.5)
                rewards.on_sleep()
                self.log_event("Rested in Bed (+30 HP)")
            return

        # --- Pick up (3×3 square around agent) ---
        if action_idx == config_rl.ACT_PICK_UP:
            item_id = ""
            _neighbors_3x3 = [(0, 0)] + [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if not (dx == 0 and dy == 0)]
            for dx, dy in _neighbors_3x3:
                item_id = self.world.pickup_at(self.x + dx, self.y + dy)
                if item_id:
                    break
            if item_id:
                added = self._add_to_inventory(item_id)
                if added:
                    rewards.on_pickup(item_id)
                    self.log_event(f"Picked up {item_name(item_id)}")
                    if item_id in ("0002", "0013", "0033", "0034"):
                        self.has_sword = True
                        self.unlock_capability("CAP_BASIC_WEAPON", rewards)
                    item = REGISTRY.get_item(item_id)
                    if item and item.get("xp"):
                        self.xp += int(item["xp"])
                        self._check_level_up(rewards)
            return

        # --- Attack / Harvest ---
        if action_idx == config_rl.ACT_ATTACK:
            dmg_to_deal = self._weapon_damage()
            target_id, drop_id, is_world_obj = self.world.attack_at(
                self.x, self.y, dmg_to_deal,
                attacker_pos=(self.x, self.y),
                facing_dx=self.facing[0], facing_dy=self.facing[1]
            )
            rewards.damage_dealt += float(dmg_to_deal)
            if target_id:
                if is_world_obj:
                    # Harvested world object (tree, bush, rock)
                    self.log_event(f"Harvested {item_name(target_id)} -> +1 {item_name(drop_id)}")
                    self._add_to_inventory(drop_id)
                    rewards.on_pickup(drop_id)
                    self.xp += 1
                    self._check_level_up(rewards)
                else:
                    # Killed entity
                    ent = next((e for e in self.world.entities if e.entity_id == target_id and not e.alive), None)
                    kx, ky = (ent.x, ent.y) if ent else (self.x, self.y)
                    self._pending_kills.append((target_id, drop_id, kx, ky))
                    self.log_event(f"Defeated {entity_name(target_id)}!")
                    self.xp += 3
                    self._check_level_up(rewards)

                    # Boss progression validator check
                    if target_id in ("E009", "E023", "E025"):
                        self.unlock_capability("CAP_BOSS_SLAYER", rewards)
                        rewards.on_boss_defeated(target_id)
                        self.log_event(f"★ Progression Validator Passed! Defeated Boss {entity_name(target_id)}!")
            return

        # --- Interact ---
        if action_idx == config_rl.ACT_INTERACT:
            _neighbors_3x3 = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
            for dx, dy in _neighbors_3x3:
                tile = self.world._tile(self.x + dx, self.y + dy)
                if tile and tile.object_id == "0023":  # Berry Bush
                    tile.object_id = ""
                    tile.tile_type = 0
                    self._add_to_inventory("0001")
                    rewards.on_pickup("0001")
                    self.log_event("Gathered fresh Berries/Apples from bush!")
                    return
            return

        # --- Jump ---
        if action_idx == config_rl.ACT_JUMP:
            if self.stamina >= 5:
                self.stamina -= 5
            return

        # --- Movement (forward/back/left/right) ---
        dx, dy = 0, 0
        if action_idx == config_rl.ACT_MOVE_FORWARD:
            dy = -1
            self.facing = (0, -1)
            self.facing_dir = "N"
        elif action_idx == config_rl.ACT_MOVE_BACKWARD:
            dy = 1
            self.facing = (0, 1)
            self.facing_dir = "S"
        elif action_idx == config_rl.ACT_MOVE_LEFT:
            dx = -1
            self.facing = (-1, 0)
            self.facing_dir = "W"
        elif action_idx == config_rl.ACT_MOVE_RIGHT:
            dx = 1
            self.facing = (1, 0)
            self.facing_dir = "E"
        else:
            return

        nx, ny = self.x + dx, self.y + dy

        # Check if attempting to pass through a progression gate
        unlocked_gate = self.world.try_unlock_gate(nx, ny, self.capabilities, self.get_inventory_item_ids())
        if unlocked_gate:
            rewards.on_gate_unlocked(unlocked_gate)
            self.unlock_capability(f"CAP_GATE_{unlocked_gate.upper()}", rewards)
            self.log_event(f"★ Unlocked Gate: {unlocked_gate}!")

        if not self.world.in_bounds_passable(nx, ny, self.world.door_unlocked):
            rewards.on_wall_hit()
            return

        # Water & Sea speed penalty
        cur_tile = self.world._tile(self.x, self.y)
        new_tile = self.world._tile(nx, ny)
        in_water = (cur_tile and cur_tile.tile_type in (5, 8)) or (new_tile and new_tile.tile_type in (5, 8))

        if in_water:
            is_deep_sea = (cur_tile and cur_tile.tile_type == 8 and self.sea_blocks_crossed >= 3) or (new_tile and new_tile.tile_type == 8 and self.sea_blocks_crossed >= 3)
            needed_subticks = 3 if is_deep_sea else 2
            self._water_move_counter += 1
            if self._water_move_counter < needed_subticks:
                drag_name = "Deep Sea (1/3 speed)" if is_deep_sea else "Water/H2O (1/2 speed)"
                self.log_event(f"Wading in {drag_name}...")
                return
            self._water_move_counter = 0
            if new_tile and new_tile.tile_type == 8:
                self.sea_blocks_crossed += 1
            elif new_tile and new_tile.tile_type != 5:
                self.sea_blocks_crossed = 0
        else:
            self._water_move_counter = 0
            self.sea_blocks_crossed = 0

        if self.sprint and self.stamina > 0 and not self.leg_injured:
            self.stamina = max(0, self.stamina - 2)

        # Fall damage calculation
        old_z = self.z
        new_z = new_tile.z if new_tile else 0
        z_fall = old_z - new_z  # positive = falling down

        self.x, self.y = nx, ny
        self.z = new_z

        if z_fall > 2.0:
            self._apply_fall_damage(z_fall, rewards)

        # Ice slip check
        if new_tile and new_tile.tile_type == 10:  # TILE_ICE
            if random.random() < 0.20 and not self.leg_injured:
                self.leg_injured = 1
                self.sprint = False
                rewards.on_leg_injury()
                self.log_event("Slipped on Ice! (Leg Injured)")

    # ------------------------------------------------------------------
    # Use item logic
    # ------------------------------------------------------------------
    def _use_item(self, rewards):
        item_id = self.inventory[self.selected_slot]["id"]
        item = REGISTRY.get_item(item_id)
        if not item:
            return

        if item.get("type") == "consumable":
            consumed = self._consume_from_slot(self.selected_slot)
            if consumed:
                if item.get("heal"):
                    self.health = min(100, self.health + int(item["heal"]))
                    self.hunger = min(100, self.hunger + 15)
                    self.log_event(f"Consumed {item_name(consumed)} (+{item.get('heal')} HP)")
                if item.get("cure_leg"):
                    self.leg_injured = 0
                    self.log_event(f"Treated leg with {item_name(consumed)} (Injury Cured)")
                if item.get("cure_poisoned"):
                    self.status["poisoned"] = 0
                    self.log_event("Poison cured!")
                if item.get("cure_burning"):
                    self.status["burning"] = 0
                    self.log_event("Extinguished flames!")
                if item.get("cure_blinded"):
                    self.status["blinded"] = 0
                    self.log_event("Vision restored!")
        elif item.get("torch"):
            self.torch_active = True
            self.log_event(f"Lit Torch (Perception Radius increased)")

    # ------------------------------------------------------------------
    # Crafting integration
    # ------------------------------------------------------------------
    def _add_to_crafting(self, rewards):
        """Add item from selected slot into the crafting queue."""
        if not self.crafting.open:
            return
        item_id = self.inventory[self.selected_slot]["id"]
        if not item_id:
            return

        result = self.crafting.add_material(item_id)
        if result is not None:
            # Recipe matched — consume all materials from inventory
            for mat_id in self.crafting._last_consumed:
                self._consume_material(mat_id)
            # Add result to inventory
            self._add_to_inventory(result)
            rewards.on_craft(result)
            self.log_event(f"Crafted {item_name(result)}!")

            # Minecraft-style emergent capability unlock
            from crafting import RECIPE_CAPABILITIES
            cap = RECIPE_CAPABILITIES.get(result)
            if cap:
                self.unlock_capability(cap, rewards)
        else:
            self.log_event(f"Added {item_name(item_id)} to crafting bench")

    def _consume_material(self, item_id):
        """Remove one unit of *item_id* from inventory."""
        for slot in self.inventory:
            if slot["id"] == item_id and slot["count"] > 0:
                slot["count"] -= 1
                if slot["count"] <= 0:
                    slot["id"] = ""
                    slot["count"] = 0
                return

    # ------------------------------------------------------------------
    # Fall damage
    # ------------------------------------------------------------------
    def _apply_fall_damage(self, z_fall, rewards):
        """Apply fall damage based on height difference."""
        sprint_landing = self.sprint

        if z_fall <= 2.0:
            return
        elif z_fall <= 3.0:
            dmg = 5
        elif z_fall <= 5.0:
            dmg = 15
        else:
            dmg = 30

        if sprint_landing:
            dmg = int(dmg * 1.5)

        self.health -= dmg
        rewards.on_damage(dmg)
        self.log_event(f"Took {dmg} Fall Damage!")

        # Leg injury
        injury_threshold = 2.5 if sprint_landing else 5.1
        if z_fall >= injury_threshold:
            self.leg_injured = 1
            self.sprint = False
            rewards.on_leg_injury()
            self.log_event("Suffered Leg Injury! (Movement slowed)")

    # ------------------------------------------------------------------
    # Tick stats
    # ------------------------------------------------------------------
    def tick_stats(self, rewards):
        self._hunger_counter += 1
        # Hunger drains 1 point every 15 ticks
        if self.hunger > 0 and self._hunger_counter % 15 == 0:
            self.hunger -= 1

        # Starvation: health drains 1 HP every 30 ticks (~1 second)
        if self.hunger <= 0 and self._hunger_counter % 30 == 0:
            self.health = max(0, self.health - 1)
            rewards.add(-0.5)

        # Safe Haven natural recovery bonus
        if self.world.is_safe_zone(self.x, self.y) and self.health < 100:
            if self._hunger_counter % 10 == 0:
                self.health = min(100, self.health + 1)

        # Tile environmental hazards
        cur_t = self.world._tile(self.x, self.y)
        if cur_t:
            if cur_t.tile_type == 6:  # TILE_LAVA
                self.health = max(0, self.health - config_rl.LAVA_DAMAGE_PER_TICK)
                self.status["burning"] = max(self.status["burning"], 5)
                rewards.on_damage(config_rl.LAVA_DAMAGE_PER_TICK)
            elif cur_t.tile_type == 9:  # TILE_ACID
                self.status["poisoned"] = max(self.status["poisoned"], config_rl.ACID_POISON_TICKS)
            elif cur_t.tile_type == 12: # TILE_SPORE
                self.status["blinded"] = max(self.status["blinded"], 8)

            if cur_t.tile_type == 8:    # TILE_OCEAN
                self.ocean_ticks += 1
            else:
                self.ocean_ticks = 0

        # Status effect ticks
        if self.status["poisoned"] > 0:
            self.health -= 1
            self.status["poisoned"] -= 1
        if self.status["burning"] > 0:
            self.health -= 2
            self.status["burning"] -= 1
        if self.status["slowed"] > 0:
            self.status["slowed"] -= 1
        if self.status["blinded"] > 0:
            self.status["blinded"] -= 1

        # Stamina regen
        if not self.sprint and self.stamina < 100:
            self.stamina += 1

    # ------------------------------------------------------------------
    # State vector (exact 130 dimensions)
    # ------------------------------------------------------------------
    def build_state(self, world, day_night, memory_features, action_mask=None):
        if action_mask is None:
            action_mask = self.get_action_mask(world, day_night)

        ent, ent_dist = world.nearest_entity(self.x, self.y)
        obj_id, obj_dist = world.nearest_object(self.x, self.y)
        z = world.get_adjacent_z(self.x, self.y)
        incoming = 0
        ent_state = 0
        ent_type = 0
        if ent:
            ent_type = int(ent.entity_id.replace("E", "")) if ent.entity_id.startswith("E") and ent.entity_id[1:].isdigit() else 0
            ent_state = ent.state
            if ent.state == 2:
                incoming = int(ent.data.get("damage", 0))

        cur_tile = world._tile(self.x, self.y)
        cur_island = cur_tile.island_id if cur_tile else -1

        state = [
            # Vitals (5)
            self._norm(self.health, 100),
            self._norm(self.hunger, 100),
            self._norm(self.stamina, 100),
            self._norm(self.xp, max(self._xp_for_next_level, 1)),
            float(self.level),

            # Position / height (5)
            float(self.z),
            float(z["ahead"]),
            float(z["left"]),
            float(z["right"]),
            float(z["behind"]),

            # Adjacent tile types (4)
            float(world.tile_type_at(self.x, self.y - 1)),
            float(world.tile_type_at(self.x - 1, self.y)),
            float(world.tile_type_at(self.x + 1, self.y)),
            float(world.tile_type_at(self.x, self.y + 1)),

            # Nearest hostile (4)
            self._norm(ent_dist, day_night.perception_radius(self.torch_active)),
            float(ent_type),
            float(ent_state),
            float(incoming),

            # Nearest object (2): dist normalized, obj_id normalized to same scale as inventory
            self._norm(obj_dist, day_night.perception_radius(self.torch_active)),
            (int(obj_id) / config_rl.ITEM_VOCAB_SIZE) if obj_id.isdigit() and int(obj_id) > 0 else 0.0,
        ]

        # Inventory: 7 slots × (norm_item_id, norm_count) = 14 values
        # item_id: 0.0 = empty, else normalized to (id / ITEM_VOCAB_SIZE) so model sees ~0.01..0.99
        # count:   normalized to count / 99, capped at 1.0
        for slot in self.inventory:
            raw_id = int(slot["id"]) if slot["id"] and slot["id"].isdigit() else 0
            norm_id = raw_id / config_rl.ITEM_VOCAB_SIZE if raw_id > 0 else 0.0
            norm_cnt = min(1.0, slot["count"] / 99.0)
            state.append(norm_id)
            state.append(norm_cnt)

        # Status effects (8)
        state.extend([
            float(self.status["poisoned"]),
            float(self.status["burning"]),
            float(self.status["slowed"]),
            float(self.status["blinded"]),
            float(self.status["speed_boost"]),
            float(self.leg_injured),
            float(self.status["cold"]),
            float(self.status["acid_burn"]),
        ])

        # Environment (9)
        state.extend([
            day_night.time_of_day,
            float(self.torch_active),
            float(self.crafting.open),
            float(self.inventory_open),
            float(cur_island),
            float(self.last_failed_action),
            float(self.facing[0]),
            float(self.facing[1]),
            float(self.ocean_ticks),
        ])

        # Memory features (3)
        state.extend(memory_features)

        # Action mask (42)
        state.extend(float(v) for v in action_mask)

        # Pad/truncate to exact STATE_SIZE
        while len(state) < config_rl.STATE_SIZE:
            state.append(0.0)
        return state[:config_rl.STATE_SIZE]
