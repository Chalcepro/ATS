"""Need detector — maps a raw state vector to an active-problem vector.

The need vector is pure floats.  The mind never sees English labels.
Each element is 0.0 (no problem) to 1.0 (urgent problem).

Current needs tracked:
    0  hunger   — rises as hunger_norm drops below threshold
    1  injury   — 1.0 when leg is injured
    2  threat   — scales with hostile proximity and attack state
    3  tool     — 1.0 when no weapon is present in inventory
"""

from __future__ import annotations

import config_rl

# Indices into the state vector (must stay in sync with agent.build_state)
_IDX_HUNGER = 1
_IDX_LEG_INJURED = -1  # resolved dynamically; see _find_leg_idx()

# Inventory block: starts after vitals(5)+position(5)+tiles(4)+hostile(4)+object(2) = 20
_INV_START = 20
_INV_SLOTS = config_rl.INVENTORY_SLOTS  # 24
_INV_FIELDS = 2  # (item_id, count) per slot

# Number of needs the detector outputs
NUM_NEEDS = 4

# Hunger threshold — below this, hunger_need starts rising
_HUNGER_WARN = 0.30


class NeedDetector:
    """Stateless mapper:  state → need vector."""

    def detect(self, state: list[float]) -> list[float]:
        """Return ``[hunger, injury, threat, tool]`` in range 0..1."""

        hunger_norm = state[_IDX_HUNGER] if len(state) > _IDX_HUNGER else 1.0
        hunger_need = max(0.0, (1.0 - hunger_norm / _HUNGER_WARN)) if hunger_norm < _HUNGER_WARN else 0.0

        # Injury — leg_injured flag is in status effect block
        # Status effects start after inventory block
        status_start = _INV_START + _INV_SLOTS * _INV_FIELDS  # 20 + 48 = 68
        # Status order: poisoned(68), burning(69), slowed(70), blinded(71), speed_boost(72), leg_injured(73)
        leg_idx = status_start + 5  # index 73
        injury_need = float(state[leg_idx]) if len(state) > leg_idx else 0.0

        # Threat — from nearest hostile state (index 16 = nearest_hostile_state)
        hostile_state_idx = 16
        hostile_dist_idx = 14
        if len(state) > hostile_state_idx:
            hostile_state = state[hostile_state_idx]
            hostile_dist = state[hostile_dist_idx]
            # Higher state (telegraph=1, attacking=2) and closer distance = higher threat
            threat_need = min(1.0, hostile_state * 0.5 + (1.0 - hostile_dist) * 0.3)
        else:
            threat_need = 0.0

        # Tool need — scan inventory for any weapon
        has_weapon = False
        for slot in range(_INV_SLOTS):
            item_id_idx = _INV_START + slot * _INV_FIELDS
            if len(state) > item_id_idx:
                item_id = int(state[item_id_idx])
                # Weapons: 0002 (Sword), 0013 (Stone Sword)
                if item_id in (2, 13):
                    has_weapon = True
                    break
        tool_need = 0.0 if has_weapon else 1.0

        return [hunger_need, injury_need, threat_need, tool_need]
