"""Crafting recipes, menu state, and inventory integration.

When the agent opens the crafting menu (action OPEN_CRAFTING) and then
adds materials (action ADD_TO_CRAFTING), items are queued.  On a recipe
match the result is returned and _last_consumed holds the list of
ingredient IDs that must be removed from the agent's inventory.
"""

# V1 Recipes — key: frozenset of ingredient IDs, value: result ID
RECIPES = {
    frozenset(["0010", "0010"]): "0008",               # Wood + Wood     → Stick
    frozenset(["0010", "0009"]): "0011",               # Wood + Coal     → Torch
    frozenset(["0008", "0012"]): "0013",               # Stick + Stone   → Stone Sword
    frozenset(["0004", "0004"]): "0005",               # Wool + Wool     → Bandage
    frozenset(["0010", "0004"]): "0014",               # Wood + Wool     → Bed
}

# tunable: how long the crafting menu stays open if no recipe is matched
CRAFTING_AUTO_CLOSE_TICKS = 20  # tunable


class CraftingSystem:
    def __init__(self, auto_close_ticks=CRAFTING_AUTO_CLOSE_TICKS):
        self.auto_close_ticks = auto_close_ticks
        self.open = False
        self.ticks_open = 0
        self.selected: list[str] = []       # item IDs queued for crafting
        self._last_consumed: list[str] = [] # set when a recipe succeeds

    def open_menu(self):
        self.open = True
        self.ticks_open = 0
        self.selected = []
        self._last_consumed = []

    def tick(self):
        """Advance the auto-close timer.  Call once per sim tick."""
        if not self.open:
            return None
        self.ticks_open += 1
        if self.ticks_open >= self.auto_close_ticks:
            self.open = False
            self.selected = []
            self._last_consumed = []
        return None

    def add_material(self, item_id: str) -> str | None:
        """Add *item_id* to the crafting queue.

        Returns the result item ID if a recipe matches, else None.
        When a result is returned, *self._last_consumed* contains the
        list of ingredient IDs consumed so the caller can remove them
        from the agent's inventory.
        """
        if not self.open:
            return None

        self.selected.append(item_id)
        key = frozenset(self.selected)
        result = RECIPES.get(key)

        if result is not None:
            self._last_consumed = list(self.selected)
            self.selected = []
            self.open = False
            self.ticks_open = 0
            return result

        # Also check partial key match to prune obviously impossible inputs
        # (keeps the queue from growing indefinitely)
        if len(self.selected) > 3:
            self.selected = []  # reset queue if too long and no match

        return None
