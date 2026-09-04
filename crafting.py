"""Crafting recipes, menu state, and inventory integration.

Full island-specific crafting chains with capability unlocking.
Each island has its own 3-4 tier crafting tree.
"""

# Progression Recipes — key: frozenset of ingredient IDs, value: result ID
RECIPES = {
    # -----------------------------------------------------------------------
    # Grassland Island (Tier 0) — Base crafting
    # -----------------------------------------------------------------------
    frozenset(["0010", "0010"]): "0008",   # Wood + Wood     -> Stick
    frozenset(["0010", "0008"]): "0034",   # Wood + Stick    -> Wooden Club
    frozenset(["0010", "0009"]): "0011",   # Wood + Coal     -> Torch
    frozenset(["0008", "0012"]): "0013",   # Stick + Stone   -> Stone Sword
    frozenset(["0004", "0004"]): "0005",   # Wool + Wool     -> Bandage
    frozenset(["0010", "0004"]): "0014",   # Wood + Wool     -> Bed
    frozenset(["0010", "0012"]): "0016",   # Wood + Stone    -> Shield
    frozenset(["0005", "0001"]): "0015",   # Bandage + Apple -> Bandage Adv

    # -----------------------------------------------------------------------
    # Dark Forest Island (Tier 1)
    # -----------------------------------------------------------------------
    frozenset(["0036", "0036"]): "0037",   # Wolf Pelt x2   -> Fur Cloak
    frozenset(["0038", "0008"]): "0039",   # Pine Resin + Stick -> Sticky Trap
    frozenset(["0040", "0001"]): "0041",   # Nightshade + Apple -> Night Potion
    frozenset(["0042", "0010"]): "0043",   # Spider Silk + Wood -> Silk Rope
    frozenset(["0013", "0042"]): "0002",   # Stone Sword + Silk -> upgrade (sharpened)

    # -----------------------------------------------------------------------
    # Volcanic Island (Tier 2)
    # -----------------------------------------------------------------------
    frozenset(["0046", "0008"]): "0047",   # Magma Stone + Stick  -> Magma Sword
    frozenset(["0048", "0016"]): "0049",   # Obsidian + Shield    -> Obsidian Shield
    frozenset(["0050", "0009"]): "0051",   # Fire Blossom + Coal  -> Fire Bomb
    frozenset(["0052", "0005"]): "0053",   # Volcanic Ash + Bandage -> Ash Balm
    frozenset(["0055", "0047"]): "0056",   # Void Shard + Magma Sword -> Void Blade

    # -----------------------------------------------------------------------
    # Mushroom Island (Tier 2)
    # -----------------------------------------------------------------------
    frozenset(["0057", "0001"]): "0058",   # Spore + Apple        -> Spore Bomb
    frozenset(["0059", "0004"]): "0060",   # Mycelium Cap + Wool  -> Mycelium Armor
    frozenset(["0061", "0001"]): "0062",   # Antidote Root + Apple -> Antidote
    frozenset(["0063", "0011"]): "0064",   # Glowshroom + Torch   -> Acid Flask
    frozenset(["0057", "0052"]): "0063",   # Spore + Ash          -> Glowshroom (emergency)

    # -----------------------------------------------------------------------
    # Tundra Island (Tier 3)
    # -----------------------------------------------------------------------
    frozenset(["0067", "0008"]): "0068",   # Ice Crystal + Stick -> Ice Arrow
    frozenset(["0070", "0013"]): "0071",   # Frost Ore + Stone Sword -> Frost Blade
    frozenset(["0072", "0037"]): "0073",   # Yeti Fur + Fur Cloak -> Yeti Coat

    # -----------------------------------------------------------------------
    # Boss Sanctum (Tier 4)
    # -----------------------------------------------------------------------
    frozenset(["0075", "0056"]): "0076",   # Void Core + Void Blade -> Rune Sword
    frozenset(["0075", "0049"]): "0078",   # Void Core + Obsidian Shield -> Rune Shield
    frozenset(["0075", "0077"]): "0077",   # Void Core + (anything) -> Void Armor placeholder
}

# Emergent capability mapping associated with synthesized items
RECIPE_CAPABILITIES = {
    # Grassland
    "0008": "CAP_WOODCRAFT",
    "0034": "CAP_BASIC_WEAPON",
    "0011": "CAP_TORCH_LIGHT",
    "0013": "CAP_STONE_TOOLS",
    "0005": "CAP_MEDICINE",
    "0014": "CAP_SHELTER",
    "0016": "CAP_DEFENSE",
    "0015": "CAP_ADV_MEDICINE",
    # Dark Forest
    "0037": "CAP_COLD_RESIST",
    "0039": "CAP_TRAPPER",
    "0041": "CAP_NIGHT_VISION",
    "0043": "CAP_ROPE_CRAFT",
    # Volcanic
    "0047": "CAP_FLAME_WEAPON",
    "0049": "CAP_FIRE_DEFENSE",
    "0051": "CAP_FIRE_CRAFT",
    "0053": "CAP_BURN_CURE",
    "0056": "CAP_VOID_TOUCHED",
    # Mushroom
    "0058": "CAP_ALCHEMIST",
    "0060": "CAP_POISON_RESIST",
    "0062": "CAP_ANTIDOTE",
    "0064": "CAP_ACID_CRAFT",
    # Tundra
    "0068": "CAP_FROST_WEAPON",
    "0071": "CAP_FROST_BLADE",
    "0073": "CAP_COLD_MASTER",
    # Boss
    "0076": "CAP_RUNE_ARMED",
    "0078": "CAP_RUNE_SHIELDED",
}

CRAFTING_AUTO_CLOSE_TICKS = 20


class CraftingSystem:
    def __init__(self, auto_close_ticks=CRAFTING_AUTO_CLOSE_TICKS):
        self.auto_close_ticks   = auto_close_ticks
        self.open               = False
        self.ticks_open         = 0
        self.selected: list     = []
        self._last_consumed: list = []

    def open_menu(self):
        self.open        = True
        self.ticks_open  = 0
        self.selected    = []
        self._last_consumed = []

    def tick(self):
        if not self.open:
            return None
        self.ticks_open += 1
        if self.ticks_open >= self.auto_close_ticks:
            self.open           = False
            self.selected       = []
            self._last_consumed = []
        return None

    def add_material(self, item_id: str) -> "str | None":
        if not self.open:
            return None
        self.selected.append(item_id)
        key    = frozenset(self.selected)
        result = RECIPES.get(key)
        if result is not None:
            self._last_consumed = list(self.selected)
            self.selected  = []
            self.open      = False
            self.ticks_open = 0
            return result
        if len(self.selected) > 3:
            self.selected = []
        return None
