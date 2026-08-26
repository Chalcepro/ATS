# ATS World, Items & Entities Catalog

This document details all world biomes, natural harvestable objects, items, entities, and crafting recipes.

---

## 1. Biomes & Procedural Flora

| Biome Index | Name | Common Objects & Flora | Common Wildlife & Entities |
| :---: | :--- | :--- | :--- |
| `0` | **Forest** | Oak Trees, Berry Bushes, Fallen Logs, Sticks, Stones, Mushrooms, Flowers, Coal Ore | Weak Slime, Sheep, Spider, Zombie |
| `1` | **Desert** | Cacti, Sandstone Boulders, Stones, Dead Shrubs | Scorpion, Husk, Skeleton |
| `2` | **Volcanic** | Coal Ore, Obsidian Boulders, Coal, Stones, Void Shards, Lava Pools | Fire Imp, Lava Golem |
| `3` | **Plains** | Berry Bushes, Flowers, Tall Grass, Sticks, Stones, Apples | Sheep, Pig, Chicken, Villager, Weak Slime |
| `4` | **Cave** | Stalagmites, Coal Ore, Stones, Mushrooms, Void Shards | Bat, Cave Spider, Cave Bat Queen, Golem |
| `5` | **Tundra** | Pine Trees, Fallen Logs, Boulders, Stones, Sticks | Wolf, Ice Wolf, Yeti |
| `6` | **Swamp** | Swamp Trees, Mushrooms, Fallen Logs, Berry Bushes, Sticks, Murky Ponds | Swamp Toad, Witch, Weak Slime |

---

## 2. Items & Harvestables Catalog

| ID | Name | Type | Effect / Stat | Drop Yield |
| :---: | :--- | :--- | :--- | :--- |
| `0001` | Apple | Consumable | +20 HP, +15 Hunger | — |
| `0002` | Sword Basic | Weapon | Damage: 5 | — |
| `0003` | Book | Curiosity | +10 XP | — |
| `0004` | Wool | Material | Crafting | — |
| `0005` | Bandage | Consumable | Cures Leg Injury | — |
| `0006` | Key | Tool | Unlocks Chests | — |
| `0007` | Chest | Container | Storage | — |
| `0008` | Stick | Material | Crafting | — |
| `0009` | Coal | Material | Crafting / Fuel | — |
| `0010` | Wood | Material | Crafting | — |
| `0011` | Torch | Tool | Increases Perception Radius | — |
| `0012` | Stone | Material | Crafting | — |
| `0013` | Stone Sword | Weapon | Damage: 8 | — |
| `0014` | Bed | Structure | Sleep (+30 HP at Night) | — |
| `0015` | Bandage Adv | Consumable | Cures Leg Injury, +30 HP | — |
| `0016` | Shield | Armour | Damage Reduction: 2 | — |
| `0017` | Rope | Tool | Safe Descent | — |
| `0018` | Map Fragment | Curiosity | +15 XP | — |
| `0019` | Potion Heal | Consumable | +50 HP | — |
| `0020` | Void Shard | Special | +5 XP | — |
| `0021` | Oak Tree | World Object | Harvestable | Wood (`0010`) |
| `0022` | Pine Tree | World Object | Harvestable | Wood (`0010`) |
| `0023` | Berry Bush | World Object | Harvestable / Interactable | Apple (`0001`) |
| `0024` | Fallen Log | World Object | Harvestable | Stick (`0008`) |
| `0025` | Boulder | World Object | Harvestable | Stone (`0012`) |
| `0026` | Coal Ore | World Object | Harvestable | Coal (`0009`) |
| `0027` | Cactus | World Object | Harvestable | Stick (`0008`) |
| `0028` | Mushroom | Consumable | +10 HP | — |
| `0029` | Flower | Curiosity | +2 XP | — |
| `0030` | Stalagmite | World Object | Harvestable | Stone (`0012`) |
| `0031` | Meat Raw | Consumable | +15 HP | — |
| `0032` | Bread | Consumable | +25 HP | — |
| `0033` | Iron Dagger | Weapon | Damage: 6 | — |
| `0034` | Wooden Club | Weapon | Damage: 3 | — |
| `0035` | Water Flask | Consumable | +10 HP | — |

---

## 3. Crafting Recipes (`crafting.py`)

| Ingredients | Result | Result ID |
| :--- | :--- | :---: |
| **Wood (`0010`) + Wood (`0010`)** | Stick | `0008` |
| **Wood (`0010`) + Coal (`0009`)** | Torch | `0011` |
| **Stick (`0008`) + Stone (`0012`)** | Stone Sword | `0013` |
| **Wool (`0004`) + Wool (`0004`)** | Bandage | `0005` |
| **Wood (`0010`) + Wool (`0004`)** | Bed | `0014` |

---

## 4. Entity Catalog (`assets/entities.json`)

| ID | Name | Type | HP | Damage | Special Traits / Drops |
| :---: | :--- | :--- | :---: | :---: | :--- |
| `E001` | Weak Slime | Hostile | 1 | 1 | Drops Apple (`0001`) |
| `E002` | Zombie | Hostile | 10 | 3 | Drops Wood (`0010`) |
| `E003` | Skeleton | Hostile | 8 | 4 | Ranged attack, Drops Stick (`0008`) |
| `E004` | Creeper | Hostile | 6 | 15 | Explodes (AoE Radius 2) |
| `E005` | Spider | Hostile | 7 | 3 | Traverses obstacles, Drops Wool (`0004`) |
| `E006` | Husk | Hostile | 10 | 3 | Inflicts Poison, Drops Stone (`0012`) |
| `E007` | Fire Imp | Hostile | 5 | 2 | Inflicts Burn, Drops Coal (`0009`) |
| `E008` | Cave Spider | Hostile | 5 | 4 | Inflicts Poison, Drops Wool (`0004`) |
| `E009` | Golem | Hostile | 25 | 8 | Boss tier, Drops Stone x3 (`0012`) |
| `E010` | Witch | Hostile | 12 | 5 | Ranged, Drops Potion Heal (`0019`) |
| `E011` | Wolf | Hostile | 8 | 4 | Pack hunter, Drops Wool (`0004`) |
| `E012` | Yeti | Hostile | 20 | 6 | Inflicts Slow, Drops Shield (`0016`) |
| `E013` | Bat | Passive | 2 | 0 | Disturbs sight in dark |
| `E014` | Void | Special | $\infty$ | 999 | Proximity instakill |
| `E015` | Sheep | Passive | 5 | 0 | Flees, Drops Wool (`0004`) |
| `E016` | Pig | Passive | 5 | 0 | Flees, Drops Apple/Meat (`0001`) |
| `E017` | Chicken | Passive | 3 | 0 | Flees fast, Drops Apple/Food (`0001`) |
| `E018` | Villager | Neutral | 10 | 0 | Can trade |
| `E019` | Guardian NPC | Neutral | 30 | 6 | Patrols & attacks hostiles |
| `E020` | Void Echo | Special | 1 | 5 | Night only, inflicts blindness |
| `E021` | Scorpion | Hostile | 8 | 3 | Inflicts Poison, Drops Stone (`0012`) |
| `E022` | Ice Wolf | Hostile | 10 | 5 | Pack hunter, Inflicts Slow |
| `E023` | Lava Golem | Hostile | 30 | 10 | Inflicts Burn, Drops Coal (`0009`) |
| `E024` | Swamp Toad | Hostile | 6 | 2 | Inflicts Poison |
| `E025` | Cave Bat Queen | Hostile | 15 | 3 | Drops Bandage (`0005`) |
