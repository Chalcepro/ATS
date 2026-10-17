# Symbolic Dictionary & Translation Guide

The **Symbolic Dictionary** ([data/symbolic_dictionary.json](file:///c:/Users/Bob/Documents/github/ATS/data/symbolic_dictionary.json)) provides a structured bidirectional mapping between numeric/symbolic codes and natural language concepts.

---

## 1. Why Symbolic Tokens?

State-based reinforcement learning models operate over discrete numeric indices, vectors, and embeddings, while human observers and language models (such as Virus) operate over natural language words and semantic sentences.

The Symbolic Dictionary bridges this gap:
1. **Embodied Agent** $\longleftrightarrow$ Uses token codes (`0001`, `0010`, `E001`, `B000`, `A007`).
2. **Human / GUI** $\longleftrightarrow$ Sees English names (`Apple`, `Wood`, `Weak Slime`, `Forest`, `Attack`).
3. **Virus Language Model** $\longleftrightarrow$ Receives formatted packets connecting token codes with semantic descriptions for future language grounding.

---

## 2. Canonical Token Organization

Tokens in the dictionary are grouped into distinct categories:

### A. Items & Resources (`0001` – `0035`)
- `0001: Apple`, `0002: Sword Basic`, `0003: Book`, `0008: Stick`, `0009: Coal`, `0010: Wood`, `0011: Torch`, `0012: Stone`, `0013: Stone Sword`, `0014: Bed`, `0019: Potion Heal`, `0020: Void Shard`
- **Harvestable World Objects**: `0021: Oak Tree`, `0022: Pine Tree`, `0023: Berry Bush`, `0024: Fallen Log`, `0025: Boulder`, `0026: Coal Ore`, `0027: Cactus`, `0028: Mushroom`, `0029: Flower`, `0030: Stalagmite`

### B. Entities & Wildlife (`E001` – `E025`)
- `E001: Weak Slime`, `E002: Zombie`, `E003: Skeleton`, `E004: Creeper`, `E005: Spider`, `E006: Husk`, `E007: Fire Imp`, `E008: Cave Spider`, `E009: Golem`, `E010: Witch`, `E011: Wolf`, `E012: Yeti`, `E013: Bat`, `E015: Sheep`, `E016: Pig`, `E017: Chicken`, `E018: Villager`, `E019: Guardian NPC`

### C. Biomes (`B000` – `B006`)
- `B000: Forest`, `B001: Desert`, `B002: Volcanic`, `B003: Plains`, `B004: Cave`, `B005: Tundra`, `B006: Swamp`

### D. Actions (`A000` – `A020`)
- `A000: Move Forward`, `A001: Move Backward`, `A002: Move Left`, `A003: Move Right`, `A004: Sprint On`, `A006: Jump`, `A007: Attack`, `A008: Pick Up`, `A009: Interact`, `A010: Use Item`, `A011: Open Inventory`, `A013: Open Crafting`, `A016: Sleep`, `A017: Wait`, `A018: Harvest Tree`, `A019: Harvest Bush`, `A020: Mine Rock`

### E. Needs & Sensations (`N001` – `N016`)
- `N001: Hunger`, `N002: Thirst`, `N003: Health`, `N004: Stamina`, `N005: Injury`, `N006: Threat`, `N007: Danger`, `N008: Safe`, `N009: Cold`, `N010: Warm`, `N011: Light`, `N012: Darkness`, `N013: Starvation`, `N015: Poisoned`, `N016: Burning`

### F. Spatial & Orientation (`D001` – `D013`)
- `D001: North`, `D002: South`, `D003: East`, `D004: West`, `D005: Ahead`, `D006: Behind`, `D007: Near`, `D008: Far`, `D009: Ground`, `D010: Inventory`, `D011: Facing`

### G. Communication & Feedback (`C001` – `C010`)
- `C001: Yes`, `C002: No`, `C003: Good`, `C004: Bad`, `C005: Stop`, `C006: Explore`, `C007: Survive`, `C008: Reward`, `C009: Performance`, `C010: Help`

---

## 3. Anti-Pollution & Sanitization

[ats_virus_translator.py](file:///c:/Users/Bob/Documents/github/ATS/ats_virus_translator.py) contains `_is_numeric_token()` filtering:
- Rejects pure integers, floats (`0.03`, `-1.20`, `+0.5`), episode numbers, and numeric fragments (`03`, `04`, `950`).
- Prevents runtime logs and performance evaluations from generating spurious hash tokens in the dictionary.
