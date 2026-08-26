# ATS (AI Training System) — Documentation Hub

Welcome to the comprehensive documentation for **ATS (AI Training System)**.

ATS is a 2D tile-world reinforcement learning simulator and embodied cognitive architecture featuring a retro CRT debug interface, dynamic procedural open-world generation, an online PPO learning loop, and a symbolic translation bridge linking embodied state vectors to natural language models (such as Virus).

---

## 📚 Documentation Table of Contents

| Document | Description |
| :--- | :--- |
| **[How to Use](file:///c:/Users/Bob/Documents/github/ATS/docs/how_to_use.md)** | Setup instructions, CLI commands, CRT controls, interactive translator, and training guide |
| **[System Overview](file:///c:/Users/Bob/Documents/github/ATS/docs/system_overview.md)** | Core architecture, Cognitive Mind cycle, Actor-Critic network, and PPO continual learning |
| **[Walkthrough & Overhaul](file:///c:/Users/Bob/Documents/github/ATS/docs/walkthrough.md)** | Summary of recent system upgrades, UI font integration, open world overhaul, and testing logs |
| **[Symbolic Dictionary Guide](file:///c:/Users/Bob/Documents/github/ATS/docs/symbolic_dictionary_guide.md)** | Value-to-Word translation bridge, token structure, sanitization rules, and Virus language model coupling |
| **[World, Items & Entities](file:///c:/Users/Bob/Documents/github/ATS/docs/world_and_items.md)** | Complete catalog of biomes, harvestable world objects, inventory items, entities/wildlife, and crafting recipes |

---

## 🚀 Fast Quickstart

```powershell
# 1. Run simulation with CRT Terminal GUI
py -3.12 main.py --episodes 2

# 2. Run high-speed headless training
py -3.12 main.py --no-gui --episodes 5

# 3. Launch interactive natural language translator CLI
py -3.12 ats_virus_translator.py --interactive
```
