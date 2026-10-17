# ATS — Agent Training System

> **Status:** Active prototype · v5.3 · August 2026  
> **Author:** Calcedony Sokomba Arial · Independent AI Research · Abuja, Nigeria  
> **Hardware:** CPU-only (Intel i5, 8GB RAM) · No GPU required

---

## What Is ATS?

ATS is a reinforcement learning environment and cognitive architecture built from scratch. An AI agent lives inside a procedurally generated 2D tile world and learns through interaction — navigating terrain, managing survival stats, collecting resources, crafting items, and responding to threats.

The system is trained using **Proximal Policy Optimization (PPO)** with online (continual) updates, running entirely on consumer laptop hardware.

---

## The Research Question

> *Can an agent that learns behavioral policies in a structured, abstract environment develop reusable representations that reduce learning time on related downstream tasks?*

This is the central hypothesis. It is currently **undemonstrated** — the infrastructure to test it is built and running. Transfer learning experiments are planned for Phase 4.

---

## System Overview

```
Environment (2D tile world)
  └─ 7 biomes · 25 entity types · crafting · day/night · survival mechanics

Agent
  └─ 126-dimensional state vector (vitals, inventory, terrain, entities)
  └─ 42 discrete actions with dynamic action masking
  └─ 2-layer MLP policy: 126 → 256 → 256 → {actor(42), critic(1)}
  └─ ~114,000 parameters

Training
  └─ PPO with continual mini-batch updates every 64 ticks
  └─ Expandable hidden layers (256 → 1024 on plateau detection)
  └─ Episode logging to CSV + TensorBoard

Mind Architecture (cognitive overlay)
  └─ Need Detector — threshold-based survival monitoring
  └─ Experience Memory — fingerprint-based recall of successful action sequences
  └─ Solution Loop — detect → recall → act → record

External Integration
  └─ Virus Adapter — exports episode corpus for character-level language model training
  └─ Symbolic Dictionary — maps item/entity IDs to natural language tokens
```

---

## Reward Function (Selected Constants)

| Event | Reward |
|---|---|
| First pickup of new item type | +1.5 |
| Kill enemy | +2.0 |
| Discover new biome | +2.0 |
| Survive full episode | +10.0 |
| Wait / idle | −0.2/tick |
| Take damage | −1.0/hit |
| Death | −10.0 |

---

## Current Status

| Component | Status |
|---|---|
| Procedural world (7 biomes, 25 entities, crafting) | ✅ Complete |
| PPO policy with action masking | ✅ Complete |
| Mission Control GUI (live graphs, episode log) | ✅ Complete |
| Mind architecture (need detector, experience memory) | ✅ Functional |
| Continual learning (online PPO updates) | ✅ Complete |
| Episode CSV logging + TensorBoard | ✅ Complete |
| Baseline comparison experiments | ⏳ Planned — Phase 4 |
| Transfer learning validation | ⏳ Planned — Phase 4 |
| Bidirectional Virus integration | ⏳ Planned — Phase 5 |

---

## Training Observations

Over 100 training sessions completed to date. In monitored short-horizon runs (100 ticks):
- All episodes terminated at max ticks — agent does not immediately collapse
- Reward per episode: approximately +16 to +48 range observed
- Best single-episode reward: **48.55**
- Agent responds to reward signal changes mid-run (adjusts movement when directions are penalized)
- Hunger death remains possible and not yet consistently avoided

**No statistically significant learning trend has been established** at current sample sizes. Longer controlled runs (500+ episodes) are the next step.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 — Core RL Loop | ✅ Complete | PPO, action space, reward engine |
| Phase 2 — World System | ✅ Complete | Procedural generation, biomes, entities, crafting |
| Phase 3 — Mind Architecture | 🔄 In Progress | Need detector, experience memory, GUI |
| Phase 4 — Validation | 📋 Planned | Baselines, transfer learning, benchmarks |
| Phase 5 — Concept Grounding | 📋 Planned | Virus integration deepened |
| Phase 6 — Specialization | 📋 Planned | Domain-specific variants |

---

## Requirements

```
Python 3.12
PyTorch (CPU build)
Pygame
NumPy
TensorBoard (optional, for loss visualization)
```

No GPU required. Designed to run on modest consumer hardware.

---

## What ATS Is and Is Not

| ATS **is** | ATS **is not** |
|---|---|
| A working RL environment + training loop | A novel AI architecture |
| A PPO agent with a standard MLP policy | A demonstration of AI understanding |
| A functional cognitive overlay (heuristic-based) | A learned cognitive architecture |
| CPU-trainable on modest hardware | A benchmarked or sample-efficient system (untested) |
| An active research prototype | A finished or validated research contribution |

---

## Related Work

- **MiniGrid** (Chevalier-Boisvert et al., 2018) — similar grid-world approach; ATS has more complex mechanics
- **Crafter** (Hafner, 2021) — closest 2D open-world equivalent; requires pixel input and GPU
- **NetHack Learning Environment** (Küttler et al., 2020) — tile-based procedural; ATS is lighter and custom-built
- **Procgen** (Cobbe et al., 2019) — procedural generation for generalization; visual input

---

## Documentation

- [Technical Research Dossier (full)](docs/) — architecture, training observations, hypothesis, roadmap
- [Executive Brief](docs/) — 1-page overview for sponsors and collaborators
- [Research Proposal](docs/) — experimental plan, methodology, timeline

---

## Contact

**Calcedony Sokomba Arial**  
Independent AI Research · Abuja, Nigeria  
Open to research feedback, collaboration, and compute access discussions.

---

*ATS is documented honestly. Hypotheses are labeled as hypotheses. Observations are distinguished from conclusions. The work is real and ongoing.*
