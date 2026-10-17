# ATS User Guide — Unified Control Center & Simulator

This guide explains how to launch, configure, monitor, and train the agent using the **ATS Unified Mission Control Dashboard**.

---

## 1. Fast Quickstart

Launch the unified graphical dashboard with one command:

```powershell
py -3.12 main.py
```

### 📟 Retro 80s Layered Wireframe Boot Sequence:
1. **CRT Raster Power-On & Scanline Sweep**: Phosphor beam sweeps across the screen.
2. **Layered Wireframe Logo Build**: Constructs the stylized apex `A` from [logo/logo_Style.txt](file:///c:/Users/Bob/Documents/github/ATS/logo/logo_Style.txt) line-by-line.
3. **Dual Strobe Identity Reveal**: Upper notch illuminates `TS` (Autonomous Training System) while lower notch illuminates `GENT` (Agent Training System).
4. **Hardware & Engine Diagnostics**: Live diagnostics check for neural matrix, 7 biomes, and memory banks with kernel progress meter.
*(Press `SPACE`, `ENTER`, or mouse click at any point to jump straight into Mission Control)*

You can now control everything directly inside the app using mouse clicks and keyboard hotkeys without reopening the terminal!

---

## 2. Navigation & Views (Collapsible Sidebar Drawer)

Click the **`[ ◀ TABS / VIEWS ]`** button at the top-right of the screen to open the slide-out navigation drawer:

| View | Key | Description |
| :--- | :---: | :--- |
| **`1: 🎮 LIVE SIM`** | `1` | Real-time CRT survival viewport (7×7 world grid with facing glyphs `[AI^]`, 24-slot inventory, target indicator, vitals, event stream, and live action/rule toggles). |
| **`2: 📊 ANALYTICS & GRAPHS`** | `2` | **Zoomable Timeframe Graphs** (`10 Ep`, `25 Ep`, `50 Ep`, `ALL`, `Zoom +/-`), Loss Curves, and **Ascending Episode History Table** (oldest at top, latest at bottom) with mouse wheel scrolling and CSV export. |
| **`3: ⚙️ HYPERPARAMS`** | `3` | Real-time hyperparameter studio with interactive stepper buttons `[-]` `[+]` for **Target Episodes**, **Max Ticks**, **Simulation Speed**, **Learning Rate**, **Entropy**, and **Day Length**. |
| **`4: 🧠 MIND & LLM`** | `4` | Cognitive architecture inspector displaying live 4-D Need meters (`Hunger`, `Injury`, `Threat`, `Tool`), SolutionLoop traces, and the Virus Language Model narration stream. |

---

## 3. Solid Action Buttons & Controls

Located at the bottom of the interface with vibrant, high-contrast solid backgrounds:

| Control | Color | Hotkey | Function |
| :--- | :---: | :---: | :--- |
| **`⏸ PAUSE / ▶ RESUME`** | Solid Amber | `SPACE` | Freezes or resumes simulation execution at the current tick. |
| **`⚡ TURBO`** | Solid Blue | `T` | Toggles ultra-high-speed background training (skips visual rendering delay while keeping graphs live). |
| **`💾 SAVE MODEL`** | Solid Green | `Ctrl + S` | Immediately saves weights to `checkpoints/model_rl.pt` and `model_best.pt`. |
| **`🔄 RESET POLICY`** | Solid Wine | `—` | Re-initializes policy to fresh random weights without restarting the app. |
| **`⏹ STOP RUN`** | Solid Red | `ESC` | Gracefully stops the current episode run and saves weights. |
| **`🏠 MENU`** | Solid Steel | `—` | Returns to the Pre-Launch Main Menu. |

---

## 4. Telemetry & CSV Export

All training runs are automatically logged to **`data/ats_episode_history.csv`** on every episode completion. You can also click the **`[ 📄 EXPORT CSV ]`** button on the Analytics tab to trigger an immediate export.

---

## 4. Headless & CLI Modes

If you need headless training for scripts or automated benchmarking:

```powershell
# Headless batch training
py -3.12 main.py --no-gui --episodes 100

# Specify custom ticks & fresh start
py -3.12 main.py --no-gui --episodes 50 --ticks 1000 --fresh

# Interactive Symbolic Translator CLI
py -3.12 ats_virus_translator.py --interactive
```
