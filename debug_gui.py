"""ATS Unified Mission Control & Live Dashboard — Modern Sci-Fi Terminal v2.5.

Features:
- Crisp, high-readability typography hierarchy (Consolas / Segoe UI with zero corrupted glyphs)
- Flashy Vector Cyber Emblem (A + TS / A + GENT) with animated pulse and zero header text collisions
- High-contrast sci-fi dark palette with glowing accents and clear visual hierarchy
- 1040x680 expanded viewport with comfortable padding and clean card panels
- Interactive pre-launch configuration with +/- steppers and preset selector pills
- Visual toggle switches for reward rules and action masks
- Solid, vibrant action buttons (Start, Pause, Turbo, Save, Reset, Stop, Menu)
- Top-Right Collapsible Navigation Sidebar Drawer ([ < TABS / VIEWS ])
- Zoomable Telemetry Graphs with dynamic Y-axis auto-scaling
- Ascending Episode History Table with mouse-wheel and button scrolling
- Automatic CSV export to data/ats_episode_history.csv with grade and efficiency tracking
"""

import csv
import math
import os
import time
import pygame
import config_rl
from item_ids import entity_name, item_name
from agent import ACTION_NAMES
from rewards import default_reward_rules

# ---------------------------------------------------------------------------
# High-Contrast Sci-Fi Terminal Palette
# ---------------------------------------------------------------------------
BG          = (10, 14, 16)        # Deep space dark background
BG_PANEL    = (16, 24, 28)        # Card panel background
BG_PANEL2   = (22, 32, 38)        # Elevated panel / Drawer
BG_CARD     = (18, 28, 32)        # Interactive card background

FG          = (50, 255, 140)      # Primary phosphor neon green
FG_BOLD     = (160, 255, 205)     # Bright highlight mint
DIM         = (70, 130, 105)      # Secondary / label text
DIM2        = (25, 55, 45)        # Muted grid lines / subtle borders
AMBER       = (255, 185, 50)      # Alert / warning amber
RED         = (255, 80,  75)      # Danger / stop red
CYAN        = (80,  225, 255)     # Telemetry cyan
FRAME       = (35,  75,  60)      # Panel bevel frame
FRAME_GLOW  = (55,  140, 105)     # Active card border glow
WHITE       = (240, 250, 255)     # Clean readable white
GOLD        = (255, 215, 80)      # Header gold / active highlights
TITLE_GREEN = (0,   235, 120)     # Brand logo green

# Solid Button Colors
BTN_LAUNCH_BG  = (25,  150, 70)
BTN_LAUNCH_BDR = (50,  240, 120)
BTN_STOP_BG    = (185, 30,  30)
BTN_STOP_BDR   = (255, 80,  80)
BTN_SAVE_BG    = (20,  140, 60)
BTN_SAVE_BDR   = (45,  220, 100)
BTN_TURBO_BG   = (20,  110, 180)
BTN_TURBO_BDR  = (65,  180, 255)
BTN_PAUSE_BG   = (180, 130, 20)
BTN_PAUSE_BDR  = (255, 200, 50)
BTN_RESET_BG   = (115, 35,  55)
BTN_RESET_BDR  = (190, 65,  95)
BTN_MENU_BG    = (35,  55,  48)
BTN_MENU_BDR   = (70,  115, 95)

BTN_IDLE_BG    = (20,  35,  30)
BTN_HOVER_BG   = (35,  70,  55)
BTN_ACTIVE_BG  = (45,  115, 80)
BTN_LOCKED_BG  = (16,  22,  20)
LOCK_FG        = (50,  75,  65)

WINDOW_WIDTH  = 1040
WINDOW_HEIGHT = 680
W, H = WINDOW_WIDTH, WINDOW_HEIGHT


# ---------------------------------------------------------------------------
# The curriculum, as the menu presents it
# ---------------------------------------------------------------------------
#
# School names because that is how the ladder was described, and because
# "nursery" says what the stage is for far better than "stage 1" does. The
# rungs underneath are the real Stage objects from curriculum.py.
#
# Locked entries are shown rather than hidden on purpose: a menu that grows new
# items as you progress hides the shape of the thing you are climbing.
TRACKS = [
    ("NURSERY",    ["nursery"],
     "One room, one goal, nothing that can hurt you", True),
    ("PRIMARY",    ["corridors", "corridors7", "avoid", "hazards", "foraging", "primary"],
     "Walls, then danger, then scarcity, then size", True),
    ("JUNIOR SEC", ["junior"],
     "Something in here with you - coming soon", False),
    ("SENIOR SEC", [],
     "Tools, crafting, the long game - coming soon", False),
    ("FULL WORLD", [],
     "The island. Everything at once", True),
]


def brain_progress():
    """What the saved brain says it has already passed. Never raises."""
    try:
        import torch
        path = config_rl.CHECKPOINT_DIR / "brain.pt"
        if not path.exists():
            return {}
        blob = torch.load(path, map_location="cpu", weights_only=False)
        prog = dict(blob.get("progress") or {})
        prog["ticks"] = blob.get("total_ticks", 0)
        return prog
    except Exception:
        return {}


def track_rungs(names):
    """The Stage objects behind a track name, in ladder order."""
    try:
        from curriculum import default_ladder
        by_name = {}
        for st in default_ladder():
            by_name.setdefault(st.name, st)
        return [by_name[n] for n in names if n in by_name]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# App States & Tabs
# ---------------------------------------------------------------------------
STATE_MENU    = "MENU"
STATE_RUNNING = "RUNNING"
STATE_PAUSED  = "PAUSED"
STATE_STOPPED = "STOPPED"

TAB_SIM       = 0
TAB_ANALYTICS = 1
TAB_CONFIG    = 2
TAB_MIND      = 3

TAB_ACCESS = {
    STATE_MENU:    {0, TAB_SIM, TAB_CONFIG, TAB_ANALYTICS},
    STATE_RUNNING: {TAB_SIM, TAB_ANALYTICS, TAB_MIND},
    STATE_PAUSED:  {TAB_SIM, TAB_ANALYTICS, TAB_CONFIG, TAB_MIND},
    STATE_STOPPED: {TAB_ANALYTICS, TAB_CONFIG, TAB_MIND},
}

_RULE_LABELS = [
    ("STILL",   "standing_still"),
    ("DMG",     "damage_penalty"),
    ("WAIT",    "wait_penalty"),
    ("HUNGER",  "hunger_penalty"),
    ("WALL",    "wall_hit"),
    ("DEATH",   "death_penalty"),
    ("PICKUP",  "pickup_reward"),
    ("KILL",    "kill_reward"),
    ("EXPLORE", "explore_reward"),
    ("PROG",    "progression"),
]

_ACTION_TOGGLES = [
    ("FWD",  config_rl.ACT_MOVE_FORWARD),
    ("BACK", config_rl.ACT_MOVE_BACKWARD),
    ("LEFT", config_rl.ACT_MOVE_LEFT),
    ("RIGHT",config_rl.ACT_MOVE_RIGHT),
    ("SPRNT",config_rl.ACT_SPRINT_ON),
    ("JUMP", config_rl.ACT_JUMP),
    ("ATK",  config_rl.ACT_ATTACK),
    ("PICK", config_rl.ACT_PICK_UP),
    ("INTR", config_rl.ACT_INTERACT),
    ("USE",  config_rl.ACT_USE_ITEM),
    ("INV",  config_rl.ACT_OPEN_INVENTORY),
    ("CRAFT",config_rl.ACT_OPEN_CRAFTING),
    ("SLEEP",config_rl.ACT_SLEEP),
    ("WAIT", config_rl.ACT_WAIT),
]

_CONFIG_FIELDS = [
    ("EPISODES",          "Target Episodes",    10,   1,    9999, "{:d}"),
    ("MAX_TICKS",         "Max Ticks / Ep",    500, 100,   50000, "{:d}"),
    ("SPEED_MULTIPLIER",  "Sim Speed",          1.0,  1.0,  50.0, "{:.1f}x"),
    ("LEARNING_RATE",     "Learning Rate",     None, None,   None, "{:.2e}"),
    ("ENTROPY_COEFF",     "Entropy Coeff",     None, None,   None, "{:.3f}"),
    ("DAY_LENGTH_TICKS",  "Day Length (ticks)",100,  200,   5000, "{:d}"),
    ("MIND_HIDDEN_SIZE",  "Hidden Width",      None, None,   None, "{:d}"),
    ("GAMMA",             "Discount Gamma",    None, None,   None, "{:.3f}"),
]

_LR_PRESETS      = [5e-5, 1e-4, 2e-4, 5e-4]
_ENTROPY_PRESETS = [0.002, 0.005, 0.010, 0.020]
_HIDDEN_PRESETS  = [128, 256, 512]


# ===========================================================================
class TerminalGUI:
# ===========================================================================

    def __init__(self, title="ATS // MISSION CONTROL"):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # Load crisp, highly readable monospaced UI fonts with graceful fallbacks
        self.f_title  = self._load_font(18, bold=True)
        self.f_header = self._load_font(14, bold=True)
        self.f_body   = self._load_font(12, bold=False)
        self.f_bold   = self._load_font(12, bold=True)
        self.f_small  = self._load_font(10, bold=False)
        self.f_tiny   = self._load_font(9,  bold=False)

        # Application state
        self.alive         = True
        self.frame         = 0
        self.app_state     = STATE_MENU
        # Which rung of the ladder the launch button will start.
        self.selected_track = 0
        # What the running stage actually permits. None means the full world,
        # where everything is live. A curriculum rung sets this to its action
        # mask, and anything it masks off is drawn locked rather than removed -
        # so the panel shows the whole instrument, greyed, instead of quietly
        # shrinking and leaving you to wonder what happened to the rest.
        self.active_mask = None
        self.active_stage = None
        self.active_tab    = 0
        self.sidebar_open  = False

        # Interactive Signals
        self.sig_start        = False
        self.sig_pause_toggle = False
        self.sig_save         = False
        self.sig_fresh_reset  = False
        self.sig_stop         = False
        self.sig_return_home  = False
        self.turbo_mode       = False
        self._confirm_fresh_armed = False
        self.is_fresh_mode    = False

        # Live toggles
        self.disabled_actions = set()
        self.reward_rules     = default_reward_rules()

        # Telemetry & History
        self.episode_history  = []
        self.reward_curve     = []
        self.loss_history     = []
        self.last_narration   = "No narration generated yet."

        # Analytics Zoom & Windowing
        self.zoom_window  = 25    # visible data points (0 = ALL)
        self.pan_offset   = 0     # 0 = latest
        self.table_scroll = 0     # scroll offset from bottom for ascending table

        # Click registry (re-populated every frame)
        self._clicks = []
        self._pulse  = 0.0

        # CSV Logging path
        self.csv_path = config_rl.DATA_DIR / "ats_episode_history.csv"

    def _load_font(self, size, bold=False):
        """Loads clean, high-readability UI fonts with anti-aliasing."""
        font_names = ["consolas", "lucidaconsole", "segoeui", "dejavusansmono", "couriernew", "arial"]
        for fn in font_names:
            try:
                f = pygame.font.SysFont(fn, size, bold=bold)
                if f:
                    return f
            except Exception:
                continue
        return pygame.font.Font(None, size + 2)

    # -----------------------------------------------------------------------
    # The mark
    # -----------------------------------------------------------------------
    def _logo_lines(self):
        """The ATS ASCII logo, read from logo/logo_Style.txt.

        Kept as a file rather than baked into this module so the logo can be
        redrawn without touching code - it is artwork, and artwork belongs
        somewhere a person can edit it.
        """
        if getattr(self, "_logo_cache", None) is None:
            try:
                raw = (config_rl.ROOT / "logo" / "logo_Style.txt").read_text(encoding="utf-8")
                self._logo_cache = [ln.rstrip() for ln in raw.splitlines()]
                while self._logo_cache and not self._logo_cache[0].strip():
                    self._logo_cache.pop(0)
                while self._logo_cache and not self._logo_cache[-1].strip():
                    self._logo_cache.pop()
            except Exception:
                self._logo_cache = ["A T S", "AGENT TRAINING SYSTEM"]
        return self._logo_cache

    def _logo_surface(self, target_w, colour=None):
        """The logo rendered once at full size, then scaled to fit.

        Scaling a rendered surface rather than picking a smaller font is what
        lets the same artwork sit in a 240px header and on a 700px splash and
        look like the same mark in both. At header size the individual
        characters stop resolving, which is fine - by then it is a shape.
        """
        colour = colour or GOLD
        key = (int(target_w), tuple(colour))
        if not hasattr(self, "_logo_surfs"):
            self._logo_surfs = {}
        hit = self._logo_surfs.get(key)
        if hit is not None:
            return hit

        lines = self._logo_lines()
        f = self._load_font(14)
        cw, ch = f.size("M")
        wide = max((len(ln) for ln in lines), default=1)
        full = pygame.Surface((max(1, wide * cw), max(1, len(lines) * ch)), pygame.SRCALPHA)
        for i, ln in enumerate(lines):
            if ln.strip():
                full.blit(f.render(ln, True, colour), (0, i * ch))

        scale = target_w / float(full.get_width())
        surf = pygame.transform.smoothscale(
            full, (int(full.get_width() * scale), max(1, int(full.get_height() * scale))))
        self._logo_surfs[key] = surf
        return surf

    def clear_session_data(self):
        """Resets in-memory table and graphs for a brand-new run."""
        self.episode_history.clear()
        self.reward_curve.clear()
        self.loss_history.clear()
        self.table_scroll = 0
        self.pan_offset = 0

    def record_episode(self, ep, reward, ticks, reason, grade="MID", efficiency=0.0, capabilities=0):
        entry = {
            "ep": ep,
            "reward": reward,
            "ticks": ticks,
            "reason": reason,
            "grade": grade,
            "efficiency": efficiency,
            "capabilities": capabilities,
            "time": time.strftime("%H:%M:%S")
        }
        self.episode_history.append(entry)
        self.reward_curve.append(reward)
        if self.table_scroll > 0:
            self.table_scroll = 0
        self.export_csv()

    def record_loss(self, total, actor, critic, entropy=None):
        self.loss_history.append((total, actor, critic))
        if len(self.loss_history) > 600:
            self.loss_history.pop(0)

    def record_narration(self, text):
        self.last_narration = text

    def export_csv(self, target_file=None):
        path = target_file or self.csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(path) == 0:
                    writer.writerow([
                        "episode", "timestamp", "total_reward", "grade",
                        "progression_efficiency", "capabilities_unlocked",
                        "ticks_survived", "termination_reason", "speed_multiplier",
                        "learning_rate", "entropy_coeff", "hidden_size"
                    ])
                if self.episode_history:
                    latest = self.episode_history[-1]
                    writer.writerow([
                        latest["ep"], latest["time"], f"{latest['reward']:.2f}",
                        latest.get("grade", "MID"), f"{latest.get('efficiency', 0.0):.3f}",
                        latest.get("capabilities", 0), latest["ticks"], latest["reason"],
                        config_rl.SPEED_MULTIPLIER, config_rl.LEARNING_RATE,
                        config_rl.ENTROPY_COEFF, config_rl.MIND_HIDDEN_SIZE
                    ])
        except Exception as e:
            print(f"[ATS GUI] CSV export note: {e}")

    def close(self):
        self.alive = False
        pygame.quit()

    # -----------------------------------------------------------------------
    # Flashy Vector Cyber Emblem Renderer (A + TS / A + GENT)
    # -----------------------------------------------------------------------
    def _draw_ats_cyber_logo(self, cx, cy, scale=0.85, pulse=0.0):
        """Renders a self-contained cyberpunk vector emblem without overflowing into text."""
        h = int(46 * scale)
        w = int(48 * scale)

        p_apex       = (cx, cy - h)
        p_left_base  = (cx - w, cy + h)
        p_right_base = (cx + w, cy + h)
        p_l_inner    = (cx - int(30 * scale), cy + h)
        p_r_inner    = (cx + int(30 * scale), cy + h)

        mid_y = cy - int(4 * scale)
        p_mid_l = (cx - int(20 * scale), mid_y)
        p_mid_r = (cx + int(20 * scale), mid_y)

        low_y = cy + int(22 * scale)
        p_low_l = (cx - int(34 * scale), low_y)
        p_low_r = (cx + int(34 * scale), low_y)

        fg_neon = (int(45 + 30 * math.sin(pulse * 3)), 255, int(125 + 30 * math.sin(pulse * 3)))
        gold_neon = (255, int(210 + 45 * math.sin(pulse * 4)), 95)
        cyan_neon = (int(75 + 40 * math.sin(pulse * 4)), 235, 255)
        glow_plate = (14, 32, 24)

        poly_pts = [p_apex, p_right_base, p_r_inner, p_mid_r, p_mid_l, p_l_inner, p_left_base]
        pygame.draw.polygon(self.screen, glow_plate, poly_pts)
        pygame.draw.polygon(self.screen, FRAME, poly_pts, 1)

        # Outer Beams
        pygame.draw.line(self.screen, fg_neon, p_apex, p_left_base, 3)
        pygame.draw.line(self.screen, fg_neon, p_apex, p_right_base, 3)
        pygame.draw.line(self.screen, fg_neon, p_left_base, p_l_inner, 2)
        pygame.draw.line(self.screen, fg_neon, p_right_base, p_r_inner, 2)

        # Inner Legs
        ch_apex = (cx, cy - int(18 * scale))
        pygame.draw.line(self.screen, fg_neon, p_l_inner, (cx - int(12 * scale), cy + int(6 * scale)), 2)
        pygame.draw.line(self.screen, fg_neon, p_r_inner, (cx + int(12 * scale), cy + int(6 * scale)), 2)
        pygame.draw.line(self.screen, fg_neon, ch_apex, (cx - int(12 * scale), cy + int(6 * scale)), 2)
        pygame.draw.line(self.screen, fg_neon, ch_apex, (cx + int(12 * scale), cy + int(6 * scale)), 2)

        # Crossbars
        pygame.draw.line(self.screen, gold_neon, p_mid_l, p_mid_r, 2)
        pygame.draw.line(self.screen, cyan_neon, p_low_l, p_low_r, 2)
        pygame.draw.circle(self.screen, fg_neon, p_apex, 3)

    # -----------------------------------------------------------------------
    # Splash Screen
    # -----------------------------------------------------------------------
    def splash(self, duration=1.0):
        start_t = time.time()
        while self.alive and time.time() - start_t < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.alive = False
                    return
                elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    return
            self.screen.fill(BG)
            self._draw_ats_cyber_logo(W // 2, H // 2 - 40, scale=1.3, pulse=time.time())
            self._blit("ATS // MISSION CONTROL v2.5", W // 2 - 130, H // 2 + 40, self.f_title, GOLD)
            self._blit("Initializing Neural Policy & Procedural Environment...", W // 2 - 190, H // 2 + 70, self.f_body, WHITE)
            pygame.display.flip()
            self.clock.tick(60)

    # -----------------------------------------------------------------------
    # Main Render Pipeline
    # -----------------------------------------------------------------------
    def render(self, env=None, needs=None, trace=None):
        if not self.alive:
            return False
        if not self.process_events():
            return False

        self.frame += 1
        self._pulse = (self._pulse + 0.05) % (2 * math.pi)
        self._clicks = []

        self.screen.fill(BG)
        pygame.draw.rect(self.screen, FRAME, (4, 4, W - 8, H - 8), 1)

        if self.app_state == STATE_MENU:
            if self.active_tab == TAB_ANALYTICS:
                self._draw_menu_header()
                self._draw_analytics(y_offset=120)
                self._hline(582)
                self._btn("[ < BACK TO SETUP ]", 20, 594, 220, 38, "tab:0", active=True, font=self.f_bold)
                self._btn("[ 3: HYPERPARAMS ]", 255, 594, 220, 38, "tab:2", active=False, font=self.f_bold)
                self._btn(">> LAUNCH TRAINING SESSION <<", 495, 594, 525, 38, "start", active=True, color=WHITE, bg_color=BTN_LAUNCH_BG, border_color=BTN_LAUNCH_BDR, font=self.f_header)
                self._blit("Hotkeys: [SPACE] Launch / Pause   |   [T] Turbo   |   [Ctrl+S] Save Checkpoint   |   [ESC] Quit ATS", 24, H - 20, self.f_small, DIM)
            elif self.active_tab == TAB_CONFIG:
                self._draw_menu_header()
                self._draw_config(y_offset=120)
                self._hline(582)
                self._btn("[ < BACK TO SETUP ]", 20, 594, 220, 38, "tab:0", active=True, font=self.f_bold)
                self._btn("[ 2: ANALYTICS ]", 255, 594, 220, 38, "tab:1", active=False, font=self.f_bold)
                self._btn(">> LAUNCH TRAINING SESSION <<", 495, 594, 525, 38, "start", active=True, color=WHITE, bg_color=BTN_LAUNCH_BG, border_color=BTN_LAUNCH_BDR, font=self.f_header)
                self._blit("Hotkeys: [SPACE] Launch / Pause   |   [T] Turbo   |   [Ctrl+S] Save Checkpoint   |   [ESC] Quit ATS", 24, H - 20, self.f_small, DIM)
            else:
                self._draw_menu()
        else:
            self._draw_chrome(env)
            avail = TAB_ACCESS.get(self.app_state, set())
            if self.active_tab not in avail:
                self.active_tab = next(iter(avail)) if avail else TAB_ANALYTICS

            if self.active_tab == TAB_SIM and env:
                self._draw_sim(env, needs or [0] * 4, trace or {})
            elif self.active_tab == TAB_ANALYTICS:
                self._draw_analytics()
            elif self.active_tab == TAB_CONFIG:
                self._draw_config()
            elif self.active_tab == TAB_MIND and env:
                self._draw_mind(env, needs or [0] * 4, trace or {})

            self._draw_toolbar()

        if self.sidebar_open:
            self._draw_sidebar_drawer()

        pygame.display.flip()
        self.clock.tick(60)
        return True

    # -----------------------------------------------------------------------
    # Event Processing
    # -----------------------------------------------------------------------
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.alive = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for rect, cb in self._clicks:
                        if rect.collidepoint(event.pos):
                            self._on_click(cb)
                            break
                elif event.button == 4:  # Wheel UP
                    if self.active_tab == TAB_ANALYTICS:
                        self.table_scroll = min(max(0, len(self.episode_history) - 6), self.table_scroll + 1)
                elif event.button == 5:  # Wheel DOWN
                    if self.active_tab == TAB_ANALYTICS:
                        self.table_scroll = max(0, self.table_scroll - 1)
            elif event.type == pygame.KEYDOWN:
                self._on_key(event)
        return self.alive

    def _on_key(self, event):
        k = event.key
        mods = pygame.key.get_mods()

        if k == pygame.K_ESCAPE:
            if self.sidebar_open:
                self.sidebar_open = False
            elif self.app_state == STATE_MENU:
                self.alive = False
            elif self.app_state in (STATE_RUNNING, STATE_PAUSED):
                self.sig_pause_toggle = True
            else:
                self.sig_return_home = True

        avail = TAB_ACCESS.get(self.app_state, set())
        if k == pygame.K_1 and TAB_SIM in avail: self.active_tab = TAB_SIM
        if k == pygame.K_2 and TAB_ANALYTICS in avail: self.active_tab = TAB_ANALYTICS
        if k == pygame.K_3 and TAB_CONFIG in avail: self.active_tab = TAB_CONFIG
        if k == pygame.K_4 and TAB_MIND in avail: self.active_tab = TAB_MIND

        if k == pygame.K_SPACE:
            if self.app_state == STATE_MENU:
                self.sig_start = True
            elif self.app_state in (STATE_RUNNING, STATE_PAUSED):
                self.sig_pause_toggle = True

        if k == pygame.K_t and self.app_state == STATE_RUNNING:
            self.turbo_mode = not self.turbo_mode
        if k == pygame.K_s and (mods & pygame.KMOD_CTRL):
            self.sig_save = True

        if self.app_state in (STATE_RUNNING, STATE_PAUSED):
            _map = {pygame.K_w: config_rl.ACT_MOVE_FORWARD,
                    pygame.K_s: config_rl.ACT_MOVE_BACKWARD,
                    pygame.K_a: config_rl.ACT_MOVE_LEFT,
                    pygame.K_d: config_rl.ACT_MOVE_RIGHT}
            for key, act in _map.items():
                if k == key and not (mods & pygame.KMOD_CTRL):
                    self._toggle_action(act)

    def _on_click(self, cb):
        if cb == "start":
            self.sig_start = True
        elif cb == "toggle_pause":
            self.sig_pause_toggle = True
        elif cb == "turbo":
            self.turbo_mode = not self.turbo_mode
        elif cb == "save":
            self.sig_save = True
        elif cb.startswith("curr:"):
            idx = int(cb.split(":")[1])
            if 0 <= idx < len(TRACKS) and TRACKS[idx][3]:
                self.selected_track = idx
        elif cb == "arm_fresh":
            self._confirm_fresh_armed = True
        elif cb == "cancel_fresh":
            self._confirm_fresh_armed = False
        elif cb == "fresh_reset":
            self._confirm_fresh_armed = False
            self.sig_fresh_reset = True
            self.is_fresh_mode = True
        elif cb == "stop":
            self.sig_stop = True
        elif cb == "return_home":
            self.sig_return_home = True
        elif cb == "toggle_sidebar":
            self.sidebar_open = not self.sidebar_open
        elif cb == "close_sidebar":
            self.sidebar_open = False
        elif cb.startswith("tab:"):
            t = int(cb.split(":")[1])
            if t in TAB_ACCESS.get(self.app_state, set()):
                self.active_tab = t
            self.sidebar_open = False
        elif cb.startswith("act:"):
            self._toggle_action(int(cb.split(":")[1]))
        elif cb.startswith("rule:"):
            key = cb.split(":")[1]
            self.reward_rules[key] = not self.reward_rules.get(key, True)
        elif cb.startswith("zoom:"):
            action = cb.split(":")[1]
            if action == "in":
                self.zoom_window = max(5, self.zoom_window // 2 if self.zoom_window > 0 else 50)
            elif action == "out":
                self.zoom_window = min(500, (self.zoom_window * 2) if self.zoom_window > 0 else 50)
            elif action == "all":
                self.zoom_window = 0
            elif action.isdigit():
                self.zoom_window = int(action)
        elif cb.startswith("scroll:"):
            action = cb.split(":")[1]
            if action == "up":
                self.table_scroll = min(max(0, len(self.episode_history) - 6), self.table_scroll + 3)
            elif action == "down":
                self.table_scroll = max(0, self.table_scroll - 3)
        elif cb == "export_csv":
            self.export_csv()
        elif cb.startswith("cfg:"):
            self._apply_cfg(cb[4:])

    def _toggle_action(self, idx):
        if idx in self.disabled_actions:
            self.disabled_actions.remove(idx)
        else:
            self.disabled_actions.add(idx)

    def _apply_cfg(self, spec):
        attr, action = spec.split(":", 1)
        cur = getattr(config_rl, attr, None)
        if cur is None:
            return

        for f in _CONFIG_FIELDS:
            if f[0] != attr:
                continue
            _, _, step, lo, hi, _ = f

            if action == "inc" and step is not None:
                new_val = cur + step
                if hi is not None:
                    new_val = min(new_val, hi)
                setattr(config_rl, attr, new_val)
            elif action == "dec" and step is not None:
                new_val = cur - step
                if lo is not None:
                    new_val = max(new_val, lo)
                setattr(config_rl, attr, new_val)
            else:
                try:
                    if isinstance(cur, int):
                        setattr(config_rl, attr, int(float(action)))
                    else:
                        setattr(config_rl, attr, float(action))
                except ValueError:
                    pass
            break

    # -----------------------------------------------------------------------
    # Clean UI Helpers
    # -----------------------------------------------------------------------
    def _blit(self, text, x, y, font=None, color=FG, max_w=None):
        font = font or self.f_body
        surf = font.render(str(text), True, color)
        if max_w and surf.get_width() > max_w:
            txt_str = str(text)
            while len(txt_str) > 1 and font.size(txt_str + "..")[0] > max_w:
                txt_str = txt_str[:-1]
            surf = font.render(txt_str + "..", True, color)
        self.screen.blit(surf, (x, y))

    def _hline(self, y, x0=14, x1=None):
        pygame.draw.line(self.screen, DIM2, (x0, y), (x1 or (W - 14), y), 1)

    def _btn(self, text, x, y, w, h, cb, active=False, locked=False, color=WHITE, bg_color=None, border_color=None, font=None):
        font = font or self.f_small
        rect = pygame.Rect(x, y, w, h)
        if not locked:
            self._clicks.append((rect, cb))
        mp = pygame.mouse.get_pos()
        hover = rect.collidepoint(mp) and not locked

        if locked:
            bg = BTN_LOCKED_BG
            bdr = DIM2
            tc = LOCK_FG
        elif bg_color is not None:
            bg = tuple(min(255, int(c * 1.2)) for c in bg_color) if hover else bg_color
            bdr = border_color or tuple(min(255, int(c * 1.3)) for c in bg_color)
            tc = color
        else:
            bg = BTN_ACTIVE_BG if active else (BTN_HOVER_BG if hover else BTN_IDLE_BG)
            bdr = GOLD if active else (FG_BOLD if hover else FRAME)
            tc = GOLD if active else (WHITE if hover else color)

        pygame.draw.rect(self.screen, bg, rect, border_radius=3)
        pygame.draw.rect(self.screen, bdr, rect, 1, border_radius=3)
        surf = font.render(text, True, tc)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _panel(self, x, y, w, h, title=None):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, BG_PANEL, rect, border_radius=4)
        pygame.draw.rect(self.screen, FRAME, rect, 1, border_radius=4)
        if title:
            self._blit(title, x + 12, y + 10, self.f_bold, GOLD)
        return rect

    def _draw_curriculum_card(self, x, y, w, h):
        """What to run, and how far up the ladder it already is.

        Locked tracks are drawn rather than hidden. A menu that grows new
        entries as you progress hides the shape of the thing you are climbing,
        and the shape is the point of a curriculum.
        """
        self._panel(x, y, w, h, "2. CURRICULUM")
        prog = brain_progress()
        passed = set(prog.get("passed") or [])

        row = y + 34
        for i, (name, rungs, blurb, open_) in enumerate(TRACKS):
            stages = track_rungs(rungs)
            done = sum(1 for st in stages if f"{st.name} {st.grid}x{st.grid}" in passed)
            sel = (self.selected_track == i)

            if not open_:
                tail = "coming soon"
            elif stages:
                tail = f"{done}/{len(stages)} passed"
            else:
                tail = "full island"

            label = f"{name:<11} {tail}"
            self._btn(label, x + 12, row, w - 24, 32, f"curr:{i}",
                      active=sel, locked=not open_, font=self.f_bold)
            row += 34
            self._blit(blurb, x + 18, row, self.f_tiny, DIM if open_ else DIM2)
            row += 18

        pygame.draw.line(self.screen, DIM2, (x + 12, row + 4), (x + w - 12, row + 4), 1)
        row += 16

        name, rungs, _, open_ = TRACKS[self.selected_track]
        stages = track_rungs(rungs)
        if stages:
            self._blit(f"{name} RUNGS", x + 14, row, self.f_bold, GOLD)
            row += 20
            for st in stages:
                key = f"{st.name} {st.grid}x{st.grid}"
                ok = key in passed
                # A tick you can read at a glance beats a progress bar you
                # have to interpret - there are eight of these, not eight
                # hundred.
                self._blit("[x]" if ok else "[ ]", x + 16, row, self.f_small,
                           FG_BOLD if ok else DIM2)
                self._blit(key, x + 44, row, self.f_small, WHITE if ok else DIM)
                row += 17
        else:
            self._blit("Nothing staged - this track runs the", x + 14, row, self.f_tiny, DIM)
            row += 14
            self._blit("full world directly." if open_ else "rungs are not built yet.",
                       x + 14, row, self.f_tiny, DIM)

    def _draw_menu_header(self):
        logo = self._logo_surface(300, GOLD)
        self.screen.blit(logo, (18, 14))

        header_x = 336
        self._blit("MISSION CONTROL v2.6", header_x, 20, self.f_title, GOLD)
        self._blit("Agent Training System  //  curriculum + embodied RL",
                   header_x, 46, self.f_body, WHITE)

        b = brain_progress()
        passed = b.get("passed") or []
        if passed:
            line = f"[● BRAIN]  {len(passed)} rung(s) passed  |  {b.get('episodes', 0)} episodes lived"
            col = FG_BOLD
        else:
            line = "[○ BRAIN]  no curriculum progress yet - start at NURSERY"
            col = AMBER
        self._blit(line, header_x, 70, self.f_small, col)
        self._blit(f"Checkpoint: {config_rl.MODEL_PATH.name}  |  Profile: {config_rl.VIRUS_PROFILE}",
                   header_x, 88, self.f_tiny, DIM)

        self._hline(116)

    # -----------------------------------------------------------------------
    # 🏠 MAIN MENU / HOME SCREEN
    # -----------------------------------------------------------------------
    def _draw_menu(self):
        # 1. Top Header
        self._draw_menu_header()

        ckpt_name = config_rl.MODEL_PATH.name
        ckpt_exists = config_rl.MODEL_PATH.exists()

        # 2. Main Content Grid (Two clean side-by-side cards)
        # Three columns now: what to run is a first-class choice, not
        # something buried behind a tab.
        card_y = 140
        card_h = 424
        card_w = 318
        COL1, COL2, COL3 = 16, 361, 706

        # LEFT CARD: PRE-LAUNCH HYPERPARAMETERS
        self._panel(COL1, card_y, card_w, card_h, "1. HYPERPARAMETERS")
        row = card_y + 38

        params = [
            ("EPISODES",         "Target Episodes",     10,  "{:d}"),
            ("MAX_TICKS",        "Max Ticks / Ep",     500,  "{:d}"),
            ("SPEED_MULTIPLIER", "Sim Speed",          2.0,  "{:.1f}x"),
            ("DAY_LENGTH_TICKS", "Day Length",         100,  "{:d} ticks"),
        ]
        for attr, label, step, fmt in params:
            val = getattr(config_rl, attr)
            self._blit(f"{label}", 32, row + 2, self.f_body, WHITE)
            self._blit(fmt.format(val), 158, row + 2, self.f_bold, FG_BOLD)
            self._btn("[-]", 224, row, 42, 22, f"cfg:{attr}:dec")
            self._btn("[+]", 270, row, 42, 22, f"cfg:{attr}:inc")
            row += 32

        row += 4
        # Learning Rate Row
        self._blit("Learning Rate", 32, row, self.f_body, WHITE)
        self._blit(f"{config_rl.LEARNING_RATE:.2e}", 158, row, self.f_bold, FG_BOLD)
        row += 24
        lr_x = 32
        for lr in _LR_PRESETS:
            active = abs(config_rl.LEARNING_RATE - lr) < lr * 0.01
            self._btn(f"{lr:.0e}", lr_x, row, 68, 22, f"cfg:LEARNING_RATE:{lr}", active=active)
            lr_x += 72
        row += 34

        # Entropy Coeff Row
        self._blit("Entropy Coeff", 32, row, self.f_body, WHITE)
        self._blit(f"{config_rl.ENTROPY_COEFF:.3f}", 158, row, self.f_bold, FG_BOLD)
        row += 24
        ent_x = 32
        for ent in _ENTROPY_PRESETS:
            active = abs(config_rl.ENTROPY_COEFF - ent) < 0.0001
            self._btn(f"{ent:.3f}", ent_x, row, 68, 22, f"cfg:ENTROPY_COEFF:{ent}", active=active)
            ent_x += 72
        row += 34

        # Hidden Width
        self._blit("Hidden / Embed", 32, row, self.f_body, WHITE)
        self._blit(f"{config_rl.MIND_HIDDEN_SIZE} / {config_rl.ITEM_EMBED_DIM}",
                   158, row, self.f_bold, FG_BOLD)
        row += 20
        self._blit("growth is OFF (MIND_GROWTH_ENABLED)", 32, row, self.f_tiny, DIM)
        row += 24

        # Checkpoint confirmation line
        ckpt_col = FG_BOLD if (ckpt_exists and not self.is_fresh_mode) else AMBER
        ckpt_msg = "Status: Fresh policy weights (Scratch Start)" if self.is_fresh_mode else (f"Status: Checkpoint '{ckpt_name}' ready" if ckpt_exists else "Status: Fresh random policy weights")
        self._blit(ckpt_msg, 32, row, self.f_tiny, ckpt_col)
        row += 22

        # Safe fresh start button (2-step confirmation)
        if not self._confirm_fresh_armed:
            self._btn("[ FRESH START - NEW MODEL ]", 32, row, 280, 26, "arm_fresh", color=AMBER, bg_color=(45, 20, 25), border_color=RED, font=self.f_small)
        else:
            self._btn("[ CONFIRM? ]", 32, row, 150, 26, "fresh_reset", color=WHITE, bg_color=BTN_STOP_BG, border_color=GOLD, font=self.f_bold)
            self._btn("[ CANCEL ]", 190, row, 122, 26, "cancel_fresh", color=DIM, font=self.f_small)

        # MIDDLE CARD: the ladder
        self._draw_curriculum_card(COL2, card_y, card_w, card_h)

        # RIGHT CARD: AGENT & ENVIRONMENT MATRIX
        rx = COL3
        self._panel(rx, card_y, card_w, card_h, "3. AGENT & ENVIRONMENT")
        info_row = card_y + 38

        specs = [
            ("Policy",     f"{config_rl.MIND_HIDDEN_SIZE}-hidden actor-critic"),
            ("Embeddings", f"item/entity {config_rl.ITEM_EMBED_DIM}, tile {config_rl.TILE_EMBED_DIM}"),
            ("State",      f"{config_rl.STATE_SIZE} dims"),
            ("Actions",    f"{config_rl.ACTION_SIZE} discrete"),
            ("Rollout",    f"{config_rl.CONTINUAL_BUFFER_SIZE} steps / {config_rl.CONTINUAL_UPDATE_EVERY}t"),
            ("Map",        "7 biomes, gates, boss arena"),
        ]
        for k, v in specs:
            self._blit(f"{k}", rx + 14, info_row, self.f_body, DIM)
            self._blit(v, rx + 96, info_row, self.f_body, WHITE)
            info_row += 22

        info_row += 8
        pygame.draw.line(self.screen, DIM2, (rx + 14, info_row), (rx + card_w - 14, info_row), 1)
        info_row += 14

        self._blit("REWARD RULES (click to toggle)", rx + 14, info_row, self.f_bold, GOLD)
        info_row += 24

        for idx, (label, key) in enumerate(_RULE_LABELS):
            en = self.reward_rules.get(key, True)
            col_pos = idx % 2
            row_pos = idx // 2
            bx = rx + 14 + col_pos * 152
            by = info_row + row_pos * 30
            btn_txt = f"[ON] {label}" if en else f"[OFF] {label}"
            btn_col = FG_BOLD if en else DIM
            btn_bg  = (20, 50, 35) if en else (28, 20, 22)
            btn_bdr = (45, 120, 75) if en else (70, 35, 40)
            self._btn(btn_txt, bx, by, 140, 24, f"rule:{key}", active=en, color=btn_col, bg_color=btn_bg, border_color=btn_bdr)

        # 3. Bottom Launch & Navigation Section
        self._hline(578)
        nav_y = 590

        self._btn("[ 2: ANALYTICS ]", 20, nav_y, 220, 38, "tab:1", active=(self.active_tab == TAB_ANALYTICS), font=self.f_bold)
        self._btn("[ 3: HYPERPARAMS ]", 255, nav_y, 220, 38, "tab:2", active=(self.active_tab == TAB_CONFIG), font=self.f_bold)

        # Main Launch Button
        track = TRACKS[self.selected_track][0]
        self._btn(f">> LAUNCH: {track} <<", 495, nav_y, 529, 38, "start", active=True,
                  color=WHITE, bg_color=BTN_LAUNCH_BG, border_color=BTN_LAUNCH_BDR,
                  font=self.f_header)

        self._blit("Hotkeys: [SPACE] Launch / Pause   |   [T] Turbo   |   [Ctrl+S] Save Checkpoint   |   [ESC] Quit ATS", 24, H - 20, self.f_small, DIM)

    # -----------------------------------------------------------------------
    # In-Session Chrome & Drawer
    # -----------------------------------------------------------------------
    def _draw_chrome(self, env):
        st_col = {STATE_RUNNING: FG_BOLD, STATE_PAUSED: AMBER, STATE_STOPPED: RED}.get(self.app_state, WHITE)
        cursor = "_" if (self.frame // 20) % 2 == 0 else " "
        ep_str = f"  |  EP: {env.episode}/{config_rl.EPISODES}  TICK: {env.tick}/{config_rl.MAX_TICKS}" if env else ""
        turbo_str = "  [TURBO ON]" if self.turbo_mode else ""
        self._blit(f"ATS MISSION CONTROL {cursor} [{self.app_state}]{turbo_str}{ep_str}", 16, 10, self.f_bold, st_col)

        # Top-Right Collapsible Sidebar Trigger
        drawer_txt = "[ X CLOSE ]" if self.sidebar_open else "[ < TABS / VIEWS ]"
        self._btn(drawer_txt, W - 160, 6, 145, 26, "toggle_sidebar", active=self.sidebar_open, color=GOLD, font=self.f_bold)
        self._hline(36)

    def _draw_sidebar_drawer(self):
        dw = 230
        dx = W - dw - 10
        dy = 40
        dh = H - 110

        pygame.draw.rect(self.screen, BG_PANEL2, (dx, dy, dw, dh), border_radius=4)
        pygame.draw.rect(self.screen, GOLD, (dx, dy, dw, dh), 1, border_radius=4)

        self._blit("=== NAVIGATION ===", dx + 45, dy + 12, self.f_bold, GOLD)
        pygame.draw.line(self.screen, DIM2, (dx + 10, dy + 32), (dx + dw - 10, dy + 32), 1)

        tabs = [
            (TAB_SIM,       "1: [ LIVE SIM ]"),
            (TAB_ANALYTICS, "2: [ ANALYTICS ]"),
            (TAB_CONFIG,    "3: [ HYPERPARAMS ]"),
            (TAB_MIND,      "4: [ MIND & LLM ]"),
        ]
        avail = TAB_ACCESS.get(self.app_state, set())
        ty = dy + 44
        for tab_id, label in tabs:
            locked = tab_id not in avail
            is_active = (self.active_tab == tab_id)
            self._btn(label, dx + 12, ty, dw - 24, 32, f"tab:{tab_id}", active=is_active, locked=locked, font=self.f_body)
            ty += 42

        pygame.draw.line(self.screen, DIM2, (dx + 10, ty + 10), (dx + dw - 10, ty + 10), 1)
        self._btn("[ CLOSE DRAWER ]", dx + 12, ty + 24, dw - 24, 28, "close_sidebar", color=AMBER)

    # -----------------------------------------------------------------------
    # TAB 0: LIVE SIMULATION
    # -----------------------------------------------------------------------
    def _draw_sim(self, env, needs, trace):
        agent = env.agent
        world = env.world
        y0 = 44

        self._blit(f"INVENTORY ({config_rl.INVENTORY_SLOTS} SLOTS)", 16, y0, self.f_bold, GOLD)
        facing_lbl = {"N": "NORTH ^", "S": "SOUTH v", "W": "WEST <", "E": "EAST >"}.get(agent.facing_dir, "?")
        self._blit(f"WORLD VIEW (7x7)    FACING: {facing_lbl}", 520, y0, self.f_bold, GOLD)
        pygame.draw.line(self.screen, DIM2, (505, 40), (505, 270), 1)

        # Inventory is meaningless in a stage that cannot pick anything up.
        # Shown anyway, greyed and labelled, for the same reason as the action
        # toggles.
        inv_live = self.active_mask is None or (
            config_rl.ACT_PICK_UP < len(self.active_mask)
            and self.active_mask[config_rl.ACT_PICK_UP])
        if not inv_live:
            self._blit("INVENTORY - not used in this stage", 16, y0 + 18,
                       self.f_small, DIM2)

        inv_y = y0 + (36 if not inv_live else 18)
        for i in range(len(agent.inventory)):
            yp = inv_y + i * 24
            sl = agent.inventory[i]
            sel = (i == agent.selected_slot)
            mk = "> " if sel else "  "
            if sl["id"] and sl["count"] > 0:
                txt = f"{mk}Slot {i}: [{sl['id']}] {item_name(sl['id'])[:14]} x{sl['count']}"
                c = FG_BOLD if sel else FG
            else:
                txt = f"{mk}Slot {i}: [EMPTY - DISABLED]"
                c = RED if sel else DIM
            self._blit(txt, 16, yp, self.f_body, c)

        # World Grid 7x7
        cw, ch = 70, 18
        fs = {"N": "[AI^]", "S": "[AIv]", "W": "[AI<]", "E": "[AI>]"}.get(agent.facing_dir, "[AI?]")
        for r in range(-3, 4):
            ry = inv_y + (r + 3) * ch
            for c in range(-3, 4):
                cx = 520 + (c + 3) * cw
                wx, wy = agent.x + c, agent.y + r
                if c == 0 and r == 0:
                    cs, cc = fs, FG_BOLD
                else:
                    t = world._tile(wx, wy)
                    ent = next((e for e in world.entities if e.alive and e.x == wx and e.y == wy), None)
                    if not t or not t.explored:
                        cs, cc = "??", DIM
                    elif ent:
                        cs = entity_name(ent.entity_id)[:5]
                        cc = RED if ent.data.get("type") == "hostile" else AMBER
                    elif t.object_id:
                        raw = item_name(t.object_id)
                        mapping = {"Tree": "Tree", "Bush": "Bush", "Log": "Log", "Stone": "Rock", "Boulder": "Rock", "Coal": "Coal", "Stick": "Stck", "Apple": "Appl", "Mushroom": "Mush", "Flower": "Flow", "Cactus": "Cact"}
                        cs = next((v for k, v in mapping.items() if k in raw), raw[:5])
                        cc = GOLD if cs in ("Appl", "Mush") else FG
                    elif t.tile_type == 1: cs, cc = "#WALL", DIM
                    elif t.tile_type == 5: cs, cc = "~H2O~", CYAN
                    elif t.tile_type == 6: cs, cc = "~LAV~", RED
                    elif t.tile_type == 7: cs, cc = "[GATE]", GOLD
                    elif t.tile_type == 8: cs, cc = "~SEA~", (80, 140, 240)
                    elif t.tile_type == 9: cs, cc = "~ACID~", (60, 220, 60)
                    elif t.tile_type == 10: cs, cc = "[ICE]", (160, 230, 255)
                    elif t.tile_type == 11: cs, cc = "[ASH]", (180, 180, 180)
                    elif t.tile_type == 12: cs, cc = "%SPOR%", (200, 80, 220)
                    else:                  cs, cc = ".", DIM
                self._blit(cs, cx, ry, self.f_body, cc)

        self._blit(f"TARGET: {agent.get_facing_tile_info(world)}", 520, inv_y + 7 * ch + 4, self.f_body, CYAN)
        nc = RED if any(n > 0.5 for n in needs[:3]) else FG
        self._blit(f"NEEDS  Hunger:{needs[0]:.2f}  Injury:{needs[1]:.2f}  Threat:{needs[2]:.2f}  Tool:{needs[3]:.2f}", 520, inv_y + 7 * ch + 24, self.f_body, nc)

        # Vitals Bar
        self._hline(276)
        zinfo = world.get_zone_info(agent.x, agent.y)
        isl_name = zinfo.get("name", "Ocean")
        isl_tier = zinfo.get("tier", 0)
        hp_c = FG_BOLD if agent.health > 50 else (AMBER if agent.health > 25 else RED)
        self._blit(f"HP: {agent.health:3d}/100   HUNGER: {agent.hunger:3d}/100   STAMINA: {agent.stamina:3d}/100   LV: {agent.level}   XP: {agent.xp}   POS: ({agent.x},{agent.y})", 16, 282, self.f_body, hp_c)
        nt = "NIGHT" if env.day_night.is_night else "DAY"
        caps_str = ",".join(list(agent.capabilities)[:4]) if agent.capabilities else "none"
        tier_badge = f"[TIER {isl_tier}]" if isl_tier >= 0 else "[DANGER]"
        water_drag_tag = f" [WATER DRAG: {int(agent.water_speed_mult * 100)}% SPD]" if agent.water_speed_mult < 1.0 else ""
        ocean_warning = f"  |  [ ! OPEN OCEAN IMMERSION: {config_rl.OCEAN_SHARK_TICKS - agent.ocean_ticks}s ! ]" if agent.ocean_ticks > 0 else ""
        self._blit(f"ISLAND: {isl_name} {tier_badge}   TIME: {env.day_night.time_of_day:.2f} ({nt}){water_drag_tag}   CAPS: [{caps_str}]{ocean_warning}", 16, 302, self.f_body, RED if agent.ocean_ticks > 0 else (AMBER if env.day_night.is_night else CYAN))

        # Recent Activity
        self._hline(324)
        self._blit("RECENT ACTIVITY & EVENTS", 16, 328, self.f_bold, GOLD)
        evts = self._fmt_events(agent.event_log, 2)
        for i, (txt, col) in enumerate(evts):
            self._blit(txt, 16, 348 + i * 18, self.f_body, col)

        # Mind Traces
        self._hline(390)
        self._blit("MIND COGNITIVE CYCLE", 16, 394, self.f_bold, GOLD)
        self._blit(f"Detect: {str(trace.get('detect', ''))[:100]}", 16, 414, self.f_body, CYAN)
        self._blit(f"Act   : {str(trace.get('act', ''))[:100]}", 16, 432, self.f_body, CYAN)

        # Action Toggles
        self._hline(454)
        stage_note = f"   [{self.active_stage}]" if self.active_stage else ""
        self._blit(f"ACTION TOGGLES (Click or W/A/S/D to toggle):{stage_note}",
                   16, 458, self.f_small, DIM)
        bx = 16
        for label, idx in _ACTION_TOGGLES:
            in_stage = self.active_mask is None or (
                idx < len(self.active_mask) and self.active_mask[idx])
            en = (idx not in self.disabled_actions) and in_stage
            self._btn(f"[{'X' if en else ' '}] {label}", bx, 474, 68, 22,
                      f"act:{idx}", active=en, locked=not in_stage)
            bx += 72

        # Reward Rules
        self._blit("REWARD RULES:", 16, 502, self.f_small, DIM)
        bx = 16
        for label, key in _RULE_LABELS:
            en = self.reward_rules.get(key, True)
            self._btn(f"[{'X' if en else ' '}] {label}", bx, 518, 92, 22, f"rule:{key}", active=en, color=AMBER)
            bx += 98

        # Stats Line
        self._hline(546)
        ent, _ = world.nearest_entity(agent.x, agent.y)
        ent_lbl = f"{entity_name(ent.entity_id)}(HP:{ent.hp})" if ent else "none"
        eff_val = env.rewards.compute_progression_efficiency(env.tick)
        self._blit(f"LAST: {agent.last_action_name:<12} | TARGET: {ent_lbl:<14} | TOTAL REW: {env.rewards.total:+7.2f} | EFF: {eff_val:.3f} | PTS: {env.rewards.progression_points:.1f}", 16, 550, self.f_body, FG_BOLD, max_w=1000)

    def _fmt_events(self, log, n):
        if not log:
            return [(">> Exploring open procedural world...", DIM)]
        grouped = []
        for txt, _ in log:
            if grouped and grouped[-1][0] == txt:
                grouped[-1] = (txt, grouped[-1][1] + 1)
            else:
                grouped.append((txt, 1))
        out = []
        for txt, cnt in grouped[-n:]:
            suf = f" (x{cnt})" if cnt > 1 else ""
            out.append((f">> {txt}{suf}", WHITE))
        return out

    # -----------------------------------------------------------------------
    # TAB 1: ADVANCED ANALYTICS & ZOOMABLE GRAPHS
    # -----------------------------------------------------------------------
    def _draw_analytics(self, y_offset=44):
        y0 = y_offset

        cards = [
            ("Total Episodes", str(len(self.episode_history))),
            ("Best Reward", f"{max(self.reward_curve):.2f}" if self.reward_curve else "—"),
            ("Latest Reward", f"{self.reward_curve[-1]:.2f}" if self.reward_curve else "—"),
            ("Avg Last 10", f"{sum(self.reward_curve[-10:]) / len(self.reward_curve[-10:]):.2f}" if len(self.reward_curve) >= 2 else "—"),
        ]
        cx = 16
        for title, val in cards:
            self._panel(cx, y0, 240, 52, None)
            self._blit(title, cx + 10, y0 + 6, self.f_small, DIM)
            self._blit(val, cx + 10, y0 + 24, self.f_header, FG_BOLD)
            cx += 254
        y0 += 60

        # Timeframe & Zoom Controls Bar
        self._blit("TIMEFRAME / ZOOM:", 16, y0 + 3, self.f_bold, GOLD)
        zx = 160
        presets = [("10 Ep", "10"), ("25 Ep", "25"), ("50 Ep", "50"), ("ALL", "all")]
        for plabel, pval in presets:
            is_active = (str(self.zoom_window) == pval or (pval == "all" and self.zoom_window == 0))
            self._btn(plabel, zx, y0 - 2, 58, 22, f"zoom:{pval}", active=is_active)
            zx += 64

        self._btn("[-] OUT", zx + 10, y0 - 2, 65, 22, "zoom:out")
        self._btn("[+] IN", zx + 80, y0 - 2, 65, 22, "zoom:in")
        self._btn("EXPORT CSV", W - 155, y0 - 2, 140, 22, "export_csv", color=GOLD)
        y0 += 30

        # Two Zoomable Telemetry Graphs
        lrect = pygame.Rect(16, y0, 490, 165)
        rrect = pygame.Rect(518, y0, 506, 165)
        pygame.draw.rect(self.screen, BG_PANEL, lrect, border_radius=4)
        pygame.draw.rect(self.screen, FRAME, lrect, 1, border_radius=4)
        pygame.draw.rect(self.screen, BG_PANEL, rrect, border_radius=4)
        pygame.draw.rect(self.screen, FRAME, rrect, 1, border_radius=4)

        win_lbl = f"(Window: Last {self.zoom_window})" if self.zoom_window > 0 else "(Window: All Time)"
        self._blit(f"EPISODE REWARD CURVE {win_lbl}", lrect.x + 10, lrect.y + 6, self.f_small, GOLD)
        self._blit("PPO LOSS HISTORY (Total / Actor / Critic)", rrect.x + 10, rrect.y + 6, self.f_small, CYAN)

        rew_slice = self.reward_curve[-self.zoom_window:] if (self.zoom_window > 0 and len(self.reward_curve) > self.zoom_window) else self.reward_curve
        loss_slice = self.loss_history[-self.zoom_window * 4:] if (self.zoom_window > 0 and len(self.loss_history) > self.zoom_window * 4) else self.loss_history

        if len(rew_slice) >= 2:
            self._line_graph(rew_slice, lrect, FG, pad_x=50, pad_top=24)
        else:
            self._blit("Awaiting episode completions to plot reward curve...", lrect.x + 14, lrect.y + 75, self.f_body, DIM)

        if len(loss_slice) >= 2:
            totals  = [l[0] for l in loss_slice]
            actors  = [l[1] for l in loss_slice]
            critics = [l[2] for l in loss_slice]
            self._line_graph(totals, rrect, CYAN, pad_x=50, pad_top=24)
            self._line_graph(actors, rrect, AMBER, pad_x=50, pad_top=24, draw_axes=False)
            self._line_graph(critics, rrect, RED, pad_x=50, pad_top=24, draw_axes=False)
        else:
            self._blit("Awaiting continual PPO updates to plot loss...", rrect.x + 14, rrect.y + 75, self.f_body, DIM)

        y0 += 175
        self._hline(y0)
        y0 += 8

        # Recent Episode Summary Table in ASCENDING Order
        self._blit("EPISODE SUMMARY LOG (Ascending Order — Latest at Bottom):", 16, y0 + 3, self.f_bold, GOLD)
        self._btn("[▲] SCROLL UP", W - 255, y0 - 2, 115, 22, "scroll:up")
        self._btn("[▼] SCROLL DOWN", W - 135, y0 - 2, 120, 22, "scroll:down")
        y0 += 26

        headers = [("EPISODE", 16), ("REWARD", 100), ("GRADE", 195), ("EFFICIENCY", 270), ("TICKS", 370), ("TERMINATION REASON", 440), ("TIME", 880)]
        for h, hx in headers:
            self._blit(h, hx, y0, self.f_small, DIM)
        self._hline(y0 + 16)
        y0 += 18

        total_eps = len(self.episode_history)
        visible_rows = 6
        start_idx = max(0, total_eps - visible_rows - self.table_scroll)
        end_idx = min(total_eps, start_idx + visible_rows)

        visible_entries = self.episode_history[start_idx:end_idx]
        for entry in visible_entries:
            rc = FG_BOLD if entry["reward"] > 0 else RED
            self._blit(f"Ep {entry['ep']:04d}", 16, y0, self.f_body, FG)
            self._blit(f"{entry['reward']:+7.2f}", 100, y0, self.f_body, rc)
            grd = entry.get("grade", "MID")
            grd_col = FG_BOLD if grd == "GOOD" else (GOLD if grd == "MID" else (AMBER if grd == "FAIR" else RED))
            self._blit(grd, 195, y0, self.f_bold, grd_col)
            self._blit(f"{entry.get('efficiency', 0.0):.3f}", 270, y0, self.f_body, CYAN)
            self._blit(f"{entry['ticks']:5d}", 370, y0, self.f_body, WHITE)
            rc2 = RED if "Perished" in entry["reason"] or "HP" in entry["reason"] else CYAN
            self._blit(entry["reason"], 440, y0, self.f_body, rc2, max_w=430)
            self._blit(entry["time"], 880, y0, self.f_body, DIM)
            y0 += 20

    def _line_graph(self, values, rect, color, pad_x=50, pad_top=24, draw_axes=True):
        pw = rect.width - pad_x * 2
        ph = rect.height - pad_top - 18
        gx = rect.x + pad_x
        gy = rect.y + pad_top

        mn, mx = min(values), max(values)
        if mn == mx: mx = mn + 1.0

        if draw_axes:
            pygame.draw.line(self.screen, DIM2, (gx, gy), (gx, gy + ph))
            pygame.draw.line(self.screen, DIM2, (gx, gy + ph), (gx + pw, gy + ph))
            self._blit(f"{mx:.1f}", rect.x + 4, gy - 4, self.f_tiny, DIM)
            self._blit(f"{mn:.1f}", rect.x + 4, gy + ph - 8, self.f_tiny, DIM)

        pts = []
        for i, v in enumerate(values):
            nx = i / (len(values) - 1)
            ny = (v - mn) / (mx - mn)
            pts.append((int(gx + nx * pw), int(gy + ph - ny * ph)))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 1)
        if pts:
            pygame.draw.circle(self.screen, GOLD, pts[-1], 3)
            self._blit(f"{values[-1]:.2f}", pts[-1][0] + 4, pts[-1][1] - 8, self.f_tiny, GOLD)

    # -----------------------------------------------------------------------
    # TAB 2: HYPERPARAMETER STUDIO
    # -----------------------------------------------------------------------
    def _draw_config(self, y_offset=44):
        running_locked = (self.app_state == STATE_RUNNING)
        y0 = y_offset

        self._blit("ATS HYPERPARAMETER STUDIO", 16, y0, self.f_header, GOLD)
        if running_locked:
            self._blit("[LOCKED DURING EXECUTION — Press SPACE to Pause before editing]", 270, y0 + 2, self.f_small, RED)
        y0 += 28

        fields_left  = _CONFIG_FIELDS[:4]
        fields_right = _CONFIG_FIELDS[4:]

        for col_idx, fields in enumerate([fields_left, fields_right]):
            cx = 16 + col_idx * 515
            fy = y0
            for attr, label, step, lo, hi, fmt in fields:
                val = getattr(config_rl, attr, "?")
                locked = running_locked or (attr == "MIND_HIDDEN_SIZE")

                self._blit(f"{label}", cx, fy, self.f_body, WHITE if not locked else LOCK_FG)
                val_str = fmt.format(val) if val != "?" else "?"
                self._blit(val_str, cx + 185, fy, self.f_bold, FG_BOLD if not locked else LOCK_FG)

                bx = cx + 260
                if step is not None:
                    self._btn("[-]", bx, fy - 2, 36, 20, f"cfg:{attr}:dec", locked=locked)
                    self._btn("[+]", bx + 40, fy - 2, 36, 20, f"cfg:{attr}:inc", locked=locked)
                    bx += 88

                if attr == "LEARNING_RATE":
                    for lr in _LR_PRESETS:
                        ac = abs(val - lr) < lr * 0.01
                        self._btn(f"{lr:.0e}", bx, fy - 2, 50, 20, f"cfg:{attr}:{lr}", active=ac, locked=locked)
                        bx += 54
                elif attr == "ENTROPY_COEFF":
                    for ent in _ENTROPY_PRESETS:
                        ac = abs(val - ent) < 0.0001
                        self._btn(f"{ent:.3f}", bx, fy - 2, 56, 20, f"cfg:{attr}:{ent}", active=ac, locked=locked)
                        bx += 60
                elif attr == "MIND_HIDDEN_SIZE":
                    for hs in _HIDDEN_PRESETS:
                        ac = (val == hs)
                        self._btn(f"{hs}", bx, fy - 2, 46, 20, f"cfg:{attr}:{hs}", active=ac, locked=True)
                        bx += 50

                if attr == "MIND_HIDDEN_SIZE":
                    self._blit("(Locked to Checkpoint)", bx + 6, fy, self.f_tiny, LOCK_FG)
                fy += 38

        self._hline(y0 + 8 * 38)
        iy = y0 + 8 * 38 + 10
        self._blit(f"Active Checkpoint: {config_rl.MODEL_PATH}   |   CSV History: {self.csv_path}   |   Profile: {config_rl.VIRUS_PROFILE}", 16, iy, self.f_small, DIM, max_w=1000)

    # -----------------------------------------------------------------------
    # TAB 3: MIND & LLM INSPECTOR
    # -----------------------------------------------------------------------
    def _draw_mind(self, env, needs, trace):
        y0 = 44
        self._panel(16, y0, 490, 500, "ACTIVE NEED DETECTOR (4-D URGENCY VECTOR)")
        n_labels = [("Hunger", needs[0], FG_BOLD), ("Injury", needs[1], RED), ("Threat", needs[2], AMBER), ("Tool / Harvest", needs[3], CYAN)]
        ny = y0 + 34
        for lbl, val, col in n_labels:
            self._blit(f"{lbl:<16}: {val:.3f}", 30, ny, self.f_body, WHITE)
            bw = int(max(0.0, min(1.0, val)) * 260)
            pygame.draw.rect(self.screen, DIM2, (210, ny + 2, 260, 14), border_radius=2)
            pygame.draw.rect(self.screen, col, (210, ny + 2, bw, 14), border_radius=2)
            pygame.draw.rect(self.screen, FRAME, (210, ny + 2, 260, 14), 1, border_radius=2)
            ny += 28

        self._hline(ny + 4, x0=26, x1=495)
        ny += 14
        self._blit("SOLUTION LOOP STAGE TRACES", 30, ny, self.f_bold, GOLD)
        ny += 22
        for stage in ["detect", "need", "recall", "act", "record"]:
            val = str(trace.get(stage, "—"))
            self._blit(f"{stage:<8}: {val[:54]}", 30, ny, self.f_body, CYAN)
            ny += 20

        self._hline(ny + 4, x0=26, x1=495)
        ny += 14
        self._blit("CONTINUAL LEARNER MATRIX", 30, ny, self.f_bold, GOLD)
        ny += 22
        self._blit(f"Buffer Capacity : {config_rl.CONTINUAL_BUFFER_SIZE} transitions", 30, ny, self.f_body, WHITE)
        ny += 20
        self._blit(f"Update Interval : Every {config_rl.CONTINUAL_UPDATE_EVERY} ticks", 30, ny, self.f_body, WHITE)
        ny += 20
        self._blit(f"PPO Mini-Epochs : {config_rl.CONTINUAL_MINI_EPOCHS} epochs per batch", 30, ny, self.f_body, WHITE)

        # Right Panel: Virus Language Model
        self._panel(518, y0, 506, 500, "VIRUS LANGUAGE MODEL GROUNDING")
        ry = y0 + 34
        self._blit(f"Active Profile : {config_rl.VIRUS_PROFILE}", 534, ry, self.f_body, WHITE)
        ry += 24

        self._blit("Latest Generated Runtime Narration:", 534, ry, self.f_bold, GOLD)
        ry += 18
        lines = [self.last_narration[i:i + 52] for i in range(0, len(self.last_narration), 52)]
        for line in lines[:7]:
            self._blit(f"> {line}", 534, ry, self.f_body, WHITE)
            ry += 18

        self._hline(ry + 6, x0=526, x1=1014)
        ry += 16
        self._blit("Symbolic Vocabulary Token Grounding:", 534, ry, self.f_bold, GOLD)
        ry += 20
        tok_rows = [
            ("Items   :", "0001:Apple  0002:Sword  0010:Wood  0021:OakTree", FG),
            ("Entities:", "E001:Slime  E005:Spider  E015:Sheep  E018:Vill", AMBER),
            ("Biomes  :", "B000:Forest  B001:Desert  B003:Plains  B005:Tundra", CYAN),
            ("Actions :", "A000:MoveN  A007:Attack  A008:PickUp  A017:Wait", WHITE),
        ]
        for k, v, col in tok_rows:
            self._blit(k, 534, ry, self.f_body, DIM)
            self._blit(v, 625, ry, self.f_body, col)
            ry += 20

    # -----------------------------------------------------------------------
    # In-Session Bottom Control Toolbar
    # -----------------------------------------------------------------------
    def _draw_toolbar(self):
        self._hline(H - 66)
        ty = H - 54

        running = (self.app_state == STATE_RUNNING)
        paused  = (self.app_state == STATE_PAUSED)
        stopped = (self.app_state == STATE_STOPPED)
        in_run  = running or paused

        self._btn("[ PAUSE ]" if running else "[ RESUME ]", 16, ty, 115, 36, "toggle_pause", active=paused, locked=stopped, color=WHITE, bg_color=BTN_PAUSE_BG, border_color=BTN_PAUSE_BDR, font=self.f_bold)
        self._btn("[ TURBO ]", 140, ty, 98, 36, "turbo", active=self.turbo_mode, locked=not running, color=WHITE, bg_color=BTN_TURBO_BG, border_color=BTN_TURBO_BDR, font=self.f_bold)
        self._btn("[ SAVE ]", 248, ty, 96, 36, "save", locked=stopped, color=WHITE, bg_color=BTN_SAVE_BG, border_color=BTN_SAVE_BDR, font=self.f_bold)
        self._btn("[ RESET POLICY ]", 354, ty, 140, 36, "fresh_reset", locked=running, color=WHITE, bg_color=BTN_RESET_BG, border_color=BTN_RESET_BDR, font=self.f_bold)
        self._btn("[ STOP ]", 504, ty, 96, 36, "stop", locked=stopped, color=WHITE, bg_color=BTN_STOP_BG, border_color=BTN_STOP_BDR, font=self.f_bold)
        self._btn("[ MENU ]", 610, ty, 96, 36, "return_home", active=(self.app_state == STATE_STOPPED), locked=running, color=WHITE, bg_color=BTN_MENU_BG, border_color=BTN_MENU_BDR, font=self.f_bold)

        lock_note = " | [SPACE]=Pause/Resume | [T]=Turbo | [Ctrl+S]=Save" if in_run else ""
        self._blit(f"Tabs: Click [< TABS / VIEWS] on Top-Right{lock_note} | [ESC]=Pause | [W/A/S/D]=Actions", 16, H - 18, self.f_small, DIM)
