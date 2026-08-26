"""ATS Unified Mission Control & Live Dashboard — Retro Cyber Edition v2.5.

Features:
- Exclusively powered by '04b_03 regular' pixel font (bundled natively at assets/fonts/04b.ttf)
- Flashy Retro-Cyber Emblem Vector Logo (A + TS / A + GENT) with glowing circuit traces
- Layered 80s Cyber-Terminal Boot Intro with vector beam sweep and diagnostics
- 1040x680 expanded viewport with thicker 2px retro pixel frames and comfortable padding
- Solid vibrant action buttons: Red (Stop), Green (Save), Blue (Turbo), Amber (Pause)
- Top-Right Collapsible Navigation Sidebar Drawer ([ ◀ TABS / VIEWS ])
- Zoomable Timeframe Telemetry Graphs with presets and dynamic Y-axis scaling
- Ascending Episode History Table (oldest to newest at bottom) with mouse wheel scrolling
- Automatic CSV export to data/ats_episode_history.csv
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
# High-Contrast Retro CRT Phosphor Palette
# ---------------------------------------------------------------------------
BG          = (8,  12,  10)       # Deep CRT background
BG_PANEL    = (14, 22,  18)       # Card panel background
BG_PANEL2   = (18, 30,  24)       # Elevated panel
FG          = (45, 255, 125)     # Primary phosphor green
FG_BOLD     = (140, 255, 195)    # Bright highlight green
DIM         = (28, 115,  65)      # Secondary / border green
DIM2        = (16,  60,  35)      # Muted grid lines
AMBER       = (255, 195, 50)     # Alert / warning amber
RED         = (255, 85,  75)      # Danger red
CYAN        = (85, 235, 255)     # Telemetry cyan
FRAME       = (36,  85,  58)      # Panel bevel frame (thickened 2px)
WHITE       = (240, 250, 245)    # Clean readable white
GOLD        = (255, 225, 95)     # Section header gold
TITLE_GREEN = (0,  235, 120)     # Brand logo green

# Solid Button Colors (High-Contrast & Vibrant)
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
BTN_MENU_BG    = (40,  60,  50)
BTN_MENU_BDR   = (75,  115, 95)

BTN_IDLE_BG    = (20,  38,  28)
BTN_HOVER_BG   = (35,  70,  50)
BTN_ACTIVE_BG  = (50,  110, 75)
BTN_LOCKED_BG  = (16,  24,  20)
LOCK_FG        = (45,  75,  60)

WINDOW_WIDTH  = 1040
WINDOW_HEIGHT = 680
W, H = WINDOW_WIDTH, WINDOW_HEIGHT

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
    STATE_MENU:    {TAB_CONFIG, TAB_ANALYTICS},
    STATE_RUNNING: {TAB_SIM, TAB_ANALYTICS, TAB_MIND},
    STATE_PAUSED:  {TAB_SIM, TAB_ANALYTICS, TAB_CONFIG, TAB_MIND},
    STATE_STOPPED: {TAB_ANALYTICS},
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

        # Load EXCLUSIVELY the '04b_03 regular' pixel font from the project assets
        self.f_title  = self._load_pixel_font(18)
        self.f_header = self._load_pixel_font(14)
        self.f_body   = self._load_pixel_font(12)
        self.f_bold   = self._load_pixel_font(12)
        self.f_small  = self._load_pixel_font(10)
        self.f_tiny   = self._load_pixel_font(9)

        # Application state
        self.alive         = True
        self.frame         = 0
        self.app_state     = STATE_MENU
        self.active_tab    = TAB_CONFIG
        self.sidebar_open  = False

        # Interactive Signals
        self.sig_start        = False
        self.sig_pause_toggle = False
        self.sig_save         = False
        self.sig_fresh_reset  = False
        self.sig_stop         = False
        self.sig_return_home  = False
        self.turbo_mode       = False

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

    def _load_pixel_font(self, size):
        """Loads exclusively '04b_03 regular' font from assets/fonts/04b.ttf."""
        candidates = [
            "assets/fonts/04b.ttf",           # Exact 04b_03 regular bundled in repo
            "assets/fonts/04B_03_regular.ttf",
            "assets/fonts/04B_03.TTF",
            r"C:\Windows\Fonts\04b.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass
        # Default safety fallback solely if file is missing
        return pygame.font.SysFont("04b_03", size)

    def clear_session_data(self):
        """Resets in-memory table and graphs for a brand-new run."""
        self.episode_history.clear()
        self.reward_curve.clear()
        self.loss_history.clear()
        self.table_scroll = 0
        self.pan_offset = 0

    def record_episode(self, ep, reward, ticks, reason):
        entry = {
            "ep": ep,
            "reward": reward,
            "ticks": ticks,
            "reason": reason,
            "time": time.strftime("%H:%M:%S")
        }
        self.episode_history.append(entry)
        self.reward_curve.append(reward)
        if self.table_scroll > 0:
            self.table_scroll = 0
        self.export_csv()

    def record_loss(self, total, actor, critic):
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
                        "episode", "timestamp", "total_reward", "ticks_survived",
                        "termination_reason", "speed_multiplier", "learning_rate",
                        "entropy_coeff", "hidden_size"
                    ])
                if self.episode_history:
                    latest = self.episode_history[-1]
                    writer.writerow([
                        latest["ep"], latest["time"], f"{latest['reward']:.2f}",
                        latest["ticks"], latest["reason"], config_rl.SPEED_MULTIPLIER,
                        config_rl.LEARNING_RATE, config_rl.ENTROPY_COEFF, config_rl.MIND_HIDDEN_SIZE
                    ])
        except Exception as e:
            print(f"[ATS GUI] CSV export note: {e}")

    def close(self):
        self.alive = False
        pygame.quit()

    # -----------------------------------------------------------------------
    # Flashy Vector Cyber Emblem Renderer (A + TS / A + GENT)
    # -----------------------------------------------------------------------
    def _draw_ats_cyber_logo(self, cx, cy, scale=1.0, progress=1.0, pulse=0.0, show_badges=True):
        """Renders a flashy, high-impact cyberpunk vector emblem for ATS."""
        # Key coordinate anchors
        h = int(52 * scale)
        w = int(58 * scale)
        
        p_apex       = (cx, cy - h)
        p_left_base  = (cx - w, cy + h)
        p_right_base = (cx + w, cy + h)
        p_l_inner    = (cx - int(36 * scale), cy + h)
        p_r_inner    = (cx + int(36 * scale), cy + h)
        
        # Mid tier connectors (TS tier)
        mid_y = cy - int(4 * scale)
        p_mid_l = (cx - int(24 * scale), mid_y)
        p_mid_r = (cx + int(24 * scale), mid_y)
        
        # Low tier connectors (GENT tier)
        low_y = cy + int(26 * scale)
        p_low_l = (cx - int(40 * scale), low_y)
        p_low_r = (cx + int(40 * scale), low_y)

        # Pulse colors
        fg_neon = (int(45 + 30 * math.sin(pulse * 3)), 255, int(125 + 30 * math.sin(pulse * 3)))
        gold_neon = (255, int(210 + 45 * math.sin(pulse * 4)), 95)
        cyan_neon = (int(75 + 40 * math.sin(pulse * 4)), 235, 255)
        glow_plate = (12, 38, 22)

        # 1. Background Backplate Polygon
        if progress >= 0.6:
            poly_pts = [p_apex, p_right_base, p_r_inner, p_mid_r, p_mid_l, p_l_inner, p_left_base]
            pygame.draw.polygon(self.screen, glow_plate, poly_pts)
            pygame.draw.polygon(self.screen, FRAME, poly_pts, 1)

        # 2. Outer Pyramid Beams
        if progress >= 0.2:
            sub_p = min(1.0, (progress - 0.2) / 0.4)
            # Left beam
            cur_l = (cx - int(w * sub_p), cy - h + int(2 * h * sub_p))
            pygame.draw.line(self.screen, fg_neon, p_apex, cur_l, 3)
            # Right beam
            cur_r = (cx + int(w * sub_p), cy - h + int(2 * h * sub_p))
            pygame.draw.line(self.screen, fg_neon, p_apex, cur_r, 3)

        # 3. Feet and Inner Chevron
        if progress >= 0.5:
            pygame.draw.line(self.screen, fg_neon, p_left_base, p_l_inner, 3)
            pygame.draw.line(self.screen, fg_neon, p_right_base, p_r_inner, 3)
            
            # Inner Legs & Chevron
            ch_apex = (cx, cy - int(24 * scale))
            pygame.draw.line(self.screen, fg_neon, p_l_inner, (cx - int(15 * scale), cy + int(8 * scale)), 2)
            pygame.draw.line(self.screen, fg_neon, p_r_inner, (cx + int(15 * scale), cy + int(8 * scale)), 2)
            pygame.draw.line(self.screen, fg_neon, ch_apex, (cx - int(15 * scale), cy + int(8 * scale)), 2)
            pygame.draw.line(self.screen, fg_neon, ch_apex, (cx + int(15 * scale), cy + int(8 * scale)), 2)

        # 4. Upper Crossbar & TS Circuit Extension
        if progress >= 0.7:
            ts_ext_x = cx + int(85 * scale)
            pygame.draw.line(self.screen, gold_neon, p_mid_l, (ts_ext_x, mid_y), 2)
            # Node crystal
            pygame.draw.circle(self.screen, gold_neon, (ts_ext_x, mid_y), int(3 * scale))
            pygame.draw.circle(self.screen, fg_neon, p_apex, int(4 * scale))

        # 5. Lower Cyber-Bridge & GENT Circuit Extension
        if progress >= 0.85:
            gent_ext_x = cx + int(85 * scale)
            pygame.draw.line(self.screen, cyan_neon, p_low_l, (gent_ext_x, low_y), 2)
            pygame.draw.circle(self.screen, cyan_neon, (gent_ext_x, low_y), int(3 * scale))

        # 6. Text Badges
        if show_badges and progress >= 0.9:
            # [A] TS Badge
            badge_x = cx + int(96 * scale)
            self.screen.blit(self.f_header.render("[A] TS", True, gold_neon), (badge_x, mid_y - 8))
            self.screen.blit(self.f_small.render(":: Autonomous Training Simulation", True, WHITE), (badge_x + 72, mid_y - 6))

            # [A] GENT Badge
            self.screen.blit(self.f_header.render("[A] GENT", True, cyan_neon), (badge_x, low_y - 8))
            self.screen.blit(self.f_small.render(":: Deep RL Mission Control Matrix v2.5", True, FG_BOLD), (badge_x + 86, low_y - 6))

    # -----------------------------------------------------------------------
    # 80s Cyber-Terminal Layered Boot Intro
    # -----------------------------------------------------------------------
    def splash(self, duration=4.8):
        """Authentic 80s retro cyber-terminal boot with vector wireframe build & diagnostics."""
        boot_logs = [
            ("BIOS ROM INTEGRITY CHECK", "OK (640KB VIRTUAL)"),
            ("PPO NEURAL POLICY ENGINE", f"ONLINE (HIDDEN={config_rl.MIND_HIDDEN_SIZE})"),
            ("PERCEPTION STATE SPACE", f"INDEXED ({config_rl.STATE_SIZE} DIMS)"),
            ("ACTION MASK PERMUTATIONS", f"LOADED ({config_rl.ACTION_SIZE} ACTIONS)"),
            ("PROCEDURAL OPEN WORLD", "7 BIOMES (555+ OBJECTS)"),
            ("CONTINUAL REPLAY BUFFER", f"READY ({config_rl.CONTINUAL_BUFFER_SIZE} ROLLS)"),
            ("SYMBOLIC VOCABULARY ENGINE", "BOUND (420 TOKENS)"),
            ("CRT PHOSPHOR RASTER", f"SYNCHRONIZED ({W}x{H})"),
        ]

        start_t = time.time()
        while self.alive and time.time() - start_t < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.alive = False
                    return
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_ESCAPE):
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    return

            elapsed = time.time() - start_t
            self.screen.fill(BG)
            pygame.draw.rect(self.screen, FRAME, (4, 4, W - 8, H - 8), 2)

            # Stage 1: CRT Raster Beam Sweep (0.0s -> 0.7s)
            if elapsed < 0.7:
                beam_h = int((elapsed / 0.7) * (H - 30))
                beam_y = (H // 2) - (beam_h // 2)
                pygame.draw.rect(self.screen, (15, 45, 25), (6, beam_y, W - 12, beam_h))
                pygame.draw.line(self.screen, FG_BOLD, (6, H // 2), (W - 6, H // 2), 2)
                for sy in range(beam_y, beam_y + beam_h, 8):
                    pygame.draw.line(self.screen, (20, 60, 35), (6, sy), (W - 6, sy), 1)
                pygame.display.flip()
                self.clock.tick(60)
                continue

            # Stage 2: Layered Cyber Emblem Wireframe Build (0.7s -> 2.6s)
            wire_progress = min(1.0, (elapsed - 0.7) / 1.9)
            logo_cx = W // 2 - 180
            logo_cy = 92
            self._draw_ats_cyber_logo(logo_cx, logo_cy, scale=1.15, progress=wire_progress, pulse=elapsed, show_badges=(elapsed >= 2.0))

            pygame.draw.line(self.screen, DIM, (24, 175), (W - 24, 175), 1)

            # Stage 3: Subsystem Kernel Diagnostics (2.6s -> duration)
            if elapsed >= 2.4:
                diag_progress = min(1.0, (elapsed - 2.4) / (duration - 2.4 - 0.3))
                num_diag_shown = min(len(boot_logs), int(diag_progress * len(boot_logs)) + 1)

                log_x = W // 2 - 290
                log_y = 190
                for i in range(num_diag_shown):
                    task, status = boot_logs[i]
                    dots = "." * (48 - len(task))
                    line_text = f">>> BOOT: {task} {dots} "
                    self.screen.blit(self.f_body.render(line_text, True, FG), (log_x, log_y))
                    self.screen.blit(self.f_bold.render(f"[{status}]", True, GOLD if "ONLINE" in status or "OK" in status else CYAN), (log_x + 480, log_y))
                    log_y += 24

                # Bottom Kernel Progress Bar
                pb_x, pb_y, pb_w, pb_h = W // 2 - 290, H - 95, 580, 18
                pygame.draw.rect(self.screen, BG_PANEL, (pb_x, pb_y, pb_w, pb_h))
                pygame.draw.rect(self.screen, FRAME, (pb_x, pb_y, pb_w, pb_h), 1)
                fill_w = int(pb_w * diag_progress)
                pygame.draw.rect(self.screen, FG, (pb_x + 1, pb_y + 1, fill_w - 2 if fill_w > 2 else 0, pb_h - 2))
                self.screen.blit(self.f_small.render(f"INITIALIZING KERNEL: {int(diag_progress * 100)}%", True, WHITE), (pb_x + 210, pb_y + 3))

                prompt_txt = "MISSION CONTROL READY. [Press SPACE or Click to Enter]"
                self.screen.blit(self.f_small.render(prompt_txt, True, GOLD if int(elapsed * 5) % 2 == 0 else DIM), (pb_x + 105, H - 60))

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
        pygame.draw.rect(self.screen, FRAME, (4, 4, W - 8, H - 8), 2)

        if self.app_state == STATE_MENU:
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
                elif event.button == 4:  # Mouse wheel UP
                    if self.active_tab == TAB_ANALYTICS:
                        self.table_scroll = min(max(0, len(self.episode_history) - 6), self.table_scroll + 1)
                elif event.button == 5:  # Mouse wheel DOWN
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
        elif cb == "fresh_reset":
            self.sig_fresh_reset = True
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
    # Sharp Pixel UI Helpers (border_radius=0)
    # -----------------------------------------------------------------------
    def _blit(self, text, x, y, font=None, color=FG, max_w=None):
        font = font or self.f_body
        surf = font.render(text, True, color)
        if max_w and surf.get_width() > max_w:
            while len(text) > 1 and font.size(text + "..")[0] > max_w:
                text = text[:-1]
            text += ".."
            surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _hline(self, y, x0=10, x1=None):
        pygame.draw.line(self.screen, DIM, (x0, y), (x1 or W - 10, y), 1)

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
            bdr = GOLD if active else (FG_BOLD if hover else DIM)
            tc = GOLD if active else (WHITE if hover else color)

        pygame.draw.rect(self.screen, bg, rect)
        pygame.draw.rect(self.screen, bdr, rect, 1)
        surf = font.render(text, True, tc)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _panel(self, x, y, w, h, title=None):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, BG_PANEL, rect)
        pygame.draw.rect(self.screen, FRAME, rect, 2)
        if title:
            self._blit(title, x + 10, y + 8, self.f_bold, GOLD)
        return rect

    # -----------------------------------------------------------------------
    # 🏠 MAIN MENU / HOME SCREEN with Flashy Cyber Emblem
    # -----------------------------------------------------------------------
    def _draw_menu(self):
        # 1. Flashy Vector Cyber Emblem at Top Header
        self._draw_ats_cyber_logo(cx=95, cy=68, scale=0.92, progress=1.0, pulse=self._pulse, show_badges=True)

        # Right-side Header Metadata & Status
        self._blit("[A] TS / [A] GENT MISSION CONTROL v2.5", 350, 16, self.f_title, GOLD)
        self._blit("Deep Reinforcement Learning & Cognitive Architecture Simulation", 350, 40, self.f_body, WHITE)
        self._blit("Status: READY FOR OPERATOR LAUNCH   |   Active Checkpoint: " + config_rl.MODEL_PATH.name, 350, 62, self.f_small, FG)
        self._blit("Cognitive PPO Engine • Procedural Open World • 04B_03 Native Pixel Typography", 350, 80, self.f_tiny, DIM)

        self._hline(136)

        # 2. Main Content Grid (Two side-by-side pixel panels)
        card_y = 144
        card_h = 430
        card_w = 490

        # LEFT PANEL: PRE-LAUNCH HYPERPARAMETERS
        self._panel(20, card_y, card_w, card_h, "1. PRE-LAUNCH HYPERPARAMETERS")
        row = card_y + 34

        params = [
            ("EPISODES",         "Target Episodes",     10,  "{:d}"),
            ("MAX_TICKS",        "Max Ticks / Ep",     500,  "{:d}"),
            ("SPEED_MULTIPLIER", "Sim Speed",          2.0,  "{:.1f}x"),
            ("DAY_LENGTH_TICKS", "Day Length",         100,  "{:d} t"),
        ]
        for attr, label, step, fmt in params:
            val = getattr(config_rl, attr)
            self._blit(f"{label:<16}", 34, row, self.f_body, WHITE)
            self._blit(fmt.format(val), 195, row, self.f_bold, FG_BOLD)
            self._btn("[-]", 290, row - 2, 38, 20, f"cfg:{attr}:dec")
            self._btn("[+]", 335, row - 2, 38, 20, f"cfg:{attr}:inc")
            row += 30

        # Learning Rate Row
        self._blit("Learning Rate", 34, row, self.f_body, WHITE)
        self._blit(f"{config_rl.LEARNING_RATE:.2e}", 195, row, self.f_bold, FG_BOLD)
        row += 24
        lr_x = 34
        for lr in _LR_PRESETS:
            active = abs(config_rl.LEARNING_RATE - lr) < lr * 0.01
            self._btn(f"{lr:.0e}", lr_x, row - 2, 64, 20, f"cfg:LEARNING_RATE:{lr}", active=active)
            lr_x += 70
        row += 30

        # Entropy Coeff Row
        self._blit("Entropy Coeff", 34, row, self.f_body, WHITE)
        self._blit(f"{config_rl.ENTROPY_COEFF:.3f}", 195, row, self.f_bold, FG_BOLD)
        row += 24
        ent_x = 34
        for ent in _ENTROPY_PRESETS:
            active = abs(config_rl.ENTROPY_COEFF - ent) < 0.0001
            self._btn(f"{ent:.3f}", ent_x, row - 2, 64, 20, f"cfg:ENTROPY_COEFF:{ent}", active=active)
            ent_x += 70
        row += 30

        # Hidden Width
        self._blit("Hidden Width", 34, row, self.f_body, WHITE)
        self._blit(f"{config_rl.MIND_HIDDEN_SIZE} units", 195, row, self.f_bold, FG_BOLD)
        self._blit("(Locked to Checkpoint)", 290, row, self.f_tiny, DIM)
        row += 30

        # Checkpoint Status
        ckpt_exists = config_rl.MODEL_PATH.exists()
        ckpt_col = FG_BOLD if ckpt_exists else AMBER
        ckpt_msg = f"Checkpoint: {config_rl.MODEL_PATH.name} (Synchronized)" if ckpt_exists else "Checkpoint: Fresh Random Weights"
        self._blit(ckpt_msg, 34, row, self.f_bold, ckpt_col)

        # RIGHT PANEL: AGENT & ENVIRONMENT MATRIX
        rx = 530
        self._panel(rx, card_y, card_w, card_h, "2. AGENT & ENVIRONMENT MATRIX")
        info_row = card_y + 34

        specs = [
            ("Policy Architecture", f"{config_rl.MIND_HIDDEN_SIZE}-Hidden 2-Layer Actor-Critic"),
            ("State Dimensions",   f"{config_rl.STATE_SIZE} Feature Dimensions"),
            ("Action Space",       f"{config_rl.ACTION_SIZE} Action Permutations"),
            ("Replay Memory",      f"{config_rl.CONTINUAL_BUFFER_SIZE} Rolls ({config_rl.CONTINUAL_UPDATE_EVERY}t PPO)"),
            ("Procedural Map",     "7 Biomes (555+ Objects, Smooth Terrain)"),
            ("Virus LM Profile",   f"Profile '{config_rl.VIRUS_PROFILE}' (420 Tokens)"),
        ]
        for k, v in specs:
            self._blit(f"{k:<20}", rx + 14, info_row, self.f_body, DIM)
            self._blit(v, rx + 185, info_row, self.f_body, WHITE)
            info_row += 24

        info_row += 8
        pygame.draw.line(self.screen, DIM, (rx + 10, info_row), (rx + card_w - 10, info_row), 1)
        info_row += 14

        self._blit("ACTIVE REWARD RULES (Click to Toggle):", rx + 14, info_row, self.f_bold, GOLD)
        info_row += 22

        for idx, (label, key) in enumerate(_RULE_LABELS):
            en = self.reward_rules.get(key, True)
            col_pos = idx % 3
            row_pos = idx // 3
            bx = rx + 14 + col_pos * 152
            by = info_row + row_pos * 28
            self._btn(f"[{'X' if en else ' '}] {label}", bx, by, 144, 22, f"rule:{key}", active=en, color=AMBER)

        # 3. Bottom Launch & Navigation Section
        self._hline(586)
        nav_y = 596

        self._btn("2: 📊 HISTORICAL ANALYTICS", 20, nav_y, 230, 36, "tab:1", active=(self.active_tab == TAB_ANALYTICS))
        self._btn("3: ⚙️ FULL HYPERPARAM STUDIO", 260, nav_y, 230, 36, "tab:2", active=(self.active_tab == TAB_CONFIG))

        # Main Launch Button
        self._btn("▶  LAUNCH TRAINING SESSION", 510, nav_y, 510, 38, "start", active=True, color=WHITE, bg_color=BTN_SAVE_BG, border_color=BTN_SAVE_BDR, font=self.f_header)

        self._blit("Hotkeys: [SPACE / Click] = Launch Session  |  [ESC] = Quit ATS", 20, H - 16, self.f_small, DIM)

    # -----------------------------------------------------------------------
    # In-Session Navigation Header & Top-Right Collapsible Sidebar Trigger
    # -----------------------------------------------------------------------
    def _draw_chrome(self, env):
        st_col = {STATE_RUNNING: FG_BOLD, STATE_PAUSED: AMBER, STATE_STOPPED: RED}.get(self.app_state, WHITE)
        cursor = "_" if (self.frame // 20) % 2 == 0 else " "
        ep_str = f"  EP:{env.episode}/{config_rl.EPISODES}  TICK:{env.tick}/{config_rl.MAX_TICKS}" if env else ""
        turbo_str = "  [⚡TURBO]" if self.turbo_mode else ""
        self._blit(f"ATS MISSION CONTROL {cursor} [{self.app_state}]{turbo_str}{ep_str}", 14, 10, self.f_bold, st_col)

        # Top-Right Collapsible Sidebar Trigger Button
        drawer_txt = "[ ✕ CLOSE ]" if self.sidebar_open else "[ ◀ TABS / VIEWS ]"
        self._btn(drawer_txt, W - 155, 6, 142, 26, "toggle_sidebar", active=self.sidebar_open, color=GOLD, font=self.f_bold)
        self._hline(36)

    def _draw_sidebar_drawer(self):
        """Renders the top-right slide-out navigation sidebar drawer."""
        dw = 230
        dx = W - dw - 8
        dy = 38
        dh = H - 106

        pygame.draw.rect(self.screen, BG_PANEL2, (dx, dy, dw, dh))
        pygame.draw.rect(self.screen, GOLD, (dx, dy, dw, dh), 2)

        self._blit("=== NAVIGATION ===", dx + 36, dy + 12, self.f_bold, GOLD)
        pygame.draw.line(self.screen, DIM, (dx + 10, dy + 32), (dx + dw - 10, dy + 32), 1)

        tabs = [
            (TAB_SIM,       "1: 🎮 LIVE SIM"),
            (TAB_ANALYTICS, "2: 📊 ANALYTICS"),
            (TAB_CONFIG,    "3: ⚙️ HYPERPARAMS"),
            (TAB_MIND,      "4: 🧠 MIND & LLM"),
        ]
        avail = TAB_ACCESS.get(self.app_state, set())
        ty = dy + 42
        for tab_id, label in tabs:
            locked = tab_id not in avail
            is_active = (self.active_tab == tab_id)
            self._btn(label, dx + 12, ty, dw - 24, 30, f"tab:{tab_id}", active=is_active, locked=locked, font=self.f_body)
            ty += 40

        pygame.draw.line(self.screen, DIM, (dx + 10, ty + 10), (dx + dw - 10, ty + 10), 1)
        self._btn("[ ✕ CLOSE DRAWER ]", dx + 12, ty + 24, dw - 24, 26, "close_sidebar", color=AMBER)

    # -----------------------------------------------------------------------
    # TAB 0: LIVE SIMULATION
    # -----------------------------------------------------------------------
    def _draw_sim(self, env, needs, trace):
        agent = env.agent
        world = env.world
        y0 = 42

        self._blit("INVENTORY (24 SLOTS)", 14, y0, self.f_bold, GOLD)
        facing_lbl = {"N": "NORTH ^", "S": "SOUTH v", "W": "WEST <", "E": "EAST >"}.get(agent.facing_dir, "?")
        self._blit(f"WORLD VIEW (7x7)    FACING: {facing_lbl}", 520, y0, self.f_bold, GOLD)
        pygame.draw.line(self.screen, DIM, (510, 38), (510, 268), 1)

        inv_y = y0 + 18
        for i in range(12):
            yp = inv_y + i * 18
            for side, base in ((0, 0), (1, 12)):
                sl = agent.inventory[i + base]
                sel = (i + base == agent.selected_slot)
                mk = ">" if sel else " "
                if sl["id"]:
                    txt = f"{mk}{i + base:02d}:[{sl['id']}]{item_name(sl['id'])[:8]} x{sl['count']}"
                    c = FG_BOLD if sel else FG
                else:
                    txt = f"{mk}{i + base:02d}:empty"
                    c = GOLD if sel else DIM
                self._blit(txt, 14 + side * 245, yp, self.f_body, c)

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
                    else:                  cs, cc = ".", DIM
                self._blit(cs, cx, ry, self.f_body, cc)

        self._blit(f"TARGET: {agent.get_facing_tile_info(world)}", 520, inv_y + 7 * ch + 4, self.f_body, CYAN)
        nc = RED if any(n > 0.5 for n in needs[:3]) else FG
        self._blit(f"NEEDS  Hunger:{needs[0]:.2f}  Injury:{needs[1]:.2f}  Threat:{needs[2]:.2f}  Tool:{needs[3]:.2f}", 520, inv_y + 7 * ch + 24, self.f_body, nc)

        # Vitals Bar
        self._hline(272)
        from biome import BIOMES
        ct = world._tile(agent.x, agent.y)
        b_name = BIOMES.get(ct.biome if ct else 0, "?")
        hp_c = FG_BOLD if agent.health > 50 else (AMBER if agent.health > 25 else RED)
        self._blit(f"HP:{agent.health:3d}/100  HUNGER:{agent.hunger:3d}/100  STAMINA:{agent.stamina:3d}/100  LV:{agent.level}  XP:{agent.xp}  POS:({agent.x},{agent.y}) Z:{agent.z}", 14, 276, self.f_body, hp_c)
        nt = "NIGHT" if env.day_night.is_night else "DAY"
        self._blit(f"BIOME:{b_name}  TIME:{env.day_night.time_of_day:.2f} ({nt})  TORCH:{'ON' if agent.torch_active else 'OFF'}  INJURED:{'YES' if agent.leg_injured else 'NO'}", 14, 296, self.f_body, AMBER if env.day_night.is_night else CYAN)

        # Recent Activity
        self._hline(318)
        self._blit("RECENT ACTIVITY & EVENTS", 14, 322, self.f_bold, GOLD)
        evts = self._fmt_events(agent.event_log, 2)
        for i, (txt, col) in enumerate(evts):
            self._blit(txt, 14, 342 + i * 18, self.f_body, col)

        # Mind Traces
        self._hline(384)
        self._blit("MIND COGNITIVE CYCLE", 14, 388, self.f_bold, GOLD)
        self._blit(f"Detect: {str(trace.get('detect', ''))[:100]}", 14, 408, self.f_body, CYAN)
        self._blit(f"Act   : {str(trace.get('act', ''))[:100]}", 14, 428, self.f_body, CYAN)

        # Action Toggles
        self._hline(450)
        self._blit("ACTION TOGGLES (Click or W/A/S/D to toggle):", 14, 454, self.f_small, DIM)
        bx = 14
        for label, idx in _ACTION_TOGGLES:
            en = idx not in self.disabled_actions
            self._btn(f"[{'X' if en else ' '}] {label}", bx, 470, 68, 22, f"act:{idx}", active=en)
            bx += 72

        # Reward Rules
        self._blit("REWARD RULES:", 14, 498, self.f_small, DIM)
        bx = 14
        for label, key in _RULE_LABELS:
            en = self.reward_rules.get(key, True)
            self._btn(f"[{'X' if en else ' '}] {label}", bx, 514, 85, 22, f"rule:{key}", active=en, color=AMBER)
            bx += 89

        # Stats Line
        self._hline(542)
        ent, _ = world.nearest_entity(agent.x, agent.y)
        ent_lbl = f"{entity_name(ent.entity_id)}(HP:{ent.hp})" if ent else "none"
        self._blit(f"LAST ACTION: {agent.last_action_name}   |   NEAREST: {ent_lbl}   |   TOTAL REWARD: {env.rewards.total:.2f}   |   LAST Δ: {env.rewards.tick_reward:+.2f}", 14, 546, self.f_body, FG_BOLD, max_w=1000)

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
    def _draw_analytics(self, y_offset=42):
        y0 = y_offset

        cards = [
            ("Total Episodes", str(len(self.episode_history))),
            ("Best Reward", f"{max(self.reward_curve):.2f}" if self.reward_curve else "—"),
            ("Latest Reward", f"{self.reward_curve[-1]:.2f}" if self.reward_curve else "—"),
            ("Avg Last 10", f"{sum(self.reward_curve[-10:]) / len(self.reward_curve[-10:]):.2f}" if len(self.reward_curve) >= 2 else "—"),
        ]
        cx = 14
        for title, val in cards:
            self._panel(cx, y0, 240, 50, None)
            self._blit(title, cx + 10, y0 + 6, self.f_small, DIM)
            self._blit(val, cx + 10, y0 + 22, self.f_header, FG_BOLD)
            cx += 254
        y0 += 58

        # Timeframe & Zoom Controls Bar
        self._blit("TIMEFRAME / ZOOM:", 14, y0 + 3, self.f_bold, GOLD)
        zx = 160
        presets = [("10 Ep", "10"), ("25 Ep", "25"), ("50 Ep", "50"), ("ALL", "all")]
        for plabel, pval in presets:
            is_active = (str(self.zoom_window) == pval or (pval == "all" and self.zoom_window == 0))
            self._btn(plabel, zx, y0 - 2, 58, 22, f"zoom:{pval}", active=is_active)
            zx += 64

        self._btn("🔍- OUT", zx + 10, y0 - 2, 65, 22, "zoom:out")
        self._btn("🔍+ IN", zx + 80, y0 - 2, 65, 22, "zoom:in")
        self._btn("📄 EXPORT CSV", W - 155, y0 - 2, 142, 22, "export_csv", color=GOLD)
        y0 += 30

        # Two Zoomable Telemetry Graphs
        lrect = pygame.Rect(14, y0, 490, 165)
        rrect = pygame.Rect(518, y0, 508, 165)
        pygame.draw.rect(self.screen, BG_PANEL, lrect)
        pygame.draw.rect(self.screen, FRAME, lrect, 2)
        pygame.draw.rect(self.screen, BG_PANEL, rrect)
        pygame.draw.rect(self.screen, FRAME, rrect, 2)

        win_lbl = f"(Window: Last {self.zoom_window})" if self.zoom_window > 0 else "(Window: All Time)"
        self._blit(f"EPISODE REWARD CURVE {win_lbl}", lrect.x + 8, lrect.y + 6, self.f_small, GOLD)
        self._blit("PPO LOSS HISTORY (Total / Actor / Critic)", rrect.x + 8, rrect.y + 6, self.f_small, CYAN)

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
        self._blit("EPISODE SUMMARY LOG (Ascending Order — Latest at Bottom):", 14, y0 + 3, self.f_bold, GOLD)
        self._btn("▲ SCROLL UP", W - 255, y0 - 2, 115, 22, "scroll:up")
        self._btn("▼ SCROLL DOWN", W - 135, y0 - 2, 122, 22, "scroll:down")
        y0 += 26

        headers = [("EPISODE", 14), ("TOTAL REWARD", 110), ("TICKS SURVIVED", 240), ("TERMINATION REASON", 390), ("TIME", 880)]
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
            self._blit(f"Ep {entry['ep']:04d}", 14, y0, self.f_body, FG)
            self._blit(f"{entry['reward']:+.2f}", 110, y0, self.f_body, rc)
            self._blit(f"{entry['ticks']:5d}", 240, y0, self.f_body, WHITE)
            rc2 = RED if "Perished" in entry["reason"] or "HP" in entry["reason"] else CYAN
            self._blit(entry["reason"], 390, y0, self.f_body, rc2, max_w=470)
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
    def _draw_config(self, y_offset=42):
        running_locked = (self.app_state == STATE_RUNNING)
        y0 = y_offset

        self._blit("ATS HYPERPARAMETER STUDIO", 14, y0, self.f_header, GOLD)
        if running_locked:
            self._blit("[LOCKED DURING EXECUTION — Press SPACE to Pause before editing]", 270, y0 + 2, self.f_small, RED)
        y0 += 28

        fields_left  = _CONFIG_FIELDS[:4]
        fields_right = _CONFIG_FIELDS[4:]

        for col_idx, fields in enumerate([fields_left, fields_right]):
            cx = 14 + col_idx * 515
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
        self._blit(f"Active Checkpoint: {config_rl.MODEL_PATH}   |   CSV History: {self.csv_path}   |   Profile: {config_rl.VIRUS_PROFILE}", 14, iy, self.f_small, DIM, max_w=1000)

    # -----------------------------------------------------------------------
    # TAB 3: MIND & LLM INSPECTOR
    # -----------------------------------------------------------------------
    def _draw_mind(self, env, needs, trace):
        y0 = 42
        self._panel(14, y0, 490, 500, "ACTIVE NEED DETECTOR (4-D URGENCY VECTOR)")
        n_labels = [("Hunger", needs[0], FG_BOLD), ("Injury", needs[1], RED), ("Threat", needs[2], AMBER), ("Tool / Harvest", needs[3], CYAN)]
        ny = y0 + 32
        for lbl, val, col in n_labels:
            self._blit(f"{lbl:<16}: {val:.3f}", 28, ny, self.f_body, WHITE)
            bw = int(max(0.0, min(1.0, val)) * 280)
            pygame.draw.rect(self.screen, DIM2, (200, ny + 2, 280, 14))
            pygame.draw.rect(self.screen, col, (200, ny + 2, bw, 14))
            pygame.draw.rect(self.screen, DIM, (200, ny + 2, 280, 14), 1)
            ny += 28

        self._hline(ny + 4, x0=24, x1=495)
        ny += 14
        self._blit("SOLUTION LOOP STAGE TRACES", 28, ny, self.f_bold, GOLD)
        ny += 22
        for stage in ["detect", "need", "recall", "act", "record"]:
            val = str(trace.get(stage, "—"))
            self._blit(f"{stage:<8}: {val[:58]}", 28, ny, self.f_body, CYAN)
            ny += 20

        self._hline(ny + 4, x0=24, x1=495)
        ny += 14
        self._blit("CONTINUAL LEARNER MATRIX", 28, ny, self.f_bold, GOLD)
        ny += 22
        self._blit(f"Buffer Capacity : {config_rl.CONTINUAL_BUFFER_SIZE} transitions", 28, ny, self.f_body, WHITE)
        ny += 20
        self._blit(f"Update Interval : Every {config_rl.CONTINUAL_UPDATE_EVERY} ticks", 28, ny, self.f_body, WHITE)
        ny += 20
        self._blit(f"PPO Mini-Epochs : {config_rl.CONTINUAL_MINI_EPOCHS} epochs per batch", 28, ny, self.f_body, WHITE)

        # Right Panel: Virus Language Model
        self._panel(518, y0, 508, 500, "VIRUS LANGUAGE MODEL GROUNDING")
        ry = y0 + 32
        self._blit(f"Active Profile : {config_rl.VIRUS_PROFILE}", 532, ry, self.f_body, WHITE)
        ry += 24

        self._blit("Latest Generated Runtime Narration:", 532, ry, self.f_bold, GOLD)
        ry += 18
        lines = [self.last_narration[i:i + 56] for i in range(0, len(self.last_narration), 56)]
        for line in lines[:7]:
            self._blit(f"> {line}", 532, ry, self.f_body, WHITE)
            ry += 18

        self._hline(ry + 6, x0=524, x1=1016)
        ry += 16
        self._blit("Symbolic Vocabulary Token Grounding:", 532, ry, self.f_bold, GOLD)
        ry += 20
        tok_rows = [
            ("Items   :", "0001:Apple  0002:Sword  0010:Wood  0021:OakTree", FG),
            ("Entities:", "E001:Slime  E005:Spider  E015:Sheep  E018:Vill", AMBER),
            ("Biomes  :", "B000:Forest  B001:Desert  B003:Plains  B005:Tundra", CYAN),
            ("Actions :", "A000:MoveN  A007:Attack  A008:PickUp  A017:Wait", WHITE),
        ]
        for k, v, col in tok_rows:
            self._blit(k, 532, ry, self.f_body, DIM)
            self._blit(v, 620, ry, self.f_body, col)
            ry += 20

    # -----------------------------------------------------------------------
    # In-Session Bottom Control Toolbar with Solid Vibrant Buttons
    # -----------------------------------------------------------------------
    def _draw_toolbar(self):
        self._hline(H - 66)
        ty = H - 54

        running = (self.app_state == STATE_RUNNING)
        paused  = (self.app_state == STATE_PAUSED)
        stopped = (self.app_state == STATE_STOPPED)
        in_run  = running or paused

        # Solid Vibrant Colored Action Buttons
        self._btn("⏸ PAUSE" if running else "▶ RESUME", 14, ty, 115, 36, "toggle_pause", active=paused, locked=stopped, color=WHITE, bg_color=BTN_PAUSE_BG, border_color=BTN_PAUSE_BDR, font=self.f_bold)
        self._btn("⚡ TURBO", 138, ty, 98, 36, "turbo", active=self.turbo_mode, locked=not running, color=WHITE, bg_color=BTN_TURBO_BG, border_color=BTN_TURBO_BDR, font=self.f_bold)
        self._btn("💾 SAVE", 244, ty, 96, 36, "save", locked=stopped, color=WHITE, bg_color=BTN_SAVE_BG, border_color=BTN_SAVE_BDR, font=self.f_bold)
        self._btn("🔄 RESET POLICY", 348, ty, 140, 36, "fresh_reset", locked=running, color=WHITE, bg_color=BTN_RESET_BG, border_color=BTN_RESET_BDR, font=self.f_bold)
        self._btn("⏹ STOP", 496, ty, 96, 36, "stop", locked=stopped, color=WHITE, bg_color=BTN_STOP_BG, border_color=BTN_STOP_BDR, font=self.f_bold)
        self._btn("🏠 MENU", 600, ty, 96, 36, "return_home", active=(self.app_state == STATE_STOPPED), locked=running, color=WHITE, bg_color=BTN_MENU_BG, border_color=BTN_MENU_BDR, font=self.f_bold)

        lock_note = " | [SPACE]=Pause/Resume | [T]=Turbo | [Ctrl+S]=Save" if in_run else ""
        self._blit(f"Tabs: Click [◀ TABS / VIEWS] on Top-Right{lock_note} | [ESC]=Pause | [W/A/S/D]=Actions", 14, H - 16, self.f_small, DIM)
