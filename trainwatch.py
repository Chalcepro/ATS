"""One small window that says how the training is going.

    python trainwatch.py                 # find the running trainer by itself
    python trainwatch.py logs/repair.log # or watch a particular log

Why a window and not more printing
----------------------------------
A training run is thirty to forty minutes of scrolling text, and the two
things worth knowing - how far through it is, and whether it has finished -
are the two things scrolling text is worst at showing. This stays open,
updates the same line and the same bar rather than printing new ones, and
changes state once at the end.

It reads a log file. It does not import the trainer, does not share its
process, and cannot slow it down or crash it - if this window is closed or
breaks, the run carries on.

It follows either trainer, because both are used:

    consolidate.py      "  --- 3200 episodes, 1304s ---"
                        "  every rung clears its bar..."
    train_curriculum.py "    ep 300   reward ..."
                        "    PASSED after 425 episodes ..."
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
POLL_MS = 1000

# consolidate: "  --- 3200 episodes, 1304s ---"
RE_CONSOL = re.compile(r"---\s*(\d+)\s+episodes,\s*(\d+)s")
# train_curriculum: "    ep 300    reward   +0.46  success  41% ..."
RE_EP = re.compile(r"^\s*ep\s+(\d+)\s+reward\s+([-+]?[\d.]+)\s+success\s+(\d+)%")
# either trainer's headline
RE_STAGE = re.compile(r"^===\s+(\S+\s+\d+x\d+)")
RE_DONE = re.compile(r"every rung clears|PASSED after (\d+)|gave up after (\d+)|"
                     r"##### DONE|ALL DONE")
RE_SHORT = re.compile(r"still short under greedy:\s*(.+)")
RE_CAP = re.compile(r"--episodes[= ](\d+)")
# main.py: "Episode 12 done | reward=..." and the cap it announces
RE_ISLAND = re.compile(r"^Episode (\d+) done \| reward=([-+]?[\d.]+)")
RE_ISLANDCAP = re.compile(r"episode cap: (\d+) ticks")

BG = "#1b1d22"
FG = "#e6e8ec"
DIM = "#8b93a1"
OK = "#5fbf7f"
BUSY = "#5b8dd6"


def find_run():
    """The trainer that is running, as (pid, logfile, episode cap), or Nones.

    Looks at the command line rather than asking to be told, because the
    thing a person wants after starting a run is to watch it, not to
    describe it again.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match "
             "'consolidate|train_curriculum|main[.]py' } | "
             "ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None, None, None
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid, cmd = line.split("|", 1)
        cap = RE_CAP.search(cmd)
        return int(pid), None, (int(cap.group(1)) if cap else None)
    return None, None, None


def alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                             capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return True          # cannot tell: assume still going, do not cry done


def newest_log():
    logs = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    return logs[0] if logs else None


class Watch:
    def __init__(self, root, log=None):
        self.root = root
        self.pid, _, self.cap = find_run()
        self.log = Path(log) if log else newest_log()
        self.done = False
        self.started = time.time()
        self.last_size = 0
        self.episodes = 0
        self.stage = ""
        self.detail = ""
        self.announced = False

        root.title("ATS training")
        root.configure(bg=BG)
        root.geometry("460x210")
        root.attributes("-topmost", True)

        pad = {"padx": 16}
        self.head = tk.Label(root, text="looking for a training run…",
                             bg=BG, fg=FG, font=("Segoe UI", 13, "bold"),
                             anchor="w")
        self.head.pack(fill="x", pady=(16, 2), **pad)

        self.sub = tk.Label(root, text="", bg=BG, fg=DIM, anchor="w",
                            font=("Segoe UI", 9))
        self.sub.pack(fill="x", **pad)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("W.Horizontal.TProgressbar", troughcolor="#2a2d34",
                        background=BUSY, bordercolor=BG, lightcolor=BUSY,
                        darkcolor=BUSY)
        self.bar = ttk.Progressbar(root, style="W.Horizontal.TProgressbar",
                                   maximum=100, length=420)
        self.bar.pack(fill="x", pady=(14, 6), **pad)

        self.pct = tk.Label(root, text="", bg=BG, fg=FG, anchor="w",
                            font=("Segoe UI", 10))
        self.pct.pack(fill="x", **pad)

        self.note = tk.Label(root, text="", bg=BG, fg=DIM, anchor="w",
                             wraplength=420, justify="left",
                             font=("Segoe UI", 8))
        self.note.pack(fill="x", pady=(6, 12), **pad)

        self.tick()

    # ---- reading the log ------------------------------------------------
    def scan(self):
        """Re-read only what is new. Tailing beats re-parsing a 150 KB log."""
        if not self.log or not self.log.is_file():
            return
        size = self.log.stat().st_size
        if size < self.last_size:          # the log was truncated: start over
            self.last_size = 0
        if size == self.last_size:
            return
        with open(self.log, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self.last_size)
            chunk = fh.read()
        self.last_size = size

        for line in chunk.splitlines():
            m = RE_CONSOL.search(line)
            if m:
                self.episodes = int(m.group(1))
                self.stage = "holding every rung at once"
                continue
            m = RE_EP.match(line)
            if m:
                self.episodes = int(m.group(1))
                self.detail = "reward %s, success %s%%" % (m.group(2), m.group(3))
                continue
            m = RE_ISLAND.match(line)
            if m:
                self.episodes = int(m.group(1))
                self.detail = "reward %s" % m.group(2)
                self.stage = self.stage or "the island"
                continue
            m = RE_ISLANDCAP.search(line)
            if m:
                self.stage = "the island - %s tick episodes" % m.group(1)
                continue
            m = RE_STAGE.match(line)
            if m:
                self.stage = m.group(1)
                continue
            m = RE_SHORT.search(line)
            if m:
                self.detail = "still short: " + m.group(1)[:70]
                continue
            if RE_DONE.search(line):
                self.done = True

    # ---- the one update --------------------------------------------------
    def tick(self):
        self.scan()
        running = alive(self.pid) if self.pid else False

        if not self.pid and not self.done:
            # Maybe it started after this window did.
            self.pid, _, cap = find_run()
            if cap:
                self.cap = cap
            if self.pid:
                self.log = newest_log() or self.log
                self.last_size = 0

        if self.done or (self.pid and not running):
            self.finish()
            return

        if not self.pid:
            self.head.configure(text="no training running", fg=DIM)
            self.sub.configure(text="start one and this will pick it up")
            self.bar.configure(value=0)
            self.pct.configure(text="")
        else:
            self.head.configure(text=self.stage or "training", fg=FG)
            el = int(time.time() - self.started)
            self.sub.configure(text="pid %d   ·   %d:%02d elapsed"
                                    % (self.pid, el // 60, el % 60))
            if self.cap:
                frac = min(1.0, self.episodes / float(self.cap))
                self.bar.configure(value=frac * 100)
                left = ""
                if self.episodes > 20 and el > 10:
                    rate = self.episodes / float(el)
                    if rate > 0:
                        rem = int((self.cap - self.episodes) / rate)
                        left = "   ·   about %d min left" % max(0, rem // 60)
                self.pct.configure(text="%d of %d episodes   ·   %.0f%%%s"
                                        % (self.episodes, self.cap,
                                           frac * 100, left))
            else:
                # No cap to divide by: show motion, not a false fraction.
                self.bar.configure(mode="indeterminate")
                self.bar.start(60)
                self.pct.configure(text="%d episodes" % self.episodes)
            self.note.configure(text=self.detail)

        self.root.after(POLL_MS, self.tick)

    def finish(self):
        """Said once, then left alone."""
        try:
            self.bar.stop()
            self.bar.configure(mode="determinate", value=100)
        except tk.TclError:
            pass
        style = ttk.Style(self.root)
        style.configure("W.Horizontal.TProgressbar", background=OK,
                        lightcolor=OK, darkcolor=OK)
        el = int(time.time() - self.started)
        self.head.configure(text="Training finished", fg=OK)
        self.sub.configure(text="%d episodes  ·  %d:%02d"
                                % (self.episodes, el // 60, el % 60))
        self.pct.configure(text=self.detail or "")
        self.note.configure(text="Run  python diag_ladder.py  to see what it "
                                 "actually plays now.")
        if not self.announced:
            self.announced = True
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.bell()
            self._toast()
        # Stays open. Nothing further is scheduled, so it cannot spam.

    def _toast(self):
        """A notification, and the voice if it is set up.

        Deliberately once. The whole point of this window is that it is not
        a stream of alerts.
        """
        msg = "ATS training finished - %d episodes" % self.episodes
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Windows.Forms;"
                 "$n=New-Object System.Windows.Forms.NotifyIcon;"
                 "$n.Icon=[System.Drawing.SystemIcons]::Information;"
                 "$n.Visible=$true;"
                 "$n.ShowBalloonTip(10000,'ATS','%s',"
                 "[System.Windows.Forms.ToolTipIcon]::Info);"
                 "Start-Sleep -Seconds 10;$n.Dispose()" % msg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        say = HERE / "say.py"
        if say.is_file():
            try:
                subprocess.Popen([sys.executable, str(say), msg],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except Exception:
                pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    Watch(root, argv[0] if argv else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
