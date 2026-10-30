"""Sorts downloaded system sounds into sounds/coalos/<meaning>__<source>.wav, so the ears
learn what a sound means (an error is an error in every style), not one recording."""
import os, re, shutil, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "sounds", "_downloads")
OUT = os.path.join(HERE, "sounds", "coalos")
# most specific first: a filename takes the first meaning whose words it holds
MEANINGS = [("battery_critical", ["battery critical", "batcrt"]), ("battery_low", ["battery low", "batlow"]),
            ("critical_stop", ["critical stop", "crtstp"]), ("device_fail", ["hardware fail", "hdfail"]),
            ("device_in", ["hardware insert", "hdinst"]), ("device_out", ["hardware remove", "hdrem"]),
            ("startup", ["startup", "xpstartu"]), ("logon", ["logon"]), ("logoff", ["logoff"]),
            ("shutdown", ["shutdown", "shutdn"]), ("error", ["error"]), ("exclamation", ["exclamation", "xpexcl"]),
            ("ding", ["ding"]), ("default", ["default", "xpdef"]), ("notify", ["notify"]),
            ("balloon", ["balloon", "balln"]), ("recycle", ["recycl"]), ("minimize", ["minimize", "xpmin"]),
            ("restore", ["restore", "restor"]), ("menu", ["menu"]), ("click", ["navigation start", "start"]),
            ("printed", ["print"]), ("popup_blocked", ["pop-up"]), ("info_bar", ["information bar"]),
            ("ring_in", ["ringin"]), ("ring_out", ["ringout", "rngout"]), ("chimes", ["chimes"]),
            ("chord", ["chord"]), ("tada", ["tada"]), ("permission", ["user account control"])]

def meaning(stem):
    s = stem.lower()
    if "speech" in s or s.startswith("ir_") or s == "xpstartb":
        return None
    for m, keys in MEANINGS:
        if any(k in s for k in keys):
            return m
    return None

os.makedirs(OUT, exist_ok=True)
kept, skipped = {}, []
for root, _, files in os.walk(SRC):
    for f in files:
        if not f.lower().endswith(".wav"):
            continue
        m = meaning(os.path.splitext(f)[0])
        if not m:
            skipped.append(f); continue
        src = re.sub(r"[^a-z0-9]+", "-", os.path.relpath(root, SRC).lower()).strip("-")
        shutil.copy(os.path.join(root, f), os.path.join(OUT, f"{m}__{src}.wav"))
        kept[m] = kept.get(m, 0) + 1
print(len(kept), "meanings:", ", ".join(f"{k} {v}" for k, v in sorted(kept.items())))
print("left out:", sorted(set(skipped)))
