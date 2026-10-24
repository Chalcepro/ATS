"""H1 gate, the real half: speak calmly, then shout; the ear must tell them apart.
   py -3.12 tools/ear_check.py
Nothing is saved; the sound only lives in memory while this runs."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sounddevice as sd
from ear import hear, FIELD, RATE

def take(prompt, secs=4):
    input(f"\n{prompt}  (press Enter, then speak for {secs} s) ")
    x = sd.rec(int(RATE * secs), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    return hear(x[:, 0])

calm = take("Say something CALM and quiet")
shout = take("Now SHOUT it")
for name in ("rms_mean", "rms_p95", "pitch_median_norm", "pitch_slope", "voiced_fraction", "centroid_mean"):
    c, s = calm[FIELD[name]], shout[FIELD[name]]
    print(f"  {name:18s} calm {c:8.3f}   shout {s:8.3f}")
ok = shout[FIELD["rms_mean"]] > 2 * calm[FIELD["rms_mean"]] and shout[FIELD["pitch_median_norm"]] > calm[FIELD["pitch_median_norm"]]
print("\nPASS: the ear tells them apart" if ok else "\nNOT YET: louder and higher should both go up when you shout")
