"""H1 gate, synthetic half: a calm voice and a shout must come apart on loudness,
pitch and pitch movement. (The real half is Chalce's own recording: tools/ear_check.py.)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ear import hear, FIELD, RATE

t = np.arange(RATE) / RATE
def voice(f0, f1, amp):
    """A buzzy voice-like tone gliding f0 -> f1 Hz, with a few harmonics and syllable pulses."""
    f = np.linspace(f0, f1, len(t))
    ph = 2 * np.pi * np.cumsum(f) / RATE
    x = sum(np.sin(k * ph) / k for k in range(1, 6))
    syll = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t).clip(0)
    return (amp * x * syll / 2.3).astype(np.float32)

calm, shout = hear(voice(120, 115, 0.05)), hear(voice(220, 300, 0.6))
loud, pitch, slope = FIELD["rms_mean"], FIELD["pitch_median_norm"], FIELD["pitch_slope"]
print("calm : loud %.3f pitch %3.0f Hz slope %+.2f" % (calm[loud], calm[pitch] * 400, calm[slope]))
print("shout: loud %.3f pitch %3.0f Hz slope %+.2f" % (shout[loud], shout[pitch] * 400, shout[slope]))
assert shout[loud] > 5 * calm[loud]
assert 100 < calm[pitch] * 400 < 140 and 200 < shout[pitch] * 400 < 320
assert shout[slope] > calm[slope]
assert not hear(np.zeros(100, np.float32)).any(), "too short to judge -> all zeros"
print("ok")
