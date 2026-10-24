"""The ear (roadmap H1): the live microphone turned into 34 numbers about HOW it
sounds (loudness, pitch, pitch rising/falling, voiced or not, pauses,
brightness, 8 frequency bands), several times a second.

The maths is senses/features.py prosody() (the same ear.v1 block the senses
repo records), plain numpy, no model, no network. It describes how something
was said, never what was said.

Muted is the default, and muted means the microphone stream is closed, not
just ignored. Nothing is recorded to disk."""
import os, sys, threading, time

import numpy as np

SENSES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "senses")
sys.path.insert(0, SENSES)
from senses.features import prosody, PROSODY_FIELDS, PROSODY_DIM   # noqa: E402

RATE = 16000
WINDOW_S = 1.0          # each hearing looks at the last second
EVERY_S = 0.25          # and is redone four times a second
FIELD = {name: i for i, name in enumerate(PROSODY_FIELDS)}


def hear(samples, rate=RATE):
    """34 floats for a stretch of sound; all zeros for silence too short to judge."""
    return prosody(np.asarray(samples, np.float32), rate)


class Ear:
    def __init__(self):
        self.muted = True
        self.vec = np.zeros(PROSODY_DIM, np.float32)
        self._buf = np.zeros(int(RATE * WINDOW_S), np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._last = 0.0
        self.error = ""

    def _on_audio(self, data, frames, t, status):
        x = data[:, 0]
        with self._lock:
            self._buf = np.concatenate([self._buf[len(x):], x])

    def set_muted(self, muted):
        if muted == self.muted:
            return
        self.muted = muted
        if muted:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self.vec[:] = 0
            return
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                          blocksize=int(RATE * 0.05), callback=self._on_audio)
            self._stream.start()
            self.error = ""
        except Exception as e:           # no mic, or it's busy: stay muted and say why
            self.error = str(e)[:80]
            self.muted = True

    def update(self, now=None):
        """Recompute the 34 numbers if it's time; returns them (zeros while muted)."""
        now = time.time() if now is None else now
        if not self.muted and now - self._last >= EVERY_S:
            self._last = now
            with self._lock:
                x = self._buf.copy()
            self.vec = hear(x)
        return self.vec

    def loudness(self):
        return float(self.vec[FIELD["rms_mean"]])

    def pitch_hz(self):
        return float(self.vec[FIELD["pitch_median_norm"]] * 400.0)

    def close(self):
        self.set_muted(True)
