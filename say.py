"""Speak a briefing instead of writing another wall of text.

    python say.py "the ladder is repaired, every rung clears its bar"
    python say.py --file notes.txt
    echo "..." | python say.py

Why this exists in the ATS repo rather than in Cpeech: Cpeech is a GUI, and
a briefing that needs a window opened and a button pressed is one that does
not get spoken. This is the same Piper voice, called directly, so a script
or a finished training run can say what happened.

The voice is libritts-high because that is the one Bob picked. Nothing here
downloads anything - if the model is missing it says so and writes the text
out instead, because a briefing tool that fails silently is worse than no
briefing tool.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

VOICES = Path.home() / "Documents" / "github" / "Cpeech" / "piper-voices"
VOICE = "en_US-libritts-high.onnx"


def model_path() -> Path | None:
    p = VOICES / VOICE
    return p if p.is_file() else None


def speak(text: str, out: Path | None = None, play: bool = True) -> Path | None:
    """Render `text` to a wav and play it. Returns the wav, or None."""
    text = (text or "").strip()
    if not text:
        return None
    model = model_path()
    if model is None:
        print("[say] no voice at %s - not speaking" % (VOICES / VOICE),
              file=sys.stderr)
        print(text)
        return None

    out = Path(out) if out else Path(tempfile.gettempdir()) / "ats_briefing.wav"
    try:
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(model), "-f", str(out)],
            input=text.encode("utf-8"), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace").strip()
        print("[say] piper failed: %s" % (err or exc), file=sys.stderr)
        print(text)
        return None

    if play:
        try:
            import sounddevice as sd
            import wave

            import numpy as np
            with wave.open(str(out), "rb") as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
            audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
            sd.play(audio, sr)
            sd.wait()
        except Exception as exc:
            print("[say] rendered but could not play (%s): %s" % (exc, out),
                  file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="what to say")
    ap.add_argument("--file", help="read the text from a file instead")
    ap.add_argument("--out", help="keep the wav here")
    ap.add_argument("--no-play", action="store_true")
    a = ap.parse_args(argv)

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = " ".join(a.text)
    else:
        text = sys.stdin.read()

    path = speak(text, out=a.out, play=not a.no_play)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
