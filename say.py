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
import time
from pathlib import Path

VOICES = Path.home() / "Documents" / "github" / "Cpeech" / "piper-voices"
VOICE = "en_US-libritts-high.onnx"

# Where spoken briefings are kept, one file per briefing.
#
# They used to all land on a single temp file, so every briefing overwrote the
# last one and the whole history was whichever one happened to be most recent -
# in the system temp directory, which Windows is free to empty. A briefing
# worth speaking is worth being able to hear again, and the recordings of Bob's
# own voice are already kept this way in Formant/voice.
BRIEFINGS = Path(__file__).resolve().parent / "briefings"

# Older briefings are pruned past this many. Text is cheap, 3 MB of wav each
# is not, and the drives on this machine are not in good health.
KEEP_BRIEFINGS = 40


def prune(keep: int = KEEP_BRIEFINGS):
    """Keep the newest few briefings. Promised by the comment on
    KEEP_BRIEFINGS, and a constant that nothing enforces is a lie."""
    try:
        wavs = sorted(BRIEFINGS.glob("*.wav"), key=lambda q: q.stat().st_mtime)
        for old in wavs[:-keep] if len(wavs) > keep else []:
            old.unlink(missing_ok=True)
            old.with_suffix(".txt").unlink(missing_ok=True)
    except Exception:
        pass


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

    if out:
        out = Path(out)
    else:
        BRIEFINGS.mkdir(parents=True, exist_ok=True)
        out = BRIEFINGS / ("%s.wav" % time.strftime("%Y-%m-%d_%H%M%S"))
        # The text beside the wav, so a briefing can be read as well as
        # replayed - and so it stays findable without listening to forty of
        # them.
        try:
            out.with_suffix(".txt").write_text(text, encoding="utf-8")
        except Exception:
            pass
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
    ap.add_argument("--list", action="store_true",
                    help="what briefings are kept, newest last")
    a = ap.parse_args(argv)

    if a.list:
        wavs = sorted(BRIEFINGS.glob("*.wav"), key=lambda q: q.stat().st_mtime)
        if not wavs:
            print("  no briefings kept yet -> %s" % BRIEFINGS)
            return 1
        print("  %d briefing(s) in %s" % (len(wavs), BRIEFINGS))
        for w in wavs:
            txt = w.with_suffix(".txt")
            first = ""
            if txt.is_file():
                first = txt.read_text(encoding="utf-8").strip().split(". ")[0][:62]
            print("    %-22s %6.1f MB  %s"
                  % (w.name, w.stat().st_size / 1e6, first))
        return 0

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text:
        text = " ".join(a.text)
    else:
        text = sys.stdin.read()

    path = speak(text, out=a.out, play=not a.no_play)
    if path and not a.out:
        prune()
        print("[say] kept at %s" % path, file=sys.stderr)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
