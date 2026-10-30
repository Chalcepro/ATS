"""Coal's ears (Chalce, 10-30): sound straight into the brain, not Whisper's text.

What reaches the brain is the sound itself, the same 24 pitch bands the CoalOS microphone
measures every 20 ms (coalos.js hearYou), as a sliding window of the last third of a
second. Coal hears while you speak: every new slice is a new moment for the brain, so
"I" is heard before "am", and the rise or fall of the voice at the end is in the bands.

Three things it learns to hear, each its own head:
  hear       which word is being said in this slice (or none)
  hear_end   speaking / a pause like a comma / a full stop / a question
  hear_cue   which sound this is, if it is not speech (sounds/coalos/<name>.wav)

Whisper is only the teacher, the way the judge is in training: it says which word was
where in practice sentences spoken by Piper's voices, and Coal learns to hear the words
from the bands. The words it can hear keep their slot numbers inside the brain file
(brain.keep["ear_words"]), so the one file still hears on its own.

Run (makes lessons once, then teaches the ears on the brain, then the exam):
  python -m household.ears lessons
  python -m household.ears teach
  python -m household.ears exam
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

BANDS = 24
STEP = 2                    # two 20 ms browser slices make one ear slice (40 ms)
EAR_FRAMES = 8              # the window: 8 x 40 ms, about a third of a second
EAR_WORDS = 512             # slots for words it can hear; slot 0 is "no word"
ENDS = ["quiet", "speaking", "pause", "stop", "question"]
EAR_CUES = 32               # slots for sound cues; slot 0 is "speech or nothing"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LESSONS = os.path.join(ROOT, "checkpoints", "ear_lessons.npz")
CUES = os.path.join(ROOT, "sounds", "coalos")
VOICES = os.path.join(os.path.expanduser("~"), "Documents", "github", "Cpeech", "piper-voices")


def ear_names() -> list[str]:
    return [f"ear.t{t}.b{b}" for t in range(EAR_FRAMES) for b in range(BANDS)] + ["ear.loud"]


EAR_SIZE = len(ear_names())


# ---- the bands, exactly as the CoalOS microphone makes them ------------------------------
def bands(samples: np.ndarray, rate: int) -> np.ndarray:
    """(slices, 24) loudness in dB, the way coalos.js does it: a Web Audio analyser
    (fftSize 1024, Blackman window, smoothing 0.8) read every 20 ms, max in each band."""
    if rate != 48000:                                   # browsers run the mic at 48 kHz
        n = int(len(samples) * 48000 / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
        rate = 48000
    N = 1024
    win = np.blackman(N)
    hop = rate // 50
    edges = [round(2 * (N / 4) ** (i / 24)) for i in range(25)]
    smooth = np.zeros(N // 2)
    out = []
    x = np.concatenate([np.zeros(N), samples.astype(np.float64)])
    for end in range(N, len(x) + 1, hop):
        mag = np.abs(np.fft.rfft(x[end - N:end] * win))[:N // 2] / N
        smooth = 0.8 * smooth + 0.2 * mag
        db = 20 * np.log10(np.maximum(smooth, 1e-12))
        out.append([max(-140.0, db[a:max(a + 1, edges[i + 1])].max()) for i, a in enumerate(edges[:24])])
    return np.array(out, np.float32)


def slices(b: np.ndarray) -> np.ndarray:
    """Browser slices (20 ms) to ear slices (40 ms)."""
    b = np.asarray(b, np.float32)
    n = len(b) // STEP * STEP
    return b[:n].reshape(-1, STEP, BANDS).max(1)


def ear_obs(s: np.ndarray, t: int) -> list[float]:
    """What the ear gives the brain at slice t: the last EAR_FRAMES slices, each band
    against the loudest of the window (so a quiet mic and a loud one sound alike),
    and how loud the window is."""
    w = s[max(0, t - EAR_FRAMES + 1):t + 1]
    if len(w) < EAR_FRAMES:
        w = np.concatenate([np.full((EAR_FRAMES - len(w), BANDS), -140, np.float32), w])
    top = float(w.max())
    rel = np.clip((w - top) / 40.0, -1.5, 0.0) + 1.5
    return list(rel.reshape(-1) / 1.5) + [float(np.clip((top + 140) / 100, 0, 1.2))]


def ear_all(s: np.ndarray) -> np.ndarray:
    return np.array([ear_obs(s, t) for t in range(len(s))], np.float32)


# ---- lessons: practice sentences, Piper's voices, Whisper as the teacher ------------------
def sentences() -> list[str]:
    from household import classroom, lexicon
    names = ["Ada", "Bayo", "Chi"]
    out = list(lexicon.WARN) + list(lexicon.URGE)
    for lines in lexicon.TALKS.values():
        out += lines
    for lines in lexicon.ASKS.values():
        out += lines
    out += lexicon.TELL + lexicon.NEGATE
    for l in classroom.LESSONS:
        out.append(l[1])
        out += list(l[2])
    sounds = os.path.join(ROOT, "checkpoints", "coal_sounds.jsonl")
    if os.path.exists(sounds):                          # things Chalce really said
        for line in open(sounds, encoding="utf-8"):
            out += [t for t in re.split(r"(?<=[.?!])\s+", json.loads(line).get("text", "")) if t]
    rng = random.Random(0)
    filled = []
    for t in out:
        for _ in range(2 if "{" in t else 1):
            filled.append(t.replace("{n}", rng.choice(names)).replace("{f}", rng.choice(lexicon.FEELINGS)))
    return sorted(set(s.strip() for s in filled if s.strip()))


def word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def make_lessons() -> None:
    """Every sentence in every voice, at two speeds; Whisper marks where each word is."""
    from faster_whisper import WhisperModel
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")
    texts = sentences()
    voices = sorted(glob.glob(os.path.join(VOICES, "*.onnx")))
    print(f"{len(texts)} sentences x {len(voices)} voices x 2 speeds", flush=True)
    rng = random.Random(1)
    clips = []
    for vp in voices:
        voice = PiperVoice.load(vp)
        for speed in (1.0, 1.25):
            for text in texts:
                audio = np.concatenate([c.audio_float_array for c in
                                        voice.synthesize(text, SynthesisConfig(length_scale=speed))])
                rate = voice.config.sample_rate
                pad = np.zeros(int(rate * rng.uniform(0.2, 0.5)), np.float32)
                audio = np.concatenate([pad, audio, pad])
                audio = audio * rng.uniform(0.2, 1.0) + rng.gauss(0, 1) * 0.003 * np.random.randn(len(audio)).astype(np.float32)
                a16 = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / rate)), np.arange(len(audio)), audio)
                segs, _ = whisper.transcribe(a16.astype(np.float32), language="en", word_timestamps=True, beam_size=1)
                words = [(w.word, w.start, w.end) for s in segs for w in (s.words or [])]
                clips.append({"bands": bands(audio, rate), "words": words, "text": text, "voice": os.path.basename(vp)})
            print(f"  {os.path.basename(vp)} x{speed}: {len(clips)} clips", flush=True)
    np.savez_compressed(LESSONS, clips=np.array(clips, dtype=object))
    print("saved", LESSONS)


def labels(clip: dict, vocab: dict) -> tuple[np.ndarray, list[int], list[int]]:
    """Ear slices for a clip, and for each slice the word being said and how speech stands."""
    s = slices(clip["bands"])
    t40 = np.arange(len(s)) * 0.04 + 0.04
    wid = [0] * len(s)
    end = [0] * len(s)
    ws = clip["words"]
    for k, (w, a, b) in enumerate(ws):
        key = word(w)
        if key and key not in vocab and len(vocab) < EAR_WORDS:
            vocab[key] = len(vocab)
        i = vocab.get(key, 0)
        last = k + 1 == len(ws)
        nxt = b + 0.4 if last else min(ws[k + 1][1], b + 0.4)
        mark = w.strip()[-1:]
        after = {"?": 4, ".": 3, "!": 3, ",": 2}.get(mark, 3 if last else 2 if nxt - b > 0.25 else 1)
        for j, t in enumerate(t40):
            if a <= t < b:
                wid[j], end[j] = i, 1
            elif b <= t < nxt and after >= 2:
                end[j] = after
    return s, wid, end


def cue_clips() -> list[tuple[str, np.ndarray, str]]:
    """Chalce's sound cues: sounds/coalos/<what it means>__<where from>.wav (or .mp3, via
    ffmpeg). Every file of one meaning is the same cue, whatever its style."""
    import subprocess
    out = []
    for f in sorted(glob.glob(os.path.join(CUES, "*.*"))):
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        name, _, src = stem.partition("__")
        try:
            raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", f, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                                 capture_output=True, check=True).stdout
        except Exception:
            continue
        out.append((name, np.frombuffer(raw, np.float32)[:48000 * 8], src))
    return out


# ---- teaching the ears ---------------------------------------------------------------------
def moving(brain):
    """The parts that learn to hear: the ear's columns of the first layer (the rest of
    that layer gets no change, since only ear inputs are on) and the three ear heads."""
    return [brain.body[0].weight, *brain.hear.parameters(), *brain.hear_end.parameters(), *brain.hear_cue.parameters()]


def teach(brain, epochs: int = 6, held_voice: str = "en_US-joy-high.onnx") -> dict:
    clips = list(np.load(LESSONS, allow_pickle=True)["clips"])
    keep = brain.keep.setdefault("ear_words", ["(none)"])
    vocab = {w: i for i, w in enumerate(keep)}
    cues = brain.keep.setdefault("ear_cues", ["(none)"])
    X, W, E, C, test = [], [], [], [], []
    for c in clips:
        s, w, e = labels(c, vocab)
        if c["voice"] == held_voice:
            test.append((s, w, e))
            continue
        X.append(ear_all(s)); W += w; E += e; C += [0] * len(s)
    rng = np.random.default_rng(0)
    cue_test = []
    for name, audio, src in cue_clips():
        if name not in cues and len(cues) < EAR_CUES:
            cues.append(name)
        if name not in cues:
            continue
        if "sonata" in src:                             # one style held back for the exam
            cue_test.append((name, slices(bands(audio, 48000))))
            continue
        for _ in range(6):                              # a few takes: louder, softer, cut in different places
            a = audio[rng.integers(0, max(1, len(audio) // 4)):] * rng.uniform(0.2, 1.0)
            s = slices(bands(a, 48000))
            X.append(ear_all(s)); W += [0] * len(s); E += [0] * len(s); C += [cues.index(name)] * len(s)
    keep[:] = sorted(vocab, key=vocab.get)
    X = torch.tensor(np.concatenate(X)); W = torch.tensor(W); E = torch.tensor(E); C = torch.tensor(C)
    print(f"{len(X)} ear slices, {len(keep) - 1} words, {len(cues) - 1} cues", flush=True)
    params = moving(brain)
    was = [p.requires_grad for p in brain.parameters()]
    for p in brain.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=2e-3)
    wt = torch.ones(EAR_WORDS); wt[0] = 0.2               # "no word" is most slices: count it less
    for ep in range(epochs):
        order = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = order[i:i + 256]
            out = brain(X[b])
            loss = (F.cross_entropy(out["hear"], W[b], weight=wt) + F.cross_entropy(out["hear_end"], E[b])
                    + F.cross_entropy(out["hear_cue"], C[b]))
            opt.zero_grad(); loss.backward(); opt.step()     # other inputs are zero: their weights get no change
        print(f"  epoch {ep + 1}: loss {loss.item():.3f}", flush=True)
    for p, r in zip(brain.parameters(), was):
        p.requires_grad_(r)
    r = exam(brain, test)
    if cue_test:
        ok = sum(listen(brain, s, browser=False)["cue"] == n for n, s in cue_test)
        r["cues_new_style"] = ok / len(cue_test)
        print(f"cues in a style it never heard (Sonata): {ok}/{len(cue_test)}", flush=True)
    return r


@torch.no_grad()
def listen(brain, b, browser: bool = True) -> dict:
    """What Coal hears in some sound: the words, in order, and how it ended."""
    s = slices(b) if browser else b
    out = brain(torch.tensor(ear_all(s)))
    words = brain.keep.get("ear_words", ["(none)"])
    cues = brain.keep.get("ear_cues", ["(none)"])
    w = out["hear"].argmax(1).tolist()
    heard, last = [], 0
    for i in w:
        if i and i != last and i < len(words):
            heard.append(words[i])
        last = i
    e = out["hear_end"].argmax(1).tolist()
    c = out["hear_cue"].softmax(1).mean(0)
    cue = cues[int(c.argmax())] if int(c.argmax()) < len(cues) and float(c.max()) > 0.5 else None
    tail = [ENDS[i] for i in e if i >= 2]
    return {"words": heard, "end": tail[-1] if tail else None, "cue": None if cue == "(none)" else cue}


def exam(brain, test: list | None = None) -> dict:
    """On a voice it never practised with: how many slices' words and endings it hears right,
    and on Chalce's real recordings, how many of Whisper's words it also heard."""
    if test is None:
        vocab = {w: i for i, w in enumerate(brain.keep.get("ear_words", ["(none)"]))}
        test = [labels(c, dict(vocab)) for c in np.load(LESSONS, allow_pickle=True)["clips"] if c["voice"] == "en_US-joy-high.onnx"]
    ok_w = n_w = ok_e = n_e = 0
    with torch.no_grad():
        for s, w, e in test:
            out = brain(torch.tensor(ear_all(s)))
            pw, pe = out["hear"].argmax(1), out["hear_end"].argmax(1)
            for j in range(len(s)):
                if w[j]:
                    n_w += 1; ok_w += int(pw[j] == w[j])
                if e[j] >= 2:
                    n_e += 1; ok_e += int(pe[j] == e[j])
    got = total = 0
    sounds = os.path.join(ROOT, "checkpoints", "coal_sounds.jsonl")
    if os.path.exists(sounds):
        for line in open(sounds, encoding="utf-8"):
            d = json.loads(line)
            want = {word(x) for x in d.get("text", "").split()} - {""}
            if want and len(d["bands"]) > 20:
                h = set(listen(brain, d["bands"])["words"])
                got += len(want & h); total += len(want)
    r = {"words_new_voice": ok_w / max(1, n_w), "endings_new_voice": ok_e / max(1, n_e),
         "chalce_words_heard": got / max(1, total)}
    print("ear exam: " + ", ".join(f"{k} {v:.0%}" for k, v in r.items()), flush=True)
    return r


if __name__ == "__main__":
    from household.model import load, save
    from household.train import SAVE
    what = sys.argv[1] if len(sys.argv) > 1 else "exam"
    if what == "lessons":
        make_lessons()
    else:
        path = sys.argv[2] if len(sys.argv) > 2 else SAVE
        brain, rounds = load(path)
        if what == "teach":
            teach(brain)
            save(brain, path, rounds)
            print("saved", path)
        else:
            exam(brain)
