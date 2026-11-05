"""Coal listens (Chalce, 11-03): always listening, recording only when it matters, quoting exactly.

    py -3.12 -m household.listen hear FILE.wav --when "2026-11-03 10:40"   (Coal listens to a stretch of sound)
    py -3.12 -m household.listen ask "who said I can get the documents tomorrow?"
    py -3.12 -m household.listen enrol NAME FILE.wav                        (teach Coal a voice)
    py -3.12 -m household.listen name "Person B" Registrar                  (rename a voice Coal heard)
    py -3.12 -m household.listen check                                      (is the chain of recordings unbroken?)

The ear (always on, cheap): loudness in 30 ms frames against a moving noise floor - is someone talking?
Nothing is kept while it only listens.
Starting to record (v0 cues; which cues matter is to be learned from what Chalce keeps):
  - talk starts after a long quiet,
  - someone says "record" ("Coal, record this", "start recording"),
  - a new voice joins.
Stopping: STOP_QUIET seconds of quiet, then a buzz (the phone vibrates) so he knows.
Voices by their sound, not by text: each voice gets prints from its frequencies (the shape of its spectrum and
its pitch), minus what all voices share (average_voice.npy). A voice is matched on its best three samples;
a new voice is named by Coal (Person A, B, ...) until Chalce renames it.
What is kept: only recorded moments - the clip, the exact words (local speech-to-text), who, when - each linked
to the one before by a fingerprint (sha256), so editing or slipping in a clip later breaks the chain.
Asked "who said ...?", Coal quotes: best match first, with the date, the voice and the clip. It only
summarises when asked to.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARS = os.environ.get("COAL_EARS") or os.path.join(HERE, "checkpoints", "ears")   # clips, the log, the voice prints
SR = 16000
FRAME = 480                                                # 30 ms
QUIET_START = 20.0                                         # talk after this much quiet starts a recording
STOP_QUIET = 4.0                                           # this much quiet stops it (minutes, in real life)
PAUSE = 0.25                                               # a pause this long ends a piece (turns come quick)
NOISE_SUB = 1.5           # 11-05: background (the quiet just before) taken off before a voice is compared; 0 = off
SPLIT_LONG = False         # 11-03 on Chalce's clip over a TikTok: catches some other voices, but noise turned his own
                           # voice into "Person D" too. Off until prints hold up in noise.
CUE = re.compile(r"\b(record|recording)\b", re.I)


# ---- sound in ------------------------------------------------------------------------------------
def read_wav(path: str) -> np.ndarray:
    if not path.lower().endswith(".wav"):                  # mp3, m4a, ogg...: decoded to 16 kHz by the speech-to-text package's reader
        from faster_whisper.audio import decode_audio
        return decode_audio(path, sampling_rate=SR).astype(np.float32)
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
        sr = w.getframerate()
    if sr != SR:                                           # to 16 kHz
        x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# ---- the ear: is someone talking? -------------------------------------------------------------------
def talking(x: np.ndarray) -> np.ndarray:
    """One yes/no per 30 ms frame, against a noise floor that follows the room."""
    n = len(x) // FRAME
    e = 10 * np.log10((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1) + 1e-10)
    floor, out = e[:10].mean() if n else -60.0, np.zeros(n, bool)
    for i, v in enumerate(e):
        out[i] = v > floor + 9
        if not out[i]:
            floor = 0.95 * floor + 0.05 * v                 # the room's hum, slowly
    return out


def utterances(speech: np.ndarray) -> list[tuple[int, int]]:
    """Frames of speech joined across short pauses: (start frame, end frame)."""
    gap = int(PAUSE * SR / FRAME)
    out, start, quiet = [], None, 0
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet > gap:
                if i - quiet - start > 8:                  # longer than ~0.25 s: a word, not a click
                    out.append((start, i - quiet))
                start, quiet = None, 0
    if start is not None and len(speech) - start > 8:
        out.append((start, len(speech)))
    return out


# ---- voices by their sound ----------------------------------------------------------------------------
def spectrum(x: np.ndarray) -> np.ndarray:
    """Magnitude per frequency, 512-sample frames (32 ms)."""
    n = len(x) // 512
    return np.abs(np.fft.rfft(x[:n * 512].reshape(n, 512) * np.hanning(512), axis=1))


def background(x: np.ndarray, speech: np.ndarray, a: int, secs: float = 6.0) -> np.ndarray | None:
    """What the room sounded like just before frame `a` when nobody was talking (music, a fan, a TV):
    the median spectrum of the quiet frames in the last `secs` seconds. None if there were too few."""
    lo = max(0, a - int(secs * SR / FRAME))
    quiet = [i for i in range(lo, a) if not speech[i]]
    if len(quiet) < 10:
        return None
    clip = np.concatenate([x[i * FRAME:(i + 1) * FRAME] for i in quiet])
    m = spectrum(clip)
    return np.median(m, 0) if len(m) else None


def voiceprint(x: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    """The shape of a voice: average energy in 32 bands (log-spaced 80 Hz - 7.6 kHz), how much it varies,
    and the pitch. Normalised, so loudness does not count. With `noise` (a background spectrum), that much
    is taken off every frame first, so the voice looks the same over music as in a quiet room."""
    n = len(x) // 512
    frames = x[:n * 512].reshape(n, 512) * np.hanning(512)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    if noise is not None and NOISE_SUB:
        mag = np.maximum(mag - NOISE_SUB * noise, 0.05 * mag)   # keep a little, so no band falls to nothing
    edges = np.geomspace(80, 7600, 33) / (SR / 512)
    bands = np.stack([mag[:, int(a):max(int(a) + 1, int(b))].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    loud = mag.sum(1) > np.median(mag.sum(1))               # the voiced frames
    lb = np.log(bands[loud] + 1e-6)
    lb -= lb.mean(1, keepdims=True)
    ac = np.array([np.correlate(f, f, "full")[511 + 40:511 + 400] for f in frames[loud][:60]])
    pitch = np.median(SR / (40 + ac.argmax(1))) if len(ac) else 150.0
    v = np.concatenate([lb.mean(0), lb.std(0) * 0.5, [np.log(pitch) * 2]])
    return v / (np.linalg.norm(v) + 1e-9)


def voices() -> dict:
    p = os.path.join(EARS, "voices.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_voices(v: dict) -> None:
    os.makedirs(EARS, exist_ok=True)
    json.dump(v, open(os.path.join(EARS, "voices.json"), "w"), indent=1)


AVERAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "average_voice.npy")   # what voices share
SAME = 0.47          # 11-03, four voices: the same voice scored 0.50+ against its own best three, a different one 0.44 at most
KEEP_PRINTS = 20


def centred(p: np.ndarray) -> np.ndarray:
    """A print minus what all voices share, so only what makes this voice itself is compared."""
    v = p - np.load(AVERAGE) if os.path.exists(AVERAGE) else p
    return v / (np.linalg.norm(v) + 1e-9)


NEW_BELOW = 0.40     # 11-05: between this and SAME it is "probably them" (no new person, nothing learned); a new
                     # person only when clearly unlike everyone - near-misses had turned Chalce into Person A
LEARN = 0.55         # 11-05: only a sure match adds a sample, so noisy lines cannot drag a voice towards someone else
ENROL_KEEP = 8       # 11-05 trial: 8 locked + 12 learned held Chalce at 95-97% over music; 20 locked fell to 6-44%
ANCHORS = "_anchors" # voices.json key: how many of a voice's first samples were enrolled and are never pushed out


def who(print_: np.ndarray, known: dict) -> str:
    """The voice whose best three samples are closest, if close enough; otherwise someone new. Each voice keeps
    up to KEEP_PRINTS samples (one sample is too few: the same voice saying a different line can look like another).
    Enrolled samples stay for good; learned ones are added only on a sure match and roll over.
    (11-05: before this, music-filled lines replaced all of Chalce's clean samples and he split into Persons A/B.)"""
    c = centred(print_)
    anchors = known.setdefault(ANCHORS, {})
    best, name = -1.0, None
    for k, prints in known.items():
        if k.startswith("_") or not prints:
            continue
        s = float(np.mean(sorted(float(c @ centred(np.array(p))) for p in prints)[-3:]))
        if s > best:
            best, name = s, k
    if name is None or best < NEW_BELOW:
        name = f"Person {chr(ord('A') + sum(k.startswith('Person ') for k in known))}"   # Coal names them; Chalce can rename
        known[name] = [list(print_)]
        return name
    if best >= LEARN:
        n = anchors.get(name, 0)
        fixed, learned = known[name][:n], known[name][n:]
        known[name] = fixed + (learned + [list(print_)])[-max(3, KEEP_PRINTS - n):]
    return name


# ---- the record of what was heard ----------------------------------------------------------------------
def log_path() -> str:
    return os.path.join(EARS, "heard.jsonl")


def heard() -> list[dict]:
    p = log_path()
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def link(prev: str, clip: bytes, e: dict) -> str:
    return hashlib.sha256(prev.encode() + clip + json.dumps([e["when"], e["who"], e["text"]]).encode()).hexdigest()


def keep(e: dict, clip: np.ndarray) -> dict:
    os.makedirs(os.path.join(EARS, "clips"), exist_ok=True)
    name = e["when"].replace(":", "").replace(" ", "_") + f"_{e['who'].replace(' ', '')}.wav"
    path = os.path.join(EARS, "clips", name)
    write_wav(path, clip)
    log = heard()
    e["clip"] = os.path.relpath(path, EARS)
    e["prev"] = log[-1]["hash"] if log else "start"
    e["hash"] = link(e["prev"], open(path, "rb").read(), e)
    with open(log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")
    return e


def check() -> list[str]:
    """Every clip and line against its link: anything edited, removed or slipped in shows up."""
    bad, prev = [], "start"
    for e in heard():
        clip = os.path.join(EARS, e["clip"])
        ok = os.path.exists(clip) and e["prev"] == prev and link(prev, open(clip, "rb").read(), e) == e["hash"]
        if not ok:
            bad.append(f"{e['when']} {e['who']}: {e['text'][:40]!r}")
        prev = e["hash"]
    return bad


# ---- hearing a stretch of sound ------------------------------------------------------------------------
_model = None
SURE, GUESS = 0.4, 0.12     # word confidence; 11-03 on Chalce's own voice note 0.6/0.3 flagged many words it had right


def words(x: np.ndarray) -> str:
    """The exact words, by local speech-to-text (faster-whisper, already on this laptop; nothing goes online)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = _model.transcribe(x, language="en", beam_size=1, vad_filter=False, word_timestamps=True)
    out = []
    for seg in segs:
        for w in seg.words or []:
            t = w.word.strip()
            if w.probability >= SURE:
                out.append(t)
            elif w.probability >= GUESS:                     # heard, not sure: a guess, marked
                out.append(f"[{t.strip('.,?!')}?]")
            elif not out or out[-1] != "[inaudible]":
                out.append("[inaudible]")
    return " ".join(out).strip()


def hear(x: np.ndarray, start: dt.datetime, log=print) -> list[dict]:
    """Listen to a stretch of sound that began at `start`. Records only when a cue says so."""
    known = voices()
    before = set(known)                                     # voices met before this stretch
    speech = talking(x)
    recording, last_end, kept, buzzes = False, 0.0, [], []
    turns = []                                              # pieces of talk, joined while the same voice goes on
    win = int(1.5 * SR / FRAME) if SPLIT_LONG else 10 ** 9  # check a long stretch every ~1.5 s (voices take turns fast)
    for a0, b0 in utterances(speech):
        for a in range(a0, b0, win):
            b = min(b0, a + win)
            if b - a < win // 2 and turns and turns[-1][1] == a:   # a short tail joins the piece before it
                turns[-1][1] = b
                continue
            v = who(voiceprint(x[a * FRAME:b * FRAME], background(x, speech, a)), known)
            if turns and turns[-1][2] == v and (a - turns[-1][1]) * FRAME / SR < 1.0:
                turns[-1][1] = b
            else:
                turns.append([a, b, v])
    met = set()                                             # new voices already announced this stretch
    for a, b, voice in turns:
        t0, t1 = a * FRAME / SR, b * FRAME / SR
        fresh = voice not in before
        if t1 - t0 < 0.6 and fresh and len(known.get(voice, [])) == 1:
            del known[voice]                               # a cough or a click is not a new person
            continue
        clip = x[a * FRAME:b * FRAME]
        announce = fresh and voice in known and voice not in met
        met.add(voice)
        if recording and t0 - last_end > STOP_QUIET:
            recording = False
            buzzes.append(last_end + STOP_QUIET)
            log(f"  [{fmt(start, last_end + STOP_QUIET)}] buzz: quiet for {STOP_QUIET:.0f} s, recording stopped")
        why = None
        if not recording:
            if t0 - last_end > QUIET_START:
                why = "talk after a long quiet"
            elif announce:
                why = "a new voice"
            else:
                head = words(clip[: 4 * SR])                    # only the first seconds are checked for "record"
                if CUE.search(head):
                    why = "asked to record"
            if why:
                recording = True
                log(f"  [{fmt(start, t0)}] recording ({why})")
        if recording:
            text = words(clip)
            if not text:
                continue
            e = keep({"when": fmt(start, t0), "who": voice, "text": text, "secs": round(t1 - t0, 1)}, clip)
            kept.append(e)
            log(f"  [{e['when']}] {voice}: {text}")
        last_end = t1
    if recording:
        log(f"  [{fmt(start, last_end + STOP_QUIET)}] buzz: quiet, recording stopped")
    save_voices(known)
    return kept


def fmt(start: dt.datetime, secs: float) -> str:
    return (start + dt.timedelta(seconds=secs)).strftime("%Y-%m-%d %H:%M:%S")


# ---- asking: exact quotes ---------------------------------------------------------------------------------
STOP = set("who what when where did said say says the a an i you me my is was to of and that it can could would "
           "tell told about he she they them his her".split())


def names() -> dict:
    """Names Chalce gave to voices ("Person B" is the Registrar). Kept apart from the chain, so naming
    a voice later never changes a recording."""
    p = os.path.join(EARS, "names.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def ask(question: str, n: int = 5) -> str:
    """Who said it, word for word: the best matches first, each with its date, voice and clip. No summary."""
    log = heard()
    if not log:
        return "I haven't recorded anything yet."
    want = [w for w in re.findall(r"[a-z']+", question.lower()) if w not in STOP]
    df = {}
    for e in log:
        for w in set(re.findall(r"[a-z']+", e["text"].lower())):
            df[w] = df.get(w, 0) + 1

    def stem(w: str) -> str:
        return re.sub(r"(ation|ing|ed|s)$", "", w)

    def score(e: dict) -> float:
        have = {stem(w) for w in re.findall(r"[a-z']+", e["text"].lower())}
        return sum(np.log((len(log) + 1) / (df.get(w, 0) + 1)) + 1 for w in want if stem(w) in have)

    hits = sorted([e for e in log if score(e) > 0], key=lambda e: (-score(e), e["when"]))[:n]
    if not hits:
        return "I didn't record anyone saying that."
    newest = max(hits, key=lambda e: e["when"])
    lines = [f"Found {len(hits)}:"]
    for e in hits:
        day = dt.datetime.strptime(e["when"], "%Y-%m-%d %H:%M:%S").strftime("%a %d %b, %H:%M")
        lines.append(f"  {day}, {names().get(e['who'], e['who'])}: \"{e['text']}\"   [clip: {e['clip']}]" + ("   <- most recent" if e is newest else ""))
    return "\n".join(lines)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "hear":
        when = sys.argv[sys.argv.index("--when") + 1] if "--when" in sys.argv else dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        hear(read_wav(sys.argv[2]), dt.datetime.strptime(when, "%Y-%m-%d %H:%M"))
    elif cmd == "ask":
        print(ask(" ".join(sys.argv[2:])))
    elif cmd == "enrol":                                    # several samples from the clip: windows of ~1.5 s of talk
        v, x = voices(), read_wav(sys.argv[3])
        sp = np.repeat(talking(x), FRAME)[:len(x)]
        talk = x[:len(sp)][sp]
        v[sys.argv[2]] = [list(voiceprint(talk[i:i + 24000])) for i in range(0, max(1, len(talk) - 12000), 12000)][:KEEP_PRINTS]
        v[sys.argv[2]] = v[sys.argv[2]][:ENROL_KEEP]
        v.setdefault(ANCHORS, {})[sys.argv[2]] = len(v[sys.argv[2]])   # enrolled samples are kept for good, the rest learned
        save_voices(v)
        print(f"Coal knows {sys.argv[2]}'s voice.")
    elif cmd == "name":                                     # name "Person B" Registrar
        nm = names()
        nm[sys.argv[2]] = " ".join(sys.argv[3:])
        json.dump(nm, open(os.path.join(EARS, "names.json"), "w"), indent=1)
        print(f"{sys.argv[2]} is {nm[sys.argv[2]]}.")
    elif cmd == "check":
        bad = check()
        print("The chain is unbroken." if not bad else "Changed or out of place:\n  " + "\n  ".join(bad))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
