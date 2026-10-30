"""The household window: talk to the brain from a browser, on this machine only.

Whatever is plugged in becomes a sense, and whatever is missing is switched
off - the brain is built to keep caring with any of them gone:
  camera      -> eyes   (the household's bodies and visible cues)
  microphone  -> ears   (coughs) and spoken words, turned to text here with
                          faster-whisper - nothing leaves the machine
  speaker     -> it speaks its sentences aloud with Piper
  keyboard    -> words  (always there: typing alone is enough)

Run: python -m household.app      then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from household.judge import HURT, condition
from household.lexicon import Lexicon, felt_by, grow
from household.talk import load
from household.train import SAVE
from household.words import DOING, narrate, sentence
from household.world import ROOMS, SENSES, Household

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_PATH = os.path.join(os.path.dirname(SAVE), "lexicon.json")
VOICE = os.path.join(os.path.expanduser("~"), "Documents", "github", "Cpeech", "piper-voices",
                     "en_US-libritts-high.onnx")
PORT = 8765


class Session:
    def __init__(self, seed: int = 7):
        self.lock = threading.Lock()
        self.brain = load()
        self.lex = Lexicon.load(LEX_PATH) if os.path.exists(LEX_PATH) else grow()
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.w = Household(seed=seed, people=3)
        self.events: list[dict] = []

    def move(self) -> tuple[str, int]:
        return self.brain.choose(self.w.recall())

    def state(self) -> dict:
        w = self.w
        act, tgt = self.move()
        said = self.brain.say(w.recall(), len(w.people))
        names = [p.name for p in w.people]
        people = []
        for i, p in enumerate(w.people):
            seen = w.in_sight(i)
            words = said[i] if seen else w.told.get(i, [])
            people.append({"name": p.name, "alive": p.alive, "in_sight": seen, "away": i in w.unseen,
                           "it_says": sentence(p.name, words) if (seen or words) else f"I can't see {p.name}.",
                           "told": w.told.get(i, []), "truth": narrate(p),
                           "health": round(p.health, 2), "stranger": p.stranger})
        doing = DOING.get(act, act)
        doing = doing.format(n=names[tgt]) if "{n}" in doing and tgt < len(names) else doing
        return {"hour": w.hour, "power": round(w.power, 2), "store": w.store, "senses": w.senses,
                "here": w.here, "danger": w.danger, "all_rooms": list(ROOMS), "caught": w.caught, "rooms": [p.room for p in w.people],
                "people": people, "doing": "I'm " + doing + ".", "action": act, "target": tgt,
                "words_known": len(self.lex.ids), "events": self.events[-40:]}

    def hours(self, n: int) -> None:
        for _ in range(max(1, min(n, 48))):
            act, tgt = self.move()
            before = self.state()["doing"]
            self.w.step(act, tgt)
            self.events.append({"hour": self.w.hour, "kind": "act", "text": before})
            if n > 1 and any(c == HURT for c in condition(self.w)):
                self.events.append({"hour": self.w.hour, "kind": "note", "text": "Stopped early: someone is hurt."})
                break

    def say(self, text: str) -> dict:
        names = [p.name for p in self.w.people]
        heard = self.lex.read(text, names)
        self.events.append({"hour": self.w.hour, "kind": "you", "text": text})
        # everything you say reaches it as its own person's words - a warning,
        # an urging, or nothing it has a meaning for yet
        self.w.heard = heard["meaning"]
        if heard["who"] is None:
            reply = f"Heard you. {self.state()['doing']}"
            if heard["new"]:
                reply += " Still learning: " + ", ".join(heard["new"]) + "."
        else:
            who = heard["who"]
            self.w.tell(who, heard["felt"])
            felt = ", ".join(heard["felt"]) or "nothing I know a word for yet"
            reply = f"{names[who]} feels {felt}."
            if heard["fixed"]:
                reply += " (I read " + ", ".join(f"'{a}' as '{b}'" for a, b in heard["fixed"].items()) + ".)"
            if heard["new"]:
                reply += " Still learning: " + ", ".join(heard["new"]) + "."
            reply += " " + self.state()["doing"]
            self.lex.learn(text, felt_by(self.w.people[who], random.Random(self.w.hour)))
            self.lex.save(LEX_PATH)
        self.events.append({"hour": self.w.hour, "kind": "brain", "text": reply})
        return {"reply": reply, "heard": heard}


_whisper = None


def transcribe(audio: bytes) -> str:
    """Speech to text on this machine (faster-whisper small.en, already downloaded)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("small.en", device="cpu", compute_type="int8", local_files_only=True)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio)
    try:
        segments, _ = _whisper.transcribe(f.name, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        os.unlink(f.name)


def speak(text: str) -> bytes | None:
    """Text to speech on this machine with Piper; None if Piper or the voice is missing."""
    if not shutil.which("piper") or not os.path.exists(VOICE):
        return None
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "say.wav")
        subprocess.run(["piper", "--model", VOICE, "--output_file", out], input=text.encode(),
                       capture_output=True, timeout=60)
        return open(out, "rb").read() if os.path.exists(out) else None


S: Session | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, kind: str = "application/json", code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), code=code)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(open(os.path.join(HERE, "web", "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with S.lock:
                self._json(S.state())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path == "/api/listen":                    # audio in, text out
            try:
                text = transcribe(raw)
            except Exception as e:
                return self._json({"error": f"could not listen: {e}"}, 500)
            return self._json({"text": text})
        if self.path == "/api/voice":                     # text in, audio out
            wav = speak(json.loads(raw or b"{}").get("text", ""))
            return self._send(wav, "audio/wav") if wav else self._json({"error": "no voice"}, 503)
        body = json.loads(raw or b"{}")
        with S.lock:
            if self.path == "/api/hours":
                S.hours(int(body.get("n", 1)))
                return self._json(S.state())
            if self.path == "/api/say":
                out = S.say(str(body.get("text", ""))[:500])
                return self._json({**out, "state": S.state()})
            if self.path == "/api/away":
                i, away = int(body.get("who", 0)), bool(body.get("away"))
                if 0 <= i < len(S.w.people):
                    S.w.away(i, away)
                return self._json(S.state())
            if self.path == "/api/senses":
                for k in SENSES:
                    if k in body and k != "words":         # words (typing) are always there
                        S.w.senses[k] = bool(body[k])
                return self._json(S.state())
            if self.path == "/api/danger":                # hidden from the brain: only your words can warn it
                room = body.get("room") or None
                S.w.danger = room if room in ROOMS else None
                return self._json(S.state())
            if self.path == "/api/reset":
                S.reset(int(body.get("seed", random.randrange(10_000))))
                return self._json(S.state())
        self._json({"error": "not found"}, 404)


def main() -> None:
    global S
    S = Session()
    print(f"household window: http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
