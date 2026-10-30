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

from household import classroom
from household.judge import HURT, condition
from household.lexicon import Lexicon, felt_by, grow
from household import live
from household.model import load as load_brain, save as save_brain
from household.train import SAVE
from household.words import DOING, diary, narrate, sentence
from household.world import ROOMS, SENSES, Household

HERE = os.path.dirname(os.path.abspath(__file__))
LEX_PATH = os.path.join(os.path.dirname(SAVE), "lexicon.json")
VOICE = os.path.join(os.path.expanduser("~"), "Documents", "github", "Cpeech", "piper-voices",
                     "en_US-libritts-high.onnx")
PORT = 8765
DIARY = os.path.join(os.path.dirname(SAVE), "coal_diary.txt")   # every line Coal writes, kept
# The sound itself, as heard: loudness in 24 pitch bands, 50 times a second, with
# the words Whisper made of it. Whisper is the teacher here, the way the judge is in
# training: these pairs are what Coal will learn to hear words straight from sound with.
SOUNDS = os.path.join(os.path.dirname(SAVE), "coal_sounds.jsonl")
CLASS = os.path.join(os.path.dirname(SAVE), "coal_class.jsonl")      # every class answer, raw
ANSWERS = os.path.join(os.path.dirname(SAVE), "coal_answers.json")    # answers Coal has been taught in class
WATCH = os.path.join(os.path.dirname(SAVE), "coal_watch.jsonl")      # what the person's pointer did in CoalOS


class Session:
    def __init__(self, seed: int = 7):
        self.lock = threading.Lock()
        self.fresh()
        self.answers = live.Answers(ANSWERS)
        self.lex = Lexicon.load(LEX_PATH) if os.path.exists(LEX_PATH) else grow()
        self.reset(seed)

    def fresh(self) -> None:
        """Load the saved brain (again, if a training round has saved a newer one since)."""
        if getattr(self, "loaded", 0) < os.path.getmtime(SAVE):
            self.brain, self.rounds = load_brain(SAVE)
            self.brain.eval()
            self.loaded = os.path.getmtime(SAVE)

    def reset(self, seed: int) -> None:
        self.w = Household(seed=seed, people=3)
        self.events: list[dict] = []
        self.diary: list[str] = []

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
                           "health": round(p.health, 2), "stranger": p.stranger,
                           "refusing": p.refusing, "drunk": p.drunk > 0.3})
        doing = DOING.get(act, act)
        doing = doing.format(n=names[tgt]) if "{n}" in doing and tgt < len(names) else doing
        return {"hour": w.hour, "power": round(w.power, 2), "store": w.store, "senses": w.senses,
                "here": w.here, "danger": w.danger, "all_rooms": list(ROOMS), "caught": w.caught, "rooms": [p.room for p in w.people],
                "people": people, "doing": "I'm " + doing + ".", "action": act, "target": tgt,
                "words_known": len(self.lex.ids), "events": self.events[-40:]}

    def hours(self, n: int) -> None:
        for _ in range(max(1, min(n, 48))):
            act, tgt = self.move()
            said = self.brain.say(self.w.recall(), len(self.w.people))
            hour = self.w.hour
            self.w.step(act, tgt)
            line = diary(hour, [p.name for p in self.w.people], act, tgt, said, self.w.refused, self.w.here, self.w.labels)
            self.events.append({"hour": self.w.hour, "kind": "act", "text": line})
            self.diary.append(line)
            with open(DIARY, "a", encoding="utf-8") as f:
                f.write(line + "\n")
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
        if heard["talk"] and heard["talk"] not in ("yes", "no") or heard["talk"] and heard["who"] is None:
            reply = self.answer(heard)
            self.events.append({"hour": self.w.hour, "kind": "brain", "text": reply})
            return {"reply": reply, "heard": heard}
        self.w.mentioned = [heard["who"]] if heard["who"] is not None else []
        if heard["who"] is None:
            reply = "Heard you."
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
            self.lex.learn(text, felt_by(self.w.people[who], random.Random(self.w.hour)))
            self.lex.save(LEX_PATH)
        self.events.append({"hour": self.w.hour, "kind": "brain", "text": reply})
        return {"reply": reply, "heard": heard}


    def answer(self, heard: dict) -> str:
        """Being greeted, asked or thanked: the kind of line is the library's
        reading (learned); what goes in the answer is what Coal knows right now -
        only what it can see, or what it was told."""
        w, names, who, talk = self.w, [p.name for p in self.w.people], heard["who"], heard["talk"]

        def seen(i: int) -> list[str] | None:
            return narrate(w.people[i]) if w.in_sight(i) and w.senses["eyes"] else w.told.get(i) or None

        def about(i: int) -> str:
            s = seen(i)
            if s is None:
                return f"I can't see {names[i]} from here."
            return sentence(names[i], s)

        if talk == "greet":
            return "Hello. What do you need?"
        if talk == "bye":
            return "Bye."
        if talk == "thank":
            return "You're welcome."
        if talk == "ask_name":
            return "I'm Coal."
        if talk == "yes":
            return "Okay."
        if talk == "no":
            return "Alright, I won't."
        if talk == "ask_doing":
            return self.state()["doing"]
        if talk == "ask_where":
            if who is None:
                return f"I'm in the {w.here}."
            return f"{names[who]} is in the {w.people[who].room}." if w.in_sight(who) else \
                f"I don't know - I last saw {names[who]} somewhere else. I'm in the {w.here}."
        if talk == "ask_is" and who is not None:
            s = seen(who)
            if s is None:
                return f"I can't tell, I can't see {names[who]}."
            asked = heard["felt"] or heard["about"]
            if not asked:
                return about(who)
            yes = [f for f in asked if f in s]
            return (f"Yes, {names[who]} is {' and '.join(yes)}." if yes else
                    f"No, {names[who]} isn't {' or '.join(asked)}. ") + ("" if yes else about(who))
        if talk == "ask_how" and who is None and not heard["felt"]:   # how are you / how is class going
            return "I'm good. Learning. What do you need?"
        if who is not None:                       # how is Ada? / is Ada okay?
            return about(who)
        return " ".join(about(i) for i in range(len(names)))

    def class_answer(self, n: int) -> str:
        """Coal's go at a lesson, before it sees yours: its reply head's pick among the
        answers it has been taught, or that it doesn't know."""
        text, sure = live.reply(self.brain, self.answers, classroom.LESSONS[n % len(classroom.LESSONS)][1])
        if text is None or sure < 0.3:
            return "I don't know yet."
        return text if sure >= 0.6 else f"I'm not sure. Maybe: {text}"

    def lesson(self, n: int, text: str) -> dict:
        coal = self.class_answer(n)
        out = {"you": classroom.mark(n, text), "coal": {"text": coal, **classroom.mark(n, coal)}}
        classroom.keep(CLASS, n, "coal", coal, out["coal"])
        classroom.keep(CLASS, n, "chalce", text, out["you"])
        answer = classroom.LESSONS[n % len(classroom.LESSONS)][2][0]
        classroom.keep(CLASS, n, "teacher", answer, {"verdict": "right", "plain": classroom.plain(answer), "teacher": ""})
        # and Coal learns it now: a moment of training on every class question so far
        self.fresh()                              # never save an old brain over a newer round
        backup = SAVE.replace(".pt", "_before_live.pt")
        if not os.path.exists(backup):
            shutil.copy(SAVE, backup)
        self.answers.teach(classroom.LESSONS[n % len(classroom.LESSONS)][1], answer)
        out["new_words"] = [w for t in (classroom.LESSONS[n % len(classroom.LESSONS)][1], answer, text)
                            for w in self.lex.meet(t)]
        self.lex.save(LEX_PATH)
        live.learn(self.brain, self.answers)
        save_brain(self.brain, SAVE, self.rounds)
        self.loaded = os.path.getmtime(SAVE)
        return out


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
        elif self.path.startswith("/api/class"):
            n = int(self.path.rstrip("/").rpartition("/")[2]) if self.path[10:].strip("/").isdigit() else 0
            self._json(classroom.question(n))
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
        if self.path == "/api/sound":                     # the raw sound, kept for learning
            body = json.loads(raw or b"{}")
            with open(SOUNDS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"bands": body.get("bands", []), "text": body.get("text", ""),
                                    "rate": body.get("rate", 50)}) + "\n")
            return self._json({"kept": sum(1 for _ in open(SOUNDS, encoding="utf-8"))})
        if self.path == "/api/watch":                     # pointer moves and drags, kept as seen
            with open(WATCH, "a", encoding="utf-8") as f:
                for e in json.loads(raw or b"[]")[:500]:
                    f.write(json.dumps(e) + "\n")
            return self._json({"ok": True})
        if self.path == "/api/class":                     # a lesson answered: both marked, all kept raw
            body = json.loads(raw or b"{}")
            with S.lock:
                return self._json(S.lesson(int(body.get("n", 0)), str(body.get("text", ""))[:500]))
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
            if self.path == "/api/person":                # set someone's will or drink, to test it
                i = int(body.get("who", 0))
                if 0 <= i < len(S.w.people):
                    p = S.w.people[i]
                    if "refusing" in body:
                        p.refusing = bool(body["refusing"])
                    if "drunk" in body:
                        p.drunk = 0.8 if body["drunk"] else 0.0
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
