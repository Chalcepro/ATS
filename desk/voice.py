"""Yes and no, both ways.

Chalce -> brain: a word (typed, or the yes/no buttons) is read as yes (+1),
no (-1) or unknown (0). Every word is logged with what the brain did in the
last few seconds, for sleep to learn from later (roadmap Sleep: nothing is
learned while awake). If the brain answered a question just before, the word
also says whether that answer was right.

Brain -> Chalce: the voice head, a tiny layer on the brain's memory state (the
GRU hidden), picks thumbs up or thumbs down when asked. It starts untrained, so
early answers are coin flips; it is its own file (checkpoints/desk_voice.pt),
so training it never touches brain.pt."""
import json, os, re, time

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_PATH = os.path.join(ROOT, "checkpoints", "desk_voice.pt")
LOG_DIR = os.path.join(ROOT, "data", "desk_feedback")

YES = {"yes", "yeah", "yep", "yup", "ya", "y", "ok", "okay", "good", "right", "correct", "nice", "sure"}
NO = {"no", "nope", "nah", "n", "stop", "bad", "wrong", "dont", "don't", "not"}
ANSWER_WINDOW_S = 15        # a yes/no this soon after it answered also grades the answer


def meaning(text):
    """+1 yes, -1 no, 0 unknown. The first yes/no word wins ("no, stop" is no)."""
    for w in re.findall(r"[a-z']+", text.lower()):
        if w in YES:
            return 1
        if w in NO:
            return -1
    return 0


class VoiceHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.out = nn.Linear(hidden, 2)         # [no, yes]

    @classmethod
    def load(cls, hidden):
        head = cls(hidden)
        if os.path.exists(VOICE_PATH):
            saved = torch.load(VOICE_PATH, map_location="cpu")
            if saved["out.weight"].shape[1] == hidden:      # the brain grew: start over
                head.load_state_dict(saved)
        else:
            os.makedirs(os.path.dirname(VOICE_PATH), exist_ok=True)
            torch.save(head.state_dict(), VOICE_PATH)        # same untrained voice every run
        head.eval()
        return head

    def answer(self, hidden):
        """+1 thumbs up, -1 thumbs down, sampled from the head's own odds."""
        with torch.no_grad():
            p = torch.softmax(self.out(torch.tensor(hidden, dtype=torch.float32)), -1)
        return (1 if torch.multinomial(p, 1).item() == 1 else -1), float(p[1])


class Voice:
    def __init__(self, hidden):
        self.head = VoiceHead.load(hidden)
        self.last_answer = None             # (time, +1/-1, p_yes, hidden, question)

    def ask(self, hidden, question):
        ans, p_yes = self.head.answer(hidden)
        self.last_answer = (time.time(), ans, p_yes, hidden, question)
        self.log({"kind": "question", "text": question, "answer": ans, "p_yes": round(p_yes, 3), "hidden": hidden})
        return ans

    def hear_word(self, text, recent):
        """Chalce said something. recent: the brain's last few (state, action) steps."""
        m = meaning(text)
        entry = {"kind": "word", "text": text, "meaning": m, "recent": recent}
        la = self.last_answer
        if m and la and time.time() - la[0] < ANSWER_WINDOW_S:
            # "yes" after its answer = the answer was right; "no" = wrong
            entry["grades_answer"] = {"question": la[4], "answer": la[1], "right": m == 1, "hidden": la[3]}
            self.last_answer = None
        self.log(entry)
        return m

    @staticmethod
    def log(entry):
        os.makedirs(LOG_DIR, exist_ok=True)
        entry["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(LOG_DIR, time.strftime("%Y-%m-%d") + ".jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
