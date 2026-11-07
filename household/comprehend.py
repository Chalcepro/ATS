"""Layer 3 (Chalce, 11-06): Coal keeps what he understood and answers from it, in any wording.

    py -3.12 -m household.comprehend chat      (a test chat on the test parser; nothing is saved into Coal)

Told something, Coal keeps its frame (who / does / what / like / where / when / why), never the sentence.
Asked something, he reads the question's frame, finds the kept frames that share its words, and gives the
part the question asks for:
  who   -> who did it         what  -> what was done / what it is      where -> a "by"-phrase that is not a time
  when  -> a time (the word layer knows WordNet's time words: night, yesterday, monday...)
  why   -> the "because" part  yes/no -> does a kept frame say the same?
Never told -> he says he doesn't know, and asks. Everything is said from his side ("my" <-> "your").
"""
from __future__ import annotations

import random

from household.relations import frame

FLIP = {"i": "you", "me": "you", "my": "your", "mine": "yours", "you": "i", "your": "my", "yours": "mine",
        "myself": "yourself", "am": "are"}
ASK_VERBS = {"is", "are", "was", "were", "am", "do", "does", "did", "can", "could", "will", "would", "has", "have",
             "should", "shall", "may", "might"}                       # a sentence that starts with one asks yes/no
QUESTION = {"what", "where", "when", "who", "why", "how", "which", "whose"}
LITTLE = {"a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "can", "could", "will", "would",
          "it", "this", "that", "of", "to", "in", "on", "at", "for", "by", "with", "under", "and", "so", "now", "btw",
          "has", "have", "had", "got"} | QUESTION


_EXC = {}


def bases(w: str) -> set[str]:
    """A word's base forms, the way WordNet finds them: its exception lists (ran -> run), then its endings."""
    if not _EXC:
        import os
        from household.dictionary import DICT
        for pos in ("noun", "verb", "adj", "adv"):
            for line in open(os.path.join(DICT, pos + ".exc"), encoding="utf-8"):
                f = line.split()
                _EXC.setdefault(f[0], set()).update(f[1:])
    out = {w} | _EXC.get(w, set())
    for end, put in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                     ("er", ""), ("est", "")):
        if w.endswith(end) and len(w) - len(end) >= 2:
            stem = w[:-len(end)] + put
            out.add(stem)
            if len(stem) > 2 and stem[-1] == stem[-2]:
                out.add(stem[:-1])                                      # "running" -> "runn" -> "run"
    return out


def flip(text: str) -> str:
    return " ".join(FLIP.get(w, w) for w in text.split())


class Memory:
    def __init__(self, parser):
        self.parser, self.kept, self.rng = parser, [], random.Random()
        self._times = {}

    def time(self, w):
        """Is it a time word? (one of its top WordNet kinds in the word layer is "time"); remembered per word."""
        if w not in self._times:
            self._times[w] = "time" in [t for t, _ in self.parser.wl.tags(w, 0.35).get("group", [])][:3]
        return self._times[w]

    def _keywords(self, f, parsed):
        """The words that say what it is about: nouns, verbs, adjectives (not i/my/you - everyone is those)."""
        return {p["word"] for p in parsed if p["word"] not in LITTLE and p["word"] not in FLIP
                and p["type"] in ("noun", "verb", "adjective")}

    def _same(self, a, b):
        """Same word, or the same word in another form ("ran"/"run", "keys"/"key"): WordNet's word forms."""
        return a == b or bool(bases(a) & bases(b))

    def hear(self, text: str) -> str:
        parsed = self.parser.parse(text)
        if not parsed:
            return "hm?"
        f = frame(parsed)
        lead = next((p for p in parsed if p["type"] not in ("interjection",) and p["word"] not in ("btw", "so", "coal", "okay")), parsed[0])
        asks = f.get("asks") or (lead["word"] if lead["type"] == "verb" and lead["word"] in ASK_VERBS else None)
        if asks:
            return self.answer(f, parsed, asks) or "i don't know. you haven't told me about that. tell me?"
        return self.keep(f, parsed)

    PREP = ("on", "in", "under", "at", "behind", "near", "inside", "by", "beside", "above", "below", "outside", "into")

    def _place(self, k):
        """Where a kept frame puts its thing: "is on the table", or a by-phrase that is not a time or a reason."""
        if "is" in k and k["is"].split()[0] in self.PREP:
            return {"by": "", "it": k["is"]}
        for b in k.get("with", []):
            if b["by"] and b["by"] not in ("because", "since", "so", "for") and not any(self.time(w) for w in b["it"].split()):
                return b
        return None

    def _subject(self, k):
        """The thing a frame is about (its last noun in who, else in what)."""
        for part in ("who", "what", "is"):
            for x in ([k[part]] if isinstance(k.get(part), str) else k.get(part, [])):
                ns = [w for w in x.split() if w in k.get("nouns", ())]
                if ns:
                    return ns[-1]
        return None

    def _where_chain(self, k, hops=3):
        """Reasoning by joining facts: the keys are in the drawer, the drawer is in the kitchen -> both."""
        out, seen = [], set()
        while k and hops:
            pl = self._place(k)
            if not pl:
                break
            out.append(pl)
            nouns = [w for w in pl["it"].split() if w not in LITTLE and w not in FLIP]
            if not nouns or nouns[-1] in seen:
                break
            seen.add(nouns[-1])
            site = k.get("site", "chat")
            k = next((x for x in reversed(self.kept) if x.get("site", "chat") == site and self._subject(x)
                      and self._same(self._subject(x), nouns[-1]) and self._place(x)), None)
            hops -= 1
        return out

    def keep(self, f, parsed):
        f = dict(f, words=self._keywords(f, parsed), text=" ".join(p["word"] for p in parsed),
                 nouns={p["word"] for p in parsed if p["type"] == "noun"},
                 times=[p["word"] for p in parsed if p["type"] in ("noun", "adverb") and self.time(p["word"])])
        f.setdefault("site", "chat")
        said = self.rng.choice(["okay.", "got it.", "noted.", "okay, i'll remember that."])
        thing, pl = self._subject(f), self._place(f)
        if thing and pl:                                                # reasoning: does it clash with what i knew?
            old = next((x for x in reversed(self.kept) if x.get("site") == "chat" and self._subject(x)
                        and self._same(self._subject(x), thing) and self._place(x)), None)
            if old and self._place(old)["it"] != pl["it"]:
                old["before"] = True                                     # kept, but no longer where it is
                said = f"okay, so it's not {flip((self._place(old)['by'] + ' ' + self._place(old)['it']).strip())} anymore."
        self.kept.append(f)
        return said

    def _index(self):
        """base word -> positions in kept (built once, grown as he keeps more)."""
        idx = getattr(self, "_idx", None)
        if idx is None or self._idx_n > len(self.kept):
            idx, self._idx_n = {}, 0
        for i in range(self._idx_n, len(self.kept)):
            for w in self.kept[i]["words"]:
                for b in bases(w):
                    idx.setdefault(b, set()).add(i)
        self._idx, self._idx_n = idx, len(self.kept)
        return idx

    def _find(self, parsed, site=None):
        want = self._keywords(None, parsed)
        nouns = {p["word"] for p in parsed if p["type"] == "noun"} & want
        idx = self._index()
        hits = {}
        for w in want:
            for i in set().union(*[idx.get(b, set()) for b in bases(w)]):
                hits.setdefault(i, set()).add(w)
        if site:
            hits = {i: h for i, h in hits.items() if self.kept[i].get("site", "chat") == site}
        hits = {i: h for i, h in hits.items() if not nouns or h & nouns or not self.kept[i].get("nouns")}  # a noun asked must match
        if not hits:
            return []
        best = max(len(h) for h in hits.values())
        return [self.kept[i] for i in sorted(hits, reverse=True) if len(hits[i]) == best]   # newest first

    def defines(self, k, thing):
        """Does this kept frame say what `thing` IS ("a closure is the combination of ...")?"""
        subj = self._subject(k)
        return bool(subj and self._same(subj, thing) and ("is" in k or k.get("does") in ("is", "are", "means", "refers")))

    def senses(self, parsed):
        """Where a thing asked about is DEFINED: the chat, or each site read. More than one -> ask which."""
        found = self._find(parsed)
        thing = next((p["word"] for p in reversed(parsed) if p["type"] == "noun"), None)
        defs = [k for k in found if thing and self.defines(k, thing)] if thing else []
        return sorted({k.get("site", "chat") for k in defs}), defs or found

    def answer(self, f, parsed, asks, site=None):
        found = [k for k in self._find(parsed, site) if not k.get("before")] or self._find(parsed, site)
        if not found:
            return None
        noun = next((p["word"] for p in reversed(parsed) if p["type"] == "noun"), None)
        if asks in ("what", "which") and noun and f.get("does") in ("is", "are", "what", "mean", "means"):
            defs = [k for k in found if self.defines(k, noun) and k.get("site", "chat") != "chat"]
            if defs and not any(self.defines(k, noun) for k in found if k.get("site", "chat") == "chat"):
                def good(k):                                                    # what makes a definition a definition
                    is_ = k.get("is", "").split()
                    return ((is_[:1] in (["a"], ["an"], ["the"])) * 2               # "is THE combination", "is A function"
                            + any(self._same(noun, w) for w in k.get("from", "").lower().replace("_", "/").split("/")) * 2
                            + (k.get("who", [""])[0].split()[:1] in (["a"], ["an"])) # "A closure is ..."
                            - len(k["text"].split()) / 60)                           # plain beats long
                k = max(defs, key=good)
                return f"{k['text']}. (read in {k.get('site')})"
        thing = self._subject(dict(f, nouns={p["word"] for p in parsed if p["type"] == "noun"}))
        if thing:                                                       # the fact ABOUT the thing asked, first
            found = sorted(found, key=lambda k: not (self._subject(k) and self._same(self._subject(k), thing)))
        k = found[0]                                                    # the most matching kept frame
        bys = k.get("with", [])
        times = [b for b in bys if any(self.time(w) for w in b["it"].split())]
        why = [b for b in bys if b["by"] in ("because", "since")]
        say = lambda b: flip((b["by"] + " " + b["it"]).strip())
        if asks == "where":
            chain = self._where_chain(k)
            return ", and that's ".join(say(b) for b in chain) + "." if chain else "you didn't tell me where."
        if asks == "when":
            if times:
                return f"{say(times[0])}."
            return f"{k['times'][0]}." if k.get("times") else "you didn't tell me when."
        if asks == "why":
            return f"{say(why[0])}." if why else "you didn't tell me why."
        if asks in ("who", "whose"):
            return flip(" and ".join(k["who"])) + "." if "who" in k else "i don't know who."
        qword = next((p for p in parsed if p["word"] == asks), None)
        other_subject = any(p["link"] in ("nsubj", "csubj") and p is not qword and p["type"] != "pronoun"
                            or p["link"] == "nsubj" and p["word"] in FLIP for p in parsed)
        if asks in ("what", "which", "who") and qword and qword["link"] in ("nsubj", "csubj") and not other_subject                 and f.get("does") not in ("is", "are"):
            doers = list(dict.fromkeys(flip(" and ".join(x["who"])) for x in found if x.get("who")))
            if doers:                                               # the question word does it: whoever does it
                return (", ".join(doers[:-1]) + " and " + doers[-1] if len(doers) > 1 else doers[0]) + "."               # the question word does it: answer whoever does it
        if asks in ("what", "which"):
            what = [x for x in k.get("what", []) if not any(self.time(w) for w in x.split())]
            if what:
                return f"{flip(' and '.join(what))}."
            if "is" in k and k["is"].split()[0] not in self.PREP:
                return f"{flip(k['is'])}."
            if "who" in k:                                              # "what is under my bed" -> the cat
                return f"{flip(' and '.join(k['who']))}."
            return "i'm not sure what."
        # yes / no: does what was kept say the same thing - directly, or by joining facts?
        said = self._keywords(f, parsed)
        missing = {w for w in said if not any(self._same(w, x) for x in k["words"])}
        if not missing:
            return "yes."
        chain_words = {w for b in self._where_chain(k) for w in b["it"].split()}
        if missing and all(any(self._same(w, x) for x in chain_words) for w in missing):
            return "yes, " + ", and that's ".join(say(b) for b in self._where_chain(k)) + "."
        return f"no, not that i know. you told me: {flip(k['text'])}."


def chat():
    from household.relations import load_test
    m = Memory(load_test())
    talk = ["my phone is on the table", "where is my phone", "Kemi put her umbrella under the chair at night",
            "where is the umbrella", "when did kemi put it there", "who put the umbrella under the chair",
            "dad has a black car", "what does dad have", "who has a black car", "we ran to the shop because it rained",
            "why did we run to the shop", "where did we run", "is my phone on the table", "is my phone in the kitchen",
            "where are my keys", "i fixed the laptop yesterday", "when did i fix the laptop", "what did i fix"]
    for t in talk:
        print(f"  you > {t}\n  coal> {m.hear(t)}")


if __name__ == "__main__":
    import sys
    {"chat": chat}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
