"""Coal digests what he read (Chalce, 11-06): every sentence from the skimmed sites becomes a kept meaning.

    py -3.12 -m household.digest run          (parse data/reading/ into data/reading/knowledge.jsonl; resumes)
    py -3.12 -m household.digest ask "what is a closure"

Same three layers as talk: word types, how the words hang together, the frame (who / does / what / is / where /
when / why). What is kept is the frame plus where it was read, never the page - so an answer can say its source.
A question is read the same way and answered from the kept frames that share its words (comprehend.Memory).
"""
from __future__ import annotations

import json
import os
import sys

from household.comprehend import Memory
from household.relations import frame
from household.skim import HERE

KNOW = os.path.join(HERE, "knowledge.jsonl")
MAX_WORDS = 40                                       # longer "sentences" are usually menus or code, not English


def _pages():
    for root, _, files in os.walk(HERE):
        for f in sorted(files):
            if f.endswith(".txt"):
                lines = open(os.path.join(root, f), encoding="utf-8").read().splitlines()
                yield os.path.basename(root), lines[1][2:] if len(lines) > 1 else "", [x for x in lines[2:]]


def run(log=print):
    import torch
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))       # leave the other cores to training
    from household.relations import load_test
    m = Memory(load_test())
    done = set()
    if os.path.exists(KNOW):
        done = {json.loads(x)["from"] for x in open(KNOW, encoding="utf-8")}
    n = 0
    with open(KNOW, "a", encoding="utf-8") as out:
        for site, url, lines in _pages():
            if url in done:
                continue
            lines = [x for x in lines if 3 <= len(x.split()) <= MAX_WORDS]
            for i in range(0, len(lines), 64):
                for parsed in m.parser.parse_many(lines[i:i + 64]):
                    if len(parsed) < 3:
                        continue
                    m.keep(frame(parsed), parsed)
                    k = m.kept.pop()
                    k = {a: (sorted(b) if isinstance(b, set) else b) for a, b in k.items()}
                    out.write(json.dumps(dict(k, site=site, **{"from": url}), ensure_ascii=False) + "\n")
                    n += 1
            log(f"  {site}: {url} digested ({n} meanings so far)", flush=True)
            done.add(url)


def memory():
    from household.relations import load_test
    m = Memory(load_test())
    for x in open(KNOW, encoding="utf-8"):
        k = json.loads(x)
        k["words"], k["nouns"] = set(k["words"]), set(k.get("nouns", []))
        m.kept.append(k)
    return m


def ask(q: str):
    m = memory()
    print(f"  ({len(m.kept)} meanings kept)\n  you > {q}\n  coal> {m.hear(q)}")
    found = m._find(m.parser.parse(q))
    if found:
        print(f"        (read in: {found[0]['text'][:100]}  -  {found[0]['from']})")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"run": run, "ask": lambda: ask(" ".join(a[1:]))}.get(a[0] if a else "", lambda: print(__doc__))()
