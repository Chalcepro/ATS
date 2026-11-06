"""Coal skims websites (Chalce, 11-06): reads pages from an allowlist and keeps the plain text to learn from.

    py -3.12 -m household.skim fetch [site ...] [--pages=20]   (read pages; saves text in data/reading/<site>/)
    py -3.12 -m household.skim read                            (what Coal found: sentences, words he never met)

Read-only and polite: only the sites below, only links inside the same site, robots.txt obeyed, one page every
few seconds, no cookies, no logins, no forms. Nothing is sent but the page address.
Python's standard library only, so it runs wherever Coal runs.
What is kept: the page's readable text, cut into sentences in Coal's own words() - no HTML, no scripts.
The text feeds his lessons: the word layer meets new words, the grammar layer reads real sentences.
"""
from __future__ import annotations

import html.parser
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser

ALLOW = {                                      # site -> first page; add more here (Chalce's list)
    "simple.wikipedia.org": "https://simple.wikipedia.org/wiki/Main_Page",
    "www.w3schools.com": "https://www.w3schools.com/html/default.asp",
    "react.dev": "https://react.dev/learn",
    "en.wikipedia.org": "https://en.wikipedia.org/wiki/Main_Page",
    "www.gutenberg.org": "https://www.gutenberg.org/ebooks/search/?sort_order=downloads",
}
AGENT = "CoalReader/0.1 (personal learning, read-only; https://github.com/Chalcepro/ATS)"
WAIT = 3.0                                     # seconds between pages
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reading")
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form", "aside"}


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.skip, self.title = [], [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "div", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def sentences(text: str) -> list[str]:
    """Readable sentences only: at least 4 words, mostly letters."""
    out = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        chunk = " ".join(chunk.split())
        if len(chunk.split()) >= 4 and sum(c.isalpha() for c in chunk) > 0.6 * len(chunk):
            out.append(chunk)
    return out


def _slug(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return (re.sub(r"[^a-zA-Z0-9]+", "_", p.path + ("_" + p.query if p.query else "")).strip("_") or "index")[:120]


def fetch(sites=None, pages=20, log=print):
    for site in sites or ALLOW:
        if site not in ALLOW:
            log(f"{site}: not on the allowlist, skipped")
            continue
        robots = urllib.robotparser.RobotFileParser()
        try:                                                             # read robots.txt as ourselves (some sites
            req = urllib.request.Request(f"https://{site}/robots.txt", headers={"User-Agent": AGENT})  # refuse Python's name)
            with urllib.request.urlopen(req, timeout=20) as r:
                robots.parse(r.read().decode("utf-8", "ignore").splitlines())
        except Exception as e:                                           # no robots.txt read -> do not read the site
            log(f"{site}: robots.txt unreadable ({e}), skipped")
            continue
        folder = os.path.join(HERE, site)
        os.makedirs(folder, exist_ok=True)
        queue, seen, got = [ALLOW[site]], set(), 0
        while queue and got < pages:
            url = queue.pop(0).split("#")[0]
            if url in seen or not robots.can_fetch(AGENT, url):
                continue
            seen.add(url)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": AGENT})
                with urllib.request.urlopen(req, timeout=20) as r:
                    if "text/html" not in r.headers.get("Content-Type", ""):
                        continue
                    page = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", "ignore")
            except Exception as e:
                log(f"  {url}: {e}")
                continue
            p = _Text()
            p.feed(page)
            sents = sentences("".join(p.parts))
            if sents:
                with open(os.path.join(folder, _slug(url) + ".txt"), "w", encoding="utf-8") as f:
                    f.write(f"# {p.title.strip()}\n# {url}\n" + "\n".join(sents) + "\n")
                got += 1
                log(f"  {site}: {p.title.strip()[:60]} ({len(sents)} sentences)")
            for h in p.links:
                u = urllib.parse.urljoin(url, h)
                q = urllib.parse.urlparse(u)
                if q.netloc == site and u not in seen and ":" not in q.path:   # no Special:/Talk:/Help: pages
                    queue.append(u)
            time.sleep(WAIT)


def texts():
    for root, _, files in os.walk(HERE):
        for f in files:
            if f.endswith(".txt"):
                lines = open(os.path.join(root, f), encoding="utf-8").read().splitlines()
                yield os.path.basename(root), [x for x in lines if not x.startswith("# ")]


def read(log=print):
    """What Coal found: how much, and which words his word layer has never met."""
    from household.model import load
    from household.train import SAVE
    from household.understand import words
    brain, _ = load(SAVE)                                                 # read only
    known = set(brain.keep["mind"]["wordlayer"]["words"])
    per_site, new = {}, {}
    for site, lines in texts():
        per_site[site] = per_site.get(site, 0) + len(lines)
        for x in lines:
            for w in words(x):
                if w.isalpha() and w not in known:
                    new[w] = new.get(w, 0) + 1
    for s, n in per_site.items():
        log(f"{s}: {n} sentences")
    top = sorted(new.items(), key=lambda kv: -kv[1])
    log(f"{len(new)} words he never met; most common: " + ", ".join(f"{w} ({n})" for w, n in top[:30]))
    json.dump(dict(top), open(os.path.join(HERE, "new_words.json"), "w", encoding="utf-8"))


if __name__ == "__main__":
    a = sys.argv[1:]
    n = next((int(x[8:]) for x in a if x.startswith("--pages=")), 20)
    {"fetch": lambda: fetch([x for x in a[1:] if not x.startswith("--")] or None, n), "read": read}.get(
        a[0] if a else "", lambda: print(__doc__))()
