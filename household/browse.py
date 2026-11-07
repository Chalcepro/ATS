"""Coal looks things up (Chalce, 11-07): when he doesn't know, he goes and reads, like a person with a phone.

    py -3.12 -m household.browse look banana       (open the page for it, print what he would keep; nothing saved)

What happens after he fails is the point: he didn't know -> he opens the page named after the thing (as a person
types it into the address bar), reads it as sentences, and keeps them in the same memory his one chat answers
from (site "web"). Then he tries the question again. Pages read are kept on disk, so offline he still has them.

Read-only, robots.txt obeyed as himself (skim.AGENT), only sites on skim.ALLOW, one page at a time, 3 s apart.
Search pages are refused by the sites' robots.txt, so he goes straight to the thing's own page.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser

from household.skim import AGENT, ALLOW, HERE, WAIT, _Text, _slug, sentences

LOOK_IN = ("simple-wikipedia", "en-wikipedia")             # plain words first, then the bigger one
_robots, _last = {}, [0.0]


def _allowed(url: str) -> bool:
    site = urllib.parse.urlparse(url).netloc
    if site not in _robots:
        r = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(f"https://{site}/robots.txt", headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=15) as f:
                r.parse(f.read().decode("utf-8", "ignore").splitlines())
        except Exception:
            r = None                                                 # unreadable robots.txt -> not read
        _robots[site] = r
    return bool(_robots[site] and _robots[site].can_fetch(AGENT, url))


def open_page(url: str):
    """(title, sentences, final url) or None. Waits between pages; follows the site's own redirects."""
    if not _allowed(url):
        return None
    time.sleep(max(0.0, WAIT - (time.time() - _last[0])))
    _last[0] = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            if "text/html" not in r.headers.get("Content-Type", ""):
                return None
            page = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", "ignore")
            url = r.geturl()
    except Exception:
        return None
    p = _Text()
    p.feed(page)
    sents = sentences("".join(p.parts))
    if not sents or "may refer to" in " ".join(sents[:3]).lower():      # a list of other pages, not the thing
        return None
    return p.title.strip(), sents, url


def look(thing: str, log=lambda *a, **k: None):
    """Open the page for `thing` on the first allowed site that has one. Saved like skimmed pages.
    Returns (site name, url, sentences) or None."""
    name = "_".join(w.capitalize() if i == 0 else w for i, w in enumerate(thing.split()))
    for site in LOOK_IN:
        host = ALLOW[site][0]
        url = f"https://{host}/wiki/{urllib.parse.quote(name)}"
        folder = os.path.join(HERE, site)
        saved = os.path.join(folder, _slug(urllib.parse.urlparse(url).path) + ".txt")
        if os.path.exists(saved):                                   # read before: offline is fine
            lines = open(saved, encoding="utf-8").read().splitlines()
            return site, lines[1][2:], lines[2:]
        got = open_page(url)
        if not got:
            continue
        title, sents, final = got
        os.makedirs(folder, exist_ok=True)
        with open(saved, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n# {final}\n" + "\n".join(sents) + "\n")
        log(f"  read {title} ({len(sents)} sentences)")
        return site, final, sents
    return None


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    if sys.argv[1:2] == ["look"]:
        got = look(" ".join(sys.argv[2:]), log=print)
        print("nothing found" if not got else f"{got[0]}: {got[1]}\n  " + "\n  ".join(got[2][:8]))
    else:
        print(__doc__)
