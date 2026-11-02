"""Any file with words in it -> plain text, for Coal to read.

    py -3.12 -m household.to_text FILE_OR_FOLDER [OUT_FOLDER]

Writes one .txt per file (into OUT_FOLDER, default "texts" beside the input).
Reads: .txt .md .csv .json .html .htm .rtf .pdf .docx .pptx .xlsx .odt .odp .ods .epub, and
source code. Office, OpenDocument and EPUB files are zipped XML, read with plain Python;
PDF needs pypdf (pip install pypdf). Anything else is tried as text and skipped if it is not.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import zipfile
from html.parser import HTMLParser

PLAIN = {".txt", ".md", ".csv", ".log", ".py", ".js", ".ts", ".lua", ".c", ".cpp", ".h", ".java", ".css", ".ini", ".yaml", ".yml"}


class _Text(HTMLParser):
    """The words of a web page, a line per block, no scripts or styles."""
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "td", "th", "title"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def from_html(s: str) -> str:
    p = _Text()
    p.feed(s)
    return "".join(p.out)


def from_xml(s: str, para: str) -> str:
    """Office/OpenDocument XML: a line break at each paragraph, then the tags go."""
    s = re.sub(rf"</?{para}[ >/]", "\n<", s)
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def from_zip(path: str, ext: str) -> str:
    z = zipfile.ZipFile(path)
    names = z.namelist()
    num = lambda n: int(re.findall(r"\d+", n)[-1]) if re.findall(r"\d+", n) else 0
    if ext == ".docx":
        parts, para = [n for n in names if re.match(r"word/(document|footnotes|endnotes)\.xml", n)], "w:p"
    elif ext == ".pptx":
        parts, para = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml", n)), key=num), "a:p"
    elif ext == ".xlsx":
        parts, para = [n for n in names if n == "xl/sharedStrings.xml" or re.match(r"xl/worksheets/sheet\d+\.xml", n)], "(si|row)"
    elif ext in (".odt", ".odp", ".ods"):
        parts, para = ["content.xml"], "text:(p|h)"
    else:                                                     # .epub: its pages are html
        return "\n".join(from_html(z.read(n).decode("utf-8", "ignore")) for n in sorted(names) if n.endswith((".xhtml", ".html", ".htm")))
    return "\n".join(from_xml(z.read(n).decode("utf-8", "ignore"), para) for n in parts)


def from_rtf(s: str) -> str:
    s = re.sub(r"\\par[d]?", "\n", s)
    s = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), s)
    s = re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", s)
    return s


def to_text(path: str) -> str | None:
    """The words in one file, or None if it has none we can read."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    if ext in (".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".epub"):
        return from_zip(path, ext)
    raw = open(path, "rb").read()
    if b"\0" in raw[:4096]:
        return None                                           # a picture, a program: no words
    s = raw.decode("utf-8", "ignore")
    if ext in (".html", ".htm"):
        return from_html(s)
    if ext == ".rtf":
        return from_rtf(s)
    if ext == ".json":
        return "\n".join(str(v) for v in _strings(json.loads(s)))
    return s


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def tidy(s: str) -> str:
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip() + "\n"


def convert(src: str, out: str | None = None) -> list[str]:
    files = [os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [src]
    out = out or os.path.join(os.path.dirname(os.path.abspath(src)), "texts")
    os.makedirs(out, exist_ok=True)
    made = []
    for f in files:
        try:
            t = to_text(f)
        except Exception as e:
            print(f"  skipped {os.path.basename(f)}: {type(e).__name__}: {e}")
            continue
        if not t or not t.strip():
            continue
        dst = os.path.join(out, os.path.splitext(os.path.basename(f))[0] + ".txt")
        open(dst, "w", encoding="utf-8").write(tidy(t))
        made.append(dst)
        print(f"  {os.path.basename(f)} -> {os.path.basename(dst)} ({len(t.split())} words)")
    return made


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
