"""CoalOS storage: the pretend computer's whole disk is one SQLite file.

Nothing in here maps to a real folder: a path like "/home/Documents/a.txt"
is only a key in the `nodes` table. Files dropped in from the real machine
are copied in as bytes and from then on live only here.

To keep the laptop's disk from filling up:
  - text, documents and code are compressed with LZMA (often 3-10x smaller)
  - photos, music and video are already compressed, so they are stored as
    they are - squeezing them again wastes time for ~0% gain
  - every file is stored once by its content hash: drop the same song in
    ten folders and it takes the space of one
"""
from __future__ import annotations

import hashlib
import lzma
import posixpath
import sqlite3
import threading
import time

# already-compressed kinds: stored raw
RAW = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac",
       ".mp4", ".mkv", ".webm", ".mov", ".avi", ".zip", ".7z", ".rar", ".gz", ".pdf", ".docx", ".xlsx", ".pptx"}

# The pretend disk is small on purpose: Coal has to live within limits, tidy
# up and restructure, not just keep piling files on.
DISK = 2 * 1024 ** 3            # 2 GB, counted as the files' real (uncompressed) size

START = ["/home", "/home/Desktop", "/home/Documents", "/home/Music", "/home/Pictures", "/home/Videos",
         "/home/Code", "/tmp"]


def clean(path: str) -> str:
    """Any path, made into one inside this disk: '..' can never climb above '/'."""
    p = posixpath.normpath("/" + (path or "/").replace("\\", "/"))
    return "/" if p in (".", "//") else p.replace("//", "/")


class Store:
    def __init__(self, db: str):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, codec TEXT, size INTEGER, data BLOB);
            CREATE TABLE IF NOT EXISTS nodes (path TEXT PRIMARY KEY, kind TEXT, hash TEXT, mtime REAL);
        """)
        if not self.exists("/"):
            for d in ["/"] + START:
                self._put_node(d, "dir", None)
            self.write("/home/Desktop/Welcome.txt",
                       b"Welcome to CoalOS.\n\nThis computer is pretend: its disk is one file on the real one.\n"
                       b"Drag files from your real desktop onto this one to give Coal something to work with.\n")
            self.db.commit()

    # ---- nodes -----------------------------------------------------------------
    def _put_node(self, path, kind, h):
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?)", (path, kind, h, time.time()))

    def exists(self, path: str) -> bool:
        return self.db.execute("SELECT 1 FROM nodes WHERE path=?", (clean(path),)).fetchone() is not None

    def kind(self, path: str) -> str | None:
        r = self.db.execute("SELECT kind FROM nodes WHERE path=?", (clean(path),)).fetchone()
        return r[0] if r else None

    def ls(self, path: str) -> list[dict]:
        d = clean(path)
        pre = "/" if d == "/" else d + "/"
        rows = self.db.execute(
            "SELECT n.path, n.kind, n.mtime, b.size, length(b.data) FROM nodes n LEFT JOIN blobs b ON n.hash=b.hash "
            "WHERE n.path LIKE ? AND n.path != ?", (pre + "%", d)).fetchall()
        out = [{"name": p[len(pre):], "path": p, "kind": k, "mtime": m, "size": s or 0, "stored": st or 0}
               for p, k, m, s, st in rows if "/" not in p[len(pre):]]
        return sorted(out, key=lambda e: (e["kind"] != "dir", e["name"].lower()))

    # ---- files -----------------------------------------------------------------
    def write(self, path: str, data: bytes) -> dict:
        p = clean(path)
        parent = posixpath.dirname(p)
        with self.lock:
            if self.kind(parent) != "dir":
                raise FileNotFoundError(f"no folder {parent}")
            if self.kind(p) == "dir":
                raise IsADirectoryError(p)
            old = self.db.execute("SELECT b.size FROM nodes n JOIN blobs b ON n.hash=b.hash WHERE n.path=?", (p,)).fetchone()
            if self.stats()["size"] - (old[0] if old else 0) + len(data) > DISK:
                raise OSError("the disk is full - delete or move something first")
            h = hashlib.sha256(data).hexdigest()
            if not self.db.execute("SELECT 1 FROM blobs WHERE hash=?", (h,)).fetchone():
                ext = posixpath.splitext(p)[1].lower()
                if ext in RAW or len(data) < 64:
                    codec, stored = "raw", data
                else:
                    packed = lzma.compress(data, preset=6)
                    codec, stored = ("lzma", packed) if len(packed) < len(data) else ("raw", data)
                self.db.execute("INSERT INTO blobs VALUES (?,?,?,?)", (h, codec, len(data), stored))
            self._put_node(p, "file", h)
            self._sweep()
            self.db.commit()
        return {"path": p, "size": len(data)}

    def read(self, path: str) -> bytes:
        r = self.db.execute("SELECT b.codec, b.data FROM nodes n JOIN blobs b ON n.hash=b.hash WHERE n.path=?",
                            (clean(path),)).fetchone()
        if r is None:
            raise FileNotFoundError(path)
        codec, data = r
        return lzma.decompress(data) if codec == "lzma" else bytes(data)

    def mkdir(self, path: str) -> None:
        p = clean(path)
        with self.lock:
            if self.kind(posixpath.dirname(p)) != "dir":
                raise FileNotFoundError(posixpath.dirname(p))
            if not self.exists(p):
                self._put_node(p, "dir", None)
                self.db.commit()

    def move(self, src: str, dst: str, keep: bool = False) -> str:
        """Move (or copy, keep=True) a file or folder. A folder as dst = into it."""
        s, d = clean(src), clean(dst)
        with self.lock:
            if not self.exists(s):
                raise FileNotFoundError(s)
            if self.kind(d) == "dir":
                d = clean(d + "/" + posixpath.basename(s))
            if d == s or d.startswith(s + "/"):
                raise ValueError("cannot move a folder into itself")
            rows = self.db.execute("SELECT path, kind, hash FROM nodes WHERE path=? OR path LIKE ?",
                                   (s, s + "/%")).fetchall()
            for p, k, h in rows:
                self._put_node(d + p[len(s):], k, h)
                if not keep:
                    self.db.execute("DELETE FROM nodes WHERE path=?", (p,))
            self._sweep()
            self.db.commit()
        return d

    def remove(self, path: str) -> None:
        p = clean(path)
        if p in ["/"] + START:
            raise PermissionError(f"{p} is part of the system")
        with self.lock:
            self.db.execute("DELETE FROM nodes WHERE path=? OR path LIKE ?", (p, p + "/%"))
            self._sweep()
            self.db.commit()

    def _sweep(self) -> None:
        """Drop stored contents no file points at any more."""
        self.db.execute("DELETE FROM blobs WHERE hash NOT IN (SELECT hash FROM nodes WHERE hash IS NOT NULL)")

    def stats(self) -> dict:
        files, real = self.db.execute(
            "SELECT count(*), coalesce(sum(b.size),0) FROM nodes n JOIN blobs b ON n.hash=b.hash").fetchone()
        stored = self.db.execute("SELECT coalesce(sum(length(data)),0) FROM blobs").fetchone()[0]
        return {"files": files, "size": real, "stored": stored, "disk": DISK}
