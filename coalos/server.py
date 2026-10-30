"""CoalOS: a pretend computer for Coal, in the browser, on this machine only.

Run: python -m coalos.server        then open http://127.0.0.1:8766

The page is the screen; this server is the disk (coalos/store.py). It only
ever reads and writes the one database file - it has no route that touches a
real folder, and code typed into the pretend terminal or editor runs inside
the browser's own sandbox (Python via Pyodide, JavaScript in a Web Worker),
never on this machine.
"""
from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from coalos.store import Store, clean

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PORT = 8766
MAX_UPLOAD = 1 << 30            # 1 GB a file

S: Store | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, kind: str = "application/json", code: int = 200, extra=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), code=code)

    def _q(self):
        u = urlparse(self.path)
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    def _fail(self, e: Exception) -> None:
        code = 404 if isinstance(e, FileNotFoundError) else 403 if isinstance(e, PermissionError) else 507 if type(e) is OSError else 400
        self._json({"error": f"{type(e).__name__}: {e}"}, code)

    def do_GET(self):
        route, q = self._q()
        try:
            if route in ("/", "/index.html"):
                return self._send(open(os.path.join(HERE, "web", "index.html"), "rb").read(), "text/html; charset=utf-8")
            if route.startswith("/web/"):
                name = os.path.basename(route)
                f = os.path.join(HERE, "web", name)
                if os.path.isfile(f):
                    return self._send(open(f, "rb").read(), mimetypes.guess_type(name)[0] or "text/plain")
            if route == "/api/ls":
                return self._json(S.ls(q.get("path", "/home")))
            if route == "/api/file":
                p = clean(q.get("path", ""))
                kind = mimetypes.guess_type(p)[0] or "application/octet-stream"
                extra = {"Content-Disposition": f'attachment; filename="{os.path.basename(p)}"'} if q.get("download") else None
                return self._send(S.read(p), kind, extra=extra)
            if route == "/api/stats":
                return self._json(S.stats())
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._fail(e)

    def do_POST(self):
        route, q = self._q()
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            return self._json({"error": "file too big for CoalOS (1 GB max)"}, 413)
        raw = self.rfile.read(n) if n else b""
        try:
            if route == "/api/upload":               # bytes in, from a drag-and-drop or paste
                return self._json(S.write(clean(q["dir"] + "/" + unquote(q["name"])), raw))
            body = json.loads(raw or b"{}")
            if route == "/api/write":
                return self._json(S.write(body["path"], body.get("text", "").encode("utf-8")))
            if route == "/api/mkdir":
                S.mkdir(body["path"])
                return self._json({"ok": True})
            if route == "/api/move":
                return self._json({"path": S.move(body["src"], body["dst"], keep=bool(body.get("copy")))})
            if route == "/api/remove":
                S.remove(body["path"])
                return self._json({"ok": True})
            self._json({"error": "not found"}, 404)
        except Exception as e:
            self._fail(e)


def main() -> None:
    global S
    os.makedirs(DATA, exist_ok=True)
    S = Store(os.path.join(DATA, "disk.db"))
    print(f"CoalOS: http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
