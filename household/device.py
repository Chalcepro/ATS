"""Coal finds out what device he is on (Chalce, 11-06): nothing is assumed, he looks.

    py -3.12 -m household.device look        (what he finds, printed; nothing is changed or saved)

No list of known apps, no "if a video app exists". On any device he looks at what is there and keeps what
he found as plain meanings - "filmora is an app on this device", "filmora : wondershare filmora video
editor" - in the same memory his chat answers from. So "what can i edit a video with?" is answered in the
one chat, by the same reading and picking as everything else, whatever the app is called, in any language.

What he looks at (Python's own library + Windows' own calls through ctypes; nothing to install):
  places   - his person's folders and the drives
  files    - how many of each kind are where (by what they are, read from the file's own ending/contents)
  apps     - every app the device lists for its person (Start menu shortcuts, installed programs), with the
             app's OWN description from inside its file (the words its maker wrote: "video editor", ...)
  windows  - what is open on the screen now, which one is in front
  online   - whether the internet answers at all (he works either way)
Elsewhere (a phone, Linux) the same look runs with what that system offers; anything missing is skipped.
"""
from __future__ import annotations

import ctypes
import os
import socket
import string
import sys

HOME = os.path.expanduser("~")


# ---- places and files -------------------------------------------------------------------------
def places():
    out = [os.path.join(HOME, d) for d in sorted(os.listdir(HOME)) if not d.startswith(".") and d != "AppData"
           and os.path.isdir(os.path.join(HOME, d)) and not os.path.isjunction(os.path.join(HOME, d))]
    if os.name == "nt":
        out += [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    return out


KINDS = {"picture": "png jpg jpeg gif webp bmp heic svg", "video": "mp4 mov mkv avi webm wmv",
         "sound": "mp3 wav m4a flac ogg aac", "document": "pdf docx doc odt rtf txt md pptx xlsx csv epub",
         "code": "py js ts html css lua c cpp h java json", "program": "exe msi apk lnk bat",
         "archive": "zip rar 7z tar gz", "3d model": "blend fbx obj glb gltf stl"}
ENDING = {e: k for k, es in KINDS.items() for e in es.split()}


def files(folder, depth=2, limit=4000):
    """How many of each kind of file, and a few names, in a folder (not deeper than `depth`)."""
    counts, names, seen = {}, {}, 0
    for root, dirs, fs in os.walk(folder):
        if root[len(folder):].count(os.sep) >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "AppData")]
        for f in fs:
            k = ENDING.get(f.rsplit(".", 1)[-1].lower(), None) if "." in f else None
            if k:
                counts[k] = counts.get(k, 0) + 1
                names.setdefault(k, []).append(f) if len(names.get(k, [])) < 5 else None
            seen += 1
            if seen > limit:
                return counts, names
    return counts, names


# ---- apps, with their makers' own words ---------------------------------------------------------
def _describe(exe):
    """The description its maker put inside the program file (Windows version info), if any."""
    if os.name != "nt" or not exe or not os.path.exists(exe):
        return ""
    try:
        v = ctypes.windll.version
        size = v.GetFileVersionInfoSizeW(exe, None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        v.GetFileVersionInfoW(exe, 0, size, buf)
        p, n = ctypes.c_void_p(), ctypes.c_uint()
        if not v.VerQueryValueW(buf, "\\VarFileInfo\\Translation", ctypes.byref(p), ctypes.byref(n)) or not n.value:
            return ""
        lang = ctypes.cast(p, ctypes.POINTER(ctypes.c_ushort * 2)).contents
        out = []
        for key in ("FileDescription", "ProductName", "CompanyName"):
            q = f"\\StringFileInfo\\{lang[0]:04x}{lang[1]:04x}\\{key}"
            if v.VerQueryValueW(buf, q, ctypes.byref(p), ctypes.byref(n)) and n.value:
                out.append(ctypes.wstring_at(p, n.value - 1))
        return " ; ".join(dict.fromkeys(x.strip() for x in out if x.strip()))
    except Exception:
        return ""


def _shortcut_target(lnk):
    """Where a Windows shortcut points: read from the .lnk file itself (its local path field)."""
    try:
        data = open(lnk, "rb").read()
        flags = int.from_bytes(data[20:24], "little")
        pos = 76
        if flags & 1:                                              # has a target id list
            pos += 2 + int.from_bytes(data[76:78], "little")
        if flags & 2:                                              # has link info
            start = pos
            local = int.from_bytes(data[start + 16:start + 20], "little")
            end = data.index(b"\0", start + local)
            return data[start + local:end].decode("mbcs", "ignore")
    except Exception:
        pass
    return ""


def apps():
    """Every app the device lists for its person: name, where it is, and its maker's own description."""
    found = {}
    if os.name == "nt":
        menus = [os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
                 os.path.join(os.environ.get("PROGRAMDATA", ""), r"Microsoft\Windows\Start Menu\Programs")]
        for m in menus:
            for root, _, fs in os.walk(m):
                for f in fs:
                    if f.lower().endswith(".lnk") and "uninstall" not in f.lower():
                        name = f[:-4]
                        exe = _shortcut_target(os.path.join(root, f))
                        found.setdefault(name.lower(), {"name": name, "exe": exe, "about": _describe(exe),
                                                        "group": os.path.basename(root) if root != m else ""})
    else:                                                           # elsewhere: what the system lists as apps
        for d in ("/usr/share/applications", os.path.join(HOME, ".local/share/applications")):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".desktop"):
                        info = dict(l.split("=", 1) for l in open(os.path.join(d, f), errors="ignore") if "=" in l)
                        found.setdefault(f, {"name": info.get("Name", f).strip(), "exe": info.get("Exec", "").strip(),
                                             "about": (info.get("Comment", "") + " " + info.get("GenericName", "")).strip(),
                                             "group": info.get("Categories", "").strip()})
    return sorted(found.values(), key=lambda a: a["name"].lower())


def opens():
    """Which programs the device says open which kinds of file: {program file name: {kinds}}.
    Read from the system's own file links (Windows: each ending's "open with" lists)."""
    out = {}
    if os.name != "nt":
        return out
    import winreg
    def names(key, sub):
        try:
            with winreg.OpenKey(key, sub) as k:
                i, got = 0, []
                while True:
                    try:
                        got.append(winreg.EnumValue(k, i)[0])
                    except OSError:
                        try:
                            got.append(winreg.EnumKey(k, i))
                        except OSError:
                            return got
                    i += 1
        except OSError:
            return []
    for ending, kind in ENDING.items():
        for sub in (rf"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.{ending}\OpenWithList",
                    rf"Software\Classes\.{ending}\OpenWithProgids"):
            for v in names(winreg.HKEY_CURRENT_USER, sub):
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub) as k:
                        v = winreg.QueryValueEx(k, v)[0] if sub.endswith("OpenWithList") else v
                except OSError:
                    pass
                if isinstance(v, str) and v and v != "MRUList":
                    out.setdefault(v.lower().split(".exe")[0].split(".")[0], set()).add(kind)
        for v in names(winreg.HKEY_CLASSES_ROOT, rf".{ending}\OpenWithProgids"):
            out.setdefault(v.lower().split(".")[0], set()).add(kind)
    return out


# ---- the screen now ----------------------------------------------------------------------------
def windows():
    """Open windows with a title: [(title, is it in front)]."""
    if os.name != "nt":
        return []
    u = ctypes.windll.user32
    out, front = [], u.GetForegroundWindow()
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def each(h, _):
        if u.IsWindowVisible(h):
            n = u.GetWindowTextLengthW(h)
            if n:
                b = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(h, b, n + 1)
                out.append((b.value, h == front))
        return True
    u.EnumWindows(proto(each), 0)
    return out


def online(timeout=2.0):
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=timeout).close()
        return True
    except OSError:
        return False


# ---- what he keeps: plain sentences, read like anything else ----------------------------------------
def meanings():
    """What he found, as sentences for his memory (the same reading as talk and pages)."""
    out = []
    opened = opens()
    for a in apps():
        n = a["name"].lower()
        out.append(f"{n} is an app on this device")
        exe = os.path.splitext(os.path.basename(a["exe"]))[0].lower() if a["exe"] else ""
        kinds = set()
        for prog, ks in opened.items():
            if prog and (prog == exe or (len(prog) > 3 and prog in n.replace(" ", ""))):
                kinds |= ks
        for k in sorted(kinds):
            out.append(f"{n} opens {k} files")
        if a["about"]:
            out.append(f"{n} is {a['about'].lower().replace(';', ',')}")
        if a["group"]:
            out.append(f"{n} is in the {a['group'].lower()} group")
    for p in places()[:20]:
        if os.path.splitdrive(p)[1] in ("\\", "/"):
            out.append(f"the {p[0].lower()} drive is a place on this device")
            continue
        counts, names = files(p, depth=1)
        for k, c in counts.items():
            out.append(f"the {os.path.basename(p).lower()} folder has {c} {k} files")
    out.append("this device is online" if online() else "this device is offline")
    return out


def look():
    sys.stdout.reconfigure(errors="replace")
    a = apps()
    print(f"{len(a)} apps. Some, with their makers' own words:")
    for x in [x for x in a if x["about"]][:25]:
        print(f"  {x['name']:32s} {x['about'][:70]}")
    print(f"\nplaces: {', '.join(places()[:12])}")
    w = windows()
    print(f"\n{len(w)} windows open; in front: {next((t for t, f in w if f), '-')}")
    print(f"online: {online()}")
    m = meanings()
    print(f"\n{len(m)} meanings he would keep, e.g.:\n  " + "\n  ".join(m[:12]))


if __name__ == "__main__":
    {"look": look}.get(sys.argv[1] if sys.argv[1:] else "", lambda: print(__doc__))()
