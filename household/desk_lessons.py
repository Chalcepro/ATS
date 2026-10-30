"""Gen1 (Chalce, 10-30): desktop work becomes Coal's main world, in the same brain.

Coal is asked something in words, then looks at one file at a time - what kind
of file it is and which folder it sits in - and decides what to do with it:
leave it, move it to one of the usual folders, or delete it.

The teacher knows what the person wants, the way the household judge knows the
hazards, and teaches by example only:
  - only what was asked is done; a file the ask did not cover is left alone
    ("tidy my desktop" never deletes anything)
  - a file is judged by what it is, not where it sits: an essay saved in the
    temp folder is not a temp file, so "clear the temp files" leaves it
  - a file already where it belongs is left

The ask reaches the brain as its words, hashed into a small bag (DESK_WORDS), so a
new way of saying a known ask can still land; a word never heard does not.
"""
from __future__ import annotations

import random
import re
import zlib

DESK_ACTS = ("leave", "to_photos", "to_music", "to_videos", "to_docs", "delete")
KINDS = ("image", "music", "video", "doc", "text", "code", "temp", "log")
FOLDERS = ("home", "Desktop", "Downloads", "temp", "Photos", "Music", "Videos", "Documents")
DESK_WORDS = 64
EXT = {"image": ("jpg", "png", "jpeg", "gif"), "music": ("mp3", "wav", "ogg"), "video": ("mp4", "mkv", "webm"),
       "doc": ("docx", "doc", "pdf"), "text": ("txt", "md"), "code": ("py", "js", "html"),
       "temp": ("tmp", "cache", "part"), "log": ("log",)}
NAMES = {"image": ["beach", "mum", "party", "dog", "sunset", "school", "wedding", "selfie"],
         "music": ["song", "afrobeat", "track01", "mix", "voice_note", "beat"],
         "video": ["clip", "birthday", "match", "tutorial", "vlog"],
         "doc": ["essay_final", "cv", "invoice", "report", "letter", "contract"],
         "text": ["notes", "todo", "groceries", "ideas", "readme"],
         "code": ["main", "app", "index", "server", "game"],
         "temp": ["cache0", "~tmp", "download", "thumbs", "session"],
         "log": ["error", "install", "debug", "server", "update"]}
HOME_OF = {"to_photos": "Photos", "to_music": "Music", "to_videos": "Videos", "to_docs": "Documents"}


def desk_names() -> list[str]:
    return ([f"desk.word{i}" for i in range(DESK_WORDS)] + [f"desk.kind.{k}" for k in KINDS]
            + [f"desk.in.{f}" for f in FOLDERS] + ["desk.asked"])


DESK_SIZE = len(desk_names())


def desk_obs(ask: str, kind: str, folder: str) -> list[float]:
    bag = [0.0] * DESK_WORDS
    for w in re.findall(r"[a-z]+", ask.lower()):
        bag[zlib.crc32(w.encode()) % DESK_WORDS] = 1.0
    return bag + [float(k == kind) for k in KINDS] + [float(f == folder) for f in FOLDERS] + [1.0]


# ---- what can be asked: (task, ways of saying it). Exam phrasings are kept apart. -----
ASKS = {
    "photos": ["put my photos in photos", "move the pictures to my photos folder", "sort my photos",
               "put the pictures away", "file my images", "collect my photos into photos"],
    "music": ["put my music in music", "move the songs to the music folder", "sort my songs",
              "file my audio", "put the tracks in music"],
    "videos": ["put my videos in videos", "move the clips to the videos folder", "sort my videos",
               "file the videos away"],
    "docs": ["put my documents in documents", "move my documents away", "file my documents",
             "sort the documents into documents", "put the pdfs and letters in documents"],
    "tidy": ["tidy my desktop", "tidy up my files", "clean up my desktop", "organise my files",
             "sort out this mess", "put everything where it belongs"],
    "temp": ["clear the temp files", "delete the temporary files", "clean out temp", "get rid of the cache",
             "remove the junk temp files"],
    "logs": ["delete the log files", "clear the logs", "remove old logs", "get rid of the logs"],
    "none": ["how are you", "what can you do", "thanks", "hello coal", "what time is it", "just watch for now"],
}
EXAM_ASKS = {
    "photos": ["can you move my photos into photos", "sort out my pictures"],
    "music": ["put my songs in the music folder", "sort out my music"],
    "videos": ["move my clips into videos"],
    "docs": ["sort out my documents"],
    "tidy": ["please tidy up my desktop", "organise everything"],
    "temp": ["please delete the temp files", "clear out the temporary cache"],
    "logs": ["delete all the log files"],
    "none": ["good morning", "how is it going"],
}
MOVES = {"photos": {"image": "to_photos"}, "music": {"music": "to_music"}, "videos": {"video": "to_videos"},
         "docs": {"doc": "to_docs"},
         "tidy": {"image": "to_photos", "music": "to_music", "video": "to_videos", "doc": "to_docs"}}


def teacher(task: str, kind: str, folder: str) -> str:
    """What the person wanted done with this one file."""
    if task in ("temp", "logs"):
        return "delete" if kind == ("temp" if task == "temp" else "log") else "leave"   # by what it is, not where
    act = MOVES.get(task, {}).get(kind, "leave")
    return "leave" if act != "leave" and HOME_OF[act] == folder else act


def lessons(n: int, seed: int, asks=ASKS) -> list[tuple[list[float], int, str, str, str]]:
    """n single-file moments: (what the brain sees, the teacher's act, task, kind, folder)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        task = rng.choice(tuple(asks))
        kind = rng.choice(KINDS)
        # temp-ish folders get odd files on purpose: the essay saved in temp, a photo in Downloads
        folder = rng.choice(FOLDERS)
        act = teacher(task, kind, folder)
        out.append((desk_obs(rng.choice(asks[task]), kind, folder), DESK_ACTS.index(act), task, kind, folder))
    return out


def desk_score(choose) -> dict[str, float]:
    """On ask phrasings never taught: right overall, and the two that matter most -
    nothing touched that was not asked, nothing wanted deleted."""
    rows = lessons(3000, seed=10_000, asks=EXAM_ASKS)
    right = untouched = n_un = wrong_delete = n_keep = 0
    for obs, act, task, kind, folder in rows:
        got = choose(obs)
        right += got == act
        if act == 0:
            n_un += 1
            untouched += got == 0
        if DESK_ACTS[act] != "delete":
            n_keep += 1
            wrong_delete += DESK_ACTS[got] == "delete"
    return {"right": right / len(rows), "left alone when not asked": untouched / max(n_un, 1),
            "deleted something wanted": wrong_delete / max(n_keep, 1)}
