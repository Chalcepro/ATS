"""Pack Coal for the phone: its code and its one brain, into one folder with a version stamp.

    py -3.12 tools/pack_coal.py            -> phone_pack/Coal/

coal_sync.ps1 copies that folder to the phone whenever the phone is plugged in and its
stamp differs. The stamp is the brain file's fingerprint, so a copy only happens when
Coal actually changed.
"""
import hashlib
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "phone_pack", "Coal")
BRAIN = ["household_brain_trend.pt", "lexicon.json", "coal_answers.json"]


def stamp(paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def main() -> str:
    code = [f for f in os.listdir(os.path.join(ROOT, "household")) if f.endswith(".py")]
    files = [os.path.join(ROOT, "household", f) for f in code] + [os.path.join(ROOT, "checkpoints", f) for f in BRAIN] + [os.path.join(ROOT, "docs", "PHONE.md")]
    version = stamp(files)
    old = os.path.join(OUT, "coal_version.json")
    if os.path.exists(old) and json.load(open(old))["version"] == version:
        return version
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "household"))
    os.makedirs(os.path.join(OUT, "checkpoints"))
    for f in code:
        shutil.copy(os.path.join(ROOT, "household", f), os.path.join(OUT, "household", f))
    for f in BRAIN:
        shutil.copy(os.path.join(ROOT, "checkpoints", f), os.path.join(OUT, "checkpoints", f))
    shutil.copy(os.path.join(ROOT, "docs", "PHONE.md"), os.path.join(OUT, "README.md"))
    json.dump({"version": version, "packed": time.strftime("%Y-%m-%d %H:%M")}, open(old, "w"))
    return version


if __name__ == "__main__":
    print(main())
