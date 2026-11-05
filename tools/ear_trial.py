"""Who-said-it trial for Coal's ear, with no words written down (fast).

Enrols Chalce from a clean recording into a throwaway ears folder, then labels every piece of talk in the
test files and prints how many seconds went to each voice. Nothing touches checkpoints/ears.

    py -3.12 tools/ear_trial.py <clean enrol file> <test file> [<test file> ...] [--quiet-sub]
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["COAL_EARS"] = tempfile.mkdtemp(prefix="ear_trial_")
from household import listen as L  # noqa: E402


def enrol(x):
    m = L.talking(x).repeat(L.FRAME)
    talk = x[:len(m)][m]
    p = [list(L.voiceprint(talk[i:i + 24000])) for i in range(0, max(1, len(talk) - 12000), 12000)][:L.KEEP_PRINTS]
    k = int(os.environ.get("ANCH", str(L.ENROL_KEEP)))
    return {"Chalce": p[:max(k, 1)], L.ANCHORS: {"Chalce": k}}


def label(x, known, **kw):
    speech = L.talking(x)
    secs = {}
    for a, b in L.utterances(speech):
        t = (b - a) * L.FRAME / L.SR
        if t < 0.6:
            continue
        v = L.who(L.voiceprint(x[a * L.FRAME:b * L.FRAME], L.background(x, speech, a)), known)
        secs[v] = secs.get(v, 0) + t
    return secs


def main(argv):
    for a in argv:
        if a.startswith("--sub="):
            L.NOISE_SUB = float(a[6:])
    files = [a for a in argv if not a.startswith("--")]
    known = enrol(L.read_wav(files[0]))
    for f in files[1:]:
        secs = label(L.read_wav(f), known)
        total = sum(secs.values()) or 1
        top = ", ".join(f"{k} {v:.0f}s" for k, v in sorted(secs.items(), key=lambda kv: -kv[1])[:6])
        print(f"{os.path.basename(f)}: Chalce {secs.get('Chalce', 0) / total:.0%} of {total:.0f}s talk | {top}")


if __name__ == "__main__":
    main(sys.argv[1:])
