"""Test conversations for Coal's ear (11-03): made with Piper voices, quiet gaps and room hum, so recording,
voices and exact quotes can be checked without real recordings.  py -3.12 tools/make_test_talks.py OUTDIR"""
import os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from household.listen import SR, read_wav, write_wav

VOICES = os.path.join(os.path.expanduser("~"), "Documents", "github", "Cpeech", "piper-voices")
WHO = {"chalce": "en_US-joe-medium", "registrar": "en_US-john-medium", "affairs": "en_US-libritts-high", "passerby": "en_GB-alan-medium"}
rng = np.random.default_rng(3)

def say(who, text):
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(["piper", "--model", os.path.join(VOICES, WHO[who] + ".onnx"), "--output_file", out] + (["--speaker", "14"] if "libritts" in WHO[who] else []),
                   input=text.encode(), capture_output=True, check=True)
    x = read_wav(out); os.remove(out); return x

def talk(parts):
    """parts: (who, text) or seconds of quiet."""
    xs = []
    for p in parts:
        xs.append(np.zeros(int(p * SR), np.float32) if isinstance(p, (int, float)) else say(*p))
        xs.append(np.zeros(int(0.35 * SR), np.float32))
    x = np.concatenate(xs)
    return x + rng.normal(0, 0.002, len(x)).astype(np.float32)

DAYS = {
    "enrol_chalce": [("chalce", "This is my voice. Coal, this is how I sound when I talk to you every day.")],
    "2026-10-28 09:10": [30, ("chalce", "Good morning ma. I'm here about my clearance."),
                         ("affairs", "Where is your student file?"), ("chalce", "I submitted it last week, ma."),
                         ("affairs", "Maybe tomorrow, check with the Registrar."), 8,
                         ("passerby", "Has anyone seen the projector remote?")],
    "2026-10-30 14:05": [25, ("chalce", "Good afternoon sir. I'm here concerning my certificate."),
                         ("registrar", "Your file is with the academic office. You can get the documentation tomorrow."),
                         ("chalce", "Thank you sir."), 6, ("chalce", "Okay, let me go and eat.")],
    "2026-11-03 10:42": [3, ("chalce", "Hmm, it's hot today."), 2,
                         ("chalce", "Coal, start recording."),
                         ("registrar", "The head academic signed it. Come back tomorrow for the documents."),
                         ("chalce", "So sir, what is the process now?"),
                         ("registrar", "Just collect the certificate from my office.")],
}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for name, parts in DAYS.items():
    write_wav(os.path.join(out, name.replace(":", "").replace(" ", "_") + ".wav"), talk(parts))
    print("made", name)
