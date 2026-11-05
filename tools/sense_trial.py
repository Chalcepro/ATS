"""Do the new senses help Coal hear? A measured trial on a COPY of Coal's ear (Coal is never saved).

The task: every 40 ms slice, say "is Chalce's voice in this?" and "is music in this?". The sound is made by
mixing a voice-only recording with a music-only stretch at known levels, so the right answer is known for
every slice. Earlier minutes teach, later minutes test (no slice is both).

Three ears are compared, each the same shape as Coal's hearing area:
  bands    what Coal hears today: the 24 CoalOS bands over the last second (starts from Coal's own ear weights)
  senses   only the 20 sense streams over the last second
  both     Coal's bands plus the senses (Coal's ear weights, the senses wired in at zero, as model.load grows inputs)

    py -3.12 tools/sense_trial.py <voice.wav> [<voice2.wav>] --music=<file.wav>:<start s>:<end s>
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from household import ears as E, senses as S  # noqa: E402
from household.listen import read_wav, talking, FRAME  # noqa: E402

SR = 16000
CLIP = 4 * SR
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def voiced_mask(v):
    """Per sample: is the voice talking here (from the clean voice alone)."""
    m = talking(v).repeat(FRAME)
    return np.pad(m, (0, max(0, len(v) - len(m))))[:len(v)]


def make_set(voice, music, n):
    """n clips: a third voice alone, a third music alone, a third both (music 3-15 dB under the voice)."""
    xs, ys = [], []
    vm = voiced_mask(voice)
    for i in range(n):
        kind = i % 3
        a = rng.integers(0, len(voice) - CLIP)
        b = rng.integers(0, len(music) - CLIP)
        v, m = voice[a:a + CLIP], music[b:b + CLIP]
        vr = np.sqrt((v ** 2).mean()) + 1e-6
        mr = np.sqrt((m ** 2).mean()) + 1e-6
        m = m / mr * vr * 10 ** (-rng.uniform(3, 15) / 20)
        x = (v if kind != 1 else 0) + (m if kind != 0 else 0)
        has_v = vm[a:a + CLIP] if kind != 1 else np.zeros(CLIP, bool)
        xs.append(x.astype(np.float32))
        ys.append((has_v, kind != 0))
    return xs, ys


def features(x, has_v, has_m, mu=None, sd=None):
    b = E.slices(E.bands(x, SR))                            # 40 ms slices, Coal's ear
    band = E.ear_all(b)
    s = S.streams(x)                                        # 20 ms rows
    n = len(s) // 2 * 2
    s = s[:n].reshape(-1, 2, s.shape[1]).mean(1)            # to 40 ms
    T = min(len(band), len(s))
    band, s = band[:T], s[:T]
    win = np.stack([np.concatenate([np.zeros((max(0, E.EAR_FRAMES - t - 1), s.shape[1])), s[max(0, t - E.EAR_FRAMES + 1):t + 1]])
                    .reshape(-1) for t in range(T)])
    step = 640
    yv = np.array([has_v[t * step:(t + 1) * step].mean() > 0.5 for t in range(T)], np.float32)
    ym = np.full(T, float(has_m), np.float32)
    return band.astype(np.float32), win.astype(np.float32), np.stack([yv, ym], 1)


def build(xs, ys):
    B, W, Y = [], [], []
    for x, (hv, hm) in zip(xs, ys):
        b, w, y = features(x, hv, hm)
        B.append(b[E.EAR_FRAMES:]); W.append(w[E.EAR_FRAMES:]); Y.append(y[E.EAR_FRAMES:])   # skip the first second (no full window)
    return np.concatenate(B), np.concatenate(W), np.concatenate(Y)


def coal_ear():
    from household import model
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "household_brain_trend.pt")
    brain, _ = model.load(path)
    return copy.deepcopy(brain.ear)                          # a copy: Coal's file is only read


def grow(ear, extra):
    """Coal's ear with `extra` new inputs wired in at zero, the way model.load adds a new sense."""
    lin = ear[0]
    new = nn.Linear(lin.in_features + extra, lin.out_features)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :lin.in_features] = lin.weight; new.bias.copy_(lin.bias)
    ear[0] = new
    return ear


def fresh(n_in, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())


def run(name, ear, Xtr, Ytr, Xte, Yte, epochs=8):
    hidden = ear[2].out_features
    net = nn.Sequential(ear, nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xtr, Ytr, Xte = map(torch.tensor, (Xtr, Ytr, Xte))
    for _ in range(epochs):
        for i in torch.randperm(len(Xtr)).split(256):
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[i]), Ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = (net(Xte) > 0).numpy()
    voice = (p[:, 0] == Yte[:, 0]).mean()
    music = (p[:, 1] == Yte[:, 1]).mean()
    mixed = (Yte[:, 1] == 1)
    voice_in_music = (p[mixed, 0] == Yte[mixed, 0]).mean()
    print(f"  {name:7s} voice right {voice:.1%} | voice right under music {voice_in_music:.1%} | music right {music:.1%}")
    return voice, voice_in_music, music


def main(argv):
    mus = [a for a in argv if a.startswith("--music=")][0][8:].rsplit(":", 2)
    music = read_wav(mus[0])[int(float(mus[1]) * SR):int(float(mus[2]) * SR)]
    voice = np.concatenate([read_wav(f) for f in argv if not a_flag(f)])
    cut_v, cut_m = int(len(voice) * 0.7), int(len(music) * 0.7)
    tr = make_set(voice[:cut_v], music[:cut_m], 120)
    te = make_set(voice[cut_v:], music[cut_m:], 45)
    print(f"voice {len(voice) / SR:.0f}s, music {len(music) / SR:.0f}s; teach 120 clips, test 45 clips (later minutes)")
    Btr, Wtr, Ytr = build(*tr)
    Bte, Wte, Yte = build(*te)
    mu, sd = Wtr.mean(0), Wtr.std(0) + 1e-6
    Wtr, Wte = (Wtr - mu) / sd, (Wte - mu) / sd
    print(f"slices: teach {len(Ytr)}, test {len(Yte)}; test is {Yte[:, 0].mean():.0%} voice, {Yte[:, 1].mean():.0%} music")
    base = coal_ear()
    hidden = base[2].out_features
    res = {}
    res["bands"] = run("bands", copy.deepcopy(base), Btr, Ytr, Bte, Yte)
    res["senses"] = run("senses", fresh(Wtr.shape[1], hidden), Wtr, Ytr, Wte, Yte)
    res["both"] = run("both", grow(copy.deepcopy(base), Wtr.shape[1]), np.hstack([Btr, Wtr]), Ytr, np.hstack([Bte, Wte]), Yte)


def a_flag(a):
    return a.startswith("--")


if __name__ == "__main__":
    main(sys.argv[1:])
