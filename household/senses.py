"""Coal's hearing senses: raw sound turned into separate streams of numbers, 50 times a second.

No sense decides anything ("music", "Chalce", "knock"): each one only measures. Coal's brain learns what the
streams mean, with Whisper and Chalce's labels as teachers at first. One mono mic for now; a second channel
adds the direction sense (left/right arrival time) when there is one.

Streams, one row per 20 ms:
  loud        loudness, dB
  attack      how fast loudness is rising (a knock, a word starting) or falling, dB per frame
  pitch       the main note, as an octave number (log2 Hz); 0 when there is no note
  voicing     how clearly there is a note at all (0 hiss .. 1 pure tone)
  harmonics   strength of overtones 2..6 against the main note, dB - the "wave shape" (sine, square, voice, guitar)
  noisiness   spectral flatness: 0 tonal (vowel, note) .. 1 noise (s, sh, a dragged sack, applause)
  brightness  where the energy sits (spectral centroid), as an octave number
  formants    the first three resonances of the throat and mouth, as octave numbers - who, not how high
  rhythm      how loudness pulses over the last second: strength at 1-3 Hz (beats), 3-8 Hz (syllables), 8-16 Hz
  steady      how stable the pitch is over the last 0.3 s (held sung note ~1, gliding speech lower)
  reverb      how slowly sound dies away after it stops (room echo, distance)
  direction   (stereo only) which side, from arrival time; 0 with one mic
"""
import numpy as np

SR = 16000
HOP = 320                      # 20 ms
WIN = 640                      # 40 ms
NAMES = (["loud", "attack", "pitch", "voicing"] + [f"harm{k}" for k in range(2, 7)] +
         ["noisiness", "brightness", "f1", "f2", "f3", "rhythm_beat", "rhythm_syllable", "rhythm_fast",
          "steady", "reverb", "direction"])
WINDOW = 25                    # the senses area hears the last second: 25 slices of 40 ms
SENSE_SIZE = WINDOW * len(NAMES)
HEARD = ["voice", "music", "chalce"]   # what the senses area says is in a slice, each yes/no ("chalce": his voice)


def _frames(x):
    n = max(0, 1 + (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] if n else np.zeros((0, WIN))


def _pitch(fr):
    """Main note per frame by autocorrelation, 60-500 Hz; voicing = how strong the repeat is."""
    f = fr - fr.mean(1, keepdims=True)
    spec = np.fft.rfft(f, 2 * WIN, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :WIN]
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = SR // 500, SR // 60
    lag = lo + ac[:, lo:hi].argmax(1)
    voicing = np.clip(ac[np.arange(len(ac)), lag], 0, 1)
    hz = SR / lag
    return hz, voicing


def _formants(fr, order=12):
    """First three resonances by linear prediction (the mouth and throat as a filter)."""
    out = np.zeros((len(fr), 3))
    pre = np.append(fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1], axis=1) * np.hamming(WIN)
    for i, f in enumerate(pre):
        r = np.correlate(f, f, "full")[WIN - 1:WIN + order]
        if r[0] <= 1e-9:
            continue
        a, e = np.zeros(order + 1), r[0]           # Levinson-Durbin
        a[0] = 1
        for k in range(1, order + 1):
            acc = r[k] + a[1:k] @ r[k - 1:0:-1]
            ref = -acc / e
            a[1:k] = a[1:k] + ref * a[k - 1:0:-1]
            a[k] = ref
            e *= 1 - ref * ref
            if e <= 0:
                break
        roots = [z for z in np.roots(a) if z.imag > 0 and abs(z) > 0.8]
        hz = sorted(np.angle(z) * SR / (2 * np.pi) for z in roots)
        hz = [h for h in hz if 200 < h < 5000][:3]
        out[i, :len(hz)] = np.log2(hz) if hz else 0
    return out


def _band_power(sig, lo, hi, rate):
    s = np.abs(np.fft.rfft(sig - sig.mean()))
    f = np.fft.rfftfreq(len(sig), 1 / rate)
    tot = s[(f > 0.5)].sum() + 1e-9
    return s[(f >= lo) & (f < hi)].sum() / tot


def streams(x, right=None):
    """All senses for a mono signal (float, 16 kHz). `right`: a second channel, for direction. Returns (T, len(NAMES))."""
    fr = _frames(x)
    T = len(fr)
    if T == 0:
        return np.zeros((0, len(NAMES)))
    win = fr * np.hanning(WIN)
    mag = np.abs(np.fft.rfft(win, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    loud = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    attack = np.diff(loud, prepend=loud[0])
    hz, voicing = _pitch(fr)
    noted = voicing > 0.45
    pitch = np.where(noted, np.log2(hz), 0.0)
    harm = np.zeros((T, 5))
    for k in range(2, 7):                              # overtone k against the fundamental
        b0 = np.clip(np.round(hz / (SR / WIN)).astype(int), 1, len(freqs) - 1)
        bk = np.clip(np.round(k * hz / (SR / WIN)).astype(int), 1, len(freqs) - 1)
        harm[:, k - 2] = np.where(noted, 20 * np.log10(mag[np.arange(T), bk] / mag[np.arange(T), b0]), 0.0)
    noisiness = np.exp(np.log(mag).mean(1)) / mag.mean(1)
    brightness = np.log2((mag * freqs).sum(1) / mag.sum(1) + 1)
    form = _formants(fr)
    rate = SR / HOP                                    # 50 per second
    rhythm = np.zeros((T, 3))
    for t in range(0, T, 5):                           # every 0.1 s, over the last second
        seg = loud[max(0, t - 50):t + 1]
        if len(seg) >= 25:
            rhythm[t:t + 5] = [_band_power(seg, 1, 3, rate), _band_power(seg, 3, 8, rate), _band_power(seg, 8, 16, rate)]
    steady = np.zeros(T)
    for t in range(T):
        seg = pitch[max(0, t - 15):t + 1]
        seg = seg[seg > 0]
        steady[t] = 1 / (1 + 40 * seg.std()) if len(seg) >= 5 else 0.0
    reverb = np.zeros(T)                               # after a drop, how slowly the tail fades (dB per frame, smaller = more echo)
    fall = -np.minimum(attack, 0)
    for t in range(1, T):
        reverb[t] = 0.9 * reverb[t - 1] + 0.1 * fall[t] if fall[t] > 0.5 else reverb[t - 1]
    direction = np.zeros(T)
    if right is not None:
        fr2 = _frames(right)[:T]
        cc = np.fft.irfft(np.fft.rfft(fr, axis=1) * np.conj(np.fft.rfft(fr2, axis=1)), axis=1)
        lag = np.argmax(np.concatenate([cc[:, -16:], cc[:, :17]], 1), 1) - 16
        direction = lag / 16.0
    return np.column_stack([loud, attack, pitch, voicing, harm, noisiness, brightness, form, rhythm, steady, reverb, direction])


# ---- the senses area of Coal's brain: what is in this slice? ---------------------------------------
def windows(x, norm=None):
    """(slices of 40 ms, SENSE_SIZE): each slice with the second before it, every sense taken against its
    usual level (norm = (mean, spread) learned in teach and kept in the brain file)."""
    s = streams(x)
    s = s[:len(s) // 2 * 2].reshape(-1, 2, len(NAMES)).mean(1)
    pad = np.concatenate([np.zeros((WINDOW - 1, len(NAMES))), s])
    w = np.lib.stride_tricks.sliding_window_view(pad, (WINDOW, len(NAMES)))[:, 0].reshape(len(s), -1)
    if norm is not None:
        w = (w - norm[0]) / norm[1]
    return w.astype(np.float32)


def heard(brain, x):
    """Per 40 ms slice, (voice yes/no, music yes/no) from Coal's senses area. None if it was never taught."""
    import torch
    norm = brain.keep.get("sense_norm")
    if norm is None:
        return None
    with torch.no_grad():
        out = brain.sense(torch.tensor(windows(x, (np.array(norm[0]), np.array(norm[1]))))) > 0
    return out.numpy()


def _mixes(voices, musics, n, rng, others=(), room=None):
    """n 4-second clips: voice alone, music alone, both (music 15 dB under the voice .. 3 dB over it), and the truth
    for every slice. Voice slices come from the loudness ear run on the clean voice, where it is reliable."""
    from household.listen import FRAME, loud_talking
    clip, out = 4 * SR, []
    marks = [np.pad(loud_talking(v).repeat(FRAME), (0, FRAME))[:len(v)] for v in voices]
    omarks = [np.pad(loud_talking(v).repeat(FRAME), (0, FRAME))[:len(v)] for v in others]
    for i in range(n):
        kind = i % 3
        his = not others or rng.random() < 0.55               # his voice, or someone else's
        if his:
            k = rng.integers(len(voices))
            v, vm = voices[k], marks[k]
        else:
            k = rng.integers(len(others))
            v, vm = others[k], omarks[k]
            # their voice gets his room's background below, so "clean" is not a giveaway
        m = musics[rng.integers(len(musics))]
        a, b = rng.integers(0, len(v) - clip), rng.integers(0, len(m) - clip)
        vv, mm, has_v = v[a:a + clip], m[b:b + clip], vm[a:a + clip]
        if not his and room is not None and len(room) > clip:
            r = rng.integers(0, len(room) - clip)
            vv = vv + room[r:r + clip] * rng.uniform(0.5, 1.5)
        mm = mm / (np.sqrt((mm ** 2).mean()) + 1e-6) * (np.sqrt((vv ** 2).mean()) + 1e-6) * 10 ** (rng.uniform(-15, 3) / 20)
        x = (vv if kind != 1 else 0) + (mm if kind != 0 else 0)
        x = x * 10 ** (rng.uniform(-12, 6) / 20)                     # near or far from the mic
        T = clip // (2 * HOP) - 1
        yv = np.array([has_v[t * 2 * HOP:(t + 1) * 2 * HOP].mean() > 0.5 for t in range(T)]) & (kind != 1)
        out.append((x.astype(np.float32), np.stack([yv, np.full(T, kind != 0), yv & his], 1).astype(np.float32)))
    return out


def _build(clips, norm=None):
    W, Y = [], []
    for x, y in clips:
        w = windows(x)
        T = min(len(w), len(y))
        W.append(w[WINDOW:T]); Y.append(y[WINDOW:T])                # the first second has no full window
    W, Y = np.concatenate(W), np.concatenate(Y)
    if norm is None:
        norm = (W.mean(0), W.std(0) + 1e-6)
    return (W - norm[0]) / norm[1], Y, norm


def teach(brain, voice_files, music_files, other_files=(), clips=600, epochs=10, seed=0):
    """Teach the senses area from clean voice files and clean music files (or file:start:end). The first 70% of
    each file teaches, the rest is the exam. Only the senses area and its heads move."""
    import torch
    from household.listen import read_wav
    def load(f):
        f, *se = f.rsplit(":", 2) if f.count(":") >= 3 else (f,)
        x = read_wav(f)
        return x[int(float(se[0]) * SR):int(float(se[1]) * SR)] if se else x
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    from household.listen import FRAME, loud_talking
    vs, ms, os_ = [load(f) for f in voice_files], [load(f) for f in music_files], [load(f) for f in other_files]
    room = np.concatenate([v[:len(v) // FRAME * FRAME][~loud_talking(v).repeat(FRAME)[:len(v) // FRAME * FRAME]] for v in vs])
    cut = lambda xs, a, b: [x[int(len(x) * a):int(len(x) * b)] for x in xs]
    W, Y, norm = _build(_mixes(cut(vs, 0, .7), cut(ms, 0, .7), clips, rng, cut(os_, 0, .7), room))
    Wt, Yt, _ = _build(_mixes(cut(vs, .7, 1), cut(ms, .7, 1), clips // 4, rng, cut(os_, .7, 1), room), norm)
    params = [*brain.sense_ear.parameters(), *brain.heard.parameters()]
    opt = torch.optim.Adam(params, 1e-3)
    W, Y = torch.tensor(W), torch.tensor(Y)
    for _ in range(epochs):
        for i in torch.randperm(len(W)).split(256):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(brain.sense(W[i]), Y[i])
            opt.zero_grad(); loss.backward(); opt.step()
    brain.keep["sense_norm"] = [norm[0].tolist(), norm[1].tolist()]
    with torch.no_grad():
        p = (brain.sense(torch.tensor(Wt)) > 0).numpy()
    both = Yt[:, 1] == 1
    res = {"voice": float((p[:, 0] == Yt[:, 0]).mean()), "voice_over_music": float((p[both, 0] == Yt[both, 0]).mean()),
           "music": float((p[:, 1] == Yt[:, 1]).mean())}
    talk = Yt[:, 0] == 1
    res["chalce"] = float((p[talk, 2] == Yt[talk, 2]).mean()) if talk.any() else 0.0
    print(f"exam (later minutes, never taught): voice right {res['voice']:.1%} | voice over music {res['voice_over_music']:.1%}"
          f" | music right {res['music']:.1%} | who (Chalce or not, while talking) {res['chalce']:.1%}  ({len(W)} slices taught, {len(Wt)} tested)")
    return res


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.argv[1] == "teach":       # py -3.12 -m household.senses teach BRAIN.pt --voice a.wav b.wav --music c.wav d.wav:280:500
        from household.model import load, save
        args = sys.argv[3:]
        flags = [k for k, a in enumerate(args) if a.startswith("--")] + [len(args)]
        part = {args[k]: args[k + 1:flags[j + 1]] for j, k in enumerate(flags[:-1])}
        brain, rounds = load(sys.argv[2])
        teach(brain, part["--voice"], part["--music"], part.get("--others", []))
        save(brain, sys.argv[2], rounds)
        print("saved", sys.argv[2])
        sys.exit()
    from household.listen import read_wav
    x = read_wav(sys.argv[1])
    a = int(float(sys.argv[2]) * SR) if len(sys.argv) > 2 else 0
    x = x[a:a + int(float(sys.argv[3]) * SR)] if len(sys.argv) > 3 else x[a:]
    s = streams(x)
    print(f"{len(s)} rows ({len(s) / 50:.1f} s), {s.shape[1]} senses")
    on = s[:, 0] > np.percentile(s[:, 0], 50)
    for i, n in enumerate(NAMES):
        print(f"  {n:16s} median {np.median(s[on, i]):7.2f}   spread {s[on, i].std():6.2f}")
