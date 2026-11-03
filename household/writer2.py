"""The fast writer (Chalce's hybrid, 11-03): trains in parallel like a transformer, writes like an RNN.

minGRU ("Were RNNs All We Needed?", Feng et al. 2024): a GRU whose gates read only the input, never the
last state, so the whole sequence's states come out of one parallel scan in training,
    h_t = (1 - z_t) * h_(t-1) + z_t * h~_t,
and when writing, each new word is one cheap step with a fixed-size state: no slower in a long chat.
It reads the same things as writer.py (question, tags, last lines, recalled memories) and copies the same way;
only the engine differs, so the two can be raced on the same lessons.

    py -3.12 -m household.writer recall fast      (the race: same lessons, same exam, this engine)
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from household.wordlayer import DIM
from household.writer import H, Writer


def _log_g(x: torch.Tensor) -> torch.Tensor:
    """log of a positive squash (x + 0.5 above 0, sigmoid below), so the scan can run in logs."""
    return torch.where(x >= 0, (Fn.relu(x) + 0.5).log(), -Fn.softplus(-x))


class MinGRU(nn.Module):
    def __init__(self, n_in: int, n: int):
        super().__init__()
        self.zh = nn.Linear(n_in, 2 * n)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """All steps at once. x (B, T, n_in), h0 (B, n) positive -> h (B, T, n)."""
        k, hh = self.zh(x).chunk(2, -1)
        log_a, log_b = -Fn.softplus(k), -Fn.softplus(-k) + _log_g(hh)
        a_star = Fn.pad(log_a.cumsum(1), (0, 0, 1, 0))
        log_v = torch.cat([h0.clamp_min(1e-6).log().unsqueeze(1), log_b], 1)
        return (a_star + torch.logcumsumexp(log_v - a_star, 1))[:, 1:].exp()

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One step, for writing: x (B, n_in), h (B, n)."""
        k, hh = self.zh(x).chunk(2, -1)
        z = torch.sigmoid(k)
        return (1 - z) * h + z * _log_g(hh).exp()


class Writer2(Writer):
    def __init__(self, wl, out_words=None):
        super().__init__(wl, out_words)
        del self.enc, self.dec                          # the old engine's parts
        self.fwd, self.bwd = MinGRU(H, H), MinGRU(H, H)  # reading, both ways
        self.h0 = nn.Linear(2 * H, 2 * H)
        self.d1 = MinGRU(DIM, 2 * H)                    # writing: the words so far
        self.d2 = MinGRU(2 * H + 2 * H, 2 * H)          # ... and what it is looking at

    def encode(self, batch):
        B, L = len(batch), max(len(n) for n, _ in batch)
        x = torch.zeros(B, L, DIM + 5)
        mask = torch.zeros(B, L, dtype=torch.bool)
        ext, ext_at = list(self.out), dict(self.out_at)
        src = torch.zeros(B, L, dtype=torch.long)
        lens = torch.tensor([len(r) for _, r in batch])
        for b, (names, rows) in enumerate(batch):
            x[b, :len(rows)] = rows
            mask[b, :len(rows)] = True
            for i, n in enumerate(names):
                if n not in ext_at:
                    ext_at[n] = len(ext)
                    ext.append(n)
                src[b, i] = ext_at[n]
        e = torch.relu(self.enc_in(x))
        one = torch.full((B, H), 0.5)
        f = self.fwd(e, one)
        idx = (lens.unsqueeze(1) - 1 - torch.arange(L)).clamp_min(0)          # each sequence reversed within its own length
        rev = e.gather(1, idx.unsqueeze(2).expand(B, L, H))
        bk = self.bwd(rev, one).gather(1, idx.unsqueeze(2).expand(B, L, H))
        mem = torch.cat([f, bk], 2).masked_fill(~mask.unsqueeze(2), 0)
        h0 = _log_g(self.h0(mem.sum(1) / mask.sum(1, keepdim=True))).exp()
        return mem, mask, src, ext, ext_at, h0

    def _probs(self, h2, ctx, prev, a, src, V, n_ext):
        hc = torch.cat([h2, ctx], -1)
        p_vocab = (self.proj(hc) @ V.T).softmax(-1)
        g = torch.sigmoid(self.gate(torch.cat([hc, prev], -1)))
        p = torch.zeros(*p_vocab.shape[:-1], n_ext)
        p[..., :len(self.out)] = g * p_vocab
        idx = src if a.dim() == 2 else src.unsqueeze(1).expand(-1, a.shape[1], -1)
        return p.scatter_add(-1, idx, (1 - g) * a)

    def _look(self, h1, mem, mask):
        s = (self.att(h1) @ mem.transpose(1, 2)) if h1.dim() == 3 else (mem @ self.att(h1).unsqueeze(2)).squeeze(2)
        m = mask.unsqueeze(1) if h1.dim() == 3 else mask
        a = s.masked_fill(~m, -1e9).softmax(-1)
        ctx = a @ mem if h1.dim() == 3 else (a.unsqueeze(2) * mem).sum(1)
        return a, ctx

    def forward(self, batch, targets=None, greedy: int = 0):
        mem, mask, src, ext, ext_at, h0 = self.encode(batch)
        B = len(batch)
        V = torch.stack([self.vec(w) for w in self.out]) + self.out_fix
        if targets:                                      # every step at once
            T = max(len(t) for t in targets) + 1
            seq = [["<start>"] + t + ["<end>"] * (T - len(t)) for t in targets]
            prev = torch.stack([torch.stack([torch.zeros(DIM) if w == "<start>" else self.vec(w) for w in s[:T]]) for s in seq])
            h1 = self.d1(prev, h0)
            a, ctx = self._look(h1, mem, mask)
            h2 = self.d2(torch.cat([h1, ctx], -1), h0)
            p = self._probs(h2, ctx, prev, a, src, V, len(ext))
            want = torch.tensor([[ext_at.get(w, 0) for w in s[1:T + 1]] for s in seq])
            live = torch.tensor([[1.0 if i <= len(t) else 0.0 for i in range(T)] for t in targets])
            nll = -(p.gather(2, want.unsqueeze(2)).squeeze(2) + 1e-9).log()
            return (nll * live).sum() / live.sum()
        h1, h2, prev = h0, h0, torch.zeros(B, DIM)       # writing: one step at a time, fixed-size state
        written, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool)
        for _ in range(greedy):
            h1 = self.d1.step(prev, h1)
            a, ctx = self._look(h1, mem, mask)
            h2 = self.d2.step(torch.cat([h1, ctx], -1), h2)
            p = self._probs(h2, ctx, prev, a, src, V, len(ext))
            for b in range(B):
                for w in written[b][-1:]:
                    p[b, ext_at[w]] = 0
            nxt = [ext[int(i)] for i in p.argmax(1)]
            for b in range(B):
                if not done[b]:
                    if nxt[b] == "<end>":
                        done[b] = True
                    else:
                        written[b].append(nxt[b])
            if done.all():
                break
            prev = torch.stack([self.vec(w) for w in nxt])
        return [re.sub(r" ([?!.,])", r"\1", " ".join(w)) for w in written]


# ---- inputs built once (11-03): a practice round was spent rebuilding the same rows and word lists in Python ----
class Prepared:
    """Every lesson turned into tensors one time: its reading rows, what each row is (as an id in one shared
    name table, the writer's own words first), and the answer as ids. A batch is then padding and a lookup."""

    def __init__(self, wr: "Writer2", data: list):
        self.names = list(wr.out)
        at = {w: i for i, w in enumerate(self.names)}

        def gid(w: str) -> int:
            if w not in at:
                at[w] = len(self.names)
                self.names.append(w)
            return at[w]
        self.rows = [rows for (names, rows), _ in data]
        self.src = [torch.tensor([gid(n) for n in names]) for (names, _), _ in data]
        self.tgt = [torch.tensor([gid(w) for w in t] + [gid("<end>")]) for _, t in data]
        self.n_out = len(wr.out)
        self.vecs = torch.stack([wr.vec(w) for w in self.names])            # every name's vector, once

    def batch(self, idx: list[int]):
        from torch.nn.utils.rnn import pad_sequence
        x = pad_sequence([self.rows[i] for i in idx], batch_first=True)
        lens = torch.tensor([len(self.rows[i]) for i in idx])
        mask = torch.arange(x.shape[1]).unsqueeze(0) < lens.unsqueeze(1)
        src_g = pad_sequence([self.src[i] for i in idx], batch_first=True)
        tgt_g = pad_sequence([self.tgt[i] for i in idx], batch_first=True, padding_value=-1)
        # this batch's extra names (not among the writer's own words) get the ids after them
        extra = torch.unique(torch.cat([src_g[mask], tgt_g[tgt_g >= 0]]))
        extra = extra[extra >= self.n_out]
        remap = torch.full((len(self.names),), -1, dtype=torch.long)
        remap[:self.n_out] = torch.arange(self.n_out)
        remap[extra] = torch.arange(self.n_out, self.n_out + len(extra))
        src = remap[src_g] * mask
        tgt = torch.where(tgt_g >= 0, remap[tgt_g.clamp_min(0)], torch.zeros_like(tgt_g))
        return x, mask, lens, src, tgt, tgt_g >= 0, self.n_out + len(extra)


def loss_prepared(wr: "Writer2", P: Prepared, idx: list[int]) -> torch.Tensor:
    """The same teacher-forced loss as Writer2.forward, from prepared tensors: no Python per word."""
    x, mask, lens, src, tgt, live, n_ext = P.batch(idx)
    B, L, _ = x.shape
    e = torch.relu(wr.enc_in(x))
    one = torch.full((B, H), 0.5)
    f = wr.fwd(e, one)
    ridx = (lens.unsqueeze(1) - 1 - torch.arange(L)).clamp_min(0).unsqueeze(2).expand(B, L, H)
    bk = wr.bwd(e.gather(1, ridx), one).gather(1, ridx)
    mem = torch.cat([f, bk], 2).masked_fill(~mask.unsqueeze(2), 0)
    h0 = _log_g(wr.h0(mem.sum(1) / mask.sum(1, keepdim=True))).exp()
    T = tgt.shape[1]
    gprev = torch.cat([torch.full((B, 1), -1), torch.where(live, P_ids(P, idx, T), torch.full_like(tgt, -1))[:, :-1]], 1)
    prev = torch.where(gprev.unsqueeze(2) >= 0, P.vecs[gprev.clamp_min(0)], torch.zeros(B, T, DIM))
    V = P.vecs[:P.n_out] + wr.out_fix[:P.n_out]
    h1 = wr.d1(prev, h0)
    a, ctx = wr._look(h1, mem, mask)
    h2 = wr.d2(torch.cat([h1, ctx], -1), h0)
    p = wr._probs(h2, ctx, prev, a, src, V, n_ext)
    nll = -(p.gather(2, tgt.unsqueeze(2)).squeeze(2) + 1e-9).log()
    return (nll * live).sum() / live.sum()


def P_ids(P: Prepared, idx: list[int], T: int) -> torch.Tensor:
    from torch.nn.utils.rnn import pad_sequence
    t = pad_sequence([P.tgt[i] for i in idx], batch_first=True, padding_value=-1)
    return Fn.pad(t, (0, T - t.shape[1]), value=-1)


def train_prepared(wr: "Writer2", data: list, epochs: int = 12, log=print) -> None:
    """writer.train's loop on prepared inputs (Writer2 only)."""
    import random
    import time
    for _, t in data:
        for w in t:
            wr.add_out(w)
    P = Prepared(wr, data)
    params = [p for n, p in wr.named_parameters() if not n.startswith("wl.")]
    opt = torch.optim.Adam(params, lr=2e-3)
    order = list(range(len(data)))
    for e in range(epochs):
        t0, total = time.time(), 0.0
        wr.train()
        random.shuffle(order)
        for i in range(0, len(order), 64):
            loss = loss_prepared(wr, P, order[i:i + 64])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.detach()) * len(order[i:i + 64])
        log(f"  epoch {e + 1}/{epochs}: loss {total / len(data):.3f}  ({time.time() - t0:.0f} s, prepared)")
