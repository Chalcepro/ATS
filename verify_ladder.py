"""Independent check of the brain against every rung, on unseen seeds."""
import torch, brain, curriculum as C
from model_rl import RLPolicy

p = RLPolicy(); brain.load(p); p.eval()

def run(stage, episodes=120, seed0=610000):
    succ = 0; det = []; died = 0; lava = 0
    for k in range(episodes):
        e = C.CurriculumEnv(stage, seed=seed0 + k)
        opt = e._route_len(set())
        s = e._state(); hx = p.initial_hidden(1)
        done = False; n = 0; info = {}; hit = 0
        while not done and n < stage.max_steps:
            t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(e.action_mask(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                lg, _, hx = p(t, m, hx=hx)
            s, r, done, info = e.step(int(torch.argmax(lg)))
            if e._tile(e.ax, e.ay) == C.T_HAZARD: hit += 1
            n += 1
        succ += bool(info.get('success')); died += bool(info.get('dead')); lava += hit
        if info.get('success') and opt: det.append(n / max(opt, 1))
    return (succ / episodes, died / episodes, lava / episodes,
            sum(det) / len(det) if det else 0.0)

print('%-16s %6s %8s %7s %7s %8s  %s' % ('rung','needs','GREEDY','died','lava','detour',''))
print('-' * 72)
bad = []
for st in C.default_ladder():
    g = st
    while g is not None:
        sr, dr, lv, dt = run(g)
        ok = sr >= g.pass_rate
        if not ok: bad.append('%s %dx%d' % (g.name, g.grid, g.grid))
        print('%-16s %5.0f%% %7.0f%% %6.0f%% %7.2f %8.2f  %s'
              % ('%s %dx%d' % (g.name, g.grid, g.grid), g.pass_rate*100,
                 sr*100, dr*100, lv, dt, 'ok' if ok else 'BELOW BAR'))
        g = g.grown()
print()
print('FAILS: %s' % (', '.join(bad) if bad else 'none - the ladder holds up'))
