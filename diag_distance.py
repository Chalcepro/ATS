"""Does success depend only on how close the goal spawns?"""
import copy, torch, brain, curriculum as C
from collections import defaultdict
from model_rl import RLPolicy
p = RLPolicy(hidden_size=256); brain.load(p); p.eval()
base = [s for s in C.default_ladder() if s.name == 'corridors7'][0]

def by_distance(label, episodes=400, **kw):
    st = copy.copy(base)
    for k, v in kw.items(): setattr(st, k, v)
    buckets = defaultdict(lambda: [0, 0])
    for k in range(episodes):
        env = C.CurriculumEnv(st, seed=3000 + k)
        d0 = env._route_len(set())
        s = env._state(); done = False; n = 0; info = {}
        while not done and n < st.max_steps:
            t = torch.tensor(s, dtype=torch.float32)
            m = torch.tensor(env.action_mask(), dtype=torch.float32)
            with torch.no_grad(): lg, _ = p(t, m)
            s, r, done, info = env.step(int(torch.argmax(lg))); n += 1
        if d0 is None: continue
        b = min(d0, 8)
        buckets[b][0] += bool(info.get('success')); buckets[b][1] += 1
    print('\n  %s' % label)
    print('   %-22s %8s %8s' % ('start distance to goal', 'n', 'success'))
    for d in sorted(buckets):
        w, t = buckets[d]
        print('   %-22s %8d %7.0f%%' % ('%d steps away' % d if d < 8 else '8+ steps away', t, 100*w/t))

by_distance('corridors  5x5 respawn OFF  (scores 84%)', grid=5, respawn=False, target=1)
by_distance('corridors7 7x7 respawn OFF  (scores 52%)', grid=7, respawn=False, target=1)
