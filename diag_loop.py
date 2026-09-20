"""Is the agent cycling, and does it follow the goal bearing?"""
import copy, torch, brain, curriculum as C
from model_rl import RLPolicy
p = RLPolicy(hidden_size=256); brain.load(p); p.eval()
base = [s for s in C.default_ladder() if s.name == 'corridors7'][0]

def probe(name, episodes=60, **kw):
    st = copy.copy(base)
    for k, v in kw.items(): setattr(st, k, v)
    toward = total = 0; revisit = []; uniq = []
    for k in range(episodes):
        env = C.CurriculumEnv(st, seed=7000 + k)
        s = env._state(); done = False; n = 0; seen = {}
        while not done and n < st.max_steps:
            g, before = env._nearest(env.goals)
            t = torch.tensor(s, dtype=torch.float32)
            m = torch.tensor(env.action_mask(), dtype=torch.float32)
            with torch.no_grad(): lg, _ = p(t, m)
            s, r, done, info = env.step(int(torch.argmax(lg)))
            g2, after = env._nearest(env.goals)
            if g and g2 and g == g2:
                total += 1
                toward += (after < before)
            seen[(env.ax, env.ay)] = seen.get((env.ax, env.ay), 0) + 1
            n += 1
        if n:
            revisit.append(max(seen.values())); uniq.append(len(seen))
    print('  %-32s  toward-goal %4.0f%%   worst cell visited %4.1fx   distinct cells %4.1f'
          % (name, 100*toward/max(total,1), sum(revisit)/len(revisit), sum(uniq)/len(uniq)))

print('\n--- does it walk toward the goal, and does it cycle? ---')
probe('5x5 respawn OFF  (works, 84%)', grid=5, respawn=False, target=1)
probe('7x7 respawn OFF  (52%)',        grid=7, respawn=False, target=1)
probe('7x7 respawn ON   (16%)',        grid=7, respawn=True,  target=3)

print('\n--- how much does the bearing actually drive the action? ---')
import random
def sensitivity(grid, respawn, target, trials=400):
    st = copy.copy(base); st.grid=grid; st.respawn=respawn; st.target=target
    same = 0
    for k in range(trials):
        env = C.CurriculumEnv(st, seed=8000+k)
        s = env._state()
        t = torch.tensor(s, dtype=torch.float32)
        m = torch.tensor(env.action_mask(), dtype=torch.float32)
        with torch.no_grad(): a0 = int(torch.argmax(p(t, m)[0]))
        d = config_rl.IDX_DIR_START if False else __import__('config_rl').IDX_DIR_START
        t2 = t.clone(); t2[d] = -t2[d]; t2[d+1] = -t2[d+1]   # flip the bearing
        with torch.no_grad(): a1 = int(torch.argmax(p(t2, m)[0]))
        same += (a0 == a1)
    return 1 - same/trials
for g, r, tg, label in ((5, False, 1, '5x5 respawn OFF'), (7, False, 1, '7x7 respawn OFF'), (7, True, 3, '7x7 respawn ON')):
    print('  %-32s  flipping the bearing changes the action %3.0f%% of the time'
          % (label, 100*sensitivity(g, r, tg)))
