"""Is the corridors -> corridors7 cliff the bigger room, or the moving goal?"""
import copy, torch, brain, curriculum as C
from model_rl import RLPolicy

p = RLPolicy(hidden_size=256); brain.load(p); p.eval()
base = [s for s in C.default_ladder() if s.name == 'corridors7'][0]

def run(stage, episodes=120, seed0=5000):
    succ = got = steps = 0
    for k in range(episodes):
        env = C.CurriculumEnv(stage, seed=seed0 + k)
        s = env._state(); done = False; n = 0; info = {}
        while not done and n < stage.max_steps:
            st = torch.tensor(s, dtype=torch.float32)
            m = torch.tensor(env.action_mask(), dtype=torch.float32)
            with torch.no_grad():
                lg, _ = p(st, m)
            s, r, done, info = env.step(int(torch.argmax(lg)))
            n += 1
        succ += bool(info.get('success')); got += info.get('collected', 0); steps += n
    return succ / episodes, got / episodes, steps / episodes

def variant(name, **kw):
    st = copy.copy(base)
    for k, v in kw.items():
        setattr(st, k, v)
    a, b, c = run(st)
    print('  %-36s %6.0f%%  %5.2f goals  %5.0f steps' % (name, a * 100, b, c))

print('\n--- grid size vs respawn (greedy play) ---')
variant('5x5  respawn OFF   (= corridors)', grid=5, respawn=False, target=1)
variant('5x5  respawn ON', grid=5, respawn=True, target=3)
variant('7x7  respawn OFF', grid=7, respawn=False, target=1)
variant('7x7  respawn ON    (= corridors7)', grid=7, respawn=True, target=3)
print('\n--- is it just short of time? ---')
variant('7x7  respawn ON   600 steps', grid=7, respawn=True, target=3, max_steps=600)
variant('7x7  respawn OFF  600 steps', grid=7, respawn=False, target=1, max_steps=600)
