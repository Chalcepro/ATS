"""Does the Mind hint help or hurt, through the path the GUI actually uses?"""
import random, statistics, sys
import numpy as np, torch
import config_rl
from model_rl import RLPolicy
from mind.continual_learner import ContinualLearner
from mind.solution_loop import SolutionLoop
from curriculum import default_ladder, CurriculumEnv
from trace import PathTracer

STAGE = {s.name: s for s in default_ladder()}["nursery"]
DIRS = ((0,-1),(0,1),(-1,0),(1,0))

def probe(pol, seed):
    env = CurriculumEnv(STAGE, seed=seed); env.reset(); env.ax, env.ay = 2, 2
    rows=[]
    for dx,dy in DIRS:
        env.goals=[(2+dx,2+dy)]
        s=torch.tensor(env._state(),dtype=torch.float32); m=torch.tensor(env.action_mask(),dtype=torch.float32)
        with torch.no_grad(): p=torch.softmax(pol(s,action_mask=m)[0],-1)
        rows.append([p[a].item() for a in (0,1,2,3)])
    a=np.array(rows)
    return (a.max(0)-a.min(0)).max(), sum(1 for i in range(4) if int(np.argmax(a[i]))==i)

def run(hint_on, episodes, seed):
    config_rl.MIND_HINT_ENABLED = hint_on
    torch.manual_seed(seed); random.seed(seed)
    pol = RLPolicy(); learner = ContinualLearner(policy=pol)
    sl = SolutionLoop(policy=pol)
    env = CurriculumEnv(STAGE, seed=seed)
    tr = PathTracer(out_dir="logs/hint_ab", per_tick=False)
    det, succ, acts = [], [], {}
    for ep in range(1, episodes+1):
        state = env.reset(); tr.begin(env, ep); done=False
        while not done:
            mask = env.action_mask()
            action = sl.choose_action(state, mask)
            acts[action] = acts.get(action, 0) + 1
            prev = state
            state, reward, done, info = env.step(action)
            tr.record(env, action, reward, info)
            sl.record_outcome(action, reward, state)
            learner.collect(state=prev, action=action, reward=reward,
                            log_prob=sl.last_log_prob, value=sl.last_value,
                            action_mask=mask, done=done)
            learner.maybe_update()
        learner.flush(); sl.on_episode_end()
        row = tr.end(env, info)
        if row["detour_ratio"] not in ("", None): det.append(float(row["detour_ratio"]))
        succ.append(1 if info.get("success") else 0)
    tr.close()
    q = max(1, len(det)//5)
    s_, c_ = probe(pol, seed)
    tot = sum(acts.values()) or 1
    spread = (max(acts.values()) - min(acts.get(a,0) for a in range(4))) / tot
    return dict(det=statistics.mean(det[-q:]) if det else float('nan'),
                succ=100*statistics.mean(succ[-200:]), sens=s_, correct=c_,
                bias=spread)

EP = int(sys.argv[1]) if len(sys.argv)>1 else 700
print("\n%d episodes per arm, through SolutionLoop (the GUI's path)\n" % EP)
print("  seed  hint   detour  success  sensitivity  argmax  action-bias")
agg = {}
for seed in (0,1):
    for on in (True, False):
        r = run(on, EP, seed)
        agg.setdefault(on, []).append(r)
        print("   %d    %-5s  %6.2f   %4.0f%%     %.4f      %d/4      %.2f"
              % (seed, 'ON' if on else 'off', r['det'], r['succ'], r['sens'], r['correct'], r['bias']))
print("\n  MEANS")
for on in (True, False):
    v = agg[on]
    print("   hint %-4s detour %5.2f  success %4.0f%%  sensitivity %.4f  argmax %.1f/4  bias %.2f"
          % ('ON' if on else 'off', statistics.mean(x['det'] for x in v),
             statistics.mean(x['succ'] for x in v), statistics.mean(x['sens'] for x in v),
             statistics.mean(x['correct'] for x in v), statistics.mean(x['bias'] for x in v)))
