"""A buried action can be revived, a new action starts reachable, and the
spread penalty pulls a hidden action back without touching the live ones.

2026-10-17: interact sat 83 logits below the moves (p ~ 1e-38) and attack 23
below, so neither was ever sampled and neither could be learned. See
RLPolicy.revive_actions and config_rl.LOGIT_SPREAD_COEF.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import torch
import config_rl
from model_rl import RLPolicy

torch.manual_seed(0)
FAIL = []
def check(name, ok, detail=''):
    print('  %-50s %-26s %s' % (name, detail, 'ok' if ok else 'FAILED'))
    if not ok: FAIL.append(name)

I = config_rl.ACT_INTERACT
s = torch.rand(8, config_rl.STATE_SIZE)

def p_of(policy, i):
    with torch.no_grad():
        return torch.softmax(policy(s)[0], -1)[:, i].mean().item()

# A trained head the way the real one looked: live actions lifted, interact buried.
p = RLPolicy()
p.actor.bias.data += 90.0
p.actor.bias.data[I] = 5.0
check('buried interact is unreachable', p_of(p, I) < 1e-30, 'p %.1e' % p_of(p, I))
live = [i for i in range(p.action_size) if i != I]
with torch.no_grad():
    before = p(s)[0][:, live]
p.revive_actions([I], live)
check('revived interact gets a fair share', p_of(p, I) > 0.2 / p.action_size, 'p %.3f' % p_of(p, I))
with torch.no_grad():
    after = p(s)[0][:, live]
check('live logits untouched', torch.equal(before, after))
with torch.no_grad():
    lg = p(s)[0]
check('revived logit = mean live logit', torch.allclose(lg[:, I], lg[:, live].mean(-1), atol=1e-4))

# A new action on an inflated head: zero init would be p ~ 1e-39.
q = RLPolicy()
q.actor.bias.data += 90.0
q.expand_actions(q.action_size + 1)
check('new action starts reachable', p_of(q, q.action_size - 1) > 0.2 / q.action_size,
      'p %.3f' % p_of(q, q.action_size - 1))

# The penalty: a hidden action with no PPO gradient climbs back; live ones barely move.
r = RLPolicy()
r.actor.bias.data[I] = -60.0
mask = torch.ones(8, r.action_size); mask[:, I] = 0
opt = torch.optim.Adam(r.parameters(), lr=1e-2)
hx = r.initial_hidden(8)
gap0 = (r.actor.bias.data.mean() - r.actor.bias.data[I]).item()
for _ in range(200):
    r.evaluate(s, torch.zeros(8, dtype=torch.long), action_masks=mask, hxs=hx)
    opt.zero_grad(); (config_rl.LOGIT_SPREAD_COEF * r.last_logit_spread).backward(); opt.step()
with torch.no_grad():
    lg = r(s)[0]
gap1 = (lg[:, [i for i in range(r.action_size) if i != I]].mean(-1) - lg[:, I]).mean().item()
check('spread penalty pulls the hidden action up', gap1 < gap0 / 2, 'gap %.0f -> %.1f' % (gap0, gap1))

print('\nALL PASSED' if not FAIL else '\nFAILED: %s' % FAIL)
sys.exit(1 if FAIL else 0)
