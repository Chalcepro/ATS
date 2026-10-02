"""Inserting a new sense before the action mask must not change a single answer.

The hearing phase adds features at IDX_MASK_START, not at the end. If the
columns are not shifted, the mask weights land on the hearing inputs and the
policy is quietly destroyed. This checks that a grown brain, fed the old state
with the new sense silent (zeros) at the insertion point, gives identical
logits and value - and that the new columns actually learn.
"""
import sys, torch
sys.path.insert(0, r'C:\Users\Bob\Documents\github\ATS')
import config_rl
from model_rl import RLPolicy

torch.manual_seed(0)
FAIL = []
def check(name, ok, detail=''):
    print('  %-46s %-22s %s' % (name, detail, 'ok' if ok else 'FAILED'))
    if not ok: FAIL.append(name)

N_HEAR = 24
at = config_rl.IDX_MASK_START

p = RLPolicy()
s = torch.rand(4, config_rl.STATE_SIZE)
s[:, at:] = (torch.rand(4, config_rl.ACTION_SIZE) > 0.3).float()   # a mask-like tail
with torch.no_grad():
    l0, v0, _ = p(s)

p.expand_state(config_rl.STATE_SIZE + N_HEAR, insert_at=at)
s2 = torch.cat([s[:, :at], torch.zeros(4, N_HEAR), s[:, at:]], dim=1)
with torch.no_grad():
    l1, v1, _ = p(s2)
check('logits unchanged (sense silent)', torch.allclose(l0, l1, atol=1e-6), 'max d %.2e' % (l0-l1).abs().max())
check('value unchanged (sense silent)', torch.allclose(v0, v1, atol=1e-6), 'max d %.2e' % (v0-v1).abs().max())

# The wrong way, for contrast: append instead of insert. Must NOT match, or
# this test would not catch the bug it exists for.
torch.manual_seed(0); q = RLPolicy()                     # same init as p
q.expand_state(config_rl.STATE_SIZE + N_HEAR)            # append - misaligned
with torch.no_grad():
    lw, _, _ = q(s2)
check('appending instead would have broken it', not torch.allclose(l0, lw, atol=1e-4), 'max d %.2e' % (l0-lw).abs().max())

s3 = s2.clone(); s3[:, at:at + N_HEAR] = torch.rand(4, N_HEAR)
logits, value, _ = p(s3)
(logits.sum() + value.sum()).backward()
g = p.fc1.weight.grad[:, at:at + N_HEAR].abs().max().item()
check('new hearing columns receive gradient', g > 0, 'max %.4f' % g)

try:
    p.expand_state(p.state_size + 1, insert_at=p.state_size + 5); bad = False
except ValueError:
    bad = True
check('insert_at out of range is refused', bad)

print('\n' + ('all checks passed' if not FAIL else 'FAILED: %s' % FAIL))
sys.exit(1 if FAIL else 0)
