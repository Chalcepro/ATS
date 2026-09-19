"""The trigger must grow for ONE kind of plateau and refuse the other three."""
import sys
sys.path.insert(0, r'C:\Users\Bob\Documents\github\ATS')
import config_rl
config_rl.MIND_GROWTH_ENABLED = True
from model_rl import RLPolicy
from mind.continual_learner import ContinualLearner

FAIL = []
def scenario(name, *, entropy, success, delta, episodes=40, cooldown_ok=True, expect_grow):
    p = RLPolicy()
    L = ContinualLearner(p)
    L.last_losses = (0.0, 0.0, 0.0, entropy)
    for _ in range(episodes):
        L.note_outcome(success)
    L._total_ticks = config_rl.MIND_GROWTH_CHECK_EVERY
    L._last_growth_tick = None if cooldown_ok else L._total_ticks
    L._last_growth_reward = 0.0
    L._reward_accumulator = delta
    L._reward_count = 1
    before = p.hidden_size
    L._maybe_grow()
    grew = p.hidden_size > before
    ok = grew == expect_grow
    print('  %-46s %-9s %s' % (name, 'GREW' if grew else 'held', 'ok' if ok else 'FAILED'))
    if not ok: FAIL.append(name)

print('\n=== the one case capacity can fix ===')
scenario('exploring, failing, flat -> grow',
         entropy=1.2, success=False, delta=0.001, expect_grow=True)

print('\n=== the three it must refuse ===')
scenario('converged and SUCCEEDING -> hold',
         entropy=1.2, success=True, delta=0.001, expect_grow=False)
scenario('entropy COLLAPSED (your brain: 0.002) -> hold',
         entropy=0.002, success=False, delta=0.001, expect_grow=False)
scenario('still IMPROVING -> hold',
         entropy=1.2, success=False, delta=0.5, expect_grow=False)

print('\n=== other guards ===')
scenario('too few episodes to judge -> hold',
         entropy=1.2, success=False, delta=0.001, episodes=5, expect_grow=False)
scenario('just grew, cooling down -> hold',
         entropy=1.2, success=False, delta=0.001, cooldown_ok=False, expect_grow=False)

print('\n=== a caller that never reports outcomes gets no blind growth ===')
p = RLPolicy(); L = ContinualLearner(p)
L.last_losses = (0.0, 0.0, 0.0, 1.2)
L._total_ticks = config_rl.MIND_GROWTH_CHECK_EVERY
L._reward_accumulator, L._reward_count = 0.001, 1
before = p.hidden_size
L._maybe_grow()
ok = p.hidden_size == before
print('  %-46s %-9s %s' % ('no note_outcome() -> hold', 'held' if ok else 'GREW', 'ok' if ok else 'FAILED'))
if not ok: FAIL.append('blind')

print('\n=== growth is capped and preserves the answer ===')
p = RLPolicy(hidden_size=config_rl.MIND_MAX_HIDDEN_SIZE)
L = ContinualLearner(p); L.last_losses = (0,0,0,1.2)
for _ in range(40): L.note_outcome(False)
L._total_ticks = config_rl.MIND_GROWTH_CHECK_EVERY
L._reward_accumulator, L._reward_count = 0.001, 1
L._maybe_grow()
ok = p.hidden_size == config_rl.MIND_MAX_HIDDEN_SIZE
print('  %-46s %-9s %s' % ('at MAX_HIDDEN_SIZE -> hold', 'held' if ok else 'GREW', 'ok' if ok else 'FAILED'))
if not ok: FAIL.append('cap')

print('\n' + '='*72)
print('FAILED (%d): %s' % (len(FAIL), ', '.join(FAIL)) if FAIL else 'all checks passed')
sys.exit(1 if FAIL else 0)
