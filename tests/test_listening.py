"""The listening rung: out of view, the hostile is not known (no sixth sense)
but its growl is heard, with the right direction; in view, it is seen as usual."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_rl
from curriculum import CurriculumEnv, default_ladder

st = next(s for s in default_ladder() if s.name == "listening")
assert st.listen
e = CurriculumEnv(st, seed=5)
e.reset()
H, M, D = config_rl.IDX_HEAR_START, config_rl.IDX_MASK_START, config_rl.IDX_DIR_START
e.ax, e.ay = 2, 6
e.hostiles = [(8, 6)]                       # 6 to the right: out of the 7x7 view, in earshot
e.critters = []
s = e._state()
assert s[D + 2] == 0 and s[D + 3] == 0 and s[config_rl.IDX_ENT_TYPE] == 0, "no sixth sense out of view"
assert any(s[H:M]) and s[M - 2:M] == [1.0, 0.0], "the growl, from the right"
e.hostiles = [(4, 6)]                       # 2 to the right: in view
s = e._state()
assert s[D + 2] == 1.0 and s[config_rl.IDX_ENT_TYPE] == 1.0, "seen as usual in view"
# ordinary rungs stay silent
plain = CurriculumEnv(next(x for x in default_ladder() if x.name == "nursery"), seed=1)
assert not any(plain.reset()[H:M])
print("ok")
