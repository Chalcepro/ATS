"""The narrator says what its senses carry, and only that."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_rl, narrator
from curriculum import CurriculumEnv, default_ladder

L = default_ladder()
# listening rung: hostile out of view and growling -> DANGER GROWL, its direction, FAR
e = CurriculumEnv(next(s for s in L if s.name == "listening"), seed=5); e.reset()
e.ax, e.ay, e.hostiles, e.critters = 2, 6, [(8, 6)], []
# each growl is one of a few takes and the odd one reads without a pitch, so judge over 20 ticks
heard = 0
for _ in range(20):
    s = e._state()
    w = s[config_rl.IDX_WORD_START:config_rl.IDX_MASK_START]
    if {"DANGER", "GROWL", "RIGHT", "FAR"} <= set(e.last_words):
        heard += 1
        assert w[narrator.W["GROWL"]] == 1.0 and w[narrator.W["SAFE"]] == 0.0
assert heard >= 17, f"growl named {heard}/20"
# in view and close -> DANGER NEAR, no GROWL needed
e.hostiles = [(3, 6)]
e._state()
assert {"DANGER", "RIGHT", "NEAR"} <= set(e.last_words), e.last_words
# nothing around -> SAFE
e.hostiles = []
e._state()
assert "SAFE" in e.last_words and "DANGER" not in e.last_words, e.last_words

# the island: hunger words follow hunger
from ats_env import ATSEnvironment
env = ATSEnvironment(vary_world=False, world_seed=3, use_loop=False); env.reset()
env.agent.hunger = 20
env._state(); assert "HUNGRY" in env.agent.last_words, env.agent.last_words
env.agent.hunger = 90
env._state(); assert "FED" in env.agent.last_words, env.agent.last_words
# words never come from anything but the state before them: same prefix, same words
st = env._state()
assert narrator.words_from_state(st[:config_rl.IDX_WORD_START]) == env.agent.last_words
print("island words:", env.agent.last_words, "| ok")
