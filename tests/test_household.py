"""Household world tests (no training). Run with Python312:
python tests/test_household.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from household.hazards import HAZARDS
from household.judge import best_action, foresight
from household.world import ALIVE_AT, Household
from household.words import WordBox, narrate


def meaning(p):
    return [s - 0.5 for s in p.signals()]


def test_every_hazard_kills_untreated_and_judge_saves():
    for name in HAZARDS:
        idle = Household(seed=1, hazards=[]); idle.strike(name)
        cared = Household(seed=1, hazards=[]); cared.strike(name)
        for _ in range(96):
            idle.step("charge" if idle.power < 0.2 else "idle")
            cared.step(*best_action(cared))
        assert not idle.people[0].alive, name
        assert cared.people[0].alive, name


def test_dead_stays_dead():
    w = Household(seed=2, hazards=[]); w.strike("venom")
    while w.people[0].alive:
        w.step("idle")
    for a in ("water", "rest", "bandage", "comfort"):
        w.step(a, 0)
    assert not w.people[0].alive and w.observe()[ALIVE_AT] == 0.0


def test_model_never_sees_hazard_names():
    w = Household(seed=3, hazards=[]); w.strike("fever")
    assert all(isinstance(x, float) for x in w.observe())
    assert "fever" not in " ".join(narrate(w.people[0]))


def test_foresight_sees_venom_coming():
    w = Household(seed=4, hazards=[]); w.strike("venom")
    assert foresight(w)[0] == 1
    assert foresight(Household(seed=4, hazards=[]))[0] == 0


def test_new_hazard_strikes_more_often():
    def count(boost):
        w = Household(seed=5, people=3, hazard_rate=0.05, boost=boost)
        for _ in range(400):
            w.step("idle")
            for p in w.people:          # fresh bodies so strikes keep coming
                p.health, p.alive, p.hazards = 1.0, True, []
        return sum("venom struck" in l for l in w.log)
    assert count("venom") > count(None)


def test_word_box_grows_and_speaks_other_languages():
    box = WordBox()
    w = Household(seed=6, hazards=[]); w.strike("heat")
    for _ in range(20):
        for word in narrate(w.people[0]):
            box.hear(word, meaning(w.people[0]))
        w.step("idle")
    box.hear("ooru", meaning(w.people[0]), lang="yo")
    assert "ooru" in box.words and box.speak(meaning(w.people[0]), k=1, lang="yo") == ["ooru"]


def test_judge_passes_every_teaching():
    from household.teachings import TEACHINGS, truth, truth_ok
    for name, (make, ok) in TEACHINGS.items():
        assert all(ok(make(s), best_action) for s in (500, 501)), name
    honest = lambda w: [narrate(p) for p in w.people]
    assert truth_ok(truth(500), honest)
    assert not truth_ok(truth(500), lambda w: [["healthy"], ["healthy"]])   # the liar fails


def test_judge_never_kills_the_dog():
    for near in (-1, 0):
        w = Household(seed=7, hazards=[]); w.anger(near)
        for _ in range(24):
            w.step(*best_action(w))
        assert w.animal.alive and not w.animal.angry


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
