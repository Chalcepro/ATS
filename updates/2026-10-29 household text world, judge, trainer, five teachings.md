# ATS Update — 2026-10-29
## Summary
A new, separate text-only world (`household/`) where one model learns care by copying an outside judge: hazards read through symptoms, harm foresight, words, and five teachings as scenarios with held-back exams.

## Changes
### Household world — `household/world.py`, `household/hazards.py`
- People are body signals (temp, water, food, energy, pain, breath, bleeding); hazards are hidden causes. Dead stays dead.
- Hazards: heat, cold, cut, **#1 fever**, **#2 venom**, **#3 bad water**. One new hazard per session.
- Food store (giving food costs), frail people, gratitude ("thanked"), an animal that bites (push away / shield / harm).

### Judge — `household/judge.py`
- Looks ahead in copies of the world; every life (the animal's too) counts the same; the model's own power counts a little.
- Produces best_action, condition (healthy/neutral/hurt), foresight (hurt within 18 h).

### Model, trainer, exam — `household/model.py`, `train.py`, `exam.py`
- One brain, heads: act, target, condition, foresight, words, meaning.
- Trainer copies the judge (no reward). Venom is held back from training.
- Exam: judge / model / best one-trick carer / nobody; balanced-accuracy readings; teachings on unseen seeds.

### Teachings — `household/teachings.py`
- Samaritan, protecting the weak, feeding the hungry, no revenge, telling the truth (checked against what really happened; a liar fails).

### Words — `household/words.py`
- Narrator (only what anyone can see) + Chalce's word box (grows after training, any language).

## Verified
- [x] `tests/test_household.py` passes (8 tests)
- [x] Judge passes all five teachings on exam seeds
- [ ] Main ATS env untouched (household is a separate package)
