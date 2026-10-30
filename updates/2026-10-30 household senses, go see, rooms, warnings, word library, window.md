# ATS Update — 2026-10-30
## Summary
The household brain gets plug-in senses, rooms, a learned word library, a way to heed its person's warnings, and a browser window. Goal reframed by Chalce: a generalist assistant/parent; care is one fundamental, listening to its person is another.

## Changes
### Senses — household/world.py
- `SENSES = eyes, ears, nose, words`: each a switch in `Household.senses`; any can be detached. Words-only (typing) must still be enough.
- `ROOMS`: people and the model are in rooms. Out of the room = out of sight. `see` walks the model to someone (care only works in the same room). The room name is announced in `labels` on entering.
- `danger` (hidden, a room that catches the model for 8 h) and `heard` (its person's words as a meaning vector). Not a scripted trap: the judge simply knows a caught carer helps nobody, and only plans around danger it was warned of.
### Word library — household/lexicon.py
- Permanent word ids; meaning = the moments a word turned up in (feelings + danger); unknown words kept.
- Typos learned from context by default; the spelling fixer is behind `LEXICON_FIX=1`.
### Training / exam — household/train.py, exam.py, memory_test.py
- Condition labels judged by the body alone; rare-label weights behind `HOUSEHOLD_COND_WEIGHTS=1` (hurt the teachings).
- Hazards 7-9 (burn, bee stings, lean season); the hazard-per-message schedule ended at 9.
- New exam lines: words only, out of sight told/not told, listens to a warning (keeps out / still goes).
### Window — household/app.py, household/web/index.html, Household.bat
- Local browser window (127.0.0.1:8765): devices found become senses (camera→eyes, mic→ears+speech via faster-whisper, speaker→Piper voice, typing always). Hidden-danger control for testing warnings.

## Verified
- [x] tests/test_household.py: 8 ok
- [x] judge 100% on hearing, words-only, and heeding exams
- [ ] brain round 9 exam (logs/household_exam_r9.log)
