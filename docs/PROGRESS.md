# Coal: progress, problems, what's next (as of 2026-10-30)

Coal is the household brain (`household/`): one continuous brain, never retrained
from scratch. It is meant to be Chalce's generalist assistant and, one day, a
companion he can reason alongside, not a nurse. One model only.

## Size
- **141,661 parameters, 578 KB** (`checkpoints/household_brain_trend.pt`).
- Gen1 adds ~90 inputs and one head and stays under 150k.
- `checkpoints/brain.pt` (391 MB) is the retired island-world brain, not Coal.

## How it is built (the pipeline)
1. **Teacher, not reward.** `collect` lives 3000 simulated household lives. At every
   moment the judge (`judge.py`) looks ahead and picks the best action, knowing only
   what a carer could know. Each moment becomes a row: named inputs plus the
   judge's answer for every head.
2. **Named inputs.**
   - The body: `world.obs_names()`.
   - What it heard: `heard.*`, the word library's meaning numbers.
   - The desk (gen1): `desk_lessons.desk_names()`.
3. **One brain, sorted heads** (`model.py`):
   - `act`
   - `target`
   - `condition`
   - `foresight`
   - `words`
   - `meaning`
   - `desk`
4. **Continuous.** Weights are saved by name. A new input starts at zero and the old
   skills carry over. Each round continues the saved Coal (30 passes, lr 3e-4).
   `HOUSEHOLD_FRESH=1` deliberately starts a new brain.
5. **Word library** (`lexicon.py`). It learns by counting, not gradients. Each word
   gets a permanent id, and its meaning is the average of the moments it was heard
   in. It reads who/felt/where/doing/talk/about.
6. **Replies.** The library decides what kind of line was said. `app.py answer()`
   fills the reply with only what Coal sees or was told. The brain does not write
   sentences yet.

## Rounds (exam = `python -m household.exam`, logs in `logs/household_*_rN.log`)

| | r15 | r16 |
|---|---|---|
| trained hazards alive | 100% | 100% |
| held-back hazard alive | 100% | 100% |
| words only (no senses) | 25% | 45% |
| still goes when urged | 25% | 2% |
| out of sight, told | 32% | 22% |
| pushes a sober "no" (times a day; judge 1.0) | 1.5 | 2.3 |
| talking, phrasings never taught | — | 91% |
| desktop (gen1), asks never taught | — | 3% (untaught) |

- **r16**: the library learned room/doing/"not" to 100%.
- **Desktop trial** (throwaway copy):
  - right 93%
  - left alone when not asked 100%
  - deleted something wanted 0%
- **r17** (started 2026-10-30): the first gen1 round. Household and desk lessons go
  into the same brain.

## Built this stretch
- **CoalOS:**
  - Every browser alert/prompt is replaced with the OS's own dialogs and file picker.
  - Meters pause when the page is hidden.
- **Talk to Coal window and Coal bar** (top right; Enter sends, Shift+Enter makes a
  new line, hold Space to speak):
  - Maximized windows make room for the bar.
  - Sense switches.
  - Whisper, on this machine, turns speech into words.
  - The raw sound is kept beside Whisper's words in `checkpoints/coal_sounds.jsonl`
    (24 pitch bands, 50/s), as teacher pairs for hearing without Whisper later.
- **Communication class:**
  - greet, bye, thank, questions (how/where/doing/is/name), yes, no
  - learned from use, not keywords
- **Not a nurse:** replies stop reporting on the household unless asked.
- **Gen1 desk lessons** (`desk_lessons.py`):
  - only what was asked is done
  - a file is judged by what it is, not where it sits
  - "tidy" never deletes

## Problems met (and what fixed them)

| Problem | Fix |
|---|---|
| Laptop crash killed r16 mid-run | The saved brain was intact; re-ran the round. |
| Coal acted like a nurse (every reply was about the household) | Replies changed; gen1 makes the desk the main world. |
| Over-caution after r16 (urged 2%, sober "no" pushed 2.3/day) | Expected: r16 continued an older brain that learned to hold back. Still open. |
| Library read "Bayo is thirsty" as a question | Statements added to the class. |
| "not" dropped to 75% | Stop words never start a word pair (back to 100%). |
| Feeling word lost inside questions | Per-word peak gives `about`. |
| Diary line had a broken literal newline | Fixed. |
| CoalOS server shuts down 20 s after the last page ping | `/coal/*` wakes the household app itself. |
| Out of sight + told is weak (22%) | Still open. |

## Next
1. Read r17's exam. Check the household scores held while desk work was learned.
2. **Classroom in CoalOS** (Chalce, 10-30): small lessons alternating English /
   computer / a little biology.
   - A question appears; Coal answers, Chalce answers; the teacher marks each one
     right, close ("good way to put it, but...") or wrong, and gives the answer.
   - Everything is kept raw, exactly as typed or said ("nah", "NAAAAH", "NaH"), so
     Coal learns how people really write, not a cleaned-up version.
3. **The in-OS cursor:** the pointer is part of the screen and its moves and drags
   are logged, so Coal learns that the human is the one who moves things.
4. Desk decisions in chat: preview first, act only on "do it", on a copy.
5. Eyes on the screen. Hearing words straight from the kept sound. Unknown
   greetings ("hiya", "what's up").
