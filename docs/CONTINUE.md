# Keeping Coal going (without Claude)

Written 2026-11-07 for Chalce, so Coal can keep growing even if I'm not around.
Everything here runs from the ATS folder: `C:\Users\Bob\Documents\github\ATS`.
Python is 3.12; `py -3.12` runs it.

## 1. Where Coal lives

| What | Where | Note |
|---|---|---|
| Coal's brain (the real one) | `checkpoints/household_brain_trend.pt` | ~90 MB. **Back it up before any training.** |
| His chooser (picks answers) | `checkpoints/chooser_test.pt` | Sits next to the brain; the chat loads it by itself. |
| Older brains | `checkpoints/household_brain_trend_before_*.pt` | Each one is a backup made before a change. |
| What he read on the web | `data/reading/` | MDN, React, Wikipedia pages he looked up. Not on GitHub (too big); on this laptop. |
| Corrections you gave him | `data/lessons/corrections.jsonl` | Each "no, it's ..." you said. For his next training. |
| What his reader learned live | `data/lessons/reader_live.jsonl` | Each line he misread and then got right. |
| The code | GitHub: `Chalcepro/ATS` | Everything except the brain files. |
| The brain backup | GitHub: `Chalcepro/ATS-brain` | Brain files (Git LFS). |

Your drives are failing, so **GitHub is the copy that survives.** After any training, copy the brain into
`..\ATS-brain\checkpoints\` and push it (section 6).

## 2. Talk to him

    py -3.12 -m household.mind chat

One chat for everything: tell him things ("my keys are in the drawer"), ask him ("where are my keys"),
ask what something is ("what is a closure"), ask about the laptop ("what opens video files").
- When he doesn't know, he looks it up (Simple Wikipedia, then Wikipedia), keeps what he read, and answers.
- When he is wrong, start your next line with **"no"**: `no, it's in the car`. He keeps the fix right away
  and saves it as a lesson. If he misread the kind of line (took a question as a feeling), "no" teaches the reader too.
- Type `quit` to stop. This chat does not save into his brain; the class session below does.

Household.bat (double-click) still has the old menu: window, talk, exam, train.

## 3. A long class with him (saves what he learns)

    py -3.12 tools\class_session.py checkpoints\household_brain_trend.pt

Leave that window open. In another window (Git Bash), talk to him one line at a time:

    sh tools/say.sh where is my phone

The whole talk is in `logs/class/talk.txt`. He saves into the brain every 10 lines and at the end.
To end: `sh tools/say.sh !end`. **Back up the brain first** (section 6).

## 4. Check how he's doing (exams)

| Command | What it checks | Time |
|---|---|---|
| `py -3.12 -m household.exam` | Household: hazards, teachings, warnings, words-only | ~30 min |
| `py -3.12 tests\test_pick.py` | 12 chats he was never tuned on (rules vs chooser) | ~20 min |
| `py -3.12 -m household.device look` | What he sees on this laptop | 1 min |
| `py -3.12 -m household.browse look banana` | Opens one page the way he does | seconds |

To exam a **copy** instead of the real brain, set this first (PowerShell):
`$env:HOUSEHOLD_SAVE = "C:\Users\Bob\Documents\github\ATS\checkpoints\MY_COPY.pt"` (and `$env:COAL_BRAIN` the same, for the chat).

## 5. Train him (always on a copy first)

Long runs go in **their own PowerShell window** with a log, so they survive and you can watch them.
Plug in the charger. The pattern:

    Copy-Item checkpoints\household_brain_trend.pt checkpoints\household_brain_trend_test.pt
    $env:HOUSEHOLD_SAVE = "C:\Users\Bob\Documents\github\ATS\checkpoints\household_brain_trend_test.pt"
    py -3.12 -m household.exam  *> logs\before.log
    py -3.12 -m household.train *> logs\train.log
    py -3.12 -m household.exam  *> logs\after.log

Compare before.log and after.log. Only if "after" is as good or better:
copy the test file over `household_brain_trend.pt` (after backing that up).

The parts, each trained on its own copy (they print a score at the end):

| Part | Command | What it is |
|---|---|---|
| Household + teachings | `py -3.12 -m household.train` | How he acts, the teachings, the desk, page pieces |
| Chooser | `py -3.12 -m household.chooser test` | Picking which fact answers a question (saves `chooser_test.pt`; set `$env:OUT` to save under another name) |
| Reader | `py -3.12 -m household.understand test` | What kind of line you said (question, stop, yes...) |
| Word types | `py -3.12 -m household.grammar test` | noun / verb / ... |
| How words hang together | `py -3.12 -m household.relations test` | who did what to what |
| Reading sites | `py -3.12 -m household.skim fetch mdn-js --pages=50` then `py -3.12 -m household.digest run` | Read more pages into his knowledge |

Logs written by PowerShell's `*>` are UTF-16; open them in Notepad, they read fine.

### The sleep round (after a class)

Corrections are remembered right away (for the exact question), but they only carry over to *new*
questions after a sleep round: the chooser practises your corrections among all its usual lessons.

    $env:FROM = "chooser_test.pt"; $env:OUT = "chooser_sleep.pt"; $env:LR = "3e-4"; $env:STEPS = "1500"
    py -3.12 -m household.chooser test *> logs\sleep.log

If its exam in sleep.log is as good or better than before, back up `chooser_test.pt`, then copy
`chooser_sleep.pt` over it. (First one, 11-07: new stories right first time went 10/20 -> 17/20.)
Never let the chooser change its weights during a chat: tried 11-07, it forgot how to chain facts
(3-step 61% -> 29%). It stays behind `COAL_LIVE_CHOOSER=1`, off.

## 6. Back up (do this after every training)

    Copy-Item checkpoints\household_brain_trend.pt checkpoints\household_brain_trend_before_<what>.pt
    Copy-Item checkpoints\household_brain_trend.pt ..\ATS-brain\checkpoints\
    cd ..\ATS-brain
    git add -A
    git commit -m "Coal after <what>"
    git push

Code changes: in the ATS folder, `git add -A`, `git commit -m "..."`, `git push`.

## 7. Rules we kept (and why)

- **Test on a copy first.** Training can make him worse; a copy costs nothing.
- **Never delete for good.** Recycle Bin only; old brains are kept as `_before_` files.
- **One model.** Coal is one brain; new skills are added into it, not as separate bots.
- **No hand rules for what he should answer.** He learns it. Where rules still help him (reading answers,
  the device), they are behind switches: `COAL_PICK=rules` turns the chooser off.
- **Failing is fine.** What matters is what he does after: look it up, keep the fix, learn from "no".
- **Long jobs in their own window**, never closed by low RAM in the middle.

## 8. What's next, in order

1. **Learn from the corrections file.** `data/lessons/corrections.jsonl` grows every time you say "no".
   Next step: the chooser's training reads it as extra lessons (`household/chooser.py`, `lesson()`).
2. **Reader: new meanings** (thanks, wait, ...). Add your own lines for each to `household/talk_lines.py`
   (`MINE`), then `py -3.12 -m household.understand test` and put the reader into the brain
   (`py -3.12 -m household.mind understand_test` builds a copy with it).
3. **Chooser takes over reading answers too** once it beats the rules on `tests\test_pick.py`
   (then the rules part can go).
4. **Device health:** RAM / CPU / disk in `household/device.py`, so he notices when something (even he)
   is using far more than the job needs.
5. **Guard, not delete:** quarantine by default; your trusted sources and files stay; ask before deleting
   or downloading anything. (Ideas written down 2026-11-07, not built.)
6. **Hands:** scroll, click, type on the screen, only in windows he opened, checking the window in front
   first, and only after you say go. (Not built.)
7. **The writer:** he says answers in his own words instead of lifting a piece of a sentence. (Not built.)

Big picture, from your own words: small and efficient on purpose; knows the basics, researches the rest,
keeps it while it's needed; sees the device he's on; values first (docs/TEACHINGS.md, docs/CHARTER.md).
