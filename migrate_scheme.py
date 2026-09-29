"""Carry a brain across a reward-scheme change, keeping what is still true.

    python migrate_scheme.py --dry-run        what it would do
    python migrate_scheme.py                  do it, with a backup

Why this has to exist
---------------------
`brain.load` refuses a checkpoint whose reward scheme does not match the
build, and it is right to: the critic predicts returns on the old scale, and
a value function that is confidently wrong is worse than one that knows
nothing. But the refusal is all-or-nothing, so bumping the scheme threw away
21 earned rungs and 26,000 episodes of policy along with the critic that
actually needed to go.

The policy and the critic are not equally affected. The policy is a mapping
from what the agent sees to what it does; nothing about "turn left at a wall"
or "a hostile is dangerous" changed because eating now pays and dying now
forfeits. The critic is a prediction of future reward in reward units, and
those units changed. So: keep the trunk, the recurrence and the actor, reset
the value head, drop the optimizer moments - they are scaled to gradients from
the old magnitudes - and stamp the new scheme.

What survives, and what does not
--------------------------------
Survives: every embedding, the trunk (fc1/fc2), the GRU, the actor head, the
`passed` rung list, the episode and tick counts, the earned survival rung.

Does not: `critic.weight` and `critic.bias`, re-initialised exactly as a fresh
model would have them, and the Adam state. Expect the first few hundred
episodes after this to look worse than before while the critic refits - that
is the value function relearning, not the policy regressing, and the
distinction is visible in the logs: actor loss stays small, critic loss starts
high and falls.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

import config_rl

CKPT = config_rl.CHECKPOINT_DIR / "brain.pt"

# The value head, and nothing else. Named explicitly rather than matched by a
# substring, so a future head that happens to contain "critic" in its name
# cannot be silently swept up in this.
VALUE_KEYS = ("critic.weight", "critic.bias")


def fresh_value_head(shape_w, shape_b):
    """A value head initialised the way a new model's would be.

    Zeros would also work and would be simpler, but a zero head predicts zero
    everywhere and gives the first updates a uniform gradient; matching the
    real initialisation means the migrated brain starts exactly where a fresh
    one starts on this one component.
    """
    lin = torch.nn.Linear(int(shape_w[1]), int(shape_w[0]))
    return lin.weight.detach().clone(), lin.bias.detach().clone()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(CKPT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    path = Path(a.path)
    if not path.is_file():
        print("  no checkpoint at %s" % path)
        return 1

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "policy" not in blob:
        print("  %s is not a brain checkpoint this can migrate" % path.name)
        return 1

    was = blob.get("reward_scheme_version")
    now = config_rl.REWARD_SCHEME_VERSION
    prog = blob.get("progress") or {}
    print("\n  %s" % path)
    print("    reward scheme      %r  ->  %r" % (was, now))
    print("    episodes           %s" % prog.get("episodes"))
    print("    total ticks        %s" % blob.get("total_ticks"))
    print("    survival rung      %s" % prog.get("survival_rung"))
    print("    passed rungs       %d" % len(prog.get("passed") or []))

    if was == now:
        print("\n  already on scheme %r - nothing to migrate." % now)
        return 0

    sd = blob["policy"]
    missing = [k for k in VALUE_KEYS if k not in sd]
    if missing:
        print("\n  cannot migrate: value head keys not found: %s" % missing)
        print("  present keys: %s" % ", ".join(sorted(sd)))
        return 1

    keep = [k for k in sd if k not in VALUE_KEYS]
    print("\n    keeping %d tensors:  %s" % (len(keep), ", ".join(keep)))
    _have = int(sd["actor.weight"].shape[0])
    if int(config_rl.ACTION_SIZE) > _have:
        print("    growing actor:       %d actions -> %d (the old %d keep "
              "their weights)" % (_have, config_rl.ACTION_SIZE, _have))
    _hs = int(sd["fc1.weight"].shape[1])
    if int(config_rl.STATE_SIZE) > _hs:
        print("    growing input:       %d state features -> %d (the action "
              "mask is the state's tail)" % (_hs, config_rl.STATE_SIZE))
    print("    resetting:           %s" % ", ".join(VALUE_KEYS))
    print("    dropping:            optimizer state (%d entries, Adam moments"
          " scaled to the old reward magnitudes)"
          % len((blob.get("optimizer") or {}).get("state", {})))

    if a.dry_run:
        print("\n  dry run - nothing written.")
        return 0

    backup = path.with_name("brain_scheme%s_%s.pt"
                            % (was, time.strftime("%Y%m%d_%H%M%S")))
    shutil.copy2(path, backup)
    print("\n    backup ->  %s" % backup.name)

    w, b = fresh_value_head(sd["critic.weight"].shape, sd["critic.bias"].shape)
    sd["critic.weight"], sd["critic.bias"] = w, b

    # Grow the actor if the build has more actions than the brain was trained
    # with. Old action logits keep their weights, so every action it already
    # understands still means the same thing; new ones start at zero, which is
    # a logit of 0 - not suppressed, not preferred, reachable as soon as the
    # mask allows it and the advantage points that way.
    want_actions = int(config_rl.ACTION_SIZE)
    have_actions = int(sd["actor.weight"].shape[0])
    if want_actions > have_actions:
        aw = torch.zeros(want_actions, sd["actor.weight"].shape[1])
        ab = torch.zeros(want_actions)
        aw[:have_actions, :] = sd["actor.weight"]
        ab[:have_actions] = sd["actor.bias"]
        sd["actor.weight"], sd["actor.bias"] = aw, ab
        print("    actions            %d -> %d (old %d keep their weights)"
              % (have_actions, want_actions, have_actions))
        shape = blob.get("shape") or {}
        shape["action_size"] = want_actions
        blob["shape"] = shape
        ps = blob.get("param_shapes")
        if isinstance(ps, dict):
            ps["actor.weight"] = list(aw.shape)
            ps["actor.bias"] = list(ab.shape)

    # Grow the state input if the build sees more than the brain was trained
    # with. STATE_SIZE is IDX_MASK_START + ACTION_SIZE - the action mask is the
    # TAIL of the state vector - so adding 7 slot actions added 7 state inputs
    # as well, and a brain with a correctly grown actor was still refused for
    # having a 495-wide input against a 502-wide one.
    #
    # A straight column copy is right only because the mask sits at the END and
    # the new actions were appended, so every old feature keeps its column. If
    # a future change inserts features in the middle, this must not be used:
    # it would misalign everything after the insertion point silently.
    want_state = int(config_rl.STATE_SIZE)
    have_state = int(sd["fc1.weight"].shape[1])
    if want_state > have_state:
        fw = torch.zeros(sd["fc1.weight"].shape[0], want_state)
        fw[:, :have_state] = sd["fc1.weight"]
        sd["fc1.weight"] = fw
        print("    state inputs       %d -> %d (old %d keep their columns)"
              % (have_state, want_state, have_state))
        shape = blob.get("shape") or {}
        shape["state_size"] = want_state
        blob["shape"] = shape
        ps = blob.get("param_shapes")
        if isinstance(ps, dict):
            ps["fc1.weight"] = list(fw.shape)

    blob["policy"] = sd
    # Dropped, not zeroed: brain.load restores the optimizer only when the key
    # is there, so removing it is how a fresh Adam gets built.
    blob.pop("optimizer", None)
    blob["reward_scheme_version"] = now
    blob["migrated_from_scheme"] = was
    blob["migrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    torch.save(blob, path)
    print("    written -> %s" % path.name)
    print("\n  Done. The 21 rungs, the episode count and the earned cap are")
    print("  intact; the critic starts over. Expect a few hundred episodes of")
    print("  worse-looking reward while it refits - in the PPO lines that is")
    print("  critic loss starting high and falling, with actor loss unchanged.")
    print("\n  If it goes wrong:  copy %s back over brain.pt" % backup.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
