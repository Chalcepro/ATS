"""Everything the agent has learned, in one file that survives the terminal.

The problem this fixes
----------------------
Saving ``policy.state_dict()`` saves the weights and the embedding tables, and
that sounds like it saves the learning. It does not. Four things were being
thrown away every time the process ended:

* **Adam's moments.** The optimiser carries a running estimate of the gradient
  and its variance for every parameter. Restart without them and the first few
  hundred updates are effectively re-tuning the optimiser rather than the
  policy, which looks exactly like the model having forgotten something.

* **The entropy coefficient.** It *adapts* during training
  (``ENTROPY_ADAPT_UP`` / ``DOWN``). A run that had carefully settled at some
  value restarts at the config default.

* **Where it was in the curriculum.** A policy that had passed four rungs
  started again at the nursery. It had not forgotten how to reach a goal - but
  it re-learned every rung to find that out, which is indistinguishable from
  forgetting if you are watching the terminal.

* **How much experience it has.** Tick counts, what it had already passed, when.

So: one file, holding all of it. Load it a week later and training continues
from the rung it was on, with the optimiser in the state it had reached.

The weights are the memory
--------------------------
There is no separate memory store here, deliberately. The embedding tables
(``item_embed``, ``entity_embed``, ``tile_embed``) *are* what the agent knows
about what things are, and they live in the same state dict as everything
else. When the nursery teaches it that one tile id is worth walking towards,
that fact is a row in an embedding table. Primary school adds more rows'
worth of meaning to the same tables rather than starting a second book.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import torch

import config_rl

VERSION = 2
DEFAULT_PATH = config_rl.CHECKPOINT_DIR / "brain.pt"


def _shape_of(policy):
    return {
        "hidden_size": int(policy.hidden_size),
        "state_size": int(policy.state_size),
        "action_size": int(policy.action_size),
    }


def save(policy, learner=None, progress=None, path: Path | None = None) -> Path:
    """Write the whole training state. Atomic: a crash mid-write cannot
    corrupt the only copy of a week's training."""
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    blob = {
        "version": VERSION,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "shape": _shape_of(policy),
        "policy": policy.state_dict(),
        "progress": dict(progress or {}),
    }
    if learner is not None:
        blob["optimizer"] = learner.optimizer.state_dict()
        blob["entropy_coeff"] = float(getattr(learner, "entropy_coeff",
                                              config_rl.ENTROPY_COEFF))
        blob["total_ticks"] = int(getattr(learner, "_total_ticks", 0))

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return path


def load(policy, learner=None, path: Path | None = None, strict_shape: bool = True):
    """Restore into `policy` (and `learner`). Returns the progress dict.

    Returns an empty dict when there is nothing to load, so a first run and a
    resumed run take the same code path.
    """
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return {}

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("version", 0) < 2:
        print("[brain] %s is an older format - weights only" % path.name)

    shape = blob.get("shape") or {}
    mine = _shape_of(policy)
    if strict_shape and shape and shape != mine:
        # Loading mismatched weights half-succeeds and then behaves oddly,
        # which is a much worse failure than refusing.
        print("[brain] refusing to load: saved %s, this policy is %s" % (shape, mine))
        return {}

    policy.load_state_dict(blob["policy"])

    if learner is not None:
        if "optimizer" in blob:
            try:
                learner.optimizer.load_state_dict(blob["optimizer"])
            except Exception as ex:
                print("[brain] optimizer state did not fit (%s) - starting it fresh" % ex)
        if "entropy_coeff" in blob:
            learner.entropy_coeff = float(blob["entropy_coeff"])
        if "total_ticks" in blob:
            learner._total_ticks = int(blob["total_ticks"])

    prog = dict(blob.get("progress") or {})
    passed = prog.get("passed") or []
    print("[brain] resumed %s - saved %s, %d rung(s) passed, %s ticks lived"
          % (path.name, blob.get("saved_at", "?"), len(passed),
             format(blob.get("total_ticks", 0), ",")))
    if passed:
        print("[brain]   already passed: %s" % ", ".join(passed))
    return prog


def describe(path: Path | None = None) -> str:
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return "no brain at %s" % path
    blob = torch.load(path, map_location="cpu", weights_only=False)
    prog = blob.get("progress") or {}
    lines = [
        "brain      %s" % path,
        "saved      %s" % blob.get("saved_at", "?"),
        "shape      %s" % blob.get("shape"),
        "ticks      %s" % format(blob.get("total_ticks", 0), ","),
        "entropy    %.4f" % blob.get("entropy_coeff", float("nan")),
        "episodes   %s" % prog.get("episodes", 0),
        "passed     %s" % ", ".join(prog.get("passed") or ["(none)"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
