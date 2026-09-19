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

VERSION = 3
DEFAULT_PATH = config_rl.CHECKPOINT_DIR / "brain.pt"


def saved_shape(path: Path | None = None) -> dict:
    """The shape of the brain on disk, without building a policy first.

    The caller needs this before it can construct the network to load into:
    a brain saved at hidden=256 cannot be poured into a hidden=128 policy, and
    guessing from the config is how a resumed run quietly becomes a fresh one.
    """
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return {}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return dict(blob.get("shape") or {})


def _shape_of(policy):
    return {
        "hidden_size": int(policy.hidden_size),
        "state_size": int(policy.state_size),
        "action_size": int(policy.action_size),
    }


def _params_of(state_dict) -> dict:
    """Every parameter's shape, which is the only honest compatibility test.

    hidden/state/action matched while the embedding front-end was widened
    underneath them (item 32->64, tile 8->16), so the old check waved through
    a brain whose tensors could not actually be poured into the network - and
    then, worse, its progress record was still believed. The menu skipped the
    nursery for a policy that had never seen one.
    """
    return {k: tuple(v.shape) for k, v in state_dict.items()
            if hasattr(v, "shape")}


def incompatible(blob, policy) -> list[str]:
    """The parameters that do not line up, or [] when the brain will load."""
    saved = blob.get("param_shapes")
    if not saved:
        saved = _params_of(blob.get("policy") or {})
    mine = _params_of(policy.state_dict())
    bad = [k for k, shp in mine.items()
           if k in saved and tuple(saved[k]) != tuple(shp)]
    bad += ["%s (absent)" % k for k in mine if k not in saved]
    return bad


def is_loadable(policy, path: Path | None = None) -> bool:
    """Whether the brain on disk belongs to this network at all."""
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return False
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return not incompatible(blob, policy)


def adopt_legacy(policy) -> str | None:
    """Pour the pre-brain checkpoints into `policy`, once.

    model_rl.pt / model_best.pt are what main.py used to load and save. The
    curriculum never wrote to them - it writes here - so the two lineages
    diverged and the GUI ran one set of weights while the menu card reported
    the other one's rungs. These are read only when there is no usable brain,
    so an old run is adopted once and then continues in one place.

    Returns the filename adopted, or None.
    """
    for ckpt in (config_rl.BEST_MODEL_PATH, config_rl.MODEL_PATH):
        if not ckpt.exists():
            continue
        try:
            data = torch.load(ckpt, map_location="cpu")
        except Exception as e:
            print("[brain] %s: %s" % (ckpt.name, e))
            continue
        if isinstance(data, dict):
            version = data.get("reward_scheme_version")
            if version != config_rl.REWARD_SCHEME_VERSION:
                print("[brain] skipping %s: reward scheme %r, this build is %r"
                      % (ckpt.name, version, config_rl.REWARD_SCHEME_VERSION))
                continue
            sd = data.get("model_state_dict", data.get("state_dict", data))
        else:
            sd = data
        try:
            policy.load_state_dict(sd)
        except Exception as e:
            print("[brain] %s did not fit (%s)" % (ckpt.name, e))
            continue
        print("[brain] adopted legacy %s - from here on it lives in %s"
              % (ckpt.name, DEFAULT_PATH.name))
        return ckpt.name
    return None


def save(policy, learner=None, progress=None, path: Path | None = None) -> Path:
    """Write the whole training state. Atomic: a crash mid-write cannot
    corrupt the only copy of a week's training."""
    path = Path(path or DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    blob = {
        "version": VERSION,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "shape": _shape_of(policy),
        "param_shapes": _params_of(policy.state_dict()),
        "policy": policy.state_dict(),
        "progress": dict(progress or {}),
        # Which reward scheme this critic was calibrated against. A critic
        # trained to expect one scale of return is worse than useless under
        # another - it confidently predicts the wrong numbers - so the load
        # refuses rather than resuming into nonsense.
        "reward_scheme_version": config_rl.REWARD_SCHEME_VERSION,
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

    # Only refuse when the file states a scheme and it disagrees. A brain
    # saved before this field existed says nothing, and refusing it would
    # throw away every rung trained up to now over a missing key.
    if "reward_scheme_version" in blob:
        if blob["reward_scheme_version"] != config_rl.REWARD_SCHEME_VERSION:
            print("[brain] refusing to load %s: saved under reward scheme %r, "
                  "this build is %r - its critic would predict the wrong returns"
                  % (path.name, blob["reward_scheme_version"],
                     config_rl.REWARD_SCHEME_VERSION))
            return {}
    else:
        print("[brain] %s predates reward-scheme tagging - loading anyway" % path.name)

    shape = blob.get("shape") or {}
    mine = _shape_of(policy)
    if strict_shape and shape and shape != mine:
        # Loading mismatched weights half-succeeds and then behaves oddly,
        # which is a much worse failure than refusing.
        print("[brain] refusing to load: saved %s, this policy is %s" % (shape, mine))
        return {}

    bad = incompatible(blob, policy)
    if strict_shape and bad:
        # The progress record goes with the weights. Returning {} here is the
        # point: a rung list that outlives the weights that earned it makes
        # the menu resume partway up a ladder the live policy never climbed.
        print("[brain] refusing to load %s: %d parameter(s) do not fit this "
              "network - %s" % (path.name, len(bad), ", ".join(bad[:4])))
        print("[brain]   its %d recorded rung(s) go with it; the ladder "
              "restarts at the nursery."
              % len((blob.get("progress") or {}).get("passed") or []))
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
        # Labelled as the COEFFICIENT, not "entropy".  They are different
        # numbers that move in opposite directions - the controller pushes the
        # coefficient DOWN when measured entropy is healthy - and a bare
        # "entropy 0.0020" reads as total collapse when it actually means the
        # opposite: the coefficient has relaxed to its floor.
        "ent.coeff  %.4f%s" % (
            blob.get("entropy_coeff", float("nan")),
            "  (at floor - exploration was healthy)"
            if abs(blob.get("entropy_coeff", -1) - config_rl.ENTROPY_COEFF) < 1e-9 else ""),
        "episodes   %s" % prog.get("episodes", 0),
        "passed     %s" % ", ".join(prog.get("passed") or ["(none)"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
