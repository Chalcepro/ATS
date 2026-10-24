"""Give the trained brain its ears (roadmap H2), keeping everything it knows.

    py -3.12 tools/add_hearing.py --dry-run     what it would do
    py -3.12 tools/add_hearing.py               do it (backs brain.pt up first)
    py -3.12 tools/add_hearing.py --remove      take the ears out again (HEARING = False)

The hearing inputs go in at config_rl.IDX_HEAR_START, before the action mask,
with zero weights (RLPolicy.expand_state insert_at): fed silence, the brain
answers exactly as before. Adam's moments for fc1 are widened the same way, so
training resumes without re-tuning the optimiser. Rungs, ticks lived, the
critic and the reward scheme all carry over untouched."""
from __future__ import annotations

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # the ATS root
import argparse, shutil, time

import torch

import brain, config_rl
from model_rl import RLPolicy

CKPT = brain.DEFAULT_PATH
AT = config_rl.IDX_HEAR_START
N = config_rl.HEAR_DIM


def fc1_index(policy):
    """fc1.weight's position in policy.parameters(): Adam's state is keyed by it."""
    return [n for n, _ in policy.named_parameters()].index("fc1.weight")


def widen(t, add):
    """Insert `add` zero columns at AT (add < 0 removes them)."""
    if add > 0:
        return torch.cat([t[:, :AT], torch.zeros(t.shape[0], add, dtype=t.dtype), t[:, AT:]], 1)
    return torch.cat([t[:, :AT], t[:, AT - add:]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    shape = dict(blob["shape"])
    old = shape["state_size"]
    new = old - N if args.remove else config_rl.STATE_SIZE
    add = new - old
    print(f"brain.pt: state {old} -> {new}  ({'+' if add >= 0 else ''}{add} at column {AT}), "
          f"hidden {shape['hidden_size']}, {len((blob.get('progress') or {}).get('passed') or [])} rung(s)")
    if add == 0:
        print("nothing to do: the brain already matches this build")
        return
    if not args.remove and add != N:
        sys.exit(f"refusing: expected to add exactly {N} hearing inputs, the build wants {add}")
    if args.dry_run:
        return

    policy = RLPolicy(hidden_size=shape["hidden_size"], state_size=old, action_size=shape["action_size"])
    brain.grow_tables(blob, policy)
    policy.load_state_dict(blob["policy"])
    if add > 0:
        policy.expand_state(new, insert_at=AT)
    else:                                   # take the hearing columns back out
        fc1 = torch.nn.Linear(new, policy.hidden_size)
        fc1.weight.data = widen(policy.fc1.weight.data, add)
        fc1.bias.data = policy.fc1.bias.data.clone()
        policy.fc1, policy.state_size = fc1, new

    opt = blob.get("optimizer")
    if opt:
        st = opt["state"].get(fc1_index(policy))
        if st:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].dim() == 2 and st[k].shape[1] == old:
                    st[k] = widen(st[k], add)

    backup = CKPT.with_name("brain_pre_%s_%s.pt" % ("unhearing" if args.remove else "hearing",
                                                    time.strftime("%Y%m%d_%H%M%S")))
    shutil.copy2(CKPT, backup)
    blob["policy"] = policy.state_dict()
    blob["shape"] = brain._shape_of(policy)
    blob["param_shapes"] = brain._params_of(blob["policy"])
    tmp = CKPT.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(CKPT)
    print(f"done. backup: {backup.name}\n  if it goes wrong: copy it back over brain.pt"
          + ("" if args.remove else " and set HEARING = False in config_rl.py"))


if __name__ == "__main__":
    main()
