"""tools/add_hearing.py on a COPY of brain.pt: with the ears silent, the migrated
brain must answer exactly like the original on real states, and --remove must
give back the original exactly."""
import os, sys, shutil, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import brain, config_rl
from model_rl import RLPolicy
import tools.add_hearing as AH

src = brain.DEFAULT_PATH
tmp = Path(tempfile.mkdtemp()) / "brain.pt"
shutil.copy2(src, tmp)
AH.CKPT = tmp

def policy_of(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sh = blob["shape"]
    p = RLPolicy(hidden_size=sh["hidden_size"], state_size=sh["state_size"], action_size=sh["action_size"])
    brain.grow_tables(blob, p); p.load_state_dict(blob["policy"]); p.eval()
    return p, blob

old_p, old_blob = policy_of(tmp)
assert old_p.state_size == config_rl.STATE_SIZE - config_rl.HEAR_DIM, "expects the pre-hearing brain"

# real states from the island, with the hearing slots cut out for the old brain
from ats_env import ATSEnvironment
env = ATSEnvironment(vary_world=False, world_seed=7, use_loop=False)
states = [env.reset()]
for _ in range(30):
    s, _, d, _ = env.step(env.random_action()); states.append(s)
S_new = torch.tensor(states, dtype=torch.float32)
S_new[:, config_rl.IDX_HEAR_START:config_rl.IDX_MASK_START] = 0          # silence
S_old = torch.cat([S_new[:, :config_rl.IDX_HEAR_START], S_new[:, config_rl.IDX_MASK_START:]], 1)

sys.argv = ["add_hearing"]; AH.main()
new_p, new_blob = policy_of(tmp)
with torch.no_grad():
    l0, v0, _ = old_p(S_old); l1, v1, _ = new_p(S_new)
print("max logit diff %.2e, value diff %.2e" % ((l0 - l1).abs().max(), (v0 - v1).abs().max()))
assert torch.allclose(l0, l1, atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)
assert new_blob["progress"] == old_blob["progress"], "rungs and progress carry over"
st = new_blob["optimizer"]["state"].get(AH.fc1_index(new_p))
assert st is None or st["exp_avg"].shape[1] == config_rl.STATE_SIZE, "Adam moments widened"
with torch.no_grad():
    S_loud = S_new.clone(); S_loud[:, config_rl.IDX_HEAR_START] = 1.0
    l2, _, _ = new_p(S_loud)
assert torch.allclose(l1, l2), "untrained ears change nothing yet (zero weights)"

sys.argv = ["add_hearing", "--remove"]; AH.main()
back_p, _ = policy_of(tmp)
with torch.no_grad():
    l3, _, _ = back_p(S_old)
assert torch.allclose(l0, l3, atol=1e-6), "--remove gives the original back"
shutil.rmtree(tmp.parent, ignore_errors=True)
print("ok")
