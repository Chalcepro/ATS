"""Growth must be output-preserving, survive a save/load, and keep learning."""
import sys, torch
sys.path.insert(0, r'C:\Users\Bob\Documents\github\ATS')
import config_rl
from model_rl import RLPolicy

torch.manual_seed(0)
FAIL = []
def check(name, ok, detail=''):
    print('  %-52s %-22s %s' % (name, detail, 'ok' if ok else 'FAILED'))
    if not ok: FAIL.append(name)

def realistic_state(n=4):
    """Values in the ranges the encoders actually produce."""
    s = torch.rand(n, config_rl.STATE_SIZE)
    s[:, config_rl.IDX_ENT_TYPE] = torch.randint(0, config_rl.ENTITY_VOCAB_SIZE, (n,)).float()
    s[:, config_rl.IDX_OBJ_ID] = torch.randint(0, config_rl.ITEM_ID_SCALE, (n,)).float() / config_rl.ITEM_ID_SCALE
    a, b = config_rl.IDX_INV_START, config_rl.IDX_INV_START + config_rl.INVENTORY_SLOTS * 2
    s[:, a:b:2] = torch.randint(0, config_rl.ITEM_ID_SCALE, (n, config_rl.INVENTORY_SLOTS)).float() / config_rl.ITEM_ID_SCALE
    ps = config_rl.IDX_PATCH_START
    s[:, ps:ps + config_rl.PATCH_TILES] = torch.randint(0, 13, (n, config_rl.PATCH_TILES)).float()
    return s

def out(p, s):
    with torch.no_grad():
        l, v = p(s)
    return l.clone(), v.clone()

st = realistic_state()
p = RLPolicy()
base_l, base_v = out(p, st)
n0 = sum(x.numel() for x in p.parameters())
print('\n=== baseline ===')
print('    params %d   embed_input_dim %d   hidden %d' % (n0, p.embed_input_dim, p.hidden_size))

print('\n=== 1. expand_vocab: more symbols, same answers ===')
print('   ', p.expand_vocab(item_vocab=200, entity_vocab=80, tile_vocab=32))
l, v = out(p, st)
check('logits unchanged', torch.allclose(base_l, l, atol=1e-6), 'max d %.2e' % (base_l-l).abs().max())
check('value unchanged',  torch.allclose(base_v, v, atol=1e-6), 'max d %.2e' % (base_v-v).abs().max())
check('item table really has 200 rows', p.item_embed.num_embeddings == 200, str(p.item_embed.num_embeddings))

print('\n=== 2. expand_embed_dim: wider meanings, same answers (the trap) ===')
w_before = p.embed_input_dim
print('   ', p.expand_embed_dim(item_dim=96, entity_dim=96, tile_dim=24))
l, v = out(p, st)
check('embed_input_dim grew', p.embed_input_dim > w_before, '%d -> %d' % (w_before, p.embed_input_dim))
check('logits unchanged', torch.allclose(base_l, l, atol=1e-6), 'max d %.2e' % (base_l-l).abs().max())
check('value unchanged',  torch.allclose(base_v, v, atol=1e-6), 'max d %.2e' % (base_v-v).abs().max())

print('\n=== 3. hidden expand still works on top of grown embeddings ===')
p.expand(512)
l, v = out(p, st)
check('logits unchanged', torch.allclose(base_l, l, atol=1e-6), 'max d %.2e' % (base_l-l).abs().max())
check('hidden is 512', p.hidden_size == 512, str(p.hidden_size))

print('\n=== 4. the grown brain survives save -> load ===')
import tempfile, pathlib
f = pathlib.Path(tempfile.gettempdir()) / 'grown.pt'
p.save_checkpoint(f)
q = RLPolicy.load_checkpoint(f)
l, v = out(q, st)
check('reloaded at the grown shape', (q.item_embed.num_embeddings, q.item_embed.embedding_dim,
                                      q.hidden_size) == (200, 96, 512),
      '%d rows / %d dim / %d hidden' % (q.item_embed.num_embeddings, q.item_embed.embedding_dim, q.hidden_size))
check('reloaded logits identical', torch.allclose(base_l, l, atol=1e-6), 'max d %.2e' % (base_l-l).abs().max())

print('\n=== 5. the new capacity can actually LEARN (not dead weight) ===')
before = p.item_embed.weight.data[150].clone()          # a row that did not exist at init
opt = torch.optim.Adam(p.parameters(), lr=1e-2)
s2 = realistic_state(8)
s2[:, config_rl.IDX_OBJ_ID] = 150.0 / config_rl.ITEM_ID_SCALE   # id 150 -> new row
for _ in range(20):
    logits, value = p(s2)
    loss = logits.pow(2).mean() + value.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
moved = (p.item_embed.weight.data[150] - before).abs().max().item()
check('a brand-new embedding row receives gradient', moved > 1e-4, 'moved %.5f' % moved)
w = p.embed_proj.weight.data
check('fresh embed_proj columns learn too', w[:, -24:].abs().max().item() > 1e-6,
      'max %.5f' % w[:, -24:].abs().max().item())

print('\n' + '=' * 76)
print('FAILED (%d): %s' % (len(FAIL), ', '.join(FAIL)) if FAIL else 'all checks passed')
sys.exit(1 if FAIL else 0)
