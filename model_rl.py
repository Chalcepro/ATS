"""Growing PPO actor-critic policy for ATS.

The policy network can expand its hidden width at runtime to grow capacity
without losing previously learned weights.  Checkpoint metadata stores the
current hidden size so reloads are transparent.
"""

import torch
import torch.nn as nn

import config_rl


class RLPolicy(nn.Module):
    """Actor-critic with learned entity & item embeddings and expandable hidden layers."""

    def __init__(self, state_size=None, action_size=None, hidden_size=None,
                 item_vocab=None, entity_vocab=None, tile_vocab=None,
                 item_dim=None, entity_dim=None, tile_dim=None):
        super().__init__()
        self.state_size = state_size or config_rl.STATE_SIZE
        self.action_size = action_size or config_rl.ACTION_SIZE
        self.hidden_size = hidden_size or config_rl.MIND_HIDDEN_SIZE

        # Learned embedding tables.  Sizes are arguments rather than direct
        # config reads so a checkpoint whose tables have already grown past
        # the config defaults can be rebuilt at *its* size, not the config's.
        self.item_embed = nn.Embedding(item_vocab or config_rl.ITEM_VOCAB_SIZE,
                                       item_dim or config_rl.ITEM_EMBED_DIM)
        self.entity_embed = nn.Embedding(entity_vocab or config_rl.ENTITY_VOCAB_SIZE,
                                         entity_dim or config_rl.ENTITY_EMBED_DIM)
        self.tile_embed = nn.Embedding(tile_vocab or config_rl.TILE_VOCAB_SIZE,
                                       tile_dim or config_rl.TILE_EMBED_DIM)
        # nn.Embedding defaults to N(0, 1), which is large next to typical
        # activations.  With the patch contributing 25 rows, the embedding
        # branch otherwise swamps fc1(state) at init and drowns the rest of
        # the observation (caught by sanity_train.py).  Start small; the
        # tables grow their own scale if the information is worth it.
        for table in (self.item_embed, self.entity_embed, self.tile_embed):
            nn.init.normal_(table.weight, mean=0.0, std=0.1)

        self.embed_proj = nn.Linear(self.embed_input_dim, self.hidden_size)

        self.fc1 = nn.Linear(self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # Recurrence.  The remembered map already carries most of what the
        # agent needs to know about the room; this carries what it cannot
        # draw on a map - what it has already tried, and how long it has
        # been getting nowhere.  Initialised so it starts near pass-through
        # rather than scrambling a freshly-learned trunk.
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.actor = nn.Linear(self.hidden_size, self.action_size)
        self.critic = nn.Linear(self.hidden_size, 1)

    @property
    def embed_input_dim(self) -> int:
        """Width of the concatenated embedding features fed to ``embed_proj``.

        nearest-entity + nearest-object + pooled inventory + flattened
        egocentric tile patch.  The patch is flattened rather than pooled on
        purpose — pooling would discard *where* each tile is, which is the
        entire reason the patch was added.

        Read off the *tables themselves*, not config.  Once a table can grow
        at runtime the config value is only its starting size, and a property
        that keeps quoting config would report a width the tensors no longer
        have - which is a silent shape bug rather than a loud one.
        """
        return self._embed_width(self.entity_embed.embedding_dim,
                                 self.item_embed.embedding_dim,
                                 self.tile_embed.embedding_dim)

    # -- embedding layout -------------------------------------------------
    #
    # embed_proj's input is a concatenation, so every column has a meaning
    # fixed by its position:
    #
    #     [ entity | object | inventory | patch tile 0 | ... | patch tile 24 ]
    #
    # Widening any one embedding therefore *moves* every block after it.  The
    # old weights have to be re-laid-out into their new offsets, not appended
    # - copying them straight across would leave the network reading
    # inventory weights off the object slot.

    @staticmethod
    def _embed_width(entity_dim, item_dim, tile_dim) -> int:
        return (entity_dim + item_dim + item_dim
                + config_rl.PATCH_TILES * tile_dim)

    @staticmethod
    def _embed_blocks(entity_dim, item_dim, tile_dim):
        """[(start_column, width), ...] for one set of embedding dims."""
        blocks = [(0, entity_dim),
                  (entity_dim, item_dim),
                  (entity_dim + item_dim, item_dim)]
        base = entity_dim + 2 * item_dim
        blocks += [(base + i * tile_dim, tile_dim)
                   for i in range(config_rl.PATCH_TILES)]
        return blocks

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def initial_hidden(self, batch=None):
        """Zero recurrent state.  Call at every episode boundary - carrying
        one episode's memory into the next is how an agent learns to be
        confused."""
        w = next(self.parameters())
        return (w.new_zeros(self.hidden_size) if batch is None
                else w.new_zeros(batch, self.hidden_size))

    def forward(self, state, action_mask=None, hx=None):
        """Return (logits, value).

        If *action_mask* is provided (0/1 tensor same shape as logits),
        masked-out actions are set to -1e8 before sampling so the model
        never selects invalid actions.
        """
        # Extract entity and object indices from the state vector for embedding lookup.
        #   index 15 = nearest entity type — stored as a RAW int  (E001 -> 1.0)
        #   index 19 = nearest object id  — stored NORMALISED     (id / ITEM_VOCAB_SIZE)
        # The object id must be de-normalised before the lookup: `.long()` on the
        # 0..1 fraction previously collapsed every object to embedding row 0, so
        # the item-embedding table (100x32) was dead weight.
        if state.dim() == 1:
            state_in = state.unsqueeze(0)
        else:
            state_in = state

        ent_idx = state_in[..., config_rl.IDX_ENT_TYPE].round().long().clamp(0, self.entity_embed.num_embeddings - 1)
        obj_idx = (state_in[..., config_rl.IDX_OBJ_ID] * config_rl.ITEM_ID_SCALE).round().long().clamp(0, self.item_embed.num_embeddings - 1)

        ent_vec = self.entity_embed(ent_idx)
        obj_vec = self.item_embed(obj_idx)

        # Inventory: 7 slots of (norm_item_id, norm_count) -> item ids live on
        # the even offsets.  These used to bypass the embedding table entirely
        # and enter the net as raw normalised floats, so a 100x32 table was
        # trained on exactly one input (the nearest ground object).  Empty
        # slots are masked out of the mean so padding doesn't wash it out.
        inv_start = config_rl.IDX_INV_START
        inv_end = inv_start + config_rl.INVENTORY_SLOTS * 2
        inv_idx = (state_in[..., inv_start:inv_end:2] * config_rl.ITEM_ID_SCALE)
        inv_idx = inv_idx.round().long().clamp(0, self.item_embed.num_embeddings - 1)
        inv_emb = self.item_embed(inv_idx)                              # (..., slots, dim)
        inv_valid = (inv_idx > 0).to(inv_emb.dtype).unsqueeze(-1)
        inv_vec = (inv_emb * inv_valid).sum(-2) / inv_valid.sum(-2).clamp(min=1.0)

        # Egocentric tile patch -> embedded and flattened (position preserved).
        patch_start = config_rl.IDX_PATCH_START
        patch_end = patch_start + config_rl.PATCH_TILES
        patch_idx = state_in[..., patch_start:patch_end].round().long().clamp(0, self.tile_embed.num_embeddings - 1)
        patch_vec = self.tile_embed(patch_idx).flatten(start_dim=-2)

        emb_feat = self.embed_proj(torch.cat([ent_vec, obj_vec, inv_vec, patch_vec], dim=-1))

        x = torch.relu(self.fc1(state_in) + emb_feat)
        x = torch.relu(self.fc2(x))

        h_in = self.initial_hidden(x.shape[0]) if hx is None else hx
        if h_in.dim() == 1:
            h_in = h_in.unsqueeze(0)
        h_out = self.gru(x, h_in)
        x = h_out

        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        if state.dim() == 1:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            h_out = h_out.squeeze(0)

        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)
        return logits, value, h_out

    def act(self, state, action_mask=None, hx=None):
        """Sample an action from the policy (used during rollouts)."""
        logits, value, h_out = self.forward(state, action_mask=action_mask, hx=hx)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, h_out

    def evaluate(self, states, actions, action_masks=None, hxs=None):
        """Evaluate log-probs, values, entropy for a batch (PPO update)."""
        logits, values, _ = self.forward(states, action_mask=action_masks, hx=hxs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Growth — expand hidden width while preserving learned weights
    # ------------------------------------------------------------------
    def expand(self, new_hidden_size):
        """Widen hidden layers to *new_hidden_size*.

        Existing weights are copied into the top-left block of the new
        weight matrices.  New neurons are initialised near zero so the
        network output is unchanged immediately after expansion.
        """
        if new_hidden_size <= self.hidden_size:
            return  # nothing to do

        old = self.hidden_size

        # --- embed_proj: (embed_input_dim → old) → (embed_input_dim → new) ---
        new_embed_proj = nn.Linear(self.embed_input_dim, new_hidden_size)
        nn.init.zeros_(new_embed_proj.weight)
        nn.init.zeros_(new_embed_proj.bias)
        new_embed_proj.weight.data[:old, :] = self.embed_proj.weight.data
        new_embed_proj.bias.data[:old] = self.embed_proj.bias.data

        # --- fc1: (state_size → old) → (state_size → new) ----
        new_fc1 = nn.Linear(self.state_size, new_hidden_size)
        nn.init.zeros_(new_fc1.weight)
        nn.init.zeros_(new_fc1.bias)
        new_fc1.weight.data[:old, :] = self.fc1.weight.data
        new_fc1.bias.data[:old] = self.fc1.bias.data

        # --- fc2: (old → old) → (new → new) ---
        new_fc2 = nn.Linear(new_hidden_size, new_hidden_size)
        nn.init.zeros_(new_fc2.weight)
        nn.init.zeros_(new_fc2.bias)
        new_fc2.weight.data[:old, :old] = self.fc2.weight.data
        new_fc2.bias.data[:old] = self.fc2.bias.data

        # --- actor: (old → action_size) → (new → action_size) ---
        new_actor = nn.Linear(new_hidden_size, self.action_size)
        nn.init.zeros_(new_actor.weight)
        nn.init.zeros_(new_actor.bias)
        new_actor.weight.data[:, :old] = self.actor.weight.data
        new_actor.bias.data[:] = self.actor.bias.data

        # --- critic: (old → 1) → (new → 1) ---
        # --- gru: (old -> old) -> (new -> new) ---
        new_gru = nn.GRUCell(new_hidden_size, new_hidden_size)
        for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            src = getattr(self.gru, name).data
            dst = getattr(new_gru, name).data
            dst.zero_()
            if src.dim() == 2:
                # GRUCell stacks 3 gates along dim 0; copy each gate block
                # into the top-left of its new block rather than the top of
                # the whole matrix, or reset and update swap places.
                for g in range(3):
                    dst[g*new_hidden_size:g*new_hidden_size+old, :old] =                         src[g*old:(g+1)*old, :old]
            else:
                for g in range(3):
                    dst[g*new_hidden_size:g*new_hidden_size+old] = src[g*old:(g+1)*old]

        new_critic = nn.Linear(new_hidden_size, 1)
        nn.init.zeros_(new_critic.weight)
        nn.init.zeros_(new_critic.bias)
        new_critic.weight.data[:, :old] = self.critic.weight.data
        new_critic.bias.data[:] = self.critic.bias.data

        self.embed_proj = new_embed_proj
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.gru = new_gru
        self.actor = new_actor
        self.critic = new_critic
        self.hidden_size = new_hidden_size

    # ------------------------------------------------------------------
    # Vocabulary growth: more SYMBOLS
    # ------------------------------------------------------------------
    def expand_vocab(self, item_vocab=None, entity_vocab=None, tile_vocab=None):
        """Add rows to the embedding tables - room for symbols not yet defined.

        Output-preserving for free: a row that nothing indexes cannot change
        an activation, and existing ids keep their existing rows.  New rows
        get the same small init the tables started with rather than zeros,
        because identical rows would make every new symbol indistinguishable
        until gradients happened to separate them.
        """
        grown = []
        for name, want in (('item_embed', item_vocab),
                           ('entity_embed', entity_vocab),
                           ('tile_embed', tile_vocab)):
            table = getattr(self, name)
            if want is None or want == table.num_embeddings:
                continue
            if want < table.num_embeddings:
                raise ValueError(
                    '%s: refusing to shrink %d -> %d; dropping rows deletes '
                    'learned symbols' % (name, table.num_embeddings, want))
            bigger = nn.Embedding(want, table.embedding_dim)
            nn.init.normal_(bigger.weight, mean=0.0, std=0.1)
            bigger.weight.data[:table.num_embeddings] = table.weight.data
            setattr(self, name, bigger)
            grown.append('%s %d->%d' % (name, table.num_embeddings, want))
        return grown

    # ------------------------------------------------------------------
    # Embedding growth: richer MEANING per symbol
    # ------------------------------------------------------------------
    def expand_embed_dim(self, item_dim=None, entity_dim=None, tile_dim=None):
        """Widen the embedding vectors, and re-lay-out ``embed_proj`` to match.

        This is the one with the trap.  Widening a table changes
        ``embed_input_dim``, so embed_proj's input must be rebuilt - and
        because its columns are a concatenation, every block after the one
        that grew now starts somewhere else.  Old weights are copied block by
        block into their NEW offsets; the fresh columns are zeroed so the
        network's output is bit-identical the instant this returns.
        """
        old_e = self.entity_embed.embedding_dim
        old_i = self.item_embed.embedding_dim
        old_t = self.tile_embed.embedding_dim
        new_e = old_e if entity_dim is None else entity_dim
        new_i = old_i if item_dim is None else item_dim
        new_t = old_t if tile_dim is None else tile_dim
        if (new_e, new_i, new_t) == (old_e, old_i, old_t):
            return []
        for label, o, n in (('entity', old_e, new_e), ('item', old_i, new_i),
                            ('tile', old_t, new_t)):
            if n < o:
                raise ValueError('%s_dim: refusing to shrink %d -> %d' % (label, o, n))

        old_blocks = self._embed_blocks(old_e, old_i, old_t)
        new_blocks = self._embed_blocks(new_e, new_i, new_t)
        old_proj_w = self.embed_proj.weight.data
        old_proj_b = self.embed_proj.bias.data

        for name, new_dim in (('entity_embed', new_e), ('item_embed', new_i),
                              ('tile_embed', new_t)):
            table = getattr(self, name)
            if new_dim == table.embedding_dim:
                continue
            wider = nn.Embedding(table.num_embeddings, new_dim)
            nn.init.normal_(wider.weight, mean=0.0, std=0.1)
            wider.weight.data[:, :table.embedding_dim] = table.weight.data
            setattr(self, name, wider)

        new_proj = nn.Linear(self.embed_input_dim, self.hidden_size)
        nn.init.zeros_(new_proj.weight)
        new_proj.bias.data[:] = old_proj_b
        for (o_start, o_w), (n_start, _n_w) in zip(old_blocks, new_blocks):
            new_proj.weight.data[:, n_start:n_start + o_w] =                 old_proj_w[:, o_start:o_start + o_w]
        self.embed_proj = new_proj
        return ['entity %d->%d' % (old_e, new_e), 'item %d->%d' % (old_i, new_i),
                'tile %d->%d' % (old_t, new_t)]

    # ------------------------------------------------------------------
    # Save / load with metadata (hidden size stored in checkpoint)
    # ------------------------------------------------------------------
    def save_checkpoint(self, path=None):
        path = path or config_rl.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "item_vocab": self.item_embed.num_embeddings,
            "entity_vocab": self.entity_embed.num_embeddings,
            "tile_vocab": self.tile_embed.num_embeddings,
            "item_dim": self.item_embed.embedding_dim,
            "entity_dim": self.entity_embed.embedding_dim,
            "tile_dim": self.tile_embed.embedding_dim,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path=None, map_location="cpu"):
        if path is None:
            path = config_rl.BEST_MODEL_PATH if config_rl.BEST_MODEL_PATH.exists() else config_rl.MODEL_PATH
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "hidden_size" in payload:
            model = cls(
                state_size=payload.get("state_size", config_rl.STATE_SIZE),
                action_size=payload.get("action_size", config_rl.ACTION_SIZE),
                hidden_size=payload.get("hidden_size", config_rl.MIND_HIDDEN_SIZE),
                item_vocab=payload.get("item_vocab"),
                entity_vocab=payload.get("entity_vocab"),
                tile_vocab=payload.get("tile_vocab"),
                item_dim=payload.get("item_dim"),
                entity_dim=payload.get("entity_dim"),
                tile_dim=payload.get("tile_dim"),
            )
            sd = payload.get("model_state_dict", payload.get("state_dict", payload))
            model.load_state_dict(sd)
        else:
            # Legacy checkpoint (plain state_dict)
            model = cls()
            model.load_state_dict(payload)
        return model
