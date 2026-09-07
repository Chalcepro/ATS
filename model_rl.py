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

    def __init__(self, state_size=None, action_size=None, hidden_size=None):
        super().__init__()
        self.state_size = state_size or config_rl.STATE_SIZE
        self.action_size = action_size or config_rl.ACTION_SIZE
        self.hidden_size = hidden_size or config_rl.MIND_HIDDEN_SIZE

        # Learned embedding tables
        self.item_embed = nn.Embedding(config_rl.ITEM_VOCAB_SIZE, config_rl.ITEM_EMBED_DIM)
        self.entity_embed = nn.Embedding(config_rl.ENTITY_VOCAB_SIZE, config_rl.ENTITY_EMBED_DIM)
        self.tile_embed = nn.Embedding(config_rl.TILE_VOCAB_SIZE, config_rl.TILE_EMBED_DIM)
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
        self.actor = nn.Linear(self.hidden_size, self.action_size)
        self.critic = nn.Linear(self.hidden_size, 1)

    @property
    def embed_input_dim(self) -> int:
        """Width of the concatenated embedding features fed to ``embed_proj``.

        nearest-entity + nearest-object + pooled inventory + flattened
        egocentric tile patch.  The patch is flattened rather than pooled on
        purpose — pooling would discard *where* each tile is, which is the
        entire reason the patch was added.
        """
        return (config_rl.ENTITY_EMBED_DIM
                + config_rl.ITEM_EMBED_DIM          # nearest object on the ground
                + config_rl.ITEM_EMBED_DIM          # pooled inventory contents
                + config_rl.PATCH_TILES * config_rl.TILE_EMBED_DIM)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state, action_mask=None):
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

        ent_idx = state_in[..., config_rl.IDX_ENT_TYPE].round().long().clamp(0, config_rl.ENTITY_VOCAB_SIZE - 1)
        obj_idx = (state_in[..., config_rl.IDX_OBJ_ID] * config_rl.ITEM_VOCAB_SIZE).round().long().clamp(0, config_rl.ITEM_VOCAB_SIZE - 1)

        ent_vec = self.entity_embed(ent_idx)
        obj_vec = self.item_embed(obj_idx)

        # Inventory: 7 slots of (norm_item_id, norm_count) -> item ids live on
        # the even offsets.  These used to bypass the embedding table entirely
        # and enter the net as raw normalised floats, so a 100x32 table was
        # trained on exactly one input (the nearest ground object).  Empty
        # slots are masked out of the mean so padding doesn't wash it out.
        inv_start = config_rl.IDX_INV_START
        inv_end = inv_start + config_rl.INVENTORY_SLOTS * 2
        inv_idx = (state_in[..., inv_start:inv_end:2] * config_rl.ITEM_VOCAB_SIZE)
        inv_idx = inv_idx.round().long().clamp(0, config_rl.ITEM_VOCAB_SIZE - 1)
        inv_emb = self.item_embed(inv_idx)                              # (..., slots, dim)
        inv_valid = (inv_idx > 0).to(inv_emb.dtype).unsqueeze(-1)
        inv_vec = (inv_emb * inv_valid).sum(-2) / inv_valid.sum(-2).clamp(min=1.0)

        # Egocentric tile patch -> embedded and flattened (position preserved).
        patch_start = config_rl.IDX_PATCH_START
        patch_end = patch_start + config_rl.PATCH_TILES
        patch_idx = state_in[..., patch_start:patch_end].round().long().clamp(0, config_rl.TILE_VOCAB_SIZE - 1)
        patch_vec = self.tile_embed(patch_idx).flatten(start_dim=-2)

        emb_feat = self.embed_proj(torch.cat([ent_vec, obj_vec, inv_vec, patch_vec], dim=-1))

        x = torch.relu(self.fc1(state_in) + emb_feat)
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        if state.dim() == 1:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)
        return logits, value

    def act(self, state, action_mask=None):
        """Sample an action from the policy (used during rollouts)."""
        logits, value = self.forward(state, action_mask=action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, states, actions, action_masks=None):
        """Evaluate log-probs, values, entropy for a batch (PPO update)."""
        logits, values = self.forward(states, action_mask=action_masks)
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
        new_critic = nn.Linear(new_hidden_size, 1)
        nn.init.zeros_(new_critic.weight)
        nn.init.zeros_(new_critic.bias)
        new_critic.weight.data[:, :old] = self.critic.weight.data
        new_critic.bias.data[:] = self.critic.bias.data

        self.embed_proj = new_embed_proj
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.actor = new_actor
        self.critic = new_critic
        self.hidden_size = new_hidden_size

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
            )
            sd = payload.get("model_state_dict", payload.get("state_dict", payload))
            model.load_state_dict(sd)
        else:
            # Legacy checkpoint (plain state_dict)
            model = cls()
            model.load_state_dict(payload)
        return model
