"""two-head proposer over the target's own pre-head hidden states.

Why this exists, given that rung 1 already measured hidden features as a clean negative
(26.52 vs 24.87 D8 NFE, val_acc 0.69 < 0.71): rung 1 used a FROZEN RANDOM 4096->32
projection appended to the picker's scalar features -- a 128x lossy sketch feeding the
position head only.  This model differs in exactly the two ways the oracle says
matter:

  1. the full pre-head state enters through a LEARNED projection and is the PRIMARY
     per-position input, not a bolt-on;
  2. there is a TOKEN head.  The position oracle saturates at E[depth] 6.18 because the
     stale argmax dies compounding (.94^k); a perfect top-2 hedge recovers 42% of the
     whole token-oracle gain.  The token head scores the stale top-C candidates per slot,
     conditioned on what the branch has committed since the forward -- the update the
     stale marginal cannot see.

Architecture: BERT-style bidirectional encoder over ALL block positions (a partially
filled block IS a masked-denoising problem; filled rows are the anchors commits happen
around, so they belong in the attention).  Two heads:

  position: per-row scalar -> logits over still-masked rows.  Same contract as the
            prefix-conditional picker: one head, re-asked per tree node.
  token:    per masked row, scores over that row's stale top-C token candidates.
            Vocab-free by construction -- token identity enters ONLY through
            stale-table coordinates (candidate rank + probability), so no 152k
            embedding table and no vocabulary coupling.

Losslessness is structural, exactly as for the slot picker: this model only decides which
future states get precomputed.  Every committed token remains the target's own sample at
the correct state; a bad proposal costs speed, never correctness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .slotpicker import N_STATIC, slot_features

# row state classes: candidate (still masked), filled (before this round's drafting
# started -- includes the verify round's own fresh commits), taken (filled by THIS
# branch's hypothetical path since the forward).
CLS_CAND, CLS_FILLED, CLS_TAKEN = 0, 1, 2


class HSProposer(nn.Module):
    def __init__(self, d_hid: int, m_top: int = 32, n_cand: int = 4, d_model: int = 256,
                 layers: int = 4, heads: int = 8, max_len: int = 32):
        super().__init__()
        self.d_hid, self.m_top, self.n_cand = d_hid, m_top, n_cand
        self.hproj = nn.Linear(d_hid, d_model)
        self.tabproj = nn.Linear(2 * m_top + N_STATIC, d_model)
        self.cls_emb = nn.Embedding(3, d_model)
        # rank of the token a TAKEN row holds, as a candidate index 0..n_cand-1; row
        # classes other than TAKEN use a dedicated null bucket.
        self.rank_emb = nn.Embedding(n_cand + 1, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.glob = nn.Linear(2, d_model)  # (depth/L, progress/L), broadcast to all rows
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                         batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pos_head = nn.Linear(d_model, 1)
        # token head: row representation x candidate coordinates -> C scores.
        self.cand_emb = nn.Embedding(n_cand, d_model // 4)
        self.cand_p = nn.Linear(2, d_model // 4)  # (p, log p)
        self.tok_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    def encode_static(self, hid, table):
        """Branch-independent input terms, computed ONCE per round at inference.

        hid (L, d_hid), table (L, F) -> (L, d_model). The drafter re-asks the model at
        every tree node; only cls/rank/glob change across nodes, so the hproj of a 4096-d
        row -- the expensive term -- should not be recomputed per node.
        """
        L = hid.shape[0]
        return (self.hproj(hid.float()) + self.tabproj(table)
                + self.pos_emb(torch.arange(L, device=hid.device)))

    def forward(self, hid, table, cls, taken_rank, depth, progress, cand_p,
                pad_mask=None, static=None):
        """All-position rows -> (pos_logits (B,L), tok_logits (B,L,C)).

        hid:        (B, L, d_hid) stale pre-head hidden rows (fp16 ok; cast here)
        table:      (B, L, 2*m_top+N_STATIC) slot_features rows, zeros for filled rows
        cls:        (B, L) int in {CAND, FILLED, TAKEN}
        taken_rank: (B, L) int candidate-rank of the token a TAKEN row holds
                    (n_cand = null bucket for non-taken rows / OOV)
        depth:      (B) float commits-since-forward / L  -- table staleness
        progress:   (B) float filled / L at the branch state
        cand_p:     (B, L, C) stale probs of each row's top-C token candidates
        pad_mask:   (B, L) True where the row is padding (bed pooling); optional
        """
        if static is None:
            B, L, _ = hid.shape
            pos = torch.arange(L, device=hid.device)
            static = (self.hproj(hid.float()) + self.tabproj(table)
                      + self.pos_emb(pos)[None])
        else:
            B, L = cls.shape
            static = static[None] if static.dim() == 2 else static
        x = (static + self.cls_emb(cls) + self.rank_emb(taken_rank)
             + self.glob(torch.stack([depth, progress], -1))[:, None, :])
        h = self.enc(x, src_key_padding_mask=pad_mask)
        pos_logits = self.pos_head(h).squeeze(-1)  # (B, L); caller masks non-candidates
        cp = cand_p.clamp_min(1e-9)
        cf = (self.cand_p(torch.stack([cp, cp.log() / 10.0], -1))
              + self.cand_emb(torch.arange(self.n_cand, device=cand_p.device))[None, None])
        tok_logits = self.tok_head(
            torch.cat([h[:, :, None, :].expand(-1, -1, self.n_cand, -1), cf], -1)
        ).squeeze(-1)  # (B, L, C)
        if pad_mask is not None:
            pos_logits = pos_logits.masked_fill(pad_mask, -1e9)
        return pos_logits, tok_logits


def round_tensors(p_top: torch.Tensor, idx: torch.Tensor, L: int, n_cand: int):
    """Branch-INDEPENDENT row tensors for one round, shared by trainer and drafter.

    Built in one place so training and inference cannot drift apart (the lesson:
    a feature computed two ways is a measured 0.78-NFE bug).

    p_top: (S, M) stale top-M probs of the candidate rows -- the slots still maskable
           when drafting starts (the verify round's own commit is NOT a candidate).
    idx:   (S) block-local indices of those rows.
    Returns table (L, 2M+N_STATIC) with zeros on non-candidate rows, and
    cand_p (L, n_cand): stale probs of each candidate row's top-C tokens.
    """
    S = int(idx.numel())
    f = slot_features(p_top, idx.float(), L, S, L - S)
    table = f.new_zeros(L, f.shape[1])
    table[idx] = f
    cand_p = p_top.new_zeros(L, n_cand)
    cand_p[idx] = p_top[:, :n_cand]
    return table, cand_p
