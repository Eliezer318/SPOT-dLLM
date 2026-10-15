"""`D_slot`: a small trained drafter that chooses *which positions* to draft, nothing else.

Motivation, from this project's own measurements: given the slot, the token is essentially
free (P(token | slot) = 0.97 -- the target's own argmax at that slot is what gets committed
90.1% of the time, and j<=8 covers 99.7%).  What actually decays with draft depth is the
*ordering*: P(next commit = top of a k-round-old confidence ranking) is 0.62 / 0.19 / 0.07
for k = 1 / 2 / 3 at b8.  The free drafter uses that stale ranking directly.  So the only
quantity worth learning is the ordering, and this model learns exactly that and nothing
else.

Input is the table the decoder already has for free: for every still-masked slot, the
target's own top-M probabilities from the previous round (M = 32 by default), plus the
identity of the top token at that slot.  No hidden states, no backbone, no extra forward.

Output is a ranking over slots for each of the next K commits -- i.e. the chain of positions
to draft.  The drafted token at each chosen slot is the target's own argmax there, which is
what makes this a pure *position* drafter.

Losslessness is structural and needs no argument: this model only decides which future
states get precomputed.  Every committed token remains the target's own sample at the
correct state, so a bad ranking costs speed and never correctness.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


N_STATIC = 8   # non-probability features per slot; see `slot_features`
N_PATH = 2     # extra features in prefix-conditional (tree) mode: see `path_features`
N_CTX = 3      # filled / left-filled / right-filled, in context mode: see `block_features`


def slot_features(p_top: torch.Tensor, slot_idx: torch.Tensor, block_len: int,
                  n_masked: int, progress: int) -> torch.Tensor:
    """(S, M) top-M probabilities -> (S, M + N_STATIC) per-slot feature rows.

    Probabilities enter as-is *and* as logs: the ranking decision is scale-sensitive at the
    top (which slot is most confident) and shape-sensitive in the tail (how contested a slot
    is), and a single representation serves one of those badly.
    """
    S, M = p_top.shape
    dev = p_top.device
    eps = 1e-9
    top1 = p_top[:, 0]
    top2 = p_top[:, 1] if M > 1 else torch.zeros_like(top1)
    ent = -(p_top * (p_top + eps).log()).sum(-1)
    mass = p_top.sum(-1)  # how much of the distribution the top-M actually covers
    static = torch.stack([
        top1,
        top1 - top2,                                   # margin: the free drafter's implicit signal
        ent,
        mass,
        slot_idx.float() / max(block_len - 1, 1),      # where in the block
        (slot_idx.float() - slot_idx.float().mean()).abs() / max(block_len, 1),
        torch.full((S,), n_masked / max(block_len, 1), device=dev),
        # `progress` = slots already written in this block.  It must be computed the same
        # way at training and inference: round_in_block is NOT, because a drafted round
        # commits ~2 tokens while a distillation round commits exactly 1, so the same round
        # index means a different fill level in the two settings (measured: ).
        torch.full((S,), progress / max(block_len, 1), device=dev),
    ], dim=-1)
    return torch.cat([p_top, (p_top + eps).log() / 10.0, static], dim=-1)


def block_features(p_top: torch.Tensor, filled: torch.Tensor, block_len: int,
                   n_masked: int, progress: int) -> torch.Tensor:
    """Whole-block rows: the written prefix as well as the candidates.

    `slot_features` shows the model only the still-masked slots and their marginals, so it
    has never seen what the block already CONTAINS -- which slots are written, with which
    tokens, or what sits either side of a candidate. A block-diffusion decoder commits
    around already-written anchors, so that context is exactly the signal that predicts the
    next position. Filled rows carry zeroed probability features plus their own token; the
    attention over the full block then lets a candidate see its neighbourhood.

    p_top: (L, M) top-M probabilities, rows for filled slots ignored.
    filled: (L) bool, True where the slot already holds a committed token.
    """
    L = int(filled.shape[0])
    dev = p_top.device
    live = (~filled).float().unsqueeze(-1)
    p = p_top * live                                    # zero out written slots
    eps = 1e-9
    top1 = p[:, 0]
    top2 = p[:, 1] if p.shape[1] > 1 else torch.zeros_like(top1)
    ent = -(p * (p + eps).log()).sum(-1)
    mass = p.sum(-1)
    pos = torch.arange(L, device=dev, dtype=torch.float32)
    left = torch.cat([filled.new_zeros(1), filled[:-1]]).float()
    right = torch.cat([filled[1:], filled.new_zeros(1)]).float()
    static = torch.stack([
        top1, top1 - top2, ent, mass,
        pos / max(L - 1, 1),
        (pos - pos.mean()).abs() / max(L, 1),
        torch.full((L,), n_masked / max(L, 1), device=dev),
        torch.full((L,), progress / max(L, 1), device=dev),  # see slot_features
    ], dim=-1)
    ctx = torch.stack([filled.float(), left, right], dim=-1)
    return torch.cat([p, (p + eps).log() / 10.0 * live, static, ctx], dim=-1)


def path_features(n_slots: int, taken: torch.Tensor, depth: int, block_len: int,
                  device=None) -> torch.Tensor:
    """Per-slot conditioning on the hypothetical path this tree node sits on.

    The chain model answered one question -- "which slot commits at step k" -- with the
    same logits for every branch, because the heads never saw which slots a branch had
    already taken.  A tree needs the opposite: two branches that filled different slots
    must predict differently from here on.  These two features carry that state.
    """
    f = torch.zeros(n_slots, N_PATH, device=device)
    f[:, 0] = taken.to(f.dtype)                       # already filled on THIS path
    f[:, 1] = depth / max(block_len, 1)               # how stale the table now is
    return f


class SlotPicker(nn.Module):
    """Transformer encoder over slots -> K ranking heads (one per future commit step).

    Attention across slots is the point: choosing a slot is a comparison against the other
    masked slots, not a per-slot judgement, so a per-slot MLP cannot express the task.
    """

    def __init__(self, m_top: int = 32, d_model: int = 128, layers: int = 3,
                 heads: int = 4, k_steps: int = 8, vocab: int = 0, tok_dim: int = 32,
                 prefix_cond: bool = False, context: bool = False, hid: int = 0):
        super().__init__()
        self.m_top = m_top
        self.k_steps = 1 if prefix_cond else k_steps
        self.prefix_cond = prefix_cond
        self.context = context
        self.hid = hid  # A2: projected target hidden-state features per slot
        in_dim = (2 * m_top + N_STATIC + (N_CTX if context else 0)
                  + (N_PATH if prefix_cond else 0) + (tok_dim if vocab else 0) + hid)
        self.tok_emb = nn.Embedding(vocab, tok_dim) if vocab else None
        self.inp = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                         batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        # prefix-conditional: ONE head asked recursively at every tree node.  Otherwise
        # k independent heads, all reading the same round-start state (the chain model).
        self.head = nn.Linear(d_model, self.k_steps)

    def forward(self, feats: torch.Tensor, top_tok: Optional[torch.Tensor],
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, S, F) -> (B, K, S) logits over slots, per future commit step."""
        if self.tok_emb is not None:
            feats = torch.cat([feats, self.tok_emb(top_tok)], dim=-1)
        h = self.enc(self.inp(feats), src_key_padding_mask=pad_mask)
        logits = self.head(h).transpose(1, 2)  # (B, K, S)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits

    @torch.no_grad()
    def chain(self, p_top: torch.Tensor, slot_idx: torch.Tensor, top_tok: torch.Tensor,
              block_len: int, progress: int, budget: int) -> list[int]:
        """Greedy chain of distinct slots: step k's argmax, excluding slots already taken.

        Returns block-local slot indices, in commit order -- the chain the decoder will
        materialise into nested draft states.
        """
        S = p_top.shape[0]
        feats = slot_features(p_top, slot_idx, block_len, S, progress)[None]
        logits = self.forward(feats, top_tok[None])[0]  # (K, S)
        taken: list[int] = []
        used = torch.zeros(S, dtype=torch.bool, device=logits.device)
        for k in range(min(budget, self.k_steps, S)):
            row = logits[k].masked_fill(used, -1e9)
            j = int(row.argmax())
            used[j] = True
            taken.append(int(slot_idx[j]))
        return taken
