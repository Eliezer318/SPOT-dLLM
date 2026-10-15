"""Model zoo for the slot picker.

Every architecture here solves the same problem and exposes the same interface as
`SlotPicker`: given (B, S, F) per-slot features -- the stale top-M table plus the static
scalars, and in prefix-conditional mode two path features (taken flag, depth) -- return
(B, K, S) logits over slots. Only the inductive bias differs, so a difference in accuracy
is attributable to the architecture and nothing else.

Why these four, given what measured:

`ptr`  POINTER HEAD (Vinyals' Pointer Networks; Set Interdependence Transformer,
       arXiv 2206.03720). The baseline scores every slot with an independent linear
       projection after self-attention. A pointer scores each slot by dot product against
       a GLOBAL QUERY pooled from the whole set, which is the canonical formulation of
       "select one element of a set" and makes the decision explicitly comparative --
       the property says the task has (30% of misses are near-ties between slots).

`rel`  RELATIVE-POSITION BIAS (Shaw et al.; T5 bias; ALiBi) plus an explicit
       distance-to-nearest-filled input. The next commit is adjacent to the previous one
       54-65% of the time against a 12-20% chance rate, yet the baseline sees only
       ABSOLUTE position and an order-free taken flag, so locality must be learned from
       scratch through position embeddings. Here it is an architectural prior.

`gru`  RECURRENT-OVER-DEPTH HEAD (ReDrafter's recurrent drafter; Hydra's sequentially
       dependent heads). The baseline re-reads the same stale table at every tree depth
       with a 2-number summary of the path. This carries a state across depth instead,
       which is the natural way to model the table's UPDATE rather than re-read a table
       that is k commits out of date.

`moe`  DECOUPLED MoE HEAD (Jakiro, arXiv 2502.06282). One shared trunk, several expert
       heads, gated by depth/progress. Motivated by an earlier experiment's depth structure: shallow and
       deep decisions have different statistics (near-tie share falls .39 -> .26 with
       depth, rank>=4 tail stays flat), so one head must average over regimes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .slotpicker import N_PATH, N_STATIC, SlotPicker

REL_BUCKETS = 16


def _trunk(in_dim: int, d: int, layers: int, heads: int):
    inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True,
                                   norm_first=True, dropout=0.0), layers)
    return inp, enc


class _Base(nn.Module):
    """Shared plumbing: the zoo keeps SlotPicker's constructor signature and its
    (feats, top_tok, pad_mask) -> (B, K, S) contract."""

    def __init__(self, m_top=32, d_model=128, layers=3, heads=4, k_steps=8, vocab=0,
                 tok_dim=32, prefix_cond=False, context=False, hid=0):
        super().__init__()
        self.m_top, self.hid, self.prefix_cond = m_top, hid, prefix_cond
        self.k_steps = 1 if prefix_cond else k_steps
        self.in_dim = (2 * m_top + N_STATIC + (N_PATH if prefix_cond else 0)
                       + (tok_dim if vocab else 0) + hid)
        self.tok_emb = nn.Embedding(vocab, tok_dim) if vocab else None
        self.d = d_model

    def _pre(self, feats, top_tok):
        if self.tok_emb is not None:
            feats = torch.cat([feats, self.tok_emb(top_tok)], dim=-1)
        return feats


class PointerPicker(_Base):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.inp, self.enc = _trunk(self.in_dim, self.d, kw.get("layers", 3),
                                    kw.get("heads", 4))
        self.q = nn.Sequential(nn.Linear(2 * self.d, self.d), nn.GELU(),
                               nn.Linear(self.d, self.d * self.k_steps))
        self.key = nn.Linear(self.d, self.d)
        self.scale = self.d ** -0.5

    def forward(self, feats, top_tok, pad_mask: Optional[torch.Tensor] = None):
        h = self.enc(self.inp(self._pre(feats, top_tok)), src_key_padding_mask=pad_mask)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask[..., None], 0.0)
            n = (~pad_mask).sum(1, keepdim=True).clamp_min(1)
            pooled = torch.cat([h.sum(1) / n, h.max(1).values], -1)
        else:
            pooled = torch.cat([h.mean(1), h.max(1).values], -1)
        q = self.q(pooled).view(h.shape[0], self.k_steps, self.d)
        logits = torch.einsum("bkd,bsd->bks", q, self.key(h)) * self.scale
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits


class RelPosPicker(_Base):
    """Attention biased by slot distance, plus distance-to-nearest-filled as an input."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.layers_n = kw.get("layers", 3)
        self.heads = kw.get("heads", 4)
        self.inp = nn.Sequential(nn.Linear(self.in_dim + 2, self.d), nn.GELU(),
                                 nn.Linear(self.d, self.d))
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                n1=nn.LayerNorm(self.d), n2=nn.LayerNorm(self.d),
                attn=nn.MultiheadAttention(self.d, self.heads, batch_first=True),
                ff=nn.Sequential(nn.Linear(self.d, 4 * self.d), nn.GELU(),
                                 nn.Linear(4 * self.d, self.d)))) for _ in range(self.layers_n)])
        self.rel = nn.Embedding(REL_BUCKETS, self.heads)
        self.head = nn.Linear(self.d, self.k_steps)

    def _extra(self, feats):
        pos = feats[..., 2 * self.m_top + 4]            # normalised slot position
        if self.prefix_cond:
            taken = feats[..., -N_PATH] > 0.5
            dist = (pos[:, :, None] - pos[:, None, :]).abs()
            # only filled slots count as neighbours; unfilled ones are pushed out of range
            dist = torch.where(taken[:, None, :], dist, torch.full_like(dist, 4.0))
            near = dist.min(-1).values                   # distance to nearest taken slot
        else:
            near = torch.zeros_like(pos)
        return pos, torch.stack([near, (near < 1e-6).to(pos.dtype)], -1)

    def forward(self, feats, top_tok, pad_mask: Optional[torch.Tensor] = None):
        x = self._pre(feats, top_tok)
        pos, extra = self._extra(feats)
        h = self.inp(torch.cat([x, extra], -1))
        d = (pos[:, :, None] - pos[:, None, :]).abs()
        b = (d * (REL_BUCKETS - 1)).clamp(0, REL_BUCKETS - 1).long()
        bias = self.rel(b).permute(0, 3, 1, 2)           # (B, heads, S, S)
        B, S = pos.shape
        bias = bias.reshape(B * self.heads, S, S)
        for blk in self.blocks:
            y = blk["n1"](h)
            a, _ = blk["attn"](y, y, y, attn_mask=bias,
                               key_padding_mask=pad_mask, need_weights=False)
            h = h + a
            h = h + blk["ff"](blk["n2"](h))
        logits = self.head(h).transpose(1, 2)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits


class GRUPicker(_Base):
    """Encode the slots once, then carry a state across tree depth.

    Depth arrives as the path feature the baseline already receives, so the recurrence is
    unrolled over it: the state is advanced `depth` times from a pooled set summary. This
    keeps the (feats -> logits) contract while giving the model somewhere to accumulate
    what the prefix did, instead of re-reading a table that is `depth` commits stale.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.inp, self.enc = _trunk(self.in_dim, self.d, kw.get("layers", 3),
                                    kw.get("heads", 4))
        self.cell = nn.GRUCell(self.d, self.d)
        self.head = nn.Linear(2 * self.d, self.k_steps)
        self.max_unroll = 8

    def forward(self, feats, top_tok, pad_mask: Optional[torch.Tensor] = None):
        h = self.enc(self.inp(self._pre(feats, top_tok)), src_key_padding_mask=pad_mask)
        if pad_mask is not None:
            hm = h.masked_fill(pad_mask[..., None], 0.0)
            pooled = hm.sum(1) / (~pad_mask).sum(1, keepdim=True).clamp_min(1)
        else:
            pooled = h.mean(1)
        depth = feats[..., -1].amax(1) if self.prefix_cond else torch.zeros(h.shape[0],
                                                                           device=h.device)
        steps = int(min(self.max_unroll, max(1, round(float(depth.max()) * 32)))) \
            if self.prefix_cond else 1
        s = torch.zeros_like(pooled)
        for _ in range(steps):
            s = self.cell(pooled, s)
        logits = self.head(torch.cat([h, s[:, None, :].expand_as(h)], -1)).transpose(1, 2)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits


class MoEPicker(_Base):
    """Shared trunk, several expert heads, gated by the path features (Jakiro-style)."""

    def __init__(self, n_experts: int = 4, **kw):
        super().__init__(**kw)
        self.inp, self.enc = _trunk(self.in_dim, self.d, kw.get("layers", 3),
                                    kw.get("heads", 4))
        self.n_experts = n_experts
        self.experts = nn.ModuleList([nn.Linear(self.d, self.k_steps)
                                      for _ in range(n_experts)])
        gate_in = N_PATH if self.prefix_cond else 1
        self.gate = nn.Sequential(nn.Linear(gate_in + 1, 32), nn.GELU(),
                                  nn.Linear(32, n_experts))

    def forward(self, feats, top_tok, pad_mask: Optional[torch.Tensor] = None):
        h = self.enc(self.inp(self._pre(feats, top_tok)), src_key_padding_mask=pad_mask)
        ctx = feats[..., -N_PATH:].mean(1) if self.prefix_cond else \
            feats[..., :1].mean(1)
        prog = feats[..., 2 * self.m_top + 7].mean(1, keepdim=True)
        w = torch.softmax(self.gate(torch.cat([ctx, prog], -1)), -1)   # (B, E)
        outs = torch.stack([e(h) for e in self.experts], 1)            # (B, E, S, K)
        logits = (outs * w[:, :, None, None]).sum(1).transpose(1, 2)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits




class Champ2(nn.Module):
    """/the best trunk plus exactly one new channel per arm.

    Identical rows to SlotPicker's prefix-conditional mode -- the stale top-M table +
    statics + path features over the still-masked slots -- with two OPTIONAL additions,
    each individually toggleable so an A/B attributes its delta to one thing:

      tok_head=True: a vocab-free token head scoring each slot's stale top-C
        candidates (inputs: candidate prob, log-prob, rank embedding, row state). The
        ladder showed this head is a small consistent win even on a losing trunk
        (g000 < tg0 by 0.9-1.9 NFE, all seeds); here it rides the trunk that wins.

      d_emb>0: per-slot TOKEN-IDENTITY features -- the frozen target embedding
        of the stale argmax and the prob-weighted mean embedding of the top-C -- entering
        through a ZERO-INITIALISED projection. an earlier experiment's lesson in code: the hs proposer
        lost by letting a new input channel outvote a correct prior, so this channel
        starts at exactly zero contribution and must earn its way in via gradient.

    Caller contract: feats = [table+path rows (F0 dims) | raw emb features (2*d_emb)],
    concatenated by the caller (trainer and drafter both build the emb block from the
    same stale top-C, so they cannot drift).
    """

    def __init__(self, m_top=32, d_model=128, layers=3, heads=4, k_steps=1, vocab=0,
                 tok_dim=32, prefix_cond=True, context=False, hid=0,
                 n_cand=4, d_emb=0, tok_head=False, extra_dim=0):
        super().__init__()
        assert prefix_cond, "champ2 is a tree model"
        self.m_top, self.n_cand, self.d_emb, self.has_tok_head = m_top, n_cand, d_emb, tok_head
        self.k_steps = 1
        self.prefix_cond = True
        self.hid = 0
        self.f0 = 2 * m_top + N_STATIC + N_PATH
        # width of the zero-init side channel: candidate-embedding block (2*d_emb,
        # ) or an arbitrary row like the RAW pre-head hidden state (extra_dim,
        #).
        self.extra_dim = extra_dim or (2 * d_emb if d_emb else 0)
        self.inp = nn.Sequential(nn.Linear(self.f0, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        if self.extra_dim:
            self.emb_proj = nn.Sequential(nn.Linear(self.extra_dim, d_model), nn.GELU(),
                                          nn.Linear(d_model, d_model))
            # zero-init the LAST layer: the channel starts silent(lesson)
            nn.init.zeros_(self.emb_proj[-1].weight)
            nn.init.zeros_(self.emb_proj[-1].bias)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                         batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)
        if tok_head:
            self.cand_emb = nn.Embedding(n_cand, d_model // 4)
            self.cand_p = nn.Linear(2, d_model // 4)
            self.tok_head = nn.Sequential(
                nn.Linear(d_model + d_model // 4, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1))

    def _trunk(self, feats, pad_mask):
        x = self.inp(feats[..., :self.f0])
        if self.extra_dim:
            x = x + self.emb_proj(feats[..., self.f0:])
        return self.enc(x, src_key_padding_mask=pad_mask)

    def forward(self, feats, top_tok=None, pad_mask=None):
        """SlotPicker-compatible: (B, S, F) -> (B, 1, S) position logits."""
        h = self._trunk(feats, pad_mask)
        logits = self.head(h).transpose(1, 2)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :], -1e9)
        return logits

    def forward_full(self, feats, cand_p, pad_mask=None):
        """(B,S,F), (B,S,C) -> (pos (B,S), tok (B,S,C)). Requires tok_head."""
        h = self._trunk(feats, pad_mask)
        pos = self.head(h).squeeze(-1)
        if pad_mask is not None:
            pos = pos.masked_fill(pad_mask, -1e9)
        cp = cand_p.clamp_min(1e-9)
        cf = (self.cand_p(torch.stack([cp, cp.log() / 10.0], -1))
              + self.cand_emb(torch.arange(self.n_cand, device=feats.device))[None, None])
        tok = self.tok_head(
            torch.cat([h[:, :, None, :].expand(-1, -1, self.n_cand, -1), cf], -1)
        ).squeeze(-1)
        return pos, tok


def emb_features(W, cand_ids, cand_p):
    """Raw token-identity block, ONE implementation for trainer and drafter(rule).

    W (V, d_emb) frozen target embedding; cand_ids/cand_p (S, C) stale top-C.
    Returns (S, 2*d_emb) fp32: [argmax embedding | prob-weighted mean embedding].
    """
    e = W[cand_ids].float()                       # (S, C, d)
    w = cand_p.float().clamp_min(1e-9)
    mean = (e * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)
    return torch.cat([e[:, 0], mean], -1)


ZOO = {"enc": SlotPicker, "ptr": PointerPicker, "rel": RelPosPicker,
       "gru": GRUPicker, "moe": MoEPicker, "champ2": Champ2}


def build_picker(arch: str = "enc", **kw):
    if arch not in ZOO:
        raise ValueError(f"unknown arch {arch!r}; have {sorted(ZOO)}")
    return ZOO[arch](**kw)
