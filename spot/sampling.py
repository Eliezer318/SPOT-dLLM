"""Sampling with coupled randomness.

Two decoders that sample from the same distribution should be judged equivalent even
when a single run diverges.  To separate a genuine distributional change from an
unlucky draw, both decoders are driven from the *same* random stream: the Gumbel noise
is a deterministic function of (seed, absolute position), so whenever two decoders
evaluate the same slot with the same conditional they are forced to make the same draw.

Correctness of the sampler under this coupling: a slot is committed exactly once, and at
that moment its token is argmax(logits/T + G_p) with G_p iid Gumbel drawn independently
of the logits.  That is the Gumbel-max trick, so the committed token is distributed as
softmax(logits/T) regardless of the noise being reused across rounds for slots that have
not yet been committed.
"""

from __future__ import annotations

from typing import Optional

import torch


class CoupledNoise:
    """Per-position Gumbel noise, deterministic given (seed, absolute position)."""

    def __init__(self, seed: int, lo: int, hi: int, vocab: int, device, dtype=torch.float32):
        """Pre-draw noise for absolute positions [lo, hi) -- the generation region."""
        self.lo, self.hi, self.vocab, self.seed = lo, hi, vocab, seed
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        u = torch.rand((hi - lo, vocab), generator=g, dtype=torch.float32)
        # Gumbel(0,1) = -log(-log(U));  clamp keeps -inf/nan out at the ends
        self.noise = (-torch.log(-torch.log(u.clamp_(1e-20, 1.0 - 1e-7)))).to(device=device, dtype=dtype)

    def block(self, b0: int, b1: int) -> torch.Tensor:
        """Noise rows for absolute positions [b0, b1)."""
        return self.noise[b0 - self.lo : b1 - self.lo]


def perturbed_scores(
    probs: torch.Tensor, temperature: float, noise: Optional[torch.Tensor]
) -> torch.Tensor:
    """Scores whose argmax is a sample from the conditional.

    `probs` are already softmax(logits/T) -- taking the log recovers logits/T up to a
    per-row constant, which argmax is invariant to.  At T = 0 the sampler degenerates to
    argmax(probs), i.e. greedy, and no noise is used.
    """
    if temperature <= 0.0 or noise is None:
        return probs
    return torch.log(probs.clamp_min(1e-30)) + noise


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """TV distance per row, for (n, V) probability tensors."""
    return 0.5 * (p - q).abs().sum(dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, in nats."""
    p = p.clamp_min(1e-30)
    q = q.clamp_min(1e-30)
    return (p * (p.log() - q.log())).sum(dim=-1)


def commit_law(probs: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Exact pmf of select_commit's (slot, token) commit under fresh Gumbel token draws.

    Mechanism (decode.select_commit, single unmasking, T>0): per masked slot s, the token
    t_s = argmax_t(log p_s + gumbel) ~ p_s; conf_s = p_s(t_s); the committed slot is
    argmax_s conf_s with FIRST-INDEX tie-break (torch.argmax semantics).  Hence

        P(s, t) = p_s(t) * prod_{s'<s} P(conf_{s'} <  p_s(t))
                         * prod_{s'>s} P(conf_{s'} <= p_s(t))     (masked s' only)

    with P(conf_{s'} < v) = sum_{t': p_{s'}(t') < v} p_{s'}(t') and <= analogously.
    Returns (S, V); rows of unmasked slots are zero.  Sums to 1 over masked support.

    This is the law of INDEPENDENT sampling semantics: the coupled-noise decoder reuses
    one Gumbel table across rounds, which is a different (also valid) coupling of the
    same per-round conditionals.  rs verification is lossless w.r.t. this law.
    """
    S, V = probs.shape
    m_idx = masked.nonzero(as_tuple=True)[0]
    sortv, _ = torch.sort(probs, dim=-1)                       # (S, V) ascending
    cum0 = torch.zeros(S, V + 1, dtype=probs.dtype, device=probs.device)
    torch.cumsum(sortv, dim=-1, out=cum0[:, 1:])
    q = probs.reshape(1, -1)                                   # queries, all (s,t) values
    acc = torch.zeros(S, V, dtype=probs.dtype, device=probs.device)
    for sp in m_idx.tolist():
        lo = torch.searchsorted(sortv[sp].contiguous(), q, side="left")
        hi = torch.searchsorted(sortv[sp].contiguous(), q, side="right")
        strict = cum0[sp][lo].view(S, V)                       # P(conf_sp <  v)
        incl = cum0[sp][hi].view(S, V)                         # P(conf_sp <= v)
        lg_strict = torch.log(strict.clamp_min(1e-45))
        lg_incl = torch.log(incl.clamp_min(1e-45))
        # slot sp is a competitor for every OTHER slot s: strict if sp < s else inclusive
        w = torch.where(
            torch.arange(S, device=probs.device).view(S, 1) > sp, lg_strict, lg_incl)
        w[sp] = 0.0
        acc += w
    law = probs * torch.exp(acc)
    law[~masked] = 0.0
    return law


def sample_rs_chain(p_prev: torch.Tensor, block_state: torch.Tensor, mask_id: int,
                    budget: int, generator=None):
    """Draft a depth-`budget` chain by SAMPLING the stale commit mechanism.

    Each link simulates select_commit on the stale marginals with fresh Gumbel noise, so
    the link's proposal is a genuine draw from q_j = commit_law(p_prev, masked_j) — the
    property rejection-sampling verification needs.  Records q_j(pair) per link.
    Returns drafts in the shared materialisation shape plus rs fields
    (pair, q_pair, src_probs).
    """
    state = block_state.clone()
    out = []
    for d in range(budget):
        masked = state == mask_id
        if int(masked.sum()) == 0:
            break
        pass  # cheap path: per-pair q below
        g = -torch.log(-torch.log(
            torch.rand(p_prev.shape, device=p_prev.device, generator=generator)
            .clamp(1e-20, 1 - 1e-7)))
        tok = (torch.log(p_prev.clamp_min(1e-30)) + g).argmax(dim=-1)
        conf = p_prev.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        cm = torch.where(masked, conf, torch.full_like(conf, -1.0))
        s = int(cm.argmax())
        t = int(tok[s])
        state = state.clone()
        state[s] = t
        out.append({"node": d, "level": d + 1, "formula": frozenset(),
                    "state": state.clone(), "chosen": {s: t}, "score": 1.0, "local": 1.0,
                    "pair": (s, t), "q_pair": commit_law_pair(p_prev, masked, s, t),
                    "src_probs": p_prev})
    return out


def commit_law_pair(probs, masked, s, t):
    """p(slot=s, token=t) under commit_law, without materialising the full (S,V) law.
    Same tie-break (first index wins): strict CDF for competitors before s, inclusive
    after.  Cost: one sort reuse per call site is the caller's concern; here S small ops."""
    if not bool(masked[s]):
        return 0.0
    v = probs[s, t]
    m_idx = masked.nonzero(as_tuple=True)[0]
    pr = float(probs[s, t])
    for sp in m_idx.tolist():
        if sp == s:
            continue
        row = probs[sp]
        side = row < v if sp < s else row <= v
        pr *= float(row[side].sum().clamp(max=1.0))
        if pr == 0.0:
            return 0.0
    return pr


def rs_commit_sample(probs, masked, generator):
    """One commit sampled by SIMULATING the mechanism with fresh Gumbel -- identical in
    law to multinomial over commit_law, at a fraction of the cost (cheap path)."""
    import torch as _t
    g = -_t.log(-_t.log(_t.rand(probs.shape, device=probs.device,
                                generator=generator).clamp(1e-20, 1 - 1e-7)))
    tok = (_t.log(probs.clamp_min(1e-30)) + g).argmax(-1)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    cm = _t.where(masked, conf, _t.full_like(conf, -1.0))
    sl = int(cm.argmax())
    return sl, int(tok[sl])
