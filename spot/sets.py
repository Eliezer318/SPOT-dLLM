"""SPOT-Sets: drafting whole commit sets for dynamic threshold decoding.

Under threshold decoding (Fast-dLLM) one target forward commits every masked position whose confidence
clears tau, else the single most confident one.  A step is then a SET of positions with their tokens, and
the stale argmax often misses it.  SPOT-Sets drafts future steps as whole sets:

  * a set drafter (5.1M parameters) reads, per block position, the stale top-32 table of the previous
    forward, the token each position holds, the target's projected hidden state and the staleness, and
    predicts P(position commits at the next step), the fallback position, and which of the stale top-16
    tokens the target will write there;
  * a tree whose nodes are whole steps: the children of a node are its top-K sets under the drafter's
    independent probabilities (the empty set maps to the fallback position), plus one sibling that swaps
    the least certain token of the best set for its runner-up; node mass = product of set masses, nodes
    below the floor gamma are dropped, the best `budget` nodes are verified;
  * one target forward verifies every node (concat layout, prefix KV cache); acceptance is state equality
    (exact: the output equals the threshold decoder's) or max-commit, which follows the verified path with
    the most commits under a relaxed rule (`rule`): "cut" (a higher threshold), "subset", or "product"
    (the drafted tokens are the top-1 tokens with geometric-mean confidence >= tau);
  * cross-block drafting: the highest-mass draft that completes the block also carries the next block, so
    a hit yields the next block's first step for free; draft refresh: the next block's cache-refresh
    forward verifies drafts too.

`generate_sets(adapter, prompt, SetsConfig(...), drafter=SetTreeDrafter(...))` returns the same summary
keys as `spot.decode.generate`; with `drafter=None` it is the threshold decoder without drafts.
"""

from __future__ import annotations

import heapq
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .decode import NEG, DecodeConfig, _assemble, _attn_bias, _block_causal_block, commit_view, select_commit
from .sampling import CoupledNoise

M_TOP, D_MAX = 32, 8                 # stale table width; staleness values 1..D_MAX
HF_REPO = "EliezerS/SPOT-dLLM"       # drafters not found locally are fetched from here (sets/<file>)


# ---------------------------------------------------------------------------------------------------------
# the set drafter
# ---------------------------------------------------------------------------------------------------------
def build_features(top_p: torch.Tensor, masked: torch.Tensor, n_committed: torch.Tensor) -> torch.Tensor:
    """top_p (B, L, M) stale probabilities, masked (B, L), n_committed (B,) -> (B, L, 2M + 8) features."""
    B, L, M = top_p.shape
    dev = top_p.device
    live = masked.float().unsqueeze(-1)
    p = top_p.float() * live
    eps = 1e-9
    top1 = p[..., 0]
    top2 = p[..., 1] if M > 1 else torch.zeros_like(top1)
    ent = -(p * (p + eps).log()).sum(-1)
    mass = p.sum(-1)
    pos = (torch.arange(L, device=dev, dtype=torch.float32) / max(L - 1, 1)).expand(B, L)
    n_masked = masked.float().sum(-1, keepdim=True).expand(B, L) / L
    n_comm = (n_committed.float() / L).unsqueeze(-1).expand(B, L)
    scal = torch.stack([top1, ent, top1 - top2, mass, pos, n_masked, n_comm, (~masked).float()], dim=-1)
    return torch.cat([p, (p + eps).log() / 10.0 * live, scal], dim=-1)


class SetDrafter(nn.Module):
    def __init__(self, m_top=M_TOP, d_model=256, layers=6, heads=8, n_cand=16, tok_rank=128, hidden_rank=128):
        super().__init__()
        self.m_top, self.n_cand, self.hidden_rank = m_top, n_cand, hidden_rank
        self.inp = nn.Sequential(nn.Linear(2 * m_top + 8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.stale_emb = nn.Embedding(D_MAX + 1, d_model)
        self.tok_proj = nn.Sequential(nn.Linear(tok_rank + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cand_proj = nn.Linear(tok_rank, d_model // 2)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.set_head = nn.Linear(d_model, 1)
        self.argmax_head = nn.Linear(d_model, 1)
        self.cand_feat = nn.Linear(2 + n_cand + 4, d_model // 2)
        self.tok_head = nn.Sequential(nn.Linear(d_model + d_model // 2, d_model // 2), nn.GELU(),
                                      nn.Linear(d_model // 2, 1))
        self.log_temperature = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.hid_proj = nn.Linear(hidden_rank, d_model)
        self.register_buffer("tok_coords", torch.zeros(0, tok_rank), persistent=False)  # PCA of the target's embeddings
        self.register_buffer("h_mean", torch.zeros(0), persistent=False)                # hidden-state basis
        self.register_buffer("h_basis", torch.zeros(0, hidden_rank), persistent=False)
        self.register_buffer("h_std", torch.zeros(hidden_rank), persistent=False)

    def table(self, ids):
        return self.tok_coords[ids.clamp(0, self.tok_coords.shape[0] - 1)]

    @torch.no_grad()
    def project(self, h):
        """(..., H) target hidden states -> (..., R) whitened PCA coordinates."""
        return ((h.to(self.h_mean.device, torch.float32) - self.h_mean) @ self.h_basis) / self.h_std

    @torch.no_grad()
    def predict(self, top_p, top_ids, states, mask_id: int, n_comm, stale, hid):
        """Per node: top_p / top_ids (B, L, M) stale table, states (B, L), n_comm (B,), stale (B,), hid (B, L, R).
        Returns P(commit) (B, L), fallback-position distribution (B, L), chosen token (B, L), token-head
        probabilities (B, L, C) and their candidate ids (B, L, C)."""
        masked = states == mask_id
        written = ~masked
        top_p, top_ids = top_p[..., :self.m_top], top_ids[..., :self.m_top]
        tok_ids = torch.where(masked, top_ids[..., 0], states)
        C = self.n_cand
        cand_ids, cand_p = top_ids[..., :C], top_p[..., :C].float()
        h = self.inp(build_features(top_p, masked, n_comm)) + self.stale_emb(stale.clamp(0, D_MAX))[:, None, :]
        h = h + self.tok_proj(torch.cat([self.table(tok_ids), written.float().unsqueeze(-1)], dim=-1))
        h = h + self.hid_proj(hid[..., :self.hidden_rank].float())
        h = self.norm(self.enc(h))
        s, a = self.set_head(h).squeeze(-1), self.argmax_head(h).squeeze(-1)
        # token head over the stale top-C candidates: probability, log-probability, rank, and whether the
        # candidate equals the token held by the left / right neighbour
        B, L, _ = cand_p.shape
        neg = torch.full_like(tok_ids[:, :1], -1)
        left, right = torch.cat([neg, tok_ids[:, :-1]], 1), torch.cat([tok_ids[:, 1:], neg], 1)
        fz = torch.zeros_like(written[:, :1])
        wl, wr = torch.cat([fz, written[:, :-1]], 1).float(), torch.cat([written[:, 1:], fz], 1).float()
        cf = torch.cat([cand_p.unsqueeze(-1), cand_p.clamp_min(1e-9).log().unsqueeze(-1) / 10.0,
                        torch.eye(C, device=cand_p.device).expand(B, L, C, C),
                        (cand_ids == left[..., None]).float().unsqueeze(-1), (cand_ids == right[..., None]).float().unsqueeze(-1),
                        wl[..., None, None].expand(B, L, C, 1), wr[..., None, None].expand(B, L, C, 1)], dim=-1)
        cf = self.cand_feat(cf) + self.cand_proj(self.table(cand_ids))
        t = self.tok_head(torch.cat([h.unsqueeze(2).expand(B, L, C, h.shape[-1]), cf], dim=-1)).squeeze(-1)
        p = torch.where(masked, torch.sigmoid(s / self.log_temperature.exp()), torch.zeros_like(s))
        a = torch.softmax(a.masked_fill(~masked, -1e9), dim=-1)
        choice = torch.gather(cand_ids, -1, t.argmax(-1, keepdim=True)).squeeze(-1)
        return p, a, choice, torch.softmax(t, -1), cand_ids


def load_set_drafter(path: str, device="cuda") -> SetDrafter:
    """A drafter file: {"cfg", "model", "token_coords", "hidden_pca": {mean, basis, std}}.  A path that does not
    exist is fetched from the Hugging Face repo (sets/<basename>)."""
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, "sets/" + os.path.basename(path))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SetDrafter(**ck["cfg"])
    m.load_state_dict(ck["model"])
    m.tok_coords = ck["token_coords"].float()
    m.h_mean, m.h_basis, m.h_std = (ck["hidden_pca"][k].float() for k in ("mean", "basis", "std"))
    return m.to(device).eval().requires_grad_(False)


class _Graphed:
    """CUDA-graph replay of SetDrafter.predict, one graph per input shape (same kernels as eager, one launch)."""

    def __init__(self, model):
        self.model, self.graphs, self.pool = model, {}, None

    def __call__(self, *args):
        tensors, mask_id = args[:3] + args[4:], args[3]
        key = tuple(tuple(x.shape) for x in tensors)
        if key not in self.graphs:
            static = [x.clone() for x in tensors]
            run = lambda: self.model.predict(*static[:3], mask_id, *static[3:])
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            self.pool = self.pool or torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = run()
            self.graphs[key] = (g, static, out)
        g, static, out = self.graphs[key]
        for s, x in zip(static, tensors):
            s.copy_(x)
        g.replay()
        return out


# ---------------------------------------------------------------------------------------------------------
# the step tree
# ---------------------------------------------------------------------------------------------------------
def _log(x: float) -> float:
    return math.log(x) if x > 0.0 else -745.0


def kbest_subsets(p, K: int, positions):
    """Top-K subsets of `positions` by independent-Bernoulli mass, exactly, in non-increasing mass order
    (best-first over flips of the MAP set, sorted by flip cost)."""
    n = len(p)
    if K <= 0:
        return []
    if n == 0:
        return [(frozenset(), 1.0)]
    p = [min(max(float(x), 0.0), 1.0) for x in p]
    hi, lo = [max(x, 1.0 - x) for x in p], [min(x, 1.0 - x) for x in p]
    cost = [_log(hi[i]) - _log(lo[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: cost[i])
    s0 = frozenset(positions[i] for i in range(n) if p[i] >= 0.5)
    out, heap = [], [(-sum(_log(h) for h in hi), (), -1)]
    max_out = min(K, 1 << n) if n < 60 else K
    while heap and len(out) < max_out:
        neg, flips, last = heapq.heappop(heap)
        if neg > 700.0:
            break
        out.append((s0.symmetric_difference([positions[order[j]] for j in flips]) if flips else s0, math.exp(-neg)))
        j = last + 1
        if j < n:
            heapq.heappush(heap, (neg + cost[order[j]], flips + (j,), j))
            if last >= 0:
                heapq.heappush(heap, (neg - cost[order[last]] + cost[order[j]], flips[:-1] + (j,), j))
    return out


def enumerate_sets(p, K: int, positions, fallback_pos: int):
    """Top-K non-empty commit sets: the empty set's mass moves to {fallback_pos}, the target's own rule."""
    merged: Dict[frozenset, float] = {}
    for s, m in kbest_subsets(p, K + 1, positions):
        s = s or frozenset([fallback_pos])
        merged[s] = merged.get(s, 0.0) + m
    return sorted(merged.items(), key=lambda kv: -kv[1])[:K]


@dataclass
class Draft:
    state: torch.Tensor           # hypothetical block state
    mass: float                   # probability that the path reaches it
    parent: int = -1              # index of the parent draft, -1 = the round's root


@dataclass
class _Node:
    key: tuple
    state: torch.Tensor
    mass: float
    level: int
    parent: int
    expanded: bool = False


class SetTreeDrafter:
    def __init__(self, model: SetDrafter, budget: int = 8, gamma: float = 0.02, children: int = 8,
                 cuda_graphs: bool = True):
        self.model, self.budget, self.gamma, self.K = model, int(budget), float(gamma), int(children)
        self.graphed = _Graphed(model) if (cuda_graphs and next(model.parameters()).is_cuda) else None
        self.prompt_len = 0

    def _predict(self, nodes, top_p, top_ids, hid, n_prompt):
        states_cpu = torch.stack([n.state for n in nodes])               # the tree lives on the CPU
        masked_cpu = states_cpu == self.mask_id
        dev = top_p.device
        B = len(nodes)
        args = (top_p.expand(B, -1, -1).contiguous(), top_ids.expand(B, -1, -1).contiguous(), states_cpu.to(dev),
                self.mask_id, ((~masked_cpu).sum(1) - n_prompt).to(dev),
                torch.tensor([min(n.level + 1, D_MAX) for n in nodes], device=dev), hid.expand(B, -1, -1).contiguous())
        out = self.graphed(*args) if self.graphed is not None else self.model.predict(*args)
        p, a, choice, tp, ci = (x.cpu() for x in out)
        return p, a, choice, tp.amax(-1), tp, ci, masked_cpu

    @torch.no_grad()
    def propose(self, p_prev: torch.Tensor, hid_prev: torch.Tensor, state: torch.Tensor, mask_id: int, b0: int):
        """Drafts for one round from the stale conditionals p_prev (L, V) and hidden rows hid_prev (L, H) of the
        deepest accepted copy of the previous round; parents precede children."""
        self.mask_id = mask_id
        L = int(state.shape[0])
        n_prompt = max(0, min(L, self.prompt_len - b0))
        tv, ti = p_prev.topk(M_TOP, dim=-1)                              # the stale table, shared by every node
        top_p, top_ids, hid = tv.float()[None], ti[None], self.model.project(hid_prev).float()[None]
        root = state.detach().to("cpu", copy=True)
        nodes = [_Node(tuple(root.tolist()), root, 1.0, 0, -1)]
        known = {nodes[0].key: 0}
        while True:
            sel = sorted(range(1, len(nodes)), key=lambda i: (-nodes[i].mass, nodes[i].level))[:self.budget]
            need = [i for i in ([0] if not nodes[0].expanded else []) + sel
                    if not nodes[i].expanded and nodes[i].level < 32 and bool((nodes[i].state == mask_id).any())]
            if not need:
                break
            p, a, choice, cp, tp, ci, masked = self._predict([nodes[i] for i in need], top_p, top_ids, hid, n_prompt)
            for row, i in enumerate(need):
                n = nodes[i]
                n.expanded = True
                pos = torch.nonzero(masked[row]).flatten().tolist()
                if not pos:
                    continue
                tokm = lambda s: math.prod(float(cp[row, j]) for j in s)
                sets = enumerate_sets(p[row, pos].tolist(), self.K, pos, pos[int(a[row, pos].argmax())])
                sets = sorted(((s, m * tokm(s)) for s, m in sets), key=lambda kv: -kv[1])
                children = [(s, m, None) for s, m in sets]
                if sets:
                    # the token hedge: the best set with its least certain token swapped for the runner-up
                    s0, m0 = sets[0]
                    j = min(s0, key=lambda j: float(cp[row, j]))
                    top2 = torch.topk(tp[row, j], 2)
                    alt = (int(ci[row, j, top2.indices[1]]), float(top2.values[1]))
                    t0 = tokm(s0)
                    others = t0 / max(float(cp[row, j]), 1e-9)
                    children = sorted(children + [(s0, (m0 / t0 if t0 > 0 else m0) * others * alt[1], (j, alt[0]))],
                                      key=lambda c: -c[1])
                for s, m, swap in children:
                    cm = n.mass * m
                    if self.gamma and cm < self.gamma:
                        continue
                    ns = n.state.clone()
                    idx = torch.tensor(sorted(s))
                    ns[idx] = choice[row][idx]
                    if swap is not None:
                        ns[swap[0]] = swap[1]
                    key = tuple(ns.tolist())
                    if key in known:
                        # the same state through another set sequence: keep the max-mass path
                        k = known[key]
                        if cm > nodes[k].mass:
                            ratio = cm / nodes[k].mass if nodes[k].mass > 0 else 0.0
                            nodes[k].mass, nodes[k].parent, nodes[k].level = cm, i, n.level + 1
                            if nodes[k].expanded and ratio:
                                stack = [k]
                                while stack:
                                    q = stack.pop()
                                    for c in range(1, len(nodes)):
                                        if nodes[c].parent == q:
                                            nodes[c].mass *= ratio
                                            nodes[c].level = nodes[q].level + 1
                                            stack.append(c)
                        continue
                    nodes.append(_Node(key, ns, cm, n.level + 1, i))
                    known[key] = len(nodes) - 1
        order = sorted(range(1, len(nodes)), key=lambda i: (-nodes[i].mass, nodes[i].level))[:self.budget]
        emitted, keep = {0: -1}, []
        for i in order:
            if nodes[i].parent in emitted and nodes[i].mass >= self.gamma:
                emitted[i] = len(keep)
                keep.append(i)
        states = torch.stack([nodes[i].state for i in keep]).to(state.device) if keep else None
        return [Draft(states[k], nodes[i].mass, emitted[nodes[i].parent]) for k, i in enumerate(keep)]


# ---------------------------------------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------------------------------------
MC_RULES = ("cut", "subset", "product")


@torch.no_grad()
def max_commit_path(root_state, probs_root, drafts: List[Draft], draft_probs, dcfg: DecodeConfig, mask_id: int,
                    noise=None, rule: str = "product") -> List[int]:
    """Draft indices of the admissible path through the verified tree with the most commits (ties: more,
    smaller steps).  Edge u -> v: state(v) extends state(u), the newly written set D is non-empty, and D passes
    the rule at u: inside the threshold rule's set with its tokens ("subset"), and also a threshold cut, i.e.
    every masked position left out is less confident than every written one ("cut"), or the top-1 tokens
    with geometric-mean confidence >= tau ("product").  The exact path is one admissible path."""
    S = torch.stack([root_state] + [d.state for d in drafts])                    # (n, L)
    P = torch.stack([probs_root] + list(draft_probs))
    M = S == mask_id
    _, TOK, CONF = commit_view(P, dcfg.temperature, noise)
    conf_m = torch.where(M, CONF, torch.full_like(CONF, -1.0))                  # the threshold rule at every node
    A = conf_m > dcfg.threshold
    fb = torch.zeros_like(M).scatter_(-1, conf_m.argmax(-1, keepdim=True), True)
    A = (A | (fb & ~A.any(-1, keepdim=True))) & M.any(-1, keepdim=True)
    T = torch.where(A, TOK, torch.full_like(TOK, -1))
    agree = ((S[None, :, :] == S[:, None, :]) | M[:, None, :]).all(-1)          # (u, v): v extends u
    D = M[:, None, :] & ~M[None, :, :]
    k = D.sum(-1)
    inside = ~(D & ~A[:, None, :]).any(-1)
    tok_ok = ~(D & (S[None, :, :] != T[:, None, :])).any(-1)
    if rule == "subset":
        ok = inside & tok_ok
    elif rule == "cut":
        conf_u = CONF[:, None, :]
        lo = torch.where(D, conf_u, torch.full_like(conf_u, float("inf"))).amin(-1)
        hi = torch.where(M[:, None, :] & ~D, conf_u, torch.full_like(conf_u, -float("inf"))).amax(-1)
        ok = inside & tok_ok & ((lo > hi) | (k == A.sum(-1, keepdim=True)))
    elif rule == "product":
        tok_top = ~(D & (S[None, :, :] != TOK[:, None, :])).any(-1)
        logc = torch.where(D, CONF[:, None, :].clamp_min(1e-30).log(), torch.zeros_like(D, dtype=CONF.dtype)).sum(-1)
        ok = tok_top & (inside | (logc >= k * math.log(dcfg.threshold)))
    else:
        raise ValueError(f"rule {rule!r}: one of {', '.join(MC_RULES)}")
    valid = agree & (k > 0) & ok
    valid[:, 0] = False
    valid.fill_diagonal_(False)
    valid &= A.any(-1)[:, None]
    n = S.shape[0]
    valid_c, k_c, free = valid.cpu().tolist(), k.cpu().tolist(), A.sum(-1).cpu().tolist()
    memo: Dict[int, tuple] = {}

    def value(u):                                     # (commits, steps, next node)
        if u not in memo:
            best = (free[u], 1 if free[u] else 0, None)
            for v in range(1, n):
                if valid_c[u][v]:
                    tv = value(v)
                    if (k_c[u][v] + tv[0], tv[1] + 1) > best[:2]:
                        best = (k_c[u][v] + tv[0], tv[1] + 1, v)
            memo[u] = best
        return memo[u]

    value(0)
    path, u = [], 0
    while memo[u][2] is not None:
        u = memo[u][2]
        path.append(u - 1)
    return path


# ---------------------------------------------------------------------------------------------------------
# the decoder
# ---------------------------------------------------------------------------------------------------------
@dataclass
class SetsConfig:
    gen_length: int = 256
    block_length: int = 32
    threshold: float = 0.9
    temperature: float = 0.0
    seed: int = 0
    acceptance: str = "product"       # "exact" | one of MC_RULES (max-commit)
    cross_block: bool = True
    draft_refresh: bool = True
    max_blocks: int = 8               # block copies per forward (the root included)

    def decode_config(self) -> DecodeConfig:
        return DecodeConfig(cache="on", unmasking="threshold", threshold=self.threshold, draft="none", suffix="skip",
                            gen_length=self.gen_length, block_length=self.block_length,
                            temperature=self.temperature, seed=self.seed)


def _bias_cross(past_len, main_len, piece_blocks, L, cur_lo_main, device, chunk_start):
    """Block-causal target: piece d holds piece_blocks[d] blocks from the current one (2 = a draft that completes
    it, then the next block masked); a piece sees the cached prefix, the true blocks before the current one and
    itself block-causally; nothing sees a piece."""
    K = past_len + main_len + sum(k * L for k in piece_blocks)
    bias = torch.zeros((1, 1, K, K), dtype=torch.float32, device=device)
    d0 = past_len + main_len
    a_lo = chunk_start - past_len
    bias[:, :, :d0, :d0] = _block_causal_block(a_lo, a_lo + d0, a_lo, a_lo + d0, L, device)
    bias[:, :, :d0, d0:] = NEG
    cur_lo, p_lo, r0 = past_len + cur_lo_main, chunk_start + cur_lo_main, d0
    for k in piece_blocks:
        r1 = r0 + k * L
        bias[:, :, r0:r1, cur_lo:] = NEG
        bias[:, :, r0:r1, r0:r1] = 0.0 if k == 1 else _block_causal_block(p_lo, p_lo + k * L, p_lo, p_lo + k * L, L, device)
        r0 = r1
    return bias


def _bias_cross_full(past_len, main_len, piece_blocks, L, shift, cur_lo_main, device):
    """Full-attention target (LLaDA-8B, Dream-7B): a two-block piece's next-block rows see the true chunk minus
    both blocks and the whole piece, the inputs of the next block's refresh forward."""
    lens = [k * L + shift for k in piece_blocks]
    K = past_len + main_len + sum(lens)
    bias = torch.zeros((1, 1, K, K), dtype=torch.float32, device=device)
    d0 = past_len + main_len
    bias[:, :, :d0, d0:] = NEG
    cur_lo, r0 = past_len + cur_lo_main, d0
    for k, ln in zip(piece_blocks, lens):
        r1, h = r0 + ln, r0 + shift + L
        bias[:, :, r0:r1, d0:] = NEG
        bias[:, :, r0:h, cur_lo:cur_lo + L] = NEG
        bias[:, :, r0:h, r0:h] = 0.0
        if k == 2:
            bias[:, :, h:r1, cur_lo:cur_lo + 2 * L] = NEG
            bias[:, :, h:r1, r0:r1] = 0.0
        if shift:
            bias[:, :, :r0, r0:r0 + shift] = NEG
            bias[:, :, r1:, r0:r0 + shift] = NEG
        r0 = r1
    return bias


def _drop_leaf(drafts: List[Draft], keep=()) -> Optional[Draft]:
    parents = {d.parent for d in drafts}
    leaves = [i for i, d in enumerate(drafts) if i not in parents and all(d is not x for x in keep)]
    return min(leaves, key=lambda i: drafts[i].mass) if leaves else None


def _remove(drafts: List[Draft], i: int):
    """Delete draft i (a leaf) and renumber the parent links."""
    del drafts[i]
    for d in drafts:
        d.parent -= int(d.parent > i)


@torch.no_grad()
def generate_sets(adapter, prompt_ids: torch.Tensor, cfg: SetsConfig, drafter: Optional[SetTreeDrafter] = None,
                  sample_id: str = "") -> dict:
    dcfg = cfg.decode_config()
    bc = bool(getattr(adapter, "block_causal", False))
    device, mask_id, shift = adapter.device, adapter.mask_id, int(getattr(adapter, "logit_shift", 0))
    assert cfg.acceptance == "exact" or cfg.acceptance in MC_RULES, cfg.acceptance
    assert shift == 0 or not bc
    L, P = cfg.block_length, int(prompt_ids.shape[1])
    if bc:                            # absolute block boundaries
        first_blk = P // L
        n_blocks = -(-(P + cfg.gen_length) // L) - first_blk
        total = (first_blk + n_blocks) * L
    else:                             # full attention: blocks start right after the prompt
        assert cfg.gen_length % L == 0
        first_blk, n_blocks, total = 0, cfg.gen_length // L, P + cfg.gen_length
    x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    noise_all = (CoupledNoise(cfg.seed, first_blk * L if bc else P, total, adapter.vocab_size, device)
                 if cfg.temperature > 0.0 else None)
    adapter.sample_temp = cfg.temperature if cfg.temperature > 0.0 else 1.0
    if drafter is not None:
        drafter.prompt_len = P
    fwd_kw = {"want_hidden": True} if drafter is not None else {}
    past, cache_len = None, 0
    n_inv = n_positions = n_drafts = n_accepted = 0
    carry = carry_hid = None          # cross-block: the next block's first-step conditionals (and hidden rows)
    rounds = []
    t0 = time.perf_counter()
    for b in range(n_blocks):
        b0 = (first_blk + b) * L if bc else P + b * L
        b1 = b0 + L
        noise_blk = noise_all.block(b0, b1) if noise_all is not None else None
        p_prev = h_prev = None
        if carry is not None:
            # the previous round's completing piece already computed this block's first step
            sel, toks, _ = select_commit(carry, x[0, b0:b1], dcfg, mask_id, noise_blk)
            x[0, b0:b1].index_copy_(0, sel, toks)
            if cfg.draft_refresh:
                p_prev, h_prev = carry, carry_hid
        carry = carry_hid = None
        while bool((x[0, b0:b1] == mask_id).any()):
            drafts: List[Draft] = []
            if drafter is not None and p_prev is not None:
                drafts = drafter.propose(p_prev, h_prev, x[0, b0:b1], mask_id, b0)
            want_prefix = max(0, b0 - shift)
            refresh = past is None or cache_len != want_prefix
            main_blocks = (b1 - (cache_len if past is not None else 0) + L - 1) // L if (drafts and refresh) else 1
            while main_blocks > 1 and len(drafts) > cfg.max_blocks - main_blocks:
                _remove(drafts, _drop_leaf(drafts))     # a drafted refresh round also carries the finished block(s)
            cross = None
            if cfg.cross_block and drafts and b1 + L <= total:
                # the highest-mass draft that completes the block also carries the next block; a lighter leaf is
                # dropped to make room
                comp = sorted((d for d in drafts if not bool((d.state == mask_id).any())), key=lambda d: -d.mass)
                if comp:
                    c = comp[0]
                    room = 1 + len(drafts) + 1 <= cfg.max_blocks - (main_blocks - 1)
                    if not room:
                        i = _drop_leaf(drafts, keep=(c,))
                        if i is not None and drafts[i].mass < c.mass:
                            _remove(drafts, i)
                            room = True
                    if room:
                        cross = next(i for i, d in enumerate(drafts) if d is c)
            past_len = cache_len if past is not None else 0
            chunk_start = past_len if refresh else want_prefix
            pieces = [{"state": d.state if i != cross else torch.cat([d.state, torch.full_like(d.state, mask_id)])}
                      for i, d in enumerate(drafts)]
            input_ids, position_ids, main_len = _assemble(x, chunk_start, b0, b1, pieces, shift, device,
                                                          chunk_end=b1 if bc else None, mask_id=mask_id)
            cur_lo = b0 - chunk_start
            blocks = [2 if i == cross else 1 for i in range(len(drafts))]
            if cross is None:
                bias = (_attn_bias(past_len, main_len, len(drafts), L, shift, cur_lo, device, block_causal=bc,
                                   chunk_start=chunk_start) if (drafts or bc) else None)
            elif bc:
                bias = _bias_cross(past_len, main_len, blocks, L, cur_lo, device, chunk_start)
            else:
                bias = _bias_cross_full(past_len, main_len, blocks, L, shift, cur_lo, device)
            out = adapter.forward(input_ids=input_ids, position_ids=position_ids, attn_bias=bias, past=past,
                                  use_cache=refresh, **fwd_kw)
            n_inv += 1
            n_positions += int(out.n_positions)
            n_drafts += len(drafts)
            if refresh:
                past, cache_len = adapter.trim_cache(out.past, want_prefix), want_prefix
            p_off = [main_len + i * (L + shift) + L * int(cross is not None and cross < i) for i in range(len(drafts))]
            probs = adapter.block_probs(out.logits, cur_lo, cur_lo + L)
            draft_probs = [adapter.block_probs_piece(out.logits, o, L, shift) for o in p_off]
            hid = draft_hid = None
            if drafter is not None:
                hid = out.hidden[0, cur_lo - shift:cur_lo + L - shift]
                draft_hid = [out.hidden[0, o:o + L] for o in p_off]

            # commit + verify
            root_state = x[0, b0:b1].clone()
            p_act, h_act, accepted, n_before = probs, hid, [], int((root_state == mask_id).sum())
            if cfg.acceptance != "exact" and drafts:
                path = max_commit_path(root_state, probs, drafts, draft_probs, dcfg, mask_id, noise_blk,
                                       cfg.acceptance)
                if path:
                    x[0, b0:b1] = drafts[path[-1]].state
                    p_act, h_act, accepted = draft_probs[path[-1]], draft_hid[path[-1]], path
                if bool((x[0, b0:b1] == mask_id).any()):
                    sel, toks, _ = select_commit(p_act, x[0, b0:b1], dcfg, mask_id, noise_blk)
                    x[0, b0:b1].index_copy_(0, sel, toks)
            else:
                unused = list(range(len(drafts)))
                while bool((x[0, b0:b1] == mask_id).any()):
                    sel, toks, _ = select_commit(p_act, x[0, b0:b1], dcfg, mask_id, noise_blk)
                    x[0, b0:b1].index_copy_(0, sel, toks)
                    m = next((i for i in unused if torch.equal(drafts[i].state, x[0, b0:b1])), None)
                    if m is None or not bool((x[0, b0:b1] == mask_id).any()):
                        break
                    unused.remove(m)
                    accepted.append(m)
                    p_act, h_act = draft_probs[m], draft_hid[m]
            if cross is not None and torch.equal(x[0, b0:b1], drafts[cross].state):
                o = p_off[cross] + L
                carry = adapter.block_probs_piece(out.logits, o, L, 0)
                carry_hid = out.hidden[0, o:o + L] if cfg.draft_refresh else None
            n_accepted += len(accepted)
            p_prev, h_prev = p_act, h_act
            rounds.append({"sample": sample_id, "block": b, "refresh": bool(refresh), "drafts_verified": len(drafts),
                           "accepted": len(accepted), "cross_piece": cross is not None,
                           "committed": n_before - int((x[0, b0:b1] == mask_id).sum())})
    wall = time.perf_counter() - t0
    gen = x[0, P:P + cfg.gen_length].tolist()
    eos = int(getattr(adapter, "eos_id", -1))
    return {"sample": sample_id, "tokens": gen, "text": adapter.decode(gen), "eos_pos": gen.index(eos) if eos in gen else -1,
            "nfe": n_inv, "verify_rows": n_inv, "positions_forwarded": n_positions, "draft_states_verified": n_drafts,
            "accepted": n_accepted, "wall_s": wall, "wall_s_net": wall, "rounds": rounds}
