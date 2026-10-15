"""Draft-graph construction.  One contract, several sources.

The verifier consumes a ranked list of candidate block states and nothing else, so a
drafter is free to build its graph however it likes as long as it hands back that list.
`prune_and_materialise` -- materialisation *and* Spiffy's geometric-mean scorer -- is shared
by every drafter, so a difference in acceptance is attributable to the proposal source and
not to a scoring change smuggled in beside it.

`D_free` (this file) is the training-free drafter: instead of resolving a fixed graph
calibrated offline against frequency, it enumerates, per round, the highest-probability
formulas of the target's *live* factorised joint.  Same information as Spiffy's incumbent
(both read `p_prev` and nothing else), same coordinates, same scorer, same zero cost.  The
only thing that differs is whether the node set was chosen once, offline, by how often a
formula fired across a calibration set, or per-round by what is actually probable now.

Worth knowing before reading the results: the calibrated SDAR graph
(`graphs/sdar_single_D10_mbpp.json`) has **max vocabulary rank j = 1**.  Every one of its
ten nodes commits argmax tokens and varies only in *which slots*.  Offline calibration
concluded that branching on the vocabulary never pays on average.  At temperature 1.0 the
drawn token is frequently not the argmax, so whether that conclusion holds *per round* is
exactly what an adaptive drafter can disagree about -- which is why `vocab_ranks` defaults
to 3 here and why the j=1 control exists (see `--draft-free-vocab-ranks 1`).
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import torch

from .graphs import DraftGraph, Node
from .sampling import perturbed_scores


def adaptive_graph(
    probs: torch.Tensor,
    block_state: torch.Tensor,
    mask_id: int,
    n_nodes: int,
    vocab_ranks: int = 3,
    max_depth: int = 6,
    skip: int = 1,
    beam: int = 8,
    order_scores: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    noise: Optional[torch.Tensor] = None) -> DraftGraph:
    """Top formulas of the live factorised joint, by level-wise beam search on GM.

    Coordinates are Spiffy's: (i, j) = (rank of a still-masked slot by confidence,
    rank of a token in the *sampler's* ordering at that slot).  Using the sampler's
    ordering rather than the probability ordering matters at T > 0 -- a draft has to
    predict what would actually be drawn, not what is most likely.

    **Ranked by geometric mean, not by joint probability.**  This is not a detail.  The
    shared scorer in `prune_and_materialise` is `exp(sum(log p) / len(pairs))` -- depth
    *normalised*, so a 6-pair formula of confident commits scores as well as a 1-pair one.
    Ranking the search by the joint product instead makes probability decay with depth, so
    the generator only ever emits shallow nodes and the deep ones the scorer would have
    preferred are never enumerated.  A first version of this function did exactly that and
    lost 33% of acceptance against the calibrated graph; the pre-registered matched-reach
    control caught it.  Generator and scorer must optimise the same
    objective or the comparison measures the mismatch, not the drafters.

    Because GM is not monotone under extension, best-first search does not apply: adding a
    pair can *raise* the score.  So this is a level-wise beam -- expand every surviving
    formula by one pair, keep the best `beam` by GM at each depth, emit them all.  Every
    kept node's prefix was in the beam that produced it and was emitted one level earlier,
    so the subset nesting Spiffy's child-scoring rule needs is present by construction.
    """
    masked = block_state == mask_id
    n_masked = int(masked.sum().item())
    if n_masked == 0 or n_nodes <= 0:
        return DraftGraph(nodes=[], meta={"source": "adaptive", "n_masked": 0})

    scores = perturbed_scores(probs, temperature, noise)
    tok = scores.argmax(dim=-1)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    # Which slot is "position rank i" -- by the target's own confidence, unless a trained
    # ranker supplies its own priority over the same masked slots.  Token probabilities below
    # still come from the target's conditional, so only the POSITION decision is delegated.
    rank_by = conf if order_scores is None else order_scores
    conf_m = torch.where(masked, rank_by, torch.full_like(rank_by, -1e30))
    order = torch.argsort(conf_m, descending=True, stable=True)[:n_masked]  # stable: match select_commit's argmax tie-break
    assert int(order[0]) == int(conf_m.argmax()), "drafter/verifier tie-break mismatch"

    j_max = max(1, min(vocab_ranks, probs.shape[-1]))
    _, ti = torch.topk(scores, j_max, dim=-1)
    # probability the *conditional* assigns to each candidate -- what the GM is over, so
    # it must be p, not the perturbed score.
    pv = probs.gather(-1, ti)
    P = pv[order].tolist()  # P[r][c]: rank-r slot, vocab-rank-c token

    depth_cap = max(1, min(max_depth, n_masked))
    frontier = [((), 0.0)]  # (pairs, sum of log p)
    out = []
    for _ in range(depth_cap):
        cand = []
        for pairs, lp in frontier:
            lo = pairs[-1][0] if pairs else 0  # 1-based rank last used == next 0-based index
            for r in range(lo, min(lo + 1 + skip, n_masked)):
                for c in range(j_max):
                    p = P[r][c]
                    nlp = lp + (math.log(p) if p > 0 else -80.0)
                    new = pairs + ((r + 1, c + 1),)
                    cand.append((math.exp(nlp / len(new)), new, nlp))
        if not cand:
            break
        cand.sort(key=lambda t: -t[0])
        frontier = [(new, nlp) for _, new, nlp in cand[:beam]]
        for gm, new, _ in cand[:beam]:
            out.append((frozenset(new), len(new)))
            if len(out) >= n_nodes:
                break
        if len(out) >= n_nodes:
            break

    return DraftGraph(
        nodes=[Node(f, lvl, 0) for f, lvl in out],
        meta={"source": "adaptive", "n_masked": n_masked, "vocab_ranks": j_max,
              "max_depth": depth_cap, "skip": skip, "beam": beam, "rank_by": "geometric_mean"})


class OracleDrafter:
    """Ablation: give the drafter the true next commits, in whole or in part.

    Decomposes drafting failure into its two components by handing over one at a time:

      pos   true SLOT, token still taken from the target's stale argmax
      tok   true TOKEN, slot still chosen by the target's stale confidence ordering
      both  both true -> every draft must be accepted; measures the speedup ceiling

    Sound only because drafting is lossless: the committed sequence is identical with and
    without drafts (verified, 50/50 byte-identical across four drafters), so a commit
    sequence recorded from a draft-free decode is exactly the one the drafted run will
    produce.  Without that guarantee the oracle would be predicting a different run.

    `both` is the number that decides whether ANY drafter is worth building: it is the
    speedup a perfect drafter would achieve at this budget, and nothing can exceed it.
    """

    def __init__(self, commits_by_sample: dict, mode: str = "both"):
        assert mode in ("pos", "tok", "both")
        self.mode = mode
        self.by_sample = commits_by_sample
        self.cur = None
        self.ptr = 0
        self.invocations = 0
        self.positions = 0

    def begin(self, sample_id):
        self.cur = self.by_sample.get(sample_id, [])
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def advance(self, n):
        self.ptr += n

    @torch.no_grad()
    def propose(self, probs, block_state, mask_id, budget, block, temperature, noise):
        """Build up to `budget` chained draft states from the recorded future commits."""
        from .decode import commit_view

        future = [c for c in self.cur[self.ptr:] if c[0] == block]
        if not future:
            return []
        _, argmax_tok, conf = commit_view(probs, temperature, noise)
        out, state = [], block_state.clone()
        for d in range(min(budget, len(future))):
            _, pos, tok = future[d]
            if self.mode == "pos":
                use_pos, use_tok = pos, int(argmax_tok[pos])
            elif self.mode == "tok":
                masked = state == mask_id
                cm = torch.where(masked, conf, torch.full_like(conf, -1.0))
                use_pos, use_tok = int(cm.argmax()), tok
            else:
                use_pos, use_tok = pos, tok
            if state[use_pos] != mask_id:
                break
            state = state.clone()
            state[use_pos] = use_tok
            out.append({"node": d, "level": d + 1, "formula": frozenset(),
                        "state": state.clone(), "chosen": {use_pos: use_tok},
                        "score": 1.0, "local": 1.0})
        return out


def cross_chain(p_win, state_win, mask_id, budget, L, temperature, noise_win,
                fan_margin=0.0, root_fan=1):
    """Draft chain over a one- or two-block window, with optional margin fan-out.

    Each expansion replicates select_commit on a hypothetical state: token = coupled-noise
    perturbed argmax from the stale window marginals, slot = highest-confidence masked slot
    of the FIRST incomplete block.  With `fan_margin` > 0, when the drafter's own top-2
    slot confidences are within the margin it BRANCHES over both slots -- replicas are
    spent on the drafter's own uncertainty.  Exact by construction either way: these are
    proposals; every commit is still the target's own sample at the correct state.  BFS
    order, at most `budget` nodes.

    Cross-block (window 2L): the b+1 half of an accepted replica is conditioned on its own
    completed copy of block k, which at acceptance time equals the true completed block.
    """
    from .sampling import perturbed_scores

    scores = perturbed_scores(p_win, temperature, noise_win)
    tok = scores.argmax(-1)
    conf = p_win.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    out = []
    frontier = [(state_win.clone(), 0)]
    nid = 0
    while frontier and len(out) < budget:
        st, depth = frontier.pop(0)
        masked = st == mask_id
        if not bool(masked.any()):
            continue
        lo, hi = (0, L) if bool(masked[:L].any()) else (L, st.shape[0])
        cm = torch.where(masked, conf, torch.full_like(conf, -1.0))
        cm[:lo] = -1.0
        cm[hi:] = -1.0
        avail = int((cm > -1.0).sum())
        top = torch.topk(cm, min(max(2, root_fan), max(avail, 1)))
        chosen = [int(top.indices[0])]
        # an earlier stage: root allocation -- branch over the top-k slots at the FIRST
        # expansion, unconditionally; each branch continues as a chain.  Motivated by
        # tree_anatomy: 58.6% of chain nodes are unreachable, so two depth-D/2 branches
        # can beat one depth-D chain at matched budget.
        if depth == 0 and root_fan > 1 and avail >= 2:
            k2 = min(root_fan, avail, budget - len(out))
            chosen = [int(i) for i in top.indices[:k2]] if k2 >= 2 else chosen
        elif (fan_margin > 0.0 and avail >= 2 and len(out) + 2 <= budget
                and float(top.values[0] - top.values[1]) < fan_margin):
            chosen.append(int(top.indices[1]))
        for s in chosen:
            ns = st.clone()
            ns[s] = int(tok[s])
            out.append({"node": nid, "level": depth + 1, "formula": frozenset(),
                        "state": ns.clone(), "chosen": {s: int(tok[s])},
                        "score": 1.0, "local": 1.0})
            nid += 1
            frontier.append((ns, depth + 1))
    return out


class SlotDrafter:
    """`D_slot` at inference: a learned *position* ranking, everything else unchanged.

    Same contract as `cross_chain` -- a chain of nested draft states, tokens taken from the
    target's own coupled-noise argmax at each chosen slot.  The only substitution is which
    slot each level drafts: the trained ranking instead of the stale confidence order.  So
    an acceptance difference is attributable to the ordering rule alone.

    Cost: one forward of a ~0.6M-parameter encoder over at most `block_length` rows.  No
    target-model call, no hidden states.
    """

    def __init__(self, ckpt_path: str, adapter=None, device=None):
        from .pickers import build_picker

        dev = device or (adapter.device if adapter is not None else "cpu")
        ck = torch.load(ckpt_path, map_location=dev)
        cfg = ck.get("cfg", ck)  # A2 checkpoints nest their config under "cfg"
        self.prefix_cond = bool(cfg.get("prefix_cond", False))
        self.context = bool(cfg.get("context", False))
        self.arch = cfg.get("arch", "enc")
        self.hid = int(cfg.get("hid", 0) or 0)  # nonzero => decode passes h_blk
        self.h_proj = ck.get("proj")
        if self.arch == "champ2":
            # /best trunk + token head and/or token-identity features.
            from .pickers import build_picker as _bp

            self.n_cand = int(cfg.get("n_cand", 4))
            self.d_emb = int(cfg.get("d_emb", 0))
            self.feat_src = cfg.get("feat_src", "emb" if self.d_emb else "none")
            self.model = _bp("champ2", m_top=cfg["m_top"], d_model=cfg["d_model"],
                             layers=cfg["layers"], heads=cfg.get("heads", 4),
                             n_cand=self.n_cand, d_emb=self.d_emb,
                             extra_dim=int(cfg.get("extra_dim", 0)),
                             tok_head=bool(cfg.get("tok_head", False)),
                             prefix_cond=True).to(dev).eval()
            if self.feat_src == "hid":
                # decode must plumb the pre-head hidden rows (h_prev) to chain()
                self.hid = int(cfg.get("extra_dim", 0))
            self.model.load_state_dict(ck["model"])
            for p_ in self.model.parameters():
                p_.requires_grad_(False)
            self.emb_W = None
            if self.d_emb or self.feat_src == "pathtok":
                assert adapter is not None, "emb picker needs the target's embeddings"
                self.emb_W = adapter.model.get_input_embeddings().weight
                want = self.d_emb or int(cfg.get("extra_dim", 0))
                assert self.emb_W.shape[1] == want, (self.emb_W.shape, want)
            self.m_top = int(cfg["m_top"])
            self.k_steps = 1
            self.tok_mod = 0
            self.trained_block = int(cfg.get("block_len", 0))
            self.val_joint = None
            self.invocations = 0
            self.positions = 0
            return
        if self.arch == "hs":
            # two-head proposer: raw hidden rows, learned projection inside the
            # model, token head over the stale top-C. No fixed random proj.
            from .hsproposer import HSProposer

            self.model = HSProposer(d_hid=cfg["d_hid"], m_top=cfg["m_top"],
                                    n_cand=cfg["n_cand"], d_model=cfg["d_model"],
                                    layers=cfg["layers"], heads=cfg.get("heads", 8),
                                    max_len=cfg["block_len"]).to(dev).eval()
            self.model.load_state_dict(ck["model"])
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.m_top = int(cfg["m_top"])
            self.k_steps = 1
            self.tok_mod = 0
            self.trained_block = int(cfg.get("block_len", 0))
            self.val_joint = None
            self.invocations = 0
            self.positions = 0
            return
        if self.hid:
            assert self.h_proj is not None and self.prefix_cond
            self.h_proj = self.h_proj.to(dev)
        # `arch` defaults to the original encoder, so every pre-checkpoint loads
        # unchanged; the zoo lives in spot/pickers.py.
        self.model = build_picker(cfg.get("arch", "enc"),
                                  m_top=cfg["m_top"], d_model=cfg["d_model"],
                                  layers=cfg["layers"], k_steps=cfg["k_steps"],
                                  vocab=cfg.get("vocab", 0) or 0,
                                  prefix_cond=self.prefix_cond,
                                  context=self.context, hid=self.hid).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.m_top = int(cfg["m_top"])
        self.k_steps = int(cfg["k_steps"])
        self.tok_mod = int(cfg.get("vocab", 0) or 0)
        self.trained_block = int(cfg.get("block_len", 0))
        self.val_joint = ck.get("val_joint")
        self.invocations = 0  # target-model calls: zero by construction
        self.positions = 0

    def reset(self):
        pass

    @torch.no_grad()
    def chain(self, p_blk, state_blk, mask_id, budget, temperature, noise_blk,
              round_in_block: int, fan_margin: float = 0.0, root_fan: int = 1,
              mode: str = "chain", gamma: float = 0.0, h_blk=None, tau: float = 1.0,
              tokgate: float = 0.0):
        from .slotpicker import slot_features

        masked = state_blk == mask_id
        idx = masked.nonzero(as_tuple=True)[0]
        if int(idx.numel()) == 0:
            return []
        L = int(state_blk.shape[0])
        if getattr(self, "arch", "enc") == "champ2":
            self._tau = float(tau) if tau and tau > 0 else 1.0
            if getattr(self, "feat_src", "none") == "hid" and h_blk is None:
                return []          # first drafted round of a block: no hidden rows yet
            samp_rank = None
            if temperature != 0.0:
                # T>0: written tokens must be the target's coupled draw, so candidate
                # IDENTITY/ORDER comes from the Gumbel-perturbed scores (best
                # convention); model features stay probability-derived, and the mass
                # factor stays the raw stale p of the written token.
                assert not getattr(self.model, "has_tok_head", False), \
                    "champ2 tok_head arms are T=0-only"
                samp_rank = perturbed_scores(p_blk, temperature, noise_blk)[idx]\
                    .topk(self.n_cand, dim=-1).indices
            return self._ddtree_champ2(p_blk, state_blk, idx, budget, round_in_block,
                                       gamma=gamma, tokgate=tokgate, h_blk=h_blk,
                                       pure_chain=(mode == "purechain"),
                                       samp_rank=samp_rank,
                                       fast=(mode in ("ddtree_bf", "pl")),
                                       one_fwd=(mode == "pl"))
        if getattr(self, "arch", "enc") == "hs":
            # proposer: chooses its own tokens from the stale top-C, so the
            # perturbed-argmax convention below does not apply. T=0 only for now --
            # at T>0 the coupled-noise verifier expects the sampler's ranking.
            assert temperature == 0.0, "hs proposer is T=0-only in this version"
            self._tau = float(tau) if tau and tau > 0 else 1.0
            return self._ddtree_hs(p_blk, state_blk, idx, budget, h_blk=h_blk,
                                   gamma=gamma, tokgate=tokgate)
        scores = perturbed_scores(p_blk, temperature, noise_blk)
        tok = scores.argmax(-1)  # the token the target itself would draw at each slot

        if self.prefix_cond:
            # the tree model: re-ask the picker at every node with that branch's own filled
            # slots as conditioning.  One 0.6M-param forward per node, no target call.
            extra = None
            if self.hid:
                assert h_blk is not None, "hid ckpt needs the verify forward's hidden states"
                extra = (h_blk.float() @ self.h_proj)[idx]
            self._tau = float(tau) if tau and tau > 0 else 1.0
            self._tokgate = float(tokgate or 0.0)
            if mode == "ddtree":
                return self._ddtree(p_blk, state_blk, idx, tok, budget, round_in_block,
                                    gamma=gamma, extra_rows=extra)
            if mode in ("ddtree_b", "pl", "purechain"):
                # second token candidate must come from the SAMPLER's ranking, not the raw
                # probability ranking: at T>0 the drafter writes the perturbed argmax, so
                # its runner-up is the second perturbed draw. Identical at T=0, where
                # perturbed_scores returns the probabilities unchanged.
                tok2 = scores.topk(2, dim=-1).indices[:, 1] if scores.shape[-1] > 1 else tok
                return self._ddtree_batched(p_blk, state_blk, idx, tok, budget,
                                            round_in_block, gamma=gamma, extra_rows=extra,
                                            one_fwd=(mode == "pl"),
                                            tokgate=getattr(self, '_tokgate', 0.0),
                                            tok2=tok2,
                                            pure_chain=(mode == "purechain"))
            return self._tree_cond(p_blk, state_blk, idx, tok, budget, round_in_block,
                                   extra_rows=extra)

        if self.context:
            # whole-block input: the written prefix is part of the question, so the row set
            # is every slot and the already-committed ones are simply ineligible.
            from .slotpicker import block_features

            filled = ~masked
            p_top_full = p_blk.topk(self.m_top, dim=-1).values.float()
            feats = block_features(p_top_full, filled, L, int(masked.sum()), int(filled.sum()))
            ids = torch.where(filled, state_blk, tok)
            top_tok = (ids % self.tok_mod) if self.tok_mod else torch.zeros_like(ids)
            logits = self.model(feats[None], top_tok[None])[0]  # (K, L)
            logits = logits.masked_fill(filled[None, :], -1e9)
            idx = torch.arange(L, device=logits.device)
            init_used = filled.clone()
        else:
            p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
            feats = slot_features(p_top, idx.float(), L, int(idx.numel()),
                                  L - int(idx.numel()))
            top_tok = tok[idx]
            if self.tok_mod:
                top_tok = top_tok % self.tok_mod
            else:
                top_tok = torch.zeros_like(top_tok)
            logits = self.model(feats[None], top_tok[None])[0]  # (K, S)
            init_used = torch.zeros(int(idx.numel()), dtype=torch.bool, device=logits.device)

        # BFS over (state, slots already taken, level).  With fan_margin = 0 and
        # root_fan = 1 this is exactly the old chain; branching spends replicas on the
        # model's OWN uncertainty instead of on depth past the predictable horizon.
        # /on mini the winning shape is ~2 wide x 4 deep, and a pure chain
        # compounds its per-level error instead of hedging it.
        probs = torch.softmax(logits, dim=-1)  # (K, S)
        if mode == "bestfirst":
            return self._bestfirst(probs, state_blk, idx, tok, budget, init_used)
        out, nid = [], 0
        seen = set()  # identical slot-sets are identical draft states; draft each once
        frontier = [(state_blk.clone(), init_used.clone(), 0)]
        kmax = min(self.k_steps, int(idx.numel()))
        while frontier and len(out) < budget:
            st, used, k = frontier.pop(0)
            if k >= kmax:
                continue
            row = probs[k].masked_fill(used, 0.0)
            n_avail = int((~used).sum())
            top = torch.topk(row, min(max(2, root_fan), max(n_avail, 1)))
            chosen = [int(top.indices[0])]
            if k == 0 and root_fan > 1 and n_avail >= 2:
                width = min(root_fan, n_avail, budget - len(out))
                if width >= 2:
                    chosen = [int(i) for i in top.indices[:width]]
            elif (fan_margin > 0.0 and n_avail >= 2 and len(out) + 2 <= budget
                    and float(top.values[0] - top.values[1]) < fan_margin):
                chosen.append(int(top.indices[1]))
            for j in chosen:
                sl = int(idx[j])
                ns = st.clone()
                ns[sl] = int(tok[sl])
                nu = used.clone()
                nu[j] = True
                sig = frozenset(nu.nonzero().flatten().tolist())
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"node": nid, "level": k + 1, "formula": frozenset(),
                            "state": ns.clone(), "chosen": {sl: int(tok[sl])},
                            "score": float(top.values[0]), "local": 1.0})
                nid += 1
                frontier.append((ns, nu, k + 1))
                if len(out) >= budget:
                    break
        return out

    @torch.no_grad()
    def _bestfirst(self, probs, state_blk, idx, tok, budget, init_used=None):
        """Best-first expansion by path probability -- lets the model pick depth vs breadth.

        /measured the trade directly: a second slot at a level is worth the rank-2
        mass (mini b32: 0.189, taking level coverage 0.616 -> 0.805), while one more level of
        chain is worth the path product (0.616^k).  Neither is uniformly better, so instead
        of fixing a shape we expand whichever partial path currently has the highest
        probability, and its top-2 children re-enter the queue with their own scores.  The
        resulting tree is ~2 wide near the root and thins with depth, which is the shape the
        winning graph has -- but derived per round from this state rather than calibrated.
        """
        import heapq

        S = int(idx.numel())
        kmax = min(self.k_steps, S)
        out, nid, ctr = [], 0, 0
        seen = set()  # same-state branches differ only in commit order; draft each once
        heap = []  # (-score, tie, state, used, level)
        heapq.heappush(heap, (-1.0, ctr, state_blk.clone(),
                              (init_used.clone() if init_used is not None
                               else torch.zeros(S, dtype=torch.bool, device=probs.device)), 0))
        while heap and len(out) < budget:
            negscore, _, st, used, k = heapq.heappop(heap)
            score = -negscore
            if k >= kmax:
                continue
            row = probs[k].masked_fill(used, 0.0)
            n_avail = int((~used).sum())
            if n_avail == 0 or k >= probs.shape[0]:
                continue
            top = torch.topk(row, min(2, n_avail))
            for r in range(top.indices.numel()):
                j = int(top.indices[r])
                child = score * float(top.values[r])
                sl = int(idx[j])
                ns = st.clone(); ns[sl] = int(tok[sl])
                nu = used.clone(); nu[j] = True
                sig = frozenset(nu.nonzero().flatten().tolist())
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"node": nid, "level": k + 1, "formula": frozenset(),
                            "state": ns.clone(), "chosen": {sl: int(tok[sl])},
                            "score": child, "local": float(top.values[r])})
                nid += 1
                ctr += 1
                heapq.heappush(heap, (-child, ctr, ns, nu, k + 1))
                if len(out) >= budget:
                    break
        return out

    @torch.no_grad()
    def order_scores(self, p_blk, state_blk, mask_id, round_in_block: int):
        """Per-slot priority for `adaptive_graph`'s position ranking.

        Lets the learned ranking drive the *free* drafter's tree builder: same beam, same
        depth, same scorer, but positions ordered by the model instead of by stale
        confidence. This is the "graph shape + trained ranking" combination showed
        the shape is what wins on mini and showed the ranking is what we do better.
        """
        from .slotpicker import slot_features

        masked = state_blk == mask_id
        idx = masked.nonzero(as_tuple=True)[0]
        out = torch.full_like(p_blk[:, 0], -1e30)
        if int(idx.numel()) == 0:
            return out
        L = int(state_blk.shape[0])
        p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
        feats = slot_features(p_top, idx.float(), L, int(idx.numel()),
                              L - int(idx.numel()))
        tk = torch.zeros(int(idx.numel()), dtype=torch.long, device=p_blk.device)
        logits = self.model(feats[None], tk[None])[0]  # (K, S)
        out[idx] = torch.softmax(logits[0], dim=-1).to(out.dtype)  # step-1 head = "next slot"
        return out

    @torch.no_grad()
    def _tree_cond(self, p_blk, state_blk, idx, tok, budget, round_in_block,
                   extra_rows=None):
        """Best-first tree with a prefix-conditional picker.

        Each frontier node is re-scored from its OWN hypothetical state, so branches that
        filled different slots diverge in their predictions -- the thing the k-head chain
        model could not express.  Expansion is by path probability, so the shape (how wide
        near the root, how deep) is decided by the model's own confidence.
        """
        import heapq
        from .slotpicker import path_features, slot_features

        S = int(idx.numel())
        L = int(state_blk.shape[0])
        p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
        base = slot_features(p_top, idx.float(), L, S, L - S)
        if extra_rows is not None:
            base = torch.cat([base, extra_rows.to(base.dtype)], -1)
        zeros_tok = torch.zeros(1, S, dtype=torch.long, device=base.device)

        def score_node(taken, depth):
            f = torch.cat([base, path_features(S, taken, depth, L, device=base.device)], -1)
            lg = self.model(f[None], zeros_tok)[0, 0]
            return torch.softmax(lg.masked_fill(taken, -1e9) / getattr(self, '_tau', 1.0), dim=-1)

        out, nid, ctr = [], 0, 0
        # a draft is accepted by STATE equality, and two branches that fill the same slots
        # in different orders produce the same state -- drafting both wastes a replica.
        seen = set()
        heap = [(-1.0, 0, state_blk.clone(), torch.zeros(S, dtype=torch.bool,
                                                         device=base.device), 0)]
        while heap and len(out) < budget:
            neg, _, st, taken, d = heapq.heappop(heap)
            if d >= S:
                continue
            row = score_node(taken, d)
            top = torch.topk(row, min(2, int((~taken).sum())))
            for r in range(top.indices.numel()):
                j = int(top.indices[r]); sl = int(idx[j])
                child = (-neg) * float(top.values[r])
                ns = st.clone(); ns[sl] = int(tok[sl])
                nt = taken.clone(); nt[j] = True
                sig = frozenset(nt.nonzero().flatten().tolist())
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"node": nid, "level": d + 1, "formula": frozenset(),
                            "state": ns.clone(), "chosen": {sl: int(tok[sl])},
                            "score": child, "local": float(top.values[r])})
                nid += 1; ctr += 1
                heapq.heappush(heap, (-child, ctr, ns, nt, d + 1))
                if len(out) >= budget:
                    break
        return out

    @torch.no_grad()
    def rs_chain(self, p_blk, state_blk, mask_id, budget, round_in_block, generator):
        """Depth-`budget` chain SAMPLED from the picker's own proposal law, for verify=rs.

        Link j proposes slot s ~ softmax(picker logits over open slots) and token
        t ~ p_blk[s, :] (the stale marginal), so the link's proposal law is the exactly
        computable product  q(s, t) = q_slot(s) * p_blk[s, t].  Both factors are recorded:
        `q_pair` drives the accept ratio and `q_slot_full` (length-L, zero on filled
        slots) lets decode build the exact residual law  q_law = q_slot[:, None] * p_blk
        -- the maximal-coupling residual is only lossless if it subtracts the TRUE
        proposal law, and the picker's law is not the stale commit law that free-RS uses.

        This is "rejection sampling on positions": the slot factor is learned, the token
        factor is the target's own stale marginal.  Requires the prefix-conditional
        picker (each link conditions on the chain's own filled slots).
        """
        assert self.prefix_cond, "rs_chain needs the prefix-conditional picker"
        from .slotpicker import path_features, slot_features

        masked = state_blk == mask_id
        idx = masked.nonzero(as_tuple=True)[0]
        S = int(idx.numel())
        if S == 0:
            return []
        L = int(state_blk.shape[0])
        p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
        base = slot_features(p_top, idx.float(), L, S, L - S)
        zeros_tok = torch.zeros(1, S, dtype=torch.long, device=base.device)

        out = []
        st = state_blk.clone()
        taken = torch.zeros(S, dtype=torch.bool, device=base.device)
        for d in range(min(budget, S)):
            f = torch.cat([base, path_features(S, taken, d, L, device=base.device)], -1)
            lg = self.model(f[None], zeros_tok)[0, 0]
            q_slot = torch.softmax(lg.masked_fill(taken, -1e9) / getattr(self, '_tau', 1.0), dim=-1)  # (S)
            j = int(torch.multinomial(q_slot, 1, generator=generator))
            sl = int(idx[j])
            row = p_blk[sl].clamp_min(0)
            row = row / row.sum().clamp_min(1e-30)
            t = int(torch.multinomial(row, 1, generator=generator))
            q_pair = float(q_slot[j]) * float(row[t])
            q_full = torch.zeros(L, dtype=p_blk.dtype, device=p_blk.device)
            q_full[idx[~taken]] = q_slot[~taken].to(p_blk.dtype)
            st = st.clone(); st[sl] = t
            taken = taken.clone(); taken[j] = True
            out.append({"node": d, "level": d + 1, "formula": frozenset(),
                        "state": st.clone(), "chosen": {sl: t}, "score": 1.0,
                        "local": 1.0, "pair": (sl, t), "q_pair": q_pair,
                        "q_slot_full": q_full, "src_probs": p_blk})
        return out

    @torch.no_grad()
    def _ddtree(self, p_blk, state_blk, idx, tok, budget, round_in_block,
                gamma: float = 0.0, extra_rows=None):
        """DDTree's enumeration (arXiv 2604.12989), in slot space.

        DDTree pops the highest-mass prefix from a heap and pushes exactly two successors:
        the **next sibling** at the current depth (same prefix, next-ranked choice) and the
        **first child** (extend the prefix with its top choice). Under a factorised drafter
        that enumerates prefixes in non-increasing prefix-mass order, so the first B pops are
        the B highest-mass prefixes -- which is precisely the set maximising their surrogate,
        expected acceptance length = sum of prefix masses over the tree.

        Our `_tree_cond` approximated this by pushing the top-2 children of every pop, which
        can spend budget on a second child of a weak node while a strong node's third sibling
        goes undrafted. This makes the ordering exact.

        The one adaptation their setting does not need: with a prefix-conditional picker the
        per-level distribution depends on the branch, so a node's children need that node's
        own forward. Sibling scores come from the parent's cached distribution, so each pop
        costs at most one 0.6M-param forward.
        """
        import heapq
        from .slotpicker import path_features, slot_features

        S = int(idx.numel())
        L = int(state_blk.shape[0])
        p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
        base = slot_features(p_top, idx.float(), L, S, L - S)
        if extra_rows is not None:
            base = torch.cat([base, extra_rows.to(base.dtype)], -1)
        zeros_tok = torch.zeros(1, S, dtype=torch.long, device=base.device)

        def dist(taken, depth):
            f = torch.cat([base, path_features(S, taken, depth, L, device=base.device)], -1)
            lg = self.model(f[None], zeros_tok)[0, 0]
            return torch.softmax(lg.masked_fill(taken, -1e9) / getattr(self, '_tau', 1.0), dim=-1)

        root = dist(torch.zeros(S, dtype=torch.bool, device=base.device), 0)
        order0 = torch.argsort(root, descending=True, stable=True)

        out, seen, ctr = [], set(), 0
        # frontier entry: (-prefix_mass, tie, state, taken, depth, parent_dist, parent_order,
        #                  rank_at_depth, parent_mass). Carrying the parent's distribution
        # makes a sibling push free -- only a child push needs a new forward.
        heap = [(-float(root[order0[0]]), 0, state_blk.clone(),
                 torch.zeros(S, dtype=torch.bool, device=base.device),
                 0, root, order0, 0, 1.0)]
        while heap and len(out) < budget:
            neg, _, st, taken, d, p_dist, p_order, rank, p_mass = heapq.heappop(heap)
            if gamma and -neg < gamma:
                break  # adaptive budget: mass-ordered pops => everything left is below gamma
            mass = -neg
            if rank >= int(p_order.numel()):
                continue
            j = int(p_order[rank])
            if bool(taken[j]):
                continue
            sl = int(idx[j])
            ns = st.clone(); ns[sl] = int(tok[sl])
            nt = taken.clone(); nt[j] = True
            sig = frozenset(nt.nonzero().flatten().tolist())
            if sig not in seen:
                seen.add(sig)
                out.append({"node": len(out), "level": d + 1, "formula": frozenset(),
                            "state": ns.clone(), "chosen": {sl: int(tok[sl])},
                            "score": mass, "local": mass / max(p_mass, 1e-12)})
            # successor 1: next sibling at this depth, scored from the parent's own dist
            if rank + 1 < int(p_order.numel()):
                nxt = int(p_order[rank + 1])
                if not bool(taken[nxt]):
                    ctr += 1
                    heapq.heappush(heap, (-(p_mass * float(p_dist[nxt])), ctr, st, taken,
                                          d, p_dist, p_order, rank + 1, p_mass))
            # successor 2: first child of the node just emitted (one forward)
            if d + 1 < S and len(out) < budget:
                ch = dist(nt, d + 1)
                ch_order = torch.argsort(ch, descending=True, stable=True)
                ctr += 1
                heapq.heappush(heap, (-(mass * float(ch[ch_order[0]])), ctr, ns, nt,
                                      d + 1, ch, ch_order, 0, mass))
        return out

    @torch.no_grad()
    def _ddtree_batched(self, p_blk, state_blk, idx, tok, budget, round_in_block,
                        gamma: float = 0.0, extra_rows=None, one_fwd: bool = False,
                        tokgate: float = 0.0, tok2=None, pure_chain: bool = False):
        """DDTree's node selection with level-batched picker calls.

        Selects the same set as `_ddtree` -- the top-`budget` prefixes by mass -- but
        instead of one picker forward per heap pop it batches every distribution a
        selection round needs into ONE call. Exactness rests on two facts: a child's
        mass never exceeds its parent's (so the top-B set is parent-closed), and only
        selected nodes' children can enter the top-B set (a non-selected prefix has mass
        below the cut, hence so do all its descendants). Iterating [select top-B from
        known masses -> batch-expand the selected-but-unexpanded] therefore reaches the
        sequential fixpoint in at most tree-depth iterations: ~3-6 batched forwards at
        D<=20 versus up to D sequential ones, all Python heap bookkeeping on CPU floats.

        tokgate > 0 turns on GATED TOKEN BRANCHING: at a slot whose stale top-1
        probability is below the gate, the search also proposes the slot's SECOND token as
        a sibling state. Measured motivation(b/c): the committed token is the stale
        argmax 99% of the time above confidence 0.8 but only 44-62% below 0.4, and 26-37%
        of chains die on exactly that -- a right-slot/wrong-token link. Blanket token
        branching lost (-1.36%, ddtree2d) because it spent replicas uniformly, i.e. mostly
        where the token was already certain. With the gate on, prefix mass includes the
        token factor at every node, so the heap orders slot-and-token hypotheses on one
        scale; tokgate=0 keeps the token factor but never emits a second token, which is
        the control that separates "mass reshaping" from "branching".

        one_fwd=True is the Plackett-Luce probe: NO forwards past the root -- every
        branch's distribution is the root distribution renormalised over the slots that
        branch has not taken. The NFE gap to the full model measures exactly what
        per-node prefix conditioning is worth; the TPS gap bounds the drafter tax.
        """
        from .slotpicker import slot_features, N_PATH

        S = int(idx.numel())
        L = int(state_blk.shape[0])
        p_top = p_blk[idx].topk(self.m_top, dim=-1).values.float()
        base = slot_features(p_top, idx.float(), L, S, L - S)
        if extra_rows is not None:
            base = torch.cat([base, extra_rows.to(base.dtype)], -1)
        tau = getattr(self, '_tau', 1.0)
        tok_cpu = tok[idx].tolist()
        idx_cpu = idx.tolist()
        # token candidates per open slot: rank-1 is what the drafter has always written
        # (kept bit-identical via `tok`), rank-2 is the hedge the gate may spend on.
        two = p_blk[idx].topk(2, dim=-1)
        ptok = two.values.tolist()          # [[p1, p2], ...] -- mass and gate read p_blk
        # candidate ids come from the sampler's ranking when it was supplied (T>0);
        # at T=0 the two rankings coincide, so this is a no-op for every T=0 result.
        tok2_cpu = (tok2[idx] if tok2 is not None else two.indices[:, 1]).tolist()
        use_tok = tokgate > 0.0

        def expand(sigs):
            # one batched picker call -> per-branch distributions as CPU lists
            n = len(sigs)
            taken = torch.zeros(n, S, dtype=torch.bool, device=base.device)
            depth = torch.zeros(n, device=base.device, dtype=base.dtype)
            for i, sg in enumerate(sigs):
                if sg:
                    taken[i, [j for j, _r in sg]] = True
                depth[i] = len(sg)
            pf = torch.zeros(n, S, N_PATH, device=base.device, dtype=base.dtype)
            pf[:, :, 0] = taken.to(base.dtype)
            pf[:, :, 1] = (depth / max(L, 1))[:, None]
            f = torch.cat([base[None].expand(n, -1, -1), pf], -1)
            zt = torch.zeros(n, S, dtype=torch.long, device=base.device)
            lg = self.model(f, zt)[:, 0]  # (n, S)
            return torch.softmax(lg.masked_fill(taken, -1e9) / tau, dim=-1).tolist()

        root = expand([frozenset()])[0]
        if pure_chain:
            # greedy depth-D CHAIN -- no branching, no token factor. Isolates
            # sequential prediction quality from every tree/search interaction.
            out, sig, state = [], frozenset(), state_blk
            dist = root
            for depth in range(min(budget, S)):
                taken = {j for j, _r in sig}
                j = max((jj for jj in range(S) if jj not in taken),
                        key=lambda jj: dist[jj])
                sl = idx_cpu[j]
                state = state.clone(); state[sl] = tok_cpu[j]
                out.append({"node": depth, "level": depth + 1,
                            "formula": frozenset(), "state": state.clone(),
                            "chosen": {sl: tok_cpu[j]}, "score": dist[j],
                            "local": dist[j]})
                sig = sig | {(j, 0)}
                if depth + 1 < min(budget, S):
                    dist = expand([sig])[0]
            return out
        if one_fwd:
            def renorm(sig):
                st = {j for j, _r in sig}
                den = sum(root[j] for j in range(S) if j not in st)
                return [0.0 if j in st else root[j] / max(den, 1e-12) for j in range(S)]

        # sig -> (mass, dist-or-None); root has mass 1 and its true distribution
        nodes = {frozenset(): (1.0, root)}
        sel = []
        while True:
            cand = {}  # child sig -> (mass, parent_sig, j, r); keep the max-mass path
            for sg, (m, dst) in nodes.items():
                if dst is None:
                    continue
                slots_taken = {j for j, _r in sg}
                for j in range(S):
                    if j in slots_taken:
                        continue
                    base_m = m * dst[j]
                    ranks = (0, 1) if (use_tok and ptok[j][0] < tokgate) else (0,)
                    for r in ranks:
                        cm = base_m * (ptok[j][r] if use_tok else 1.0)
                        if gamma and cm < gamma:
                            continue
                        cs = sg | {(j, r)}
                        if cs not in cand or cm > cand[cs][0]:
                            cand[cs] = (cm, sg, j, r)
            # depth-ascending tiebreak: a p=1.0 link gives child mass == parent mass,
            # and the state build below needs parents ordered before their children
            sel = sorted(cand.items(), key=lambda kv: (-kv[1][0], len(kv[0])))[:budget]
            need = [cs for cs, (cm, ps, j, r) in sel
                    if cs not in nodes and len(cs) < S]
            for cs, (cm, ps, j, r) in sel:
                if cs not in nodes:
                    nodes[cs] = (cm, None)
                else:
                    nodes[cs] = (cm, nodes[cs][1])
            if not need:
                break
            dsts = [renorm(cs) for cs in need] if one_fwd else expand(need)
            for cs, dst in zip(need, dsts):
                nodes[cs] = (nodes[cs][0], dst)

        out, states = [], {frozenset(): state_blk}
        for cs, (cm, ps, j, r) in sel:  # mass-descending; parents precede children
            sl = idx_cpu[j]
            ns = states[ps].clone()
            ns[sl] = tok_cpu[j] if r == 0 else tok2_cpu[j]
            states[cs] = ns
            pm = nodes[ps][0]
            out.append({"node": len(out), "level": len(cs), "formula": frozenset(),
                        "state": ns, "chosen": {sl: tok_cpu[j]},
                        "score": cm, "local": cm / max(pm, 1e-12)})
        return out

    @torch.no_grad()
    def _ddtree_champ2(self, p_blk, state_blk, idx, budget, round_in_block,
                       gamma: float = 0.0, tokgate: float = 0.0, h_blk=None,
                       pure_chain: bool = False, samp_rank=None,
                       fast: bool = False, one_fwd: bool = False):
        """/search: `_ddtree_batched`'s selection on the champ2 model.

        Feature layout per row: [slot_features | path (taken, depth) | raw emb block],
        matching Champ2.f0. The emb block is branch-INDEPENDENT (stale top-C identity),
        so it is computed once per round; only path features vary per node.

        Token semantics per arm, so each A/B changes one thing:
          tok_head=False: EXACTLY `_ddtree_batched`'s best rule -- stale
            argmax + raw stale-p factor when tokgate>0, second STALE candidate below
            the gate. The arm isolates the embedding features.
          fast=True (mode ddtree_bf): the SAME selection computed with a
            vectorised candidate table + lexsort instead of the per-(node, slot, cand)
            Python loop and a full sort per level -- identical node set, ~4x less
            drafting time (the loop, not the picker forwards, dominated at D20).
          one_fwd=True (mode pl): Plackett-Luce probe -- no picker forwards past the
            root; child distributions are the root renormalised over untaken slots.
          tok_head=True: the MODEL's token distribution supplies both the
            factor and the candidate ranking (tokgate 0 = positions-only control,
            0.001 = calibrated g000 analog, 0.6 = gated 2nd candidate), as in
            `_ddtree_hs`. The arm isolates the learned token head.
        """
        from .slotpicker import N_PATH, slot_features

        S = int(idx.numel())
        L = int(state_blk.shape[0])
        C = int(self.n_cand)
        dev = p_blk.device
        top = p_blk[idx].topk(self.m_top, dim=-1)
        p_top = top.values.float()
        cand_ids_cpu = top.indices[:, :C].tolist()
        cand_p = p_top[:, :C]                       # raw stale probs, best-style
        if samp_rank is not None:
            # T>0 (see chain()): sampler-ordered candidates; factor = raw p of each
            cand_ids_cpu = samp_rank.tolist()
            cand_p = p_blk[idx].gather(-1, samp_rank).float()
        cand_p_cpu = cand_p.tolist()
        idx_cpu = idx.tolist()
        base = slot_features(p_top, idx.float(), L, S, L - S)
        emb = None
        pathtok = getattr(self, "feat_src", "none") == "pathtok"
        if pathtok:
            pass                                  # branch-dependent; built in expand()
        elif getattr(self, "feat_src", "none") == "hid":
            emb = h_blk[idx].to(base.dtype)      # RAW pre-head rows, uncompressed
        elif self.d_emb:
            from .pickers import emb_features
            emb = emb_features(self.emb_W, top.indices[:, :C], cand_p).to(base.dtype)
        tau = getattr(self, "_tau", 1.0)
        has_tok = bool(getattr(self.model, "has_tok_head", False))

        def expand(sigs):
            n = len(sigs)
            taken_h = torch.zeros(n, S, dtype=torch.bool)
            depth_h = torch.zeros(n)
            for i, sg in enumerate(sigs):
                if sg:
                    taken_h[i, [j for j, _c in sg]] = True
                depth_h[i] = len(sg)
            taken = taken_h.to(dev, non_blocking=True)
            depth = depth_h.to(dev, dtype=base.dtype, non_blocking=True)
            pf = torch.zeros(n, S, N_PATH, device=dev, dtype=base.dtype)
            pf[:, :, 0] = taken.to(base.dtype)
            pf[:, :, 1] = (depth / max(L, 1))[:, None]
            f = torch.cat([base[None].expand(n, -1, -1), pf], -1)
            if pathtok:
                # each branch row carries the embedding of the token IT wrote at
                # that slot -- prefix mass becomes sequence-coherence-aware, which is
                # what prunes the tree.
                pe = torch.zeros(n, S, self.emb_W.shape[1],
                                 device=dev, dtype=base.dtype)
                for i, sg in enumerate(sigs):
                    for j, c in sg:
                        pe[i, j] = self.emb_W[cand_ids_cpu[j][c]].to(base.dtype)
                f = torch.cat([f, pe], -1)
            elif emb is not None:
                f = torch.cat([f, emb[None].expand(n, -1, -1)], -1)
            if has_tok:
                pos_lg, tok_lg = self.model.forward_full(
                    f, cand_p[None].expand(n, -1, -1))
                pos_lg = pos_lg.masked_fill(taken, -1e9)
                return list(zip(torch.softmax(pos_lg / tau, -1).tolist(),
                                torch.softmax(tok_lg, -1).tolist()))
            lg = self.model(f)[:, 0].masked_fill(taken, -1e9)
            return [(d, None) for d in torch.softmax(lg / tau, -1).tolist()]

        root = expand([frozenset()])[0]
        if pure_chain:
            out, sig, state = [], frozenset(), state_blk
            sp, _tp = root
            for depth in range(min(budget, S)):
                taken = {j for j, _c in sig}
                j = max((jj for jj in range(S) if jj not in taken),
                        key=lambda jj: sp[jj])
                sl = idx_cpu[j]
                tk = cand_ids_cpu[j][0]          # stale argmax, best-identical
                state = state.clone(); state[sl] = tk
                out.append({"node": depth, "level": depth + 1,
                            "formula": frozenset(), "state": state.clone(),
                            "chosen": {sl: tk}, "score": sp[j], "local": sp[j]})
                sig = sig | {(j, 0)}
                if depth + 1 < min(budget, S):
                    sp, _tp = expand([sig])[0]
            return out
        nodes = {frozenset(): (1.0, root)}
        sel = []
        use_tok = tokgate > 0.0
        if one_fwd:
            def renorm(sig):
                st_ = {j for j, _c in sig}
                den = sum(root[0][j] for j in range(S) if j not in st_)
                return ([0.0 if j in st_ else root[0][j] / max(den, 1e-12) for j in range(S)],
                        None)
        if fast and not has_tok:
            import numpy as np
            # per-round token factor table: F[j, c] and validity, identical to the loop below
            F = np.zeros((S, 2)); V_ = np.zeros((S, 2), dtype=bool)
            for j in range(S):
                if use_tok:
                    two = cand_p_cpu[j]
                    ranks = (0, 1) if two[0] < tokgate else (0,)
                    for c in ranks:
                        F[j, c] = two[c]; V_[j, c] = True
                else:
                    F[j, 0] = 1.0; V_[j, 0] = True
            while True:
                act = [(sg, m, dst[0]) for sg, (m, dst) in nodes.items() if dst is not None]
                n = len(act)
                SP = np.asarray([sp for _, _, sp in act], dtype=np.float64)          # (n, S)
                Mv = np.asarray([m for _, m, _ in act])[:, None, None] * SP[:, :, None] * F[None]
                tk = np.zeros((n, S), dtype=bool)
                for i, (sg, _, _) in enumerate(act):
                    for j, _c in sg:
                        tk[i, j] = True
                ok = V_[None] & ~tk[:, :, None]
                if gamma:
                    ok &= Mv >= gamma
                flat = np.flatnonzero(ok)
                vals = Mv.ravel()[flat]
                lens = np.asarray([len(sg) + 1 for sg, _, _ in act])[flat // (2 * S)]
                order = np.lexsort((flat, lens, -vals))   # (-mass, len, loop order): the loop's key
                sel, seen = [], set()
                for o in order:
                    f_ = int(flat[o]); i, r = divmod(f_, 2 * S); j, c = divmod(r, 2)
                    cs = act[i][0] | {(j, c)}
                    if cs in seen:
                        continue                       # first hit is the max-mass path, as the dict kept
                    seen.add(cs); sel.append((cs, (float(vals[o]), act[i][0], j, c)))
                    if len(sel) == budget:
                        break
                need = [cs for cs, (cm, ps, j, c) in sel if cs not in nodes and len(cs) < S]
                for cs, (cm, ps, j, c) in sel:
                    nodes[cs] = (cm, None) if cs not in nodes else (cm, nodes[cs][1])
                if not need:
                    break
                dsts = [renorm(cs) for cs in need] if one_fwd else expand(need)
                for cs, dst in zip(need, dsts):
                    nodes[cs] = (nodes[cs][0], dst)
        while not (fast and not has_tok):
            cand = {}
            for sg, (m, dst) in nodes.items():
                if dst is None:
                    continue
                sp, tp = dst
                slots_taken = {j for j, _c in sg}
                for j in range(S):
                    if j in slots_taken:
                        continue
                    base_m = m * sp[j]
                    if has_tok and use_tok:
                        order = sorted(range(C), key=lambda c: -tp[j][c])
                        ranks = (order[:2] if tp[j][order[0]] < tokgate else order[:1])
                        facts = [tp[j][c] for c in ranks]
                    elif has_tok:                    # positions-only control
                        ranks, facts = (0,), [1.0]
                    elif use_tok:                    # best rule, stale table
                        two = cand_p_cpu[j]
                        ranks = (0, 1) if two[0] < tokgate else (0,)
                        facts = [two[c] for c in ranks]
                    else:
                        ranks, facts = (0,), [1.0]
                    for c, fc in zip(ranks, facts):
                        cm = base_m * fc
                        if gamma and cm < gamma:
                            continue
                        cs = sg | {(j, c)}
                        if cs not in cand or cm > cand[cs][0]:
                            cand[cs] = (cm, sg, j, c)
            sel = sorted(cand.items(), key=lambda kv: (-kv[1][0], len(kv[0])))[:budget]
            need = [cs for cs, (cm, ps, j, c) in sel
                    if cs not in nodes and len(cs) < S]
            for cs, (cm, ps, j, c) in sel:
                if cs not in nodes:
                    nodes[cs] = (cm, None)
                else:
                    nodes[cs] = (cm, nodes[cs][1])
            if not need:
                break
            dsts = [renorm(cs) for cs in need] if one_fwd else expand(need)
            for cs, dst in zip(need, dsts):
                nodes[cs] = (nodes[cs][0], dst)

        out, states = [], {frozenset(): state_blk}
        for cs, (cm, ps, j, c) in sel:
            sl = idx_cpu[j]
            tk = cand_ids_cpu[j][c]
            ns = states[ps].clone()
            ns[sl] = tk
            states[cs] = ns
            pm = nodes[ps][0]
            out.append({"node": len(out), "level": len(cs), "formula": frozenset(),
                        "state": ns, "chosen": {sl: tk},
                        "score": cm, "local": cm / max(pm, 1e-12)})
        return out

    @torch.no_grad()
    def _ddtree_hs(self, p_blk, state_blk, idx, budget, h_blk, gamma: float = 0.0,
                   tokgate: float = 0.0):
        """`_ddtree_batched`'s node selection driven by the two-head proposer.

        Differences from `_ddtree_batched`, and nothing else:
          * per-branch distributions come from the hidden-state model (position head),
            conditioned on the branch's committed TOKENS, not just its slot set;
          * `tokgate` keeps the best's knob semantics EXACTLY, so every rung of the
            ladder changes one thing:
              tokgate = 0      -- POSITIONS ONLY: stale argmax tokens, no token factor.
                                  The pure architecture A/B against the best's own
                                  tokgate=0 (isolates the hidden states).
              tokgate = 0.001  -- g000 analog: the model's token head picks the token
                                  and its probability enters the mass; branching gate
                                  effectively never fires.
              tokgate = 0.6    -- g060 analog: additionally emits the model's SECOND
                                  candidate as a sibling where the model's top token
                                  probability is below the gate.
            With the token head on, tokens are per branch -- two branches may write
            different tokens at the same slot.

        Node keys are frozensets of (j, c): row index into `idx` and candidate rank.
        Same parent-closure argument as `_ddtree_batched`: every mass factor is <= 1, so
        the level-batched fixpoint selects the exact top-`budget` set by mass.
        """
        from .hsproposer import CLS_CAND, CLS_FILLED, CLS_TAKEN, round_tensors

        if h_blk is None:
            return []
        S = int(idx.numel())
        L = int(state_blk.shape[0])
        C = int(self.model.n_cand)
        dev = p_blk.device
        top = p_blk[idx].topk(self.m_top, dim=-1)
        p_top = top.values.float()
        cand_ids_cpu = top.indices[:, :C].tolist()   # (S, C) token ids
        idx_cpu = idx.tolist()
        table, cand_p = round_tensors(p_top, idx, L, C)
        static = self.model.encode_static(h_blk, table)
        base_cls = torch.full((L,), CLS_FILLED, dtype=torch.long, device=dev)
        base_cls[idx] = CLS_CAND
        tau = getattr(self, "_tau", 1.0)

        def expand(sigs):
            n = len(sigs)
            cls = base_cls[None].repeat(n, 1)
            rank = torch.full((n, L), C, dtype=torch.long, device=dev)
            dep = torch.zeros(n, device=dev)
            for i, sg in enumerate(sigs):
                for j, c in sg:
                    cls[i, idx_cpu[j]] = CLS_TAKEN
                    rank[i, idx_cpu[j]] = c
                dep[i] = len(sg)
            pos_lg, tok_lg = self.model(
                None, None, cls, rank, dep / max(L, 1), (L - S + dep) / max(L, 1),
                cand_p[None].expand(n, -1, -1), static=static)
            pos_lg = pos_lg.masked_fill(cls != CLS_CAND, -1e9)
            slot_p = torch.softmax(pos_lg[:, idx] / tau, dim=-1).tolist()   # (n, S)
            tok_p = torch.softmax(tok_lg[:, idx], dim=-1).tolist()          # (n, S, C)
            return list(zip(slot_p, tok_p))

        root = expand([frozenset()])[0]
        nodes = {frozenset(): (1.0, root)}
        sel = []
        while True:
            cand = {}
            for sg, (m, dst) in nodes.items():
                if dst is None:
                    continue
                sp, tp = dst
                slots_taken = {j for j, _c in sg}
                use_tok = tokgate > 0.0
                for j in range(S):
                    if j in slots_taken:
                        continue
                    base_m = m * sp[j]
                    if use_tok:
                        order = sorted(range(C), key=lambda c: -tp[j][c])
                        ranks = (order[:2] if tp[j][order[0]] < tokgate else order[:1])
                    else:
                        ranks = (0,)  # candidate 0 == stale argmax, by construction
                    for c in ranks:
                        cm = base_m * (tp[j][c] if use_tok else 1.0)
                        if gamma and cm < gamma:
                            continue
                        cs = sg | {(j, c)}
                        if cs not in cand or cm > cand[cs][0]:
                            cand[cs] = (cm, sg, j, c)
            sel = sorted(cand.items(), key=lambda kv: (-kv[1][0], len(kv[0])))[:budget]
            need = [cs for cs, (cm, ps, j, c) in sel
                    if cs not in nodes and len(cs) < S]
            for cs, (cm, ps, j, c) in sel:
                if cs not in nodes:
                    nodes[cs] = (cm, None)
                else:
                    nodes[cs] = (cm, nodes[cs][1])
            if not need:
                break
            dsts = [renorm(cs) for cs in need] if one_fwd else expand(need)
            for cs, dst in zip(need, dsts):
                nodes[cs] = (nodes[cs][0], dst)

        out, states = [], {frozenset(): state_blk}
        for cs, (cm, ps, j, c) in sel:  # mass-descending; parents precede children
            sl = idx_cpu[j]
            tk = cand_ids_cpu[j][c]
            ns = states[ps].clone()
            ns[sl] = tk
            states[cs] = ns
            pm = nodes[ps][0]
            out.append({"node": len(out), "level": len(cs), "formula": frozenset(),
                        "state": ns, "chosen": {sl: tk},
                        "score": cm, "local": cm / max(pm, 1e-12)})
        return out





class ConfTreeDrafter:
    """BlockSpec-style baseline: a training-free tree over decoding
    trajectories, exact-verified in our harness.

    BlockSpec itself (Pan et al., OpenReview hmAviop5rm) has no public code, targets
    LLaDA/Dream, and greedy-verifies against a lossy dual-cache reference; this is the
    algorithm-matched analogue under our strict contract. Proposal law: at any
    hypothetical prefix, P(next slot) = stale confidence renormalised over open slots
    (branch-independent -- no learned component, no re-forwards); token = the target's
    coupled draw at that slot. Node mass = product along the branch; the top-D nodes
    by mass are selected with the same parent-closed heap as `_ddtree` (exact).
    Trajectory axis only (their token-choice axis not replicated in v1; noted in the
    report). Zero drafter tax beyond a topk on the stale table.
    """

    prefix_cond = False
    hid = 0

    def __init__(self):
        self.invocations = 0   # zero drafter forwards by construction
        self.positions = 0

    def reset(self):
        pass

    def begin(self, sample_id):
        pass

    @torch.no_grad()
    def chain(self, p_blk, state_blk, mask_id, budget, temperature, noise_blk,
              round_in_block, fan_margin=0.0, root_fan=1, mode="conftree",
              gamma=0.0, h_blk=None, tau=1.0, tokgate=0.0):
        import heapq
        masked = state_blk == mask_id
        idx = masked.nonzero(as_tuple=True)[0]
        S = int(idx.numel())
        if S == 0:
            return []
        scores = perturbed_scores(p_blk, temperature, noise_blk)
        tok = scores.argmax(-1)                       # the target's own draw per slot
        conf = p_blk[idx].max(-1).values.float()      # stale confidence, open slots
        conf = conf.clamp_min(1e-9)

        def dist(taken):
            c = conf.masked_fill(taken, 0.0)
            return c / c.sum().clamp_min(1e-9)

        root = dist(torch.zeros(S, dtype=torch.bool, device=conf.device))
        order0 = torch.argsort(root, descending=True, stable=True)
        out, seen, ctr = [], set(), 0
        heap = [(-float(root[order0[0]]), 0, state_blk.clone(),
                 torch.zeros(S, dtype=torch.bool, device=conf.device),
                 0, root, order0, 0, 1.0)]
        while heap and len(out) < budget:
            neg, _, st, taken, d, p_dist, p_order, rank, p_mass = heapq.heappop(heap)
            mass = -neg
            if rank >= int(p_order.numel()):
                continue
            j = int(p_order[rank])
            if bool(taken[j]):
                continue
            sl = int(idx[j])
            ns = st.clone(); ns[sl] = int(tok[sl])
            nt = taken.clone(); nt[j] = True
            sig = frozenset(nt.nonzero().flatten().tolist())
            if sig not in seen:
                seen.add(sig)
                out.append({"node": len(out), "level": d + 1, "formula": frozenset(),
                            "state": ns.clone(), "chosen": {sl: int(tok[sl])},
                            "score": mass, "local": mass / max(p_mass, 1e-12)})
            if rank + 1 < int(p_order.numel()):
                nxt = int(p_order[rank + 1])
                if not bool(taken[nxt]):
                    ctr += 1
                    heapq.heappush(heap, (-(p_mass * float(p_dist[nxt])), ctr, st,
                                          taken, d, p_dist, p_order, rank + 1, p_mass))
            if d + 1 < S and len(out) < budget:
                ch = dist(nt)
                ch_order = torch.argsort(ch, descending=True, stable=True)
                ctr += 1
                heapq.heappush(heap, (-(mass * float(ch[ch_order[0]])), ctr, ns, nt,
                                      d + 1, ch, ch_order, 0, mass))
        return out


class SiblingChainDrafter:
    """SimSD's ALGORITHMIC CORE, isolated from its deployment optimizations.

    SimSD (arXiv 2606.02544; github airevo2/SimSD-release) has a 1.7B SDAR sibling decode
    the block and the 8B verify the resulting chain -- shipped behind a 2-GPU stream-
    overlapped pipeline whose overlap HIDES the drafter's cost. This class runs exactly
    the technique -- the sibling decodes forward from the current state, committing its
    own top-confidence slot with its own argmax token, emitting the nested chain -- inside
    our harness: same exact verification, same NFE accounting (target forwards only),
    and the sibling's forwards timed into t_draft_s, so honest TPS pays the tax the
    pipeline hides. One GPU, one process: the drafter comparison the paper needs.

    Lossless as always: proposals only; every commit is the target's own sample.
    """

    def __init__(self, aux, use_cache: bool = True):
        self.ad = aux
        self.mask_id = aux.mask_id
        self.bias_fn = True   # routes decode's full-sequence propose() call
        self.invocations = 0  # sibling forwards -- drafter tax, never NFE
        self.positions = 0
        # the sibling gets the SAME lossless prefix cache the target enjoys
        # (exact on a block-causal model). Without it every draft level pays a
        # full-prefix forward -- an unfair tax our /rows charged SimSD.
        self.use_cache = bool(use_cache)
        self._past = None
        self._cache_lo = -1

    def reset(self):
        pass

    def begin(self, sample_id):
        self._past = None
        self._cache_lo = -1

    @torch.no_grad()
    def propose(self, x, b0, b1, L, budget, temperature, noise_blk):
        from .decode import _attn_bias

        aux = self.ad
        sh = int(getattr(aux, "logit_shift", 0))
        bc = bool(getattr(aux, "block_causal", False))
        out = []
        work = x.clone().to(aux.device)
        state = work[0, b0:b1]
        score = 1.0
        lo = max(0, b0 - sh)
        cached = self.use_cache and bc and lo > 0
        if cached and self._cache_lo != lo:
            # one refresh forward per block: build + trim the prefix KVs (frozen
            # inside a block; commits land only in [b0, b1))
            pl0 = int(work.shape[1])
            bias0 = _attn_bias(0, pl0, 0, L, sh, 0, aux.device,
                               block_causal=bc, chunk_start=0)
            o0 = aux.forward(work, torch.arange(pl0, device=aux.device), bias0,
                             None, True)
            self.invocations += 1
            self.positions += pl0
            self._past = aux.trim_cache(o0.past, lo)
            self._cache_lo = lo
        for lvl in range(budget):
            masked = state == self.mask_id
            if not bool(masked.any()):
                break
            pl = int(work.shape[1])
            if cached:
                seg = work[:, lo:b1]
                bias = _attn_bias(lo, b1 - lo, 0, L, sh, b0 - lo, aux.device,
                                  block_causal=bc, chunk_start=lo)
                o = aux.forward(seg, torch.arange(lo, b1, device=aux.device),
                                bias, self._past, False)
                self.invocations += 1
                self.positions += b1 - lo
                probs = aux.block_probs(o.logits, b0 - lo, b1 - lo)
            else:
                bias = _attn_bias(0, pl, 0, L, sh, 0, aux.device,
                                  block_causal=bc, chunk_start=0)
                o = aux.forward(work, torch.arange(pl, device=aux.device), bias,
                                None, False)
                self.invocations += 1
                self.positions += pl
                probs = aux.block_probs(o.logits, b0, b1)
            conf, tok = probs.max(-1)
            cm = torch.where(masked, conf, torch.full_like(conf, -1.0))
            slot = int(cm.argmax())
            t = int(tok[slot])
            state = state.clone()
            state[slot] = t
            work = work.clone()
            work[0, b0 + slot] = t
            score *= float(conf[slot])
            out.append({"node": lvl, "level": lvl + 1, "formula": frozenset(),
                        "state": state.clone().to(x.device), "chosen": {slot: t},
                        "score": score, "local": float(conf[slot])})
        return out


class LiveOracleDrafter:
    """Oracle defined INSIDE the current run, not replayed from a recorded one.

    The recorded-replay oracle (`OracleDrafter`) assumes the drafted pass reproduces the
    draft-free pass's commit sequence. That holds on a dense target and fails on an MoE:
    LLaDA2.0-mini routes experts by batch composition, so adding replicas changes the logits
    and the drafted run leaves the recorded path after ~11 commits. The replay then
    proposes states from a trajectory the run is no longer on, and the "oracle" scores 48%
    WORSE than a real drafter.

    This version derives the continuation from the run's own current state: at each level it
    forwards the hypothetical state and applies the decoder's own commit rule (highest
    confidence masked slot; argmax token at T=0). Those lookahead forwards are the oracle's
    cheat and are deliberately NOT counted in NFE -- the question is how many *verification*
    rounds a perfect drafter would need, not what it costs to be perfect.

    Residual imperfection, stated: the lookahead forward has its own batch composition, so on
    an MoE its argmax can still differ from the drafted forward's on near-ties. That makes
    this a very-near-oracle rather than an exact one, which is the intended reading -- it is
    assumed correct wherever it is supposed to be correct.

    `mode` decomposes the bound the same way the recorded-replay oracle does:
      both -- true slot AND true token at every level (the full upper bound).
      pos  -- true slot, but the token a real drafter would emit: the ROOT state's argmax
              at that slot (level-0 lookahead = the verify forward's own distribution).
              Acceptance loss below `both` is the price of token error alone.
      tok  -- true token, but the slot a naive chain drafter would pick: the root
              confidence order over still-open slots. Loss below `both` is the price of
              slot error alone.
    """

    def __init__(self, adapter, bias_fn, mask_id: int, mode: str = "both"):
        assert mode in ("pos", "tok", "both")
        self.ad = adapter
        self.bias_fn = bias_fn
        self.mask_id = mask_id
        self.mode = mode
        self.invocations = 0   # lookahead forwards, reported but excluded from NFE
        self.positions = 0

    def reset(self):
        pass

    def begin(self, sample_id):
        pass          # no recorded trajectory to seek into

    def advance(self, n):
        pass          # the continuation is re-derived from the live state each round

    @torch.no_grad()
    def propose(self, x, b0, b1, L, budget, temperature, noise_blk):
        """Nested draft states following the run's own future, `budget` levels deep.

        Two states are tracked: `truth` advances with the target's own commits (what the
        lookahead reveals) and `draft` is what we emit. In `both` mode they coincide. In
        `pos` mode draft tokens come from the root distribution, so draft may leave truth
        -- exactly the divergence whose cost is being measured; the lookahead stays on
        truth so the slot sequence remains the true continuation's. In `tok` mode the
        lookahead follows the drafted state itself (root-confidence slot order), so the
        emitted token is exactly what the target would sample at that state and slot.
        """
        out = []
        truth = x[0, b0:b1].clone()
        draft = truth.clone()
        work = x.clone()
        root_tok = None   # pos mode: drafter-visible tokens, from the level-0 forward
        root_conf = None  # tok mode: drafter-visible slot confidences, same forward
        for lvl in range(budget):
            look = draft if self.mode == "tok" else truth
            masked = look == self.mask_id
            if not bool(masked.any()):
                break
            work[0, b0:b1] = look
            pl = int(work.shape[1])
            pos_ids = torch.arange(pl, device=work.device)
            bias = self.bias_fn(pl, pl, 0)
            o = self.ad.forward(work, pos_ids, bias, None, False)
            self.invocations += 1
            self.positions += pl
            logits = o.logits[0, b0:b1].float()
            probs = torch.softmax(logits, dim=-1)
            scores = perturbed_scores(probs, temperature, noise_blk)
            tok = scores.argmax(-1)
            conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            if lvl == 0:
                root_tok = tok.clone()
                root_conf = conf.clone()
            if self.mode == "tok":
                # drafter-side slot: root confidence order over still-open slots
                cav = torch.where(masked, root_conf, torch.full_like(root_conf, -1.0))
                slot = int(cav.argmax())
                t_draft = int(tok[slot])       # oracle token at this state and slot
                draft = draft.clone(); draft[slot] = t_draft
            else:
                confm = torch.where(masked, conf, torch.full_like(conf, -1.0))
                slot = int(confm.argmax())     # oracle slot: the true next commit
                t_true = int(tok[slot])
                truth = truth.clone(); truth[slot] = t_true
                t_draft = t_true if self.mode == "both" else int(root_tok[slot])
                draft = draft.clone(); draft[slot] = t_draft
            out.append({"node": lvl, "level": lvl + 1, "formula": frozenset(),
                        "state": draft.clone(), "chosen": {slot: t_draft},
                        "score": 1.0, "local": 1.0})
        return out
