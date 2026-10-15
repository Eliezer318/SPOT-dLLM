"""The decoder.

One function, four independent runtime flags:

    cache      on | off              prefix KV caching
    unmasking  threshold | single
    draft      spiffy | free | none  draft-graph source
    threshold  float                 confidence bar when unmasking=threshold

The code path is identical for every combination; a flag only changes whether a cache
is kept, how many slots a commit unmasks, and whether draft blocks are appended to the
forwarded chunk.  Per-round records are emitted for every configuration (including the
baseline), so ablations are comparable record-for-record.
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .drafters import OracleDrafter, adaptive_graph, cross_chain
from .graphs import DraftGraph, Formula
from .sampling import (CoupledNoise, commit_law, commit_law_pair, kl_divergence,
                       perturbed_scores, rs_commit_sample, sample_rs_chain,
                       total_variation)

# Large-but-finite: representable in fp32 and bf16 alike, and still exp()s to exactly 0 in
# the softmax.  torch.finfo(float32).min overflows to -inf when SDPA casts the mask to bf16.
NEG = -1e30


@dataclass
class DecodeConfig:
    cache: str = "on"  # on | off
    unmasking: str = "threshold"  # threshold | single
    threshold: float = 0.9
    # spiffy: fixed graph, calibrated offline against frequency.
    # free:   D_free -- the same coordinates and the same scorer, but the node set is
    #         enumerated per round from the live factorised joint.  Zero model calls either
    #         way; the pair isolates what the offline calibration step is worth.
    draft: str = "none"  # spiffy | free | slot | oracle | none
    # match: exact state matching under the coupled-noise process (byte-reproducible).
    # rs: rejection-sampling verification -- lossless in law under INDEPENDENT sampling
    # semantics; per-link acceptance is the maximal coupling 1 - TV(p_j, q_j).
    verify: str = "match"  # match | rs
    # Stage B: draft chains may continue into block k+1.  A replica piece then spans TWO
    # blocks -- its own drafted copy of block k followed by its block-k+1 state -- and the
    # k+1 half attends to that copy, never the cache, which is exact under block-causality
    # once the copy equals the true completed block (the acceptance condition).
    cross_block: str = "off"  # off | on
    # Margin fan-out: when the drafter's top-2 slot confidences are within this margin,
    # branch over both slots (replicas spent on the drafter's own uncertainty).  0 = off.
    # >0 switches the free drafter to the chain builder (labeled arm; the GM-beam graph
    # is the within-block default at 0).
    draft_free_fan_margin: float = 0.0
    draft_free_root_fan: int = 1
    slot_mode: str = "chain"  # chain | bestfirst | graph | ddtree | ddtree_b | pl
    # Temperature-sweep switches ().  draft_noise: "coupled" (default) hands the
    # drafter the verifier's Gumbel noise so its argmax proposal IS the target's draw;
    # "none" gives the drafter no noise (plain argmax proposals, verifier still samples):
    # lossless, lower acceptance.  verify_greedy: a drafted state is accepted if it equals
    # the target's ARGMAX commit while undrafted commits still sample -- the "exact match
    # at T>0" of greedy verifiers; LOSSY by construction (greedy/sampled mixture).
    draft_noise: str = "coupled"  # coupled | none
    verify_greedy: bool = False
    slot_gamma: float = 0.0   # adaptive budget: stop drafting below this prefix mass
    slot_tokgate: float = 0.0 # branch the 2nd token where stale top-1 < gate
    slot_tau: float = 1.0     # picker-softmax calibration temperature (tree shape) (D_slot topology)  # branches at the root of the cross chain
    # forward with output_hidden_states and hand the STALE block hidden rows (the
    # ones aligned with src_probs) to the probe, for hidden-state distillation dumps.
    want_hidden: bool = False
    # refresh the drafting table with a cheap AUX model's forward at the CURRENT
    # state before each drafted round. "full" replaces the stale table wholesale
    # (ranking, tokens, and mass all come from the fresh aux distribution). The aux
    # forward is timed into t_draft_s -- it is drafter tax, and honest TPS must pay it.
    slot_refresh: str = "off"  # off | full
    draft_budget: int = 3  # D*: nodes kept after pruning
    # D_free search reach.  Defaults deliberately exceed the calibrated graph's j=1 in the
    # vocabulary dimension; set to 1 for the matched-reach control that separates "adaptive"
    # from "wider".
    draft_free_vocab_ranks: int = 3
    draft_free_max_depth: int = 6
    draft_free_skip: int = 1
    draft_free_beam: int = 8  # formulas kept per depth during enumeration
    # Nodes enumerated per round.  This MUST be large enough for the beam to reach
    # `draft_free_max_depth`: truncating the enumeration caps the drafts' depth, and a
    # depth-capped drafter loses to the calibrated graph for a reason that has nothing to
    # do with adaptive selection.  Cost is O(n^2) subset checks per round on ~40 nodes,
    # which is microseconds -- there is no reason to be stingy here.
    draft_free_pool: int = 4  # floor: pool * budget
    # How draft candidates are ORDERED by position rank.  Measured on 400 trajectories
    # Ordering masked slots by max probability predicts the
    # next commit 73.4% of the time, while ordering by the sampled token's probability --
    # what the decoder itself ranks by at T=1.0 -- manages only 49.9%.  The drafter is free
    # to disagree with the decoder here: it only chooses what to PROPOSE.  Commits still come
    # from select_commit under the decoder's own rule, so losslessness is untouched.
    draft_free_order: str = "maxprob"  # maxprob | sampler
    # Under block-causality the current block cannot be affected by LATER blocks, yet the
    # cached path forwards [b0, total) -- every remaining all-masked block -- on every round.
    # Measured: 66 positions per call when only 32 can matter.  Truncating at b1 is exact for
    # a block-causal model and roughly halves per-round cost, which also makes each draft
    # replica a larger share of a smaller total.  Never valid for a bidirectional model, so
    # it is gated on adapter.block_causal as well as this flag.
    suffix: str = "keep"  # keep | skip
    # Pad every non-refresh forward to this many draft pieces with fully-masked dummy
    # replicas (excluded from acceptance).  0 = off.  Makes the forward shape invariant to
    # how many drafts materialised, so the sampled path is bitwise stable across draft
    # configurations -- oracle_diag showed varying shape flips ~7%/commit of temp-1.0
    # samples, which is what breaks oracle chains.
    pad_pieces: int = 0
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    # concat: draft blocks appended to one sequence behind a blockwise mask (the paper's
    #   Fig. 3).  Cheap, but a draft cannot be attended to by the suffix, so its
    #   conditional only approximates p(.|X(t-1)).
    # batch:  each draft is a full sequence with the block substituted in place, so its
    #   conditional IS a fresh forward on that state -- exact by construction, at the cost
    #   of one whole sequence per draft.  Requires cache=off to stay exact.
    draft_layout: str = "concat"  # concat | batch
    seed: int = 0
    audit: str = "off"  # off | conditional
    # Pad the audit's *reference* forward to this batch size by duplicating the row. The
    # reference is mathematically identical either way, so any divergence it then shows is
    # pure float nondeterminism -- measured at exactly the slots the audit scores, which is
    # the matched floor every other cell has to be judged against.
    audit_ref_batch: int = 1
    # .  The plain audit reference is a (1, total) forward, while a drafted round's
    # forward is (1, main_len + D*piece_len) and reads accepted-draft commits off a *draft
    # piece*, not the main chunk.  So a depth>0 record mixes two things: whatever the draft
    # layer really costs, and pure float noise from a different tensor shape and a different
    # offset within it.  With this on, each audit record additionally runs a reference that
    # is shape-, offset- and content-matched to the cell's own forward -- same number of
    # draft pieces, same states, read at the same place -- and is *mathematically identical*
    # to the plain reference on a block-causal model.  Its divergence from the plain
    # reference is therefore the matched floor for draft rows, and `tv_matched` is the cell's
    # divergence with that floor removed by construction rather than by subtraction.
    audit_matched_shape: bool = False

    def validate(self):
        assert self.cache in ("on", "off")
        assert self.unmasking in ("threshold", "single")
        assert self.draft in ("spiffy", "free", "slot", "sibling",
                              "oracle", "none")
        assert self.draft_layout in ("concat", "batch")
        assert self.draft_free_order in ("maxprob", "sampler")
        assert self.suffix in ("keep", "skip")
        assert self.verify in ("match", "rs")
        assert self.cross_block in ("off", "on")
        assert self.audit in ("off", "conditional")
        assert self.gen_length % self.block_length == 0
        assert self.temperature >= 0.0

    def is_distribution_preserving(self, block_causal: bool = False) -> tuple:
        """Mechanistic verdict, before any measurement.  (verdict, reasons-it-is-not)

        `block_causal` is a property of the MODEL, not of the flags, and it changes two of
        the three verdicts.  On a block-causal checkpoint earlier blocks never attend to the
        current one, so (a) the cached prefix K/V are the same tensors a full forward would
        build -- no staleness -- and (b) an appended draft's conditional IS p(.|X(t-1)),
        because nothing in the true sequence needed to attend to it.  Both were verified
        bitwise exact against the uncached decoder at every layer.

        The factorisation error of multi-slot commits is untouched by attention structure
        and stays lossy everywhere.
        """
        why = []
        if self.cache == "on" and not block_causal:
            why.append("prefix KV cache is built once per block and reused as the block is "
                       "denoised, so the model conditions on stale keys/values")
        if self.unmasking == "threshold":
            why.append("multi-slot commits draw each slot from its own marginal, i.e. from a "
                       "factorised approximation of a joint the model never evaluated")
        if self.draft != "none" and self.draft_layout == "concat" and not block_causal:
            why.append("draft blocks sit at the end of the sequence and cannot be attended to "
                       "by the suffix, so the accepted draft's conditional is not p(.|X(t-1))")
        if self.verify_greedy and self.temperature > 0.0 and self.draft != "none":
            why.append("verify_greedy: drafted slots are accepted by argmax match while undrafted "
                       "slots sample at T, so the output is a greedy/sampled mixture, not p(.|X)")
        return (len(why) == 0), why


def commit_view(probs, temperature, noise):
    """(scores, token, confidence) for every slot of a block.

    token is a draw from the conditional (argmax of the Gumbel-perturbed scores; plain
    argmax at T=0).  confidence is the probability the conditional assigns to that drawn
    token -- this is what ranks slots and what the threshold rule tests, following
    Fast-dLLM, and it reduces to the top-1 probability at T=0.
    """
    scores = perturbed_scores(probs, temperature, noise)
    tok = scores.argmax(dim=-1)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return scores, tok, conf


# ----------------------------------------------------------------------------------
# drafting
# ----------------------------------------------------------------------------------
def _position_order(probs: torch.Tensor, masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked block slots ordered by max-probability, descending.

    Returns (ordered_positions, per-position max prob).
    """
    conf, _ = probs.max(dim=-1)
    conf_m = torch.where(masked, conf, torch.full_like(conf, -1.0))
    # stable=True: CUDA argsort is unstable, and SDAR ties at conf 1.0 on ~72% of commits; the
    # verifier (select_commit) breaks ties by argmax = lowest index, so the drafter must too.
    order = torch.argsort(conf_m, descending=True, stable=True)
    n = int(masked.sum().item())
    assert n == 0 or int(order[0]) == int(conf_m.argmax()), "drafter/verifier tie-break mismatch"
    return order[:n], conf


def prune_and_materialise(
    graph: DraftGraph,
    probs: torch.Tensor,
    block_state: torch.Tensor,
    mask_id: int,
    budget: int,
    temperature: float = 0.0,
    noise: Optional[torch.Tensor] = None,
    order_scores: Optional[torch.Tensor] = None) -> List[dict]:
    """Materialise every graph node against the live distribution, then keep the best `budget`.

    Materialisation (paper Sec 4.4.2): the formula's (i, j) coordinates are resolved into
    concrete (slot, token) pairs using the current ranking of still-masked slots.
    Pruning (paper Sec 4.4.4): score(q) = GM(localScore(q), GM of children's localScore),
    where localScore is the geometric mean of the token probabilities the node commits to.

    The whole thing runs off three device->host transfers per round rather than one per
    (node, pair), which keeps the drafting overhead well under the model call.
    """
    L = block_state.shape[0]
    masked = block_state == mask_id
    scores, _, conf = commit_view(probs, temperature, noise)
    rank_by = conf if order_scores is None else order_scores
    conf_m = torch.where(masked, rank_by, torch.full_like(rank_by, -1e30))
    n_masked = int(masked.sum().item())
    # materialisation must resolve rank i against the SAME ordering the graph was built
    # with, or the formulas mean something different than they did during search
    order = torch.argsort(conf_m, descending=True, stable=True)[:n_masked].tolist()
    assert n_masked == 0 or order[0] == int(conf_m.argmax()), "drafter/verifier tie-break mismatch"
    max_j = graph.max_vocab_rank()
    # vocabulary rank is a rank in the *sampler's* ordering, so that a draft predicts what
    # would actually be drawn; at T=0 this is the plain top-k of the conditional.
    _, ti = torch.topk(scores, max_j, dim=-1)
    tv = probs.gather(-1, ti)
    tv, ti = tv.tolist(), ti.tolist()

    mats: List[Optional[tuple]] = []
    for node in graph.nodes:
        pairs = sorted(node.formula)
        if any(i > n_masked for i, _ in pairs):
            mats.append(None)
            continue
        chosen: Dict[int, int] = {}
        logp = 0.0
        for i, j in pairs:
            pos = order[i - 1]
            chosen[pos] = ti[pos][j - 1]
            p = tv[pos][j - 1]
            logp += math.log(p) if p > 0 else -80.0
        mats.append((chosen, math.exp(logp / len(pairs))))

    children = graph.children_map()
    scored = []
    for idx, m in enumerate(mats):
        if m is None:
            continue
        local = m[1]
        kids = [mats[c][1] for c in children[idx] if mats[c] is not None]
        if kids:
            child_score = math.exp(sum(math.log(max(k, 1e-30)) for k in kids) / len(kids))
            score = (local * child_score) ** 0.5
        else:
            score = local
        scored.append((score, idx))
    scored.sort(key=lambda t: -t[0])

    out = []
    for score, idx in scored[:budget]:
        chosen, local = mats[idx]
        state = block_state.clone()
        for pos, tok in chosen.items():
            state[pos] = tok
        out.append(
            {
                "node": idx,
                "level": graph.nodes[idx].level,
                "formula": sorted(graph.nodes[idx].formula),
                "state": state,
                "chosen": chosen,
                "score": score,
                "local": local,
            }
        )
    return out


# ----------------------------------------------------------------------------------
# unmasking rule
# ----------------------------------------------------------------------------------
def select_commit(probs, block_state, cfg: DecodeConfig, mask_id: int, noise=None):
    """Which slots to unmask this step, and with which tokens."""
    masked = block_state == mask_id
    _, argmax_tok, conf = commit_view(probs, cfg.temperature, noise)
    conf_m = torch.where(masked, conf, torch.full_like(conf, -1.0))
    if cfg.unmasking == "single":
        sel = conf_m.argmax().view(1)
    else:
        sel = (conf_m > cfg.threshold).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            sel = conf_m.argmax().view(1)
    return sel, argmax_tok[sel], conf[sel]


# ----------------------------------------------------------------------------------
# forward-chunk assembly
# ----------------------------------------------------------------------------------
def _assemble(x, chunk_start, b0, b1, drafts, shift, device, chunk_end=None,
              n_dummy=0, mask_id=0, dummy_len=0):
    """Build (input_ids, position_ids, layout) for one model call.

    Layout of the forwarded chunk:  [ true tail from chunk_start ] [ draft piece ] x D*
    Each draft piece carries the position ids of the block it drafts for; for models whose
    logits are shifted by one (Dream), the piece is prefixed with the true token at b0-1 so
    that the draft has its own predecessor.

    `n_dummy` appends that many fully-masked dummy pieces AFTER the real drafts (so real
    piece offsets are unchanged).  A dummy is the state of a draft with no commits; its
    rows are never read.  Purpose: keep the forward shape constant across rounds.
    """
    total = x.shape[1] if chunk_end is None else chunk_end
    pieces = [x[:, chunk_start:total]]
    positions = [torch.arange(chunk_start, total, device=device)]
    main_len = total - chunk_start
    for d in drafts:
        st = d["state"]
        piece = torch.cat([x[0, b0 - shift : b0], st]).view(1, -1)
        pieces.append(piece)
        positions.append(torch.arange(b0 - shift, b0 + st.shape[0], device=device))
    if n_dummy:
        blank = torch.full((dummy_len if dummy_len else b1 - b0,), mask_id,
                           dtype=x.dtype, device=x.device)
        dummy = torch.cat([x[0, b0 - shift : b0], blank]).view(1, -1)
        for _ in range(n_dummy):
            pieces.append(dummy)
            positions.append(torch.arange(b0 - shift, b0 + blank.shape[0], device=device))
    return torch.cat(pieces, dim=1), torch.cat(positions), main_len


def _assemble_batch(x, b0, b1, drafts):
    """Exact layout: one full sequence per draft, with the block substituted in place.

    Row 0 is the true sequence; row m is the sequence the model would see if the block
    were in draft state m.  Row m's logits are therefore literally p(. | X'; X^m) -- the
    same thing a fresh forward on that state returns -- which is what makes acceptance
    distribution-preserving.  Costs one whole sequence per draft.
    """
    rows = [x]
    for d in drafts:
        r = x.clone()
        r[0, b0:b1] = d["state"]
        rows.append(r)
    return torch.cat(rows, dim=0)


def _block_causal_block(k_lo, k_hi, q_lo, q_hi, block_len, device):
    """Additive bias for a block-causal true sequence: block j attends to blocks <= j.

    Absolute positions, so block boundaries land on multiples of `block_len` -- which is
    what a block-diffusion checkpoint is trained with.  Needed for SDAR, whose attention
    carries no implicit causality of its own: given no mask it attends everywhere.
    """
    q = torch.arange(q_lo, q_hi, device=device).unsqueeze(1) // block_len
    k = torch.arange(k_lo, k_hi, device=device).unsqueeze(0) // block_len
    return torch.where(k <= q, 0.0, NEG)


def _attn_bias(past_len, main_len, n_drafts, block_len, shift, cur_lo_main, device,
               block_causal=False, chunk_start=0, piece_blocks=1):
    """Blockwise draft mask (paper Fig. 3), as an additive float bias.

    true rows      -> everything except draft columns
    draft rows     -> prefix + true sequence minus the block being drafted + itself

    With `block_causal`, the true sequence additionally carries block-causal structure
    rather than being fully bidirectional.  `chunk_start` is the absolute position of
    column/row 0 of the *main* part, so block boundaries stay aligned to absolute position.
    """
    piece_len = piece_blocks * block_len + shift
    K = past_len + main_len + n_drafts * piece_len
    bias = torch.zeros((1, 1, K, K), dtype=torch.float32, device=device)
    d0 = past_len + main_len
    if block_causal:
        # rows/cols [0, d0) are the true sequence, starting at absolute `chunk_start`
        a_lo = chunk_start - past_len
        bias[:, :, :d0, :d0] = _block_causal_block(
            a_lo, a_lo + d0, a_lo, a_lo + d0, block_len, device
        )
    # nothing in the true sequence may see a draft
    bias[:, :, :d0, d0:] = NEG
    cur_lo = past_len + cur_lo_main
    cur_hi = cur_lo + piece_blocks * block_len
    for d in range(n_drafts):
        r0 = d0 + d * piece_len
        r1 = r0 + piece_len
        bias[:, :, r0:r1, cur_lo:cur_hi] = NEG  # not the block it is standing in for
        bias[:, :, r0:r1, d0:] = NEG  # nor any other draft
        if block_causal:
            # A draft stands in for the current block, so under block-causal attention it
            # must see strictly earlier blocks only.  Without this it would attend to true
            # blocks *after* the one it replaces -- which the real block cannot do, so its
            # conditional would not be the one being verified.  (Moot when cache=on, where
            # the chunk holds only the current block, but not when cache=off.)
            bias[:, :, r0:r1, cur_hi:d0] = NEG
        if piece_blocks == 1:
            bias[:, :, r0:r1, r0:r1] = 0.0  # but itself
        else:
            # a multi-block piece is its own little block-causal sequence: the k+1 half
            # attends its own completed-k copy, the k half never sees the k+1 half
            a_lo = chunk_start + cur_lo_main - shift
            bias[:, :, r0:r1, r0:r1] = _block_causal_block(
                a_lo, a_lo + piece_len, a_lo, a_lo + piece_len, block_len, device
            )
        if shift:
            # the duplicated predecessor token is a query only: hide its key from everyone else
            bias[:, :, :r0, r0 : r0 + shift] = NEG
            bias[:, :, r1:, r0 : r0 + shift] = NEG
    return bias


# ----------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------
@torch.no_grad()
def generate(
    adapter,
    prompt_ids: torch.Tensor,
    cfg: DecodeConfig,
    graph: Optional[DraftGraph] = None,
    sample_id: str = "",
    probe: Optional[Callable[[dict], None]] = None,
    drafter=None,
    aux=None) -> dict:
    cfg.validate()
    if cfg.draft == "spiffy":
        assert graph is not None, "draft=spiffy needs a calibrated graph"
    if cfg.draft == "oracle":
        assert drafter is not None, "draft=oracle needs a drafter"
        if hasattr(drafter, "begin"):
            drafter.begin(sample_id)
    if cfg.draft == "slot":
        assert drafter is not None, f"draft={cfg.draft} needs a drafter"
        drafter.reset()  # its prefix cache is per-sample, like the target's
    device = adapter.device
    mask_id = adapter.mask_id
    shift = getattr(adapter, "logit_shift", 0)
    L = cfg.block_length
    P = int(prompt_ids.shape[1])
    # Block-diffusion checkpoints align blocks to absolute multiples of L, and the first
    # generated block is P // L -- **floor**.  The masked tail of the prompt's own block is
    # decoded as generation.  Using ceil instead strands MASK tokens between prompt and
    # generation, which is out of distribution and degrades output silently.
    bc = getattr(adapter, "block_causal", False)
    if bc:
        first_blk = P // L
        n_blocks = -(-(P + cfg.gen_length) // L) - first_blk
        total = (first_blk + n_blocks) * L
    else:
        first_blk = 0
        n_blocks = cfg.gen_length // L
        total = P + cfg.gen_length
    x = torch.full((1, total), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    # Noise is indexed by ABSOLUTE position, so it must start where the first decoded block
    # starts.  Under block-causal alignment that is (P // L) * L, which is <= P -- starting
    # at P would make the first block's slice run off the front of the array.
    noise_lo = first_blk * L if bc else P
    noise_all = (
        CoupledNoise(cfg.seed, noise_lo, total, adapter.vocab_size, device)
        if cfg.temperature > 0.0
        else None
    )
    # rs verification uses fresh randomness (independent sampling semantics); seeded per
    # (cfg.seed, sample) so runs replicate.  match arms never touch this generator.
    rs_gen = None
    if cfg.verify == "rs":
        assert cfg.temperature > 0.0, "verify=rs is meaningless at T=0 (greedy == match)"
        assert cfg.draft in ("none", "free", "slot"), \
            "verify=rs: none|free|slot"
        if cfg.draft == "slot":
            assert cfg.slot_mode in ("chain", "purechain"), \
            "verify=rs needs a sequential proposal law (chain/purechain)"
        rs_gen = torch.Generator(device=device)
        rs_gen.manual_seed((cfg.seed * 1000003 + zlib.crc32(str(sample_id).encode())) % 2**31)
    past = None
    cache_len = 0
    rounds: List[dict] = []
    audit_records: List[dict] = []
    n_inv = 0
    n_positions = 0
    # T>0: the adapter must emit softmax(logits/T) -- see sampling.perturbed_scores's
    # contract. Exactly 1.0 at T=0 (greedy reads argmax, division would be a no-op).
    adapter.sample_temp = cfg.temperature if cfg.temperature > 0.0 else 1.0
    n_verify_rows = 0
    n_draft_states = 0
    t0 = time.perf_counter()
    commits_total = 0
    round_global = 0
    instr_total = 0.0  # time spent on statistics that no decoding decision depends on

    for b in range(n_blocks):
        if bc:
            b0 = (first_blk + b) * L      # absolute block boundaries
        else:
            b0 = P + b * L                # generation region starts right after the prompt
        b1 = b0 + L
        # cross-block window: [b0, b0+W).  Only when a next block exists; the last block
        # always decodes within-block.  v1 scope: block-causal models, match verify.
        cross = cfg.cross_block == "on" and bc and (b1 + L) <= total
        if cfg.cross_block == "on":
            assert bc and cfg.verify == "match" and cfg.draft in ("free", "none"), \
                "cross_block v1: block-causal model, verify=match, draft free|none"
            assert cfg.audit != "conditional" or not cfg.audit_matched_shape, \
                "cross_block: matched-shape audit not supported"
        W = 2 * L if cross else L
        bW = b0 + W
        noise_blk = noise_all.block(b0, b1) if noise_all is not None else None
        noise_win = (noise_all.block(b0, bW) if (noise_all is not None and cross)
                     else noise_blk)
        # drafter-side noise: identical to the verifier's under "coupled", absent under "none"
        noise_d = noise_blk if cfg.draft_noise == "coupled" else None
        noise_wd = noise_win if cfg.draft_noise == "coupled" else None
        p_prev: Optional[torch.Tensor] = None  # drafting source distribution
        h_prev: Optional[torch.Tensor] = None  # block hidden states (slot+hid)
        p_prev_commits_ago = 0  # commits since the forward that produced p_prev
        round_in_block = 0

        while bool((x[0, b0:b1] == mask_id).any()):
            t_round0 = time.perf_counter()
            # ---- 1. draft ------------------------------------------------------
            drafts: List[dict] = []
            draft_graph_size = 0
            t_draft = 0.0
            n_aux_fwd = 0
            # must exist for every drafter, including `spiffy`, which shares the
            # materialisation call below but never sets an ordering override
            ord_sc = None
            if cfg.draft != "none" and p_prev is not None:
                t_d0 = time.perf_counter()
                if cfg.draft in ("oracle", "sibling"):
                    if hasattr(drafter, "bias_fn"):
                        # live oracle: derives the continuation from THIS run's own state, so
                        # MoE routing drift cannot desynchronise it the way replaying a
                        # recorded trajectory does.
                        drafts = drafter.propose(x, b0, b1, L, cfg.draft_budget,
                                                 cfg.temperature, noise_d)
                    else:
                        # states come straight from the recorded future; no graph, no scorer
                        drafts = drafter.propose(p_prev, x[0, b0:b1], mask_id,
                                                 cfg.draft_budget, b, cfg.temperature,
                                                 noise_d)
                    g_round = None
                    draft_graph_size = len(drafts)
                elif cfg.draft == "slot" and cfg.slot_mode == "graph":
                    # learned ranking inside the free drafter's tree builder
                    src_probs = p_prev
                    ord_sc = drafter.order_scores(p_prev, x[0, b0:b1], mask_id,
                                                  round_in_block)
                    g_round = adaptive_graph(
                        src_probs, x[0, b0:b1], mask_id, order_scores=ord_sc,
                        n_nodes=max(cfg.draft_free_pool * cfg.draft_budget,
                                    cfg.draft_free_beam * cfg.draft_free_max_depth),
                        vocab_ranks=cfg.draft_free_vocab_ranks,
                        max_depth=cfg.draft_free_max_depth,
                        skip=cfg.draft_free_skip, beam=cfg.draft_free_beam,
                        temperature=cfg.temperature, noise=noise_d)
                elif cfg.draft == "slot" and cfg.verify == "rs":
                    # rejection sampling on positions: slot ~ picker softmax, token ~
                    # stale marginal; each link records its exact proposal law so the
                    # accept ratio and the residual subtract the SAME q (losslessness)
                    drafts = drafter.rs_chain(p_prev, x[0, b0:b1], mask_id,
                                              cfg.draft_budget, round_in_block, rs_gen)
                    g_round = None
                    draft_graph_size = len(drafts)
                elif cfg.draft == "slot":
                    p_draft_src = p_prev
                    if aux is not None and cfg.slot_refresh == "full":
                        # one aux forward at the CURRENT state un-stales the table.
                        # Full-sequence, no drafts, aux's own causality; block_probs
                        # handles the aux logit shift. Cost lands in t_draft_s.
                        pl = int(x.shape[1])
                        a_bias = _attn_bias(0, pl, 0, L,
                                            int(getattr(aux, "logit_shift", 0)), 0,
                                            aux.device,
                                            block_causal=bool(getattr(aux, "block_causal",
                                                                      False)),
                                            chunk_start=0)
                        a_out = aux.forward(x, torch.arange(pl, device=aux.device),
                                            a_bias, None, False)
                        p_draft_src = aux.block_probs(a_out.logits, b0, b1)
                        n_aux_fwd = 1
                    # learned position ranking; tokens are still the target's own argmax
                    drafts = drafter.chain(p_draft_src, x[0, b0:b1], mask_id, cfg.draft_budget,
                                           cfg.temperature, noise_d, round_in_block,
                                           fan_margin=cfg.draft_free_fan_margin,
                                           root_fan=cfg.draft_free_root_fan,
                                           mode=cfg.slot_mode, gamma=cfg.slot_gamma,
                                           h_blk=h_prev, tau=cfg.slot_tau,
                                           tokgate=cfg.slot_tokgate)
                    g_round = None
                    draft_graph_size = len(drafts)
                elif cfg.draft == "free" and (cross or cfg.draft_free_fan_margin > 0):
                    drafts = cross_chain(p_prev, x[0, b0:bW], mask_id,
                                         cfg.draft_budget, L, cfg.temperature, noise_wd,
                                         fan_margin=cfg.draft_free_fan_margin,
                                         root_fan=cfg.draft_free_root_fan)
                    g_round = None
                    draft_graph_size = len(drafts)
                elif cfg.verify == "rs" and cfg.draft == "free":
                    # rs chain from the free drafter's stale-target marginals -- the trained-drafter
                    # negative result was measured under exact γ-state match only, so it
                    # is re-tested under the rule that fits a distribution-producing
                    # drafter before being accepted.
                    if cfg.draft == "med":
                        q_src = drafter.block_probs(h_prev)
                    else:
                        q_src = p_prev
                    drafts = sample_rs_chain(q_src, x[0, b0:b1], mask_id,
                                             cfg.draft_budget, generator=rs_gen)
                    g_round = None
                    draft_graph_size = len(drafts)
                elif cfg.draft == "free":
                    if cfg.draft == "med":
                        # zero model calls: the heads read hidden states the verification
                        # forward already produced last round.
                        src_probs = drafter.block_probs(h_prev)
                    else:
                        # built fresh each round from the live conditional; no model call
                        src_probs = p_prev
                    if cfg.draft == "rank":
                        ord_sc = drafter.order_scores(h_prev, p_prev, x[0, b0:b1], mask_id, L)
                    elif cfg.draft_free_order == "maxprob":
                        ord_sc = src_probs.max(dim=-1).values
                    else:
                        ord_sc = None
                    g_round = adaptive_graph(
                        src_probs, x[0, b0:b1], mask_id, order_scores=ord_sc,
                        n_nodes=max(cfg.draft_free_pool * cfg.draft_budget,
                                    cfg.draft_free_beam * cfg.draft_free_max_depth),
                        vocab_ranks=cfg.draft_free_vocab_ranks,
                        max_depth=cfg.draft_free_max_depth,
                        skip=cfg.draft_free_skip,
                        beam=cfg.draft_free_beam,
                        temperature=cfg.temperature, noise=noise_d)
                else:
                    g_round = graph
                    src_probs = p_prev
                if g_round is None:
                    pass
                else:
                    draft_graph_size = g_round.size
                    drafts = prune_and_materialise(
                    g_round, src_probs, x[0, b0:b1], mask_id, cfg.draft_budget,
                    temperature=cfg.temperature, noise=noise_d,
                    order_scores=ord_sc)
                t_draft = time.perf_counter() - t_d0

            # ---- 2. forward ----------------------------------------------------
            # With cache=on the KVs of the strict prefix [0, b0) are frozen while the block is
            # denoised (Fast-dLLM prefix caching).  They are (re)built by one call at the start of
            # each block, which is also that block's first real round -- no call is wasted.
            if cfg.cache == "on":
                want_prefix = max(0, b0 - shift)
                refresh = past is None or cache_len != want_prefix
                use_past = past
                past_len = cache_len if past is not None else 0
                chunk_start = past_len if refresh else want_prefix
                assert not (refresh and drafts), "cache refresh must happen on a draft-free round"
            else:
                chunk_start, use_past, past_len, refresh = 0, None, 0, False

            batched = cfg.draft_layout == "batch" and bool(drafts)
            if batched:
                assert cfg.cache == "off", "batch layout is the exact layout; it needs cache=off"
                input_ids = _assemble_batch(x, b0, b1, drafts)
                position_ids = torch.arange(0, total, device=device)
                main_len, cur_lo_main, bias = total, b0, None
            else:
                # exact only when the model is block-causal: later blocks cannot
                # influence the current one, so forwarding them is pure waste
                c_end = (bW if cross else b1) if (cfg.suffix == "skip" and bc) else None
                # refresh rounds are draft-free in every configuration (asserted above),
                # so both a padded and an unpadded run reach them with the same shape;
                # padding them would only inflate the prefill.
                n_dummy = (max(0, cfg.pad_pieces - len(drafts))
                           if (cfg.pad_pieces and not refresh) else 0)
                input_ids, position_ids, main_len = _assemble(
                    x, chunk_start, b0, b1, drafts, shift, device, chunk_end=c_end,
                    n_dummy=n_dummy, mask_id=mask_id, dummy_len=(W if cross else 0)
                )
                cur_lo_main = b0 - chunk_start
                # A block-causal model carries no implicit causality: handed no mask it
                # attends everywhere, so a draft-free round would silently let the current
                # block see future blocks.  Build the bias on every round, not just drafted
                # ones.
                bias = (
                    _attn_bias(past_len, main_len, len(drafts) + n_dummy, L, shift,
                               cur_lo_main, device, block_causal=bc, chunk_start=chunk_start,
                               piece_blocks=(2 if cross else 1))
                    if (drafts or n_dummy or bc)
                    else None
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwd0 = time.perf_counter()
            fkw = ({"want_hidden": True} if (cfg.want_hidden
                   or (cfg.draft == "slot" and getattr(drafter, "hid", 0))) else {})
            out = adapter.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_bias=bias,
                past=use_past,
                use_cache=(cfg.cache == "on" and refresh),
                **fkw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t_fwd0
            n_inv += 1
            n_positions += out.n_positions * int(input_ids.shape[0])
            n_verify_rows += int(input_ids.shape[0])
            n_draft_states += len(drafts)
            if cfg.cache == "on" and refresh:
                past = adapter.trim_cache(out.past, want_prefix)
                cache_len = want_prefix

            logits = out.logits
            probs_cur = adapter.block_probs(logits, cur_lo_main, cur_lo_main + W)
            draft_probs = []
            if batched:
                for d_i in range(len(drafts)):
                    draft_probs.append(adapter.block_probs(logits, b0, b1, row=d_i + 1))
            else:
                piece_len = W + shift if cross else L + shift
                for d_i in range(len(drafts)):
                    st = main_len + d_i * piece_len
                    draft_probs.append(adapter.block_probs_piece(logits, st,
                                                                 W if cross else L, shift))

            # ---- 3. commit + verify -------------------------------------------
            # ranking of the drafting source, used for the (position-rank, token-rank) stats
            src = p_prev if p_prev is not None else probs_cur
            src_is_prev = p_prev is not None
            depth = 0
            committed_records: List[dict] = []
            accepted_nodes: List[int] = []
            first_rejection_depth = None
            first_draft_accepted = None
            unconsumed = list(range(len(drafts)))
            probs_active = probs_cur
            # which draft piece `probs_active` was read off (None => the main chunk).  The
            # matched-shape reference has to be read at the same offset to be a control.
            active_draft_idx: Optional[int] = None
            masked_at_round_start = (x[0, b0:b1] == mask_id).clone()
            t_instr = 0.0

            rs_reject = False
            rs_slot_hits = rs_links = 0
            rs_link_log = []
            t_accept = t_resid = 0.0
            while True:
                link_idx = depth  # rs: which chain link this commit corresponds to
                rs_link = (cfg.verify == "rs" and bool(drafts) and link_idx < len(drafts))
                if rs_link:
                    # -- rejection-sampling verification, one chain link ---------------
                    d_rs = drafts[link_idx]
                    ps, pt = d_rs["pair"]
                    masked_now = x[0, b0:b1] == mask_id
                    t_a0 = time.perf_counter()
                    p_pair = commit_law_pair(probs_active, masked_now, ps, pt)
                    t_accept += time.perf_counter() - t_a0
                    a = min(1.0, p_pair / max(d_rs["q_pair"], 1e-45))
                    u = float(torch.rand((), device=device, generator=rs_gen))
                    rs_link_log.append({"lvl": link_idx + 1, "pi": p_pair,
                                        "q": d_rs["q_pair"], "acc": bool(u < a)})
                    if u < a:
                        rs_slot_hits += 1
                        rs_links += 1
                        sel = torch.tensor([ps], device=device)
                        toks = torch.tensor([pt], device=device, dtype=torch.long)
                        confs = probs_active[ps, pt].view(1)
                    else:
                        # committed pair is exactly p-distributed: residual of the
                        # maximal coupling, then the chain is dead
                        t_r0 = time.perf_counter()
                        p_law = commit_law(probs_active, masked_now)  # full law only here
                        if "q_slot_full" in d_rs:
                            # slot-RS: the proposal law is picker-slot x stale-marginal,
                            # NOT the stale commit law -- subtract what was proposed from
                            q_law = (d_rs["q_slot_full"].unsqueeze(1)
                                     * d_rs["src_probs"]) * masked_now.unsqueeze(1)
                        else:
                            q_law = commit_law(d_rs["src_probs"], masked_now)
                        resid = (p_law - q_law).clamp_min(0).reshape(-1)
                        z = float(resid.sum())
                        if z <= 1e-30:  # laws coincide numerically; any p-sample is exact
                            resid, z = p_law.reshape(-1), float(p_law.sum())
                        idx = int(torch.multinomial(resid / z, 1, generator=rs_gen))
                        sel = torch.tensor([idx // p_law.shape[1]], device=device)
                        toks = torch.tensor([idx % p_law.shape[1]], device=device,
                                            dtype=torch.long)
                        confs = probs_active[sel[0], toks[0]].view(1)
                        t_resid += time.perf_counter() - t_r0
                        rs_reject = True
                        rs_slot_hits += int(int(sel[0]) == ps)
                        rs_links += 1
                elif cfg.verify == "rs":
                    # cheap path: simulate the mechanism (identical in law) instead of
                    # materialising the full law
                    masked_now = x[0, b0:b1] == mask_id
                    sl, tk = rs_commit_sample(probs_active, masked_now, rs_gen)
                    sel = torch.tensor([sl], device=device)
                    toks = torch.tensor([tk], device=device, dtype=torch.long)
                    confs = probs_active[sl, tk].view(1)
                elif cross:
                    # commits never touch block k+1 while block k has masked slots; once k
                    # completes, conditionals come from the accepted replica's k+1 half
                    seg2 = not bool((x[0, b0:b1] == mask_id).any())
                    if seg2:
                        noise2 = noise_all.block(b1, bW) if noise_all is not None else None
                        sel, toks, confs = select_commit(probs_active[L:], x[0, b1:bW],
                                                         cfg, mask_id, noise2)
                        sel = sel + L
                    else:
                        sel, toks, confs = select_commit(probs_active[:L], x[0, b0:b1],
                                                         cfg, mask_id, noise_blk)
                else:
                    sel, toks, confs = select_commit(probs_active, x[0, b0:b1], cfg, mask_id, noise_blk)
                    if cfg.verify_greedy and cfg.temperature > 0.0 and drafts and unconsumed:
                        # greedy-match at T>0: accept a drafted state iff it equals the target's
                        # ARGMAX commit; otherwise fall back to the sampled commit above.  Lossy.
                        sel_g, toks_g, confs_g = select_commit(probs_active, x[0, b0:b1], cfg,
                                                               mask_id, None)
                        st_g = x[0, b0:bW].clone(); st_g.index_copy_(0, sel_g, toks_g)
                        if any(bool(torch.equal(drafts[d_i]["state"], st_g)) for d_i in unconsumed):
                            sel, toks, confs = sel_g, toks_g, confs_g

                # -- instrumentation: rank/confidence of each commit under the drafting
                # distribution.  Timed separately so it can be subtracted from the wall clock;
                # none of it feeds back into decoding.
                t_i0 = time.perf_counter()
                order, _ = _position_order(src, x[0, b0:bW] == mask_id)
                pos_rank_of = {int(p): r + 1 for r, p in enumerate(order.tolist())}
                sel_l, toks_l, confs_l = sel.tolist(), toks.tolist(), confs.tolist()
                ranks = (src[sel] > src[sel, toks].unsqueeze(-1)).sum(-1).add(1).tolist()
                src_conf = src[sel].max(-1).values.tolist()
                for k in range(len(sel_l)):
                    committed_records.append(
                        {
                            "block_pos": sel_l[k],
                            "token": toks_l[k],
                            "depth": depth,
                            "pos_rank": pos_rank_of.get(sel_l[k], -1),
                            "tok_rank": ranks[k],
                            "conf_commit": confs_l[k],
                            "conf_src": src_conf[k],
                        }
                    )
                t_instr += time.perf_counter() - t_i0

                if cfg.audit == "conditional":
                    t_a0 = time.perf_counter()
                    # Apply this round's commits one slot at a time to a scratch sequence and,
                    # before each, ask the *reference* decoder (uncached, draft-free, full
                    # bidirectional forward on the true state) what conditional it would have
                    # used for that slot.  Slot i=0 isolates the conditioning error of the
                    # cache/draft layers; i>0 additionally exposes what multi-slot commits give
                    # up by drawing every slot from its own marginal instead of the joint.
                    scratch = x.clone()
                    order_by_conf = torch.argsort(confs, descending=True, stable=True).tolist()
                    for i_step, k in enumerate(order_by_conf):
                        pos, tok = int(sel[k].item()), int(toks[k].item())
                        ref_in = (
                            scratch if cfg.audit_ref_batch <= 1
                            else scratch.repeat(cfg.audit_ref_batch, 1)
                        )
                        # A block-causal model handed no bias attends everywhere, which
                        # would make the "reference" a full-bidirectional forward -- a
                        # different model, not a reference.  Supply the same block-causal
                        # structure the true decoder uses.
                        ref_bias = (
                            _block_causal_block(0, total, 0, total, L, device)
                            .view(1, 1, total, total)
                            if bc else None
                        )
                        ref = adapter.forward(
                            ref_in, torch.arange(0, total, device=device), ref_bias,
                            None, False
                        )
                        ref_p = adapter.block_probs(ref.logits, b0 + pos, b0 + pos + 1)[0]
                        used_p = probs_active[pos]

                        # ---- shape/offset/content-matched reference ----------------
                        # Same forward the cell built, but with every draft piece present
                        # and no cache, so the current block can be read either off the
                        # main chunk or off draft piece `active_draft_idx` -- at the same
                        # offset, in the same-length tensor, alongside the same neighbours.
                        # On a block-causal model both reads are mathematically equal to
                        # `ref_p`, so any difference is float noise and nothing else.
                        m_main = m_row = None
                        if cfg.audit_matched_shape and drafts:
                            piece_len_m = L + shift
                            m_pieces = [scratch]
                            m_pos = [torch.arange(0, total, device=device)]
                            for d in drafts:
                                m_pieces.append(
                                    torch.cat([scratch[0, b0 - shift : b0], d["state"]]).view(1, -1)
                                )
                                m_pos.append(torch.arange(b0 - shift, b1, device=device))
                            m_bias = _attn_bias(
                                0, total, len(drafts), L, shift, b0, device,
                                block_causal=bc, chunk_start=0)
                            m_out = adapter.forward(
                                torch.cat(m_pieces, dim=1), torch.cat(m_pos), m_bias,
                                None, False)
                            m_main = adapter.block_probs(m_out.logits, b0, b1)[pos]
                            if active_draft_idx is not None:
                                m_row = adapter.block_probs_piece(
                                    m_out.logits,
                                    total + active_draft_idx * piece_len_m, L, shift)[pos]
                        ref_m = m_row if m_row is not None else m_main

                        def _tv(a, c):
                            if a is None or c is None:
                                return None
                            return float(total_variation(a[None], c[None]).item())

                        audit_records.append(
                            {
                                "sample": sample_id, "round": round_global, "depth": depth,
                                "slot_in_commit": i_step, "commit_width": int(sel.numel()),
                                "block": b, "block_pos": pos,
                                "n_drafts_round": len(drafts),
                                "active_draft_idx": active_draft_idx,
                                # cell vs a reference matched to the cell's own shape/offset
                                "tv_matched": _tv(used_p, ref_m),
                                # the matched floor: identical maths, different tensor shape
                                "tv_floor_pad_main": _tv(m_main, ref_p),
                                "tv_floor_pad_row": _tv(m_row, ref_p),
                                # offset alone -- both reads come out of the SAME forward
                                "tv_floor_row_vs_main": _tv(m_row, m_main),
                                "tv": float(total_variation(used_p[None], ref_p[None]).item()),
                                "kl_ref_used": float(kl_divergence(ref_p[None], used_p[None]).item()),
                                "p_used_of_committed": float(used_p[tok].item()),
                                "p_ref_of_committed": float(ref_p[tok].item()),
                                "ref_argmax_agrees": bool(int(ref_p.argmax()) == int(used_p.argmax())),
                                "logit_margin_ref": float(
                                    (ref_p.topk(2).values[0] - ref_p.topk(2).values[1]).item()
                                ),
                            }
                        )
                        scratch[0, b0 + pos] = tok
                    t_instr += time.perf_counter() - t_a0

                x[0, b0:bW].index_copy_(0, sel, toks)
                commits_total += int(sel.numel())
                depth += 1

                win_done = not bool((x[0, b0:bW] == mask_id).any())
                if win_done or (not cross and not bool((x[0, b0:b1] == mask_id).any())):
                    break
                if cfg.verify == "rs":
                    if rs_reject or not rs_link:
                        if drafts and first_rejection_depth is None:
                            first_rejection_depth = depth
                        if drafts and first_draft_accepted is None:
                            first_draft_accepted = False
                        break
                    # accepted link: the drafted state IS the true state by construction,
                    # so its precomputed row is the conditional for the next link
                    if first_draft_accepted is None:
                        first_draft_accepted = True
                    accepted_nodes.append(drafts[link_idx]["node"])
                    probs_active = draft_probs[link_idx]
                    active_draft_idx = link_idx
                    continue
                # try to accept a draft whose state now equals the true state (the full
                # window under cross_block: crossing requires the completed-k copy to match)
                state_now = x[0, b0:bW]
                match = None
                for d_i in unconsumed:
                    if bool(torch.equal(drafts[d_i]["state"], state_now)):
                        match = d_i
                        break
                if drafts and first_draft_accepted is None:
                    first_draft_accepted = bool(match is not None and match == 0)
                if match is None:
                    if drafts:
                        first_rejection_depth = depth
                    break
                accepted_nodes.append(drafts[match]["node"])
                unconsumed.remove(match)
                probs_active = draft_probs[match]
                active_draft_idx = match
                # the accepted draft's distribution is exact for the new state
            n_from_draft = sum(1 for r in committed_records if r["depth"] > 0)
            rec = {
                "sample": sample_id,
                "round": round_global,
                "round_in_block": round_in_block,
                "block": b,
                "gen_pos_committed": commits_total,
                "masked_in_block_after": int((x[0, b0:b1] == mask_id).sum().item()),
                "masked_in_gen_after": int((x[0, P : P + cfg.gen_length] == mask_id).sum().item()),
                "drafts_verified": len(drafts),
                # drafter cost columns.  D_free and the calibrated graph both cost zero
                # model calls; a drafter that is a model will set this nonzero.
                "drafter_invocations": n_aux_fwd,
                "drafter_nodes_proposed": draft_graph_size,
                "drafter_graph_depth": (max((d["level"] for d in drafts), default=0)),
                "t_draft_s": t_draft,
                "draft_nodes": [d["node"] for d in drafts],
                "draft_levels": [d["level"] for d in drafts],
                "draft_scores": [round(d["score"], 5) for d in drafts],
                "accepted": len(accepted_nodes),
                "accepted_nodes": accepted_nodes,
                "first_draft_accepted": first_draft_accepted,
                "first_rejection_depth": first_rejection_depth,
                "n_committed": len(committed_records),
                "n_committed_from_draft": n_from_draft,
                "crossed_commits": sum(1 for r in committed_records if r["block_pos"] >= L),
                # rs decomposition: slot-level hits vs joint accepts, per drafted round
                "rs_links_tried": rs_links,
                "rs_slot_hits": rs_slot_hits,
                "rs_link_log": rs_link_log,
                "t_rs_accept_s": t_accept,
                "t_rs_resid_s": t_resid,
                "commits": committed_records,
                # how many commits had landed since the forward pass that produced the logits
                # these drafts were built from (== all commits of the previous round); the
                # drafting *state* itself is always exactly one commit behind.
                "src_staleness_commits": p_prev_commits_ago if src_is_prev else None,
                "model_invocations": 1,
                "positions_forwarded": out.n_positions,
                "verify_rows": 1 + len(drafts),
                "key_len": out.key_len,
                "cache_refresh": bool(refresh),
                "t_forward_s": t_fwd,
                "t_instr_s": t_instr,
            }
            t_probe0 = time.perf_counter()
            if probe is not None:
                probe(
                    {
                        "record": rec,
                        "src_probs": src,
                        # hidden rows from the SAME forward as src_probs -- h_prev
                        # when src is the stale table, this round's rows otherwise. A
                        # drafter's features must be stale-by-construction, so the dump
                        # must be too.
                        "hidden": ((h_prev if src_is_prev else
                                    out.hidden[0, cur_lo_main : cur_lo_main + L])
                                   if cfg.want_hidden else None),
                        "masked_before_round": masked_at_round_start,
                        "commits": committed_records,
                        "block": b,
                        "round_in_block": round_in_block,
                    }
                )
            t_instr += time.perf_counter() - t_probe0
            # bookkeeping for the next round
            if (cfg.want_hidden
                    or (cfg.draft == "slot" and getattr(drafter, "hid", 0))):
                # hidden rows must come from the SAME row as p_prev (probs_active). The
                # dump convention is `hidden index = prob-producing index + logit_shift`
                # (row 0, no -shift), so the accepted-draft slices reproduce it exactly:
                # batched draft probs read logits[m+1, b0-s:b1-s] -> hidden [m+1, b0:b1];
                # piece probs read logits[0, st:st+L] -> hidden [0, st+s : st+s+L].
                if active_draft_idx is None:
                    h_prev = out.hidden[0, cur_lo_main : cur_lo_main + L]
                elif batched:
                    h_prev = out.hidden[active_draft_idx + 1, b0:b1]
                else:
                    st = main_len + active_draft_idx * (L + shift)
                    h_prev = out.hidden[0, st + shift : st + shift + L]
            if cfg.draft == 'oracle':
                drafter.advance(len(committed_records))
            p_prev = probs_active
            p_prev_commits_ago = len(committed_records)
            rec["t_round_s"] = time.perf_counter() - t_round0
            rec["t_instr_s"] = t_instr
            instr_total += t_instr
            rounds.append(rec)
            round_global += 1
            round_in_block += 1

    wall = time.perf_counter() - t0
    # block-causal runs pad `total` up to a block boundary, so clamp to the requested length
    gen_ids = x[0, P : P + cfg.gen_length].tolist()
    # wall_s      : everything, including the per-round statistics this repo logs
    # wall_s_net  : with that instrumentation subtracted -- the number to quote as a token
    #               rate, since the paper's implementation logs none of it
    eos_id = int(getattr(adapter, "eos_id", -1))
    eos_pos = gen_ids.index(eos_id) if eos_id in gen_ids else -1
    return {
        "sample": sample_id,
        "tokens": gen_ids,
        "text": adapter.decode(gen_ids),
        "eos_pos": eos_pos,       # first EOS in the generation; -1 = ran to length
        "nfe": n_inv,
        "positions_forwarded": n_positions,
        "verify_rows": n_verify_rows,
        "draft_states_verified": n_draft_states,
        # drafter-side cost, kept as its own column: 0 for the training-free drafters,
        # nonzero for any drafter that is itself a model.  Never folded into the target's.
        "drafter_invocations": (drafter.invocations if drafter is not None else 0),
        "drafter_positions": (drafter.positions if drafter is not None else 0),
        "audit": audit_records,
        "wall_s": wall,
        "wall_s_net": wall - instr_total,
        "instr_s": instr_total,
        "tokens_generated": cfg.gen_length,
        "rounds": rounds,
    }


