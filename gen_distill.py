#!/usr/bin/env python
"""Self-distillation trajectories from the 8B target, for the trained drafters.

What a trained drafter has to learn is *what the target will commit next, given the state
the decoder is actually in*.  So the data is the decoder's own trajectory: every
intermediate block state it visits, and which (slot, token) it committed from there.

**What is stored, and what deliberately is not.**  Not hidden states: 4096 floats x 32 slots
x ~50 rounds x 5000 samples is ~65 GB and would dominate the whole project's storage for
something the frozen backbone can regenerate in a forward pass.  Not full marginals: 152k
vocab x 32 slots x 250k rounds is worse.  Only the trajectory -- prompt, committed tokens,
and the per-round (state -> commit) structure -- which is small, and from which training
recomputes whatever representation it needs by re-forwarding the frozen target.

That choice is what makes `med` trainable in hours rather than days, and it is why the
storage estimate ("start at 2-5k trajectories") is comfortable.

Contamination: prompts must come from a pool disjoint from MBPP / GSM8K / HumanEval.  This
script records the pool identity and every prompt's hash so the overlap check in
`check_contamination.py` can run against the actual training set rather than against an
intention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spot.decode import DecodeConfig, generate  # noqa: E402
from spot.models import load_adapter  # noqa: E402
from spot.runner import provenance  # noqa: E402
from spot.tasks import load_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar")
    ap.add_argument("--pool", default="codealpaca",
                    help="prompt pool; MUST be disjoint from mbpp/gsm8k/humaneval")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0, help="resume offset, for multi-job runs")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=9600,
                    help="stop cleanly before the job's wall limit; writes what it has")
    ap.add_argument("--topk", type=int, default=8,
                    help="marginals stored per masked slot. 0 = trajectory only (the old "
                         "format). Needed to distil a model that PREDICTS the target's "
                         "next-round distribution rather than re-ranking the stale one.")
    ap.add_argument("--hid-dump", default="",
                    help="directory for per-trajectory sidecar .pt files holding the "
                         "STALE pre-head hidden rows of all block positions, fp16, one "
                         "tensor per round, aligned 1:1 with the jsonl round list. This is "
                         "the input the two-head proposer trains on; storage is why it is "
                         "a flag and not the default (~17 MB/traj at b32 on the 8B).")
    ap.add_argument("--draft", default="none", choices=["none", "slot"],
                    help="train-time testing: generate WITH a drafter active, so "
                         "the recorded tables come from the DEPLOY distribution -- "
                         "draft-row logits, multi-commit staleness. Lossless decoding "
                         "commits the identical sequence, so labels stay valid.")
    ap.add_argument("--slot-ckpt", default="")
    ap.add_argument("--draft-budget", type=int, default=8)
    ap.add_argument("--cache", default="on", choices=["on", "off"],
                    help="MUST match the eval config: LLaDA2.0-mini rejects the cached "
                         "attention-mask shape and is always run uncached.")
    ap.add_argument("--suffix", default="skip", choices=["skip", "keep"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ad = load_adapter(args.model)
    samples = load_task(args.pool, limit=args.limit + args.start)[args.start:]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The decoder configuration the drafter will be deployed under.  Training on states from
    # a *different* configuration would reintroduce exactly the train/test mismatch P3 exists
    # to remove, so this must track the eval config.
    cfg = DecodeConfig(cache=args.cache, unmasking="single", draft=args.draft,
                       suffix=args.suffix, slot_mode="ddtree_b", slot_tokgate=0.6,
                       draft_budget=args.draft_budget, pad_pieces=-1,
                       gen_length=args.gen_length, block_length=args.block_length,
                       temperature=args.temperature, seed=args.seed,
                       want_hidden=bool(args.hid_dump))
    drafter = None
    if args.draft == "slot":
        from spot.drafters import SlotDrafter

        drafter = SlotDrafter(args.slot_ckpt, ad)
    if args.hid_dump:
        os.makedirs(args.hid_dump, exist_ok=True)

    meta = {"pool": args.pool, "model": args.model, "decode": dict(cfg.__dict__),
            "start": args.start, "requested": args.limit,
            "provenance": provenance(None)}
    f = open(args.out, "w")
    f.write(json.dumps({"_meta": meta}) + "\n")

    t0 = time.time()
    n = 0
    for s in samples:
        if time.time() - t0 > args.time_budget_s:
            print(f"== time budget reached after {n} trajectories", flush=True)
            break
        prompt = ad.encode_raw(s.raw_prompt) if s.raw_prompt else ad.encode_prompt(s.messages)
        marg = []
        hids = []
        if args.topk > 0:
            # capture the target's own block marginals at every round: the distillation
            # target. Stored top-k per MASKED slot only -- full 152k-way distributions over
            # 387k rounds would be terabytes, and the tail carries no drafting signal
            # (measured: j=1 accounts for 90.1% of commits, j<=8 for 99.7%).
            def _probe(ev, _m=marg, _h=hids, _k=args.topk, _dump=bool(args.hid_dump)):
                sp = ev["src_probs"]
                mk = ev["masked_before_round"]
                idx = mk.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    _m.append(None)
                    if _dump:
                        _h.append(None)
                    return
                v, i = sp[idx].topk(_k, dim=-1)
                _m.append({"slots": idx.tolist(),
                           "tok": i.to(torch.int32).cpu().tolist(),
                           "p": [[round(x, 6) for x in r] for r in v.float().cpu().tolist()]})
                if _dump:
                    # ALL block rows, not just masked ones: the filled positions are the
                    # bidirectional context the proposer conditions on.
                    _h.append(ev["hidden"].detach().to(torch.float16).cpu())
            r = generate(ad, prompt, cfg, graph=None, sample_id=s.sample_id, probe=_probe, drafter=drafter)
        else:
            r = generate(ad, prompt, cfg, graph=None, sample_id=s.sample_id)
        # per round: which block, the mask pattern before the commit, and what was committed.
        # `block_pos` is a slot index inside the block; the committed token is the target's
        # own draw, which is the label.
        rounds = [{"block": rd["block"],
                   "round_in_block": rd["round_in_block"],
                   "masked_after": rd["masked_in_block_after"],
                   "commits": [[c["block_pos"], c["token"], c["depth"]] for c in rd["commits"]],
                   "marg": (marg[k] if k < len(marg) else None)}
                  for k, rd in enumerate(r["rounds"])]
        f.write(json.dumps({
            "sample": s.sample_id,
            "prompt_sha": hashlib.sha256(
                (s.raw_prompt or json.dumps(s.messages)).encode()).hexdigest()[:16],
            "prompt_ids": prompt[0].tolist(),
            "tokens": r["tokens"],
            "nfe": r["nfe"],
            "rounds": rounds,
        }) + "\n")
        if args.hid_dump:
            torch.save({"sample": s.sample_id, "rounds": hids},
                       os.path.join(args.hid_dump, f"{s.sample_id}.pt"))
        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            print(f"== {n} trajectories, {el:.0f}s, {el/n:.2f}s/traj", flush=True)
    f.close()
    print(f"== wrote {n} trajectories to {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
