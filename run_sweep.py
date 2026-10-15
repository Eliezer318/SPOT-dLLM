#!/usr/bin/env python
"""Draft-budget sweep for block-diffusion speculative decoding.

Runs one (task, drafter, budget) grid on a single GPU and writes, per budget and seed, a
run directory with the generations, scores, per-round acceptance log and a
distribution-preserving verdict derived from the configuration. Every arm carries its own
draft-free (D=0) baseline in the same invocation on the same card, and reports realised
cost (rounds, verified draft states, forwarded positions) rather than the nominal budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spot.decode import DecodeConfig, generate  # noqa: E402
from spot.drafters import (  # noqa: E402
                             SiblingChainDrafter, SlotDrafter)
from spot.graphs import DraftGraph  # noqa: E402
from spot.models import load_adapter  # noqa: E402
from spot.runner import provenance  # noqa: E402
from spot.tasks import load_task, score_task  # noqa: E402


def _check_graph_matches_model(graph_path: str, model: str, draft: str) -> None:
    """Refuse a graph calibrated on a different target than the one being run.

    `--graph` defaults to the LLaDA-8B graph, so a command that omits it silently drafts
    LLaDA-8B's calibrated formulas against SDAR or LLaDA2.0-mini, which reads as a weak
    baseline rather than as a misconfiguration. Only `draft=spiffy` resolves a calibrated
    graph, so only that arm is gated.
    """
    if draft != "spiffy":
        return
    base = os.path.basename(graph_path)
    family = ("llada2" if model.startswith("llada2") else
              "sdar" if model.startswith("sdar") else
              "dream" if model.startswith("dream") else "llada")
    got = ("llada2" if base.startswith("llada2") else
           "sdar" if base.startswith("sdar") else
           "dream" if base.startswith("dream") else "llada")
    if got != family:
        raise SystemExit(
            f"--graph {base} is calibrated for '{got}' but --model is '{model}'.\n"
            f"Pass the matching graph from graphs/ explicitly (e.g. graphs/sdar8b_b32_D24.json)."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdar")
    ap.add_argument("--task", default="mbpp")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many prompts before --limit (augmentation lanes: the "
                         "prompt range [offset, offset+limit) decodes identically whether or "
                         "not the earlier prompts were run, so cells can be merged later)")
    ap.add_argument("--budgets", default="0,1,2,3,5,8")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--extra-seed-budgets", default="0,1", help="budgets to repeat over all seeds")
    ap.add_argument("--gen-length", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--graph", default="graphs/sdar8b_b32_D24.json",
                    help="calibrated draft graph for the Spiffy baseline (--draft spiffy); must match --model")
    ap.add_argument("--layout", default="batch", choices=["batch", "concat"])
    ap.add_argument("--cache", default="off", choices=["on", "off"])
    ap.add_argument("--unmasking", default="single", choices=["single", "threshold"])
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="confidence bar for --unmasking threshold (commit every position above it)")
    ap.add_argument("--draft", default="slot", choices=["spiffy","free","slot","sibling"],
                    help="draft-graph source for budget>0 arms")
    ap.add_argument("--free-vocab-ranks", type=int, default=3)
    ap.add_argument("--free-max-depth", type=int, default=6)
    ap.add_argument("--free-skip", type=int, default=1)
    ap.add_argument("--free-order", default="maxprob", choices=["maxprob","sampler"])
    ap.add_argument("--free-beam", type=int, default=8)
    ap.add_argument("--suffix", default="keep", choices=["keep","skip"])
    ap.add_argument("--verify", default="match", choices=["match","rs"])
    ap.add_argument("--draft-noise", default="coupled", choices=["coupled","none"],
                    help="T>0: give the drafter the verifier's Gumbel noise (coupled) or none (argmax proposals)")
    ap.add_argument("--verify-greedy", action="store_true",
                    help="T>0: accept drafted states by ARGMAX match (lossy greedy/sampled mixture; temperature-sweep arm)")
    ap.add_argument("--slot-gamma", type=float, default=0.0)
    ap.add_argument("--slot-tokgate", type=float, default=0.0,
                    help="branch the 2nd token at slots whose stale top-1 < gate")
    ap.add_argument("--slot-refresh", default="off", choices=["off", "full"],
                    help="refresh the drafting table with --aux-model's forward")
    ap.add_argument("--aux-cache", default="on", choices=["on", "off"],
                    help="lossless prefix KV cache for the sibling drafter")
    ap.add_argument("--aux-device", default="cuda",
                    help="device for the sibling/aux drafter (cuda:1 = 2-GPU fairness arm)")
    ap.add_argument("--aux-model", default="",
                    help="adapter name of the aux refresher (e.g. sdar17)")
    ap.add_argument("--slot-tau", type=float, default=1.0)
    ap.add_argument("--cross-block", default="off", choices=["off","on"])
    ap.add_argument("--free-fan-margin", type=float, default=0.0)
    ap.add_argument("--free-root-fan", type=int, default=1)
    ap.add_argument("--pad-pieces", type=int, default=0,
                    help="pad non-refresh forwards to a constant piece count; 0=off; -1=pad to each arm's budget. "
                         "Applied to drafted arms only -- the D=0 baseline stays the "
                         "honest unpadded deployment baseline.")
    ap.add_argument("--slot-ckpt", default="drafters/sdar8b_b32.pt")
    ap.add_argument("--slot-mode", default="ddtree_b", choices=["chain", "bestfirst", "graph", "ddtree", "ddtree_b", "ddtree_bf", "pl", "purechain", "conftree"])
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    _check_graph_matches_model(args.graph, args.model, args.draft)

    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    extra = {int(b) for b in args.extra_seed_budgets.split(",") if b != ""}
    ad = load_adapter(args.model)
    if False:
        pass
    elif args.draft == "slot" and args.slot_mode == "conftree":
        from spot.drafters import ConfTreeDrafter
        drafter = ConfTreeDrafter()   # training-free; no ckpt
    elif args.draft == "slot":
        drafter = SlotDrafter(args.slot_ckpt, ad)
        print(f"== slot picker: m_top={drafter.m_top}, k_steps={drafter.k_steps}, "
              f"trained block={drafter.trained_block}, val_joint={drafter.val_joint}")
    elif args.draft == "sibling":
        # SimSD-core -- the 1.7B sibling drafts its own chain, we verify exactly
        # --aux-device cuda:1 = SimSD's own deployment shape (sibling on its own GPU);
        # forwards stay serialized -- overlap is their pipeline optimization, cited not run
        drafter = SiblingChainDrafter(load_adapter(args.aux_model or "sdar17",
                                                   device=args.aux_device),
                                      use_cache=(args.aux_cache == "on"))
    else:
        drafter = None
    aux = None
    if args.aux_model:
        # cheap sibling model whose forward un-stales the drafting table. Same
        # tokenizer family is a hard requirement -- proposals are token ids.
        aux = load_adapter(args.aux_model)
        assert aux.mask_id == ad.mask_id, (aux.mask_id, ad.mask_id)
    graph = DraftGraph.load(args.graph) if args.draft == "spiffy" else None
    samples = load_task(args.task, limit=args.limit, offset=args.offset)
    os.makedirs(args.out_root, exist_ok=True)

    index = []
    for budget in budgets:
        for seed in (seeds if (budget in extra or len(seeds) == 1) else seeds[:1]):
            name = f"D{budget}_seed{seed}"
            out_dir = os.path.join(args.out_root, name)
            if os.path.exists(os.path.join(out_dir, "summary.json")):
                print(f"== skip {name}", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            cfg = DecodeConfig(
                cache=args.cache, unmasking=args.unmasking, threshold=args.threshold,
                draft=(args.draft if budget > 0 else "none"),
                draft_free_vocab_ranks=args.free_vocab_ranks,
                draft_free_max_depth=args.free_max_depth,
                draft_free_skip=args.free_skip,
                draft_free_order=args.free_order,
                draft_free_beam=args.free_beam,
                slot_mode=args.slot_mode,
                suffix=args.suffix, verify=args.verify, cross_block=args.cross_block, draft_free_fan_margin=args.free_fan_margin, draft_free_root_fan=args.free_root_fan,
                pad_pieces=((budget if args.pad_pieces < 0 else args.pad_pieces)
                            if budget > 0 else 0),
                draft_layout=args.layout, draft_budget=max(budget, 1),
                gen_length=args.gen_length, block_length=args.block_length,
                temperature=args.temperature, seed=seed, slot_gamma=args.slot_gamma, slot_tau=args.slot_tau, slot_tokgate=args.slot_tokgate, slot_refresh=args.slot_refresh, draft_noise=args.draft_noise, verify_greedy=args.verify_greedy,
            )
            preserving, why = cfg.is_distribution_preserving(
                getattr(ad, "block_causal", False))
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump({"budget": budget, "seed": seed, "decode": dict(cfg.__dict__),
                           "distribution_preserving": preserving, "reasons_not": why,
                           "provenance": provenance(args.graph)}, f, indent=2)

            rf = open(os.path.join(out_dir, "rounds.jsonl"), "w")
            sf = open(os.path.join(out_dir, "samples.jsonl"), "w")
            tot = dict(nfe=0, rows=0, pos=0, wall=0.0, score=0.0, n=0,
                       states=0, accepted=0, rounds=0, drafted_rounds=0)
            t0 = time.time()
            for s in samples:
                prompt = ad.encode_raw(s.raw_prompt) if s.raw_prompt else ad.encode_prompt(s.messages)
                r = generate(ad, prompt, cfg, graph=(graph if budget > 0 else None),
                             sample_id=s.sample_id, drafter=drafter, aux=aux)
                for rec in r["rounds"]:
                    rf.write(json.dumps(rec) + "\n")
                sc = score_task(args.task, s, r["text"])
                acc = sum(x["accepted"] for x in r["rounds"])
                dr = sum(1 for x in r["rounds"] if x["drafts_verified"] > 0)
                sf.write(json.dumps({
                    "sample": s.sample_id, "text": r["text"], "nfe": r["nfe"], "rows": r["verify_rows"],
                    "eos_pos": r.get("eos_pos", -1),
                    "positions": r["positions_forwarded"], "wall_s_net": r["wall_s_net"],
                    "states_verified": r["draft_states_verified"], "accepted": acc, "score": sc,
                }) + "\n")
                tot["nfe"] += r["nfe"]; tot["rows"] += r["verify_rows"]; tot["pos"] += r["positions_forwarded"]
                tot["wall"] += r["wall_s_net"]; tot["score"] += sc; tot["n"] += 1
                tot["states"] += r["draft_states_verified"]; tot["accepted"] += acc
                tot["rounds"] += len(r["rounds"]); tot["drafted_rounds"] += dr
            rf.close(); sf.close()
            n = tot["n"]
            summary = {
                "name": name, "budget_nominal": budget, "seed": seed,
                "distribution_preserving": preserving,
                "n_samples": n,
                "nfe_per_sample": tot["nfe"] / n,
                "rows_per_sample": tot["rows"] / n,
                "positions_per_sample": tot["pos"] / n,
                "states_verified_per_sample": tot["states"] / n,
                "states_verified_per_drafted_round": tot["states"] / max(tot["drafted_rounds"], 1),
                "accepted_per_sample": tot["accepted"] / n,
                "accepted_per_drafted_round": tot["accepted"] / max(tot["drafted_rounds"], 1),
                "wall_s_per_sample": tot["wall"] / n,
                "tps": (args.gen_length * n) / tot["wall"],
                "accuracy": tot["score"] / n,
                "seconds": time.time() - t0,
            }
            # atomic write: two jobs racing on the same out-root (an accidental duplicate
            # submission, ) otherwise interleave into an unparseable file that
            # looks like a finished cell to the skip check.
            tmp_sum = os.path.join(out_dir, "summary.json.tmp")
            with open(tmp_sum, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp_sum, os.path.join(out_dir, "summary.json"))
            index.append(summary)
            with open(os.path.join(args.out_root, "index.json"), "w") as f:
                json.dump(index, f, indent=2)
            print("== " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
