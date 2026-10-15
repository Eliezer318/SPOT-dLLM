#!/usr/bin/env python
"""Train `D_slot` -- the position-only drafter -- on distilled decoder trajectories.

One training example = one drafting decision the decoder actually faced:

  input   the target's own top-M table over the slots still masked *after* this round's
          commit -- exactly the (N, M) table the decoder already holds, one round stale;
  label   the slots the target commits over the next K rounds of the same block, in order.

The control it must beat is the free drafter, which ranks those same slots by their stale
confidence.  Both see identical information, so the per-step accuracy gap is attributable
to the ranking rule and nothing else -- reported here before any GPU sweep is spent.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, sys, time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spot.pickers import build_picker
from spot.slotpicker import SlotPicker, slot_features  # noqa: E402


def build(paths, m_top, k_steps, limit_rounds=0):
    """jsonl trajectories -> list of (p_top, slot_idx, top_tok, round_in_block, labels)."""
    ex, block_len = [], None
    for path in paths:
        with open(path) as f:
            meta = json.loads(f.readline())
            block_len = int(meta["_meta"]["decode"]["block_length"])
            for line in f:
                rounds = json.loads(line)["rounds"]
                for r, rd in enumerate(rounds):
                    mg = rd.get("marg")
                    if not mg or not rd["commits"]:
                        continue
                    committed = rd["commits"][0][0]
                    slots = mg["slots"]
                    keep = [i for i, s in enumerate(slots) if s != committed]
                    if len(keep) < 2:
                        continue
                    # labels: the next K commits, same block, restricted to slots we kept
                    cand = {slots[i]: j for j, i in enumerate(keep)}
                    labels = []
                    for rr in range(r + 1, min(r + 1 + k_steps, len(rounds))):
                        nxt = rounds[rr]
                        if nxt["block"] != rd["block"] or not nxt["commits"]:
                            break
                        pos = nxt["commits"][0][0]
                        if pos not in cand:
                            break
                        labels.append(cand[pos])
                    if not labels:
                        continue
                    p = torch.tensor([mg["p"][i][:m_top] for i in keep], dtype=torch.float32)
                    if p.shape[1] < m_top:  # data stored fewer ranks than requested
                        p = torch.cat([p, p.new_zeros(p.shape[0], m_top - p.shape[1])], 1)
                    ex.append((p,
                               torch.tensor([slots[i] for i in keep]),
                               torch.tensor([mg["tok"][i][0] for i in keep]),
                               rd["round_in_block"], labels))
                    if limit_rounds and len(ex) >= limit_rounds:
                        return ex, block_len
    return ex, block_len


def build_ctx(paths, m_top, k_steps, limit_rounds=0):
    """Whole-block examples: the written prefix plus the candidates.

    Reconstructs what the block actually CONTAINED at each drafting decision — which slots
    were already committed and with which tokens — which the marginal-only builder throws
    away. `marg["slots"]` lists the slots still masked before the round, so the complement
    within the block is the written prefix; tokens come from the recorded commits.
    """
    ex, block_len = [], None
    for path in paths:
        with open(path) as f:
            meta = json.loads(f.readline())
            block_len = int(meta["_meta"]["decode"]["block_length"])
            for line in f:
                rounds = json.loads(line)["rounds"]
                filled = {}           # block_pos -> token, for the CURRENT block
                cur_block = None
                for r, rd in enumerate(rounds):
                    if rd["block"] != cur_block:
                        cur_block, filled = rd["block"], {}
                    mg = rd.get("marg")
                    if not mg or not rd["commits"]:
                        continue
                    pos_r, tok_r = rd["commits"][0][0], rd["commits"][0][1]
                    state = dict(filled)          # before this round's own commit
                    state[pos_r] = tok_r          # the drafter runs after it
                    labels = []
                    for rr in range(r + 1, min(r + 1 + k_steps, len(rounds))):
                        nxt = rounds[rr]
                        if nxt["block"] != rd["block"] or not nxt["commits"]:
                            break
                        np_ = nxt["commits"][0][0]
                        if np_ in state:
                            break
                        labels.append(np_)
                    filled[pos_r] = tok_r
                    if not labels:
                        continue
                    # full-block rows: probs for masked slots, zeros (ignored) for written
                    p = torch.zeros(block_len, m_top)
                    tk = torch.zeros(block_len, dtype=torch.long)
                    for i, sl in enumerate(mg["slots"]):
                        if sl in state:
                            continue
                        row = mg["p"][i][:m_top]
                        p[sl, :len(row)] = torch.tensor(row)
                        tk[sl] = mg["tok"][i][0]
                    fill_mask = torch.zeros(block_len, dtype=torch.bool)
                    for sl, t in state.items():
                        fill_mask[sl] = True
                        tk[sl] = t
                    ex.append((p, fill_mask, tk, rd["round_in_block"], labels))
                    if limit_rounds and len(ex) >= limit_rounds:
                        return ex, block_len
    return ex, block_len


def collate_ctx(batch, block_len, k_steps, tok_mod):
    from spot.slotpicker import block_features
    B = len(batch)
    feats, toks = [], []
    lab = torch.full((B, k_steps), -100, dtype=torch.long)
    fill = torch.zeros(B, block_len, dtype=torch.bool)
    for b, (p, fm, tk, rib, labels) in enumerate(batch):
        feats.append(block_features(p, fm, block_len, int((~fm).sum()), int(fm.sum())))
        toks.append(tk % tok_mod)
        fill[b] = fm
        for k, y in enumerate(labels[:k_steps]):
            lab[b, k] = y
    return torch.stack(feats), torch.stack(toks), fill, lab


def build_pc(paths, m_top, k_steps, limit_rounds=0):
    """Prefix-conditional examples: one per (round, depth) along the TRUE path.

    The chain model saw only the round-start state, so every branch of a tree got the same
    prediction. Here each example carries the slots already filled on its path, which is
    exactly what a tree node knows at expansion time -- train and inference now ask the
    same question.
    """
    ex, block_len = [], None
    for path in paths:
        with open(path) as f:
            meta = json.loads(f.readline())
            block_len = int(meta["_meta"]["decode"]["block_length"])
            for line in f:
                rounds = json.loads(line)["rounds"]
                for r, rd in enumerate(rounds):
                    mg = rd.get("marg")
                    if not mg or not rd["commits"]:
                        continue
                    committed = rd["commits"][0][0]
                    keep = [i for i, s in enumerate(mg["slots"]) if s != committed]
                    if len(keep) < 2:
                        continue
                    cand = {mg["slots"][i]: j for j, i in enumerate(keep)}
                    seq = []
                    for rr in range(r + 1, min(r + 1 + k_steps + 1, len(rounds))):
                        nxt = rounds[rr]
                        if nxt["block"] != rd["block"] or not nxt["commits"]:
                            break
                        pos = nxt["commits"][0][0]
                        if pos not in cand:
                            break
                        seq.append(cand[pos])
                    if len(seq) < 1:
                        continue
                    p = torch.tensor([mg["p"][i][:m_top] for i in keep], dtype=torch.float32)
                    if p.shape[1] < m_top:
                        p = torch.cat([p, p.new_zeros(p.shape[0], m_top - p.shape[1])], 1)
                    idx = torch.tensor([mg["slots"][i] for i in keep])
                    for d in range(min(len(seq), k_steps)):
                        ex.append((p, idx, rd["round_in_block"], seq[:d], seq[d], d,
                                   block_len))
                        if limit_rounds and len(ex) >= limit_rounds:
                            return ex, block_len
    return ex, block_len


def collate_pc(batch, block_len, m_top):
    from spot.slotpicker import path_features
    S = max(len(x[1]) for x in batch); B = len(batch)
    feats, pad = [], []
    lab = torch.zeros(B, dtype=torch.long)
    taken_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, (p, idx, rib, prefix, y, d, *rest) in enumerate(batch):
        bl = rest[0] if rest else block_len   # per-example width when pooling beds
        n = len(idx)
        taken = torch.zeros(n, dtype=torch.bool)
        for t in prefix:
            taken[t] = True
        f = torch.cat([slot_features(p, idx, bl, n, bl - n + len(prefix)),
                       path_features(n, taken, d, bl)], dim=-1)
        pd = S - n
        feats.append(torch.cat([f, f.new_zeros(pd, f.shape[1])]) if pd else f)
        pad.append(torch.tensor([False] * n + [True] * pd))
        taken_mask[b, :n] = taken
        lab[b] = y
    return torch.stack(feats), torch.stack(pad), taken_mask, lab


def collate(batch, block_len, k_steps, tok_mod):
    S = max(len(x[1]) for x in batch)
    B = len(batch)
    feats, toks, pad, lab = [], [], [], torch.full((B, k_steps), -100, dtype=torch.long)
    excl = torch.zeros(B, k_steps, S, dtype=torch.bool)  # true-prefix slots, per step
    for b, (p, idx, tk, rib, labels) in enumerate(batch):
        for k in range(k_steps):
            for y in labels[:k]:
                excl[b, k, y] = True
        f = slot_features(p, idx, block_len, len(idx), block_len - len(idx))
        pd = S - f.shape[0]
        feats.append(torch.cat([f, f.new_zeros(pd, f.shape[1])]) if pd else f)
        toks.append(torch.cat([tk % tok_mod, tk.new_zeros(pd)]) if pd else tk % tok_mod)
        pad.append(torch.tensor([False] * len(idx) + [True] * pd))
        for k, y in enumerate(labels[:k_steps]):
            lab[b, k] = y
    return (torch.stack(feats), torch.stack(toks), torch.stack(pad), lab, excl)


def conf_baseline(batch, k_steps):
    """The free drafter's rule: rank these same slots by stale top-1 confidence."""
    hits = [0] * k_steps; tot = [0] * k_steps; joint = [0] * k_steps
    for p, idx, tk, rib, labels in batch:
        order = torch.argsort(p[:, 0], descending=True).tolist()
        alive = True
        for k, y in enumerate(labels[:k_steps]):
            tot[k] += 1
            ok = k < len(order) and order[k] == y
            hits[k] += int(ok)
            alive = alive and ok
            joint[k] += int(alive)
    return hits, tot, joint


def evaluate(model, data, block_len, k_steps, tok_mod, bs=256):
    """Per-step and prefix-joint accuracy, using inference's own greedy distinct selection.

    Prefix-joint is the number that matters: a draft chain only reaches level k if every
    earlier level was right, so joint accuracy is what acceptance is made of.
    """
    model.eval()
    mh = [0] * k_steps; mt = [0] * k_steps; mj = [0] * k_steps
    bh = [0] * k_steps; bt = [0] * k_steps; bj = [0] * k_steps
    with torch.no_grad():
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            f, tk, pad, lab, _ = collate(chunk, block_len, k_steps, tok_mod)
            logits = model(f, tk, pad)  # (B, K, S)
            used = torch.zeros(logits.shape[0], logits.shape[2], dtype=torch.bool)
            alive = torch.ones(logits.shape[0], dtype=torch.bool)
            for k in range(k_steps):
                pred = logits[:, k].masked_fill(used, -1e9).argmax(-1)
                used.scatter_(1, pred[:, None], True)
                m = lab[:, k] != -100
                ok = (pred == lab[:, k]) & m
                mt[k] += int(m.sum()); mh[k] += int(ok.sum())
                alive = alive & (ok | ~m)
                mj[k] += int((alive & m).sum())
            h, t, j = conf_baseline(chunk, k_steps)
            bh = [a + b for a, b in zip(bh, h)]
            bt = [a + b for a, b in zip(bt, t)]
            bj = [a + b for a, b in zip(bj, j)]
    model.train()
    return (mh, mt, mj), (bh, bt, bj)


def main_ctx(args):
    """Context model: whole-block input, k heads, evaluated exactly like the marginal one."""
    t0 = time.time()
    tok_mod = args.tok_mod or 4096          # token identity matters here; hash by default
    data, block_len = build_ctx(args.data, args.m_top, args.k_steps, args.limit_rounds)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac)); val, train = data[:n_val], data[n_val:]
    print(f"== CONTEXT {len(data)} decisions, block={block_len}, train {len(train)} / "
          f"val {n_val}, tok_mod={tok_mod}, {time.time()-t0:.0f}s", flush=True)
    model = build_picker(getattr(args, 'arch', 'enc'), m_top=args.m_top, d_model=args.d_model, layers=args.layers,
                       heads=args.heads,
                       k_steps=args.k_steps, vocab=tok_mod, context=True)
    print(f"== params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(args.epochs):
        random.shuffle(train); tot = n = 0
        for i in range(0, len(train), args.bs):
            f, tk, fill, lab = collate_ctx(train[i:i + args.bs], block_len, args.k_steps, tok_mod)
            logits = model(f, tk).masked_fill(fill[:, None, :], -1e9)
            loss = lossf(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); n += 1
        model.eval(); hit = [0] * args.k_steps; cnt = [0] * args.k_steps; joint = [0] * args.k_steps
        with torch.no_grad():
            for i in range(0, len(val), 256):
                ch = val[i:i + 256]
                f, tk, fill, lab = collate_ctx(ch, block_len, args.k_steps, tok_mod)
                lg = model(f, tk).masked_fill(fill[:, None, :], -1e9)
                used = fill.clone(); alive = torch.ones(len(ch), dtype=torch.bool)
                for k in range(args.k_steps):
                    pred = lg[:, k].masked_fill(used, -1e9).argmax(-1)
                    used.scatter_(1, pred[:, None], True)
                    m = lab[:, k] != -100
                    ok = (pred == lab[:, k]) & m
                    cnt[k] += int(m.sum()); hit[k] += int(ok.sum())
                    alive = alive & (ok | ~m); joint[k] += int((alive & m).sum())
        model.train()
        dep = sum(joint[k] / max(cnt[0], 1) for k in range(args.k_steps))
        acc = " ".join(f"k{k+1}:{hit[k]/max(cnt[k],1):.3f}" for k in range(args.k_steps) if cnt[k])
        print(f"== ep {ep} loss {tot/max(n,1):.4f} | {acc} | E[depth] {dep:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "m_top": args.m_top, "k_steps": args.k_steps,
                    "d_model": args.d_model, "layers": args.layers, "vocab": tok_mod,
                    "block_len": block_len, "arch": getattr(args, "arch", "enc"), "epoch": ep, "context": True,
                    "val_acc": [hit[k] / max(cnt[k], 1) for k in range(args.k_steps)],
                    "exp_depth_model": dep}, args.out)
        if time.time() - t0 > args.time_budget_s:
            print("== time budget reached", flush=True); break
    print(f"== wrote {args.out}", flush=True)


def build_ttt(paths, m_top, k_steps, limit_rounds=0):
    """One example per drafting round, carrying the WHOLE true commit sequence.

    Unlike `build_pc`, the prefix is not baked in here: training-time test needs the prefix
    the *current* model would produce, so the rollout happens inside the training loop and
    this builder only supplies the round's state plus the true continuation to score against.
    """
    ex, block_len = [], None
    for path in paths:
        with open(path) as f:
            meta = json.loads(f.readline())
            block_len = int(meta["_meta"]["decode"]["block_length"])
            for line in f:
                rounds = json.loads(line)["rounds"]
                for r, rd in enumerate(rounds):
                    mg = rd.get("marg")
                    if not mg or not rd["commits"]:
                        continue
                    committed = rd["commits"][0][0]
                    keep = [i for i, s in enumerate(mg["slots"]) if s != committed]
                    if len(keep) < 2:
                        continue
                    cand = {mg["slots"][i]: j for j, i in enumerate(keep)}
                    seq = []
                    for rr in range(r + 1, min(r + 2 + k_steps, len(rounds))):
                        nxt = rounds[rr]
                        if nxt["block"] != rd["block"] or not nxt["commits"]:
                            break
                        pos = nxt["commits"][0][0]
                        if pos not in cand:
                            break
                        seq.append(cand[pos])
                    if not seq:
                        continue
                    p = torch.tensor([mg["p"][i][:m_top] for i in keep], dtype=torch.float32)
                    if p.shape[1] < m_top:
                        p = torch.cat([p, p.new_zeros(p.shape[0], m_top - p.shape[1])], 1)
                    idx = torch.tensor([mg["slots"][i] for i in keep])
                    ex.append((p, idx, len(keep), seq))
                    if limit_rounds and len(ex) >= limit_rounds:
                        return ex, block_len
    return ex, block_len


def collate_ttt(batch, block_len, k_steps):
    """(base features, pad mask, true-sequence tensor, lengths) -- prefixes added later."""
    S = max(len(x[1]) for x in batch)
    B = len(batch)
    feats, pad = [], []
    seqs = torch.full((B, k_steps + 1), -1, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for b, (p, idx, n, seq) in enumerate(batch):
        f = slot_features(p, idx, block_len, n, block_len - n)
        pd = S - n
        feats.append(torch.cat([f, f.new_zeros(pd, f.shape[1])]) if pd else f)
        pad.append(torch.tensor([False] * n + [True] * pd))
        L = min(len(seq), k_steps + 1)
        seqs[b, :L] = torch.tensor(seq[:L])
        lens[b] = L
    return torch.stack(feats), torch.stack(pad), seqs, lens


def _step_logits(model, base, taken, depth, block_len, pad):
    """One next-slot distribution per example, conditioned on this branch's filled slots.

    `depth` may be an int (whole batch at the same step, as during rollout) or a per-example
    tensor (training, where each example is scored at its own sampled depth). Feeding a
    batch-mean depth would repeat the mistake of showing the model a feature value that
    does not match its state.
    """
    from spot.slotpicker import path_features
    B, S, _ = base.shape
    dv = ([int(depth)] * B if isinstance(depth, int)
          else [int(x) for x in depth])
    pf = torch.stack([path_features(S, taken[b], dv[b], block_len, device=base.device)
                      for b in range(B)])
    logits = model(torch.cat([base, pf], dim=-1),
                   torch.zeros(B, S, dtype=torch.long), pad)[:, 0]
    return logits.masked_fill(taken | pad, -1e9)


def main_ttt(args):
    """Training-time test: the model is scored on prefixes IT generated, not the true ones.

    EAGLE-3's finding is that a drafter trained only on ground-truth prefixes never sees the
    error-carrying states it meets at inference. Our tree has exactly that gap: every branch
    below level 1 stands on the model's own picks, which are wrong ~30% of the time.

    Labelling rule: a self-generated prefix that is a PERMUTATION of the true prefix reaches
    the same block state (acceptance is state equality), so the true next commit is still a
    valid label. A prefix that leaves the true set reaches a counterfactual state whose
    continuation is unknown from stored data -- those are masked out of the loss and counted,
    since a high mask rate is itself a finding.
    """
    t0 = time.time()
    data, block_len = build_ttt(args.data, args.m_top, args.k_steps, args.limit_rounds)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac)); val, train = data[:n_val], data[n_val:]
    print(f"== TTT {len(data)} rounds, block={block_len}, train {len(train)} / val {n_val}, "
          f"{time.time()-t0:.0f}s", flush=True)
    model = build_picker(getattr(args, 'arch', 'enc'), m_top=args.m_top, d_model=args.d_model, layers=args.layers,
                       heads=args.heads,
                       k_steps=1, vocab=args.tok_mod, prefix_cond=True)
    print(f"== params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    for ep in range(args.epochs):
        # scheduled sampling: teacher forcing first, then increasingly the model's own picks
        eps = args.ttt_eps_max * min(1.0, ep / max(args.ttt_warmup, 1))
        random.shuffle(train)
        tot = n = 0; used = masked = 0
        for i in range(0, len(train), args.bs):
            chunk = train[i:i + args.bs]
            base, pad, seqs, lens = collate_ttt(chunk, block_len, args.k_steps)
            B, S, _ = base.shape
            dmax = int(lens.max()) - 1
            if dmax < 0:
                continue
            depth = torch.randint(0, max(dmax, 1), (B,))
            depth = torch.minimum(depth, (lens - 1).clamp(min=0))
            roll = torch.rand(B) < eps

            # ---- roll the model forward on its own picks (no grad) -------------------
            with torch.no_grad():
                taken_r = pad.clone()
                picks = torch.zeros(B, max(dmax, 1), dtype=torch.long)
                for t in range(max(dmax, 1)):
                    lg = _step_logits(model, base, taken_r, t, block_len, pad)
                    pr = lg.argmax(-1)
                    picks[:, t] = pr
                    taken_r = taken_r.clone()
                    taken_r.scatter_(1, pr[:, None], True)

            # ---- assemble each example's prefix and decide whether it is labelable ----
            taken = pad.clone()
            lab = torch.full((B,), -100, dtype=torch.long)
            for b in range(B):
                d = int(depth[b])
                true_pre = [int(seqs[b, t]) for t in range(d)]
                pre = [int(picks[b, t]) for t in range(d)] if bool(roll[b]) else true_pre
                if set(pre) != set(true_pre):
                    masked += 1                    # counterfactual state: no label exists
                    continue
                for s in pre:
                    taken[b, s] = True
                lab[b] = seqs[b, d]
                used += 1
            if int((lab != -100).sum()) == 0:
                continue

            logits = _step_logits(model, base, taken, depth, block_len, pad)
            loss = lossf(logits, lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); n += 1

        tf_acc, roll_acc = _eval_ttt(model, val, block_len, args.k_steps)
        print(f"== ep {ep} eps={eps:.2f} loss {tot/max(n,1):.4f} | teacher-forced "
              f"{tf_acc:.3f} | rollout {roll_acc:.3f} | gap {tf_acc-roll_acc:+.3f} | "
              f"masked {masked/max(used+masked,1):.3f} | {time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "m_top": args.m_top, "k_steps": 1,
                    "d_model": args.d_model, "layers": args.layers, "vocab": args.tok_mod,
                    "block_len": block_len, "arch": getattr(args, "arch", "enc"), "epoch": ep, "prefix_cond": True, "ttt": True,
                    "tf_acc": tf_acc, "roll_acc": roll_acc}, args.out)
        if time.time() - t0 > args.time_budget_s:
            print("== time budget reached", flush=True); break
    print(f"== wrote {args.out}", flush=True)


@torch.no_grad()
def _eval_ttt(model, val, block_len, k_steps, bs=256):
    """Accuracy under teacher forcing vs under the model's own prefixes.

    The gap between these two is what training-time test exists to close; reporting only
    the first is what let the context model look good offline and decode worse.
    """
    model.eval()
    tf_hit = tf_tot = rl_hit = rl_tot = 0
    for i in range(0, len(val), bs):
        chunk = val[i:i + bs]
        base, pad, seqs, lens = collate_ttt(chunk, block_len, k_steps)
        B = base.shape[0]
        dmax = int(lens.max()) - 1
        if dmax < 1:
            continue
        taken_tf = pad.clone(); taken_rl = pad.clone()
        for t in range(dmax):
            live = (lens - 1) > t
            lg_tf = _step_logits(model, base, taken_tf, t, block_len, pad)
            lg_rl = _step_logits(model, base, taken_rl, t, block_len, pad)
            y = seqs[:, t]
            tf_hit += int(((lg_tf.argmax(-1) == y) & live).sum())
            rl_hit += int(((lg_rl.argmax(-1) == y) & live).sum())
            tf_tot += int(live.sum()); rl_tot += int(live.sum())
            # teacher forcing advances along the TRUE path; rollout advances on its own pick
            taken_tf = taken_tf.clone(); taken_rl = taken_rl.clone()
            step_tf = torch.where(live, y, torch.zeros_like(y)).clamp(min=0)
            taken_tf.scatter_(1, step_tf[:, None], live[:, None])
            taken_rl.scatter_(1, lg_rl.argmax(-1)[:, None], live[:, None])
    model.train()
    return tf_hit / max(tf_tot, 1), rl_hit / max(rl_tot, 1)


def main_pc(args):
    """Prefix-conditional training: single next-slot head over path-aware states."""
    t0 = time.time()
    data, block_len = build_pc(args.data, args.m_top, args.k_steps, args.limit_rounds)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac)); val, train = data[:n_val], data[n_val:]
    print(f"== PREFIX-COND {len(data)} (state, next-slot) pairs, block={block_len}, "
          f"train {len(train)} / val {n_val}, {time.time()-t0:.0f}s", flush=True)
    model = build_picker(getattr(args, 'arch', 'enc'), m_top=args.m_top, d_model=args.d_model, layers=args.layers,
                       heads=args.heads,
                       k_steps=1, vocab=args.tok_mod, prefix_cond=True)
    print(f"== params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        random.shuffle(train); tot = n = 0
        for i in range(0, len(train), args.bs):
            f, pad, taken, lab = collate_pc(train[i:i + args.bs], block_len, args.m_top)
            logits = model(f, torch.zeros(f.shape[0], f.shape[1], dtype=torch.long), pad)[:, 0]
            logits = logits.masked_fill(taken, -1e9)
            loss = lossf(logits, lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); n += 1
        # validation by depth: accuracy where this branch is d slots deep
        model.eval(); hit = collections.Counter(); cnt = collections.Counter()
        with torch.no_grad():
            for i in range(0, len(val), 256):
                ch = val[i:i + 256]
                f, pad, taken, lab = collate_pc(ch, block_len, args.m_top)
                lg = model(f, torch.zeros(f.shape[0], f.shape[1], dtype=torch.long), pad)[:, 0]
                pr = lg.masked_fill(taken, -1e9).argmax(-1)
                for b, ex_i in enumerate(ch):
                    d = ex_i[5]   # per-example tuple gained block_len (pooling support)
                    cnt[d] += 1; hit[d] += int(pr[b] == lab[b])
        model.train()
        acc = " ".join(f"d{d}:{hit[d]/cnt[d]:.3f}" for d in sorted(cnt) if cnt[d] > 30)
        print(f"== ep {ep} loss {tot/max(n,1):.4f} | depth-acc {acc} | {time.time()-t0:.0f}s",
              flush=True)
        torch.save({"model": model.state_dict(), "m_top": args.m_top, "k_steps": 1,
                    "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                    "vocab": args.tok_mod,
                    "block_len": block_len, "arch": getattr(args, "arch", "enc"), "epoch": ep, "prefix_cond": True,
                    "depth_acc": {d: hit[d] / cnt[d] for d in cnt}}, args.out)
        if time.time() - t0 > args.time_budget_s:
            print("== time budget reached", flush=True); break
    print(f"== wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--arch", default="enc",
                    help="picker architecture: enc|ptr|rel|gru|moe (spot/pickers.py)")
    ap.add_argument("--m-top", type=int, default=32)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tok-mod", type=int, default=0, help="hashed token-embedding size; 0 = off")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit-rounds", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=7200)
    ap.add_argument("--context", action="store_true",
                    help="feed the WRITTEN PREFIX: whole-block rows with filled/neighbour "
                         "flags and committed-token embeddings, not just the marginals")
    ap.add_argument("--ttt", action="store_true",
                    help="training-time test: score the model on prefixes IT generated "
                         "(scheduled sampling), not on ground-truth prefixes")
    ap.add_argument("--ttt-eps-max", type=float, default=0.5)
    ap.add_argument("--ttt-warmup", type=int, default=8,
                    help="epochs over which eps anneals 0 -> ttt-eps-max")
    ap.add_argument("--prefix-cond", action="store_true",
                    help="train the TREE model: one head, conditioned on the slots already "
                         "filled on this branch, asked recursively at every node")
    ap.add_argument("--a2", action="store_true", help="live-target trainer (hidden feats + replay labels)")
    ap.add_argument("--a2-model", default="llada2")
    ap.add_argument("--a2-hid", type=int, default=32)
    ap.add_argument("--a2-eps-max", type=float, default=0.0)
    ap.add_argument("--a2-groups-per-epoch", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    torch.set_num_threads(8)

    t0 = time.time()
    if args.a2:
        return main_a2(args)
    if args.ttt:
        return main_ttt(args)
    if args.context:
        return main_ctx(args)
    if args.prefix_cond:
        return main_pc(args)
    data, block_len = build(args.data, args.m_top, args.k_steps, args.limit_rounds)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac))
    val, train = data[:n_val], data[n_val:]
    print(f"== {len(data)} decisions (block_len={block_len}), train {len(train)} / val {n_val}, "
          f"built in {time.time()-t0:.0f}s", flush=True)

    model = build_picker(getattr(args, 'arch', 'enc'), m_top=args.m_top, d_model=args.d_model, layers=args.layers,
                       heads=args.heads,
                       k_steps=args.k_steps, vocab=args.tok_mod)
    print(f"== params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    for ep in range(args.epochs):
        random.shuffle(train); tot = n = 0
        for i in range(0, len(train), args.bs):
            f, tk, pad, lab, excl = collate(train[i:i + args.bs], block_len, args.k_steps,
                                            args.tok_mod or 1)
            logits = model(f, tk, pad)  # (B, K, S)
            logits = logits.masked_fill(excl, -1e9)
            loss = lossf(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); n += 1
        (mh, mt, mj), (bh, bt, bj) = evaluate(model, val, block_len, args.k_steps,
                                              args.tok_mod or 1)
        acc = " ".join(f"k{k+1}:{mh[k]/max(mt[k],1):.3f}/{bh[k]/max(bt[k],1):.3f}"
                       f"(J{mj[k]/max(mt[k],1):.3f}/{bj[k]/max(bt[k],1):.3f})"
                       for k in range(args.k_steps) if mt[k])
        # E[accepted chain depth] = sum_k P(first k levels all correct); with one commit per
        # accepted level this is the per-round NFE saving the chain buys, model vs free.
        dep_m = sum(mj[k] / max(mt[0], 1) for k in range(args.k_steps))
        dep_f = sum(bj[k] / max(bt[0], 1) for k in range(args.k_steps))
        acc += f" | E[depth] model {dep_m:.3f} vs free {dep_f:.3f}"
        print(f"== ep {ep} loss {tot/max(n,1):.4f} | model/free {acc} | {time.time()-t0:.0f}s",
              flush=True)
        torch.save({"model": model.state_dict(), "m_top": args.m_top, "k_steps": args.k_steps,
                    "d_model": args.d_model, "layers": args.layers, "vocab": args.tok_mod,
                    "block_len": block_len, "arch": getattr(args, "arch", "enc"), "epoch": ep,
                    "val_acc": [mh[k] / max(mt[k], 1) for k in range(args.k_steps)],
                    "val_joint": [mj[k] / max(mt[k], 1) for k in range(args.k_steps)],
                    "free_acc": [bh[k] / max(bt[k], 1) for k in range(args.k_steps)],
                    "free_joint": [bj[k] / max(bt[k], 1) for k in range(args.k_steps)],
                    "exp_depth_model": dep_m, "exp_depth_free": dep_f},
                   args.out)
        if time.time() - t0 > args.time_budget_s:
            print("== time budget reached", flush=True); break
    print(f"== wrote {args.out}", flush=True)




# ---------------------------------------------------------------------------------------
# A2: live-target training .
# The frozen target sits in the training loop: every example's LABEL is the commit the
# target actually makes at that example's state (replay -- exact even for counterfactual
# prefixes, the thing A1 could not label), and every example's hidden FEATURES are the
# target's own last-layer states, projected to --a2-hid dims by a fixed seeded matrix.
# Features stay STALE-by-construction (root-state hidden + root-state table), matching
# what inference has; only the label comes from the deep forward.
# ---------------------------------------------------------------------------------------

def build_a2(paths, m_top, k_steps):
    """Per-(traj, round) groups with enough info to reconstruct real windows."""
    groups, block_len, mask_id, gen_len = [], None, None, None
    for path in paths:
        with open(path) as f:
            meta = json.loads(f.readline())["_meta"]
            block_len = int(meta["decode"]["block_length"])
            gen_len = int(meta["decode"]["gen_length"])
            for line in f:
                o = json.loads(line)
                rounds = o["rounds"]
                commits = []  # (round_i, block, pos, token); block*L+pos indexes the
                # FULL sequence (block-aligned absolute frame -- generation starts
                # mid-block when the prompt is not block-aligned)
                for ri, rd in enumerate(rounds):
                    for c in rd["commits"]:
                        pos, tok = (c[0], c[1]) if isinstance(c, (list, tuple)) else (c["block_pos"], c["token"])
                        commits.append((ri, rd["block"], pos, tok))
                for ri, rd in enumerate(rounds):
                    mg = rd.get("marg")
                    if not mg or not rd["commits"]:
                        continue
                    c0 = rd["commits"][0]
                    committed = c0[0] if isinstance(c0, (list, tuple)) else c0["block_pos"]
                    keep = [i for i, s in enumerate(mg["slots"]) if s != committed]
                    if len(keep) < 2:
                        continue
                    cand = {mg["slots"][i]: j for j, i in enumerate(keep)}
                    seq = []
                    for rr in range(ri + 1, min(ri + 1 + k_steps + 1, len(rounds))):
                        nxt = rounds[rr]
                        if nxt["block"] != rd["block"] or not nxt["commits"]:
                            break
                        nc = nxt["commits"][0]
                        pos = nc[0] if isinstance(nc, (list, tuple)) else nc["block_pos"]
                        if pos not in cand:
                            break
                        seq.append(cand[pos])
                    if not seq:
                        continue
                    wmax = max([b for (_, b, _, _) in commits] + [rd["block"]])
                    groups.append({"prompt": o["prompt_ids"], "commits": commits,
                                   "wmax": wmax,
                                   "round": ri, "block": rd["block"], "marg": mg,
                                   "keep": keep, "seq": seq,
                                   "rib": rd.get("round_in_block", 0)})
    return groups, block_len, gen_len


def main_a2(args):
    import numpy as np
    from spot.models import load_adapter
    from spot.decode import _attn_bias
    from spot.pickers import build_picker
    from spot.slotpicker import SlotPicker, slot_features, path_features

    dev = "cuda"
    ad = load_adapter(args.a2_model)
    mask_id = ad.mask_id
    groups, L, gen_len = build_a2(args.data, args.m_top, args.k_steps)
    random.shuffle(groups)
    n_val = max(8, int(len(groups) * args.val_frac))
    val, train = groups[:n_val], groups[n_val:]
    print(f"== A2: {len(train)} train / {n_val} val groups, block={L}", flush=True)

    d_t = None  # discovered on first forward
    proj = None
    g_proj = torch.Generator().manual_seed(42)
    model = None
    opt = None
    bc = bool(getattr(ad, "block_causal", False))

    def window(g, depth_prefix):
        """Full-sequence input_ids. Coordinate frame (measured on the data, ):
        blk*L+pos is relative to origin=(P//L)*L -- the prompt's last block boundary --
        and the final partial block fills past the nominal gen end (block-padded window).
        """
        P = len(g["prompt"]); origin = (P // L) * L
        total = origin + (g["wmax"] + 1) * L
        pr = torch.tensor(g["prompt"], dtype=torch.long)
        seq = torch.cat([pr, torch.full((total - P,), mask_id, dtype=torch.long)])
        for (ri, blk, pos, tok) in g["commits"]:
            if ri <= g["round"]:
                seq[origin + blk * L + pos] = tok
        mg = g["marg"]
        for j in depth_prefix:
            slot = mg["slots"][g["keep"][j]]
            t = mg["tok"][g["keep"][j]]
            seq[origin + g["block"] * L + slot] = t[0] if isinstance(t, list) else t
        return seq

    def fwd_batch(wins):
        nonlocal d_t, proj
        x = torch.stack(wins).to(dev)
        pl = x.shape[1]
        pos = torch.arange(pl, device=dev)
        bias = _attn_bias(0, pl, 0, L, 0, 0, dev, block_causal=bc, chunk_start=0)
        out = ad.forward(x, pos, bias, None, False, want_hidden=True)
        if d_t is None:
            d_t = out.hidden.shape[-1]
            proj = (torch.randn(d_t, args.a2_hid, generator=g_proj) / (d_t ** 0.5)).to(dev)
        return out

    def group_pass(g, eps, train_mode):
        """One (traj, round) group -> stacked examples for depths 0..len(seq)-1.

        eps > 0 (A2 rung 2): each prefix element is, with prob eps, the MODEL'S OWN pick
        instead of the recorded one -- the state distribution shifts toward what inference
        visits, and the live target still labels the resulting counterfactual exactly.
        Requires the root forward first (rollout picks need root features), so eps > 0
        costs one extra small forward batch per group.
        """
        P = len(g["prompt"]); blk_lo = (P // L) * L + g["block"] * L
        depths = list(range(min(len(g["seq"]), args.k_steps)))
        keep_n = len(g["keep"])
        prefixes = {0: []}
        if eps > 0.0 and len(depths) > 1:
            out0 = fwd_batch([window(g, [])])
            hid0 = torch.einsum("ld,dh->lh", out0.hidden[0, blk_lo:blk_lo + L].float(), proj)
            mg0 = g["marg"]
            p_top0 = torch.tensor([mg0["p"][i][:args.m_top] for i in g["keep"]], dtype=torch.float32)
            if p_top0.shape[1] < args.m_top:
                p_top0 = torch.cat([p_top0, p_top0.new_zeros(p_top0.shape[0], args.m_top - p_top0.shape[1])], 1)
            idx0 = torch.tensor([mg0["slots"][i] for i in g["keep"]], dtype=torch.float32)
            base0 = slot_features(p_top0.to(dev), idx0.to(dev), L, keep_n, L - keep_n)
            h0 = hid0[[int(mg0["slots"][i]) for i in g["keep"]]] if args.a2_hid else None
            pref = []
            for d in depths[1:]:
                if random.random() < eps:
                    taken = torch.zeros(keep_n, dtype=torch.bool, device=dev)
                    for t in pref:
                        taken[t] = True
                    parts = [base0] + ([h0] if h0 is not None else []) +                             [path_features(keep_n, taken, len(pref), L, device=dev)]
                    with torch.no_grad():
                        lg = model(torch.cat(parts, -1)[None], None)[0, 0]
                    lg[taken] = -1e9
                    pref = pref + [int(lg.argmax())]
                else:
                    # continue with the recorded element closest to on-path behavior
                    nxt = [t for t in g["seq"] if t not in pref]
                    if not nxt:
                        break
                    pref = pref + [nxt[0]]
                prefixes[d] = list(pref)
        else:
            for d in depths[1:]:
                prefixes[d] = g["seq"][:d]
        wins = [window(g, [])]
        for d in depths[1:]:
            if d in prefixes:
                wins.append(window(g, prefixes[d]))
        depths = [0] + [d for d in depths[1:] if d in prefixes]
        out = fwd_batch(wins)
        probs = torch.softmax(out.logits[:, blk_lo:blk_lo + L].float(), -1)
        hid = torch.einsum("bld,dh->blh", out.hidden[:, blk_lo:blk_lo + L].float(), proj)
        mg = g["marg"]
        p_top = torch.tensor([mg["p"][i][:args.m_top] for i in g["keep"]], dtype=torch.float32)
        if p_top.shape[1] < args.m_top:
            p_top = torch.cat([p_top, p_top.new_zeros(p_top.shape[0], args.m_top - p_top.shape[1])], 1)
        idx = torch.tensor([mg["slots"][i] for i in g["keep"]], dtype=torch.float32)
        base = slot_features(p_top.to(dev), idx.to(dev), L, keep_n, L - keep_n)
        hroot = hid[0][[int(mg["slots"][i]) for i in g["keep"]]]  # STALE root hidden
        feats_rows, labels, pads = [], [], []
        for bi, d in enumerate(depths):
            # live label: the decoder's own commit rule on THIS state's fresh forward
            pr_d = probs[bi]
            tok = pr_d.argmax(-1)
            conf = pr_d.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            open_mask = torch.ones(L, dtype=torch.bool, device=dev)
            wrow = wins[bi]
            open_mask = (wrow[blk_lo: blk_lo + L] == mask_id).to(dev)
            confm = torch.where(open_mask, conf, torch.full_like(conf, -1.0))
            slot_live = int(confm.argmax())
            lab = None
            for j, i in enumerate(g["keep"]):
                if mg["slots"][i] == slot_live and j not in prefixes.get(d, []):
                    lab = j
            if lab is None:
                continue  # off-set commit; count-free skip (coverage is printed)
            taken = torch.zeros(keep_n, dtype=torch.bool)
            for t in prefixes.get(d, []):
                taken[t] = True
            f = torch.cat([base, hroot, path_features(keep_n, taken.to(dev), d, L, device=dev)], -1)
            feats_rows.append((f, taken, lab))
        return feats_rows

    # lazily create the model after first group (needs d_t? no -- hid fixed by flag)
    model = build_picker(getattr(args, 'arch', 'enc'), m_top=args.m_top, d_model=args.d_model, layers=args.layers,
                       heads=args.heads,
                       k_steps=1, prefix_cond=True, hid=args.a2_hid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs):
        eps = args.a2_eps_max * min(1.0, ep / max(args.epochs // 2, 1))
        random.shuffle(train)
        n_ex = n_hit = 0
        losses = []
        for g in train[: args.a2_groups_per_epoch]:
            with torch.no_grad():
                rows = group_pass(g, eps, True)
            if not rows:
                continue
            S = len(g["keep"])
            f = torch.stack([r[0] for r in rows])
            lab = torch.tensor([r[2] for r in rows], device=dev)
            logit = model(f, None)[:, 0, :]
            for bi, r in enumerate(rows):
                logit[bi, r[1].to(dev)] = -1e9
            loss = torch.nn.functional.cross_entropy(logit, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach())); n_ex += len(rows)
            if time.time() - t0 > args.time_budget_s:
                break
        # val: teacher-forced depth accuracy AND (the gate) live rollout accuracy
        model.eval(); acc = n = 0
        with torch.no_grad():
            for g in val[:120]:
                rows = group_pass(g, 0.0, False)
                if not rows: continue
                f = torch.stack([r[0] for r in rows])
                logit = model(f, None)[:, 0, :]
                for bi, r in enumerate(rows):
                    logit[bi, r[1].to(dev)] = -1e9
                pred = logit.argmax(-1)
                lab = torch.tensor([r[2] for r in rows], device=dev)
                acc += int((pred == lab).sum()); n += len(rows)
        model.train()
        print(f"== A2 ep{ep} eps={eps:.2f} loss={sum(losses)/max(len(losses),1):.3f} "
              f"val_acc={acc/max(n,1):.3f} ({n}) ex={n_ex} {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > args.time_budget_s:
            break
    torch.save({"model": model.state_dict(),
                "cfg": {"m_top": args.m_top, "d_model": args.d_model, "layers": args.layers,
                        "k_steps": 1, "prefix_cond": True, "hid": args.a2_hid,
                        "block_len": L},
                "proj": proj.cpu()}, args.out)
    print("== wrote", args.out)


if __name__ == "__main__":
    main()
