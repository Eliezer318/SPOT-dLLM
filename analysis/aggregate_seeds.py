#!/usr/bin/env python
"""Aggregate multi-seed cells into mean/std WITHOUT discarding individual runs.

Scans an out-root produced by run_sweep with --seeds 0,1,2: for every arm dir
containing D{budget}_seed{s}/summary.json, groups by budget, and reports per
metric: individual per-seed values, mean, std (population, ddof=1 when n>1).
TPS ratios are computed per seed against the SAME seed's D0 cell (each seed's
own control), then aggregated.

Usage: python analysis/aggregate_seeds.py <arm-dir> [...]   (writes
<arm-dir>/aggregate.json and prints a table)
"""
import json, sys, glob, os, re, statistics as st
from collections import defaultdict

def agg(vals):
    return {"runs": vals, "mean": st.mean(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0}

for root in sys.argv[1:]:
    cells = defaultdict(dict)   # D -> seed -> summary
    for f in glob.glob(os.path.join(root, "D*_seed*", "summary.json")):
        if not re.fullmatch(r"D\d+_seed\d+", os.path.basename(os.path.dirname(f))): continue  # skip archived cells (e.g. D0_seed1.slow-<date>)
        name = os.path.basename(os.path.dirname(f))
        D = int(name.split("_")[0][1:]); seed = int(name.split("seed")[1])
        cells[D][seed] = json.load(open(f))
    if not cells:
        print(f"{root}: no cells"); continue
    out = {}
    base = cells.get(0, {})
    for D in sorted(cells):
        seeds = sorted(cells[D])
        row = {"seeds": seeds}
        for m in ("nfe_per_sample", "tps", "accuracy",
                  "accepted_per_drafted_round", "wall_s_per_sample"):
            vals = [cells[D][s].get(m) for s in seeds if cells[D][s].get(m) is not None]
            if vals: row[m] = agg(vals)
        if D != 0 and base:
            ratios = [cells[D][s]["tps"] / base[s]["tps"]
                      for s in seeds if s in base]
            if ratios: row["tps_ratio_vs_own_D0"] = agg(ratios)
        out[f"D{D}"] = row
    json.dump(out, open(os.path.join(root, "aggregate.json"), "w"), indent=1)
    print(f"== {root}")
    for D, row in out.items():
        nfe = row.get("nfe_per_sample", {}); tr = row.get("tps_ratio_vs_own_D0", {})
        line = f"  {D:>4}: NFE {nfe.get('mean', 0):7.1f} ± {nfe.get('std', 0):.2f}  {[round(v,1) for v in nfe.get('runs',[])]}"
        if tr: line += f"   TPSx {tr['mean']:.2f} ± {tr['std']:.2f}  {[round(v,2) for v in tr['runs']]}"
        print(line)
