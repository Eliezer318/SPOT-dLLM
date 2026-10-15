"""Run one configuration cell and write its artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from typing import List, Optional

import torch

from .decode import DecodeConfig, generate
from .graphs import DraftGraph
from .tasks import Sample, score_task


def provenance(graph_path: Optional[str] = None) -> dict:
    import transformers

    p = {
        "argv": sys.argv,
        # Machine identity is opt-in: set SPOT_RECORD_HOST=1 to store it.
        "host": socket.gethostname() if os.environ.get("SPOT_RECORD_HOST") else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID") if os.environ.get("SPOT_RECORD_HOST") else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "time_started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if graph_path and os.path.exists(graph_path):
        p["graph_sha1"] = hashlib.sha1(open(graph_path, "rb").read()).hexdigest()[:12]
    return p


def run_cell(
    adapter,
    model_name: str,
    task: str,
    cfg: DecodeConfig,
    samples: List[Sample],
    out_dir: str,
    graph: Optional[DraftGraph] = None,
    graph_path: Optional[str] = None,
    score: bool = True,
    log_rounds: bool = True,
    offset: int = 0) -> dict:
    cfg.validate()
    os.makedirs(out_dir, exist_ok=True)
    config_blob = {
        "model": model_name,
        "task": task,
        "limit": len(samples),
        "offset": offset,
        "decode": dict(cfg.__dict__),
        "graph": graph_path,
        "graph_meta": graph.meta if graph else None,
        "graph_nodes": graph.to_json()["nodes"] if graph else None,
        "provenance": provenance(graph_path),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_blob, f, indent=2)

    rounds_f = open(os.path.join(out_dir, "rounds.jsonl"), "w") if log_rounds else None
    samples_f = open(os.path.join(out_dir, "samples.jsonl"), "w")
    tot = {"nfe": 0, "positions": 0, "verify_rows": 0, "wall": 0.0, "wall_net": 0.0, "tokens": 0, "score": 0.0, "n": 0}

    for i, s in enumerate(samples):
        prompt_ids = (
            adapter.encode_raw(s.raw_prompt) if s.raw_prompt else adapter.encode_prompt(s.messages)
        )
        res = generate(adapter, prompt_ids, cfg, graph=graph, sample_id=s.sample_id)
        sc = score_task(task, s, res["text"]) if score else None
        if rounds_f:
            for r in res["rounds"]:
                rounds_f.write(json.dumps(r) + "\n")
        rec = {
            "sample": s.sample_id,
            "prompt_len": int(prompt_ids.shape[1]),
            "nfe": res["nfe"],
            "positions_forwarded": res["positions_forwarded"],
            "verify_rows": res["verify_rows"],
            "draft_states_verified": res["draft_states_verified"],
            "wall_s": res["wall_s"],
            "wall_s_net": res["wall_s_net"],
            "instr_s": res["instr_s"],
            "tokens_generated": res["tokens_generated"],
            "tps": res["tokens_generated"] / res["wall_s_net"],
            "n_rounds": len(res["rounds"]),
            "score": sc,
            "text": res["text"],
        }
        samples_f.write(json.dumps(rec) + "\n")
        samples_f.flush()
        tot["nfe"] += res["nfe"]
        tot["positions"] += res["positions_forwarded"]
        tot["verify_rows"] += res["verify_rows"]
        tot["wall"] += res["wall_s"]
        tot["wall_net"] += res["wall_s_net"]
        tot["tokens"] += res["tokens_generated"]
        tot["n"] += 1
        if sc is not None:
            tot["score"] += sc
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(samples)}] nfe/sample={tot['nfe']/tot['n']:.1f} "
                f"tps={tot['tokens']/tot['wall_net']:.2f} acc={tot['score']/tot['n']:.3f}",
                flush=True)

    summary = {
        "config": config_blob,
        "n_samples": tot["n"],
        "nfe_total": tot["nfe"],
        "nfe_per_sample": tot["nfe"] / max(tot["n"], 1),
        "positions_forwarded_total": tot["positions"],
        "positions_per_sample": tot["positions"] / max(tot["n"], 1),
        "verify_rows_total": tot["verify_rows"],
        "wall_s_total": tot["wall"],
        "wall_s_net_total": tot["wall_net"],
        "instrumentation_s_total": tot["wall"] - tot["wall_net"],
        "tokens_total": tot["tokens"],
        "tps": tot["tokens"] / tot["wall_net"],
        "tps_including_instrumentation": tot["tokens"] / tot["wall"],
        "accuracy": tot["score"] / max(tot["n"], 1) if score else None,
        "time_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    samples_f.close()
    if rounds_f:
        rounds_f.close()
    return summary
