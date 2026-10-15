"""Datasets, prompts and metrics.

lm-eval is not installed in the environment, so the prompt formats below are written
directly.  They follow lm-eval's task definitions (and the LLaDA / Fast-dLLM eval
scripts) as closely as possible; the exact strings are recorded here so that any
prompt-induced difference from the paper is inspectable rather than hidden.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

# Default is the standard Hugging Face cache; override with SPOT_HUB.
HUB = os.environ.get("SPOT_HUB", os.path.expanduser("~/.cache/huggingface/hub"))


def _snapshot(repo_dirname: str) -> str:
    """Newest snapshot that actually has data files.

    The shared HF cache keeps several snapshot dirs per repo and some are stubs, so
    picking the lexicographically last one is not safe.
    """
    base = os.path.join(HUB, repo_dirname, "snapshots")
    best, best_n = None, -1
    for snap in sorted(os.listdir(base)):
        p = os.path.join(base, snap)
        n = sum(len(fs) for _, _, fs in os.walk(p))
        if n > best_n:
            best, best_n = p, n
    if best is None or best_n == 0:
        raise FileNotFoundError(f"no populated snapshot under {base}")
    return best


@dataclass
class Sample:
    sample_id: str
    messages: List[dict]
    reference: Any
    meta: Dict[str, Any]
    raw_prompt: Optional[str] = None  # set when the task is a plain completion, not a chat turn


# ----------------------------------------------------------------------------------
# MBPP  (3-shot, exemplars from the `prompt` split = task_id 1..10, as in the MBPP paper
# and lm-eval's mbpp task).  Answers are wrapped in [BEGIN]/[DONE].
# ----------------------------------------------------------------------------------
MBPP_INSTR = (
    "You are an expert Python programmer, and here is your task: {text} "
    "Your code should pass these tests:\n\n{tests}\n"
)


def _mbpp_rows(split: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_snapshot("datasets--google-research-datasets--mbpp"), "full", f"{split}-00000-of-00001.parquet"))


def load_mbpp(
    limit: Optional[int] = None,
    offset: int = 0,
    split: str = "test",
    n_shot: int = 3,
    style: str = "raw3shot") -> List[Sample]:
    """MBPP prompts.

    style="raw3shot"  : the 3-shot completion prompt, no chat template.  This is what
                        lm-eval's mbpp task does, and it is what LLaDA-8B-Instruct needs:
                        wrapping the same 3-shot text in the chat template costs it two
                        thirds of its pass@1 (0.40 -> 0.12).
    style="chat3shot" : the same text as a single user turn under the chat template.
    """
    test = _mbpp_rows(split)
    shots = _mbpp_rows("prompt").iloc[:n_shot]
    prefix = ""
    for _, r in shots.iterrows():
        prefix += MBPP_INSTR.format(text=r["text"], tests="\n".join(r["test_list"]))
        prefix += "[BEGIN]\n" + r["code"] + "\n[DONE]\n"
    rows = test.iloc[offset : offset + limit] if limit else test.iloc[offset:]
    out = []
    for _, r in rows.iterrows():
        q = prefix + MBPP_INSTR.format(text=r["text"], tests="\n".join(r["test_list"])) + "[BEGIN]\n"
        out.append(
            Sample(
                sample_id=f"mbpp-{int(r['task_id'])}",
                messages=[{"role": "user", "content": q}],
                reference={"test_list": list(r["test_list"]), "test_setup_code": r.get("test_setup_code", "")},
                meta={"task_id": int(r["task_id"]), "prompt_style": style},
                raw_prompt=q if style == "raw3shot" else None)
        )
    return out


def score_mbpp(sample: Sample, text: str, timeout: int = 10) -> float:
    # End marker is the PREFIX "[DONE", not "[DONE]": a block-diffusion target that runs out
    # of block at the marker emits "[DONE" and then EOS, so the closing bracket never lands.
    # With the full marker those samples kept a trailing "[DONE" inside the extracted code,
    # which is a SyntaxError and scored 0 (SDAR-1.7B did this on 98/100 MBPP prompts:
    # 0.01 instead of 0.30; SDAR-4B 0.24 -> 0.27; SDAR-8B unchanged).  Splitting on the prefix
    # is strictly safer: whenever "[DONE]" is present the two split at the same index, so
    # only samples that lacked the closing bracket can change.
    code = _extract_between(text, "[BEGIN]", "[DONE")
    program = "\n".join(
        [code, sample.reference.get("test_setup_code") or "", *sample.reference["test_list"]]
    )
    return float(_run_python(program, timeout))


# ----------------------------------------------------------------------------------
# HumanEval (0-shot instruct style, completion extracted from a markdown code block)
# ----------------------------------------------------------------------------------
HUMANEVAL_INSTR = (
    "Complete the following python function. Provide the complete function implementation "
    "inside a ```python code block.\n\n{prompt}"
)


def load_humaneval(limit: Optional[int] = None, offset: int = 0, **kw) -> List[Sample]:
    df = pd.read_parquet(os.path.join(_snapshot("datasets--openai--openai_humaneval"), "openai_humaneval", "test-00000-of-00001.parquet"))
    rows = df.iloc[offset : offset + limit] if limit else df.iloc[offset:]
    out = []
    for _, r in rows.iterrows():
        out.append(
            Sample(
                sample_id=r["task_id"].replace("/", "-"),
                messages=[{"role": "user", "content": HUMANEVAL_INSTR.format(prompt=r["prompt"])}],
                reference={"test": r["test"], "entry_point": r["entry_point"], "prompt": r["prompt"]},
                meta={"task_id": r["task_id"]})
        )
    return out


def score_humaneval(sample: Sample, text: str, timeout: int = 15) -> float:
    code = _extract_code_block(text) or text
    program = "\n".join([code, sample.reference["test"], f"check({sample.reference['entry_point']})"])
    return float(_run_python(program, timeout))


# ----------------------------------------------------------------------------------
# GSM8K (0-shot instruct, answer after ####/final number)
# ----------------------------------------------------------------------------------
def load_gsm8k(limit: Optional[int] = None, offset: int = 0, **kw) -> List[Sample]:
    df = pd.read_parquet(os.path.join(_snapshot("datasets--openai--gsm8k"), "main", "test-00000-of-00001.parquet"))
    rows = df.iloc[offset : offset + limit] if limit else df.iloc[offset:]
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        q = r["question"] + "\nPlease reason step by step, and put your final answer after '####'."
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            Sample(f"gsm8k-{offset+i}", [{"role": "user", "content": q}], {"answer": gold}, {})
        )
    return out


def score_gsm8k(sample: Sample, text: str, **kw) -> float:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("####")[-1] if "####" in text else text)
    if not nums:
        return 0.0
    pred = nums[0 if "####" in text else -1].replace(",", "").rstrip(".")
    gold = sample.reference["answer"]
    try:
        return float(abs(float(pred) - float(gold)) < 1e-6)
    except ValueError:
        return float(pred == gold)


# ----------------------------------------------------------------------------------
# MATH500 (0-shot instruct, \boxed{} answer)
# ----------------------------------------------------------------------------------
def load_math500(limit: Optional[int] = None, offset: int = 0, **kw) -> List[Sample]:
    path = os.path.join(_snapshot("datasets--HuggingFaceH4--MATH-500"), "test.jsonl")
    recs = [json.loads(l) for l in open(path)]
    recs = recs[offset : offset + limit] if limit else recs[offset:]
    out = []
    for i, r in enumerate(recs):
        q = r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        out.append(Sample(f"math500-{offset+i}", [{"role": "user", "content": q}], {"answer": r["answer"]}, {}))
    return out


def score_math500(sample: Sample, text: str, **kw) -> float:
    pred = _last_boxed(text)
    gold = sample.reference["answer"]
    if pred is None:
        return 0.0
    return float(_norm_math(pred) == _norm_math(gold))


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _extract_between(text: str, start: str, end: str) -> str:
    if start in text:
        text = text.split(start, 1)[1]
    if end in text:
        text = text.split(end, 1)[0]
    return text.strip()


def _extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, re.S)
    return m.group(1) if m else None


def _last_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return text[i + 1 :]


def _norm_math(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\\!", "").replace("\\,", "").replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".").rstrip("%")
    if s.startswith("\\frac") is False and re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _run_python(program: str, timeout: int) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout, env={**os.environ, "OMP_NUM_THREADS": "1"}
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------------------
# CodeAlpaca-20k -- a PROMPT POOL, not a benchmark.  Used only to generate
# self-distillation trajectories from the target (gen_distill.py).  Chosen because it is
# instruction-style code data disjoint from MBPP / GSM8K / HumanEval by construction rather
# than by audit; the overlap check runs anyway (check_contamination.py).  It has no scorer
# because nothing is ever scored on it -- scoring it would mean it had become an eval set.
# ----------------------------------------------------------------------------------
def load_codealpaca(limit: Optional[int] = None, offset: int = 0, **kw) -> List[Sample]:
    path = os.path.join(_snapshot("datasets--sahil2801--CodeAlpaca-20k"),
                        "code_alpaca_20k.json")
    with open(path) as f:
        rows = json.load(f)
    rows = rows[offset : offset + limit] if limit else rows[offset:]
    out = []
    for i, r in enumerate(rows):
        q = r["instruction"]
        if r.get("input"):
            q = q + "\n\n" + r["input"]
        out.append(Sample(f"codealpaca-{offset+i}", [{"role": "user", "content": q}], None, {}))
    return out


def score_never(sample: Sample, text: str, **kw) -> float:
    raise AssertionError("codealpaca is a prompt pool for distillation, not an eval set")


# datasets must land in the project store regardless of which module is imported
# first (the race: login-side smoke cached to ~/.cache, compute nodes re-downloaded
# concurrently and corrupted the arrow build). Mirror of the guard in models.py.
import os as _os
_HF_STORE = os.environ.get("SPOT_HF_STORE", os.path.expanduser("~/.cache/huggingface/hub"))
if _os.path.isdir(_HF_STORE):
    _os.environ.setdefault("HF_HOME", _HF_STORE)


def load_ultrachat(limit=None, offset: int = 0, **kw):
    """UltraChat-200k prompts -- the EAGLE-3/DFlash training pool. PROMPT POOL ONLY:
    broad-domain chat/instruction, disjoint from every eval benchmark. Added
    to fix the drafter's single-domain diet (CodeAlpaca-only), which the decode
    loss traced to token-identity features not transferring across domains."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    hi = len(ds) if limit is None else min(offset + limit, len(ds))
    return [Sample(f"ultrachat-{i}", [{"role": "user", "content": ds[i]["prompt"]}],
                   None, {})
            for i in range(offset, hi)]


def load_nemotron(limit=None, offset: int = 0, split: str = "chat", **kw):
    """Nemotron Post-Training V2 prompts (DFlash's production-blend base). PROMPT POOL
    ONLY. Splits: chat | code | math. Field layout probed at runtime -- the corpus
    stores conversations under 'messages' (first user turn is the prompt)."""
    # Direct single-shard parquet read. Three prior approaches failed on this corpus:
    # bulk load_dataset blew 92 GB and broke its arrow build; the gated original needs
    # auth; streaming stalled >50 min before yielding a row. One shard per split is
    # bounded (few hundred MB), cached in the store, and holds far more prompts than
    # any pool draw we make.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    assert split in ("code", "chat"), "single-shard loader wired for code/chat only"
    shard = hf_hub_download(
        "openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated",
        f"data/{split}-00000-of-{'00002' if split == 'code' else '00012'}.parquet",
        repo_type="dataset")
    tbl = pq.read_table(shard, columns=None)
    rows = tbl.to_pylist()
    out = []
    for i, row in enumerate(rows[offset:], start=offset):
        q = None
        if row.get("messages"):
            for m in row["messages"]:
                if (m or {}).get("role") == "user":
                    q = m.get("content"); break
        q = q or row.get("prompt") or row.get("input")
        # Prompt pool, not a benchmark: skip pathological multi-KB user turns -- one
        # chat-split prompt OOMed the 8B forward on a 48 GB card.
        if q and len(q) <= 4000:
            out.append(Sample(f"nemotron-{split}-{i}",
                              [{"role": "user", "content": q}], None, {}))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_nemotron_code(limit=None, offset=0, **kw):
    return load_nemotron(limit, offset, split="code", **kw)


TASKS = {
    "codealpaca": (load_codealpaca, score_never, "none"),
    "ultrachat": (load_ultrachat, score_never, "none"),
    "nemotron_chat": (load_nemotron, score_never, "none"),
    "nemotron_code": (load_nemotron_code, score_never, "none"),
    "mbpp": (load_mbpp, score_mbpp, "pass@1"),
    "humaneval": (load_humaneval, score_humaneval, "pass@1"),
    "gsm8k": (load_gsm8k, score_gsm8k, "exact_match"),
    "math500": (load_math500, score_math500, "exact_match"),
}


def load_task(name: str, limit: Optional[int] = None, offset: int = 0, **kw) -> List[Sample]:
    return TASKS[name][0](limit=limit, offset=offset, **kw)


def score_task(name: str, sample: Sample, text: str) -> float:
    return TASKS[name][1](sample, text)
