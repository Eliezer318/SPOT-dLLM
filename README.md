<div align="center">

<p>
  <a href="https://www.technion.ac.il/en/"><img src="docs/assets/logos/technion-wide.svg" alt="Technion – Israel Institute of Technology" height="46"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://www.bgu.ac.il/en/"><img src="docs/assets/logos/bgu-wide.svg" alt="Ben-Gurion University of the Negev" height="46"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://insight.bgu.ac.il/"><img src="docs/assets/logos/insight-wide.png" alt="INSIGHT Lab, Ben-Gurion University of the Negev" height="40"></a>
</p>

# On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction

**Eliezer Shlomi**<sup>1,2\*</sup>, **Eliron Rahimi**<sup>1,2\*</sup>, **Avi Mendelson**<sup>1</sup>, **Chaim Baskin**<sup>2</sup>

<sup>1</sup>Technion – Israel Institute of Technology<br>
<sup>2</sup>INSIGHT Lab, School of Electrical and Computer Engineering, Ben-Gurion University of the Negev

<sub><sup>\*</sup>Equal contribution. Correspondence: {eliezer, elironrahimi}@campus.technion.ac.il</sub>

[![Project page](https://img.shields.io/badge/Project-page-1D4ED8)](https://eliezer318.github.io/SPOT-dLLM/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.22543-B31B1B)](https://arxiv.org/abs/2610.22543)
[![Drafters](https://img.shields.io/badge/%F0%9F%A4%97%20Drafters-SPOT--dLLM-FFD21E)](https://huggingface.co/EliezerS/SPOT-dLLM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliezer318/SPOT-dLLM/blob/main/colab/spot_quickstart.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)

</div>

<p align="center">
  <img src="docs/assets/spot_replay.gif" width="96%" alt="Recorded SDAR-8B runs on one GSM8K prompt, replayed at twice the recorded wall-clock speed: draft-free decoding, the Spiffy baseline, SPOT and SPOT-γ. All four write the same 256 tokens.">
</p>
<p align="center"><sub>Recorded SDAR-8B runs on one GSM8K prompt (T = 0, block 32, budget 8), replayed at 2× the measured
wall-clock time. All four arms write the identical 256 tokens; bright cells were written from an accepted draft.
The <a href="https://eliezer318.github.io/SPOT-dLLM/">project page</a> has an interactive version.</sub></p>

> **TL;DR.** SPOT speeds up diffusion language models losslessly. A 0.62M-parameter drafter predicts the order
> in which the model commits positions, and the model verifies a whole tree of drafts in one pass, exactly at any
> temperature. It cuts target passes by up to 6.8×, reaches up to 3.4× the throughput of standard decoding, and
> its drafters transfer across six diffusion LMs. **SPOT-Sets** carries the idea to dynamic threshold decoding:
> it drafts whole commit sets and raises SDAR-8B's throughput by 59% over the threshold decoder alone.

## 📰 News

- **2026-10** · Paper v2 adds **SPOT-Sets** for dynamic threshold decoding: code in `spot/sets.py`, its drafters
  for SDAR-8B, SDAR-4B, LLaDA2.0-mini, LLaDA-8B and Dream-7B on [🤗 SPOT-dLLM](https://huggingface.co/EliezerS/SPOT-dLLM) (`sets/`).
- **2026-10** · Code, the drafter checkpoints ([🤗 SPOT-dLLM](https://huggingface.co/EliezerS/SPOT-dLLM)) and the
  [project page](https://eliezer318.github.io/SPOT-dLLM/) are being prepared for release.

## ✨ Highlights

- **Order is the key.** A diffusion LM chooses both *what* to write and *where*. The previous pass already
  has the right token at 92–100 % of positions, so a draft succeeds by predicting which position the target
  commits next.
- **A tiny learned drafter.** A 0.62M-parameter order model reads only the target's own output
  distributions, predicts the next commit, and grows a tree of candidate block states.
- **One forward verifies the whole tree, exactly.** Every candidate state is a separate copy of the block
  behind the cached prefix; an accepted draft equals the target's own next state. Coupled-noise
  match (shared Gumbel noise) keeps this exact at T > 0, so the output distribution is unchanged.
- **Transfers.** Its drafters transfer across six DLMs.

## 📊 Results

SDAR-8B (block 32), 256 generated tokens, draft budget 8, T = 0. Throughput relative to the draft-free
one-token-per-step control of the same target (target forwards per sample in brackets). All methods are exact,
so accuracy is the target's own (GSM8K 0.89–0.90, MBPP 0.43, HumanEval 0.71–0.74, MATH500 0.10–0.12).

| Method | GSM8K | MBPP | HumanEval | MATH500 |
|---|---|---|---|---|
| Draft-free | 1.00× (270.8) | 1.00× (269.2) | 1.00× (271.4) | 1.00× (271.8) |
| **SPOT** | 2.95× (**70.1**) | 3.38× (**56.7**) | 3.35× (**62.5**) | 2.50× (**84.6**) |
| **SPOT-γ** (mass floor 0.02) | **3.03×** (70.1) | **3.48×** (56.9) | **3.42×** (62.6) | **2.53×** (84.7) |
| Spiffy graph | 2.48× (92.5) | 2.40× (83.6) | 2.64× (87.6) | 2.29× (100.6) |
| SSD free chain | 2.44× (92.4) | 2.39× (84.5) | 2.63× (87.0) | 2.28× (100.6) |
| BlockSpec tree | 1.64× (136.5) | 1.46× (135.1) | 1.65× (136.0) | 1.64× (136.8) |

n = 350 prompts per task (HumanEval 164), mean of three seeds, one job per NVIDIA RTX PRO 6000 Blackwell card
(Table 2 of the paper).

- **Sampling (T = 1).** With coupled-noise match, SPOT-γ keeps 2.20–3.27× at budgets 8 and 20 while preserving the
  target's sampling distribution; SPOT takes 16–35% fewer forwards than Spiffy at 1.08–1.37× its throughput.
- **Transfer.** SDAR-8B, 4B and 1.7B, LLaDA2.0-mini, LLaDA-8B and Dream-7B, each driven by a drafter trained on
  another model.

**SPOT-Sets (threshold decoding, τ = 0.9).** A faster decoder commits every position above a confidence bar, so a
step is a set of positions. SPOT-Sets drafts these steps whole, with a 5.1M-parameter set drafter trained on general
chat prompts only (ShareGPT and UltraChat-200K). SDAR-8B, block 32, 256 tokens, budget 8, T = 0, mean of three
seeds; throughput over draft-free one-commit decoding (target forwards per sample in brackets; Table 2 of the
paper).

| Method | GSM8K | MBPP | HumanEval | MATH500 |
|---|---|---|---|---|
| Threshold decoder, no drafts | 4.51× (58.8) | 5.89× (44.0) | 6.13× (43.4) | 3.34× (80.1) |
| **SPOT-Sets-γ** | **7.30×** (**30.6**) | **9.28×** (**22.8**) | **9.85×** (**22.7**) | **5.24×** (**42.3**) |
| SPOT-Sets-γ, exact | 6.93× (32.5) | 8.97× (23.8) | 9.35× (24.0) | 4.96× (45.1) |

SPOT-Sets-γ accepts with max-commit (the product rule), which can change the output; the exact variant keeps the
threshold decoder's output.

## 🚀 Quick start

**No GPU of your own?** The [Colab notebook](https://colab.research.google.com/github/Eliezer318/SPOT-dLLM/blob/main/colab/spot_quickstart.ipynb) runs SDAR-1.7B with its drafter on a free Colab GPU
and compares draft-free decoding with SPOT on your own prompt and on a few GSM8K questions.

### Install

```
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
```

Targets are pulled from the Hugging Face hub on first use. `SPOT_HUB` and `SPOT_HF_STORE`
point at the snapshot cache if it is not in the default location; `SDAR_PATH`,
`LLADA2_PATH`, `LLADA_PATH` and `DREAM_PATH` override the repo id or local snapshot of each
target. `vendor/` holds the two patched model definitions the exact prefix cache needs, and
is imported from the repository root, so run the commands from there.

### Decode

One `(task, drafter, budget)` grid on a single GPU, writing a run directory per budget and
seed with generations, scores, the per-round acceptance log and a distribution-preserving
verdict:

```
python run_sweep.py --model sdar --block-length 32 --gen-length 256 \
  --task mbpp --limit 100 --temperature 0.0 --verify match \
  --layout concat --cache on --unmasking single --suffix skip --pad-pieces -1 \
  --draft slot --slot-mode ddtree_b --slot-ckpt drafters/sdar8b_b32.pt --slot-tokgate 0.6 \
  --budgets 0,8,20 --seeds 0 --out-root runs/sdar8b/mbpp/spot
```

Arms are selected with `--draft`:

| flag | arm |
|---|---|
| `--draft slot --slot-mode ddtree_b` | SPOT, the learned commit-order tree |
| `--draft slot --slot-mode ddtree_b --pad-pieces 0 --slot-gamma G` | SPOT with the mass floor |
| `--draft slot --slot-mode purechain` | the same drafter as a chain |
| `--draft slot --slot-mode conftree` | confidence tree |
| `--draft spiffy --graph graphs/sdar8b_b32_D24.json` | Spiffy baseline: calibrated static draft graph |
| `--draft free --free-vocab-ranks 1 --free-order sampler --free-max-depth 4` | confidence-ordered free chain |
| `--budgets 0` (no `--draft`) | draft-free control |
| `--draft sets --unmasking threshold` | SPOT-Sets: drafted commit sets for threshold decoding (below) |

`--verify match` is state equality, coupled to the target's sampler at `T>0`;
`--verify rs` is the rejection-sampling verifier, defined for chains only.

`--unmasking single` commits one position per forward, the exact regime of the paper's results.

**SPOT-Sets** runs on threshold decoding; its drafter is fetched from Hugging Face on first use. Budget 0 is the
threshold decoder without drafts:

```
python run_sweep.py --model sdar --block-length 32 --gen-length 256 \
  --task mbpp --limit 100 --temperature 0.0 \
  --layout concat --cache on --unmasking threshold --threshold 0.9 \
  --draft sets --sets-accept product --cross-block on \
  --budgets 0,8 --seeds 0 --out-root runs/sdar8b/mbpp/sets
```

`--sets-accept exact` keeps the threshold decoder's output; `product`, `subset` and `cut` are the max-commit rules,
from loosest to strictest. `--cross-block on` lets a draft that completes the block carry the next one, and the
next block's first forward verify drafts too. For another target, pass its `--model` and its `--sets-ckpt` from
the [SPOT-Sets table](#drafter-checkpoints).

<a id="drafter-checkpoints"></a>

## 🤗 Drafter checkpoints


Every drafter is the same 0.62M-parameter order model (3 layers, `d_model` 128, 8 heads,
stale top-32 inputs), trained with the recipe below on the target's own draft-free
trajectories. Pass the matching `--model` and `--block-length`.

| checkpoint | target | `--model` | `--block-length` | held-out next-commit acc. |
|---|---|---|---|---|
| `drafters/sdar8b_b32.pt` | SDAR-8B-Chat-b32 | `sdar` | 32 | 0.750 |
| `drafters/sdar8b_b16.pt` | SDAR-8B-Chat-b16 | `sdar_b16` | 16 | 0.746 |
| `drafters/sdar4b_b32.pt` | SDAR-4B-Chat-b32 | `sdar4b` | 32 | 0.669 |
| `drafters/sdar1.7b_b32.pt` | SDAR-1.7B-Chat-b32 | `sdar17` | 32 | 0.665 |
| `drafters/llada2mini_b32.pt` | LLaDA2.0-mini | `llada2` | 32 | 0.636 |
| `drafters/llada2mini_b16.pt` | LLaDA2.0-mini | `llada2` | 16 | 0.665 |

The accuracy column is each drafter's next-commit accuracy on its own held-out trajectories at training
time; the paper reports the SDAR-8B b32 value (75%, Section 3.1).

A checkpoint can also drive a different target directly (the transfer experiments);
LLaDA-8B (`--model llada`) and Dream-7B (`--model dream`) run with the checkpoints above.

`drafters/tuned/` holds the per-setting comparison drafters of the appendix (a drafter
tuned separately for one target and width, same architecture): `sdar8b_b32`,
`sdar8b_b16`, `llada2mini_b32`, `llada2mini_b16`. The six checkpoints above are all you need to use SPOT.

The SDAR-4B, SDAR-1.7B and SDAR-8B-b16 targets are read from the local snapshot cache;
download them first, e.g. `huggingface-cli download JetLM/SDAR-4B-Chat-b32`.

The SPOT-Sets drafters (threshold 0.9, block 32) are on Hugging Face only and are fetched on first use. Each file
holds the set drafter's weights together with the target's token-embedding PCA and hidden-state basis it reads.

| `--sets-ckpt` | target | `--model` | drafter |
|---|---|---|---|
| `drafters/sets/sdar8b_b32.pt` | SDAR-8B-Chat-b32 | `sdar` | trained on general chat |
| `drafters/sets/sdar4b_b32.pt` | SDAR-4B-Chat-b32 | `sdar4b` | trained on general chat |
| `drafters/sets/llada2mini_b32.pt` | LLaDA2.0-mini | `llada2` | trained on general chat |
| `drafters/sets/llada8b_b32.pt` | LLaDA-8B-Instruct | `llada` | the SDAR-8B drafter, calibrated on 98 target prompts |
| `drafters/sets/dream_b32.pt` | Dream-v0-Instruct-7B | `dream` | the SDAR-8B drafter, calibrated on 98 target prompts |

The same checkpoints, with a model card, are on Hugging Face at [🤗 EliezerS/SPOT-dLLM](https://huggingface.co/EliezerS/SPOT-dLLM):

```bash
huggingface-cli download EliezerS/SPOT-dLLM --local-dir drafters
```

## 🧭 How it works

```
   stale top-32 distributions (previous verified pass)
                 │
                 ▼
     ┌──────────────────────┐   tree of candidate     ┌───────────────────────────────┐
     │  order model (0.62M) │ ──── block states ────▶ │ target: one forward, one copy │
     │  predicts next commit│                         │ of the block per candidate    │
     └──────────────────────┘                         └───────────────┬───────────────┘
                 ▲                                                    │ accept the deepest state that
                 └────── conditionals of the accepted copy ◀──────────┘ equals the target's own next state
```

- **Draft.** The order model reads the stale distributions of each masked position, predicts which position the
  target commits next, conditions on the positions a branch has already filled, and grows a tree best-first by
  prefix mass. Tokens come from the stale argmax (or the coupled-noise draw at T > 0).
- **Verify.** Every node is appended as its own copy of the block behind the cached prefix. The target applies its
  own commit rule to every copy in one forward; a node is accepted only if it equals the target's next state, so
  the decoded output is the target's.
- **Repeat.** The accepted copy's conditionals feed the next round's drafter.

## 🔬 Reproduce

Every table and figure in the paper, with the exact commands its runs recorded, is in
[REPRODUCE.md](REPRODUCE.md). A smoke test that needs no data or GPU checks the install in a few seconds:

```
python -m unittest discover -s tests -v
```

### Baseline draft graphs

`graphs/` holds the calibrated Spiffy draft graphs used for the baseline, one per target
and width (budget 24, generation length 256, calibrated on MBPP-train and MATH-500 prompts
disjoint from the evaluated ones, 18 nodes each): `sdar8b_b32_D24.json`,
`sdar8b_b16_D24.json`, `sdar4b_b32_D24.json`, `sdar1.7b_b32_D24.json`,
`llada2mini_b32_D24.json`, `llada2mini_b16_D24.json`. `run_sweep.py` refuses a graph
calibrated for a different model family.

### Train a drafter

```
python gen_distill.py --model sdar --block-length 32 --limit 400 --out data/traj_sdar8b_b32
python train_slotpicker.py --data data/traj_sdar8b_b32 --arch champ2 --prefix-cond \
  --m-top 32 --d-model 128 --layers 3 --heads 8 --epochs 12 --out drafters/sdar8b_b32.pt
```

The block length is read from the trajectory metadata, so the drafter matches the width it
was collected at; `--prefix-cond` makes it the recursive tree model.

The same command with `--model` and `--block-length` changed produces the other
checkpoints, with the same settings for each.

### Aggregate seeds

```
python analysis/aggregate_seeds.py runs/sdar8b/mbpp/spot
```

Writes `aggregate.json` with per-seed values plus mean and standard deviation for forwards
per sample, throughput and accuracy. Ratios are taken per seed against that seed's own
draft-free run.

## 🗂 Repository layout

| Path | Contents |
|---|---|
| `run_sweep.py` | Decoding and evaluation entry point (forwards, wall clock, accuracy): one (task, drafter, budget) grid per call |
| `gen_distill.py` | Collect draft-free target trajectories for training |
| `train_slotpicker.py` | Train the commit-order model |
| `spot/decode.py` | Block-diffusion decoding loop and verification |
| `spot/drafters.py` | Draft-tree construction, including the level-batched search |
| `spot/sets.py` | SPOT-Sets: set drafter, step tree, max-commit and the threshold-decoding loop |
| `spot/pickers.py`, `spot/slotpicker.py` | Order-model architectures and stale-distribution features |
| `spot/sampling.py` | Coupled-noise sampling and commit laws |
| `spot/models.py`, `spot/tasks.py` | Target adapters; benchmark loading and scoring |
| `spot/graphs.py`, `graphs/` | Calibrated draft graphs for the Spiffy baseline |
| `vendor/` | Patched SDAR and LLaDA model definitions for the exact prefix cache |
| `drafters/` | Trained order models, one per target and block width |
| `analysis/aggregate_seeds.py` | Per-seed aggregation of runs |
| `docs/` | Project page (`index.html`) and the scripts that build it |
| `hf/` | Hugging Face model card and the upload script for the drafters |

## Citation

```bibtex
@article{shlomi2026spot,
  title         = {On the {SPOT}: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction},
  author        = {Shlomi, Eliezer and Rahimi, Eliron and Mendelson, Avi and Baskin, Chaim},
  journal       = {arXiv preprint arXiv:2610.22543},
  year          = {2026},
  doi           = {10.48550/arXiv.2610.22543},
  eprint        = {2610.22543},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.22543}
}
```

## License

The code and the drafter checkpoints are released under the [MIT License](LICENSE). The model definitions
under `vendor/` are third-party code kept under their original licenses (SDAR: Apache-2.0, see the file
headers; LLaDA: generated from the GSAI-ML/LLaDA-8B-Instruct snapshot, see `vendor/llada/SOURCE.txt`).
