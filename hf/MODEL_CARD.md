---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - speculative-decoding
  - diffusion-language-model
  - masked-diffusion
  - block-diffusion
  - drafter
  - lossless-acceleration
---

<div align="center">

# SPOT drafters

**On the SPOT: Enhancing Speculative Decoding for Diffusion Language Models via Commit-Order Prediction**

Eliezer Shlomi<sup>1,2\*</sup>, Eliron Rahimi<sup>1,2\*</sup>, Avi Mendelson<sup>1</sup>, Chaim Baskin<sup>2</sup>

<sup>1</sup>Technion – Israel Institute of Technology<br>
<sup>2</sup>INSIGHT Lab, School of Electrical and Computer Engineering, Ben-Gurion University of the Negev

<sub><sup>\*</sup>Equal contribution. Correspondence: {eliezer, elironrahimi}@campus.technion.ac.il</sub>

[![Project page](https://img.shields.io/badge/Project-page-1D4ED8)](https://eliezer318.github.io/SPOT-dLLM/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.22543-B31B1B)](https://arxiv.org/abs/2610.22543)
[![Code](https://img.shields.io/badge/GitHub-SPOT--dLLM-181717?logo=github)](https://github.com/Eliezer318/SPOT-dLLM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliezer318/SPOT-dLLM/blob/main/colab/spot_quickstart.ipynb)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Eliezer318/SPOT-dLLM/blob/main/LICENSE)

</div>

![Recorded SDAR-8B runs on one GSM8K prompt: draft-free decoding, the Spiffy baseline, SPOT and SPOT-γ write the same 256 tokens; SPOT finishes in a quarter of the forwards.](assets/spot_replay.gif)

Commit-order drafters for **SPOT** (Speculating Positions Order Trajectory). Each checkpoint is the same
0.62M-parameter order model (3 layers, `d_model` 128, 8 heads, stale top-32 inputs). It reads only the target's
own output distributions from the previous pass and predicts which masked position a block-diffusion target
commits next. SPOT grows those predictions into a tree of candidate block states, and the target verifies the
whole tree in one forward, exactly: an accepted state equals the target's own next state, and
coupled-noise match keeps this exact at any temperature. On SDAR-8B this cuts target passes by up to 6.8× and
reaches up to 3.2× the throughput of standard decoding, with the target's output unchanged.
Paper: [arXiv:2610.22543](https://arxiv.org/abs/2610.22543).

## Files

| file | target | block | held-out next-commit acc. |
|---|---|---|---|
| `sdar8b_b32.pt` | SDAR-8B-Chat-b32 | 32 | 0.750 |
| `sdar8b_b16.pt` | SDAR-8B-Chat-b16 | 16 | 0.746 |
| `sdar4b_b32.pt` | SDAR-4B-Chat-b32 | 32 | 0.669 |
| `sdar1.7b_b32.pt` | SDAR-1.7B-Chat-b32 | 32 | 0.665 |
| `llada2mini_b32.pt` | LLaDA2.0-mini | 32 | 0.636 |
| `llada2mini_b16.pt` | LLaDA2.0-mini | 16 | 0.665 |

`tuned/` holds the per-setting comparison drafters of the paper's appendix; the six checkpoints above are all you need to use SPOT.
A drafter can also drive other targets directly; LLaDA-8B and Dream-7B run with transferred drafters.

## Results

SDAR-8B, block width 32, generation length 256, draft budget 8, T = 0: throughput ratio over draft-free
decoding (1.00), mean of three seeds (Table 2 of the paper).

| Task | Spiffy graph | SPOT | SPOT-γ |
|---|---|---|---|
| GSM8K | 1.96 | 2.50 | **2.76** |
| MBPP | 1.86 | 2.61 | **2.98** |
| HumanEval | 2.11 | 2.74 | **3.09** |
| MATH500 | 1.80 | 2.03 | **2.27** |

At budget 20, SPOT needs 4.1–6.8× fewer target forwards than draft-free decoding on SDAR-8B. At T = 1 it uses
15–34% fewer forwards than Spiffy at 1.15–1.48× its throughput.

## Limitations

- Gains shrink when verification is costly or acceptance is low: at wide blocks, on low-acceptance tasks, and at
  T ≥ 2, where noise-aware ranking beats the order model trained at T = 0.
- On SDAR-1.7B, where a target forward is cheap, Spiffy has the higher throughput in three columns at T = 0 and
  two at T = 1, with one tie.
- LLaDA-8B and Dream-7B run with an approximate prefix cache, so those results measure reach; losslessness is
  shown on the exact-cache targets.
- Exact tree packing under bidirectional attention remains open.

## Use

The checkpoints are loaded by the decoder in the [GitHub repository](https://github.com/Eliezer318/SPOT-dLLM):

```bash
git clone https://github.com/Eliezer318/SPOT-dLLM.git && cd SPOT-dLLM
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
huggingface-cli download HF_REPO_ID --local-dir drafters
python run_sweep.py --model sdar --block-length 32 --gen-length 256 --task gsm8k --limit 100 \
  --temperature 0.0 --verify match --layout concat --cache on --unmasking single --suffix skip \
  --pad-pieces -1 --draft slot --slot-mode ddtree_b --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 --budgets 0,8 --seeds 0 --out-root runs/sdar8b/gsm8k/spot
```

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
