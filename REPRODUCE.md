# Reproducing the paper

Each table and figure, and the commands that made it. Every command below is the one the paper's run recorded
(`provenance.argv` in its `config.json`), with the drafter and graph files renamed to the ones in this repository
(each is byte-identical to the file the run used) and the output folder moved under `runs/`. Replace `{task}` with
each task listed and `{seeds}` with the seeds listed; every run writes its own draft-free (budget 0) baseline when
budget 0 is in `--budgets`.

Then, for the mean and standard deviation over seeds of forwards per sample, throughput ratio and accuracy:

```
python analysis/aggregate_seeds.py runs/<folder>/<task>/<arm>
```

Throughput depends on the GPU: the paper's runs used one NVIDIA RTX 6000 Ada per run. Forward counts and accuracy
carry over between GPUs up to floating-point noise.

## Table 2 (T = 0 block), Tables 18-19 (all budgets), Table 20 and Figure 3 (budget response): SDAR-8B b32, T = 0

Folder `runs/T3_sdar8b/`, 25 run configurations.

- tasks: gsm8k; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/free
```

- tasks: gsm8k, math500; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: gsm8k; seeds: 0

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,12,20 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/free
```

- tasks: humaneval; seeds: 0

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0 \
  --limit 164 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/spiffy
```

- tasks: math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 12 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/conftree
```

- tasks: math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/conftree
```

- tasks: math500; seeds: 0, 1

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: math500; seeds: 0, 1

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 12 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: mbpp; seeds: 0,1,2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --pad-pieces -1 \
  --verify match \
  --seeds {seeds} \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --temperature 0.0 \
  --budgets 0,6,8,12,16,20 \
  --extra-seed-budgets 0,6,8,12,16,20 \
  --out-root runs/T3_sdar8b/{task}/ours
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.02
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.05 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.05
```

- tasks: mbpp; seeds: 0, 1

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_bf \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 64 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1_fast
```

- tasks: mbpp; seeds: 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_bf \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 64 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1_fast
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode pl \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 64 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.1_pl
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.2 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0.2
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_nopad_g0
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b/{task}/ours_pad_g0.1
```

## Table 2 (T = 1 block): SDAR-8B b32, T = 1

Folder `runs/T3_sdar8b_T1/`, 14 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/free
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/free_rs
```

- tasks: gsm8k, math500; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/ours
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/ours_nopad_g0.1
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/free
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/free_rs
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/ours_nopad_g0.1
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/spiffy
```

- tasks: mbpp; seeds: 0

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --pad-pieces -1 \
  --verify match \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --temperature 1.0 \
  --budgets 0 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T3_sdar8b_T1/{task}/ours
```

- tasks: mbpp; seeds: 0,1,2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --pad-pieces -1 \
  --verify match \
  --seeds {seeds} \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --temperature 1.0 \
  --budgets 0,6,8,12,16,20 \
  --extra-seed-budgets 0,6,8,12,16,20 \
  --out-root runs/T3_sdar8b_T1/{task}/tree_match
```

## Table 8: SDAR-4B b32, T = 0 (Appendix F.3)

Folder `runs/TS_sdar4b/`, 15 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/free
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours_nopad_g0.02
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/spiffy
```

- tasks: gsm8k; seeds: 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/free
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours_nopad_g0.02
```

- tasks: humaneval; seeds: 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 20 \
  --limit 164 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours_nopad_g0.02
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/spiffy
```

- tasks: math500; seeds: 0

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/conftree
```

- tasks: math500; seeds: 1

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/free
```

- tasks: math500; seeds: 1

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b/{task}/ours
```

## Table 9: SDAR-4B b32, T = 1 (Appendix F.3)

Folder `runs/TS_sdar4b_T1/`, 12 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/free
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/free_rs
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/ours
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/ours_nopad_g0.02
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/free
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/free_rs
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/ours_nopad_g0.02
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar4b_T1/{task}/spiffy
```

## Table 10: SDAR-1.7B b32, T = 0 (Appendix F.3)

Folder `runs/TS_sdar17/`, 14 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/free
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours
```

- tasks: gsm8k; seeds: 0

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours_nopad_g0.02
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours_nopad_g0.02
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/free
```

- tasks: humaneval; seeds: 0

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --limit 164 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours
```

- tasks: humaneval; seeds: 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours_nopad_g0.02
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/spiffy
```

- tasks: math500; seeds: 0

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/ours
```

- tasks: math500; seeds: 1

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17/{task}/spiffy
```

## Table 11: SDAR-1.7B b32, T = 1 (Appendix F.3)

Folder `runs/TS_sdar17_T1/`, 12 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/conftree
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/free
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/free_rs
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/ours
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/ours_nopad_g0.02
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/spiffy
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/conftree
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/free
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/free_rs
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/ours
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/ours_nopad_g0.02
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/TS_sdar17_T1/{task}/spiffy
```

## Table 12: LLaDA2.0-mini b32, T = 0 (Appendix F.3)

Folder `runs/T4_mini/`, 2 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T4_mini/{task}/ours
```

- tasks: humaneval; seeds: 0

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T4_mini/{task}/ours
```

## Tables 3 and 14: checkpoints pointed at other targets (Section 5.4, Appendix F.3)

Folder `runs/T9_transfer/`, 48 run configurations.

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model dream \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k; seeds: 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: gsm8k, math500, mbpp; seeds: 0

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: humaneval; seeds: 0

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: mbpp; seeds: 0

```
python run_sweep.py --model llada2 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/llada2mini_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k; seeds: 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: humaneval; seeds: 0, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 1

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 164 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: math500; seeds: 1

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: mbpp; seeds: 2

```
python run_sweep.py --model sdar17 \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar1.7b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k; seeds: 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: humaneval; seeds: 0, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 1

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --limit 164 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar4b \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar4b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: gsm8k, math500, mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/llada2mini_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_mini
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar1.7b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar17
```

- tasks: humaneval; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 164 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar4b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/T9_transfer/{task}/drafter_sdar4b
```

## Table 17: ablations on SDAR-8B MBPP (Section 5.5, Appendix F.6)

Folder `runs/T6_ablation/`, 5 run configurations.

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.02 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T6_ablation/{task}/gamma0.02
```

- tasks: mbpp; seeds: 0,1,2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --pad-pieces -1 \
  --verify match \
  --seeds {seeds} \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0 \
  --temperature 0.0 \
  --budgets 0,6,8,12,16,20 \
  --extra-seed-budgets 0,6,8,12,16,20 \
  --out-root runs/T6_ablation/{task}/no_tokgate
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode purechain \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T6_ablation/{task}/purechain
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-tau 1.3 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T6_ablation/{task}/tau1.3
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/tuned/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/T6_ablation/{task}/unscaled_data
```

## Table 22: verifier and structure at T = 1, SDAR-8B MBPP (Appendix F.7)

Folder `runs/T7_verifiers/`, 6 run configurations.

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode purechain \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T7_verifiers/{task}/chain_match
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode purechain \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T7_verifiers/{task}/chain_rs
```

- tasks: mbpp; seeds: 0,1,2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --pad-pieces -1 \
  --verify match \
  --seeds {seeds} \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --temperature 1.0 \
  --budgets 0,6,8,12,16,20 \
  --extra-seed-budgets 0,6,8,12,16,20 \
  --out-root runs/T7_verifiers/{task}/tree_match
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode purechain \
  --slot-ckpt drafters/tuned/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T7_verifiers/{task}/chain_match
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode purechain \
  --slot-ckpt drafters/tuned/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T7_verifiers/{task}/chain_rs
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/tuned/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 8,12,20 \
  --seeds {seeds} \
  --out-root runs/T7_verifiers/{task}/tree_match
```

## Tables 1 and 21: temperature sweep, SDAR-8B MBPP (Section 3.2, Appendix F.6)

Folder `runs/Tsweep_sdar8b/`, 27 run configurations.

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 0.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_coupled
```

- tasks: mbpp; seeds: 0

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_coupled
```

- tasks: mbpp; seeds: 0, 1

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --verify-greedy \
  --budgets 8 \
  --limit 350 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_greedy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/ours_uncoupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_coupled
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 256 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 2.5 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --draft-noise none \
  --budgets 8 \
  --seeds {seeds} \
  --out-root runs/Tsweep_sdar8b/{task}/spiffy_uncoupled
```

## Table 16: generation length, SDAR-8B MBPP, T = 1 (Appendix F.5)

Folder `runs/Tlen_sdar8b_T1/`, 12 run configurations.

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/conftree
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/free
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/free_rs
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/ours
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/ours_nopad_g0.1
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 512 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/spiffy
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode conftree \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/conftree
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/free
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify rs \
  --pad-pieces -1 \
  --draft free \
  --free-vocab-ranks 1 \
  --free-order sampler \
  --free-max-depth 4 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/free_rs
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --budgets 0,8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/ours
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces 0 \
  --draft slot \
  --slot-mode ddtree_b \
  --slot-ckpt drafters/sdar8b_b32.pt \
  --slot-tokgate 0.6 \
  --slot-gamma 0.1 \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/ours_nopad_g0.1
```

- tasks: mbpp; seeds: 0, 1, 2

```
python run_sweep.py --model sdar \
  --block-length 32 \
  --gen-length 64 \
  --layout concat \
  --cache on \
  --unmasking single \
  --suffix skip \
  --graph graphs/sdar8b_b32_D24.json \
  --task {task} \
  --limit 350 \
  --temperature 1.0 \
  --verify match \
  --pad-pieces -1 \
  --draft spiffy \
  --budgets 8,20 \
  --seeds {seeds} \
  --out-root runs/Tlen_sdar8b_T1/{task}/spiffy
```

## Not mapped here

These come from development and analysis runs outside this release, and their commands are not listed: Figure 2 and
Table 7 (the live-oracle grid, Appendix F.2), Table 4 (the mass-floor calibration, Appendix D.5), Table 15 (the recipe
against per-setting drafters, which are in `drafters/tuned/`), Table 13 (LLaDA2.0-mini at T = 1), Figures 4 and 5 (the
mechanism and error analyses, Appendices F.8 and F.9), Table 5 (MRP's reported numbers) and Table 6 (related work,
no experiment).
