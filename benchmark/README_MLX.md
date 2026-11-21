# Pusula and Decision 3.0 on Apple silicon (MLX)

`pusula.mlx_d3` runs Decision 3.0 (`vllm-sr/d3`) and the Pusula adapter on an Apple silicon Mac with
[MLX](https://github.com/ml-explore/mlx), without a GPU server. It converts the base d3 model to 4, 6 or 8 bits
once and applies the Pusula adapter at run time. The same converted model answers with the adapter on (Pusula) or
off (base Decision 3.0), so the `own_base` routing of the published setup is kept.

## MLX results (4, 6 and 8-bit)

Full [Turkish Decision Benchmark](README.md) (19,861 questions), on an Apple M3 Max with 36 GB unified memory.
Micro accuracy / macro accuracy / ECE (15 bins), computed as `d3train eval` computes them. The published BF16
results in the main README are unchanged; the GPU BF16 row below is the same evaluation on a GPU, for comparison.

| run | weights | base Decision 3.0 | Pusula | Pusula gain (micro) |
|---|---|---|---|---|
| MLX 4-bit | 14.4 GB | 86.92 / 88.08 / 0.009 | 91.39 / 92.20 / 0.032 | +4.47 |
| MLX 6-bit | 20.9 GB | 87.48 / 88.50 / 0.011 | 91.42 / 92.23 / 0.031 | +3.94 |
| MLX 8-bit | 27.3 GB | 87.44 / 88.47 / 0.008 | 91.38 / 92.20 / 0.032 | +3.94 |
| GPU BF16 | | 87.50 / 88.52 / 0.009 | 91.41 / 92.22 / 0.031 | +3.91 |

Accuracy per question type (base Decision 3.0 / Pusula):

| question | MLX 4-bit | MLX 6-bit | MLX 8-bit | GPU BF16 |
|---|---|---|---|---|
| aciliyet | 76.25 / 82.86 | 76.21 / 83.41 | 76.16 / 83.32 | 76.44 / 83.23 |
| alan | 89.52 / 90.47 | 90.06 / 90.47 | 90.02 / 90.57 | 89.97 / 90.47 |
| alt_alan | 93.76 / 93.89 | 93.85 / 93.89 | 93.85 / 93.89 | 93.80 / 93.94 |
| duygu | 68.23 / 85.82 | 71.38 / 86.24 | 70.88 / 86.05 | 71.38 / 86.28 |
| eylem_gerekli | 87.92 / 90.88 | 89.11 / 90.84 | 89.20 / 90.66 | 89.20 / 90.75 |
| guvenlik | 96.81 / 99.18 | 97.40 / 99.32 | 97.58 / 99.32 | 97.54 / 99.27 |
| insana_devret | 77.85 / 84.78 | 77.03 / 83.82 | 77.12 / 83.77 | 77.12 / 83.87 |
| kisisel_veri | 93.44 / 96.17 | 93.71 / 96.22 | 93.66 / 96.31 | 93.57 / 96.26 |
| niyet | 97.90 / 97.99 | 97.99 / 98.09 | 97.95 / 98.09 | 97.90 / 98.13 |
| ozel_soru | 99.13 / 100.00 | 98.26 / 100.00 | 98.26 / 100.00 | 98.26 / 100.00 |

- 6-bit and 8-bit are within 0.06 points of GPU BF16 (micro and macro) and within 0.5 points per question type.
  Between 6-bit and 8-bit the paired difference is not significant (paired z −0.68 for base Decision 3.0, −1.02 for
  Pusula).
- At 4-bit, Pusula micro and macro accuracy are within 0.02 points of GPU BF16. Base Decision 3.0 loses about 0.5
  points at 4-bit (8-bit minus 4-bit +0.52, paired z 5.3), mostly on `duygu` and `eylem_gerekli`. This makes the
  Pusula gain look larger at 4-bit.
- One question type differs at 4-bit: Pusula `insana_devret` is 84.78 at 4-bit and 83.77 at 8-bit (exact McNemar
  p 0.0005); 6-bit and 8-bit are within 0.1 points of GPU BF16 there.
- Per-question agreement with GPU BF16 has not been measured yet.
- Pusula answers every question with the adapter on, which is the v2.1 default (`PUSULA_OWN_BASE=templates`; the
  benchmark has no `eksik_bilgi` question). With the v2.0 setting (`ozel_soru` answered by base Decision 3.0, as
  in the published 92.0) the macro accuracy is 92.12 at 4-bit, 92.06 at 6-bit and 92.02 at 8-bit.
- Speed on the M3 Max, base Decision 3.0: 172 (4-bit), 162 (6-bit) and 161 (8-bit) prompt tokens/s. A full run
  takes about 14 hours for base Decision 3.0 and about 16 hours for Pusula at 4 and 6-bit.
- Memory: peak memory footprint of the Pusula run is 20.9 GB (4-bit), 26.8 GB (6-bit, token budget 4096) and
  30.8 GB (8-bit, token budget 1024), with the default 2 GB buffer cache limit (100 questions, `/usr/bin/time -l`).

## Requirements

- An Apple silicon Mac. A 36 GB machine runs all three bit levels (measured on an M3 Max). Smaller machines were
  not measured. 8-bit uses a token budget of 1024 (`scripts/mlx_bench.sh` sets it).
- The MLX extra:

  ```
  pip install -e ".[mlx]"
  ```

- A local snapshot of `vllm-sr/d3` and of the Pusula adapter (LoRA and readout, as published on Hugging Face).

## How it works

- `convert` quantizes the d3 text backbone once (MLX affine quantization, group size 64). It writes into
  `<out>.partial` and renames it to `<out>` only when the weights, `config.json`, the tokenizer, `decision_config.json`,
  `readout.safetensors`, `LICENSE` and `NOTICE` are all there, so an interrupted conversion never looks complete.
- The LoRA adapter is applied at run time in full precision on top of the quantized weights, as on the GPU 4-bit
  path (NF4 base and LoRA). A switch turns it off. With the adapter switched off, the model gives exactly the
  probabilities of base d3.
- The forward pass follows `pusula/d3model.py`: the softmax-attention layers are non-causal (as d3 is trained), the
  Gated DeltaNet layers stay causal, and the final-norm hidden state at the last prompt position is multiplied by
  the 255-row code readout.
- Prompts are built by the same code as `d3train eval` (`d3train.build_items` and `d3model.render`), so the MLX
  and GPU runs score the same prompts in the same order.

## Commands

Convert the base model (once per bit level):

```
python -m pusula.mlx_d3 convert --d3 ~/models/d3 --out ~/models/d3-mlx-4bit --bits 4
```

Score the benchmark with base Decision 3.0 and with Pusula:

```
python -m pusula.mlx_d3 eval --mlx ~/models/d3-mlx-4bit \
  --data benchmark/tr_decision_bench_v1.jsonl --out local/mlx/4bit-base.json
python -m pusula.mlx_d3 eval --mlx ~/models/d3-mlx-4bit --adapter ~/models/pusula-adapter \
  --data benchmark/tr_decision_bench_v1.jsonl --out local/mlx/4bit-pusula.json
```

Each run writes a report (`<run>.json`: accuracy per question type, micro and macro accuracy, ECE, speed) and one
prediction line per question (`<run>.preds.jsonl`). Runs are resumable: a run that stops continues from the
questions already in its `.preds.jsonl` file. Next to it, `<run>.preds.run.json` records what fixes the run (model
path and bits, adapter, sha256 of the data file, `--base`, `--limit`, `--seed`); a run with other settings on the same
file stops with an error instead of mixing rows, and so does a preds file with duplicate rows or rows that are not
questions of the run. A lock stops a second run on the same file. The `local/` folder is not in git.

`eval` refuses a model whose `decision_config.json` it cannot run as the GPU path does: only
`noncausal_full_attention` without a `prompt_layout` (vllm-sr/d3) is supported, so d3-ultra (`prefix_bidir`,
`header-v1`) stops with an error, as does an adapter trained for another base model. A `temperature` other than 1.0
is applied as on the GPU: to base answers (adapter off or not loaded).

Useful options of `eval`:

| option | meaning |
|---|---|
| `--limit N`, `--seed S` | a fixed random subset of N questions |
| `--base` | adapter loaded but switched off (base Decision 3.0) |
| `--token-budget` | padded tokens per forward pass (default 4096) |
| `--cache-gb` | MLX buffer cache limit (default 2 GB); without it, peak memory grows to about 30 GB at 4-bit |
| `--causal` | debugging only: causal attention everywhere, which is not how d3 runs |

The `own_base` routing (`PUSULA_OWN_BASE`), built from the two runs without loading a model. The default follows
the v2.1 default `templates`: only the `eksik_bilgi` template is answered by base Decision 3.0, own questions keep
the adapter. As on the GPU, a question is recognized as the `eksik_bilgi` template by name or by content: `eval`
marks each row (`base_template`) and `combine` uses the mark. The benchmark has no `eksik_bilgi` question, so on the
benchmark this equals the Pusula run.
`--base-questions ozel_soru eksik_bilgi` gives the v2.0 default (`1`, own questions on base Decision 3.0), which the
published 92.0 used:

```
python -m pusula.mlx_d3 combine --adapter-preds local/mlx/4bit-pusula.preds.jsonl \
  --base-preds local/mlx/4bit-base.preds.jsonl --base-questions ozel_soru eksik_bilgi \
  --out local/mlx/4bit-pusula-ownbase-v20.json
```

Paired comparison of two runs on the same questions (per question type, answers fixed and broken, optional confusion
matrices):

```
python -m pusula.mlx_d3 compare local/mlx/4bit-base.preds.jsonl local/mlx/4bit-pusula.preds.jsonl \
  --a-label "base d3" --b-label Pusula --confusion aciliyet
```

Comparison with GPU runs: `pusula.mlx_preds export` writes an MLX prediction file in the format of
`d3train eval --preds`, and `pusula.mlx_preds compare` compares files of either format (matching rows, same answers,
probability differences, McNemar test). For a `d3train` file the answer is its recorded `pred`; the rounded `p`
values are used only for probability differences and ECE, because two options can round to the same value:

```
python -m pusula.mlx_preds export local/mlx/4bit-pusula.preds.jsonl \
  --data benchmark/tr_decision_bench_v1.jsonl --out local/mlx/4bit-pusula.d3train-preds.jsonl
python -m pusula.mlx_preds compare gpu-pusula.preds.jsonl local/mlx/4bit-pusula.preds.jsonl \
  --data benchmark/tr_decision_bench_v1.jsonl --a-label "GPU BF16" --b-label "MLX 4-bit"
```

## Long runs

`scripts/mlx_bench.sh` runs base Decision 3.0 and Pusula at each bit level one after another. It converts a model
first when it is missing or lacks a required file (every shard of `model.safetensors.index.json` included; if
the check itself fails, the script stops and deletes nothing), restarts a run after a crash, and keeps the Mac awake
(`caffeinate`). Reports go to `local/mlx/` (`OUT=` changes this).

```
nohup bash scripts/mlx_bench.sh 4bit-base 4bit-pusula > ~/models/mlx_bench.out 2>&1 &
bash scripts/mlx_bench.sh status
```

Times and speeds are in [MLX results](#mlx-results-4-6-and-8-bit).

## Checks

- **Bidirectional attention is applied:** with `--causal`, accuracy drops by about 3 points on 300 questions.
- **Batching does not change answers:** one question per forward pass and 4096-token batches give the same answers
  (largest probability difference 0.0015).
- **The adapter switch is exact:** the adapter loaded and switched off gives the same probabilities as base
  Decision 3.0 without the adapter.
