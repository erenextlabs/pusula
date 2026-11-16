# Pusula and Decision 3.0 on Apple silicon (MLX)

`pusula.mlx_d3` runs Decision 3.0 (`vllm-sr/d3`) and the Pusula adapter on an Apple silicon Mac with
[MLX](https://github.com/ml-explore/mlx), without a GPU server. It converts the base d3 model to 4, 6 or 8 bits
once and applies the Pusula adapter at run time. The same converted model answers with the adapter on (Pusula) or
off (base Decision 3.0), so the `own_base` routing of the published setup is kept.

Results on this benchmark will be added when the 4, 6 and 8-bit runs are complete.

## Requirements

- An Apple silicon Mac. A 36 GB machine runs 4 and 6-bit. 8-bit (about 27 GB of weights) is at the limit and uses a
  smaller token budget.
- The MLX extra:

  ```
  pip install -e ".[mlx]"
  ```

- A local snapshot of `vllm-sr/d3` and of the Pusula adapter (LoRA and readout, as published on Hugging Face).

## How it works

- `convert` quantizes the d3 text backbone once (MLX affine quantization, group size 64).
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
questions already in its `.preds.jsonl` file. The `local/` folder is not in git.

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
the adapter. The benchmark has no `eksik_bilgi` question, so on the benchmark this equals the Pusula run.
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
probability differences, McNemar test):

```
python -m pusula.mlx_preds export local/mlx/4bit-pusula.preds.jsonl \
  --data benchmark/tr_decision_bench_v1.jsonl --out local/mlx/4bit-pusula.d3train-preds.jsonl
python -m pusula.mlx_preds compare gpu-pusula.preds.jsonl local/mlx/4bit-pusula.preds.jsonl \
  --data benchmark/tr_decision_bench_v1.jsonl --a-label "GPU BF16" --b-label "MLX 4-bit"
```

## Long runs

`scripts/mlx_bench.sh` runs base Decision 3.0 and Pusula at each bit level one after another. It converts a model
first when it is missing, restarts a run after a crash, and keeps the Mac awake (`caffeinate`). Reports go to
`local/mlx/` (`OUT=` changes this).

```
nohup bash scripts/mlx_bench.sh 4bit-base 4bit-pusula > ~/models/mlx_bench.out 2>&1 &
bash scripts/mlx_bench.sh status
```

A full benchmark run (19,861 questions) takes about 14 hours for base Decision 3.0 and about 16 hours for Pusula at
4-bit on an M3 Max.

## Checks

- **Bidirectional attention is applied:** with `--causal`, accuracy drops by about 3 points on 300 questions.
- **Batching does not change answers:** one question per forward pass and 4096-token batches give the same answers
  (largest probability difference 0.0015).
- **The adapter switch is exact:** the adapter loaded and switched off gives the same probabilities as base
  Decision 3.0 without the adapter.
