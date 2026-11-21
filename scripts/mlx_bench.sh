#!/usr/bin/env bash
# Pusula on Apple silicon: MLX benchmark runs, one run at a time, resumable.
# Start it and leave it. A killed or crashed run continues where it stopped on the next start,
# because every finished question is already in the run's .preds.jsonl file.
#
#   bash scripts/mlx_bench.sh <run> [<run> ...]
#   bash scripts/mlx_bench.sh status
#
#   runs: 4bit-base 4bit-pusula 6bit-base 6bit-pusula 8bit-base 8bit-pusula
#         base   = base Decision 3.0 (adapter not loaded)
#         pusula = Pusula LoRA adapter + readout
#   The quantized model is converted first when it is not there yet.
#
# Leave it running in the background (survives closing the terminal, keeps the Mac awake):
#   nohup bash scripts/mlx_bench.sh 4bit-base 4bit-pusula > ~/models/mlx_bench.out 2>&1 &
#   tail -f ~/models/mlx_bench.out
#
# Environment (optional):
#   MODELS=~/models  LIMIT=0 (0 = the full benchmark; N = a fixed random subset of N questions)
#   TOKEN_BUDGET=4096 (8bit default 1024, to fit 36 GB)  TRIES=5 (restarts per run after a crash)
#   OUT=local/mlx (reports and predictions; local/ is not in git)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-$HOME/models}
DATA=${DATA:-benchmark/tr_decision_bench_v1.jsonl}
LIMIT=${LIMIT:-0}
TRIES=${TRIES:-5}
OUT=${OUT:-$ROOT/local/mlx}
PY="$ROOT/.venv/bin/python"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
mkdir -p "$OUT"

log() { echo "[$(date +%F' '%T)] $*"; }
name() { if [[ $LIMIT == 0 ]]; then echo "$1"; else echo "$1-n$LIMIT"; fi; }

status() {
  for run in 4bit-base 4bit-pusula 6bit-base 6bit-pusula 8bit-base 8bit-pusula; do
    local n; n=$(name "$run")
    if [[ -f "$OUT/$n.json" ]]; then
      "$PY" -I -c "import json,sys;r=json.load(open(sys.argv[1]));print(f'{sys.argv[2]:<14} DONE  n={r[\"n\"]}  micro={100*r[\"micro_acc\"]:.2f}  macro={100*r[\"macro_acc\"]:.2f}  ece={r[\"ece\"]:.3f}')" "$OUT/$n.json" "$n"
    elif [[ -f "$OUT/$n.preds.jsonl" ]]; then
      printf '%-14s PARTIAL  %s questions done\n' "$n" "$(wc -l < "$OUT/$n.preds.jsonl" | tr -d ' ')"
    else
      printf '%-14s not started\n' "$n"
    fi
  done
}

run_one() {
  local run=$1 bits kind n mlx budget extra=()
  [[ $run =~ ^(4|6|8)bit-(base|pusula)$ ]] || { echo "unknown run: $run"; exit 2; }
  bits=${BASH_REMATCH[1]}; kind=${BASH_REMATCH[2]}
  n=$(name "$run")
  mlx="$MODELS/d3-mlx-${bits}bit"
  if [[ $bits == 8 ]]; then budget=${TOKEN_BUDGET:-1024}; else budget=${TOKEN_BUDGET:-4096}; fi
  [[ $kind == pusula ]] && extra=(--adapter "$MODELS/pusula-adapter")
  [[ $LIMIT != 0 ]] && extra+=(--limit "$LIMIT")

  if [[ -f "$OUT/$n.json" ]]; then log "$n already done, skipped"; return; fi

  # convert writes into $mlx.partial and renames it only when every required file is there;
  # a directory that still lacks one (made by an older convert) is converted again.
  if ! "$PY" -I -c 'import sys; from pusula.mlx_d3 import missing_files; sys.exit(1 if missing_files(sys.argv[1]) else 0)' "$mlx"; then
    log "converting d3 to ${bits}-bit: $mlx"
    rm -rf "$mlx" "$mlx.partial"
    caffeinate -i "$PY" -m pusula.mlx_d3 convert --d3 "$MODELS/d3" --out "$mlx" --bits "$bits"
  fi

  local try=1
  while true; do
    log "$n: start (try $try/$TRIES, token budget $budget)"
    if caffeinate -i "$PY" -m pusula.mlx_d3 eval --mlx "$mlx" --data "$DATA" --token-budget "$budget" \
         --out "$OUT/$n.json" ${extra[@]+"${extra[@]}"}; then
      log "RUN_DONE $n  report: $OUT/$n.json"
      return
    fi
    if (( try >= TRIES )); then log "RUN_FAILED $n after $TRIES tries"; exit 1; fi
    try=$((try + 1))
    log "$n: crashed, restarting in 30 s (finished questions are kept)"
    sleep 30
  done
}

[[ $# -gt 0 ]] || { sed -n '2,20p' "$0"; exit 2; }
if [[ $1 == status ]]; then status; exit 0; fi
for run in "$@"; do run_one "$run"; done
log "ALL_DONE $*"
status
