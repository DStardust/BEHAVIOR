#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  printf 'Usage: bash %s <train-jsonl> <benchmark-jsonl> <model> <output-dir>\n' "$0" >&2
  printf 'Options via environment: CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=16 EVAL_LIMIT=22\n' >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train_data="$(realpath "$1")"
benchmark="$(realpath "$2")"
model="$3"
output="$(realpath -m "$4")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
steps="${TRAIN_STEPS:-16}"
limit="${EVAL_LIMIT:-22}"
mkdir -p "$output"
cd "$repo_root"

python_cmd=(conda run --no-capture-output -n emstorm-train python -u code/train_emstorm.py)
common=(--data "$train_data" --stage 0 --model "$model" --load-in-4bit)
eval_args=(--limit "$limit" --stratified --max-new-tokens 256)

printf 'PILOT_PHASE=base_validation\n'
"${python_cmd[@]}" predict "${common[@]}" "${eval_args[@]}" \
  --predictions "$output/base_val.jsonl"

printf 'PILOT_PHASE=train\n'
"${python_cmd[@]}" train "${common[@]}" --output "$output/adapter" \
  --max-steps "$steps" --batch-size 1 --gradient-accumulation 2 --logging-steps 1

printf 'PILOT_PHASE=adapter_validation\n'
"${python_cmd[@]}" predict "${common[@]}" "${eval_args[@]}" --adapter "$output/adapter" \
  --predictions "$output/adapter_val.jsonl"

printf 'PILOT_PHASE=benchmark_subset\n'
"${python_cmd[@]}" predict "${common[@]}" "${eval_args[@]}" --adapter "$output/adapter" \
  --benchmark "$benchmark" --split test --predictions "$output/adapter_benchmark.jsonl"

printf 'PILOT_COMPLETE=%s\n' "$output"
