#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  printf 'Usage: bash %s <train-jsonl> <benchmark-jsonl> <model> <output-dir>\n' "$0" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
train_data="$(realpath "$1")"
benchmark="$(realpath "$2")"
model="$3"
output="$(realpath -m "$4")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$output"
cd "$repo_root"

phase=starting
status() {
  printf '{"phase":"%s","state":"%s","exit_code":%s}\n' "$phase" "$1" "$2" > "$output/status.json.tmp"
  mv "$output/status.json.tmp" "$output/status.json"
}
trap 'code=$?; if (( code != 0 )); then status failed "$code"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

python_cmd=(conda run --no-capture-output -n emstorm-train python -u code/train_emstorm.py)
common=(--data "$train_data" --stage 0 --model "$model" --load-in-4bit --seed 3407)

phase=train
status running 0
printf 'FULL_PHASE=%s\n' "$phase"
"${python_cmd[@]}" train "${common[@]}" --output "$output/adapter" \
  --epochs 1 --batch-size 1 --gradient-accumulation 8 --logging-steps 10 --save-steps 50

phase=validation
status running 0
printf 'FULL_PHASE=%s\n' "$phase"
"${python_cmd[@]}" predict "${common[@]}" --adapter "$output/adapter" \
  --max-new-tokens 768 --predictions "$output/adapter_val.jsonl" --resume

phase=benchmark
status running 0
printf 'FULL_PHASE=%s\n' "$phase"
"${python_cmd[@]}" predict "${common[@]}" --adapter "$output/adapter" \
  --max-new-tokens 768 --benchmark "$benchmark" --split test \
  --predictions "$output/adapter_benchmark.jsonl" --resume

phase=complete
status completed 0
printf 'FULL_COMPLETE=%s\n' "$output"
