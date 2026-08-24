#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MANIFEST OUTPUT_ROOT GPU NUM_SHARDS SHARD_INDEX" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$1"
output_root="$2"
gpu="$3"
num_shards="$4"
shard_index="$5"

set -a
source "$repo_root/.env"
set +a

cd "$repo_root"
exec conda run --no-capture-output -n behavior \
  python code/run_enva_expert_coverage.py \
  --manifest "$manifest" \
  --output-root "$output_root" \
  --gpu "$gpu" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --robot Tiago \
  --expert-robot Tiago \
  --expert-backend physical_control \
  --process-retries 1 \
  --expert-process-retries 1 \
  --min-eligible-accept-rate "${DELTASG_MIN_ELIGIBLE_ACCEPT_RATE:-0.80}"
