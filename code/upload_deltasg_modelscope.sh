#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: code/upload_deltasg_modelscope.sh <staged-directory> [repo-id]

Upload a directory produced by prepare_deltasg_modelscope.py to a ModelScope
dataset repository. Authenticate once with `modelscope login` before running.

Optional environment variables:
  MODELSCOPE_REPO_ID             Default: DStardust/EM-STORM
  MODELSCOPE_MAX_WORKERS         Default: 8
  MODELSCOPE_COMMIT_MESSAGE      Upload commit title
  MODELSCOPE_COMMIT_DESCRIPTION  Upload commit description
EOF
}

if (( $# < 1 || $# > 2 )); then
  usage
  exit 64
fi

staged_dir="${1%/}"
repo_id="${2:-${MODELSCOPE_REPO_ID:-DStardust/EM-STORM}}"
max_workers="${MODELSCOPE_MAX_WORKERS:-8}"

if [[ ! -d "$staged_dir" ]]; then
  echo "Staged directory does not exist: $staged_dir" >&2
  exit 66
fi

manifest="$staged_dir/accepted_manifest.jsonl"
summary="$staged_dir/dataset_summary.json"
if [[ ! -s "$manifest" || ! -s "$summary" ]]; then
  echo "Refusing upload: expected non-empty accepted_manifest.jsonl and dataset_summary.json in $staged_dir" >&2
  exit 65
fi
if [[ -z "$repo_id" ]]; then
  echo "ModelScope repository id must not be empty" >&2
  exit 64
fi
if [[ ! "$max_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODELSCOPE_MAX_WORKERS must be a positive integer" >&2
  exit 64
fi
if ! command -v modelscope >/dev/null 2>&1; then
  echo "The modelscope CLI is not installed or is not on PATH" >&2
  exit 69
fi
if ! command -v stdbuf >/dev/null 2>&1; then
  echo "The stdbuf command is required for line-buffered upload logs" >&2
  exit 69
fi

sample_count="$(wc -l < "$manifest" | tr -d '[:space:]')"
commit_message="${MODELSCOPE_COMMIT_MESSAGE:-Upload DeltaSG accepted samples $(date +%F)}"
commit_description="${MODELSCOPE_COMMIT_DESCRIPTION:-${sample_count} generation-and-expert accepted samples with complete robot and global visualizations}"

echo "Uploading ${sample_count} accepted samples from $staged_dir to $repo_id"
exec stdbuf -oL -eL modelscope upload \
  "$repo_id" \
  "$staged_dir" \
  --repo-type dataset \
  --max-workers "$max_workers" \
  --commit-message "$commit_message" \
  --commit-description "$commit_description"
