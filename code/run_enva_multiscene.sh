#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-code/outputs/enva_retrieval_multiscene_20260708}"
ROBOT="${ROBOT:-fetch}"
NUM="${NUM:-300}"
TASK_CATEGORIES="${TASK_CATEGORIES:-retrieval_delivery}"
SEED_BASE="${SEED_BASE:-79000}"
VISUALIZE="${VISUALIZE:-0}"

read -r -a SCENE_LIST <<< "${SCENES:-Rs_int Benevolence_0_int Pomaria_0_int Wainscott_0_int Merom_0_int}"

mkdir -p "$OUT_ROOT/logs"

scene_count_for_index() {
  local idx="$1"
  local n="${#SCENE_LIST[@]}"
  local base=$((NUM / n))
  local rem=$((NUM % n))
  local count="$base"
  if (( idx < rem )); then
    count=$((count + 1))
  fi
  echo "$count"
}

run_scene() {
  local scene="$1"
  local count="$2"
  local seed="$3"
  local out_dir="$OUT_ROOT/$scene"
  mkdir -p "$out_dir"
  echo "===== GENERATE Env-A scene=$scene categories=$TASK_CATEGORIES count=$count $(date '+%F %T') ====="
  env -u ALL_PROXY -u all_proxy PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    conda run --no-capture-output -n behavior python code/run_online_deltasg.py \
      --scene "$scene" --robot "$ROBOT" \
      --env-type A --task-categories "$TASK_CATEGORIES" \
      --allow-repeat-tasks \
      --num-envs "$count" \
      --checkpoint-interval 10 \
      --warmup-steps 20 --settle-steps 5 \
      --llm-model qwen3.7-max \
      --max-llm-retries 2 --max-retries 1 \
      --placement-timeout 60 --relation-timeout 10 \
      --max-placement-attempts 4 --max-total-placement-time 120 \
      --output-dir "$out_dir" \
      --seed "$seed"
}

vis_scene() {
  local scene="$1"
  local in_dir="$OUT_ROOT/$scene"
  local vis_dir="$OUT_ROOT/visualizations/$scene"
  mkdir -p "$vis_dir"
  echo "===== VISUALIZE Env-A scene=$scene $(date '+%F %T') ====="
  env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 \
    conda run -n behavior python code/visualize_deltasg_batch.py \
      --scene "$scene" --robot "$ROBOT" \
      --input-dir "$in_dir" \
      --output-dir "$vis_dir"
}

for idx in "${!SCENE_LIST[@]}"; do
  scene="${SCENE_LIST[$idx]}"
  count="$(scene_count_for_index "$idx")"
  if (( count <= 0 )); then
    continue
  fi
  run_scene "$scene" "$count" "$((SEED_BASE + idx))"
done

if [[ "$VISUALIZE" == "1" ]]; then
  for scene in "${SCENE_LIST[@]}"; do
    vis_scene "$scene"
  done
fi

echo "===== DONE $(date '+%F %T') ====="
echo "$OUT_ROOT"
