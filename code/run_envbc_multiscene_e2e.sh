#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-code/outputs/envbc_multiscene_e2e_$(date +%Y%m%d_%H%M%S)}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"
ROBOT="${ROBOT:-Tiago}"
ENVB_NUM="${ENVB_NUM:-3}"
ENVC_NUM="${ENVC_NUM:-8}"
SEED_BASE="${SEED_BASE:-96800}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENE_LIST <<< "$SCENES"
else
  mapfile -t SCENE_LIST < code/configs/env_a_scenes.txt
fi

mkdir -p "$OUT_ROOT/logs"
printf '%s\n' "${SCENE_LIST[@]}" > "$OUT_ROOT/scenes.txt"
python - "$OUT_ROOT/config.json" "$MODEL" "$ROBOT" "$ENVB_NUM" "$ENVC_NUM" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "llm_model": sys.argv[2],
            "robot": sys.argv[3],
            "envb_requested_per_scene": int(sys.argv[4]),
            "envc_requested_per_scene": int(sys.argv[5]),
            "expert_backend": "oracle_symbolic",
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

run_generation() {
  local scene="$1"
  local env_type="$2"
  local count="$3"
  local output_dir="$4"
  local log_path="$5"
  local seed="$6"
  shift 6
  local status=0
  mkdir -p "$output_dir"
  PYTHONUNBUFFERED=1 code/run_omnigibson_single_gpu.sh \
    conda run --no-capture-output -n behavior \
    python code/run_online_deltasg.py \
      --scene "$scene" --robot "$ROBOT" \
      --env-type "$env_type" --num-envs "$count" \
      --task-objects 1 --context-objects 0 \
      --checkpoint-interval 1 \
      --warmup-steps 20 --settle-steps 5 \
      --llm-model "$MODEL" \
      --max-llm-retries 5 --max-retries 4 \
      --placement-timeout 60 --relation-timeout 10 \
      --max-placement-attempts 6 --max-total-placement-time 180 \
      --min-global-cameras 2 --max-global-cameras 3 \
      --max-camera-pose-attempts 8 --camera-pose-render-steps 4 \
      --min-manipulation-height 0.10 --max-manipulation-height 1.55 \
      --solvability-profile oracle_symbolic \
      --output-dir "$output_dir" --seed "$seed" \
      "$@" >"$log_path" 2>&1 || status=$?
  printf '%s\n' "$status" > "${log_path%.log}.exit"
}

for index in "${!SCENE_LIST[@]}"; do
  scene="${SCENE_LIST[$index]}"
  scene_root="$OUT_ROOT/$scene"
  generation_root="$scene_root/generation"
  mkdir -p "$generation_root/envB_fire" "$generation_root/envC_all" "$scene_root/logs"
  if [[ -f "$scene_root/complete" ]]; then
    continue
  fi

  printf 'GEN_B\n' > "$scene_root/state"
  if [[ ! -f "$scene_root/logs/envb.exit" ]]; then
    run_generation \
      "$scene" B "$ENVB_NUM" "$generation_root/envB_fire" \
      "$scene_root/logs/envb.log" "$((SEED_BASE + index * 100 + 1))"
  fi

  printf 'GEN_C\n' > "$scene_root/state"
  if [[ ! -f "$scene_root/logs/envc.exit" ]]; then
    run_generation \
      "$scene" C "$ENVC_NUM" "$generation_root/envC_all" \
      "$scene_root/logs/envc.log" "$((SEED_BASE + index * 100 + 2))" \
      --env-c-types retrieval_delivery,open_close,appliance,fire
  fi

  printf 'AUDIT_GEN\n' > "$scene_root/state"
  python code/audit_deltasg_outputs.py \
    --root "$generation_root" --ok-only \
    --json-out "$scene_root/generation_audit.json" \
    >"$scene_root/logs/generation_audit.log" 2>&1
  printf '%s\n' "$?" > "$scene_root/logs/generation_audit.exit"

  printf 'EXPERT\n' > "$scene_root/state"
  expert_status=0
  EXPERT_BACKEND=oracle_symbolic \
  EXPERT_LABELS=all \
  EXPERT_TASKS=all \
  DELTASG_LLM_MODEL="$MODEL" \
    bash code/run_deltasg_expert_batch.sh \
      "$generation_root" "$scene_root/expert" \
      >"$scene_root/logs/expert.log" 2>&1 || expert_status=$?
  printf '%s\n' "$expert_status" > "$scene_root/logs/expert.exit"

  printf 'DONE\n' > "$scene_root/state"
  date --iso-8601=seconds > "$scene_root/complete"
done

python code/monitor_envbc_multiscene_e2e.py "$OUT_ROOT" > "$OUT_ROOT/final_report.txt"
printf 'DONE\n' > "$OUT_ROOT/state"
cat "$OUT_ROOT/final_report.txt"

