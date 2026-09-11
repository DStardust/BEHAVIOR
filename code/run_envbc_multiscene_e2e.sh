#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-code/outputs/envbc_multiscene_e2e_$(date +%Y%m%d_%H%M%S)}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"
ROBOT="${ROBOT:-Tiago}"
ENVA_NUM="${ENVA_NUM:-0}"
ENVB_NUM="${ENVB_NUM:-8}"
ENVB_TYPES="${ENVB_TYPES:-fire,dirty_dishes,dirty_clothes,broken_object}"
ENVC_NUM="${ENVC_NUM:-8}"
SEED_BASE="${SEED_BASE:-96800}"
RUN_EXPERT="${RUN_EXPERT:-1}"
MIN_PLACEMENT_DIVERSITY_DISTANCE="${MIN_PLACEMENT_DIVERSITY_DISTANCE:-0.50}"

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
python - "$OUT_ROOT/config.json" "$MODEL" "$ROBOT" "$ENVB_NUM" "$ENVC_NUM" "$RUN_EXPERT" "$ENVB_TYPES" "$ENVA_NUM" "$MIN_PLACEMENT_DIVERSITY_DISTANCE" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "llm_model": sys.argv[2],
            "robot": sys.argv[3],
            "envb_requested_per_scene": int(sys.argv[4]),
            "enva_requested_per_scene": int(sys.argv[8]),
            "envb_types": sys.argv[7].split(","),
            "envc_requested_per_scene": int(sys.argv[5]),
            "expert_backend": "oracle_symbolic",
            "run_expert": sys.argv[6] == "1",
            "min_placement_diversity_distance": float(sys.argv[9]),
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
      --max-llm-retries 5 --max-retries 4 --max-retries-per-task 4 \
      --placement-timeout 60 --relation-timeout 10 \
      --max-placement-attempts 6 --max-total-placement-time 180 \
      --min-global-cameras 2 --max-global-cameras 3 \
      --max-camera-pose-attempts 8 --camera-pose-render-steps 4 \
      --min-manipulation-height 0.10 --max-manipulation-height 1.55 \
      --min-placement-diversity-distance "$MIN_PLACEMENT_DIVERSITY_DISTANCE" \
      --solvability-profile oracle_symbolic \
      --output-dir "$output_dir" --seed "$seed" \
      "$@" >"$log_path" 2>&1 || status=$?
  printf '%s\n' "$status" > "${log_path%.log}.exit"
}

phase_succeeded() {
  local exit_path="$1"
  [[ -f "$exit_path" && "$(tr -d '[:space:]' < "$exit_path")" == "0" ]]
}

for index in "${!SCENE_LIST[@]}"; do
  scene="${SCENE_LIST[$index]}"
  scene_root="$OUT_ROOT/$scene"
  generation_root="$scene_root/generation"
  mkdir -p "$generation_root/envB_all" "$generation_root/envC_all" "$scene_root/logs"
  if [[ -f "$scene_root/complete" ]] \
    && { [[ "$ENVA_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/enva.exit"; } \
    && { [[ "$ENVB_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/envb.exit"; } \
    && { [[ "$ENVC_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/envc.exit"; } \
    && phase_succeeded "$scene_root/logs/generation_audit.exit" \
    && { [[ "$RUN_EXPERT" != "1" ]] || phase_succeeded "$scene_root/logs/expert.exit"; }; then
    continue
  fi

  if [[ "$ENVA_NUM" -gt 0 ]] && ! phase_succeeded "$scene_root/logs/enva.exit"; then
    printf 'GEN_A\n' > "$scene_root/state"
    run_generation \
      "$scene" A "$ENVA_NUM" "$generation_root/envA_all" \
      "$scene_root/logs/enva.log" "$((SEED_BASE + index * 100))" \
      --allow-repeat-tasks
  fi

  printf 'GEN_B\n' > "$scene_root/state"
  if [[ "$ENVB_NUM" -gt 0 ]] && ! phase_succeeded "$scene_root/logs/envb.exit"; then
    run_generation \
      "$scene" B "$ENVB_NUM" "$generation_root/envB_all" \
      "$scene_root/logs/envb.log" "$((SEED_BASE + index * 100 + 1))" \
      --env-b-types "$ENVB_TYPES" --allow-repeat-tasks
  fi

  printf 'GEN_C\n' > "$scene_root/state"
  if [[ "$ENVC_NUM" -gt 0 ]] && ! phase_succeeded "$scene_root/logs/envc.exit"; then
    run_generation \
      "$scene" C "$ENVC_NUM" "$generation_root/envC_all" \
      "$scene_root/logs/envc.log" "$((SEED_BASE + index * 100 + 2))" \
      --env-c-types retrieval_delivery,open_close,appliance,fire --allow-repeat-tasks
  fi

  printf 'AUDIT_GEN\n' > "$scene_root/state"
  generation_audit_status=0
  python code/audit_deltasg_outputs.py \
    --root "$generation_root" --ok-only --fail-on-issues \
    --json-out "$scene_root/generation_audit.json" \
    >"$scene_root/logs/generation_audit.log" 2>&1 || generation_audit_status=$?
  printf '%s\n' "$generation_audit_status" > "$scene_root/logs/generation_audit.exit"

  if [[ "$RUN_EXPERT" == "1" ]] && ! phase_succeeded "$scene_root/logs/expert.exit"; then
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
  fi

  scene_status=0
  { [[ "$ENVA_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/enva.exit"; } || scene_status=2
  { [[ "$ENVB_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/envb.exit"; } || scene_status=2
  { [[ "$ENVC_NUM" -eq 0 ]] || phase_succeeded "$scene_root/logs/envc.exit"; } || scene_status=2
  phase_succeeded "$scene_root/logs/generation_audit.exit" || scene_status=2
  { [[ "$RUN_EXPERT" != "1" ]] || phase_succeeded "$scene_root/logs/expert.exit"; } || scene_status=2
  if [[ "$scene_status" -eq 0 ]]; then
    printf 'DONE\n' > "$scene_root/state"
  else
    printf 'PARTIAL\n' > "$scene_root/state"
  fi
  date --iso-8601=seconds > "$scene_root/complete"
done

python code/monitor_envbc_multiscene_e2e.py "$OUT_ROOT" > "$OUT_ROOT/final_report.txt"
overall_status=0
for scene in "${SCENE_LIST[@]}"; do
  [[ "$(tr -d '[:space:]' < "$OUT_ROOT/$scene/state")" == "DONE" ]] || overall_status=2
done
if [[ "$overall_status" -eq 0 ]]; then
  printf 'DONE\n' > "$OUT_ROOT/state"
else
  printf 'PARTIAL\n' > "$OUT_ROOT/state"
fi
cat "$OUT_ROOT/final_report.txt"
exit "$overall_status"
