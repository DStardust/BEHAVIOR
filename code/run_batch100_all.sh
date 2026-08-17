#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-code/outputs/batch100_all_multiscene_20260708}"
ROBOT="${ROBOT:-Tiago}"
NUM="${NUM:-100}"
MIN_OK_PER_SCENE="${MIN_OK_PER_SCENE:-1}"
TASK_OBJECTS="${TASK_OBJECTS:-1}"
CONTEXT_OBJECTS="${CONTEXT_OBJECTS:-0}"
SEED_BASE="${SEED_BASE:-78000}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MAX_TOPUP_ROUNDS="${MAX_TOPUP_ROUNDS:-30}"
MAX_CONSECUTIVE_PROCESS_FAILURES="${MAX_CONSECUTIVE_PROCESS_FAILURES:-3}"
MAX_VIS_RETRIES="${MAX_VIS_RETRIES:-3}"
MAX_GLOBAL_CAMERAS="${MAX_GLOBAL_CAMERAS:-3}"
MAX_CAMERA_POSE_ATTEMPTS="${MAX_CAMERA_POSE_ATTEMPTS:-6}"
CAMERA_POSE_RENDER_STEPS="${CAMERA_POSE_RENDER_STEPS:-4}"
MIN_MANIPULATION_HEIGHT="${MIN_MANIPULATION_HEIGHT:-0.10}"
MAX_MANIPULATION_HEIGHT="${MAX_MANIPULATION_HEIGHT:-1.55}"
VISUALIZE_INCREMENTAL="${VISUALIZE_INCREMENTAL:-1}"
SCENE_SCOPE="${SCENE_SCOPE:-interior}"
LABELS="${LABELS:-all}"
STRICT_COVERAGE="${STRICT_COVERAGE:-1}"
REQUIRE_ALL_ASSET_MODELS="${REQUIRE_ALL_ASSET_MODELS:-1}"
REQUIRE_ALL_NATIVE_TARGETS="${REQUIRE_ALL_NATIVE_TARGETS:-1}"

# By default use every locally installed BEHAVIOR home interior scene. Override
# with an explicit SCENES list for a small smoke run, or SCENE_SCOPE=all when a
# task family has been validated outside the home-interior domain.
if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENE_LIST <<< "$SCENES"
else
  mapfile -t SCENE_LIST < <(
    code/run_omnigibson_single_gpu.sh \
      conda run --no-capture-output -n behavior python code/list_deltasg_scenes.py --scope "$SCENE_SCOPE"
  )
  FILTERED_SCENE_LIST=()
  for scene in "${SCENE_LIST[@]}"; do
    [[ -n "$scene" ]] && FILTERED_SCENE_LIST+=("$scene")
  done
  SCENE_LIST=("${FILTERED_SCENE_LIST[@]}")
fi
if (( ${#SCENE_LIST[@]} == 0 )); then
  echo "ERROR: no scenes discovered (SCENE_SCOPE=$SCENE_SCOPE)" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/visualizations"
FAILED_JOBS_FILE="$OUT_ROOT/failed_jobs.tsv"
: > "$FAILED_JOBS_FILE"
FAILED_JOB_COUNT=0
printf '%s\n' "${SCENE_LIST[@]}" > "$OUT_ROOT/scenes.txt"
echo "===== DeltaSG scenes (${#SCENE_LIST[@]}): ${SCENE_LIST[*]} ====="

label_enabled() {
  local label="$1"
  if [[ "$LABELS" == "all" ]]; then
    return 0
  fi
  local normalized=" ${LABELS//,/ } "
  [[ "$normalized" == *" $label "* ]]
}

scene_count_for_index() {
  local idx="$1"
  local n="${#SCENE_LIST[@]}"
  local base=$((NUM / n))
  local rem=$((NUM % n))
  local count="$base"
  if (( idx < rem )); then
    count=$((count + 1))
  fi
  if (( count < MIN_OK_PER_SCENE )); then
    count="$MIN_OK_PER_SCENE"
  fi
  echo "$count"
}

run_gen_scene() {
  local label="$1"
  local scene="$2"
  local count="$3"
  local seed="$4"
  shift 4
  local out_dir="$OUT_ROOT/$label/$scene"
  mkdir -p "$out_dir"
  echo "===== GENERATE $label scene=$scene count=$count $(date '+%F %T') ====="
  if ! PYTHONUNBUFFERED=1 code/run_omnigibson_single_gpu.sh \
    conda run --no-capture-output -n behavior python code/run_online_deltasg.py \
      --scene "$scene" --robot "$ROBOT" \
      --num-envs "$count" \
      --task-objects "$TASK_OBJECTS" --context-objects "$CONTEXT_OBJECTS" \
      --checkpoint-interval 10 \
      --warmup-steps 20 --settle-steps 5 \
      --llm-model qwen3.8-max \
      --max-llm-retries 5 --max-retries 4 \
      --placement-timeout 60 --relation-timeout 10 \
      --max-placement-attempts 4 --max-total-placement-time 120 \
      --max-model-failures 2 \
      --max-global-cameras "$MAX_GLOBAL_CAMERAS" \
      --max-camera-pose-attempts "$MAX_CAMERA_POSE_ATTEMPTS" \
      --camera-pose-render-steps "$CAMERA_POSE_RENDER_STEPS" \
      --min-manipulation-height "$MIN_MANIPULATION_HEIGHT" \
      --max-manipulation-height "$MAX_MANIPULATION_HEIGHT" \
      --output-dir "$out_dir" \
      --seed "$seed" \
      "$@"; then
    echo "===== GENERATION FAILED $label scene=$scene $(date '+%F %T') =====" >&2
    return 1
  fi
}

count_ok_runs() {
  local out_dir="$1"
  if [ ! -d "$out_dir" ]; then
    echo 0
    return
  fi
  python - "$out_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd() / "code"))
from audit_deltasg_outputs import check_run

count = 0
for path in root.glob("online_*.json"):
    if path.name in {"summary.json", "dataset.json", "dataset_index.json", "checkpoint.json"}:
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    validation = data.get("validation") or {}
    integrity = validation.get("scene_integrity") or {}
    settling = validation.get("settling") or {}
    if (
        data.get("ok")
        and str(data.get("run_id", "")).startswith("online_")
        and integrity.get("ok") is True
        and settling.get("all_within_threshold") is True
        and not validation.get("duplicate_sample")
        and not check_run(path, data)
    ):
        count += 1
print(count)
PY
}

run_gen_scene_until() {
  local label="$1"
  local scene="$2"
  local target="$3"
  local seed="$4"
  shift 4
  local out_dir="$OUT_ROOT/$label/$scene"
  mkdir -p "$out_dir"

  local ok_count
  ok_count="$(count_ok_runs "$out_dir")"
  local round=0
  local consecutive_process_failures=0
  while (( ok_count < target )); do
    local remaining=$((target - ok_count))
    local batch="$remaining"
    if (( batch > CHUNK_SIZE )); then
      batch="$CHUNK_SIZE"
    fi
    round=$((round + 1))
    if (( round > MAX_TOPUP_ROUNDS )); then
      echo "ERROR: $label scene=$scene did not reach $target clean samples after $MAX_TOPUP_ROUNDS rounds" >&2
      return 1
    fi
    echo "===== TOP UP $label scene=$scene ok=$ok_count target=$target batch=$batch round=$round $(date '+%F %T') ====="
    if ! run_gen_scene "$label" "$scene" "$batch" "$((seed + round))" --resume "$@"; then
      consecutive_process_failures=$((consecutive_process_failures + 1))
      echo "===== RETRYING $label scene=$scene after failed generator process =====" >&2
      if (( consecutive_process_failures >= MAX_CONSECUTIVE_PROCESS_FAILURES )); then
        echo "ERROR: $label scene=$scene had $consecutive_process_failures consecutive generator process failures" >&2
        return 1
      fi
      continue
    fi
    consecutive_process_failures=0
    ok_count="$(count_ok_runs "$out_dir")"
  done
  echo "===== TARGET REACHED $label scene=$scene ok=$ok_count target=$target $(date '+%F %T') ====="
}

run_gen_multiscene() {
  local label="$1"
  local seed_offset="$2"
  shift 2
  for idx in "${!SCENE_LIST[@]}"; do
    local scene="${SCENE_LIST[$idx]}"
    local count
    count="$(scene_count_for_index "$idx")"
    if (( count <= 0 )); then
      continue
    fi
    if ! run_gen_scene_until "$label" "$scene" "$count" "$((SEED_BASE + seed_offset * 100 + idx))" "$@"; then
      printf '%s\t%s\t%s\n' "$label" "$scene" "generation_incomplete" >> "$FAILED_JOBS_FILE"
      FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
      echo "===== QUARANTINED $label scene=$scene; continuing with remaining scenes =====" >&2
    elif [[ "$VISUALIZE_INCREMENTAL" == "1" ]]; then
      if ! run_vis_scene_with_retry "$label" "$scene"; then
        : # Failure is recorded by the visualization helper; continue the matrix.
      fi
    fi
  done
}

run_vis_scene() {
  local label="$1"
  local scene="$2"
  local in_dir="$OUT_ROOT/$label/$scene"
  local vis_dir="$OUT_ROOT/visualizations/$label/$scene"
  if [ ! -d "$in_dir" ]; then
    return
  fi
  mkdir -p "$vis_dir"
  echo "===== VISUALIZE $label scene=$scene $(date '+%F %T') ====="
  code/run_omnigibson_single_gpu.sh \
    conda run --no-capture-output -n behavior python code/visualize_deltasg_batch.py \
      --scene "$scene" --robot none \
      --input-dir "$in_dir" \
      --output-dir "$vis_dir"
}

run_vis_multiscene() {
  local label="$1"
  for scene in "${SCENE_LIST[@]}"; do
    local attempt=1
    local rendered=false
    while (( attempt <= MAX_VIS_RETRIES )); do
      if run_vis_scene "$label" "$scene"; then
        rendered=true
        break
      fi
      echo "===== RETRY VISUALIZATION $label scene=$scene attempt=$attempt/$MAX_VIS_RETRIES =====" >&2
      attempt=$((attempt + 1))
    done
    if [[ "$rendered" != true ]]; then
      printf '%s\t%s\t%s\n' "$label" "$scene" "visualization_failed" >> "$FAILED_JOBS_FILE"
      FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
      echo "===== VISUALIZATION FAILED $label scene=$scene; continuing =====" >&2
    fi
  done
}

run_vis_scene_with_retry() {
  local label="$1"
  local scene="$2"
  local attempt=1
  while (( attempt <= MAX_VIS_RETRIES )); do
    if run_vis_scene "$label" "$scene"; then
      return 0
    fi
    echo "===== RETRY VISUALIZATION $label scene=$scene attempt=$attempt/$MAX_VIS_RETRIES =====" >&2
    attempt=$((attempt + 1))
  done
  printf '%s\t%s\t%s\n' "$label" "$scene" "visualization_failed" >> "$FAILED_JOBS_FILE"
  FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
  echo "===== VISUALIZATION FAILED $label scene=$scene; continuing =====" >&2
  return 1
}

if label_enabled "envA_retrieval_delivery"; then
  run_gen_multiscene "envA_retrieval_delivery" 1 \
    --env-type A --task-categories retrieval_delivery --allow-repeat-tasks
fi
if label_enabled "envA_open_close"; then
  run_gen_multiscene "envA_open_close" 2 \
    --env-type A --task-categories open_close --allow-repeat-tasks
fi
if label_enabled "envA_appliance"; then
  run_gen_multiscene "envA_appliance" 3 \
    --env-type A --task-categories appliance --allow-repeat-tasks
fi
if label_enabled "envB_fire"; then
  run_gen_multiscene "envB_fire" 4 --env-type B
fi
if label_enabled "envC_retrieval_delivery"; then
  run_gen_multiscene "envC_retrieval_delivery" 5 \
    --env-type C --env-c-types retrieval_delivery --allow-repeat-tasks
fi
if label_enabled "envC_open_close"; then
  run_gen_multiscene "envC_open_close" 6 \
    --env-type C --env-c-types open_close --allow-repeat-tasks
fi
if label_enabled "envC_appliance"; then
  run_gen_multiscene "envC_appliance" 7 \
    --env-type C --env-c-types appliance --allow-repeat-tasks
fi
if label_enabled "envC_fire_disambiguation"; then
  run_gen_multiscene "envC_fire_disambiguation" 8 --env-type C --env-c-types fire
fi

if [[ "$VISUALIZE_INCREMENTAL" != "1" ]]; then
  for label in \
    envA_retrieval_delivery envA_open_close envA_appliance envB_fire \
    envC_retrieval_delivery envC_open_close envC_appliance envC_fire_disambiguation; do
    if label_enabled "$label"; then
      run_vis_multiscene "$label"
    fi
  done
fi

echo "===== AUDIT ACCEPTED SAMPLES $(date '+%F %T') ====="
if ! python code/audit_deltasg_outputs.py \
  --root "$OUT_ROOT" \
  --vis-root "$OUT_ROOT/visualizations" \
  --ok-only \
  --json-out "$OUT_ROOT/audit_accepted.json" \
  --fail-on-issues; then
  printf '%s\t%s\t%s\n' "all" "all" "accepted_sample_audit_failed" >> "$FAILED_JOBS_FILE"
  FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
fi

echo "===== BUILD COVERAGE INVENTORY $(date '+%F %T') ====="
if ! code/run_omnigibson_single_gpu.sh \
  conda run --no-capture-output -n behavior \
  python code/build_deltasg_coverage_inventory.py \
    --output "$OUT_ROOT/coverage_inventory.json"; then
  printf '%s\t%s\t%s\n' "all" "all" "coverage_inventory_failed" >> "$FAILED_JOBS_FILE"
  FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
fi

COVERAGE_LABELS=()
for label in \
  envA_retrieval_delivery envA_open_close envA_appliance envB_fire \
  envC_retrieval_delivery envC_open_close envC_appliance envC_fire_disambiguation; do
  if label_enabled "$label"; then
    COVERAGE_LABELS+=("$label")
  fi
done
COVERAGE_ARGS=(
  --root "$OUT_ROOT"
  --scenes-file "$OUT_ROOT/scenes.txt"
  --labels "$(IFS=,; echo "${COVERAGE_LABELS[*]}")"
  --min-clean-per-cell "$MIN_OK_PER_SCENE"
  --asset-inventory "$OUT_ROOT/coverage_inventory.json"
  --json-out "$OUT_ROOT/coverage_audit.json"
  --fail-on-gaps
)
if [[ "$STRICT_COVERAGE" == "1" ]]; then
  COVERAGE_ARGS+=(--require-all-known-tasks)
fi
if [[ "$REQUIRE_ALL_ASSET_MODELS" == "1" ]]; then
  COVERAGE_ARGS+=(--require-all-asset-models)
fi
if [[ "$REQUIRE_ALL_NATIVE_TARGETS" == "1" ]]; then
  COVERAGE_ARGS+=(--require-all-native-targets)
fi

echo "===== AUDIT COVERAGE $(date '+%F %T') ====="
if ! python code/audit_deltasg_coverage.py "${COVERAGE_ARGS[@]}"; then
  printf '%s\t%s\t%s\n' "all" "all" "coverage_audit_failed" >> "$FAILED_JOBS_FILE"
  FAILED_JOB_COUNT=$((FAILED_JOB_COUNT + 1))
fi

echo "===== DONE $(date '+%F %T') ====="
echo "$OUT_ROOT"
if (( FAILED_JOB_COUNT > 0 )); then
  echo "ERROR: $FAILED_JOB_COUNT scene/task jobs incomplete; see $FAILED_JOBS_FILE" >&2
  exit 1
fi
