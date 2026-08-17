#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 INPUT_ROOT OUTPUT_ROOT [LIMIT]" >&2
  exit 2
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="$2"
LIMIT="${3:-0}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"
BACKEND="${EXPERT_BACKEND:-physical_control}"
REQUESTED_ROBOT="${EXPERT_ROBOT:-}"
MAX_PER_CELL="${EXPERT_MAX_PER_CELL:-0}"
LABELS="${EXPERT_LABELS:-envA_retrieval_delivery envA_open_close envA_appliance}"
TASKS="${EXPERT_TASKS:-all}"
MIN_MANIPULATION_HEIGHT="${EXPERT_MIN_MANIPULATION_HEIGHT:-0.10}"
MAX_MANIPULATION_HEIGHT="${EXPERT_MAX_MANIPULATION_HEIGHT:-1.55}"
MIN_ACCEPT_RATE="${EXPERT_MIN_ACCEPT_RATE:-0.0}"
SAMPLE_TIMEOUT="${EXPERT_SAMPLE_TIMEOUT:-1800}"
SAMPLE_RETRIES="${EXPERT_SAMPLE_RETRIES:-2}"
mkdir -p "$OUTPUT_ROOT/logs"
AUDIT_PROFILE_ARGS=(--require-backend "$BACKEND")
if [[ "$BACKEND" == "physical_control" ]]; then
  AUDIT_PROFILE_ARGS+=(--require-physical-trajectory)
fi

contains_name() {
  local requested="$1"
  local value="$2"
  [[ "$requested" == "all" || " ${requested//,/ } " == *" $value "* ]]
}

python code/validate_deltasg_expert_plans.py \
  --input "$INPUT_ROOT" \
  --output "$OUTPUT_ROOT/plan_preflight.json" \
  >"$OUTPUT_ROOT/logs/plan_preflight.log"

if [[ "$BACKEND" == "oracle_symbolic" ]]; then
  persistent_status=0
  DELTASG_CHILD_TIMEOUT=0 code/run_omnigibson_single_gpu.sh \
    conda run --no-capture-output -n behavior \
    python code/run_deltasg_expert_persistent.py \
      --input-root "$INPUT_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --backend "$BACKEND" \
      --llm-model "$MODEL" \
      --labels "$LABELS" \
      --tasks "$TASKS" \
      --robot "$REQUESTED_ROBOT" \
      --limit "$LIMIT" \
      --max-per-cell "$MAX_PER_CELL" \
      --min-manipulation-height "$MIN_MANIPULATION_HEIGHT" \
      --max-manipulation-height "$MAX_MANIPULATION_HEIGHT" \
      >"$OUTPUT_ROOT/logs/persistent_worker.log" 2>&1 || persistent_status=$?
  audit_status=0
  python code/audit_deltasg_expert.py \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/expert_audit.json" \
    --min-accept-rate "$MIN_ACCEPT_RATE" \
    "${AUDIT_PROFILE_ARGS[@]}" || audit_status=$?
  if [[ "$persistent_status" -ne 0 || "$audit_status" -ne 0 ]]; then
    exit 2
  fi
  exit 0
fi

mapfile -d '' FILES < <(find "$INPUT_ROOT" -type f -name 'online_env*.json' -print0 | sort -z)
count=0
accepted=0
failed=0
skipped=0
declare -A CELL_COUNTS=()
for input in "${FILES[@]}"; do
  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    break
  fi
  if ! jq -e '.ok == true' "$input" >/dev/null 2>&1; then
    continue
  fi
  relative="${input#"$INPUT_ROOT"/}"
  label="${relative%%/*}"
  if ! contains_name "$LABELS" "$label"; then
    continue
  fi
  task_name="$(jq -r '(.task_environment.task.primary_behavior_task // .task.primary_behavior_task // "")' "$input")"
  sample_robot="$(jq -r '(.robot.model // .task_environment.robot.model // "")' "$input")"
  generation_profile="$(jq -r '(.task_environment.generation.solvability_profile // "")' "$input")"
  if [[ -n "$generation_profile" && "$generation_profile" != "$BACKEND" ]]; then
    echo "[expert-batch] generation profile $generation_profile does not match backend $BACKEND in $relative" \
      | tee -a "$OUTPUT_ROOT/logs/master.log"
    failed=$((failed + 1))
    count=$((count + 1))
    continue
  fi
  if [[ "$BACKEND" == "physical_control" && -z "$generation_profile" ]]; then
    echo "[expert-batch] physical VLA run requires an explicit physical_control generation profile in $relative" \
      | tee -a "$OUTPUT_ROOT/logs/master.log"
    failed=$((failed + 1))
    count=$((count + 1))
    continue
  fi
  if [[ -z "$sample_robot" ]]; then
    echo "[expert-batch] missing generation robot in $relative" | tee -a "$OUTPUT_ROOT/logs/master.log"
    failed=$((failed + 1))
    count=$((count + 1))
    continue
  fi
  if [[ -n "$REQUESTED_ROBOT" && "${REQUESTED_ROBOT,,}" != "${sample_robot,,}" ]]; then
    echo "[expert-batch] requested robot $REQUESTED_ROBOT does not match generation robot $sample_robot in $relative" \
      | tee -a "$OUTPUT_ROOT/logs/master.log"
    failed=$((failed + 1))
    count=$((count + 1))
    continue
  fi
  case "${sample_robot,,}" in
    tiago) sample_robot="Tiago" ;;
    r1) sample_robot="R1" ;;
    fetch) sample_robot="fetch" ;;
  esac
  if ! contains_name "$TASKS" "$task_name"; then
    continue
  fi
  remainder="${relative#*/}"
  scene="${remainder%%/*}"
  cell="$label/$scene"
  if [[ "$MAX_PER_CELL" -gt 0 && "${CELL_COUNTS[$cell]:-0}" -ge "$MAX_PER_CELL" ]]; then
    continue
  fi
  sample="${relative%.json}"
  output="$OUTPUT_ROOT/$sample"
  mkdir -p "$output"
  result="$output/expert_result.json"
  if [[ -f "$result" ]] && python code/audit_deltasg_expert.py \
      --root "$output" --min-accept-rate 1.0 \
      "${AUDIT_PROFILE_ARGS[@]}" >/dev/null 2>&1; then
    echo "[expert-batch] skip accepted $relative" | tee -a "$OUTPUT_ROOT/logs/master.log"
    skipped=$((skipped + 1))
    count=$((count + 1))
    CELL_COUNTS[$cell]=$(( ${CELL_COUNTS[$cell]:-0} + 1 ))
    continue
  fi
  log_name="$(printf '%s' "$sample" | tr '/' '_')"
  echo "[expert-batch] $((count + 1)) $relative" | tee -a "$OUTPUT_ROOT/logs/master.log"
  rm -f "$result"
  succeeded=false
  attempt=1
  while [[ "$attempt" -le "$SAMPLE_RETRIES" ]]; do
    echo "[expert-batch] attempt=$attempt/$SAMPLE_RETRIES timeout=${SAMPLE_TIMEOUT}s $relative" \
      | tee -a "$OUTPUT_ROOT/logs/master.log"
    if DELTASG_CHILD_TIMEOUT="$SAMPLE_TIMEOUT" code/run_omnigibson_single_gpu.sh \
        conda run --no-capture-output -n behavior \
        python code/run_deltasg_expert.py \
          --input-json "$input" \
          --output-dir "$output" \
          --backend "$BACKEND" \
          --robot "$sample_robot" \
          --llm-model "$MODEL" \
          --min-manipulation-height "$MIN_MANIPULATION_HEIGHT" \
          --max-manipulation-height "$MAX_MANIPULATION_HEIGHT" \
          >>"$OUTPUT_ROOT/logs/${log_name}.log" 2>&1; then
      succeeded=true
      break
    fi
    echo "[expert-batch] failed attempt=$attempt $relative" | tee -a "$OUTPUT_ROOT/logs/master.log"
    if [[ -f "$result" ]] && jq -e '.accepted == false' "$result" >/dev/null 2>&1; then
      echo "[expert-batch] deterministic rejection recorded; not retrying $relative" \
        | tee -a "$OUTPUT_ROOT/logs/master.log"
      break
    fi
    attempt=$((attempt + 1))
  done
  if [[ "$succeeded" == true ]]; then
    accepted=$((accepted + 1))
  else
    failed=$((failed + 1))
  fi
  count=$((count + 1))
  CELL_COUNTS[$cell]=$(( ${CELL_COUNTS[$cell]:-0} + 1 ))
done

audit_status=0
python code/audit_deltasg_expert.py \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/expert_audit.json" \
  --min-accept-rate "$MIN_ACCEPT_RATE" \
  "${AUDIT_PROFILE_ARGS[@]}" || audit_status=$?
echo "[expert-batch] total=$count accepted_now=$accepted failed_now=$failed skipped_accepted=$skipped"
if [[ "$failed" -gt 0 || "$audit_status" -ne 0 ]]; then
  exit 2
fi
