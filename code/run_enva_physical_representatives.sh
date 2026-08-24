#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 COVERAGE_RESULT_JSON OUTPUT_ROOT" >&2
  exit 2
fi

COVERAGE_RESULT="$1"
OUTPUT_ROOT="$2"
SAMPLE_TIMEOUT="${DELTASG_PHYSICAL_TIMEOUT:-3600}"
GPU="${DELTASG_GPU:-auto}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"

mkdir -p "$OUTPUT_ROOT"

if ! jq -e '
  .completed == .total
  and .total > 0
  and .structural_resolution_rate == 1
  and .eligible_accept_rate >= .min_eligible_accept_rate
  and .runtime_ineligible_rate <= .max_runtime_ineligible_rate
  and .expert_audit_process_code == 0
  and .expert_backend == "physical_control"
  and .generation_solvability_profile == "physical_control"
' "$COVERAGE_RESULT" >/dev/null; then
  echo "[physical-regression] coverage input has not passed its formal gates" >&2
  exit 2
fi

select_input() {
  local selector="$1"
  jq -r --arg selector "$selector" '
    [
      .results[]
      | select(.status == "accepted" and (.generation_json // "") != "")
      | select(
          ($selector == "retrieve" and (.task | startswith("retrieve_")))
          or ($selector == "deliver" and (.task | startswith("deliver_")))
          or ($selector == "open_close" and .label == "envA_open_close")
          or ($selector == "appliance" and .label == "envA_appliance")
        )
    ][0].generation_json // empty
  ' "$COVERAGE_RESULT"
}

declare -A REPRESENTATIVE_INPUTS
for selector in retrieve deliver open_close appliance; do
  input_json="$(select_input "$selector")"
  if [[ -z "$input_json" || ! -f "$input_json" ]]; then
    echo "[physical-regression] no accepted $selector input in $COVERAGE_RESULT" >&2
    exit 2
  fi
  REPRESENTATIVE_INPUTS["$selector"]="$input_json"
  if [[ "$(jq -r '(.task_environment.generation.solvability_profile // "")' "$input_json")" != "physical_control" ]]; then
    echo "[physical-regression] $selector input lacks physical_control generation profile" >&2
    exit 2
  fi
done

run_one() {
  local selector="$1"
  local input_json
  input_json="${REPRESENTATIVE_INPUTS[$selector]}"

  local task
  task="$(jq -r '.task_environment.task.primary_behavior_task' "$input_json")"
  local sample_robot
  sample_robot="$(jq -r '(.robot.model // .task_environment.robot.model // "")' "$input_json")"
  case "${sample_robot,,}" in
    tiago) sample_robot="Tiago" ;;
    r1) sample_robot="R1" ;;
  esac
  if [[ "$sample_robot" != "R1" && "$sample_robot" != "Tiago" ]]; then
    echo "[physical-regression] generation robot must be R1 or Tiago, got '$sample_robot' in $input_json" >&2
    return 2
  fi
  local output="$OUTPUT_ROOT/${selector}_${task}"
  if [[ -f "$output/expert_result.json" ]] && \
     python code/audit_deltasg_expert.py \
       --root "$output" --min-accept-rate 1.0 \
       --require-backend physical_control --require-low-level-vla-actions \
       --output "$output/resume_audit.json" >/dev/null 2>&1; then
    echo "[physical-regression] resume accepted selector=$selector task=$task"
    return 0
  fi

  echo "[physical-regression] run selector=$selector task=$task input=$input_json"
  rm -f "$output/expert_result.json"
  DELTASG_GPU="$GPU" DELTASG_CHILD_TIMEOUT="$SAMPLE_TIMEOUT" \
    bash code/run_omnigibson_single_gpu.sh \
      conda run --no-capture-output -n behavior \
      python code/run_deltasg_expert.py \
        --input-json "$input_json" \
        --output-dir "$output" \
        --backend physical_control \
        --robot "$sample_robot" \
        --llm-model "$MODEL" \
        --min-manipulation-height 0.10 \
        --max-manipulation-height 1.55 \
      2>&1 | tee "$output.log"
}

for selector in retrieve deliver open_close appliance; do
  run_one "$selector"
done

python code/audit_deltasg_expert.py \
  --root "$OUTPUT_ROOT" \
  --min-accept-rate 1.0 \
  --require-backend physical_control \
  --require-low-level-vla-actions \
  --output "$OUTPUT_ROOT/audit.json"

jq -e '
  .ok == true
  and .total == 4
  and .accepted == 4
  and (.qa_gate_violations | length) == 0
  and (.vla_backend_label_violations | length) == 0
  and (.artifact_violations | length) == 0
' "$OUTPUT_ROOT/audit.json" >/dev/null

jq -n \
  --arg coverage_result "$COVERAGE_RESULT" \
  --arg output_root "$OUTPUT_ROOT" \
  '{schema_version:"enva_physical_representatives.v1",ok:true,coverage_result:$coverage_result,output_root:$output_root}' \
  >"$OUTPUT_ROOT/report.json"

echo "[physical-regression] accepted 4/4; report=$OUTPUT_ROOT/report.json"
