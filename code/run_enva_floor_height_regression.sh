#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-code/outputs/enva_floor_height_regression}"
SCENE="${SCENE:-Beechwood_0_int}"
ROBOT="${ROBOT:-Tiago}"
GPU="${DELTASG_GPU:-auto}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"
CHILD_TIMEOUT="${DELTASG_CHILD_TIMEOUT:-900}"
PROCESS_RETRIES="${DELTASG_PROCESS_RETRIES:-3}"

run_generation() {
  local task="$1"
  local category="$2"
  local asset_model="$3"
  local output="$4"
  local seed="$5"
  local attempt exit_code=1
  for ((attempt = 1; attempt <= PROCESS_RETRIES; attempt++)); do
    rm -rf "$output"
    mkdir -p "$output"
    if DELTASG_GPU="$GPU" DELTASG_CHILD_TIMEOUT="$CHILD_TIMEOUT" \
      code/run_omnigibson_single_gpu.sh \
      conda run --no-capture-output -n behavior \
      python code/run_online_deltasg.py \
        --scene "$SCENE" --robot "$ROBOT" --env-type A \
        --task "$task" --task-categories retrieval_delivery \
        --target-asset-category "$category" --target-asset-model "$asset_model" \
        --target-placement-mode floor --num-envs 1 \
        --task-objects 1 --context-objects 0 --warmup-steps 20 --settle-steps 5 \
        --llm-model "$MODEL" --max-llm-retries 1 --max-retries 1 \
        --placement-timeout 60 --relation-timeout 10 \
        --max-placement-attempts 6 --max-total-placement-time 180 \
        --max-global-cameras 3 --max-camera-pose-attempts 6 \
        --camera-pose-render-steps 4 \
        --min-manipulation-height 0.10 --max-manipulation-height 1.55 \
        --output-dir "$output" --seed "$seed"; then
      return 0
    else
      exit_code=$?
    fi
    echo "[floor-regression] generation attempt $attempt/$PROCESS_RETRIES failed (exit=$exit_code)" >&2
    sleep 5
  done
  return "$exit_code"
}

run_expert() {
  local input_json="$1"
  local output="$2"
  local attempt exit_code=1
  for ((attempt = 1; attempt <= PROCESS_RETRIES; attempt++)); do
    rm -rf "$output"
    if DELTASG_GPU="$GPU" DELTASG_CHILD_TIMEOUT="$CHILD_TIMEOUT" \
      code/run_omnigibson_single_gpu.sh \
      conda run --no-capture-output -n behavior \
      python code/run_deltasg_expert.py \
        --input-json "$input_json" --output-dir "$output" \
        --backend oracle_symbolic --robot "$ROBOT" --llm-model "$MODEL" \
        --min-manipulation-height 0.10 --max-manipulation-height 1.55; then
      return 0
    else
      exit_code=$?
    fi
    echo "[floor-regression] expert attempt $attempt/$PROCESS_RETRIES failed (exit=$exit_code)" >&2
    sleep 5
  done
  return "$exit_code"
}

mkdir -p "$ROOT"
rm -f "$ROOT/report.json"
TALL_GENERATION="$ROOT/tall_bottle/generation"
TALL_EXPERT="$ROOT/tall_bottle/expert"
SHORT_GENERATION="$ROOT/short_phone/generation"

run_generation retrieve_drink bottle_of_water cytqio "$TALL_GENERATION" 810901 \
  >"$ROOT/tall_bottle.log" 2>&1
TALL_JSON="$(find "$TALL_GENERATION" -maxdepth 1 -type f -name 'online_env_a_*.json' ! -name '*rejected*' | sort | head -n 1)"
if [[ -z "$TALL_JSON" ]] || ! jq -e '.ok == true' "$TALL_JSON" >/dev/null; then
  echo "[floor-regression] tall floor bottle did not generate a successful sample" >&2
  exit 2
fi
python code/audit_deltasg_outputs.py --root "$TALL_GENERATION" --ok-only --fail-on-issues \
  >"$ROOT/tall_generation_audit.log"

run_expert "$TALL_JSON" "$TALL_EXPERT" \
  >"$ROOT/tall_expert.log" 2>&1
python code/audit_deltasg_expert.py --root "$TALL_EXPERT" --min-accept-rate 1.0 \
  --output "$ROOT/tall_expert_audit.json" >"$ROOT/tall_expert_audit.log"

set +e
run_generation retrieve_phone cell_phone dbhfuh "$SHORT_GENERATION" 810902 \
  >"$ROOT/short_phone.log" 2>&1
short_code=$?
set -e
if find "$SHORT_GENERATION" -maxdepth 1 -type f -name 'online_env_a_*.json' ! -name '*rejected*' \
    -exec jq -e '.ok == true' {} \; | grep -q true; then
  echo "[floor-regression] short floor phone was incorrectly accepted" >&2
  exit 2
fi
if ! find "$SHORT_GENERATION" -maxdepth 1 -type f -name '*.json' \
    -exec jq -c '.. | objects | select(.error? == "task_object_floor_height_out_of_range")' {} \; \
    2>/dev/null | grep -q 'task_object_floor_height_out_of_range'; then
  echo "[floor-regression] short phone rejection lacks height-gate evidence" >&2
  exit 2
fi

jq -n \
  --arg scene "$SCENE" \
  --arg tall_json "$TALL_JSON" \
  --argjson short_process_code "$short_code" \
  '{
    schema_version: "enva_floor_height_regression.v1",
    ok: true,
    scene: $scene,
    min_manipulation_height: 0.10,
    max_manipulation_height: 1.55,
    tall_bottle: {expected: "accepted", generation_json: $tall_json, expert_accepted: true},
    short_phone: {
      expected: "task_object_floor_height_out_of_range",
      process_code: $short_process_code,
      accepted: false
    }
  }' >"$ROOT/report.json"
cat "$ROOT/report.json"
