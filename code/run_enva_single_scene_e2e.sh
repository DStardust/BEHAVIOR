#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: $0 OUTPUT_ROOT [SCENE]}"
SCENE="${2:-Beechwood_0_int}"
MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"
BACKEND="${EXPERT_BACKEND:-physical_control}"
if [[ "$BACKEND" != "physical_control" && "$BACKEND" != "oracle_symbolic" ]]; then
  echo "unsupported EXPERT_BACKEND=$BACKEND" >&2
  exit 2
fi
TASKS="deliver_drink,deliver_food,deliver_medicine,retrieve_drink,retrieve_food,retrieve_medicine,retrieve_book,retrieve_key,retrieve_phone,open_door,close_door,open_window,close_window,open_fridge,close_fridge,open_cabinet,close_cabinet,turn_on_light,turn_off_light,turn_on_tv,turn_off_tv,turn_on_stove,turn_off_stove"
GENERATION="$OUT/generation"
EXPERT="$OUT/expert"

mkdir -p "$GENERATION" "$EXPERT"
RUN_MARKER="$OUT/.e2e_running"
printf 'GEN\n' >"$RUN_MARKER"
cleanup_run_marker() {
  rm -f "$RUN_MARKER"
}
trap cleanup_run_marker EXIT
set -a
source "$ROOT/.env"
set +a

cd "$ROOT"
generation_status=0
if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
code/run_omnigibson_single_gpu.sh \
  conda run --no-capture-output -n behavior \
  python code/run_online_deltasg.py \
    --scene "$SCENE" --robot Tiago --env-type A \
    --task-sequence "$TASKS" \
    --task-objects 1 --context-objects 0 \
    --warmup-steps 20 --settle-steps 5 \
    --llm-model "$MODEL" --max-llm-retries 4 --max-retries 3 \
    --placement-timeout 60 --relation-timeout 10 \
    --max-placement-attempts 6 --max-total-placement-time 180 \
    --max-global-cameras 3 --max-camera-pose-attempts 6 \
    --camera-pose-render-steps 4 \
    --min-manipulation-height 0.10 --max-manipulation-height 1.55 \
    --solvability-profile "$BACKEND" --checkpoint-interval 1 \
    --output-dir "$GENERATION" --seed 140814 \
    >"$OUT/generation.log" 2>&1 || generation_status=$?
fi

expert_status=0
if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
printf 'EXP\n' >"$RUN_MARKER"
DELTASG_LLM_MODEL="$MODEL" \
EXPERT_BACKEND="$BACKEND" \
EXPERT_LABELS=all \
EXPERT_TASKS=all \
EXPERT_SAMPLE_RETRIES=1 \
EXPERT_MIN_ACCEPT_RATE=0.0 \
bash code/run_deltasg_expert_batch.sh "$GENERATION" "$EXPERT" \
  >"$OUT/expert_batch.log" 2>&1 || expert_status=$?
fi

python - "$GENERATION" "$EXPERT" "$OUT/e2e_report.json" "$SCENE" \
  "$MODEL" "$BACKEND" "$generation_status" "$expert_status" <<'PY'
import json
import sys
from pathlib import Path

generation_root, expert_root, report_path = map(Path, sys.argv[1:4])
scene, model, required_backend = sys.argv[4:7]
generation_status, expert_status = map(int, sys.argv[7:9])
tasks = [
    "deliver_drink", "deliver_food", "deliver_medicine",
    "retrieve_drink", "retrieve_food", "retrieve_medicine", "retrieve_book",
    "retrieve_key", "retrieve_phone", "open_door", "close_door",
    "open_window", "close_window", "open_fridge", "close_fridge",
    "open_cabinet", "close_cabinet", "turn_on_light", "turn_off_light",
    "turn_on_tv", "turn_off_tv", "turn_on_stove", "turn_off_stove",
]
generated = {}
for path in generation_root.glob("online_env*.json"):
    item = json.loads(path.read_text(encoding="utf-8"))
    task = ((item.get("task_environment") or {}).get("task") or item.get("task") or {}).get(
        "primary_behavior_task"
    )
    if task and item.get("ok") is True:
        generated[task] = str(path.resolve())

experts = {}
for path in expert_root.rglob("expert_result.json"):
    item = json.loads(path.read_text(encoding="utf-8"))
    task = item.get("task_name")
    if not task:
        continue
    backend = item.get("backend") or {}
    recorded_input = Path(str(item.get("input") or ""))
    input_matches = (
        task in generated
        and recorded_input.is_absolute()
        and recorded_input.resolve() == Path(generated[task]).resolve()
    )
    physical = (
        input_matches
        and item.get("accepted") is True
        and backend.get("name") == "physical_control"
        and backend.get("generation_profile_verified") is True
        and backend.get("complete_action_trace") is True
        and backend.get("physical_trajectory_available") is True
        and int(backend.get("physical_action_count") or 0) > 0
        and int(backend.get("physical_nonzero_action_count") or 0) > 0
    )
    strict = physical and backend.get("low_level_vla_actions_eligible") is True
    hybrid = physical and backend.get("assisted_interaction") is True
    steps = item.get("steps") or []
    symbolic = (
        input_matches
        and item.get("accepted") is True
        and backend.get("name") == "oracle_symbolic"
        and backend.get("generation_profile_verified") is True
        and item.get("qa_eligible") is True
        and bool(steps)
        and all(
            step.get("pre_observation") and step.get("post_observation")
            for step in steps
        )
    )
    qualified = symbolic if required_backend == "oracle_symbolic" else physical and (strict or hybrid)
    experts[task] = {
        "accepted": item.get("accepted") is True,
        "input_matches_generation": input_matches,
        "physical_trajectory": physical,
        "strict_vla": strict,
        "hybrid_official_state": hybrid,
        "symbolic_pre_post_ok": symbolic,
        "qualified": qualified,
        "result": str(path.resolve()),
        "rejection": item.get("rejection"),
    }

rows = []
for task in tasks:
    expert = experts.get(task) or {}
    rows.append({
        "task": task,
        "generation_ok": task in generated,
        "generation_json": generated.get(task),
        **expert,
    })
qualified = sum(row.get("qualified") is True for row in rows)
generation_ok = sum(row["generation_ok"] for row in rows)
expert_completed = sum(
    row.get("input_matches_generation") is True and bool(row.get("result"))
    for row in rows
)
expert_success_rate = qualified / expert_completed if expert_completed else 0.0
end_to_end_rate = qualified / generation_ok if generation_ok else 0.0
report = {
    "schema_version": "deltasg_single_scene_e2e.v1",
    "scene": scene,
    "llm_model": model,
    "expert_backend": required_backend,
    "generation_process_code": generation_status,
    "expert_batch_process_code": expert_status,
    "total_tasks": len(tasks),
    "generation_ok": generation_ok,
    "generation_coverage_rate": generation_ok / len(tasks),
    "expert_completed": expert_completed,
    "expert_completion_rate": expert_completed / generation_ok if generation_ok else 0.0,
    "expert_success_rate": expert_success_rate,
    "expert_accepted": sum(row.get("accepted") is True for row in rows),
    "physical_trajectory_ok": sum(row.get("physical_trajectory") is True for row in rows),
    "strict_vla_ok": sum(row.get("strict_vla") is True for row in rows),
    "hybrid_official_state_ok": sum(row.get("hybrid_official_state") is True for row in rows),
    "symbolic_pre_post_ok": sum(row.get("symbolic_pre_post_ok") is True for row in rows),
    "qualified_ok": qualified,
    "end_to_end_rate": end_to_end_rate,
    "required_rate": 0.80,
    "passed": (
        generation_ok > 0
        and expert_completed == generation_ok
        and end_to_end_rate >= 0.80
    ),
    "tasks": rows,
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 2)
PY
