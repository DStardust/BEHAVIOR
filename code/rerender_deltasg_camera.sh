#!/usr/bin/env bash
# Physical re-render of DeltaSG instances with randomized lighting + FOV.
#
# This is the render-time half of the two-path "camera recolor" design. It
# re-runs run_deltasg_expert.py (which applies _apply_expert_lighting + FOV
# randomization) on the *generation* JSONs of already-generated instances,
# producing N visually-distinct variants per instance.
#
# The offline half (recolor of already-rendered rgb.png frames) lives in
# recolor_deltasg_camera.py and is much cheaper; prefer it unless you need the
# physically-correct relighting and FOV change.
#
# Single-GPU parallelism follows run_envbc_multiscene_e2e.sh: a GPU wrapper
# (DELTASG_GPU_WRAPPER) pins each process to one physical GPU and arbitrates
# concurrency. With DELTASG_GPU_JOBS > 1 the parallel wrapper
# (run_omnigibson_single_gpu_parallel.sh) is used and DELTASG_GPU_SLOTS lets that
# many processes share the one card.
#
# Usage:
#   code/rerender_deltasg_camera.sh INPUT_ROOT OUTPUT_ROOT [NUM_VARIANTS] [LIMIT]
#
#   INPUT_ROOT     e.g. code/outputs/backup_1004/1004_part0
#   OUTPUT_ROOT    e.g. code/outputs/1004_rerendered/1004_part0
#   NUM_VARIANTS   visual variants per instance (default 1)
#   LIMIT          max instances to process (0 = all; default 0)
#
# Env overrides:
#   DELTASG_GPU=2                 physical GPU index (default 2)
#   DELTASG_GPU_JOBS=1            concurrent processes on that one GPU (default 1)
#   DELTASG_GPU_SLOTS=            GPU share slots (default = DELTASG_GPU_JOBS)
#   DELTASG_GPU_WRAPPER=          force a wrapper (else auto: parallel if jobs>1)
#   EXPERT_BACKEND=oracle_symbolic  must match each instance's generation profile
#   EXPERT_ROBOT=                 force a robot (empty = auto-detect from JSON)
#   DELTASG_LLM_MODEL=qwen3.8-max
#   DELTASG_CHILD_TIMEOUT=5400    hard per-instance timeout (s)
set -uo pipefail

INPUT_ROOT="${1:?usage: rerender_deltasg_camera.sh INPUT_ROOT OUTPUT_ROOT [NUM_VARIANTS] [LIMIT]}"
OUTPUT_ROOT="${2:?usage: rerender_deltasg_camera.sh INPUT_ROOT OUTPUT_ROOT [NUM_VARIANTS] [LIMIT]}"
NUM_VARIANTS="${3:-1}"
LIMIT="${4:-0}"
# INPUT_ROOT may be a single part dir (backup_1004/1004_part0) or the parent
# (backup_1004). Normalize so the output tree can mirror the INPUT_ROOT-relative
# path of every generation JSON (which preserves the part level).
INPUT_ROOT="$(realpath -m "$INPUT_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${DELTASG_GPU:-2}"
jobs="${DELTASG_GPU_JOBS:-1}"
backend="${EXPERT_BACKEND:-oracle_symbolic}"
robot="${EXPERT_ROBOT:-}"
model="${DELTASG_LLM_MODEL:-qwen3.8-max}"
child_timeout="${DELTASG_CHILD_TIMEOUT:-5400}"

# GPU wrapper: explicit override wins. The parallel wrapper is used whenever
# more than one process may share the card — either within this script
# (jobs > 1) or across several scripts pinned to the same GPU via a shared
# DELTASG_GPU_SLOTS value (the "N tmux sessions on one card" case). Otherwise
# the single-process wrapper (one per-GPU flock) is enough.
if [[ -n "${DELTASG_GPU_WRAPPER:-}" ]]; then
  gpu_wrapper="$DELTASG_GPU_WRAPPER"
elif (( jobs > 1 )) || [[ -n "${DELTASG_GPU_SLOTS:-}" ]]; then
  gpu_wrapper="code/run_omnigibson_single_gpu_parallel.sh"
else
  gpu_wrapper="code/run_omnigibson_single_gpu.sh"
fi
# Share slots on the card; default to this script's concurrency. Across several
# scripts this must be set to the SAME value so their slot flocks (keyed on GPU
# index only) agree on the total concurrent processes allowed on the card.
slots="${DELTASG_GPU_SLOTS:-$jobs}"

if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_root/.env"
  set +a
fi

mkdir -p "$OUTPUT_ROOT/logs"
log="$OUTPUT_ROOT/logs/rerender.log"
if (( jobs > 1 )) && [[ "$gpu_wrapper" != *parallel* ]]; then
  echo "[rerender] WARNING: jobs=$jobs but wrapper=$gpu_wrapper is single-process; "
  echo "           processes will serialize on the per-GPU lock."
fi
echo "[rerender] input=$INPUT_ROOT output=$OUTPUT_ROOT gpu=$gpu jobs=$jobs slots=$slots wrapper=$gpu_wrapper backend=$backend variants=$NUM_VARIANTS" \
  | tee -a "$log"

# Seed for one (run_id, variant), via the shared visual-variation module so the
# offline recolor and the physical re-render agree.
seed_for() {
  python3 -c "import sys; sys.path.insert(0, '$repo_root/code'); \
from deltasg_visual_variation import visual_seed; \
print(visual_seed('$1', $2))"
}

run_one() {
  local gen_json="$1" run_id="$2" v="$3" seed="$4" out_dir="$5"
  local task_log="$OUTPUT_ROOT/logs/${run_id}.v${v}.log"
  echo "[rerender] start $run_id v$v seed=$seed -> $out_dir" >>"$log"
  mkdir -p "$out_dir"

  local robot_args=()
  if [[ -n "$robot" ]]; then
    robot_args+=(--robot "$robot")
  fi

  DELTASG_GPU="$gpu" DELTASG_GPU_SLOTS="$slots" DELTASG_CHILD_TIMEOUT="$child_timeout" \
    bash "$repo_root/$gpu_wrapper" \
      python "$repo_root/code/run_deltasg_expert.py" \
        --input-json "$gen_json" \
        --output-dir "$out_dir" \
        --backend "$backend" \
        --llm-model "$model" \
        --visual-seed "$seed" \
        --min-manipulation-height 0.10 \
        --max-manipulation-height 1.55 \
        "${robot_args[@]}" \
      >"$task_log" 2>&1 || echo "[rerender] FAILED $run_id v$v" >>"$log"
}

# Phase 1: enumerate (instance, variant) tasks, skipping already-done outputs.
task_list="$(mktemp)"
: >"$task_list"
count=0
while IFS= read -r gen_json; do
  run_id="$(basename "$gen_json" .json)"
  env="$(basename "$(dirname "$gen_json")")"
  # gen_json = INPUT_ROOT/<scene-path>/generation/<env>/online_env_*.json, where
  # <scene-path> may include a part directory (1004_part0/Pomaria_0_int). Mirror
  # the whole INPUT_ROOT-relative prefix into the output tree so the physical
  # re-render lines up with recolor_deltasg_camera.py's layout.
  x_dir="$(dirname "$(dirname "$(dirname "$gen_json")")")"
  rel_x="${x_dir#"$INPUT_ROOT"/}"
  [[ "$rel_x" == "$x_dir" ]] && rel_x=""

  for ((v = 0; v < NUM_VARIANTS; v++)); do
    # Variant 0 keeps the run-id name; variants > 0 get a ``__v{v}`` suffix
    # (mirrors recolor_deltasg_camera.py variant_target_dir).
    run_dir="$run_id"
    if (( v > 0 )); then
      run_dir="${run_id}__v${v}"
    fi
    out_dir="$OUTPUT_ROOT/$rel_x/expert/$env/$run_dir"
    [[ -f "$out_dir/expert_result.json" ]] && continue
    seed="$(seed_for "$run_id" "$v")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$gen_json" "$run_id" "$v" "$seed" "$out_dir" >>"$task_list"
  done

  count=$((count + 1))
  if [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]]; then
    break
  fi
done < <(find "$INPUT_ROOT" -name 'online_env*.json' \
  ! -name 'checkpoint*' ! -name '*_rejected_*' ! -name '*_after_bboxes*' \
  -path '*/generation/*' | sort)

total="$(wc -l <"$task_list")"
echo "[rerender] $total tasks to run (instances=$count)" | tee -a "$log"

# Phase 2: launch with a bounded background pool (up to $jobs at a time).
active=0
while IFS=$'\t' read -r gen_json run_id v seed out_dir; do
  [[ -n "$gen_json" ]] || continue
  run_one "$gen_json" "$run_id" "$v" "$seed" "$out_dir" &
  active=$((active + 1))
  if (( active >= jobs )); then
    wait -n || true
    active=$((active - 1))
  fi
done <"$task_list"
wait
rm -f "$task_list"

# Summarize what actually landed on disk.
python3 - "$OUTPUT_ROOT" "$NUM_VARIANTS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
variants = int(sys.argv[2])
accepted = failed = total = 0
for result in root.rglob("expert_result.json"):
    total += 1
    try:
        data = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        failed += 1
        continue
    if data.get("accepted") is True:
        accepted += 1
    else:
        failed += 1
print(f"[rerender] summary: {total} expert_result.json "
      f"(accepted={accepted}, failed={failed}, expected_variants={variants})")
PY

echo "[rerender] done ($count instances, $total tasks)"
