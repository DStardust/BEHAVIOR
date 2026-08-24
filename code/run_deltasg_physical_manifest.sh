#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST_TSV OUTPUT_ROOT" >&2
  exit 2
fi

manifest="$1"
output_root="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${DELTASG_GPU:-2}"
model="${DELTASG_LLM_MODEL:-qwen3.8-max}"

set -a
source "$repo_root/.env"
set +a
mkdir -p "$output_root"

while IFS=$'\t' read -r label input robot; do
  [[ -n "$label" && "${label:0:1}" != "#" ]] || continue
  case_root="$output_root/$label"
  mkdir -p "$case_root"
  rm -f "$case_root/expert_result.json" "$case_root/audit.json"
  profile="$(jq -r '(.task_environment.generation.solvability_profile // "")' "$input")"
  if [[ "$profile" != "physical_control" ]]; then
    echo "[physical-manifest] reject label=$label generation_profile=${profile:-missing}"
    printf '%s\n' 2 >"$case_root/exit_code"
    printf '%s\n' 1 >"$case_root/audit_exit_code"
    continue
  fi
  echo "[physical-manifest] label=$label input=$input robot=$robot"
  set +e
  DELTASG_GPU="$gpu" DELTASG_CHILD_TIMEOUT=5400 \
    bash code/run_omnigibson_single_gpu.sh \
      conda run --no-capture-output -n behavior \
      python code/run_deltasg_expert.py \
        --input-json "$input" --output-dir "$case_root" \
        --backend physical_control --robot "$robot" \
        --llm-model "$model" \
        --min-manipulation-height 0.10 --max-manipulation-height 1.55 \
      >"$case_root/run.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" >"$case_root/exit_code"
  audit_code=0
  python code/audit_deltasg_expert.py \
    --root "$case_root" --min-accept-rate 1.0 \
    --require-backend physical_control --require-low-level-vla-actions \
    --output "$case_root/audit.json" >/dev/null || audit_code=$?
  printf '%s\n' "$audit_code" >"$case_root/audit_exit_code"
done <"$manifest"

python - "$manifest" "$output_root" <<'PY'
import json
import sys
from pathlib import Path

manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    label, input_path, robot = line.split("\t")
    result_path = root / label / "expert_result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    backend = result.get("backend") or {}
    run_exit = int((root / label / "exit_code").read_text()) if (root / label / "exit_code").is_file() else -1
    audit_exit = int((root / label / "audit_exit_code").read_text()) if (root / label / "audit_exit_code").is_file() else -1
    accepted = result.get("accepted") is True
    vla_eligible = (
        run_exit == 0
        and audit_exit == 0
        and accepted
        and Path(str(result.get("input") or "")).resolve() == Path(input_path).resolve()
        and backend.get("name") == "physical_control"
        and backend.get("generation_profile_verified") is True
        and backend.get("physical_solubility_validation") is True
        and backend.get("low_level_vla_actions_eligible") is True
        and backend.get("assisted_interaction") is not True
        and backend.get("complete_action_trace") is True
    )
    rows.append({
        "label": label,
        "input": input_path,
        "robot": robot,
        "scene": result.get("scene"),
        "task_name": result.get("task_name"),
        "accepted": accepted,
        "vla_eligible": vla_eligible,
        "run_exit_code": run_exit,
        "audit_exit_code": audit_exit,
        "rejection": result.get("rejection"),
        "result": str(result_path) if result_path.is_file() else None,
    })
report = {
    "schema_version": "deltasg_physical_manifest.v1",
    "total": len(rows),
    "accepted": sum(row["accepted"] for row in rows),
    "accept_rate": sum(row["accepted"] for row in rows) / len(rows) if rows else 0.0,
    "vla_eligible": sum(row["vla_eligible"] for row in rows),
    "vla_eligible_rate": (
        sum(row["vla_eligible"] for row in rows) / len(rows) if rows else 0.0
    ),
    "results": rows,
}
(root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
