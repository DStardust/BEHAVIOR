#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: $0 OUTPUT_ROOT}"
SCENES_FILE="${SCENES_FILE:-$ROOT/code/configs/env_a_scenes.txt}"
SKIP_SCENE="${SKIP_SCENE:-}"
SCENES="${SCENES:-}"

mkdir -p "$OUT"
overall_status=0
if [[ -n "$SCENES" ]]; then
  read -r -a scene_list <<<"${SCENES//,/ }"
else
  mapfile -t scene_list <"$SCENES_FILE"
fi
for scene in "${scene_list[@]}"; do
  [[ -n "$scene" && "$scene" != \#* && "$scene" != "$SKIP_SCENE" ]] || continue
  echo "[symbolic-all-scenes] start scene=$scene"
  if EXPERT_BACKEND=oracle_symbolic \
      bash "$ROOT/code/run_enva_single_scene_e2e.sh" "$OUT/$scene" "$scene"; then
    echo "[symbolic-all-scenes] passed scene=$scene"
  else
    echo "[symbolic-all-scenes] failed scene=$scene" >&2
    overall_status=2
  fi
done

python - "$OUT" "$SCENES_FILE" "$OUT/all_scenes_report.json" "$SCENES" <<'PY'
import json
import sys
from pathlib import Path

root, scenes_file, report_path = map(Path, sys.argv[1:4])
requested_scenes = sys.argv[4]
if requested_scenes:
    scenes = requested_scenes.replace(",", " ").split()
else:
    scenes = [
        line.strip()
        for line in scenes_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
rows = []
for scene in scenes:
    path = root / scene / "e2e_report.json"
    if not path.is_file():
        continue
    item = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "scene": scene,
        "generation_ok": item.get("generation_ok", 0),
        "expert_completed": item.get("expert_completed", 0),
        "qualified_ok": item.get("qualified_ok", 0),
        "total_tasks": item.get("total_tasks", 0),
        "generation_coverage_rate": item.get("generation_coverage_rate", 0.0),
        "expert_success_rate": item.get("expert_success_rate", 0.0),
        "end_to_end_rate": item.get("end_to_end_rate", 0.0),
        "passed": item.get("passed") is True,
        "report": str(path.resolve()),
    })
qualified = sum(row["qualified_ok"] for row in rows)
total = sum(row["total_tasks"] for row in rows)
generated = sum(row["generation_ok"] for row in rows)
expert_completed = sum(row["expert_completed"] for row in rows)
report = {
    "schema_version": "deltasg_symbolic_all_scenes.v1",
    "expert_backend": "oracle_symbolic",
    "expected_scenes": len(scenes),
    "completed_scenes": len(rows),
    "qualified_ok": qualified,
    "generation_ok": generated,
    "expert_completed": expert_completed,
    "total_tasks": total,
    "generation_coverage_rate": generated / total if total else 0.0,
    "expert_completion_rate": expert_completed / generated if generated else 0.0,
    "expert_success_rate": qualified / expert_completed if expert_completed else 0.0,
    "end_to_end_rate": qualified / generated if generated else 0.0,
    "passed_scenes": sum(row["passed"] for row in rows),
    "scenes": rows,
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

exit "$overall_status"
