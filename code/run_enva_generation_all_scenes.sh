#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: $0 OUTPUT_ROOT}"
SCENES_FILE="${SCENES_FILE:-$ROOT/code/configs/env_a_scenes.txt}"
SCENES="${SCENES:-}"

mkdir -p "$OUT"
if [[ -n "$SCENES" ]]; then
  read -r -a scene_list <<<"${SCENES//,/ }"
else
  mapfile -t scene_list <"$SCENES_FILE"
fi

overall_status=0
for scene in "${scene_list[@]}"; do
  [[ -n "$scene" && "$scene" != \#* ]] || continue
  echo "[generation-all-scenes] start scene=$scene"
  if GENERATION_ONLY=1 EXPERT_BACKEND=oracle_symbolic \
      bash "$ROOT/code/run_enva_single_scene_e2e.sh" "$OUT/$scene" "$scene"; then
    echo "[generation-all-scenes] passed scene=$scene"
  else
    echo "[generation-all-scenes] failed scene=$scene" >&2
    overall_status=2
  fi
done

python - "$OUT" "$SCENES_FILE" "$OUT/all_scenes_generation_audit.json" "$SCENES" <<'PY'
import json
import sys
from pathlib import Path

root, scenes_file, report_path = map(Path, sys.argv[1:4])
requested = sys.argv[4]
scenes = (
    requested.replace(",", " ").split()
    if requested
    else [
        line.strip()
        for line in scenes_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
)
rows = []
for scene in scenes:
    path = root / scene / "generation_audit.json"
    if not path.is_file():
        continue
    item = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "scene": scene,
        "generated": item["generated"],
        "eligible_tasks": item["eligible_tasks"],
        "structurally_ineligible": item["structurally_ineligible"],
        "unresolved": item["unresolved"],
        "generation_rate": item["generation_rate"],
        "passed": item["passed"],
        "report": str(path.resolve()),
    })
generated = sum(row["generated"] for row in rows)
eligible = sum(row["eligible_tasks"] for row in rows)
report = {
    "schema_version": "deltasg_enva_generation_all_scenes.v1",
    "expected_scenes": len(scenes),
    "completed_scenes": len(rows),
    "passed_scenes": sum(row["passed"] for row in rows),
    "generated": generated,
    "eligible_tasks": eligible,
    "structurally_ineligible": sum(row["structurally_ineligible"] for row in rows),
    "unresolved": sum(row["unresolved"] for row in rows),
    "generation_rate": generated / eligible if eligible else 0.0,
    "required_rate": 0.80,
    "passed": len(rows) == len(scenes) and all(row["passed"] for row in rows),
    "scenes": rows,
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

exit "$overall_status"
