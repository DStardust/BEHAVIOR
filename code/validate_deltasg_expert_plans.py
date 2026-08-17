"""Offline preflight for DeltaSG expert plans; does not import OmniGibson."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from deltasg_expert import ExpertPlanError, compile_expert_plan


def input_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(path.rglob("online_env*.json"))


def main():
    parser = argparse.ArgumentParser(description="Compile-check DeltaSG solution plans before simulation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    rows = []
    counts = Counter()
    by_task = {}
    for path in input_files(Path(args.input)):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({"path": str(path), "valid": False, "error": repr(exc)})
            counts["invalid"] += 1
            continue
        if run.get("ok") is not True:
            continue
        try:
            plan = compile_expert_plan(run)
            row = {
                "path": str(path),
                "valid": True,
                "task_name": plan.task_name,
                "task_family": plan.task_family,
                "num_steps": len(plan.steps),
                "warnings": list(plan.warnings),
            }
            counts["valid"] += 1
            counts["repaired"] += int(bool(plan.warnings))
        except ExpertPlanError as exc:
            task = ((run.get("task_environment") or {}).get("task") or run.get("task") or {})
            row = {
                "path": str(path),
                "valid": False,
                "task_name": task.get("primary_behavior_task"),
                "error": str(exc),
            }
            counts["invalid"] += 1
        rows.append(row)
        task_name = str(row.get("task_name") or "unknown")
        task_counts = by_task.setdefault(task_name, {"valid": 0, "invalid": 0, "repaired": 0})
        task_counts["valid" if row["valid"] else "invalid"] += 1
        task_counts["repaired"] += int(bool(row.get("warnings")))

    result = {
        "schema_version": "deltasg_expert_plan_preflight.v1",
        "ok": counts["invalid"] == 0 and counts["valid"] > 0,
        "valid": counts["valid"],
        "invalid": counts["invalid"],
        "repaired": counts["repaired"],
        "by_task": dict(sorted(by_task.items())),
        "items": rows,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.fail_on_invalid and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
