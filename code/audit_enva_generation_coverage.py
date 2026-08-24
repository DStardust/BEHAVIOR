#!/usr/bin/env python3
"""Audit Env-A generation coverage with evidence-based structural exclusions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TASKS = (
    "deliver_drink", "deliver_food", "deliver_medicine",
    "retrieve_drink", "retrieve_food", "retrieve_medicine", "retrieve_book",
    "retrieve_key", "retrieve_phone", "open_door", "close_door",
    "open_window", "close_window", "open_fridge", "close_fridge",
    "open_cabinet", "close_cabinet", "turn_on_light", "turn_off_light",
    "turn_on_tv", "turn_off_tv", "turn_on_stove", "turn_off_stove",
)

NATIVE_TASK_TOKENS = {
    "open_door": {"door"},
    "close_door": {"door"},
    "open_window": {"window"},
    "close_window": {"window"},
    "open_fridge": {"fridge", "refrigerator"},
    "close_fridge": {"fridge", "refrigerator"},
    "open_cabinet": {"cabinet"},
    "close_cabinet": {"cabinet"},
    "turn_on_light": {"electric", "switch", "light", "lamp"},
    "turn_off_light": {"electric", "switch", "light", "lamp"},
    "turn_on_tv": {"tv", "television"},
    "turn_off_tv": {"tv", "television"},
    "turn_on_stove": {"stove", "oven", "burner"},
    "turn_off_stove": {"stove", "oven", "burner"},
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _tokens(value: str):
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _task_name(item):
    task = (item.get("task_environment") or {}).get("task") or item.get("task") or {}
    return task.get("primary_behavior_task") or ""


def audit_generation(generation_root: Path, required_rate: float):
    accepted = {}
    native_categories = set()
    native_objects = {}
    scene = None
    for path in sorted(generation_root.glob("online_env_a_*.json")):
        item = _read_json(path) or {}
        if scene is None:
            scene = (
                item.get("scene_model")
                or ((item.get("task_environment") or {}).get("base_scene") or {}).get(
                    "scene_model"
                )
            )
        for node in (item.get("before_graph") or {}).get("nodes") or []:
            object_id = str(node.get("id") or "")
            if node.get("type") != "object" or object_id.startswith("online_env_"):
                continue
            category = str(node.get("category") or "").lower()
            if not category:
                continue
            native_categories.add(category)
            native_objects.setdefault(category, []).append(object_id)
        task = _task_name(item)
        if task and item.get("ok") is True:
            accepted[task] = str(path.resolve())

    category_tokens = {category: _tokens(category) for category in native_categories}
    rows = []
    for task in TASKS:
        if task in accepted:
            rows.append({"task": task, "status": "generated", "artifact": accepted[task]})
            continue
        required_tokens = NATIVE_TASK_TOKENS.get(task)
        if required_tokens is not None:
            matching_categories = sorted(
                category for category, tokens in category_tokens.items()
                if tokens & required_tokens
            )
            if not matching_categories:
                rows.append({
                    "task": task,
                    "status": "structurally_ineligible",
                    "reason": "scene_missing_native_target_category",
                    "required_category_tokens": sorted(required_tokens),
                    "scene_native_category_count": len(native_categories),
                })
                continue
            rows.append({
                "task": task,
                "status": "unresolved",
                "reason": "native_target_present_but_generation_failed",
                "matching_native_categories": matching_categories,
                "matching_native_object_ids": sorted({
                    object_id
                    for category in matching_categories
                    for object_id in native_objects.get(category, [])
                }),
            })
            continue
        rows.append({"task": task, "status": "unresolved", "reason": "generated_task_failed"})

    structural = sum(row["status"] == "structurally_ineligible" for row in rows)
    generated = sum(row["status"] == "generated" for row in rows)
    eligible = len(rows) - structural
    rate = generated / eligible if eligible else 0.0
    return {
        "schema_version": "deltasg_enva_generation_audit.v1",
        "scene": scene or generation_root.parent.name,
        "generation_root": str(generation_root.resolve()),
        "total_tasks": len(rows),
        "eligible_tasks": eligible,
        "structurally_ineligible": structural,
        "generated": generated,
        "unresolved": eligible - generated,
        "generation_rate": rate,
        "required_rate": required_rate,
        "passed": rate >= required_rate,
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_root")
    parser.add_argument("--required-rate", type=float, default=0.80)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_generation(Path(args.generation_root), args.required_rate)
    payload = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
