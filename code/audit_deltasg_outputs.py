#!/usr/bin/env python3
"""Audit generated DeltaSG samples and optional visualization bbox outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_LABELS = {
    "envA_retrieval_delivery": {"env_type": "Env-A", "category": "retrieval_delivery"},
    "envA_open_close": {"env_type": "Env-A", "category": "open_close"},
    "envA_appliance": {"env_type": "Env-A", "category": "appliance"},
    "envB_fire": {"env_type": "Env-B", "category": "anomaly_response"},
    "envC_fire_disambiguation": {"env_type": "Env-C", "category": "semantic_disambiguation"},
    "envC_retrieval_delivery": {"env_type": "Env-C", "category": "retrieval_delivery"},
    "envC_open_close": {"env_type": "Env-C", "category": "open_close"},
    "envC_appliance": {"env_type": "Env-C", "category": "appliance"},
    "envC_all": {"env_type": "Env-C", "category": "mixed"},
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_files(root: Path):
    return sorted(
        p for p in root.rglob("online_env*.json")
        if not p.name.startswith("checkpoint")
        and "_rejected_" not in p.name
        and "_after_bboxes" not in p.name
        and "visualizations" not in p.parts
    )


def infer_label(path: Path):
    for part in path.parts:
        if part in EXPECTED_LABELS:
            return part
    return None


def check_run(path: Path, run: dict):
    issues = []
    te = run.get("task_environment")
    if not run.get("ok"):
        issues.append("run_not_ok")
    if not isinstance(te, dict):
        issues.append("missing_task_environment")
        return issues

    task = te.get("task") or {}
    env_type = te.get("env_type")
    label = infer_label(path)
    expected = EXPECTED_LABELS.get(label or "")
    if expected and env_type != expected["env_type"]:
        issues.append(f"wrong_env_type:{env_type}")

    if not task.get("instruction"):
        issues.append("missing_instruction")
    if not te.get("robot"):
        issues.append("missing_robot")
    if not te.get("camera"):
        issues.append("missing_camera")
    if not te.get("solution_plan"):
        issues.append("missing_solution_plan")

    validation = te.get("validation") or run.get("validation") or {}
    if validation.get("ok") is False:
        issues.append("validation_not_ok")
    if validation.get("duplicate_sample"):
        issues.append("duplicate_sample")
    integrity = validation.get("scene_integrity") or {}
    if not integrity:
        issues.append("missing_scene_integrity_report")
    elif not integrity.get("ok", False):
        issues.append("scene_integrity_failed")
    settling = validation.get("settling") or {}
    if not settling:
        issues.append("missing_settling_report")
    elif not settling.get("all_within_threshold", False):
        issues.append("settling_or_contact_failed")

    if env_type == "Env-A":
        primary = task.get("primary_behavior_task") or ""
        is_retrieval = any(token in primary for token in ("retrieve", "deliver", "put_object"))
        if expected and expected["category"] == "retrieval_delivery" and not any(
            token in primary for token in ("retrieve", "deliver", "put_object")
        ):
            issues.append(f"unexpected_envA_retrieval_task:{primary}")
        if expected and expected["category"] == "open_close" and not any(
            token in primary for token in ("open", "close")
        ):
            issues.append(f"unexpected_envA_open_close_task:{primary}")
        if expected and expected["category"] == "appliance" and not any(
            token in primary for token in ("turn_on", "turn_off")
        ):
            issues.append(f"unexpected_envA_appliance_task:{primary}")
        # Retrieval/delivery should usually introduce or reuse a task object.
        if is_retrieval:
            task_objects = te.get("task_objects") or []
            added_objects = te.get("added_objects") or []
            if not task_objects and not added_objects:
                issues.append("envA_retrieval_no_task_or_added_objects")
            instruction_lower = str(task.get("instruction") or "").lower()
            for obj in added_objects:
                placement = obj.get("placement") or {}
                support = str(placement.get("support_category") or "").lower().replace("_", " ")
                if support == "floors":
                    support = "floor"
                if support and support not in instruction_lower:
                    issues.append("envA_retrieval_instruction_support_mismatch")
                    break
            if primary.startswith("deliver_"):
                source_supports = {
                    (obj.get("placement") or {}).get("support_object_id")
                    for obj in added_objects
                }
                destinations = [
                    obj for obj in task.get("plan_objects") or []
                    if obj.get("reference_only") and obj.get("object_id") not in source_supports
                ]
                if not destinations:
                    issues.append("envA_delivery_missing_distinct_destination")

    elif env_type == "Env-B":
        state_changed = te.get("state_changed_objects") or []
        if not any(((obj.get("states") or {}).get("on_fire")) for obj in state_changed):
            issues.append("envB_missing_on_fire_state")
        added = te.get("added_objects") or []
        task_objects = te.get("task_objects") or []
        tool_text = " ".join(
            str(obj.get("category") or obj.get("object_name") or obj.get("object_id") or "")
            for obj in [*added, *task_objects]
        )
        if "fire_extinguisher" not in tool_text:
            issues.append("envB_missing_extinguisher")
        plan_text = " ".join(step.get("primitive", "") for step in te.get("solution_plan") or [])
        if "INTERACT" not in plan_text:
            issues.append("envB_plan_missing_interact")

    elif env_type == "Env-C":
        reasoning = te.get("semantic_reasoning") or task.get("semantic_reasoning") or run.get("task_instance", {}).get("semantic_reasoning")
        constraints = task.get("semantic_constraints") or []
        if not constraints:
            issues.append("envC_missing_semantic_constraints")
        if not reasoning:
            issues.append("envC_missing_semantic_reasoning")
        roles = []
        for obj in te.get("added_objects") or []:
            roles.extend(obj.get("semantic_roles") or [])
            role = obj.get("semantic_role") or obj.get("role")
            if role:
                roles.append(role)
        for obj in task.get("plan_objects") or []:
            roles.extend(obj.get("semantic_roles") or [])
            role = obj.get("semantic_role") or obj.get("role")
            if role:
                roles.append(role)
        roles_text = " ".join(roles)
        if "candidate_solution" not in roles_text:
            issues.append("envC_missing_candidate_solution")
        if "semantic_distractor" not in roles_text:
            issues.append("envC_missing_semantic_distractor")
        state_changed = te.get("state_changed_objects") or []
        primary = task.get("primary_behavior_task") or ""
        is_fire = "fire" in primary or "fire" in (label or "")
        if is_fire and not any(((obj.get("states") or {}).get("on_fire")) for obj in state_changed):
            issues.append("envC_missing_fire_state")

    return issues


def bbox_file_for_run(run_path: Path, vis_root: Path):
    return next(vis_root.rglob(f"{run_path.stem}_after_bboxes.json"), None)


def summarize_bboxes(run_path: Path, vis_root: Path | None):
    if vis_root is None:
        return None
    bbox_path = bbox_file_for_run(run_path, vis_root)
    if bbox_path is None:
        return {"exists": False}
    data = load_json(bbox_path)
    targets = data.get("target_objects") or []
    visible = data.get("objects") or []
    missing = data.get("missing_target_objects") or []
    camera_selection = data.get("camera_selection") or {}
    return {
        "exists": True,
        "targets": len(targets),
        "visible": len(visible),
        "missing": len(missing),
        "camera_visible": camera_selection.get("visible_count"),
        "camera_target": camera_selection.get("target_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Generation output root.")
    parser.add_argument("--vis-root", default=None, help="Optional visualization output root.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail-on-issues", action="store_true")
    parser.add_argument("--ok-only", action="store_true", help="Audit only generator-accepted samples.")
    args = parser.parse_args()

    root = Path(args.root)
    vis_root = Path(args.vis_root) if args.vis_root else None
    files = iter_run_files(root)
    if args.ok_only:
        files = [path for path in files if load_json(path).get("ok") is True]
    by_label = defaultdict(list)
    issue_counts = Counter()
    bbox_totals = Counter()
    diversity_totals = {
        "env_type": Counter(), "primary_task": Counter(), "target_room": Counter(),
        "source_room": Counter(), "target_category": Counter(), "target_model": Counter(),
        "target_object_id": Counter(), "support_category": Counter(), "position_bin_25cm": Counter(),
    }
    examples = []

    for path in files:
        run = load_json(path)
        te = run.get("task_environment") or {}
        label = infer_label(path) or "unknown"
        issues = check_run(path, run)
        bbox = summarize_bboxes(path, vis_root)
        if vis_root is not None and run.get("ok") is True:
            if not bbox or not bbox.get("exists"):
                issues.append("missing_bbox_file")
            elif bbox.get("targets", 0) > bbox.get("visible", 0):
                issues.append("bbox_target_not_visible")
        for issue in issues:
            issue_counts[issue] += 1
        diversity = run.get("diversity") or te.get("diversity") or {}
        for key in ("env_type", "primary_task", "target_room"):
            if diversity.get(key):
                diversity_totals[key][str(diversity[key])] += 1
        for category in diversity.get("target_categories") or []:
            diversity_totals["target_category"][str(category)] += 1
        target_models = diversity.get("target_models") or [
            {"category": item.get("category"), "model": item.get("model")}
            for item in te.get("added_objects") or []
            if item.get("category") and item.get("model")
        ]
        for item in target_models:
            if isinstance(item, dict) and item.get("category") and item.get("model"):
                key = f"{item['category']}::{item['model']}"
                diversity_totals["target_model"][key] += 1
        target_object_ids = diversity.get("target_object_ids") or [
            item.get("object_id")
            for item in (te.get("task") or {}).get("plan_objects") or []
            if item.get("object_id")
        ]
        for object_id in target_object_ids:
            diversity_totals["target_object_id"][str(object_id)] += 1
        for room in diversity.get("source_rooms") or []:
            diversity_totals["source_room"][str(room)] += 1
        for category in diversity.get("support_categories") or []:
            diversity_totals["support_category"][str(category)] += 1
        for position_bin in diversity.get("position_bins_25cm") or []:
            diversity_totals["position_bin_25cm"][str(position_bin)] += 1
        if bbox:
            if bbox["exists"]:
                bbox_totals["bbox_files"] += 1
                bbox_totals["bbox_targets"] += bbox.get("targets", 0)
                bbox_totals["bbox_visible"] += bbox.get("visible", 0)
                if bbox.get("visible", 0) > 0:
                    bbox_totals["runs_with_visible_bbox"] += 1
            else:
                bbox_totals["missing_bbox_files"] += 1
        by_label[label].append({"path": str(path), "ok": bool(run.get("ok")), "issues": issues, "bbox": bbox})
        if issues and len(examples) < 20:
            examples.append({"path": str(path), "issues": issues})

    fingerprints = defaultdict(list)
    for path in files:
        run = load_json(path)
        validation = run.get("validation") or {}
        fingerprint = validation.get("sample_fingerprint")
        if fingerprint:
            fingerprints[fingerprint].append(str(path))
    duplicate_groups = [paths for paths in fingerprints.values() if len(paths) > 1]
    if duplicate_groups:
        issue_counts["duplicate_fingerprint"] += sum(len(group) for group in duplicate_groups)

    label_summary = {}
    for label, items in sorted(by_label.items()):
        ok_count = sum(1 for item in items if item["ok"] and not item["issues"])
        label_summary[label] = {
            "total": len(items),
            "clean": ok_count,
            "with_issues": len(items) - ok_count,
        }

    report = {
        "root": str(root),
        "vis_root": str(vis_root) if vis_root else None,
        "num_runs": len(files),
        "labels": label_summary,
        "issues": dict(issue_counts),
        "bbox": dict(bbox_totals),
        "issue_examples": examples,
        "duplicate_fingerprint_groups": duplicate_groups[:20],
        "diversity": {key: dict(counter) for key, counter in diversity_totals.items()},
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if args.fail_on_issues and issue_counts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
