#!/usr/bin/env python3
"""Audit the explicit scene, task, and task-asset coverage contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_deltasg_outputs import EXPECTED_LABELS, check_run, infer_label, iter_run_files, load_json
from deltasg_expert import (
    SUPPORTED_APPLIANCE_TASKS,
    SUPPORTED_OPEN_CLOSE_TASKS,
    SUPPORTED_RETRIEVAL_DELIVERY_TASKS,
)


KNOWN_TASKS = {
    "retrieval_delivery": set(SUPPORTED_RETRIEVAL_DELIVERY_TASKS),
    "open_close": set(SUPPORTED_OPEN_CLOSE_TASKS),
    "appliance": set(SUPPORTED_APPLIANCE_TASKS),
}

LABEL_TASK_GROUP = {
    "envA_retrieval_delivery": "retrieval_delivery",
    "envA_open_close": "open_close",
    "envA_appliance": "appliance",
    "envC_retrieval_delivery": "retrieval_delivery",
    "envC_open_close": "open_close",
    "envC_appliance": "appliance",
}

LABEL_ASSET_GROUPS = {
    "envA_retrieval_delivery": ("retrieval_delivery",),
    "envC_retrieval_delivery": ("retrieval_delivery",),
    "envB_fire": ("fire_common",),
    "envC_fire_disambiguation": ("fire_common", "fire_env_c"),
}
FORMAL_LABELS = tuple(label for label in EXPECTED_LABELS if label != "envC_all")
NATIVE_TARGET_RULES = {
    "envA_open_close": ("Open", ("door", "window", "fridge", "refrigerator", "cabinet")),
    "envC_open_close": ("Open", ("door", "window", "fridge", "refrigerator", "cabinet")),
    "envA_appliance": ("ToggledOn", ("electric_switch", "light", "lamp", "tv", "television", "stove")),
    "envC_appliance": ("ToggledOn", ("electric_switch", "light", "lamp", "tv", "television", "stove")),
    "envB_fire": ("OnFire", ()),
    "envC_fire_disambiguation": ("OnFire", ()),
}
NATIVE_TASK_RULES = {
    "open_door": ("Open", ("door",)),
    "close_door": ("Open", ("door",)),
    "open_window": ("Open", ("window",)),
    "close_window": ("Open", ("window",)),
    "open_fridge": ("Open", ("fridge", "refrigerator")),
    "close_fridge": ("Open", ("fridge", "refrigerator")),
    "open_cabinet": ("Open", ("cabinet",)),
    "close_cabinet": ("Open", ("cabinet",)),
    "turn_on_light": ("ToggledOn", ("electric_switch", "light", "lamp")),
    "turn_off_light": ("ToggledOn", ("electric_switch", "light", "lamp")),
    "turn_on_tv": ("ToggledOn", ("tv", "television")),
    "turn_off_tv": ("ToggledOn", ("tv", "television")),
    "turn_on_stove": ("ToggledOn", ("stove",)),
    "turn_off_stove": ("ToggledOn", ("stove",)),
}
DEFAULT_MIN_MANIPULATION_HEIGHT = 0.10
DEFAULT_MAX_MANIPULATION_HEIGHT = 1.55


def parse_list(value: str):
    return [item for item in value.replace(",", " ").split() if item]


def scene_for_path(root: Path, path: Path):
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        return None
    return relative.parts[1]


def extract_dimensions(run: dict):
    te = run.get("task_environment") or {}
    task = te.get("task") or {}
    diversity = run.get("diversity") or te.get("diversity") or {}
    target_models = diversity.get("target_models") or [
        {"category": item.get("category"), "model": item.get("model")}
        for item in te.get("added_objects") or []
        if item.get("category") and item.get("model")
    ]
    model_keys = {
        f"{item['category']}::{item['model']}"
        for item in target_models
        if isinstance(item, dict) and item.get("category") and item.get("model")
    }
    object_ids = set(diversity.get("target_object_ids") or [])
    if not object_ids:
        object_ids = {
            item.get("object_id")
            for item in task.get("plan_objects") or []
            if item.get("object_id")
        }
    categories = set(diversity.get("target_categories") or [])
    if not categories:
        categories = {
            item.get("category")
            for item in [*(te.get("added_objects") or []), *(task.get("plan_objects") or [])]
            if item.get("category")
        }
    return {
        "task": task.get("primary_behavior_task"),
        "categories": categories,
        "models": model_keys,
        "object_ids": object_ids,
    }


def expected_models(label: str, inventory: dict):
    result = set()
    groups = inventory.get("groups") or {}
    for group in LABEL_ASSET_GROUPS.get(label, ()):
        for category, models in (groups.get(group) or {}).items():
            result.update(f"{category}::{model}" for model in models)
    return result


def _native_height_eligible(node, graph, min_height, max_height):
    bbox = node.get("bbox") or {}
    aabb_min = bbox.get("min")
    aabb_max = bbox.get("max")
    if not aabb_min or not aabb_max:
        return False
    rooms = set(node.get("rooms") or [])
    floor_surfaces = []
    for candidate in graph.get("nodes") or []:
        if candidate.get("type") != "object":
            continue
        category = str(candidate.get("category") or "").lower()
        if category not in {"floor", "floors"}:
            continue
        if rooms and not rooms.intersection(candidate.get("rooms") or []):
            continue
        surface = ((candidate.get("bbox") or {}).get("max") or [None, None, None])[2]
        if surface is not None:
            floor_surfaces.append(float(surface))
    lower_z = float(aabb_min[2])
    below = [height for height in floor_surfaces if height <= lower_z + 0.10]
    floor_height = max(below) if below else (
        min(floor_surfaces, key=lambda height: abs(height - lower_z))
        if floor_surfaces else 0.0
    )
    operation_height = (float(aabb_min[2]) + float(aabb_max[2])) / 2.0 - floor_height
    return float(min_height) <= operation_height <= float(max_height)


def eligible_native_targets(
    label: str,
    run: dict,
    min_height=DEFAULT_MIN_MANIPULATION_HEIGHT,
    max_height=DEFAULT_MAX_MANIPULATION_HEIGHT,
):
    rule = NATIVE_TARGET_RULES.get(label)
    if not rule:
        return set()
    required_state, category_tokens = rule
    graph = run.get("before_graph") or (run.get("debug") or {}).get("before_graph") or {}
    result = set()
    for node in graph.get("nodes") or []:
        if node.get("type") != "object":
            continue
        category = str(node.get("category") or "").lower()
        if category_tokens and not any(token in category for token in category_tokens):
            continue
        if required_state not in set(node.get("available_states") or []):
            continue
        if not node.get("rooms"):
            continue
        if not _native_height_eligible(node, graph, min_height, max_height):
            continue
        if node.get("id"):
            result.add(node["id"])
    return result


def eligible_native_task_pairs(
    label: str,
    run: dict,
    min_height=DEFAULT_MIN_MANIPULATION_HEIGHT,
    max_height=DEFAULT_MAX_MANIPULATION_HEIGHT,
):
    group = LABEL_TASK_GROUP.get(label)
    if group not in {"open_close", "appliance"}:
        return set()
    graph = run.get("before_graph") or (run.get("debug") or {}).get("before_graph") or {}
    result = set()
    for task_name in KNOWN_TASKS[group]:
        required_state, category_tokens = NATIVE_TASK_RULES[task_name]
        for node in graph.get("nodes") or []:
            category = str(node.get("category") or "").lower()
            if node.get("type") != "object":
                continue
            if not any(token in category for token in category_tokens):
                continue
            if required_state not in set(node.get("available_states") or []):
                continue
            if not node.get("rooms"):
                continue
            if not _native_height_eligible(node, graph, min_height, max_height):
                continue
            if node.get("id"):
                result.add((task_name, node["id"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenes-file", default=None)
    parser.add_argument("--labels", default="all")
    parser.add_argument("--min-clean-per-cell", type=int, default=1)
    parser.add_argument("--require-all-known-tasks", action="store_true")
    parser.add_argument("--asset-inventory", default=None)
    parser.add_argument("--require-all-asset-models", action="store_true")
    parser.add_argument("--require-all-native-targets", action="store_true")
    parser.add_argument("--min-manipulation-height", type=float, default=DEFAULT_MIN_MANIPULATION_HEIGHT)
    parser.add_argument("--max-manipulation-height", type=float, default=DEFAULT_MAX_MANIPULATION_HEIGHT)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail-on-gaps", action="store_true")
    args = parser.parse_args()
    if args.min_manipulation_height < 0 or args.max_manipulation_height <= args.min_manipulation_height:
        parser.error("manipulation height bounds must satisfy 0 <= min < max")

    root = Path(args.root)
    scenes_file = Path(args.scenes_file) if args.scenes_file else root / "scenes.txt"
    scenes = [
        line.strip()
        for line in scenes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = sorted(FORMAL_LABELS) if args.labels == "all" else parse_list(args.labels)
    unknown_labels = sorted(set(labels) - set(EXPECTED_LABELS))
    if unknown_labels:
        raise SystemExit(f"Unknown labels: {unknown_labels}")

    cell_counts = Counter()
    observed = defaultdict(lambda: {
        "tasks": Counter(), "categories": Counter(), "models": Counter(), "object_ids": Counter(),
    })
    expected_native = defaultdict(set)
    observed_native = defaultdict(set)
    expected_native_pairs = defaultdict(set)
    observed_native_pairs = defaultdict(set)
    rejected_files = []
    for path in iter_run_files(root):
        label = infer_label(path)
        if label not in labels:
            continue
        scene = scene_for_path(root, path)
        if scene not in scenes:
            continue
        run = load_json(path)
        issues = check_run(path, run)
        if not run.get("ok") or issues:
            rejected_files.append({"path": str(path), "issues": issues})
            continue
        cell_counts[(label, scene)] += 1
        expected_native[label].update(
            f"{scene}::{object_id}"
            for object_id in eligible_native_targets(
                label, run, args.min_manipulation_height, args.max_manipulation_height
            )
        )
        expected_native_pairs[label].update(
            f"{scene}::{task_name}::{object_id}"
            for task_name, object_id in eligible_native_task_pairs(
                label, run, args.min_manipulation_height, args.max_manipulation_height
            )
        )
        dims = extract_dimensions(run)
        observed_native[label].update(
            f"{scene}::{object_id}"
            for object_id in dims["object_ids"]
        )
        if dims["task"]:
            observed_native_pairs[label].update(
                f"{scene}::{dims['task']}::{object_id}"
                for object_id in dims["object_ids"]
            )
        for key in ("task",):
            if dims[key]:
                observed[label][f"{key}s"][dims[key]] += 1
        for key in ("categories", "models", "object_ids"):
            observed[label][key].update(dims[key])

    missing_cells = []
    for label in labels:
        for scene in scenes:
            count = cell_counts[(label, scene)]
            if count < args.min_clean_per_cell:
                missing_cells.append({
                    "label": label,
                    "scene": scene,
                    "clean": count,
                    "required": args.min_clean_per_cell,
                })

    inventory = load_json(Path(args.asset_inventory)) if args.asset_inventory else {}
    missing_tasks = {}
    missing_models = {}
    missing_native_targets = {}
    label_summary = {}
    for label in labels:
        task_group = LABEL_TASK_GROUP.get(label)
        expected_task_set = KNOWN_TASKS.get(task_group, set()) if args.require_all_known_tasks else set()
        observed_tasks = set(observed[label]["tasks"])
        task_gaps = sorted(expected_task_set - observed_tasks)
        if task_gaps:
            missing_tasks[label] = task_gaps

        expected_model_set = (
            expected_models(label, inventory)
            if args.require_all_asset_models
            else set()
        )
        observed_models = set(observed[label]["models"])
        model_gaps = sorted(expected_model_set - observed_models)
        if model_gaps:
            missing_models[label] = model_gaps
        native_gaps = (
            sorted(
                (expected_native[label] - observed_native[label])
                | (expected_native_pairs[label] - observed_native_pairs[label])
            )
            if args.require_all_native_targets
            else []
        )
        if native_gaps:
            missing_native_targets[label] = native_gaps

        label_summary[label] = {
            "clean": sum(cell_counts[(label, scene)] for scene in scenes),
            "covered_scenes": sum(cell_counts[(label, scene)] > 0 for scene in scenes),
            "required_scenes": len(scenes),
            "unique_tasks": len(observed_tasks),
            "unique_target_categories": len(observed[label]["categories"]),
            "unique_target_models": len(observed_models),
            "unique_target_object_ids": len(observed[label]["object_ids"]),
            "eligible_native_target_ids": len(expected_native[label]),
            "covered_native_target_ids": len(expected_native[label] & observed_native[label]),
            "eligible_native_task_target_pairs": len(expected_native_pairs[label]),
            "covered_native_task_target_pairs": len(
                expected_native_pairs[label] & observed_native_pairs[label]
            ),
            "task_counts": dict(observed[label]["tasks"]),
            "target_category_counts": dict(observed[label]["categories"]),
            "target_model_counts": dict(observed[label]["models"]),
        }

    gaps = {
        "missing_cells": missing_cells,
        "missing_tasks": missing_tasks,
        "missing_models": missing_models,
        "missing_native_targets": missing_native_targets,
    }
    report = {
        "schema_version": "deltasg_coverage_audit.v1",
        "root": str(root),
        "scenes": scenes,
        "labels": labels,
        "min_clean_per_cell": args.min_clean_per_cell,
        "matrix_cells": len(scenes) * len(labels),
        "covered_cells": len(scenes) * len(labels) - len(missing_cells),
        "label_summary": label_summary,
        "gaps": gaps,
        "num_rejected_files": len(rejected_files),
        "rejected_examples": rejected_files[:20],
        "ok": (
            not missing_cells
            and not missing_tasks
            and not missing_models
            and not missing_native_targets
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.fail_on_gaps and not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
