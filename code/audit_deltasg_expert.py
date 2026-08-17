"""Aggregate expert solver results and enforce dataset acceptance gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from audit_deltasg_outputs import check_run, infer_label, iter_run_files, load_json
from deltasg_expert import summarize_expert_results


DEFAULT_ENV_A_LABELS = {
    "envA_retrieval_delivery",
    "envA_open_close",
    "envA_appliance",
}


def rgb_artifact_error(path_value):
    if not path_value:
        return "missing"
    path = Path(path_value)
    if not path.is_file():
        return "missing"
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width < 64 or image.height < 64:
                return f"resolution_too_small:{image.width}x{image.height}"
            image.thumbnail((64, 64))
            extrema = ImageStat.Stat(image).extrema
            if max(high - low for low, high in extrema) < 2:
                return "blank_or_uniform"
    except (OSError, ValueError) as exc:
        return f"decode_failed:{exc!r}"
    return None


def segmentation_artifact_error(path_value):
    if not path_value:
        return "missing"
    path = Path(path_value)
    if not path.is_file():
        return "missing"
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            return f"expected_2d:{array.shape}"
        if array.shape[0] < 64 or array.shape[1] < 64:
            return f"resolution_too_small:{array.shape[1]}x{array.shape[0]}"
        if not np.issubdtype(array.dtype, np.integer):
            return f"non_integer_dtype:{array.dtype}"
        if not np.any(array):
            return "all_background"
    except (OSError, ValueError) as exc:
        return f"decode_failed:{exc!r}"
    return None


def action_artifact_error(path_value, expected_count):
    if not path_value:
        return "missing"
    path = Path(path_value)
    if not path.is_file():
        return "missing"
    try:
        actions = np.load(path, mmap_mode="r", allow_pickle=False)
        if actions.ndim != 2:
            return f"expected_2d:{actions.shape}"
        if len(actions) != expected_count:
            return f"count_mismatch:{len(actions)}!={expected_count}"
        if len(actions) == 0:
            return "empty"
        if not np.issubdtype(actions.dtype, np.number):
            return f"non_numeric_dtype:{actions.dtype}"
        if not np.all(np.isfinite(actions)):
            return "non_finite"
    except (OSError, ValueError) as exc:
        return f"decode_failed:{exc!r}"
    return None


def is_strict_vla_eligible(item, source_profile):
    backend = item.get("backend") or {}
    return (
        item.get("accepted") is True
        and backend.get("name") == "physical_control"
        and source_profile == "physical_control"
        and backend.get("generation_solvability_profile") == source_profile
        and backend.get("generation_profile_verified") is True
        and backend.get("physical_solubility_validation") is True
        and backend.get("low_level_vla_actions_eligible") is True
        and backend.get("assisted_interaction") is not True
        and backend.get("complete_action_trace") is True
    )


def is_physical_trajectory_eligible(item, source_profile):
    backend = item.get("backend") or {}
    return (
        item.get("accepted") is True
        and backend.get("name") == "physical_control"
        and source_profile == "physical_control"
        and backend.get("generation_solvability_profile") == source_profile
        and backend.get("generation_profile_verified") is True
        and backend.get("complete_action_trace") is True
        and backend.get("physical_trajectory_available") is True
        and int(backend.get("physical_action_count") or 0) > 0
        and int(backend.get("physical_nonzero_action_count") or 0) > 0
    )


def parse_names(value):
    return {item for item in value.replace(",", " ").split() if item}


def load_results(root: Path):
    results = []
    for path in sorted(root.rglob("expert_result.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["result_path"] = str(path)
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description="Audit DeltaSG expert solver outputs")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-accept-rate", type=float, default=0.0)
    parser.add_argument("--input-root", default=None)
    parser.add_argument(
        "--labels",
        default="envA_retrieval_delivery,envA_open_close,envA_appliance",
    )
    parser.add_argument("--require-all-inputs", action="store_true")
    parser.add_argument(
        "--require-backend",
        choices=("oracle_symbolic", "physical_control"),
        default=None,
    )
    parser.add_argument("--require-low-level-vla-actions", action="store_true")
    parser.add_argument("--require-physical-trajectory", action="store_true")
    args = parser.parse_args()

    results = load_results(Path(args.root))
    summary = summarize_expert_results(results)
    failures = Counter()
    qa_violations = []
    vla_mislabels = []
    artifact_violations = []
    accepted_coverage = {
        "scenes": Counter(),
        "task_families": Counter(),
        "tasks": Counter(),
        "target_categories": Counter(),
        "target_models": Counter(),
        "target_object_ids": Counter(),
    }
    for item in results:
        rejection = item.get("rejection") or {}
        if not item.get("accepted"):
            failures[str(rejection.get("stage") or ("runner_error" if item.get("error") else "unknown"))] += 1
        if item.get("qa_eligible") and not item.get("accepted"):
            qa_violations.append(item["result_path"])
        backend = item.get("backend") or {}
        if backend.get("name") == "oracle_symbolic" and backend.get("low_level_vla_actions_eligible") is not False:
            vla_mislabels.append(item["result_path"])
        if backend.get("assisted_interaction") is True and backend.get("low_level_vla_actions_eligible") is not False:
            vla_mislabels.append(item["result_path"])
        if not item.get("accepted"):
            continue
        errors = []
        backend_name = backend.get("name")
        if backend_name not in {"oracle_symbolic", "physical_control"}:
            errors.append("accepted result has missing/unknown backend")
        if args.require_backend and backend_name != args.require_backend:
            errors.append(
                f"backend {backend_name!r} != required {args.require_backend!r}"
            )
        accepted_coverage["scenes"][str(item.get("scene") or "unknown")] += 1
        accepted_coverage["task_families"][str(item.get("task_family") or "unknown")] += 1
        accepted_coverage["tasks"][str(item.get("task_name") or "unknown")] += 1
        input_path = Path(str(item.get("input") or ""))
        source_profile = None
        if input_path.is_file():
            source = load_json(input_path)
            te = source.get("task_environment") or {}
            source_profile = (te.get("generation") or {}).get("solvability_profile")
            task = te.get("task") or source.get("task") or {}
            diversity = source.get("diversity") or te.get("diversity") or {}
            target_records = [
                {
                    "object_id": object_id,
                    "category": None,
                    "model": None,
                }
                for object_id in diversity.get("target_object_ids") or []
            ]
            model_records = [
                {"object_id": None, "category": item.get("category"), "model": item.get("model")}
                for item in diversity.get("target_models") or []
                if isinstance(item, dict)
            ]
            target_records.extend(model_records)
            model_categories = {item.get("category") for item in model_records}
            target_records.extend(
                {"object_id": None, "category": category, "model": None}
                for category in diversity.get("target_categories") or []
                if category not in model_categories
            )
            if not target_records:
                target_records = task.get("plan_objects") or []
            for target in target_records:
                object_id = target.get("object_id") or target.get("object_name")
                category = target.get("category")
                model = target.get("model")
                if object_id:
                    accepted_coverage["target_object_ids"][str(object_id)] += 1
                if category:
                    accepted_coverage["target_categories"][str(category)] += 1
                if category and model:
                    accepted_coverage["target_models"][f"{category}::{model}"] += 1
        elif args.require_low_level_vla_actions:
            errors.append("VLA source input is missing")
        persisted_profile = backend.get("generation_solvability_profile")
        if source_profile is not None and persisted_profile != source_profile:
            errors.append("expert generation profile does not match source input")
        plan_steps = (item.get("compiled_plan") or {}).get("steps") or []
        result_steps = item.get("steps") or []
        native_targets = {
            step.get("target_object")
            for step in plan_steps
            if step.get("primitive") in {"OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}
        }
        if native_targets:
            replayed_targets = {
                state.get("object_id")
                for state in item.get("replayed_initial_states") or []
            }
            if not native_targets <= replayed_targets:
                errors.append("native task initial state was not replayed")
        if len(result_steps) != len(plan_steps):
            errors.append(f"step count {len(result_steps)} != compiled {len(plan_steps)}")
        for step in result_steps:
            if step.get("visibility_errors"):
                errors.append(f"step {((step.get('step') or {}).get('step_id'))} visibility failed")
            if step.get("postcondition_ok") is not True:
                errors.append(f"step {((step.get('step') or {}).get('step_id'))} postcondition missing/failed")
            primitive = ((step.get("step") or {}).get("primitive"))
            if primitive in {
                "GRASP", "PLACE_ON_TOP", "PLACE_INSIDE", "OPEN", "CLOSE",
                "TOGGLE_ON", "TOGGLE_OFF",
            }:
                height = step.get("manipulation_height") or {}
                if height.get("eligible") is not True:
                    errors.append(
                        f"step {((step.get('step') or {}).get('step_id'))} manipulation height missing/failed"
                    )
            action_path = step.get("actions_path")
            if not action_path or not Path(action_path).is_file():
                errors.append(f"step {((step.get('step') or {}).get('step_id'))} actions missing")
            if (
                args.require_low_level_vla_actions
                or args.require_physical_trajectory
                or backend.get("low_level_vla_actions_eligible") is True
            ):
                action_error = action_artifact_error(
                    action_path, int(step.get("actions_executed") or 0)
                )
                if action_error:
                    errors.append(
                        f"step {((step.get('step') or {}).get('step_id'))} "
                        f"actions invalid: {action_error}"
                    )
        events = item.get("observation_events") or []
        if not events:
            errors.append("no observation events")
        for event in events:
            robot_paths = ((event.get("robot_primary") or {}).get("paths") or {})
            robot_rgb_error = rgb_artifact_error(robot_paths.get("rgb"))
            if robot_rgb_error:
                errors.append(
                    f"event {event.get('event_id')} robot RGB invalid: {robot_rgb_error}"
                )
            for modality in ("seg_semantic", "seg_instance"):
                segmentation_error = segmentation_artifact_error(robot_paths.get(modality))
                if segmentation_error:
                    errors.append(
                        f"event {event.get('event_id')} robot {modality} invalid: "
                        f"{segmentation_error}"
                    )
            for view in event.get("global_cameras") or []:
                paths = view.get("paths") or {}
                global_rgb_error = rgb_artifact_error(paths.get("rgb"))
                if global_rgb_error:
                    errors.append(
                        f"event {event.get('event_id')} global RGB invalid: {global_rgb_error}"
                    )
        if backend.get("name") == "physical_control":
            if item.get("robot") not in {"R1", "Tiago"}:
                errors.append("physical_control used an unsupported robot")
            expected_vla_eligible = (
                backend.get("generation_profile_verified") is True
                and persisted_profile == "physical_control"
                and backend.get("complete_action_trace") is True
                and backend.get("assisted_interaction") is not True
            )
            if backend.get("physical_solubility_validation") is not expected_vla_eligible:
                errors.append("physical solvability validation label is inconsistent")
            if backend.get("low_level_vla_actions_eligible") is not expected_vla_eligible:
                errors.append("physical_control actions are not marked VLA eligible")
        strict_vla_eligible = is_strict_vla_eligible(item, source_profile)
        physical_trajectory_eligible = is_physical_trajectory_eligible(
            item, source_profile
        )
        if args.require_low_level_vla_actions and not strict_vla_eligible:
            errors.append("result is not strict low-level VLA eligible")
        if args.require_physical_trajectory and not physical_trajectory_eligible:
            errors.append("result has no complete physical trajectory")
        if errors:
            artifact_violations.append({"path": item["result_path"], "errors": errors})
    summary.update(
        {
            "failure_stages": dict(sorted(failures.items())),
            "qa_gate_violations": qa_violations,
            "vla_backend_label_violations": vla_mislabels,
            "artifact_violations": artifact_violations,
            "accepted_coverage": {
                key: dict(sorted(counter.items()))
                for key, counter in accepted_coverage.items()
            },
            "required_backend": args.require_backend,
            "required_low_level_vla_actions": args.require_low_level_vla_actions,
            "required_physical_trajectory": args.require_physical_trajectory,
        }
    )
    input_coverage = None
    if args.input_root:
        input_root = Path(args.input_root)
        labels = parse_names(args.labels)
        expected = {}
        for path in iter_run_files(input_root):
            if infer_label(path) not in labels:
                continue
            source = load_json(path)
            if source.get("ok") is True and not check_run(path, source):
                expected[str(path.resolve())] = str(path)
        by_input = {
            str(Path(str(item.get("input"))).resolve()): item
            for item in results
            if item.get("input")
        }
        missing = [expected[key] for key in sorted(set(expected) - set(by_input))]
        rejected = [
            expected[key]
            for key in sorted(set(expected) & set(by_input))
            if by_input[key].get("accepted") is not True
        ]
        input_coverage = {
            "input_root": str(input_root),
            "labels": sorted(labels),
            "expected_clean_inputs": len(expected),
            "results": len(set(expected) & set(by_input)),
            "accepted": sum(
                by_input[key].get("accepted") is True
                for key in set(expected) & set(by_input)
            ),
            "missing_results": missing,
            "rejected_results": rejected,
            "complete": not missing and not rejected and bool(expected),
        }
        summary["input_coverage"] = input_coverage
    summary["ok"] = (
        bool(results)
        and summary["accept_rate"] >= args.min_accept_rate
        and not qa_violations
        and not vla_mislabels
        and not artifact_violations
        and (not args.require_all_inputs or bool(input_coverage and input_coverage["complete"]))
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
