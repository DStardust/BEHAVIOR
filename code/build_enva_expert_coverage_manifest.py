#!/usr/bin/env python3
"""Build deterministic Env-A scene/task/target coverage jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deltasg_expert import (
    SUPPORTED_APPLIANCE_TASKS,
    SUPPORTED_OPEN_CLOSE_TASKS,
    SUPPORTED_RETRIEVAL_DELIVERY_TASKS,
)


RETRIEVAL_TASK_ASSETS = {
    "retrieve_medicine": ("bottle_of_medicine",),
    "retrieve_key": ("keys", "key_chain"),
    "retrieve_phone": ("cell_phone",),
    "retrieve_book": ("paperback_book",),
    "retrieve_drink": ("bottle_of_water", "water_bottle"),
    "retrieve_food": ("canned_food",),
    "deliver_medicine": ("bottle_of_medicine",),
    "deliver_food": ("canned_food",),
    "deliver_drink": ("bottle_of_water", "water_bottle"),
}

assert set(RETRIEVAL_TASK_ASSETS) == set(SUPPORTED_RETRIEVAL_DELIVERY_TASKS)


def _native_candidate_order(scene_record, object_id):
    """Try fixtures in roomy work areas before confined circulation spaces."""
    target = (scene_record.get("eligible_target_objects") or {}).get(object_id) or {}
    rooms = [str(room).lower() for room in target.get("rooms") or []]
    if any(
        token in room
        for room in rooms
        for token in ("living_room", "dining_room", "kitchen")
    ):
        room_rank = 0
    elif any(
        token in room
        for room in rooms
        for token in ("bedroom", "office")
    ):
        room_rank = 1
    elif any(
        token in room
        for room in rooms
        for token in ("storage", "bathroom", "corridor", "closet", "pantry")
    ):
        room_rank = 3
    else:
        room_rank = 2
    return room_rank, object_id


def _rotate_native_candidates_within_room_rank(scene_record, object_ids, offset):
    ranked = []
    room_ranks = sorted({_native_candidate_order(scene_record, object_id)[0] for object_id in object_ids})
    for room_rank in room_ranks:
        group = [
            object_id
            for object_id in object_ids
            if _native_candidate_order(scene_record, object_id)[0] == room_rank
        ]
        rotation = offset % len(group)
        ranked.extend(group[rotation:] + group[:rotation])
    return ranked


def _job(
    scene,
    task,
    label,
    category=None,
    model=None,
    object_id=None,
    object_candidates=None,
    preflight_ineligibility=None,
):
    identity = "::".join(filter(None, (scene, task, category, model, object_id or "task_target")))
    return {
        "job_id": hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12],
        "scene": scene,
        "label": label,
        "task": task,
        "target_asset_category": category,
        "target_asset_model": model,
        "target_native_object_id": object_id,
        "target_native_candidates": list(object_candidates or []),
        "preflight_ineligibility": preflight_ineligibility,
    }


def build_jobs(scenes, inventory, native, scope, retrieval_variants=1):
    models_by_category = (inventory.get("groups") or {}).get("retrieval_delivery") or {}
    native_scenes = native.get("scenes") or {}
    missing_native_scenes = sorted(set(scenes) - set(native_scenes))
    if missing_native_scenes:
        raise ValueError(
            f"native eligibility inventory is missing scenes: {missing_native_scenes}"
        )
    supported_native_tasks = set(SUPPORTED_OPEN_CLOSE_TASKS) | set(SUPPORTED_APPLIANCE_TASKS)
    unknown_native_tasks = sorted({
        task
        for scene in scenes
        for task in ((native_scenes.get(scene) or {}).get("eligible_tasks") or {})
        if task not in supported_native_tasks
    })
    if unknown_native_tasks:
        raise ValueError(
            f"native eligibility inventory has unsupported tasks: {unknown_native_tasks}"
        )
    required_categories = {
        category for categories in RETRIEVAL_TASK_ASSETS.values() for category in categories
    }
    missing_categories = sorted(
        category for category in required_categories if not models_by_category.get(category)
    )
    if missing_categories:
        raise ValueError(
            f"retrieval asset inventory has no models for categories: {missing_categories}"
        )
    jobs = []
    retrieval_items = list(RETRIEVAL_TASK_ASSETS.items())
    for scene_index, scene in enumerate(scenes):
        scene_retrieval_items = (
            [retrieval_items[scene_index % len(retrieval_items)]]
            if scope == "smoke"
            else retrieval_items
        )
        for task_index, (task, categories) in enumerate(scene_retrieval_items):
            pairs = [
                (category, model)
                for category in categories
                for model in models_by_category.get(category, [])
            ]
            if not pairs:
                continue
            selected = pairs if scope == "full" else [pairs[(scene_index + task_index) % len(pairs)]]
            for category, model in selected:
                jobs.append(_job(scene, task, "envA_retrieval_delivery", category, model))

        scene_record = native_scenes.get(scene) or {}
        eligible_tasks = scene_record.get("eligible_tasks") or {}
        if scope != "smoke":
            for task in sorted(supported_native_tasks):
                label = (
                    "envA_open_close"
                    if task.startswith(("open_", "close_"))
                    else "envA_appliance"
                )
                object_ids = sorted(
                    set(eligible_tasks.get(task) or []),
                    key=lambda object_id: _native_candidate_order(scene_record, object_id),
                )
                if scope == "full" and object_ids:
                    jobs.extend(
                        _job(scene, task, label, object_id=object_id)
                        for object_id in object_ids
                    )
                elif object_ids:
                    jobs.append(_job(scene, task, label, object_candidates=object_ids))
                else:
                    jobs.append(_job(
                        scene,
                        task,
                        label,
                        preflight_ineligibility={
                            "stage": "scene_task_inventory",
                            "reason": (
                                "versioned scene graph has no category-, state-, and "
                                "manipulation-height-eligible native target"
                            ),
                        },
                    ))

        if scope == "smoke":
            emitted_labels = set()
            for label, prefix in (
                ("envA_open_close", ("open_", "close_")),
                ("envA_appliance", ("turn_on_", "turn_off_")),
            ):
                candidates = [task for task in sorted(eligible_tasks) if task.startswith(prefix)]
                if candidates:
                    task = candidates[scene_index % len(candidates)]
                    object_ids = sorted(
                        set(eligible_tasks[task]),
                        key=lambda object_id: _native_candidate_order(scene_record, object_id),
                    )
                    if object_ids:
                        rotated = _rotate_native_candidates_within_room_rank(
                            scene_record, object_ids, scene_index
                        )
                        jobs.append(_job(scene, task, label, object_candidates=rotated))
                        emitted_labels.add(label)
            missing_labels = {"envA_open_close", "envA_appliance"} - emitted_labels
            if missing_labels:
                raise ValueError(
                    f"scene {scene!r} has no structural candidates for {sorted(missing_labels)}"
                )

    deduplicated = {job["job_id"]: job for job in jobs}
    expanded = []
    for job in deduplicated.values():
        variants = retrieval_variants if job["label"] == "envA_retrieval_delivery" else 1
        for variant_index in range(variants):
            variant = dict(job)
            variant["variant_index"] = variant_index
            if variants > 1:
                variant["job_id"] = hashlib.sha1(
                    f"{job['job_id']}::placement_variant::{variant_index}".encode("utf-8")
                ).hexdigest()[:12]
            expanded.append(variant)
    label_order = {
        "envA_retrieval_delivery": 0,
        "envA_open_close": 1,
        "envA_appliance": 2,
    }
    return sorted(
        expanded,
        key=lambda job: (
            scenes.index(job["scene"]),
            label_order[job["label"]],
            job["task"],
            job["job_id"],
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-file", required=True)
    parser.add_argument(
        "--asset-inventory",
        default=str(Path(__file__).resolve().parent / "configs" / "env_a_asset_inventory.json"),
    )
    parser.add_argument(
        "--native-eligibility",
        default=str(Path(__file__).resolve().parent / "configs" / "env_a_native_eligibility.json"),
    )
    parser.add_argument("--scope", choices=("smoke", "tasks", "full"), default="smoke")
    parser.add_argument("--retrieval-variants", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.retrieval_variants < 1:
        parser.error("--retrieval-variants must be at least 1")

    scenes = [line.strip() for line in Path(args.scenes_file).read_text().splitlines() if line.strip()]
    inventory = json.loads(Path(args.asset_inventory).read_text())
    native = json.loads(Path(args.native_eligibility).read_text())
    jobs = build_jobs(
        scenes, inventory, native, args.scope, retrieval_variants=args.retrieval_variants
    )
    payload = {
        "schema_version": "enva_expert_coverage_manifest.v1",
        "scope": args.scope,
        "scenes": scenes,
        "num_jobs": len(jobs),
        "retrieval_variants": args.retrieval_variants,
        "jobs": jobs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("scope", "num_jobs")}, indent=2))


if __name__ == "__main__":
    main()
