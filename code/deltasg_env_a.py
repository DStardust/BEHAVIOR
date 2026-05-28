"""
Generate DeltaSG Env-A task environments from the layer-2 scene graph assets.

Env-A is the basic task environment from intro.md: it edits an initial scene by
adding task-relevant objects, records the DeltaSG edit, and emits a compact task
environment JSON that can later be physically instantiated and validated.

This module is intentionally offline and standard-library only. It consumes the
task asset database produced by code/build_task_asset_db.py, including its
scene_overlay, and does not launch OmniGibson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TASK_OBJECTS = 5
DEFAULT_ADDITIONAL_OBJECTS = 12
DEFAULT_ENVS = 1
MAX_RELATED_PER_TARGET = 5

ROOM_KEYWORDS = {
    "kitchen": {
        "bake",
        "bowl",
        "cook",
        "dish",
        "food",
        "fruit",
        "kitchen",
        "make",
        "meal",
        "microwave",
        "mix",
        "nacho",
        "oven",
        "pan",
        "plate",
        "pot",
        "prepare",
        "sauce",
        "vegetable",
        "wash",
    },
    "bathroom": {"bath", "clean", "disinfect", "mirror", "shower", "toilet", "wash"},
    "bedroom": {"bed", "cloth", "fold", "laundry", "pillow", "shoe", "sleep"},
    "living_room": {"book", "decorate", "lamp", "table", "tv"},
    "entryway": {"bag", "backpack", "carry", "shoe", "trash"},
}

STRUCTURAL_CATEGORIES = {"agent", "ceilings", "floors", "walls"}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def generate_env_a(
    task_asset_database_path,
    output_path=None,
    num_envs=DEFAULT_ENVS,
    seed=0,
    task_objects=DEFAULT_TASK_OBJECTS,
    additional_objects=DEFAULT_ADDITIONAL_OBJECTS,
    scene_model=None,
):
    rng = random.Random(seed)
    db = load_json(task_asset_database_path)
    overlay = db.get("scene_overlay")
    if not overlay:
        raise ValueError("task_asset_database.json must include scene_overlay for Env-A generation")

    scene_model = scene_model or overlay.get("scene_model")
    records = _usable_records(db.get("records", []))
    task_index = _records_by_task(records)
    task_pool = _eligible_tasks(task_index, min_records=task_objects)
    if not task_pool:
        raise ValueError(f"No tasks have at least {task_objects} usable object records")
    category_records = _records_by_category(records)
    scene = _scene_context(overlay)

    envs = []
    used_signatures = set()
    for env_idx in range(num_envs):
        primary_task = _sample_primary_task(task_pool, task_index, rng)
        selected = _sample_records_for_task(task_index[primary_task], rng, task_objects)
        related = _sample_related_records_for_task(
            primary_task=primary_task,
            selected_records=selected,
            task_index=task_index,
            rng=rng,
            count=additional_objects,
        )
        all_records = _dedupe_records(selected + related)

        signature = (primary_task, *tuple(sorted(record["synset"] for record in all_records)))
        retry = 0
        while signature in used_signatures and retry < 20:
            primary_task = _sample_primary_task(task_pool, task_index, rng)
            selected = _sample_records_for_task(task_index[primary_task], rng, task_objects)
            related = _sample_related_records_for_task(
                primary_task=primary_task,
                selected_records=selected,
                task_index=task_index,
                rng=rng,
                count=additional_objects,
            )
            all_records = _dedupe_records(selected + related)
            signature = (primary_task, *tuple(sorted(record["synset"] for record in all_records)))
            retry += 1
        used_signatures.add(signature)

        env = _build_single_env_a(
            env_idx=env_idx,
            scene_model=scene_model,
            primary_task=primary_task,
            selected_records=selected,
            additional_records=related,
            all_records=all_records,
            category_records=category_records,
            scene=scene,
            rng=rng,
            seed=seed,
        )
        envs.append(env)

    result = {
        "schema_version": "deltasg_env_a.v1",
        "source": {
            "task_asset_database": str(task_asset_database_path),
            "scene_model": scene_model,
            "generator": "code/deltasg_env_a.py",
        },
        "generation": {
            "seed": seed,
            "num_envs": num_envs,
            "task_objects_per_env": task_objects,
            "additional_object_budget": additional_objects,
        },
        "envs": envs,
    }

    if output_path:
        write_json(output_path, result)
    return result


def _usable_records(records):
    usable = []
    for record in records:
        if not record.get("direct_categories"):
            continue
        tasks = _valid_tasks(record)
        if not tasks:
            continue
        if record.get("object_type") in {"liquid", "microPhysicalSubstance", "visualSubstance"}:
            continue
        record = dict(record)
        record["tasks"] = tasks
        usable.append(record)
    return usable


def _records_by_task(records):
    index = defaultdict(list)
    for record in records:
        for task in record.get("tasks", []):
            index[task].append(record)
    return index


def _valid_tasks(record):
    return [task for task in record.get("tasks", []) if task and task != "..."]


def _eligible_tasks(task_index, min_records):
    tasks = []
    for task, records in task_index.items():
        unique = {record["synset"] for record in records}
        if len(unique) >= min_records:
            tasks.append(task)
    return sorted(tasks)


def _records_by_category(records):
    index = defaultdict(list)
    for record in records:
        for category in record.get("direct_categories", []):
            index[category].append(record)
    return index


def _scene_context(overlay):
    object_instances = overlay.get("object_instances", [])
    rooms = overlay.get("rooms", [])
    room_centers = overlay.get("navigation", {}).get("room_centers", {})
    receptacles = []
    room_objects = defaultdict(list)

    for obj in object_instances:
        category = obj.get("category")
        obj_rooms = [room for room in obj.get("rooms", []) if room]
        if category in STRUCTURAL_CATEGORIES or not obj_rooms:
            continue
        for room in obj_rooms:
            room_objects[room].append(obj)
        receptacle = (obj.get("semantic") or {}).get("receptacle") or {}
        if receptacle.get("can_support"):
            receptacles.append(obj)

    return {
        "rooms": rooms,
        "room_centers": room_centers,
        "room_objects": dict(room_objects),
        "receptacles": receptacles,
        "navigation": overlay.get("navigation", {}),
        "category_counts": overlay.get("category_counts", {}),
    }


def _sample_primary_task(task_pool, task_index, rng):
    weights = []
    for task in task_pool:
        records = task_index[task]
        num_sampled = sum(sum(len(v) for v in record.get("sampled_placements", {}).values()) for record in records)
        num_interactive = sum(1 for record in records if _interaction_kind(record) != "none")
        weights.append(max(1, len(records) + min(num_sampled, 20) + num_interactive))
    return rng.choices(task_pool, weights=weights, k=1)[0]


def _sample_records_for_task(records, rng, count):
    dedup = _dedupe_records(records)
    scored = [(_record_sampling_score(record), record) for record in dedup]
    selected = []
    while scored and len(selected) < count:
        weights = [max(score, 1) for score, _ in scored]
        record = rng.choices([record for _, record in scored], weights=weights, k=1)[0]
        selected.append(record)
        scored = [(score, item) for score, item in scored if item["synset"] != record["synset"]]
    return selected


def _sample_related_records_for_task(primary_task, selected_records, task_index, rng, count):
    selected_ids = {record["synset"] for record in selected_records}
    same_task_records = [
        record
        for record in _dedupe_records(task_index.get(primary_task, []))
        if record["synset"] not in selected_ids
    ]
    same_task_records.sort(key=lambda record: (-_record_sampling_score(record), record["synset"]))

    related = same_task_records[:count]
    if len(related) >= count:
        return related

    candidates = {}
    for target in selected_records:
        for task in target.get("tasks", []):
            for record in task_index.get(task, []):
                if record["synset"] == target["synset"]:
                    continue
                item = candidates.setdefault(record["synset"], {"record": record, "score": 0, "tasks": set()})
                item["score"] += 1
                item["tasks"].add(task)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["score"],
            -len(item["record"].get("direct_categories", [])),
            item["record"]["synset"],
        ),
    )

    selected_or_related = selected_ids | {record["synset"] for record in related}
    for item in ranked:
        if item["record"]["synset"] in selected_or_related:
            continue
        related.append(item["record"])
        if len(related) >= count:
            break

    return related[:count]


def _record_sampling_score(record):
    score = 0
    score += min(len(record.get("tasks", [])), 10)
    score += min(sum(len(v) for v in record.get("sampled_placements", {}).values()), 10)
    interaction = _interaction_kind(record)
    if interaction in {"manipulable", "controllable", "articulable"}:
        score += 3
    if _is_receptacle(record):
        score += 1
    return score


def _dedupe_records(records):
    dedup = {}
    for record in records:
        dedup.setdefault(record["synset"], record)
    return list(dedup.values())


def _build_single_env_a(
    env_idx,
    scene_model,
    primary_task,
    selected_records,
    additional_records,
    all_records,
    category_records,
    scene,
    rng,
    seed,
):
    env_id = f"Env-A_{scene_model}_{env_idx:04d}_{_short_hash(primary_task + str(seed))}"
    target_room = _choose_room(primary_task, selected_records, scene, rng)
    added_objects = []
    delta_nodes = []
    delta_edges = []

    for local_idx, record in enumerate(all_records):
        category = _choose_category(record, scene, rng)
        object_id = f"add_{local_idx:03d}_{_slug(category)}"
        placement = _choose_placement(record, category, target_room, scene, rng)
        role = "task_object" if record["synset"] in {r["synset"] for r in selected_records} else "context_object"
        added = {
            "object_id": object_id,
            "synset": record["synset"],
            "category": category,
            "object_type": record.get("object_type"),
            "semantic_roles": [role],
            "room_id": placement["room_id"],
            "placement": placement,
            "asset_metadata": {
                "definition": record.get("definition"),
                "tasks": record.get("tasks", [])[:10],
                "properties": record.get("properties", {}),
                "edit_metadata": record.get("edit_metadata", {}),
                "mass_estimate": record.get("mass_estimates", {}).get(category),
            },
        }
        added_objects.append(added)
        delta_nodes.append(
            {
                "id": object_id,
                "type": "added_object",
                "category": category,
                "synset": record["synset"],
                "room_id": placement["room_id"],
                "semantic": record.get("edit_metadata", {}),
            }
        )
        delta_edges.append({"source": f"room::{placement['room_id']}", "target": object_id, "relation": "contains"})
        if placement.get("support_object_id"):
            delta_edges.append(
                {
                    "source": placement["support_object_id"],
                    "target": object_id,
                    "relation": "supports",
                    "mode": placement["mode"],
                }
            )

    task_objects = [obj for obj in added_objects if "task_object" in obj["semantic_roles"]]
    task_plan = _build_env_a_solution_plan(task_objects, scene, target_room)

    return {
        "env_id": env_id,
        "env_type": "Env-A",
        "base_scene": {
            "scene_model": scene_model,
            "base_env_usd": f"{scene_model}.usd",
            "source_graph": "scene_overlay",
        },
        "task": {
            "task_id": f"{env_id}_task",
            "task_type": "basic_task_environment",
            "primary_behavior_task": primary_task,
            "instruction": _build_instruction(primary_task, task_objects, target_room),
            "target_room": target_room,
        },
        "added_objects": added_objects,
        "delta_sg": {
            "operation": "add_task_assets",
            "nodes": delta_nodes,
            "edges": delta_edges,
            "validation_status": "pending_physical_instantiation",
        },
        "physical_instantiation": {
            "status": "pending",
            "runner": "code/deltasg_env_a_omnigibson.py",
            "expected_command": (
                "conda run -n behavior python code/deltasg_env_a_omnigibson.py "
                "--input code/outputs/deltasg_env_a.json --env-indices all"
            ),
        },
        "robot": {
            "robot_id": "robot_0",
            "initial_room": _choose_robot_room(target_room, scene),
            "pose": None,
            "navigation_hint": _room_path(scene, _choose_robot_room(target_room, scene), target_room),
        },
        "camera": _default_camera_plan(scene, target_room),
        "solution_plan": task_plan,
        "provenance": {
            "target_synsets": [record["synset"] for record in selected_records],
            "context_synsets": [record["synset"] for record in additional_records],
            "num_category_records_available": len(category_records),
        },
    }


def _choose_primary_task(selected_records, additional_records):
    counts = Counter()
    selected_synsets = {record["synset"] for record in selected_records}
    for record in selected_records + additional_records:
        weight = 3 if record["synset"] in selected_synsets else 1
        for task in record.get("tasks", []):
            counts[task] += weight
    if not counts:
        return "generic_home_care_task"
    return counts.most_common(1)[0][0]


def _choose_room(task, selected_records, scene, rng):
    rooms = scene["rooms"]
    if not rooms:
        return None

    task_tokens = _tokens(task)
    record_tokens = set()
    for record in selected_records:
        record_tokens |= _tokens(record.get("synset", ""))
        for category in record.get("direct_categories", []):
            record_tokens |= _tokens(category)
        for parent in record.get("parents", []):
            record_tokens |= _tokens(parent)
    semantic_tokens = task_tokens | record_tokens
    room_scores = Counter()
    for room in rooms:
        room_base = room.split("_")[0]
        for key, keywords in ROOM_KEYWORDS.items():
            if key in room or room_base in key:
                room_scores[room] += len(semantic_tokens & keywords) * 5

    for record in selected_records:
        for category, placements in record.get("sampled_placements", {}).items():
            for placement in placements[:5]:
                room = _nearest_room(placement.get("pos"), scene["room_centers"])
                if room:
                    room_scores[room] += 1

    for room, objects in scene["room_objects"].items():
        room_scores[room] += len(objects) * 0.05

    if room_scores:
        best_score = max(room_scores.values())
        best_rooms = [room for room, score in room_scores.items() if score == best_score]
        return rng.choice(sorted(best_rooms))
    return rng.choice(rooms)


def _choose_category(record, scene, rng):
    categories = list(record.get("direct_categories", []))
    if not categories:
        return record["synset"].split(".")[0].replace("__", "_")
    scene_counts = scene["category_counts"]
    categories.sort(key=lambda category: (-scene_counts.get(category, 0), category))
    top_count = scene_counts.get(categories[0], 0)
    top = [category for category in categories if scene_counts.get(category, 0) == top_count]
    return rng.choice(top)


def _choose_placement(record, category, target_room, scene, rng):
    support = _choose_support_object(target_room, scene, prefer_inside=_is_receptacle(record), rng=rng)
    sampled_pose = _sampled_pose_for_category(record, category)
    pose = sampled_pose or _pose_near_support(support, scene["room_centers"].get(target_room))

    mode = "on_top"
    if support:
        support_semantic = support.get("semantic") or {}
        receptacle = support_semantic.get("receptacle") or {}
        if _is_receptacle(record) and receptacle.get("supports_inside"):
            mode = "inside"
        elif receptacle.get("supports_on_top"):
            mode = "on_top"
        elif receptacle.get("supports_inside"):
            mode = "inside"

    return {
        "room_id": target_room,
        "mode": mode,
        "support_object_id": support.get("object_id") if support else None,
        "support_category": support.get("category") if support else None,
        "pose": {
            "position": pose,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "pose_source": "sampled_metadata" if sampled_pose else "support_object_heuristic",
    }


def _choose_support_object(target_room, scene, prefer_inside, rng):
    candidates = []
    for obj in scene["receptacles"]:
        if target_room not in obj.get("rooms", []):
            continue
        receptacle = ((obj.get("semantic") or {}).get("receptacle") or {})
        score = 0
        if prefer_inside and receptacle.get("supports_inside"):
            score += 4
        if receptacle.get("supports_on_top"):
            score += 3
        if receptacle.get("supports_inside"):
            score += 2
        category = obj.get("category") or ""
        if category in {"countertop", "breakfast_table", "coffee_table", "table_lamp"}:
            score += 2
        candidates.append((score, obj))

    if not candidates:
        for room_obj in scene["room_objects"].get(target_room, []):
            if room_obj.get("category") not in STRUCTURAL_CATEGORIES:
                candidates.append((1, room_obj))
    if not candidates:
        return None

    best_score = max(score for score, _ in candidates)
    best = [obj for score, obj in candidates if score == best_score]
    return rng.choice(sorted(best, key=lambda obj: obj.get("object_id") or ""))


def _sampled_pose_for_category(record, category):
    placements = record.get("sampled_placements", {}).get(category, [])
    if not placements:
        return None
    for placement in placements:
        pos = placement.get("pos")
        if pos and len(pos) >= 3:
            return pos[:3]
    return None


def _pose_near_support(support, room_center):
    if support:
        position = ((support.get("pose") or {}).get("position") or [0.0, 0.0, 0.0])[:3]
        return [float(position[0]), float(position[1]), float(position[2]) + 0.15]
    if room_center:
        return [float(room_center[0]), float(room_center[1]), 0.8]
    return [0.0, 0.0, 0.8]


def _build_instruction(primary_task, task_objects, target_room):
    readable_task = primary_task.replace("_", " ").replace("-", " ")
    categories = ", ".join(obj["category"].replace("_", " ") for obj in task_objects[:5])
    return f"Complete the {readable_task} task in {target_room} using the relevant objects: {categories}."


def _build_env_a_solution_plan(task_objects, scene, target_room):
    plan = [{"step_id": 1, "primitive": "MOVE", "nl": f"Move to {target_room}", "target_room": target_room}]
    step_id = 2
    for obj in task_objects:
        plan.append(
            {
                "step_id": step_id,
                "primitive": "MOVE",
                "nl": f"Move to {obj['category']}",
                "target_object": obj["object_id"],
                "target_room": obj["room_id"],
            }
        )
        step_id += 1
        interaction = ((obj.get("asset_metadata") or {}).get("edit_metadata") or {}).get("interaction", {})
        primitive = "INTERACT" if interaction.get("kind") in {"controllable", "articulable"} else "PICK"
        plan.append(
            {
                "step_id": step_id,
                "primitive": primitive,
                "nl": f"{primitive.title()} {obj['category'].replace('_', ' ')}",
                "target_object": obj["object_id"],
                "inventory": [] if primitive == "INTERACT" else [obj["object_id"]],
            }
        )
        step_id += 1
    return plan


def _choose_robot_room(target_room, scene):
    paths = scene.get("navigation", {}).get("shortest_room_paths", {})
    candidates = []
    for room, room_paths in paths.items():
        if room == target_room:
            continue
        path = room_paths.get(target_room)
        if path:
            candidates.append((path.get("distance") or 0, room))
    if not candidates:
        return target_room
    return sorted(candidates, reverse=True)[0][1]


def _room_path(scene, start_room, target_room):
    if start_room == target_room:
        return {"rooms": [target_room], "distance": 0.0}
    return (
        scene.get("navigation", {})
        .get("shortest_room_paths", {})
        .get(start_room, {})
        .get(target_room)
    )


def _default_camera_plan(scene, target_room):
    center = scene["room_centers"].get(target_room) or [0.0, 0.0, 0.0]
    return [
        {
            "camera_id": f"cam_{target_room}",
            "camera_type": "global_camera",
            "room_id": target_room,
            "pose": {
                "position": [float(center[0]), float(center[1]), 2.4],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    ]


def _nearest_room(pos, room_centers):
    if not pos or len(pos) < 2:
        return None
    best = None
    for room, center in room_centers.items():
        if not center:
            continue
        dx = float(pos[0]) - float(center[0])
        dy = float(pos[1]) - float(center[1])
        dist = dx * dx + dy * dy
        candidate = (dist, room)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best else None


def _interaction_kind(record):
    return (record.get("edit_metadata", {}).get("interaction") or {}).get("kind")


def _is_receptacle(record):
    return bool((record.get("edit_metadata", {}).get("receptacle") or {}).get("can_support"))


def _tokens(text):
    return {token for token in re.split(r"[_\-\W]+", str(text).lower()) if token}


def _slug(text):
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).lower())
    return re.sub(r"_+", "_", text).strip("_") or "object"


def _short_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Generate DeltaSG Env-A task environment JSON.")
    parser.add_argument(
        "--task-asset-db",
        default="task_asset_database.json",
        help="Path to task_asset_database.json with scene_overlay.",
    )
    parser.add_argument("--output", default="code/outputs/deltasg_env_a.json", help="Output Env-A JSON path.")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_ENVS, help="Number of Env-A environments to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--task-objects", type=int, default=DEFAULT_TASK_OBJECTS)
    parser.add_argument("--additional-objects", type=int, default=DEFAULT_ADDITIONAL_OBJECTS)
    parser.add_argument("--scene-model", default=None)
    args = parser.parse_args()

    result = generate_env_a(
        task_asset_database_path=args.task_asset_db,
        output_path=args.output,
        num_envs=args.num_envs,
        seed=args.seed,
        task_objects=args.task_objects,
        additional_objects=args.additional_objects,
        scene_model=args.scene_model,
    )
    print(f"saved {args.output} with {len(result['envs'])} Env-A environments")
    for env in result["envs"]:
        print(
            f"{env['env_id']}: {len(env['added_objects'])} added objects, "
            f"task={env['task']['primary_behavior_task']}, room={env['task']['target_room']}"
        )


if __name__ == "__main__":
    main()
