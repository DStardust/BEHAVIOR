"""Sample identity shared by generation and dataset auditing."""

import hashlib
import json


def sample_fingerprint(run):
    te = run.get("task_environment") or {}
    task = te.get("task") or run.get("task") or {}
    records = te.get("added_objects") or []
    identities = {}
    objects = []
    for item in records:
        placement = item.get("placement") or {}
        pose = item.get("pose") or item.get("final_pose_before_warmup") or placement.get("pose") or {}
        identity = {
            "category": item.get("category"), "model": item.get("model"),
            "room": item.get("room_id") or placement.get("room_id"),
            "position": [round(float(v), 3) for v in pose.get("position", [])],
        }
        identities[item.get("object_id") or item.get("object_name")] = identity
        objects.append({**identity, "mode": placement.get("mode"),
                        "roles": sorted(item.get("semantic_roles") or []),
                        "support": placement.get("support_object_id")})

    def reference(name):
        return identities.get(name, name)

    def ordered(items):
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))

    def semantic_states(item):
        states = dict(item.get("states") or {})
        if isinstance(states.get("covered"), dict):
            states["covered"] = {key: value for key, value in states["covered"].items()
                                 if key != "particle_snapshot"}
        return states

    for item in objects:
        item["support"] = reference(item["support"])
    reasoning = te.get("semantic_reasoning") or task.get("semantic_reasoning") or {}
    payload = {
        "scene": (te.get("base_scene") or {}).get("scene_model"),
        "env_type": te.get("env_type"),
        "primary_task": task.get("primary_behavior_task"),
        "target_room": task.get("target_room"),
        "solution_path": reasoning.get("solution_path"),
        "objects": ordered(objects),
        "plan_objects": ordered([
            {"id": reference(item.get("object_id")), "category": item.get("category"),
             "role": item.get("semantic_role"), "room": item.get("room") or item.get("room_id")}
            for item in task.get("plan_objects") or []
        ]),
        "state_changes": ordered([
            {"object": reference(item.get("object_id") or item.get("object_name")),
             "states": semantic_states(item)}
            for item in te.get("state_changed_objects") or []
        ]),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), payload
