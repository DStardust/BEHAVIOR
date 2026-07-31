"""Pure-data checks for DeltaSG sample uniqueness and integrity contracts."""

import hashlib
import json


def fingerprint_payload(run):
    te = run["task_environment"]
    task = te["task"]
    objects = []
    for item in te.get("added_objects", []):
        placement = item.get("placement", {})
        pose = item.get("pose", {})
        objects.append({
            "category": item.get("category"),
            "roles": sorted(item.get("semantic_roles", [])),
            "room": item.get("room_id"),
            "mode": placement.get("mode"),
            "support": placement.get("support_object_id"),
            "position": [round(float(v), 3) for v in pose.get("position", [])[:3]],
        })
    return {
        "scene": te["base_scene"]["scene_model"],
        "env_type": te["env_type"],
        "primary_task": task["primary_behavior_task"],
        "target_room": task["target_room"],
        "objects": sorted(objects, key=lambda value: json.dumps(value, sort_keys=True)),
    }


def fingerprint(run):
    canonical = json.dumps(fingerprint_payload(run), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def integrity(before, after, limit=0.05):
    missing, moved = [], []
    for name, start in before.items():
        if name not in after:
            missing.append(name)
            continue
        displacement = sum((a - b) ** 2 for a, b in zip(start, after[name])) ** 0.5
        if displacement > limit:
            moved.append(name)
    return not missing and not moved


def sample(position):
    return {
        "task_environment": {
            "base_scene": {"scene_model": "Rs_int"},
            "env_type": "Env-A",
            "task": {"primary_behavior_task": "retrieve_book", "target_room": "living_room_0"},
            "added_objects": [{
                "category": "book", "semantic_roles": ["task_object"], "room_id": "living_room_0",
                "placement": {"mode": "on_top", "support_object_id": "coffee_table_0"},
                "pose": {"position": position},
            }],
        }
    }


def main():
    assert fingerprint(sample([1.0, 2.0, 3.0])) == fingerprint(sample([1.0, 2.0, 3.0]))
    assert fingerprint(sample([1.0, 2.0, 3.0])) != fingerprint(sample([1.1, 2.0, 3.0]))
    assert integrity({"table": [0, 0, 0]}, {"table": [0.04, 0, 0]})
    assert not integrity({"table": [0, 0, 0]}, {"table": [0.06, 0, 0]})
    assert not integrity({"table": [0, 0, 0]}, {})
    print("PASS: DeltaSG uniqueness and source-scene integrity contracts")


if __name__ == "__main__":
    main()
