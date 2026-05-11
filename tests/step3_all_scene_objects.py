import os
import sys
import json
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True

import omnigibson as og


def to_list(x):
    try:
        return x.tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return x


def get_state_by_class_name(obj, class_name):
    for cls, state in getattr(obj, "states", {}).items():
        if cls.__name__ == class_name:
            return state
    return None


def get_room_info_fallback(obj):
    room = {}
    for attr in ["room_type", "room_instance", "in_rooms", "rooms"]:
        if hasattr(obj, attr):
            try:
                room[attr] = to_list(getattr(obj, attr))
            except Exception:
                room[attr] = None
    return room


def collect_obj_info(obj):
    info = {
        "name": getattr(obj, "name", None),
        "category": getattr(obj, "category", None),
        "prim_path": getattr(obj, "prim_path", None),
        "room_info": get_room_info_fallback(obj),
        "available_states": [],
    }

    # pose
    try:
        pos, quat = obj.get_position_orientation()
        info["position"] = to_list(pos)
        info["orientation_xyzw"] = to_list(quat)
    except Exception:
        info["position"] = None
        info["orientation_xyzw"] = None

    # states list
    try:
        info["available_states"] = sorted([cls.__name__ for cls in obj.states.keys()])
    except Exception:
        info["available_states"] = []

    # AABB
    aabb_state = get_state_by_class_name(obj, "AABB")
    if aabb_state is not None:
        try:
            aabb_min, aabb_max = aabb_state.get_value()
            info["aabb_min"] = to_list(aabb_min)
            info["aabb_max"] = to_list(aabb_max)
        except Exception:
            info["aabb_min"] = None
            info["aabb_max"] = None
    else:
        info["aabb_min"] = None
        info["aabb_max"] = None

    # contact count
    contact_state = get_state_by_class_name(obj, "ContactBodies")
    if contact_state is not None:
        try:
            contacts = contact_state.get_value()
            info["contact_body_count"] = len(contacts)
        except Exception:
            info["contact_body_count"] = None
    else:
        info["contact_body_count"] = None

    return info


def get_all_scene_objects(scene):
    objs = []

    # 常见情况：scene.objects
    if hasattr(scene, "objects"):
        try:
            scene_objects = getattr(scene, "objects")
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    # 兼容某些版本可能的内部存储
    if not objs and hasattr(scene, "_objects"):
        try:
            scene_objects = getattr(scene, "_objects")
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    # 去重
    dedup = {}
    for obj in objs:
        key = getattr(obj, "prim_path", None) or getattr(obj, "name", None) or str(id(obj))
        dedup[key] = obj

    return list(dedup.values())


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "objects": [],
        "robots": [],
    }

    try:
        print("[1/4] create env")
        env = og.Environment(configs=cfg)

        print("[2/4] reset")
        env.reset()

        print("[3/4] warmup")
        for _ in range(30):
            og.sim.step()

        print("[4/4] export all scene objects")
        all_objs = get_all_scene_objects(env.scene)

        print(f"total scene objects found = {len(all_objs)}")

        exported = []
        category_counts = {}

        for obj in all_objs:
            info = collect_obj_info(obj)
            exported.append(info)

            cat = info["category"]
            if cat is None:
                cat = "__NONE__"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        result = {
            "ok": True,
            "scene_model": "Rs_int",
            "num_scene_objects": len(all_objs),
            "category_counts": dict(sorted(category_counts.items(), key=lambda x: x[0])),
            "objects": exported,
        }

        with open("scene_all_objects.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"saved scene_all_objects.json with {len(exported)} objects")
        hard_exit(0)

    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        hard_exit(1)


if __name__ == "__main__":
    main()