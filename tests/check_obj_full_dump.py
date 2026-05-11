import os
import sys
import json
import argparse
import traceback
import inspect
import numpy as np

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True

import omnigibson as og


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def to_jsonable(x, max_items=80, max_str=1000):
    if x is None:
        return None

    if isinstance(x, (bool, int, float, str)):
        if isinstance(x, str) and len(x) > max_str:
            return x[:max_str] + "...<truncated>"
        return x

    if isinstance(x, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "sample": to_jsonable(x.flatten()[:20].tolist()),
        }

    if hasattr(x, "tolist"):
        try:
            return to_jsonable(x.tolist())
        except Exception:
            pass

    if isinstance(x, (list, tuple, set)):
        arr = list(x)
        if len(arr) > max_items:
            return [to_jsonable(v) for v in arr[:max_items]] + [f"...<truncated {len(arr)} items>"]
        return [to_jsonable(v) for v in arr]

    if isinstance(x, dict):
        out = {}
        for i, (k, v) in enumerate(x.items()):
            if i >= max_items:
                out["..."] = f"<truncated {len(x)} items>"
                break
            out[str(k)] = to_jsonable(v)
        return out

    return {
        "__type__": type(x).__name__,
        "__repr__": repr(x)[:max_str],
    }


def get_all_scene_objects(scene):
    objs = []

    if hasattr(scene, "objects"):
        try:
            scene_objects = scene.objects
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    if not objs and hasattr(scene, "_objects"):
        try:
            scene_objects = scene._objects
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    dedup = {}
    for obj in objs:
        key = getattr(obj, "prim_path", None) or getattr(obj, "name", None) or str(id(obj))
        dedup[key] = obj

    return list(dedup.values())


def safe_getattr(obj, attr):
    try:
        return True, getattr(obj, attr), None
    except Exception as e:
        return False, None, repr(e)


def dump_obj_dict(obj):
    try:
        return to_jsonable(getattr(obj, "__dict__", {}))
    except Exception as e:
        return {"error": repr(e)}


def dump_dir_attrs(obj, keyword=None):
    readable_attrs = {}
    callable_methods = []
    failed_attrs = {}

    for attr in dir(obj):
        if attr.startswith("__") and attr.endswith("__"):
            continue

        if keyword is not None and keyword.lower() not in attr.lower():
            continue

        ok, val, err = safe_getattr(obj, attr)
        if not ok:
            failed_attrs[attr] = err
            continue

        if callable(val):
            callable_methods.append(attr)
        else:
            readable_attrs[attr] = {
                "type": type(val).__name__,
                "value": to_jsonable(val),
            }

    return {
        "readable_attrs": readable_attrs,
        "callable_methods": sorted(callable_methods),
        "failed_attrs": failed_attrs,
    }


def dump_class_members(obj, keyword=None):
    members = {}

    cls = type(obj)
    for name, member in inspect.getmembers(cls):
        if name.startswith("__") and name.endswith("__"):
            continue

        if keyword is not None and keyword.lower() not in name.lower():
            continue

        if inspect.isfunction(member):
            kind = "function"
        elif isinstance(member, property):
            kind = "property"
        else:
            kind = type(member).__name__

        members[name] = {
            "kind": kind,
            "repr": repr(member)[:500],
        }

    return members


def dump_abilities(obj):
    out = {}

    for attr in ["abilities", "_abilities"]:
        try:
            if hasattr(obj, attr):
                out[attr] = to_jsonable(getattr(obj, attr))
        except Exception as e:
            out[attr] = {"error": repr(e)}

    return out


def dump_states(obj):
    result = {}
    states = getattr(obj, "states", {}) or {}

    for cls, state in states.items():
        cls_name = getattr(cls, "__name__", str(cls))

        item = {
            "state_instance_type": type(state).__name__,
            "state_repr": repr(state)[:500],
            "value": None,
        }

        try:
            value = state.get_value()
            item["value"] = {
                "ok": True,
                "type": type(value).__name__,
                "value": to_jsonable(value),
            }
        except TypeError as e:
            item["value"] = {
                "ok": False,
                "reason": "get_value_requires_arguments",
                "error": repr(e),
            }
        except Exception as e:
            item["value"] = {
                "ok": False,
                "reason": "get_value_failed",
                "error": repr(e),
            }

        result[cls_name] = item

    return result


def dump_pose_and_aabb(obj):
    out = {}

    try:
        pos, quat = obj.get_position_orientation()
        out["position"] = to_jsonable(pos)
        out["orientation_xyzw"] = to_jsonable(quat)
    except Exception as e:
        out["pose_error"] = repr(e)

    try:
        for cls, state in getattr(obj, "states", {}).items():
            if getattr(cls, "__name__", "") == "AABB":
                aabb_min, aabb_max = state.get_value()
                out["aabb_min"] = to_jsonable(aabb_min)
                out["aabb_max"] = to_jsonable(aabb_max)
                out["aabb_extent"] = to_jsonable(np.asarray(aabb_max) - np.asarray(aabb_min))
    except Exception as e:
        out["aabb_error"] = repr(e)

    return out


def dump_links_and_joints(obj):
    out = {}

    for attr in ["links", "_links"]:
        try:
            if hasattr(obj, attr):
                links = getattr(obj, attr)
                if isinstance(links, dict):
                    out[attr] = {
                        k: {
                            "type": type(v).__name__,
                            "repr": repr(v)[:500],
                        }
                        for k, v in links.items()
                    }
                else:
                    out[attr] = to_jsonable(links)
        except Exception as e:
            out[attr] = {"error": repr(e)}

    for attr in ["joints", "_joints"]:
        try:
            if hasattr(obj, attr):
                joints = getattr(obj, attr)
                if isinstance(joints, dict):
                    out[attr] = {
                        k: {
                            "type": type(v).__name__,
                            "repr": repr(v)[:500],
                        }
                        for k, v in joints.items()
                    }
                else:
                    out[attr] = to_jsonable(joints)
        except Exception as e:
            out[attr] = {"error": repr(e)}

    return out


def dump_usd_prim(obj):
    result = {
        "prim_path": getattr(obj, "prim_path", None),
        "prim_valid": False,
        "prim_type_name": None,
        "attributes": {},
        "metadata": {},
        "children": [],
    }

    try:
        prim = getattr(obj, "prim", None)
        if prim is None:
            return result

        result["prim_valid"] = bool(prim.IsValid())
        result["prim_type_name"] = prim.GetTypeName()

        for attr in prim.GetAttributes():
            name = attr.GetName()
            try:
                result["attributes"][name] = {
                    "type_name": str(attr.GetTypeName()),
                    "value": to_jsonable(attr.Get()),
                }
            except Exception as e:
                result["attributes"][name] = {"error": repr(e)}

        try:
            result["metadata"] = to_jsonable(prim.GetAllMetadata())
        except Exception as e:
            result["metadata_error"] = repr(e)

        try:
            for child in prim.GetChildren():
                result["children"].append({
                    "name": child.GetName(),
                    "path": str(child.GetPath()),
                    "type_name": child.GetTypeName(),
                })
        except Exception as e:
            result["children_error"] = repr(e)

    except Exception as e:
        result["error"] = repr(e)

    return result


def dump_one_obj(obj, keyword=None):
    name = getattr(obj, "name", None)
    category = getattr(obj, "category", None)
    prim_path = getattr(obj, "prim_path", None)

    return {
        "basic": {
            "name": name,
            "category": category,
            "class_type": type(obj).__name__,
            "prim_path": prim_path,
        },
        "__dict__": dump_obj_dict(obj),
        "abilities": dump_abilities(obj),
        "states": dump_states(obj),
        "pose_and_aabb": dump_pose_and_aabb(obj),
        "links_and_joints": dump_links_and_joints(obj),
        "dir_attrs": dump_dir_attrs(obj, keyword=keyword),
        "class_members": dump_class_members(obj, keyword=keyword),
        "usd_prim": dump_usd_prim(obj),
    }


def build_summary(objects_dump):
    categories = {}
    abilities = {}
    states = {}
    attrs = {}
    methods = {}

    for item in objects_dump:
        cat = item["basic"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

        for ab_group in item.get("abilities", {}).values():
            if isinstance(ab_group, dict):
                for ab in ab_group.keys():
                    abilities[ab] = abilities.get(ab, 0) + 1

        for st in item.get("states", {}).keys():
            states[st] = states.get(st, 0) + 1

        readable = item.get("dir_attrs", {}).get("readable_attrs", {})
        for a in readable.keys():
            attrs[a] = attrs.get(a, 0) + 1

        callable_methods = item.get("dir_attrs", {}).get("callable_methods", [])
        for m in callable_methods:
            methods[m] = methods.get(m, 0) + 1

    def sort_count(d):
        return dict(sorted(d.items(), key=lambda x: (-x[1], str(x[0]))))

    return {
        "categories": sort_count(categories),
        "abilities": sort_count(abilities),
        "states": sort_count(states),
        "attrs": sort_count(attrs),
        "methods": sort_count(methods),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--output", default=None)
    parser.add_argument("--name", default=None, help="Only dump object whose name contains this string")
    parser.add_argument("--category", default=None, help="Only dump object whose category contains this string")
    parser.add_argument("--keyword", default=None, help="Only dump attributes / class members whose name contains this keyword")
    parser.add_argument("--max-objects", type=int, default=-1)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    output_path = args.output or f"obj_full_dump_{args.scene}.json"

    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        },
        "objects": [],
        "robots": [],
    }

    try:
        print(f"[1/4] create env: scene={args.scene}")
        env = og.Environment(configs=cfg)

        print("[2/4] reset")
        env.reset()

        print("[3/4] warmup")
        for _ in range(30):
            og.sim.step()

        print("[4/4] collect objects")
        objs = get_all_scene_objects(env.scene)

        filtered = []
        for obj in objs:
            name = str(getattr(obj, "name", ""))
            category = str(getattr(obj, "category", ""))

            if args.name and args.name.lower() not in name.lower():
                continue

            if args.category and args.category.lower() not in category.lower():
                continue

            filtered.append(obj)

        if args.max_objects > 0:
            filtered = filtered[:args.max_objects]

        print(f"total_objects={len(objs)}, selected_objects={len(filtered)}")

        objects_dump = []
        for i, obj in enumerate(filtered):
            name = getattr(obj, "name", None)
            category = getattr(obj, "category", None)
            print(f"[{i + 1}/{len(filtered)}] dumping {name}, category={category}")
            objects_dump.append(dump_one_obj(obj, keyword=args.keyword))

        result = {
            "ok": True,
            "scene_model": args.scene,
            "num_total_objects": len(objs),
            "num_selected_objects": len(filtered),
            "filters": {
                "name": args.name,
                "category": args.category,
                "keyword": args.keyword,
                "max_objects": args.max_objects,
            },
            "summary": build_summary(objects_dump),
            "objects": objects_dump,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nSaved to: {output_path}")

        if args.print_summary:
            summary = result["summary"]

            print("\n=== categories ===")
            for k, v in summary["categories"].items():
                print(f"{k}: {v}")

            print("\n=== abilities ===")
            for k, v in summary["abilities"].items():
                print(f"{k}: {v}")

            print("\n=== states ===")
            for k, v in summary["states"].items():
                print(f"{k}: {v}")

            print("\n=== attrs, top 100 ===")
            for k, v in list(summary["attrs"].items())[:100]:
                print(f"{k}: {v}")

            print("\n=== methods, top 100 ===")
            for k, v in list(summary["methods"].items())[:100]:
                print(f"{k}: {v}")

        hard_exit(0)

    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        hard_exit(1)


if __name__ == "__main__":
    main()