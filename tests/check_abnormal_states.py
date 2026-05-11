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


def to_jsonable(x, max_list_len=50, max_str_len=500):
    """Safely convert common values to JSON-serializable form."""
    if x is None:
        return None

    if isinstance(x, (bool, int, float, str)):
        if isinstance(x, str) and len(x) > max_str_len:
            return x[:max_str_len] + "...<truncated>"
        return x

    if isinstance(x, (list, tuple)):
        arr = list(x)
        if len(arr) > max_list_len:
            return [to_jsonable(v) for v in arr[:max_list_len]] + [f"...<truncated {len(arr)} items>"]
        return [to_jsonable(v) for v in arr]

    if isinstance(x, dict):
        out = {}
        for i, (k, v) in enumerate(x.items()):
            if i >= max_list_len:
                out["..."] = f"<truncated {len(x)} items>"
                break
            out[str(k)] = to_jsonable(v)
        return out

    try:
        if hasattr(x, "tolist"):
            return to_jsonable(x.tolist())
    except Exception:
        pass

    try:
        if isinstance(x, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "sample": to_jsonable(x.flatten()[:20].tolist()),
            }
    except Exception:
        pass

    # Avoid dumping huge / non-serializable Omni objects directly
    return {
        "__type__": type(x).__name__,
        "__repr__": repr(x)[:max_str_len],
    }


def get_all_scene_objects(scene):
    objs = []

    if hasattr(scene, "objects"):
        try:
            scene_objects = getattr(scene, "objects")
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    if not objs and hasattr(scene, "_objects"):
        try:
            scene_objects = getattr(scene, "_objects")
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


def safe_get_attr(obj, attr):
    try:
        val = getattr(obj, attr)
        return True, val, None
    except Exception as e:
        return False, None, repr(e)


def dump_python_attrs(obj):
    """
    Dump all safely readable non-callable attributes.
    Callable methods are only listed, not called.
    """
    attrs = {}
    methods = []
    failed = {}

    for attr in dir(obj):
        if attr.startswith("__") and attr.endswith("__"):
            continue

        ok, val, err = safe_get_attr(obj, attr)
        if not ok:
            failed[attr] = err
            continue

        if callable(val):
            methods.append(attr)
            continue

        attrs[attr] = {
            "type": type(val).__name__,
            "value": to_jsonable(val),
        }

    return {
        "readable_attrs": attrs,
        "callable_methods": sorted(methods),
        "failed_attrs": failed,
    }


def dump_states(obj):
    states = {}
    raw_states = getattr(obj, "states", {}) or {}

    for cls, state in raw_states.items():
        cls_name = getattr(cls, "__name__", str(cls))
        item = {
            "state_class": cls_name,
            "state_instance_type": type(state).__name__,
            "get_value": None,
        }

        try:
            v = state.get_value()
            item["get_value"] = {
                "ok": True,
                "value": to_jsonable(v),
                "type": type(v).__name__,
            }
        except TypeError as e:
            item["get_value"] = {
                "ok": False,
                "reason": "requires_arguments_or_unsupported_signature",
                "error": repr(e),
            }
        except Exception as e:
            item["get_value"] = {
                "ok": False,
                "reason": "get_value_failed",
                "error": repr(e),
            }

        states[cls_name] = item

    return states


def dump_abilities(obj):
    for attr in ["abilities", "_abilities"]:
        try:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    return to_jsonable(val)
        except Exception:
            pass
    return {}


def dump_basic_pose(obj):
    out = {}

    try:
        pos, quat = obj.get_position_orientation()
        out["position"] = to_jsonable(pos)
        out["orientation_xyzw"] = to_jsonable(quat)
    except Exception as e:
        out["pose_error"] = repr(e)

    try:
        out["scale"] = to_jsonable(getattr(obj, "scale", None))
    except Exception as e:
        out["scale_error"] = repr(e)

    try:
        out["mass"] = to_jsonable(getattr(obj, "mass", None))
    except Exception as e:
        out["mass_error"] = repr(e)

    return out


def dump_usd_prim_attrs(obj):
    """
    Dump USD prim attributes if obj.prim exists.
    """
    result = {
        "prim_valid": False,
        "prim_type_name": None,
        "attributes": {},
        "metadata": {},
    }

    try:
        prim = getattr(obj, "prim", None)
        if prim is None:
            return result

        result["prim_valid"] = bool(prim.IsValid())
        result["prim_type_name"] = prim.GetTypeName()

        # USD attributes
        for attr in prim.GetAttributes():
            name = attr.GetName()
            try:
                val = attr.Get()
                result["attributes"][name] = {
                    "type_name": str(attr.GetTypeName()),
                    "value": to_jsonable(val),
                }
            except Exception as e:
                result["attributes"][name] = {
                    "error": repr(e),
                }

        # USD metadata
        try:
            metadata_keys = prim.GetAllMetadata()
            result["metadata"] = to_jsonable(metadata_keys)
        except Exception as e:
            result["metadata_error"] = repr(e)

    except Exception as e:
        result["error"] = repr(e)

    return result


def dump_single_object(obj):
    name = getattr(obj, "name", None)
    category = getattr(obj, "category", None)
    prim_path = getattr(obj, "prim_path", None)

    return {
        "name": name,
        "category": category,
        "prim_path": prim_path,
        "class_type": type(obj).__name__,

        "basic": {
            "name": to_jsonable(name),
            "category": to_jsonable(category),
            "prim_path": to_jsonable(prim_path),
        },

        "pose_and_physics": dump_basic_pose(obj),
        "abilities": dump_abilities(obj),
        "states": dump_states(obj),
        "python_introspection": dump_python_attrs(obj),
        "usd_prim": dump_usd_prim_attrs(obj),
    }


def summarize_discovered_fields(objects_dump):
    """
    Summarize all discovered attr names / state names / ability names.
    """
    all_categories = {}
    all_abilities = {}
    all_states = {}
    all_python_attrs = {}
    all_methods = {}

    for item in objects_dump:
        cat = item.get("category")
        all_categories[cat] = all_categories.get(cat, 0) + 1

        for ab in item.get("abilities", {}).keys():
            all_abilities[ab] = all_abilities.get(ab, 0) + 1

        for st in item.get("states", {}).keys():
            all_states[st] = all_states.get(st, 0) + 1

        readable_attrs = item.get("python_introspection", {}).get("readable_attrs", {})
        for a in readable_attrs.keys():
            all_python_attrs[a] = all_python_attrs.get(a, 0) + 1

        methods = item.get("python_introspection", {}).get("callable_methods", [])
        for m in methods:
            all_methods[m] = all_methods.get(m, 0) + 1

    def sort_dict(d):
        return dict(sorted(d.items(), key=lambda x: (-x[1], str(x[0]))))

    return {
        "categories": sort_dict(all_categories),
        "abilities": sort_dict(all_abilities),
        "states": sort_dict(all_states),
        "python_attrs": sort_dict(all_python_attrs),
        "methods": sort_dict(all_methods),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-objects", type=int, default=-1)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    output_path = args.output or f"all_obj_attrs_{args.scene}.json"

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
        print(f"[1/4] create env, scene = {args.scene}")
        env = og.Environment(configs=cfg)

        print("[2/4] reset")
        env.reset()

        print("[3/4] warmup")
        for _ in range(30):
            og.sim.step()

        print("[4/4] dump all object attributes")
        objs = get_all_scene_objects(env.scene)
        if args.max_objects > 0:
            objs = objs[:args.max_objects]

        objects_dump = []
        for i, obj in enumerate(objs):
            name = getattr(obj, "name", None)
            category = getattr(obj, "category", None)
            print(f"[{i+1}/{len(objs)}] dumping {name}, category={category}")
            objects_dump.append(dump_single_object(obj))

        summary = summarize_discovered_fields(objects_dump)

        result = {
            "ok": True,
            "scene_model": args.scene,
            "num_objects": len(objects_dump),
            "summary": summary,
            "objects": objects_dump,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nSaved to: {output_path}")
        print(f"num_objects = {len(objects_dump)}")

        if args.print_summary:
            print("\n=== Discovered abilities ===")
            for k, v in summary["abilities"].items():
                print(f"{k}: {v}")

            print("\n=== Discovered states ===")
            for k, v in summary["states"].items():
                print(f"{k}: {v}")

            print("\n=== Top discovered python attrs ===")
            for k, v in list(summary["python_attrs"].items())[:100]:
                print(f"{k}: {v}")

        hard_exit(0)

    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        hard_exit(1)


if __name__ == "__main__":
    main()