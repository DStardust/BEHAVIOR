"""
Physically instantiate DeltaSG Env-A plans in OmniGibson.

This is the server-side companion to code/deltasg_env_a.py. It takes the
offline Env-A JSON, launches OmniGibson, adds the DeltaSG objects as
DatasetObjects, applies the planned OnTop / Inside relations where possible,
warms up physics, and writes a validation JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import torch as th

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True

import omnigibson as og
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from omnigibson.utils.constants import PrimType


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def build_env_config(scene_model, robot_model=None):
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
        "objects": [],
        "robots": [],
    }
    if robot_model:
        cfg["robots"] = [
            {
                "model": robot_model,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "obs_modalities": ["rgb", "depth", "seg_semantic"],
            }
        ]
    return cfg


def instantiate_env_a_file(
    input_path,
    output_path,
    env_indices=None,
    robot_model=None,
    warmup_steps=120,
    settle_threshold=0.25,
    dump_sim_state=False,
):
    data = load_json(input_path)
    envs = data.get("envs", [])
    if env_indices is not None:
        selected = [envs[idx] for idx in env_indices]
    else:
        selected = envs

    results = []
    for env_plan in selected:
        results.append(
            instantiate_single_env_a(
                env_plan=env_plan,
                robot_model=robot_model,
                warmup_steps=warmup_steps,
                settle_threshold=settle_threshold,
                dump_sim_state=dump_sim_state,
                output_dir=Path(output_path).parent,
            )
        )

    output = {
        "schema_version": "deltasg_env_a_omnigibson_validation.v1",
        "source_env_a": str(input_path),
        "warmup_steps": warmup_steps,
        "settle_threshold": settle_threshold,
        "results": results,
        "ok": all(item.get("ok", False) for item in results),
    }
    write_json(output_path, output)
    return output


def instantiate_single_env_a(
    env_plan,
    robot_model=None,
    warmup_steps=120,
    settle_threshold=0.25,
    dump_sim_state=False,
    output_dir=Path("."),
):
    env_id = env_plan["env_id"]
    scene_model = env_plan["base_scene"]["scene_model"]
    result = {
        "env_id": env_id,
        "scene_model": scene_model,
        "ok": False,
        "objects": [],
        "errors": [],
    }

    env = None
    try:
        env = og.Environment(configs=build_env_config(scene_model=scene_model, robot_model=robot_model))
        env.reset()
        _warmup_env(env, 10)

        support_cache = {}
        created_objects = []
        for spec in env_plan.get("added_objects", []):
            item = instantiate_added_object(env=env, spec=spec, support_cache=support_cache)
            result["objects"].append(item)
            if item.get("ok"):
                created_objects.append(item["object_name"])

        _warmup_env(env, warmup_steps)
        result["settling"] = collect_settling_report(
            env=env,
            object_names=created_objects,
            settle_threshold=settle_threshold,
        )
        result["ok"] = (
            len(created_objects) == len(env_plan.get("added_objects", []))
            and result["settling"]["all_within_threshold"]
        )

        result["sim_state"] = save_sim_state(env_id, output_dir) if dump_sim_state else None
    except Exception as exc:
        result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                og.clear()
            except Exception:
                pass

    return result


def instantiate_added_object(env, spec, support_cache):
    object_id = spec["object_id"]
    category = spec["category"]
    placement = spec.get("placement", {})
    item = {
        "object_id": object_id,
        "category": category,
        "object_name": object_id,
        "ok": False,
        "relation_applied": None,
        "fallback_used": False,
        "errors": [],
    }

    try:
        prim_type = PrimType.CLOTH if spec.get("object_type") == "cloth" else PrimType.RIGID
        obj = DatasetObject(
            name=object_id,
            category=category,
            prim_type=prim_type,
            in_rooms=spec.get("room_id"),
        )
        env.scene.add_object(obj)

        pose = placement.get("pose", {})
        position = th.tensor(pose.get("position", [0.0, 0.0, 0.8]), dtype=th.float32)
        orientation = th.tensor(pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=th.float32)
        obj.set_position_orientation(position=position, orientation=orientation)

        og.sim.step()

        support_obj = get_support_object(env, placement.get("support_object_id"), support_cache)
        if support_obj is not None:
            applied = try_apply_relation(obj=obj, support_obj=support_obj, mode=placement.get("mode"))
            item["relation_applied"] = applied
            if not applied["ok"]:
                item["fallback_used"] = True
                item["errors"].append(applied)

        pos, quat = obj.get_position_orientation()
        item["final_pose_before_warmup"] = {
            "position": _to_list(pos),
            "orientation_xyzw": _to_list(quat),
        }
        item["model"] = getattr(obj, "model", None)
        item["ok"] = True
    except Exception as exc:
        item["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})

    return item


def get_support_object(env, support_object_id, support_cache):
    if not support_object_id:
        return None
    if support_object_id in support_cache:
        return support_cache[support_object_id]
    support = env.scene.object_registry("name", support_object_id, None)
    support_cache[support_object_id] = support
    return support


def try_apply_relation(obj, support_obj, mode):
    state_type = object_states.Inside if mode == "inside" else object_states.OnTop
    try:
        if state_type not in obj.states:
            return {
                "ok": False,
                "mode": mode,
                "state": state_type.__name__,
                "error": "state_not_available_on_object",
            }
        obj.states[state_type].set_value(support_obj, True)
        return {"ok": True, "mode": mode, "state": state_type.__name__, "support": support_obj.name}
    except Exception as exc:
        return {
            "ok": False,
            "mode": mode,
            "state": state_type.__name__,
            "support": getattr(support_obj, "name", None),
            "error": repr(exc),
        }


def collect_settling_report(env, object_names, settle_threshold):
    before = {}
    for name in object_names:
        obj = env.scene.object_registry("name", name, None)
        if obj is not None:
            before[name] = _position(obj)

    _warmup_env(env, 20)

    objects = []
    all_within = True
    for name, start_pos in before.items():
        obj = env.scene.object_registry("name", name, None)
        if obj is None:
            objects.append({"object_name": name, "ok": False, "error": "missing_after_warmup"})
            all_within = False
            continue
        end_pos = _position(obj)
        displacement = float(np.linalg.norm(np.asarray(end_pos) - np.asarray(start_pos)))
        ok = displacement <= settle_threshold
        all_within = all_within and ok
        objects.append(
            {
                "object_name": name,
                "start_position": start_pos,
                "end_position": end_pos,
                "displacement": displacement,
                "within_threshold": ok,
            }
        )

    return {
        "settle_threshold": settle_threshold,
        "all_within_threshold": all_within,
        "objects": objects,
    }


def save_sim_state(env_id, output_dir):
    state = og.sim.dump_state(serialized=True)
    path = Path(output_dir) / f"{env_id}_sim_state.npy"
    np.save(path, np.asarray(state))
    return {"serialized_state_npy": str(path)}


def _warmup_env(env, steps):
    empty_action = th.empty(0)
    for _ in range(steps):
        try:
            env.step(empty_action)
        except Exception:
            og.sim.step()


def _position(obj):
    pos, _ = obj.get_position_orientation()
    return _to_list(pos)


def _to_list(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _parse_indices(text):
    if text is None or text.lower() == "all":
        return None
    indices = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(item))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Instantiate DeltaSG Env-A plans in OmniGibson.")
    parser.add_argument("--input", default="code/outputs/deltasg_env_a.json", help="Input Env-A JSON.")
    parser.add_argument(
        "--output",
        default="code/outputs/deltasg_env_a_og_validation.json",
        help="Output validation JSON.",
    )
    parser.add_argument("--env-indices", default="0", help='Env indices, e.g. "0", "0,2", "0-3", or "all".')
    parser.add_argument("--robot", default=None, help="Optional robot model, e.g. fetch.")
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--settle-threshold", type=float, default=0.25)
    parser.add_argument("--dump-sim-state", action="store_true")
    args = parser.parse_args()

    result = instantiate_env_a_file(
        input_path=args.input,
        output_path=args.output,
        env_indices=_parse_indices(args.env_indices),
        robot_model=args.robot,
        warmup_steps=args.warmup_steps,
        settle_threshold=args.settle_threshold,
        dump_sim_state=args.dump_sim_state,
    )
    print(f"saved {args.output}")
    print(json.dumps({"ok": result["ok"], "num_results": len(result["results"])}, indent=2))


if __name__ == "__main__":
    main()
