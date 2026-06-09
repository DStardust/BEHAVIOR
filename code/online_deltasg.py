"""
Online DeltaSG engine for live OmniGibson environments.

This module implements the layer-2 DeltaSG step described in intro.md as an
online process: it reads the current OmniGibson scene, decides task-oriented
scene edits, applies those edits directly to the running simulator, relaxes
physics, and returns an updated graph plus validation report.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import traceback
import importlib.util
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.constants import PrimType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ASSET_DATABASE_PATH = REPO_ROOT / "OmniGibson" / "omnigibson" / "scene_graphs" / "task_asset_database.py"


def _load_task_asset_database_class():
    spec = importlib.util.spec_from_file_location("task_asset_database", TASK_ASSET_DATABASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskAssetDatabase


TaskAssetDatabase = _load_task_asset_database_class()


STRUCTURAL_CATEGORIES = {"agent", "ceilings", "ceiling", "floors", "floor", "walls", "wall"}
SUPPORT_SURFACE_TOKENS = {
    "bar",
    "bed",
    "bench",
    "cabinet",
    "cart",
    "chair",
    "counter",
    "countertop",
    "desk",
    "dresser",
    "floor",
    "island",
    "mat",
    "nightstand",
    "ottoman",
    "plate",
    "rack",
    "shelf",
    "sofa",
    "stool",
    "table",
    "tray",
}
BAD_SUPPORT_TOKENS = {
    "ceiling",
    "ceilings",
    "wall",
    "walls",
    "window",
    "door",
    "towel",
    "rack",
    "mirror",
    "lamp",
    "light",
    "agent",
}
GOOD_SUPPORT_TOKENS = {
    "bathroom": {"counter", "countertop", "sink", "cabinet", "shelf", "table", "floor"},
    "bedroom": {"bed", "nightstand", "dresser", "cabinet", "shelf", "table", "floor"},
    "entryway": {"floor", "bench", "table", "cabinet", "shelf"},
    "kitchen": {"counter", "countertop", "sink", "table", "cabinet", "shelf", "stove", "floor"},
    "living_room": {"table", "coffee", "shelf", "sofa", "cabinet", "floor"},
}
INSIDE_RECEPTACLE_TOKENS = {
    "basket",
    "bin",
    "bowl",
    "box",
    "cabinet",
    "can",
    "container",
    "cup",
    "drawer",
    "fridge",
    "jar",
    "mug",
    "oven",
    "pot",
    "sink",
    "washer",
}
ROOM_KEYWORDS = {
    "kitchen": {
        "bake",
        "bowl",
        "cook",
        "dish",
        "food",
        "fruit",
        "kitchen",
        "meal",
        "microwave",
        "oven",
        "pan",
        "plate",
        "pot",
        "prepare",
        "wash",
    },
    "bathroom": {"bath", "clean", "disinfect", "mirror", "shower", "toilet", "wash"},
    "bedroom": {"bed", "cloth", "fold", "laundry", "pillow", "shoe", "sleep"},
    "living_room": {"book", "decorate", "lamp", "table", "tv", "wine"},
    "entryway": {"bag", "backpack", "carry", "shoe", "trash"},
}


@dataclass
class OnlineDeltaSGConfig:
    task_objects: int = 5
    context_objects: int = 12
    warmup_steps: int = 120
    settle_steps: int = 30
    settle_threshold: float = 0.25
    near_distance: float = 1.25
    support_z_tolerance: float = 0.08
    seed: int = 0
    allow_cloth: bool = False


class OnlineDeltaSGEngine:
    """
    Live DeltaSG engine attached to a running OmniGibson environment.

    The engine never requires a pre-exported 3DSG JSON. Every operation starts
    from the current simulator state, applies edits immediately, and then
    re-snapshots the scene.
    """

    def __init__(self, env, metadata_dir=None, config=None):
        self.env = env
        self.config = config or OnlineDeltaSGConfig()
        self.rng = random.Random(self.config.seed)
        self.asset_db = TaskAssetDatabase.load(metadata_dir=metadata_dir or TaskAssetDatabase().metadata_dir)
        self._run_counter = 0
        self._category_models_cache = {}

    def snapshot(self):
        """Return a 3DSG-like graph built from the current live simulator state."""
        objects = [self._collect_object_info(obj) for obj in self._scene_objects()]
        rooms = sorted({room for obj in objects for room in obj["rooms"] if room})
        nodes = [
            {
                "id": f"room::{room}",
                "type": "room",
                "name": room,
                "category": "room",
                "semantic": {"interaction": {"kind": "none", "confidence": "explicit"}},
            }
            for room in rooms
        ]
        edges = []

        object_nodes = {}
        for obj in objects:
            node = {
                "id": obj["name"],
                "type": "object",
                "name": obj["name"],
                "category": obj["category"],
                "prim_path": obj["prim_path"],
                "pose": {
                    "position": obj["position"],
                    "orientation_xyzw": obj["orientation_xyzw"],
                },
                "bbox": {
                    "min": obj["aabb_min"],
                    "max": obj["aabb_max"],
                    "extent": obj["aabb_extent"],
                },
                "rooms": obj["rooms"],
                "available_states": obj["available_states"],
                "semantic": self._infer_object_semantic(obj),
            }
            nodes.append(node)
            object_nodes[node["id"]] = node
            for room in node["rooms"]:
                edges.append({"source": f"room::{room}", "target": node["id"], "relation": "contains"})

        for room, room_nodes in self._group_by_room(object_nodes.values()).items():
            for idx, src in enumerate(room_nodes):
                for dst in room_nodes[idx + 1 :]:
                    relation = self._spatial_relation(src, dst)
                    if relation is not None:
                        relation["room"] = room
                        edges.append({"source": src["id"], "target": dst["id"], **relation})

        navigation = self._build_navigation(rooms, object_nodes.values())
        edges.extend(navigation["edges"])
        return {
            "schema_version": "online_3dsg.v1",
            "source": "live_omnigibson_env",
            "timestamp": time.time(),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "num_objects": len(objects),
                "num_rooms": len(rooms),
                "category_counts": dict(Counter(obj["category"] for obj in objects if obj["category"])),
            },
            "navigation": navigation["navigation"],
        }

    def generate_env_a(self, task=None, target_room=None):
        """
        Create an Env-A task scene online and physically validate it.

        Returns a record containing the before graph, applied DeltaSG, validation
        report, and after graph. The current OmniGibson env remains edited.
        """
        before_graph = self.snapshot()
        selected_records = self._select_task_records(task)
        primary_task = task or self._choose_primary_task(selected_records)
        target_room = target_room or self._choose_target_room(primary_task, selected_records, before_graph)
        context_records = self._select_context_records(primary_task, selected_records)
        records = self._dedupe_records(selected_records + context_records)

        run_id = f"online_env_a_{self._run_counter:04d}"
        self._run_counter += 1

        delta = {
            "delta_id": run_id,
            "operation": "online_add_task_assets",
            "task": {
                "task_type": "basic_task_environment",
                "primary_behavior_task": primary_task,
                "target_room": target_room,
            },
            "nodes": [],
            "edges": [],
        }
        validation = {
            "ok": False,
            "created_objects": [],
            "failed_objects": [],
            "settling": None,
        }

        created_names = []
        for idx, record in enumerate(records):
            role = "task_object" if record in selected_records else "context_object"
            add_result = self.add_task_asset(
                record=record,
                object_name=f"{run_id}_{idx:03d}_{self._slug(self._choose_category(record))}",
                target_room=target_room,
                semantic_role=role,
            )
            if add_result["ok"]:
                created_names.append(add_result["object_name"])
                validation["created_objects"].append(add_result)
                delta["nodes"].append(add_result["delta_node"])
                delta["edges"].extend(add_result["delta_edges"])
            else:
                validation["failed_objects"].append(add_result)

        self._step(self.config.warmup_steps)
        validation["settling"] = self._collect_settling_report(created_names)
        validation["ok"] = (
            len(validation["failed_objects"]) == 0
            and validation["settling"]["all_within_threshold"]
        )
        after_graph = self.snapshot()
        task_instance = self._build_task_instance(run_id, primary_task, target_room, validation["created_objects"])
        task_environment = self._build_task_environment_record(
            env_id=run_id,
            env_type="Env-A",
            task_instance=task_instance,
            target_room=target_room,
            created_objects=validation["created_objects"],
            validation=validation,
            delta_sg=delta,
            graph=after_graph,
        )
        return {
            "schema_version": "online_deltasg_env_a.v1",
            "run_id": run_id,
            "ok": validation["ok"],
            "task_environment": task_environment,
            "base_scene": task_environment["base_scene"],
            "task": task_environment["task"],
            "robot": task_environment["robot"],
            "camera": task_environment["camera"],
            "added_objects": task_environment["added_objects"],
            "task_objects": task_environment["task_objects"],
            "solution_plan": task_environment["solution_plan"],
            "before_graph": before_graph,
            "delta_sg": delta,
            "validation": validation,
            "after_graph": after_graph,
            "task_instance": task_instance,
            "debug": {
                "before_graph": before_graph,
                "after_graph": after_graph,
                "failed_objects": validation["failed_objects"],
            },
        }

    def generate_env_b_fire(self, target_room=None):
        """
        Create an Env-B event scene online by setting a live object on fire and
        spawning a fire extinguisher solution object.
        """
        before_graph = self.snapshot()
        target_room = target_room or self._choose_room_with_objects(before_graph)
        fire_target = self._choose_fire_target(before_graph, target_room)
        if fire_target is None:
            fire_target = self._spawn_fire_target(target_room)

        run_id = f"online_env_b_fire_{self._run_counter:04d}"
        self._run_counter += 1
        fire_state = self._set_boolean_state(fire_target["name"], object_states.OnFire, True)
        extinguisher_record = self._record_for_category("fire_extinguisher")
        extinguisher = self.add_task_asset(
            record=extinguisher_record,
            object_name=f"{run_id}_extinguisher",
            target_room=target_room,
            semantic_role="interaction_tool",
        )
        self._step(self.config.warmup_steps)
        after_graph = self.snapshot()
        ok = bool(fire_state.get("ok") and extinguisher.get("ok"))
        delta_sg = {
            "delta_id": run_id,
            "operation": "online_add_fire_anomaly",
            "nodes": [
                {
                    "id": fire_target["name"],
                    "type": "state_changed_object",
                    "category": fire_target.get("category"),
                    "room_id": target_room,
                    "semantic_roles": ["goal_target", "anomaly"],
                    "states": {"on_fire": True},
                },
                extinguisher.get("delta_node"),
            ],
            "edges": [
                {"source": f"room::{target_room}", "target": fire_target["name"], "relation": "contains"},
                *extinguisher.get("delta_edges", []),
            ],
        }
        validation = {
            "ok": ok,
            "fire_state": fire_state,
            "solution_tool": extinguisher,
        }
        task_instance = self._build_fire_task_instance(run_id, target_room, fire_target["name"], extinguisher)
        task_environment = self._build_task_environment_record(
            env_id=run_id,
            env_type="Env-B",
            task_instance=task_instance,
            target_room=target_room,
            created_objects=[extinguisher] if extinguisher.get("ok") else [],
            validation=validation,
            delta_sg=delta_sg,
            graph=after_graph,
            state_changed_objects=[
                {
                    "object_id": fire_target["name"],
                    "category": fire_target.get("category"),
                    "room_id": target_room,
                    "states": {"on_fire": True},
                    "semantic_roles": ["goal_target", "anomaly"],
                }
            ],
        )
        return {
            "schema_version": "online_deltasg_env_b_fire.v1",
            "run_id": run_id,
            "ok": ok,
            "task_environment": task_environment,
            "base_scene": task_environment["base_scene"],
            "task": task_environment["task"],
            "robot": task_environment["robot"],
            "camera": task_environment["camera"],
            "added_objects": task_environment["added_objects"],
            "task_objects": task_environment["task_objects"],
            "solution_plan": task_environment["solution_plan"],
            "before_graph": before_graph,
            "delta_sg": delta_sg,
            "validation": validation,
            "after_graph": after_graph,
            "task_instance": task_instance,
            "debug": {
                "before_graph": before_graph,
                "after_graph": after_graph,
                "fire_state": fire_state,
            },
        }

    def generate_env_c_fire_disambiguation(self, target_room=None):
        """
        Create an Env-C semantic-disambiguation scene online.

        This first creates a fire Env-B scene, then adds a lower-utility valid
        candidate and an invalid semantic distractor.
        """
        env_b = self.generate_env_b_fire(target_room=target_room)
        run_id = f"online_env_c_fire_disambiguation_{self._run_counter:04d}"
        self._run_counter += 1
        target_room = env_b["task_instance"]["target_room"]
        alt_room = self._choose_different_room(self.snapshot(), target_room) or target_room

        bucket = self.add_task_asset(
            record=self._record_for_category("bucket"),
            object_name=f"{run_id}_water_bucket",
            target_room=alt_room,
            semantic_role="candidate_solution",
        )
        toy = self.add_task_asset(
            record=self._record_for_category("toy_bucket", fallback_category="bucket"),
            object_name=f"{run_id}_toy_bucket",
            target_room=target_room,
            semantic_role="semantic_distractor",
        )
        self._step(self.config.warmup_steps)
        after_graph = self.snapshot()
        optimal = env_b["task_instance"]["task_objects"][0]["object_id"]
        ok = bool(env_b["ok"] and bucket["ok"] and toy["ok"])
        env_b["delta_sg"]["delta_id"] = run_id
        env_b["delta_sg"]["operation"] = "online_add_fire_semantic_disambiguation"
        env_b["delta_sg"]["nodes"].extend([bucket.get("delta_node"), toy.get("delta_node")])
        env_b["delta_sg"]["edges"].extend(bucket.get("delta_edges", []) + toy.get("delta_edges", []))
        env_b["task_instance"]["task_id"] = f"{run_id}_task"
        env_b["task_instance"]["task_type"] = "Env-C"
        env_b["task_instance"]["instruction"] = "Quickly extinguish the fire using the most suitable tool."
        env_b["task_instance"]["semantic_constraints"] = ["fastest_solution", "fire_suppression_affordance"]
        env_b["task_instance"]["semantic_reasoning"] = {
            "reasoning_type": ["semantic_disambiguation", "affordance_grounding", "utility_reasoning"],
            "ground_truth": {
                "optimal_object": optimal,
                "rejected_candidates": [
                    {
                        "object_id": bucket["object_name"],
                        "reason": "valid_solution_but_lower_efficiency",
                    },
                    {
                        "object_id": toy["object_name"],
                        "reason": "invalid_affordance",
                    },
                ],
            },
        }
        validation = {
            "ok": ok,
            "env_b_fire": env_b["validation"],
            "candidate_solution": bucket,
            "semantic_distractor": toy,
        }
        created_objects = []
        solution_tool = env_b["validation"].get("solution_tool", {})
        if solution_tool.get("ok"):
            created_objects.append(solution_tool)
        for item in (bucket, toy):
            if item.get("ok"):
                created_objects.append(item)
        task_environment = self._build_task_environment_record(
            env_id=run_id,
            env_type="Env-C",
            task_instance=env_b["task_instance"],
            target_room=target_room,
            created_objects=created_objects,
            validation=validation,
            delta_sg=env_b["delta_sg"],
            graph=after_graph,
            state_changed_objects=[
                obj
                for obj in env_b["task_environment"].get("state_changed_objects", [])
            ],
        )
        return {
            "schema_version": "online_deltasg_env_c_fire_disambiguation.v1",
            "run_id": run_id,
            "ok": ok,
            "task_environment": task_environment,
            "base_scene": task_environment["base_scene"],
            "task": task_environment["task"],
            "robot": task_environment["robot"],
            "camera": task_environment["camera"],
            "added_objects": task_environment["added_objects"],
            "task_objects": task_environment["task_objects"],
            "solution_plan": task_environment["solution_plan"],
            "before_graph": env_b["before_graph"],
            "delta_sg": env_b["delta_sg"],
            "validation": validation,
            "after_graph": after_graph,
            "task_instance": env_b["task_instance"],
            "debug": {
                "before_graph": env_b["before_graph"],
                "after_graph": after_graph,
                "env_b_run_id": env_b["run_id"],
            },
        }

    def add_task_asset(self, record, object_name, target_room, semantic_role="task_object"):
        """Add one task asset directly to the live OmniGibson scene."""
        category = self._choose_category(record)
        placement = self._choose_live_placement(record, target_room)
        result = {
            "ok": False,
            "object_name": object_name,
            "category": category,
            "synset": record["synset"],
            "semantic_role": semantic_role,
            "placement": placement,
            "errors": [],
        }

        try:
            if not self._category_has_models(category):
                raise ValueError(f"No available models found for category {category}")
            prim_type = PrimType.CLOTH if record.get("object_type") == "cloth" else PrimType.RIGID
            obj = DatasetObject(
                name=object_name,
                category=category,
                prim_type=prim_type,
                in_rooms=target_room,
            )
            self.env.scene.add_object(obj)
            self._clear_usd_selection()
            obj.set_position_orientation(
                position=th.tensor(placement["pose"]["position"], dtype=th.float32),
                orientation=th.tensor(placement["pose"]["orientation_xyzw"], dtype=th.float32),
            )
            self._clear_usd_selection()
            self._step(1)

            relation = self._apply_relation(obj, placement)
            self._clear_usd_selection()
            self._step(5)
            position, orientation = obj.get_position_orientation()
            result.update(
                {
                    "ok": True,
                    "model": getattr(obj, "model", None),
                    "relation": relation,
                    "final_pose_before_warmup": {
                        "position": self._to_list(position),
                        "orientation_xyzw": self._to_list(orientation),
                    },
                    "delta_node": {
                        "id": object_name,
                        "type": "added_object",
                        "category": category,
                        "synset": record["synset"],
                        "room_id": target_room,
                        "semantic_roles": [semantic_role],
                        "semantic": record.get("edit_metadata", {}),
                    },
                    "delta_edges": self._delta_edges_for_added_object(object_name, target_room, placement, relation),
                }
            )
        except Exception as exc:
            result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})

        return result

    def _build_fire_task_instance(self, run_id, target_room, fire_object, extinguisher):
        ext_id = extinguisher["object_name"]
        return {
            "task_id": f"{run_id}_task",
            "task_type": "Env-B",
            "instruction": "Resolve the fire emergency using the extinguisher.",
            "target_room": target_room,
            "task_objects": [
                {"object_id": ext_id, "category": extinguisher["category"], "role": "interaction_tool"},
                {"object_id": fire_object, "category": "fire_target", "role": "goal_target"},
            ],
            "solution_plan": [
                {"step_id": 1, "primitive": "MOVE", "nl": "Move to fire extinguisher", "target_object": ext_id},
                {"step_id": 2, "primitive": "PICK", "nl": "Pick up fire extinguisher", "target_object": ext_id},
                {
                    "step_id": 3,
                    "primitive": "MOVE",
                    "nl": "Move to fire target",
                    "target_object": fire_object,
                    "inventory": [ext_id],
                },
                {
                    "step_id": 4,
                    "primitive": "INTERACT",
                    "nl": "Extinguish fire",
                    "tool_object": ext_id,
                    "target_object": fire_object,
                    "inventory": [ext_id],
                },
            ],
        }

    def _choose_room_with_objects(self, graph):
        counts = Counter()
        for node in graph.get("nodes", []):
            if node.get("type") == "object":
                for room in node.get("rooms", []):
                    counts[room] += 1
        if counts:
            max_count = max(counts.values())
            return self.rng.choice(sorted(room for room, count in counts.items() if count == max_count))
        rooms = [node["name"] for node in graph.get("nodes", []) if node.get("type") == "room"]
        return self.rng.choice(rooms) if rooms else None

    def _choose_different_room(self, graph, room):
        rooms = [node["name"] for node in graph.get("nodes", []) if node.get("type") == "room" and node["name"] != room]
        return self.rng.choice(sorted(rooms)) if rooms else None

    def _choose_fire_target(self, graph, target_room):
        candidates = []
        for node in graph.get("nodes", []):
            if node.get("type") != "object" or target_room not in node.get("rooms", []):
                continue
            if "OnFire" in node.get("available_states", []):
                candidates.append(node)
        if not candidates:
            return None
        candidates.sort(key=lambda node: (node.get("category") or "", node["id"]))
        return self.rng.choice(candidates)

    def _spawn_fire_target(self, target_room):
        record = self._record_for_category("plywood", fallback_category="book")
        spawned = self.add_task_asset(
            record=record,
            object_name=f"online_fire_target_{self._run_counter:04d}",
            target_room=target_room,
            semantic_role="anomaly_carrier",
        )
        if not spawned["ok"]:
            raise RuntimeError(f"Could not spawn a fire target: {spawned['errors']}")
        return {"name": spawned["object_name"], "category": spawned["category"]}

    def _set_boolean_state(self, object_name, state_cls, value):
        obj = self.env.scene.object_registry("name", object_name, None)
        result = {
            "ok": False,
            "object_name": object_name,
            "state": state_cls.__name__,
            "value": value,
            "errors": [],
        }
        if obj is None:
            result["errors"].append({"error": "object_missing"})
            return result
        if state_cls not in obj.states:
            result["errors"].append({"error": "state_not_available"})
            return result
        try:
            result["ok"] = bool(obj.states[state_cls].set_value(value))
            self._clear_usd_selection()
            self._step(5)
        except Exception as exc:
            result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        return result

    def _record_for_category(self, category, fallback_category=None):
        for candidate in (category, fallback_category):
            if not candidate:
                continue
            records = self._usable_records(self.asset_db.by_category(candidate))
            if records:
                return self.rng.choice(records)
            candidate_tokens = self._tokens(candidate)
            fuzzy = []
            for item_category, item_records in self.asset_db.categories.items():
                item_tokens = self._tokens(item_category)
                if candidate_tokens and (candidate_tokens <= item_tokens or candidate_tokens & item_tokens):
                    fuzzy.extend(self._usable_records(item_records))
            if fuzzy:
                return self.rng.choice(self._dedupe_records(fuzzy))
        raise ValueError(f"No usable asset metadata record found for category={category!r}")

    def save_run(self, run_result, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(run_result, f, ensure_ascii=False, indent=2, default=self._json_default)
        return output_path

    def _scene_objects(self):
        objs = []
        for attr in ("objects", "_objects"):
            if not hasattr(self.env.scene, attr):
                continue
            value = getattr(self.env.scene, attr)
            try:
                objs.extend(value.values() if isinstance(value, dict) else list(value))
            except Exception:
                pass
            if objs:
                break
        dedup = {}
        for obj in objs:
            key = getattr(obj, "prim_path", None) or getattr(obj, "name", None) or id(obj)
            dedup[key] = obj
        return list(dedup.values())

    def _collect_object_info(self, obj):
        pos, quat = self._safe_pose(obj)
        aabb_min, aabb_max, extent = self._safe_aabb(obj)
        return {
            "name": getattr(obj, "name", None),
            "category": getattr(obj, "category", None),
            "prim_path": getattr(obj, "prim_path", None),
            "rooms": self._rooms_for_obj(obj),
            "position": pos,
            "orientation_xyzw": quat,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "aabb_extent": extent,
            "available_states": self._state_names(obj),
        }

    def _safe_pose(self, obj):
        try:
            pos, quat = obj.get_position_orientation()
            return self._to_list(pos), self._to_list(quat)
        except Exception:
            return None, None

    def _safe_aabb(self, obj):
        state = self._state_by_name(obj, "AABB")
        if state is None:
            return None, None, None
        try:
            lo, hi = state.get_value()
            lo_list = self._to_list(lo)
            hi_list = self._to_list(hi)
            extent = (np.asarray(hi_list, dtype=np.float32) - np.asarray(lo_list, dtype=np.float32)).tolist()
            return lo_list, hi_list, extent
        except Exception:
            return None, None, None

    def _rooms_for_obj(self, obj):
        rooms = []
        for attr in ("in_rooms", "rooms", "room_instance", "room_type"):
            if not hasattr(obj, attr):
                continue
            try:
                value = getattr(obj, attr)
                if isinstance(value, str):
                    rooms.append(value)
                elif value:
                    rooms.extend(list(value))
            except Exception:
                pass
        return sorted({room for room in rooms if room})

    def _state_names(self, obj):
        try:
            return sorted(cls.__name__ for cls in obj.states.keys())
        except Exception:
            return []

    def _infer_object_semantic(self, obj_info):
        category = obj_info.get("category") or ""
        tokens = self._tokens(category)
        states = set(obj_info.get("available_states", []))
        extent = obj_info.get("aabb_extent")

        supports_on_top = bool(tokens & SUPPORT_SURFACE_TOKENS) or self._is_large_horizontal_surface(extent)
        supports_inside = bool(tokens & INSIDE_RECEPTACLE_TOKENS) or bool(states & {"Contains", "Filled"})
        if category == "agent":
            interaction = {"kind": "agent", "confidence": "explicit"}
        elif states & {"ToggledOn", "HeatSourceOrSink"}:
            interaction = {"kind": "controllable", "confidence": "state_inferred"}
        elif states & {"Open"} and not tokens & STRUCTURAL_CATEGORIES:
            interaction = {"kind": "articulable", "confidence": "state_inferred"}
        elif tokens & STRUCTURAL_CATEGORIES:
            interaction = {"kind": "none", "confidence": "category_inferred"}
        else:
            interaction = {"kind": "manipulable", "confidence": "default_inferred"}
        abnormal = []
        for state_name, abnormal_name in {"OnFire": "on_fire", "Burnt": "burnt", "Covered": "covered"}.items():
            if state_name in states:
                abnormal.append(abnormal_name)
        return {
            "receptacle": {
                "can_support": supports_on_top or supports_inside,
                "supports_on_top": supports_on_top,
                "supports_inside": supports_inside,
                "confidence": "online_inferred",
            },
            "interaction": interaction,
            "abnormal": {
                "potential": sorted(abnormal),
                "current": [],
                "confidence": "state_type_inferred" if abnormal else "none",
            },
        }

    def _select_task_records(self, task):
        if task:
            pool = self._usable_records(self.asset_db.by_task(task))
            if len(pool) < self.config.task_objects:
                raise ValueError(f"Task {task!r} only has {len(pool)} usable records")
        else:
            task_pool = [
                name
                for name, records in self.asset_db.tasks.items()
                if len({record["synset"] for record in self._usable_records(records)}) >= self.config.task_objects
            ]
            if not task_pool:
                raise ValueError("No task has enough usable asset records")
            task = self.rng.choice(sorted(task_pool))
            pool = self._usable_records(self.asset_db.by_task(task))
        return self._weighted_sample_records(pool, self.config.task_objects)

    def _select_context_records(self, task, selected_records):
        selected = {record["synset"] for record in selected_records}
        candidates = [record for record in self._usable_records(self.asset_db.by_task(task)) if record["synset"] not in selected]
        return self._weighted_sample_records(candidates, self.config.context_objects)

    def _usable_records(self, records):
        usable = []
        for record in records:
            if not record.get("direct_categories") or not record.get("tasks"):
                continue
            if record.get("object_type") in {"liquid", "microPhysicalSubstance", "visualSubstance"}:
                continue
            if record.get("object_type") == "cloth" and not self.config.allow_cloth:
                continue
            if not self._record_has_any_model(record):
                continue
            usable.append(record)
        return self._dedupe_records(usable)

    def _weighted_sample_records(self, records, count):
        records = self._dedupe_records(records)
        selected = []
        while records and len(selected) < count:
            weights = [max(1, self._record_score(record)) for record in records]
            record = self.rng.choices(records, weights=weights, k=1)[0]
            selected.append(record)
            records = [item for item in records if item["synset"] != record["synset"]]
        return selected

    def _record_score(self, record):
        score = min(len(record.get("tasks", [])), 10)
        interaction = (record.get("edit_metadata", {}).get("interaction") or {}).get("kind")
        if interaction in {"manipulable", "articulable", "controllable"}:
            score += 3
        if (record.get("edit_metadata", {}).get("receptacle") or {}).get("can_support"):
            score += 1
        return score

    def _choose_primary_task(self, selected_records):
        counts = Counter()
        for record in selected_records:
            for task in record.get("tasks", []):
                counts[task] += 1
        return counts.most_common(1)[0][0] if counts else "generic_home_care_task"

    def _choose_target_room(self, task, selected_records, graph):
        rooms = [node["name"] for node in graph["nodes"] if node.get("type") == "room"]
        if not rooms:
            return None
        semantic_tokens = self._tokens(task)
        for record in selected_records:
            semantic_tokens |= self._tokens(record.get("synset", ""))
            for category in record.get("direct_categories", []):
                semantic_tokens |= self._tokens(category)
        scores = Counter()
        for room in rooms:
            for key, keywords in ROOM_KEYWORDS.items():
                if key in room:
                    scores[room] += len(semantic_tokens & keywords) * 5
        for node in graph["nodes"]:
            if node.get("type") != "object":
                continue
            for room in node.get("rooms", []):
                scores[room] += 0.05
        if not scores:
            return self.rng.choice(rooms)
        max_score = max(scores.values())
        return self.rng.choice(sorted(room for room, score in scores.items() if score == max_score))

    def _choose_live_placement(self, record, target_room):
        graph = self.snapshot()
        support = self._choose_support_node(record, target_room, graph)
        mode = "on_top"
        if support:
            receptacle = ((support.get("semantic") or {}).get("receptacle") or {})
            wants_inside = (record.get("edit_metadata", {}).get("receptacle") or {}).get("supports_inside")
            if wants_inside and receptacle.get("supports_inside"):
                mode = "inside"
            elif receptacle.get("supports_on_top"):
                mode = "on_top"
            elif receptacle.get("supports_inside"):
                mode = "inside"
        pose = self._pose_near_support(support, target_room, graph)
        return {
            "room_id": target_room,
            "mode": mode,
            "support_object_id": support["id"] if support else None,
            "support_category": support.get("category") if support else None,
            "pose": {
                "position": pose,
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "pose_source": "live_support_object",
        }

    def _choose_support_node(self, record, target_room, graph):
        wants_inside = (record.get("edit_metadata", {}).get("receptacle") or {}).get("supports_inside")
        candidates = []
        for node in graph["nodes"]:
            if node.get("type") != "object" or target_room not in node.get("rooms", []):
                continue
            if not self._is_valid_support_node(node, target_room):
                continue
            receptacle = ((node.get("semantic") or {}).get("receptacle") or {})
            if not receptacle.get("can_support"):
                continue
            score = 0
            if wants_inside and receptacle.get("supports_inside"):
                score += 4
            if receptacle.get("supports_on_top"):
                score += 3
            if receptacle.get("supports_inside"):
                score += 2
            score += self._support_preference_score(node, target_room)
            candidates.append((score, node))
        if not candidates:
            return None
        best_score = max(score for score, _ in candidates)
        return self.rng.choice(sorted([node for score, node in candidates if score == best_score], key=lambda n: n["id"]))

    def _apply_relation(self, obj, placement):
        support_id = placement.get("support_object_id")
        if not support_id:
            return {"ok": False, "mode": placement.get("mode"), "reason": "no_support_object"}
        support_obj = self.env.scene.object_registry("name", support_id, None)
        if support_obj is None:
            return {"ok": False, "mode": placement.get("mode"), "support": support_id, "reason": "support_missing"}
        state_cls = object_states.Inside if placement.get("mode") == "inside" else object_states.OnTop
        if state_cls not in obj.states:
            return {
                "ok": False,
                "mode": placement.get("mode"),
                "support": support_id,
                "state": state_cls.__name__,
                "reason": "state_not_available",
            }
        try:
            ok = bool(obj.states[state_cls].set_value(support_obj, True, reset_before_sampling=True))
            return {
                "ok": ok,
                "mode": placement.get("mode"),
                "support": support_id,
                "state": state_cls.__name__,
            }
        except Exception as exc:
            return {
                "ok": False,
                "mode": placement.get("mode"),
                "support": support_id,
                "state": state_cls.__name__,
                "error": repr(exc),
            }

    def _collect_settling_report(self, object_names):
        before = {name: self._object_position(name) for name in object_names}
        self._step(self.config.settle_steps)
        objects = []
        all_within = True
        for name, start in before.items():
            end = self._object_position(name)
            if start is None or end is None:
                objects.append({"object_name": name, "ok": False, "error": "missing_pose"})
                all_within = False
                continue
            displacement = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
            within = displacement <= self.config.settle_threshold
            all_within = all_within and within
            objects.append(
                {
                    "object_name": name,
                    "start_position": start,
                    "end_position": end,
                    "displacement": displacement,
                    "within_threshold": within,
                }
            )
        return {
            "settle_steps": self.config.settle_steps,
            "settle_threshold": self.config.settle_threshold,
            "all_within_threshold": all_within,
            "objects": objects,
        }

    def _build_task_instance(self, run_id, primary_task, target_room, created_objects):
        task_objects = [item for item in created_objects if item.get("semantic_role") == "task_object"]
        plan = [{"step_id": 1, "primitive": "MOVE", "nl": f"Move to {target_room}", "target_room": target_room}]
        step_id = 2
        for item in task_objects:
            plan.append(
                {
                    "step_id": step_id,
                    "primitive": "MOVE",
                    "nl": f"Move to {item['category'].replace('_', ' ')}",
                    "target_object": item["object_name"],
                    "target_room": target_room,
                }
            )
            step_id += 1
            plan.append(
                {
                    "step_id": step_id,
                    "primitive": "INTERACT",
                    "nl": f"Interact with {item['category'].replace('_', ' ')}",
                    "target_object": item["object_name"],
                    "inventory": [],
                }
            )
            step_id += 1
        return {
            "task_id": f"{run_id}_task",
            "task_type": "Env-A",
            "instruction": f"Complete the {primary_task.replace('_', ' ')} task in {target_room}.",
            "target_room": target_room,
            "task_objects": [
                {
                    "object_id": item["object_name"],
                    "category": item["category"],
                    "synset": item["synset"],
                }
                for item in task_objects
            ],
            "solution_plan": plan,
        }

    def _build_task_environment_record(
        self,
        env_id,
        env_type,
        task_instance,
        target_room,
        created_objects,
        validation,
        delta_sg,
        graph,
        state_changed_objects=None,
    ):
        scene_model = self._scene_model()
        added_objects = self._standard_added_objects(created_objects, validation)
        state_changed_objects = state_changed_objects or []
        return {
            "schema_version": "task_environment.v1",
            "env_id": env_id,
            "env_type": env_type,
            "base_scene": {
                "scene_model": scene_model,
                "base_env_usd": f"{scene_model}.usd" if scene_model else None,
                "source": "live_omnigibson_env",
            },
            "task": {
                "task_id": task_instance["task_id"],
                "task_type": task_instance["task_type"],
                "instruction": task_instance["instruction"],
                "target_room": target_room,
                "semantic_constraints": task_instance.get("semantic_constraints", []),
            },
            "robot": self._robot_record(target_room, graph),
            "camera": self._camera_records(target_room, graph),
            "added_objects": added_objects,
            "state_changed_objects": state_changed_objects,
            "task_objects": task_instance.get("task_objects", []),
            "delta_sg": delta_sg,
            "validation": self._standard_validation(validation),
            "solution_plan": task_instance.get("solution_plan", []),
            "semantic_reasoning": task_instance.get("semantic_reasoning"),
            "storage": {
                "format": "base_scene_plus_delta_json",
                "replay_order": [
                    "load base_scene",
                    "spawn added_objects",
                    "apply state_changed_objects",
                    "apply delta_sg relations",
                    "warmup and validate",
                ],
            },
        }

    def _standard_added_objects(self, created_objects, validation):
        settling = ((validation or {}).get("settling") or {}).get("objects", [])
        settled_by_name = {item.get("object_name"): item for item in settling}
        added = []
        for item in created_objects:
            pose = item.get("final_pose_before_warmup") or {}
            settled = settled_by_name.get(item.get("object_name"), {})
            added.append(
                {
                    "object_id": item.get("object_name"),
                    "object_name": item.get("object_name"),
                    "category": item.get("category"),
                    "synset": item.get("synset"),
                    "model": item.get("model"),
                    "semantic_roles": [item.get("semantic_role")] if item.get("semantic_role") else [],
                    "room_id": (item.get("placement") or {}).get("room_id"),
                    "placement": {
                        "mode": (item.get("placement") or {}).get("mode"),
                        "support_object_id": (item.get("placement") or {}).get("support_object_id"),
                        "support_category": (item.get("placement") or {}).get("support_category"),
                        "pose": (item.get("placement") or {}).get("pose"),
                        "pose_source": (item.get("placement") or {}).get("pose_source"),
                    },
                    "pose": {
                        "position": settled.get("end_position") or pose.get("position"),
                        "orientation_xyzw": pose.get("orientation_xyzw"),
                    },
                    "states": {},
                    "relation": item.get("relation"),
                    "validation": {
                        "spawn_ok": item.get("ok", False),
                        "settling": settled or None,
                    },
                }
            )
        return added

    def _standard_validation(self, validation):
        if not isinstance(validation, dict):
            return {"ok": False}
        return {
            "ok": validation.get("ok", False),
            "num_created_objects": len(validation.get("created_objects", []))
            if isinstance(validation.get("created_objects"), list)
            else None,
            "num_failed_objects": len(validation.get("failed_objects", []))
            if isinstance(validation.get("failed_objects"), list)
            else None,
            "settling": validation.get("settling"),
            "failure_summary": [
                {
                    "object_name": item.get("object_name"),
                    "category": item.get("category"),
                    "synset": item.get("synset"),
                    "errors": item.get("errors", []),
                }
                for item in validation.get("failed_objects", [])
            ]
            if isinstance(validation.get("failed_objects"), list)
            else [],
        }

    def _scene_model(self):
        for attr in ("scene_model", "model", "_scene_model"):
            if hasattr(self.env.scene, attr):
                try:
                    value = getattr(self.env.scene, attr)
                    if value:
                        return str(value)
                except Exception:
                    pass
        return None

    def _robot_record(self, target_room, graph):
        if not getattr(self.env, "robots", None):
            return None
        robot = self.env.robots[0]
        pose = self._object_pose_record(robot)
        rooms = self._rooms_for_obj(robot)
        initial_room = rooms[0] if rooms else self._nearest_room(pose.get("position"), graph)
        return {
            "robot_id": getattr(robot, "name", "robot_0"),
            "model": getattr(robot, "model_name", None) or getattr(robot, "model", None),
            "initial_room": initial_room,
            "target_room": target_room,
            "pose": pose,
            "navigation_hint": self._room_path(graph, initial_room, target_room) if initial_room else None,
        }

    def _camera_records(self, target_room, graph):
        cameras = []
        if getattr(self.env, "robots", None):
            robot = self.env.robots[0]
            sensors = getattr(robot, "sensors", None) or getattr(robot, "_sensors", None) or {}
            if isinstance(sensors, dict):
                sensor_items = sensors.items()
            else:
                try:
                    sensor_items = [(getattr(sensor, "name", f"sensor_{idx}"), sensor) for idx, sensor in enumerate(sensors)]
                except Exception:
                    sensor_items = []
            for name, sensor in sensor_items:
                if not self._looks_like_camera_sensor(sensor):
                    continue
                cameras.append(
                    {
                        "camera_id": str(name),
                        "camera_type": "robot_camera",
                        "attached_to": getattr(robot, "name", "robot_0"),
                        "room_id": None,
                        "pose": self._object_pose_record(sensor),
                        "modalities": self._sensor_modalities(sensor),
                        "resolution": self._sensor_resolution(sensor),
                    }
                )

        center = (graph.get("navigation") or {}).get("room_centers", {}).get(target_room)
        if center:
            cameras.append(
                {
                    "camera_id": f"planned_global_{target_room}",
                    "camera_type": "global_camera_plan",
                    "room_id": target_room,
                    "pose": {
                        "position": [float(center[0]), float(center[1]), 2.4],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "modalities": ["rgb", "depth", "seg_semantic"],
                    "resolution": None,
                    "status": "planned_not_spawned",
                }
            )
        return cameras

    def _object_pose_record(self, obj):
        try:
            pos, quat = obj.get_position_orientation()
            return {"position": self._to_list(pos), "orientation_xyzw": self._to_list(quat)}
        except Exception:
            return {"position": None, "orientation_xyzw": None}

    def _looks_like_camera_sensor(self, sensor):
        name = str(getattr(sensor, "name", "")).lower()
        class_name = sensor.__class__.__name__.lower()
        return "camera" in name or "vision" in class_name or hasattr(sensor, "image_height")

    def _sensor_modalities(self, sensor):
        for attr in ("modalities", "obs_modalities", "_modalities", "_obs_modalities"):
            if hasattr(sensor, attr):
                try:
                    value = getattr(sensor, attr)
                    return sorted(list(value))
                except Exception:
                    pass
        return None

    def _sensor_resolution(self, sensor):
        height = getattr(sensor, "image_height", None)
        width = getattr(sensor, "image_width", None)
        if height is None or width is None:
            return None
        return {"height": int(height), "width": int(width)}

    def _nearest_room(self, position, graph):
        if not position or len(position) < 2:
            return None
        centers = (graph.get("navigation") or {}).get("room_centers", {})
        best = None
        for room, center in centers.items():
            if not center:
                continue
            dx = float(position[0]) - float(center[0])
            dy = float(position[1]) - float(center[1])
            candidate = (dx * dx + dy * dy, room)
            if best is None or candidate < best:
                best = candidate
        return best[1] if best else None

    def _room_path(self, graph, start_room, target_room):
        if start_room == target_room:
            return {"rooms": [target_room], "distance": 0.0}
        return (
            (graph.get("navigation") or {})
            .get("shortest_room_paths", {})
            .get(start_room, {})
            .get(target_room)
        )

    def _delta_edges_for_added_object(self, object_name, target_room, placement, relation):
        edges = [{"source": f"room::{target_room}", "target": object_name, "relation": "contains"}]
        support_id = placement.get("support_object_id")
        if support_id:
            edges.append(
                {
                    "source": support_id,
                    "target": object_name,
                    "relation": "supports" if placement.get("mode") == "on_top" else "contains",
                    "mode": placement.get("mode"),
                    "physical_relation_ok": relation.get("ok", False),
                }
            )
        return edges

    def _object_position(self, name):
        obj = self.env.scene.object_registry("name", name, None)
        if obj is None:
            return None
        try:
            pos, _ = obj.get_position_orientation()
            return self._to_list(pos)
        except Exception:
            return None

    def _choose_category(self, record):
        categories = [category for category in record.get("direct_categories", []) if self._category_has_models(category)]
        if not categories:
            return record["synset"].split(".")[0].replace("__", "_")
        return self.rng.choice(sorted(categories))

    def _pose_near_support(self, support, target_room, graph):
        if support:
            pos = (support.get("pose") or {}).get("position")
            bbox = support.get("bbox") or {}
            if bbox.get("max"):
                return [float(pos[0]), float(pos[1]), float(bbox["max"][2]) + 0.15]
            if pos:
                return [float(pos[0]), float(pos[1]), float(pos[2]) + 0.3]
        room_center = (graph.get("navigation") or {}).get("room_centers", {}).get(target_room)
        if room_center:
            return [float(room_center[0]), float(room_center[1]), 0.8]
        return [0.0, 0.0, 0.8]

    def _record_has_any_model(self, record):
        return any(self._category_has_models(category) for category in record.get("direct_categories", []))

    def _category_has_models(self, category):
        if not category:
            return False
        if category not in self._category_models_cache:
            try:
                self._category_models_cache[category] = len(get_all_object_category_models(category=category)) > 0
            except Exception:
                self._category_models_cache[category] = False
        return self._category_models_cache[category]

    def _is_valid_support_node(self, node, target_room):
        category = node.get("category") or ""
        tokens = self._tokens(category)
        if tokens & BAD_SUPPORT_TOKENS:
            return False
        bbox = node.get("bbox") or {}
        extent = bbox.get("extent")
        pose = node.get("pose") or {}
        pos = pose.get("position")
        if not extent or len(extent) < 3 or not pos or len(pos) < 3:
            return False
        x, y, z = [float(v) for v in extent[:3]]
        if max(x, y) < 0.18:
            return False
        if float(pos[2]) > 1.8:
            return False
        room_type = str(target_room).rsplit("_", 1)[0]
        preferred = GOOD_SUPPORT_TOKENS.get(room_type, set())
        if preferred and tokens & preferred:
            return True
        receptacle = ((node.get("semantic") or {}).get("receptacle") or {})
        return bool(receptacle.get("supports_on_top") and self._is_large_horizontal_surface(extent))

    def _support_preference_score(self, node, target_room):
        category = node.get("category") or ""
        tokens = self._tokens(category)
        room_type = str(target_room).rsplit("_", 1)[0]
        score = 0
        score += 5 * len(tokens & GOOD_SUPPORT_TOKENS.get(room_type, set()))
        if "floor" in tokens:
            score -= 1
        bbox = node.get("bbox") or {}
        extent = bbox.get("extent")
        if extent and len(extent) >= 3:
            x, y, z = [float(v) for v in extent[:3]]
            if x >= 0.35 and y >= 0.35:
                score += 2
            if z > 1.2:
                score -= 1
        return score

    def _group_by_room(self, object_nodes):
        grouped = defaultdict(list)
        for node in object_nodes:
            for room in node.get("rooms", []):
                grouped[room].append(node)
        return grouped

    def _spatial_relation(self, src, dst):
        src_center = self._center(src)
        dst_center = self._center(dst)
        if src_center is None or dst_center is None:
            return None
        dx = src_center[0] - dst_center[0]
        dy = src_center[1] - dst_center[1]
        xy_dist = float(np.sqrt(dx * dx + dy * dy))
        src_min, src_max = src["bbox"].get("min"), src["bbox"].get("max")
        dst_min, dst_max = dst["bbox"].get("min"), dst["bbox"].get("max")
        if src_min and src_max and dst_min and dst_max and self._xy_overlap(src, dst):
            if abs(float(dst_min[2]) - float(src_max[2])) <= self.config.support_z_tolerance:
                return {"relation": "supports_candidate", "mode": "on_top", "distance": xy_dist}
            if abs(float(src_min[2]) - float(dst_max[2])) <= self.config.support_z_tolerance:
                return {"relation": "supported_by_candidate", "mode": "on_top", "distance": xy_dist}
        if xy_dist <= self.config.near_distance:
            return {"relation": "near", "distance": xy_dist}
        return None

    def _build_navigation(self, rooms, object_nodes):
        room_centers = self._estimate_room_centers(rooms, object_nodes)
        adjacency = defaultdict(dict)
        edges = []
        room_list = list(rooms)
        for idx, src in enumerate(room_list):
            for dst in room_list[idx + 1 :]:
                dist = self._room_distance(room_centers, src, dst)
                if dist is None:
                    continue
                adjacency[src][dst] = {"distance": dist, "mode": "centroid_route_candidate"}
                adjacency[dst][src] = {"distance": dist, "mode": "centroid_route_candidate"}
                edges.append(
                    {
                        "source": f"room::{src}",
                        "target": f"room::{dst}",
                        "relation": "room_route_candidate",
                        "mode": "centroid_route_candidate",
                        "distance": dist,
                    }
                )
        return {
            "edges": edges,
            "navigation": {
                "room_centers": room_centers,
                "room_edges": [
                    {"source": src, "target": dst, **meta}
                    for src, dsts in sorted(adjacency.items())
                    for dst, meta in sorted(dsts.items())
                    if src < dst
                ],
                "shortest_room_paths": {
                    src: {dst: self._shortest_room_path(src, dst, adjacency) for dst in rooms if dst != src}
                    for src in rooms
                },
            },
        }

    def _estimate_room_centers(self, rooms, object_nodes):
        centers = defaultdict(list)
        for node in object_nodes:
            if self._tokens(node.get("category") or "") & STRUCTURAL_CATEGORIES:
                continue
            center = self._center(node)
            if center is None:
                continue
            for room in node.get("rooms", []):
                if room in rooms:
                    centers[room].append(center)
        result = {}
        for room in rooms:
            points = centers.get(room, [])
            result[room] = [sum(p[axis] for p in points) / len(points) for axis in range(3)] if points else None
        return result

    def _shortest_room_path(self, src, dst, adjacency):
        frontier = [(0.0, src, [])]
        seen = set()
        while frontier:
            frontier.sort(key=lambda item: item[0])
            cost, room, path = frontier.pop(0)
            if room in seen:
                continue
            seen.add(room)
            next_path = path + [room]
            if room == dst:
                return {"rooms": next_path, "distance": cost}
            for neighbor, meta in adjacency.get(room, {}).items():
                if neighbor not in seen:
                    frontier.append((cost + (meta.get("distance") or 1.0), neighbor, next_path))
        return None

    def _room_distance(self, room_centers, src, dst):
        src_center = room_centers.get(src)
        dst_center = room_centers.get(dst)
        if src_center is None or dst_center is None:
            return None
        dx = src_center[0] - dst_center[0]
        dy = src_center[1] - dst_center[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _center(self, node):
        pos = node["pose"].get("position")
        if pos and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        bbox_min = node["bbox"].get("min")
        bbox_max = node["bbox"].get("max")
        if bbox_min and bbox_max:
            return [(float(lo) + float(hi)) * 0.5 for lo, hi in zip(bbox_min[:3], bbox_max[:3])]
        return None

    def _xy_overlap(self, src, dst):
        src_min, src_max = src["bbox"].get("min"), src["bbox"].get("max")
        dst_min, dst_max = dst["bbox"].get("min"), dst["bbox"].get("max")
        return (
            min(float(src_max[0]), float(dst_max[0])) > max(float(src_min[0]), float(dst_min[0]))
            and min(float(src_max[1]), float(dst_max[1])) > max(float(src_min[1]), float(dst_min[1]))
        )

    def _is_large_horizontal_surface(self, extent):
        if not extent or len(extent) < 3:
            return False
        x, y, z = [float(v) for v in extent[:3]]
        return x >= 0.35 and y >= 0.35 and z <= max(x, y) * 0.6

    def _state_by_name(self, obj, state_name):
        try:
            for cls, state in obj.states.items():
                if cls.__name__ == state_name:
                    return state
        except Exception:
            return None
        return None

    def _dedupe_records(self, records):
        dedup = {}
        for record in records:
            dedup.setdefault(record["synset"], record)
        return list(dedup.values())

    def _step(self, n_steps):
        for _ in range(n_steps):
            og.sim.step()

    def _clear_usd_selection(self):
        try:
            selection = lazy.omni.usd.get_context().get_selection()
            for method_name in ("clear_selected_prim_paths", "clear_selection"):
                method = getattr(selection, method_name, None)
                if method is not None:
                    method()
                    return
            set_selected = getattr(selection, "set_selected_prim_paths", None)
            if set_selected is not None:
                set_selected([], False)
        except Exception:
            pass

    def _tokens(self, text):
        return {token for token in re.split(r"[_\-\W]+", str(text).lower()) if token}

    def _slug(self, text):
        text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).lower())
        return re.sub(r"_+", "_", text).strip("_") or "object"

    def _to_list(self, value):
        if isinstance(value, th.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return list(value)

    def _json_default(self, value):
        if isinstance(value, th.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
