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
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.constants import PrimType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_client import create_llm_client
import llm_client as llm_prompts

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


# ================================================================
# Env-A Task Categories (from intro.md 2026/6/16)
# ================================================================
# Env-A focuses on Retrieval & Delivery and Open/Close only.
# Other categories (anomaly, appliance, cleaning, etc.) go to Env-B/C.
VALID_TASKS: dict[str, set[str]] = {
    # Retrieval & Delivery
    "retrieval_delivery": {
        "retrieve_medicine", "retrieve_key", "retrieve_remote",
        "retrieve_phone", "retrieve_book", "retrieve_drink",
        "retrieve_food", "deliver_medicine", "deliver_food",
        "deliver_drink", "put_object_on_table", "put_object_in_container",
    },
    # Open / Close
    "open_close": {
        "open_door", "close_door", "open_window", "close_window",
        "open_fridge", "close_fridge", "open_cabinet", "close_cabinet",
    },
    # Appliance (switch on/off)
    "appliance": {
        "turn_on_light", "turn_off_light", "turn_on_tv",
        "turn_off_tv", "turn_on_stove", "turn_off_stove",
    },
}

# Flattened set of all task names for quick lookup
ALL_VALID_TASK_NAMES: set[str] = {t for tasks in VALID_TASKS.values() for t in tasks}


@dataclass
class OnlineDeltaSGConfig:
    task_objects: int = 2
    context_objects: int = 3
    warmup_steps: int = 20
    settle_steps: int = 5
    settle_threshold: float = 0.25
    near_distance: float = 1.25
    support_z_tolerance: float = 0.08
    seed: int | None = None
    allow_cloth: bool = False
    max_fallback_supports: int = 2
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    # ---- Retry control ----
    max_llm_retries_per_scene: int = 3
    max_retries_per_task: int = 1
    max_total_generation_time_sec: float = 900.0

    # ---- Placement fail-fast ----
    per_object_placement_timeout_sec: float = 90.0
    per_relation_attempt_timeout_sec: float = 15.0
    max_placement_attempts_per_object: int = 4
    max_total_placement_time_sec: float = 180.0

    # ---- Context object behavior ----
    abort_on_task_object_failure: bool = True
    skip_context_on_failure: bool = True


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
        if self.config.seed is None:
            self.config.seed = random.randint(0, 2**31)
        self.rng = random.Random(self.config.seed)
        print(f"[online-deltasg] seed={self.config.seed}")
        self.asset_db = TaskAssetDatabase.load(metadata_dir=metadata_dir or TaskAssetDatabase().metadata_dir)
        self._run_counter = 0
        self._category_models_cache = {}
        self._placed_on_support = {}
        self._placement_support_map = {}
        self._support_occupied_area = {}  # support_id → occupied XY area (m²)
        self._scene_categories_cache = None  # lazily computed set of categories in scene
        # Retry / fail-fast state
        self._rejected_task_cache: set[str] = set()
        self._failed_placement_cache: set[tuple[str, str]] = set()  # (category, support_id) pairs
        self._llm_retry_count: int = 0
        self._task_placement_start_time: float = 0.0
        self._scene_start_time: float = 0.0
        self._checkpoint: dict = {
            "attempted_tasks": [],
            "rejected_tasks": [],
            "failed_placements": [],
            "successful_samples": [],
        }
        self._enabled_categories: set[str] | None = None  # None = all categories
        self._llm_client = create_llm_client(
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
        )
        if self._llm_client:
            print(f"[online-deltasg] LLM enabled: {self._llm_client.model}")

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

    def generate_env_a(self, task=None, target_room=None, skip_tasks=None):
        """
        Create an Env-A task scene online and physically validate it.

        New paradigm (2026/6/16): task category → required objects → scene.
        Instead of sampling random BEHAVIOR tasks, we first pick a task
        category from VALID_TASKS, determine the minimum required objects,
        then place them into the scene.

        Args:
            task: optional explicit task name from VALID_TASKS.
            target_room: optional explicit room id.
            skip_tasks: optional set of task names to exclude from selection.
        """
        self._placed_on_support = {}
        self._placement_support_map = {}
        self._support_occupied_area = {}
        self._scene_categories_cache = None  # refresh scene categories
        # Save full scene state so we can restore after this run
        self._cleanup_spawned_objects()
        self._pre_run_state = og.sim.dump_state(serialized=False)
        before_graph = self.snapshot()

        # ---- Step 1: Select task category and required objects ----
        if task and task in ALL_VALID_TASK_NAMES:
            primary_task = task
            task_category = self._category_for_task(task)
        elif task:
            # Legacy BEHAVIOR task name — use old path
            primary_task = task
            task_category = None
        else:
            # New paradigm: pick category → pick task → find objects
            task_category, primary_task = self._select_task_and_category(
                skip_tasks=skip_tasks, cached_graph=before_graph,
            )
            if not primary_task:
                return self._build_llm_rejected_result(
                    llm_validation={"issues": ["no_valid_task_found"]},
                    before_graph=before_graph,
                    primary_task="unknown",
                    target_room=target_room or "kitchen_0",
                    hard_reject=True,
                )

        target_room = target_room or self._choose_target_room(primary_task, [], before_graph)

        # ---- Step 2: Find required objects via LLM ----
        if self._llm_client and task_category:
            scene_furniture = self._build_scene_furniture_dict(before_graph)
            required = llm_prompts.find_required_objects(
                client=self._llm_client,
                task_name=primary_task,
                task_category=task_category,
                scene_furniture=scene_furniture,
            )
            if required and required.get("objects"):
                required_objs = required["objects"]
                reasoning = required.get("reasoning", "")
                obj_names = [o.get("name", "?") for o in required_objs]
                print(f"[llm] required objects ({len(required_objs)}): {', '.join(obj_names)}")
                if reasoning:
                    print(f"[llm] reasoning: {reasoning[:200]}")
            else:
                required_objs = []
        else:
            required_objs = []

        # ---- Step 3: Match required objects to asset database records ----
        selected_records = self._match_required_objects(required_objs)

        # If no objects matched and no scene-native targets, reject the task
        if not selected_records and not any(
            robj.get("role") in ("tool", "support") or
            robj.get("category_hint", "") in self._get_scene_categories()
            for robj in required_objs
        ):
            if task_category not in ("appliance", "open_close"):
                print(f"[match] task '{primary_task}' has no matchable objects, rejecting", flush=True)
                self._rejected_task_cache.add(primary_task)
                return self._build_llm_rejected_result(
                    llm_validation={"issues": ["no_matchable_objects"]},
                    before_graph=before_graph,
                    primary_task=primary_task,
                    target_room=target_room,
                    hard_reject=True,
                )

        # For appliance/open_close tasks, the target is already in the scene.
        # Don't add context objects — they just add noise.
        if task_category in ("appliance", "open_close"):
            context_records = []
        else:
            context_records = self._select_context_records_for_objects(selected_records, before_graph)
        records = self._dedupe_records(selected_records + context_records)

        # LLM task feasibility validation
        generated_instruction = None
        if self._llm_client:
            llm_validation, generated_instruction = self._llm_validate_task(
                primary_task, selected_records, context_records, target_room, before_graph,
                task_category=task_category,
            )
            if llm_validation is not None:
                feasible = llm_validation.get("feasible", True)
                is_linear = llm_validation.get("is_linear", True)
                issues = llm_validation.get("issues", [])
                # Check for hard-reject keywords — these are unfixable for this task
                hard_reject = self._is_hard_reject(issues)
                if not feasible or not is_linear:
                    rejection_kind = "hard-reject" if hard_reject else "soft-reject"
                    if not is_linear and feasible:
                        print(f"[llm] task is feasible but NOT linear — {rejection_kind}")
                    else:
                        print(f"[llm] task not feasible — {rejection_kind}: {issues}")
                    # Always add to rejected task cache
                    self._rejected_task_cache.add(primary_task)
                    self._checkpoint["rejected_tasks"].append({
                        "task": primary_task, "issues": issues, "kind": rejection_kind,
                    })
                    return self._build_llm_rejected_result(
                        llm_validation=llm_validation,
                        before_graph=before_graph,
                        primary_task=primary_task,
                        target_room=target_room,
                        generated_instruction=generated_instruction,
                        hard_reject=hard_reject,
                    )

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
        import time as _time
        self._task_placement_start_time = _time.time()
        abort_task = False
        abort_reason = None

        for idx, record in enumerate(records):
            role = "task_object" if record in selected_records else "context_object"
            cat = self._choose_category(record)
            obj_name = f"{run_id}_{idx:03d}_{self._slug(cat)}"

            # Pre-check: skip if this is a context object and we already failed task objects
            if abort_task and role == "context_object":
                if self.config.skip_context_on_failure:
                    print(f"[placement] ({idx + 1}/{len(records)}) SKIP context {cat} "
                          f"(task aborted: {abort_reason})", flush=True)
                    validation["failed_objects"].append({
                        "ok": False,
                        "object_name": obj_name,
                        "category": cat,
                        "semantic_role": role,
                        "errors": [{"error": "skipped_context_after_task_abort", "reason": abort_reason}],
                    })
                    continue
                else:
                    print(f"[placement] ({idx + 1}/{len(records)}) placing context {cat} "
                          f"(task aborted but continuing per config)", flush=True)

            # Check total placement time budget
            placement_elapsed = _time.time() - self._task_placement_start_time
            if placement_elapsed > self.config.max_total_placement_time_sec:
                print(f"[placement-timeout] total placement time {placement_elapsed:.1f}s "
                      f"> max {self.config.max_total_placement_time_sec}s, aborting", flush=True)
                abort_task = True
                abort_reason = "total_placement_timeout"
                validation["failed_objects"].append({
                    "ok": False,
                    "object_name": obj_name,
                    "category": cat,
                    "semantic_role": role,
                    "errors": [{"error": "total_placement_timeout",
                               "elapsed": placement_elapsed,
                               "limit": self.config.max_total_placement_time_sec}],
                })
                if role == "task_object" and self.config.abort_on_task_object_failure:
                    break
                continue

            t0 = _time.time()
            print(f"[placement] ({idx + 1}/{len(records)}) placing {cat} as {role}...",
                  end=" ", flush=True)
            add_result = self.add_task_asset(
                record=record,
                object_name=obj_name,
                target_room=target_room,
                semantic_role=role,
                cached_graph=before_graph,
            )
            elapsed = _time.time() - t0
            if add_result["ok"]:
                if not add_result.get("reused"):
                    created_names.append(add_result["object_name"])
                validation["created_objects"].append(add_result)
                delta["nodes"].append(add_result["delta_node"])
                delta["edges"].extend(add_result["delta_edges"])
                mode = (add_result.get("placement") or {}).get("mode", "?")
                support = (add_result.get("placement") or {}).get("support_object_id", "none")
                if add_result.get("reused"):
                    existing_name = add_result.get("reused_object_name", "?")
                    print(f"REUSED (existing={existing_name}) [{elapsed:.1f}s]", flush=True)
                else:
                    print(f"OK (mode={mode}, support={support}) [{elapsed:.1f}s]", flush=True)
            else:
                validation["failed_objects"].append(add_result)
                errors = [e.get("error", str(e)[:60]) for e in add_result.get("errors", [])[:3]]
                print(f"FAILED ({errors}) [{elapsed:.1f}s]", flush=True)
                # Track failed placement pair
                support_id = (add_result.get("placement") or {}).get("support_object_id", "__none__")
                self._failed_placement_cache.add((cat, support_id))
                self._checkpoint["failed_placements"].append({
                    "category": cat, "support": support_id, "errors": errors,
                })
                # For task objects: don't abort immediately — try remaining task objects.
                # Only abort if we finish all task objects and none succeeded.
                if role == "task_object" and self.config.abort_on_task_object_failure:
                    # Check if any remaining objects are also task_objects
                    remaining_task = any(
                        r in selected_records
                        for r in records[idx + 1:]
                    )
                    if not remaining_task:
                        # This was the last task object — check if any succeeded
                        any_task_ok = any(
                            o.get("ok") and o.get("semantic_role") == "task_object"
                            for o in validation["created_objects"]
                        )
                        if not any_task_ok:
                            abort_task = True
                            abort_reason = "all_task_objects_failed"
                            print(f"[abort-task] all task objects failed, "
                                  f"aborting context placements", flush=True)

        print(f"[placement] done: {len(created_names)} created, "
              f"{len(validation['failed_objects'])} failed"
              f"{' (ABORTED: ' + abort_reason + ')' if abort_task else ''}. "
              f"Running {self.config.warmup_steps} warmup steps...", flush=True)
        # Track attempted task in checkpoint
        self._checkpoint["attempted_tasks"].append({
            "task": primary_task, "aborted": abort_task, "abort_reason": abort_reason,
        })
        t0 = _time.time()
        # Keep all placed objects still, then release one by one to prevent clipping
        placed_objs = []
        for name in created_names:
            obj = self.env.scene.object_registry("name", name, None)
            if obj is not None:
                try:
                    obj.keep_still()
                    placed_objs.append(obj)
                except Exception:
                    pass
        # First half: all objects frozen, let scene settle
        self._step(max(self.config.warmup_steps // 2, 5))
        # Release objects one at a time with small gaps
        for obj in placed_objs:
            try:
                obj.wake()
            except Exception:
                pass
            self._step(2)  # 2 steps between each release to prevent collisions
        # Final settling
        self._step(3)
        print(f"[placement] warmup done [{_time.time()-t0:.1f}s]. Collecting settling report...", flush=True)
        t0 = _time.time()
        validation["settling"] = self._collect_settling_report(created_names)
        print(f"[placement] settling done [{_time.time()-t0:.1f}s].", flush=True)

        # Validation: succeed if at least 1 task_object was placed.
        # Failed objects are acceptable — the LLM already regenerated the
        # instruction to only reference successfully placed objects.
        placed_task_objects = [
            o for o in validation["created_objects"]
            if o.get("semantic_role") == "task_object"
        ]
        placed_context_objects = [
            o for o in validation["created_objects"]
            if o.get("semantic_role") != "task_object"
        ]
        failed_task_objects = [
            o for o in validation["failed_objects"]
            if o.get("semantic_role") == "task_object"
        ]
        settling_ok = validation["settling"]["all_within_threshold"]

        if len(placed_task_objects) == 0 and not selected_records:
            # No task objects needed — the target is already in the scene
            # (e.g., turn_on_tv, open_cabinet — the appliance/fixture is pre-existing)
            validation["ok"] = True
            print(f"[placement] no task objects needed (scene-native targets used)", flush=True)
        elif len(placed_task_objects) == 0:
            validation["ok"] = False
            print(f"[placement] FAILED: no task_objects placed successfully", flush=True)
        else:
            validation["ok"] = True
            if failed_task_objects:
                print(f"[placement] PARTIAL: {len(placed_task_objects)} task + "
                      f"{len(placed_context_objects)} context placed, "
                      f"{len(failed_task_objects)} task objects skipped", flush=True)
            if not settling_ok:
                unstable = [o["object_name"] for o in validation["settling"]["objects"]
                           if not o.get("within_threshold", True)]
                print(f"[placement] WARNING: settling issues on {unstable} "
                      f"(proceeding anyway)", flush=True)
        after_graph = self.snapshot()

        # Post-placement: check if any task_object failed to place
        failed_task_objects = [
            fo for fo in validation.get("failed_objects", [])
            if fo.get("semantic_role") == "task_object"
        ]
        if failed_task_objects and self._llm_client:
            # Some task objects are missing — regenerate instruction with placed objects only
            placed_task_objs = [
                {
                    "synset": o.get("synset", ""),
                    "category": o.get("category", ""),
                    "role": "task_object",
                }
                for o in validation["created_objects"]
                if o.get("semantic_role") == "task_object"
            ]
            if placed_task_objs:
                failed_categories = {
                    fo.get("category", "").lower().replace("_", " ")
                    for fo in failed_task_objects
                }
                regen = llm_prompts.generate_instruction(
                    client=self._llm_client,
                    task_name=primary_task,
                    task_objects=placed_task_objs,
                    target_room=target_room,
                    task_category=task_category,
                )
                if regen and regen.get("instruction"):
                    new_inst = regen["instruction"]
                    # Check if the regenerated instruction still references failed objects
                    inst_lower = new_inst.lower()
                    refs_failed = [c for c in failed_categories if c and c in inst_lower]
                    if refs_failed:
                        print(f"[llm] regenerated instruction still references failed objects: {refs_failed}")
                        print(f"[llm] regenerating with explicit exclusion...")
                        # Regenerate again, passing excluded objects info
                        regen2 = llm_prompts.generate_instruction(
                            client=self._llm_client,
                            task_name=f"{primary_task} (without {', '.join(refs_failed)})",
                            task_objects=placed_task_objs,
                            target_room=target_room,
                            task_category=task_category,
                        )
                        if regen2 and regen2.get("instruction"):
                            new_inst = regen2["instruction"]
                    old_inst = generated_instruction
                    generated_instruction = new_inst
                    print(f"[llm] instruction regenerated (task objects failed):")
                    print(f"  old: {old_inst}")
                    print(f"  new: {generated_instruction}")
                    failed_names = [fo.get("category", "?") for fo in failed_task_objects]
                    print(f"  failed objects excluded: {failed_names}")

        task_instance = self._build_task_instance(
            run_id, primary_task, target_room, validation["created_objects"],
            generated_instruction, required_objs=required_objs,
            task_category=task_category,
        )
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
        self._placed_on_support = {}
        self._placement_support_map = {}
        self._cleanup_spawned_objects()
        self._pre_run_state = og.sim.dump_state(serialized=False)
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
        self._cleanup_spawned_objects()
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

    def add_task_asset(self, record, object_name, target_room, semantic_role="task_object", cached_graph=None):
        """Add one task asset directly to the live OmniGibson scene.

        Applies placement attempts with per-object timeout, placement cache
        checks, and pre-validation.  If all attempts fail, the object is
        removed from the scene with cleanup.

        Args:
            cached_graph: Optional pre-computed graph from generate_env_a to avoid re-snapshotting.
        """
        category = self._choose_category(record)
        result = {
            "ok": False,
            "object_name": object_name,
            "category": category,
            "synset": record["synset"],
            "semantic_role": semantic_role,
            "placement": None,
            "errors": [],
        }

        # Pre-check: placement cache — skip known failing pairs
        try:
            # Check if an object of this category already exists in the target room.
            # Large furniture (tables, cabinets, countertops) are already in the scene —
            # reuse them instead of spawning a new one.
            existing = self._find_existing_in_room(category, target_room)
            if existing is not None:
                pos, ori = self._safe_pose(existing)
                actual_rooms = self._rooms_for_obj(existing)
                actual_room = actual_rooms[0] if actual_rooms else target_room
                result.update({
                    "ok": True,
                    "reused": True,
                    "reused_object_name": getattr(existing, "name", None),
                    "model": getattr(existing, "model", None),
                    "relation": {"ok": True, "mode": "reused", "reason": "scene_native"},
                    "placement": {"room_id": actual_room, "mode": "reused", "support_object_id": None},
                    "final_pose_before_warmup": {"position": pos, "orientation_xyzw": ori},
                    "delta_node": {
                        "id": object_name,
                        "type": "reused_object",
                        "category": category,
                        "synset": record["synset"],
                        "room_id": actual_room,
                        "semantic_roles": [semantic_role],
                        "reused_from": getattr(existing, "name", None),
                    },
                    "delta_edges": [{"source": f"room::{target_room}", "target": object_name, "relation": "contains"}],
                })
                print(f"[reuse] {category} → existing {getattr(existing, 'name', '?')}", flush=True)
                return result

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

            # Build candidate placement list: primary + fallbacks + floor
            # Use cached graph from generate_env_a when available to avoid expensive re-scans
            placement_graph = cached_graph or self.snapshot()
            placement = self._choose_live_placement(record, target_room, graph=placement_graph)
            candidates = [placement]
            for alt_support in placement.get("support_candidates", [])[:self.config.max_fallback_supports]:
                alt_placement = self._build_placement_for_support(record, target_room, alt_support, graph=placement_graph)
                if alt_placement:
                    candidates.append(alt_placement)
            # Only allow floor placement for categories that reasonably go on the floor.
            # Do not use floor as a universal last resort: plates, food, knives, cups, etc.
            # should fail placement rather than becoming invalid task data.
            if self._category_allows_floor(category):
                for _ in range(3):
                    floor_placement = self._build_floor_placement(record, target_room, graph=placement_graph)
                    if floor_placement:
                        candidates.append(floor_placement)

            placed = False
            chosen_placement = None
            relation_result = None

            # Cap attempts per object
            max_attempts = min(len(candidates), self.config.max_placement_attempts_per_object)
            placement_start = time.time()

            for attempt_idx, placement_attempt in enumerate(candidates):
                if attempt_idx >= max_attempts:
                    result["errors"].append({
                        "error": "max_placement_attempts_reached",
                        "limit": max_attempts,
                    })
                    break

                support_id = placement_attempt.get("support_object_id")
                is_floor = support_id is None
                mode = placement_attempt.get("mode", "?")
                if is_floor:
                    attempt_label = "floor"
                else:
                    attempt_label = f"{mode}({support_id})"

                # Placement cache check: skip known failing (category, support) pairs
                cache_key = (category, support_id or "__floor__")
                if cache_key in self._failed_placement_cache:
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: "
                          f"SKIPPED (in failed_placement_cache)", end="", flush=True)
                    result["errors"].append({
                        "error": "cached_failure_skipped",
                        "category": category,
                        "support": support_id or "__floor__",
                    })
                    continue

                # Per-object timeout check
                attempt_elapsed = time.time() - placement_start
                if attempt_elapsed > self.config.per_object_placement_timeout_sec:
                    result["errors"].append({
                        "error": "per_object_placement_timeout",
                        "elapsed": attempt_elapsed,
                        "limit": self.config.per_object_placement_timeout_sec,
                    })
                    print(f"\n[placement-timeout] {object_name}: "
                          f"{attempt_elapsed:.1f}s > {self.config.per_object_placement_timeout_sec}s, "
                          f"stopping remaining attempts", flush=True)
                    break

                pos = placement_attempt["pose"]["position"]
                # Skip invalid positions (NaN/Inf cause PhysX crash)
                if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in pos):
                    result["errors"].append({
                        "error": "invalid_position",
                        "position": pos,
                        "support": support_id,
                    })
                    continue

                # Pre-check: verify support object exists before applying relation
                if not is_floor and support_id:
                    support_obj = self.env.scene.object_registry("name", support_id, None)
                    if support_obj is None:
                        result["errors"].append({
                            "error": "support_object_missing",
                            "support": support_id,
                        })
                        self._failed_placement_cache.add((category, support_id))
                        continue

                obj.set_position_orientation(
                    position=th.tensor(placement_attempt["pose"]["position"], dtype=th.float32),
                    orientation=th.tensor(placement_attempt["pose"]["orientation_xyzw"], dtype=th.float32),
                )

                if is_floor:
                    # Floor placement: skip relation, just check AABB
                    self._step(1)
                    overlapping = self._check_aabb_overlap(obj, exclude_names=None, margin=0.02, target_room=target_room)
                    if not overlapping:
                        # Keep still to prevent floor impact
                        try:
                            obj.keep_still()
                        except Exception:
                            pass
                        self._step(3)
                        try:
                            obj.wake()
                        except Exception:
                            pass
                        self._step(2)
                        placed = True
                        chosen_placement = placement_attempt
                        relation_result = {"ok": True, "mode": "floor", "reason": "floor_placement"}
                        break
                    else:
                        result["errors"].append({
                            "error": "aabb_overlap_floor",
                            "overlapping_objects": overlapping,
                        })
                        self._failed_placement_cache.add((category, "__floor__"))
                        continue

                # Apply relation (OnTop/Inside) — with per-relation timeout
                rel_start = time.time()
                relation_result = self._apply_relation(obj, placement_attempt)
                rel_elapsed = time.time() - rel_start
                self._step(2)  # reduced from 3+1 to 2: relation set_value already handles positioning

                if rel_elapsed > self.config.per_relation_attempt_timeout_sec:
                    print(f"\n[relation-timeout] {object_name} on {support_id}: "
                          f"{rel_elapsed:.1f}s > {self.config.per_relation_attempt_timeout_sec}s",
                          flush=True)

                if not relation_result.get("ok"):
                    reason = relation_result.get("error") or relation_result.get("reason", "unknown")
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: relation FAILED ({reason})",
                          end="", flush=True)
                    result["errors"].append({
                        "error": "relation_failed",
                        "relation": relation_result,
                        "support": support_id,
                    })
                    self._failed_placement_cache.add((category, support_id))
                    continue

                if placement_attempt.get("mode") == "on_top" and support_obj is not None:
                    pose_check = self._validate_on_top_pose(obj, support_obj)
                    if not pose_check.get("ok"):
                        print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: invalid support pose ({pose_check.get('reason')})",
                              end="", flush=True)
                        result["errors"].append({
                            "error": "invalid_support_pose_after_placement",
                            "pose_check": pose_check,
                            "support": support_id,
                        })
                        self._failed_placement_cache.add((category, support_id))
                        continue

                # Verify with AABB overlap check (cheap)
                overlapping = self._check_aabb_overlap(
                    obj, exclude_names={support_id} if support_id else None, margin=0.02,
                    target_room=target_room,
                )
                if overlapping:
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: AABB overlap ({overlapping[:2]})",
                          end="", flush=True)
                    result["errors"].append({
                        "error": "aabb_overlap_after_placement",
                        "overlapping_objects": overlapping,
                        "support": support_id,
                    })
                    self._failed_placement_cache.add((category, support_id))
                    continue

                # Verify with contact check (catches objects resting on same surface)
                has_contact, contact_names = self._has_unexpected_contacts(obj, support_obj, settle_steps=3)
                if has_contact:
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: unexpected contact ({contact_names[:2]})",
                          end="", flush=True)
                    result["errors"].append({
                        "error": "unexpected_contact_after_placement",
                        "contact_objects": contact_names,
                        "support": support_id,
                    })
                    self._failed_placement_cache.add((category, support_id))
                    continue

                # All checks passed
                if attempt_idx > 0:
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: OK",
                          flush=True)
                placed = True
                chosen_placement = placement_attempt
                break

            if not placed:
                # Enhanced cleanup: remove object and step to refresh physics
                self._remove_object_safe(obj)
                result["errors"].append({
                    "error": "all_placement_attempts_failed",
                    "num_attempts": len(candidates),
                })
                return result

            # Track placement on support for crowding penalty
            support_id = chosen_placement.get("support_object_id")
            support_key = support_id or "__floor__"
            self._placed_on_support[support_key] = self._placed_on_support.get(support_key, 0) + 1
            # Track occupied area on this support
            obj_lo, obj_hi, _ = self._safe_aabb(obj)
            if obj_lo and obj_hi:
                footprint = (obj_hi[0] - obj_lo[0]) * (obj_hi[1] - obj_lo[1])
                self._support_occupied_area[support_key] = (
                    self._support_occupied_area.get(support_key, 0.0) + footprint
                )
            if support_id:
                self._placement_support_map[object_name] = support_id

            position, orientation = obj.get_position_orientation()
            result.update(
                {
                    "ok": True,
                    "model": getattr(obj, "model", None),
                    "relation": relation_result,
                    "placement": chosen_placement,
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
                    "delta_edges": self._delta_edges_for_added_object(
                        object_name, target_room, chosen_placement, relation_result,
                    ),
                }
            )
        except Exception as exc:
            result["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
            self._remove_object_safe_by_name(object_name)
            # Step physics to refresh tensor views after removal
            try:
                self._step(1)
            except Exception:
                pass

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

    def save_checkpoint(self, output_dir):
        """Save checkpoint to resume later, skipping already-failed tasks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint.json"
        state = {
            "schema_version": "deltasg_checkpoint.v1",
            "timestamp": time.time(),
            "run_counter": self._run_counter,
            "rejected_task_cache": sorted(self._rejected_task_cache),
            "failed_placement_cache": [
                {"category": c, "support": s}
                for c, s in self._failed_placement_cache
            ],
            "attempted_tasks": self._checkpoint["attempted_tasks"],
            "rejected_tasks": self._checkpoint["rejected_tasks"],
            "failed_placements": self._checkpoint["failed_placements"],
            "successful_samples": self._checkpoint["successful_samples"],
        }
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=self._json_default)
        print(f"[checkpoint] saved {checkpoint_path} "
              f"({len(self._checkpoint['successful_samples'])} successful, "
              f"{len(self._rejected_task_cache)} rejected tasks)", flush=True)
        return checkpoint_path

    def load_checkpoint(self, output_dir):
        """Load a previous checkpoint to resume. Returns the skip_tasks set."""
        output_dir = Path(output_dir)
        checkpoint_path = output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            print("[checkpoint] no previous checkpoint found, starting fresh", flush=True)
            return set()
        with checkpoint_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self._run_counter = state.get("run_counter", self._run_counter)
        self._rejected_task_cache = set(state.get("rejected_task_cache", []))
        self._failed_placement_cache = {
            (item["category"], item["support"])
            for item in state.get("failed_placement_cache", [])
        }
        self._checkpoint["attempted_tasks"] = state.get("attempted_tasks", [])
        self._checkpoint["rejected_tasks"] = state.get("rejected_tasks", [])
        self._checkpoint["failed_placements"] = state.get("failed_placements", [])
        self._checkpoint["successful_samples"] = state.get("successful_samples", [])
        print(f"[checkpoint] loaded {checkpoint_path}: "
              f"resumed from run #{self._run_counter}, "
              f"{len(self._rejected_task_cache)} rejected tasks, "
              f"{len(self._failed_placement_cache)} failed placements", flush=True)
        return self._rejected_task_cache

    # Categories that should be reused from the scene when possible.
    # These are large / structural objects that are slow to spawn and
    # already exist in most scenes.
    REUSE_CATEGORIES = {
        "countertop", "counter", "kitchen_island", "island",
        "table", "dining_table", "console_table", "coffee_table",
        "breakfast_table", "desk", "dresser", "bookcase",
        "sofa", "couch", "armchair", "bench",
        "bed", "nightstand", "wardrobe",
        "cabinet", "bottom_cabinet", "top_cabinet", "filing_cabinet",
        "shelf", "wall_shelf",
        "stove", "oven", "dishwasher", "washer", "dryer",
        "refrigerator", "electric_refrigerator", "freezer",
        "wine_fridge", "mini_fridge", "fridge", "beer_fridge",
        "sink", "furniture_sink", "bathtub", "shower_stall",
        "basket", "wicker_basket", "hamper", "laundry_basket",
        "trash_can", "public_trash_can", "recycling_bin",
        "display_case", "medicine_cabinet",
        # Structural fixtures — always in scene, never spawn
        "door", "openable_window", "window",
        "electric_switch", "floor_lamp", "table_lamp", "standing_tv",
        "mirror", "picture", "towel_rack", "carpet",
    }

    # Categories that can reasonably be placed on the floor.
    # Everything NOT in this set will be marked as FAILED if no support
    # surface works — no floor fallback for plates, knives, food, etc.
    FLOOR_OKAY = {
        # Containers & storage — naturally sit on floor
        "basket", "wicker_basket", "hamper", "laundry_basket",
        "trash_can", "public_trash_can", "recycling_bin",
        "bucket", "pail", "mop", "broom", "dustpan", "vacuum_cleaner",
        "box", "cardboard_box", "crate", "suitcase", "briefcase",
        "bag", "backpack", "duffel_bag", "shopping_bag",
        # Plants & decor
        "pot_plant", "plant_pot", "flower_pot", "planter",
        "rug", "carpet", "mat", "doormat", "floor_lamp",
        # Footwear
        "shoe", "shoes", "boot", "boots", "slipper",
        # Pet items
        "pet_bed", "dog_bed", "cat_bed", "pet_bowl", "dog_bowl",
        # Furniture that can sit on floor
        "stool", "step_stool", "footstool", "ottoman",
        # Tools & equipment
        "fire_extinguisher", "heater", "space_heater", "fan",
        "air_filter", "air_purifier", "humidifier", "dehumidifier",
        # Toys & misc
        "toy", "teddy_bear", "doll", "ball", "soccer_ball", "basketball",
        "book", "magazine", "newspaper",  # stacks on floor
        "pillow", "cushion", "blanket", "cloth_blanket",
        # Large electronics
        "desktop_computer", "computer_tower", "printer", "scanner",
        # Cleaning
        "spray_bottle", "bottle_of_cleaner", "disinfectant_bottle",
        "bottle_of_disinfectant",
        # Food/drink containers (large ones)
        "water_bottle", "jug", "pitcher", "cooler", "ice_chest",
        # Outdoor/garage items that can be indoors
        "umbrella", "walking_stick", "cane",
    }

    @classmethod
    def _category_allows_floor(cls, category: str) -> bool:
        """Return whether a spawned object category is reasonable on the floor."""
        cat = (category or "").lower().replace("__", "_")
        if cat in cls.FLOOR_OKAY:
            return True
        tokens = set(re.split(r"[_\W]+", cat))
        floor_tokens = {
            "bag", "basket", "bin", "box", "bucket", "carpet", "crate", "hamper",
            "mat", "mop", "pillow", "plant", "rug", "shoe", "suitcase", "trash",
            "umbrella", "vacuum",
        }
        return bool(tokens & floor_tokens)

    @classmethod
    def _has_explicit_support_affinity(cls, obj_category: str) -> bool:
        return any(
            cls._category_matches_affinity_key(obj_category, key)
            for key in cls.OBJECT_SUPPORT_AFFINITY
        )

    @staticmethod
    def _category_matches_affinity_key(obj_category: str, key: str) -> bool:
        obj_lower = (obj_category or "").lower().replace("__", "_")
        key_lower = (key or "").lower().replace("__", "_")
        if not obj_lower or not key_lower:
            return False
        if obj_lower == key_lower:
            return True
        obj_tokens = set(re.split(r"[_\W]+", obj_lower))
        key_tokens = set(re.split(r"[_\W]+", key_lower))
        return key_lower in obj_tokens or obj_lower in key_tokens

    @staticmethod
    def _record_category_for_affinity(record) -> str:
        categories = record.get("direct_categories") or []
        return categories[0] if categories else record.get("synset", "").split(".")[0].replace("__", "_")

    def _get_scene_categories(self):
        """Return the set of all object categories currently in the scene (cached)."""
        if self._scene_categories_cache is None:
            self._scene_categories_cache = {
                getattr(obj, "category", "").lower()
                for obj in self._scene_objects()
                if getattr(obj, "category", None)
            }
        return self._scene_categories_cache

    def _find_existing_in_room(self, category, target_room):
        """Find an existing scene object of the given category.

        For REUSE_CATEGORIES, searches the ENTIRE scene (not just target_room)
        because large furniture like tables stay where they are — the task
        just references them at their existing location.

        Returns the object if found, else None.
        """
        if category.lower() not in self.REUSE_CATEGORIES:
            return None
        # First try target room (best match)
        for obj in self._scene_objects():
            obj_cat = getattr(obj, "category", None)
            if obj_cat and obj_cat.lower() == category.lower():
                rooms = self._rooms_for_obj(obj)
                if target_room in rooms:
                    return obj
        # Then try any room (furniture stays where it is)
        for obj in self._scene_objects():
            obj_cat = getattr(obj, "category", None)
            if obj_cat and obj_cat.lower() == category.lower():
                return obj
        return None

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

    def _check_aabb_overlap(self, obj, exclude_names=None, margin=0.02, target_room=None):
        """Check whether obj's AABB overlaps any other scene object's AABB.

        Args:
            obj: The object to check.
            exclude_names: Set of object names to exclude (e.g. the support, the object itself).
            margin: Extra padding (meters) added to each AABB side before intersection test.
            target_room: Optional room filter — only check objects in this room.

        Returns:
            list[str]: Names of overlapping objects. Empty means no overlap.
        """
        exclude = set(exclude_names or [])
        exclude.add(getattr(obj, "name", None))
        obj_lo, obj_hi, _ = self._safe_aabb(obj)
        if obj_lo is None or obj_hi is None:
            return []
        obj_lo = np.asarray(obj_lo, dtype=np.float32) - margin
        obj_hi = np.asarray(obj_hi, dtype=np.float32) + margin
        overlapping = []
        for other in self._scene_objects():
            other_name = getattr(other, "name", None)
            if other_name in exclude:
                continue
            other_category = getattr(other, "category", "") or ""
            other_tokens = self._tokens(other_category)
            if other_tokens & {"floor", "floors"}:
                continue
            # Room filter: only check objects in the same room (fast pre-filter)
            if target_room:
                other_rooms = self._rooms_for_obj(other)
                if target_room not in other_rooms:
                    continue
            other_lo, other_hi, _ = self._safe_aabb(other)
            if other_lo is None or other_hi is None:
                continue
            other_lo = np.asarray(other_lo, dtype=np.float32)
            other_hi = np.asarray(other_hi, dtype=np.float32)
            if (obj_lo[0] < other_hi[0] and obj_hi[0] > other_lo[0] and
                    obj_lo[1] < other_hi[1] and obj_hi[1] > other_lo[1] and
                    obj_lo[2] < other_hi[2] and obj_hi[2] > other_lo[2]):
                overlapping.append(other_name)
        return overlapping

    def _has_unexpected_contacts(self, obj, support_obj=None, settle_steps=2):
        """Check via RigidContactAPI whether obj contacts anything other than its support.

        Returns:
            tuple[bool, list[str]]: (has_unexpected, names_of_contacting_objects)
        """
        ignore_set = {obj}
        if support_obj is not None:
            ignore_set.add(support_obj)
        self._step(settle_steps)
        try:
            in_contact = RigidContactAPI.is_in_contact(
                scene_idx=self.env.scene.idx,
                query_set=[obj],
                with_set=None,
                ignore_set=ignore_set,
                current_only=True,
            )
        except Exception:
            return False, []
        if not in_contact:
            return False, []
        unexpected = []
        for other in self._scene_objects():
            if other is obj or other is support_obj:
                continue
            other_name = getattr(other, "name", None)
            if other_name is None:
                continue
            try:
                pair_contact = RigidContactAPI.is_in_contact(
                    scene_idx=self.env.scene.idx,
                    query_set=[obj],
                    with_set=[other],
                    ignore_set=None,
                    current_only=True,
                )
            except Exception:
                pair_contact = False
            if pair_contact:
                unexpected.append(other_name)
        return bool(unexpected), unexpected

    def _remove_object_safe(self, obj):
        """Remove an object from the scene, swallowing errors."""
        try:
            self.env.scene.remove_object(obj)
        except Exception:
            pass

    def _remove_object_safe_by_name(self, name):
        """Remove an object from the scene by name, swallowing errors."""
        try:
            obj = self.env.scene.object_registry("name", name, None)
            if obj is not None:
                self.env.scene.remove_object(obj)
        except Exception:
            pass

    def _find_support_for_object(self, object_name):
        """Look up the intended support object for a placed object."""
        support_id = self._placement_support_map.get(object_name)
        if support_id is None:
            return None
        return self.env.scene.object_registry("name", support_id, None)

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

    # ================================================================
    # New Env-A paradigm: task category → required objects → scene
    # ================================================================

    @staticmethod
    def _category_for_task(task_name: str) -> str | None:
        """Reverse lookup: given a task name, return its category."""
        for cat, tasks in VALID_TASKS.items():
            if task_name in tasks:
                return cat
        return None

    def _pick_task_from_category(self, category: str, skip_tasks: set[str] | None = None, used_task_names: set[str] | None = None) -> str | None:
        """Pick a random un-skipped task from a category, preferring unused ones."""
        skip = skip_tasks or set()
        available = [t for t in self._get_active_categories().get(category, set()) if t not in skip]
        if not available:
            return None
        # Prefer tasks not used yet (2x weight), but don't exclude used ones
        used = used_task_names or set()
        weights = [1 if t in used else 2 for t in available]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def _cleanup_spawned_objects(self):
        """Restore scene to pre-run state, fully reverting all changes.

        Uses env.reset() as primary method (fully resets physics + object states).
        Falls back to load_state then name-prefix removal if reset is unavailable.
        """
        # Primary: full env reset (restores physics, object poses, states)
        if hasattr(self.env, 'reset'):
            try:
                self.env.reset()
                self._step(5)
                return
            except Exception:
                pass

        # Fallback 1: load previously saved state
        if hasattr(self, '_pre_run_state') and self._pre_run_state is not None:
            try:
                og.sim.load_state(self._pre_run_state, serialized=False)
                self._pre_run_state = None
                self._step(3)
                return
            except Exception as e:
                print(f"[cleanup] load_state failed: {e}", flush=True)

        # Fallback 2: remove spawned objects by name prefix
        spawned_prefix = "online_env_a_"
        removed = 0
        for obj in list(self._scene_objects()):
            name = getattr(obj, "name", "")
            if name.startswith(spawned_prefix):
                try:
                    self.env.scene.remove_object(obj)
                    removed += 1
                except Exception:
                    pass
        if removed:
            self._clear_usd_selection()
            self._step(3)
            print(f"[cleanup] removed {removed} objects from previous runs", flush=True)

    def set_enabled_categories(self, categories: set[str]):
        """Filter task categories to only include the given set."""
        self._enabled_categories = categories & set(VALID_TASKS.keys())

    def _get_active_categories(self) -> dict[str, set[str]]:
        """Return the currently active task categories dict."""
        if self._enabled_categories is None:
            return VALID_TASKS
        return {k: v for k, v in VALID_TASKS.items() if k in self._enabled_categories}

    def _select_task_and_category(self, skip_tasks: set[str] | None = None, cached_graph: dict | None = None) -> tuple[str | None, str | None]:
        """Select a task category and a specific task name.

        Returns (category, task_name) or (None, None) if no valid task found.

        Args:
            cached_graph: Optional pre-computed graph from generate_env_a to avoid re-snapshotting.
        """
        skip = (skip_tasks or set()) | self._rejected_task_cache
        graph = cached_graph or self.snapshot()
        scene_furniture = self._build_scene_furniture_dict(graph)
        active_tasks = self._get_active_categories()

        # Track used task names to avoid repetition (at task level, not category)
        used_task_names = {s.get("task", "") for s in self._checkpoint.get("successful_samples", [])}
        used_categories = {self._category_for_task(t) for t in used_task_names if self._category_for_task(t)}

        # Step 1: LLM picks a category based on scene furniture
        category = None
        if self._llm_client:
            result = llm_prompts.select_task_category(
                client=self._llm_client,
                scene_furniture=scene_furniture,
                used_categories=used_categories,
            )
            if result and result.get("selected_category") in active_tasks:
                category = result["selected_category"]
                reason = result.get("reason", "")
                print(f"[llm] task category: {category} — {reason}")

        # Fallback: weighted random, slightly preferring unused categories
        if not category:
            available_cats = [
                c for c, tasks in active_tasks.items()
                if any(t not in skip for t in tasks)
            ]
            if not available_cats:
                return None, None
            # Weight: unused categories get 2x weight, used get 1x
            weights = [2 if c not in used_categories else 1 for c in available_cats]
            category = self.rng.choices(available_cats, weights=weights, k=1)[0]

        # Step 2: Pick a specific task from the category
        task_name = self._pick_task_from_category(category, skip, used_task_names)
        if not task_name:
            # Try other categories
            for alt_cat in sorted(active_tasks):
                if alt_cat == category:
                    continue
                task_name = self._pick_task_from_category(alt_cat, skip, used_task_names)
                if task_name:
                    category = alt_cat
                    break

        if task_name:
            print(f"[llm] selected task: {task_name} (category={category})")
        return category, task_name

    def _match_required_objects(self, required_objs: list[dict]) -> list[dict]:
        """Match LLM-returned required objects to asset database records.

        Each required_obj has: {"name": str, "category_hint": str, "role": str}
        We search the asset database for matching records.

        If a required object already exists in the scene (e.g., "standing_tv"
        for turn_on_tv), we skip spawning it — it will be used as a reference_only
        scene object.
        """
        if not required_objs:
            return []

        scene_cats = self._get_scene_categories()
        selected = []
        scene_native_count = 0  # track objects skipped because they're already in scene
        for obj in required_objs:
            hint = obj.get("category_hint", "").lower().replace(" ", "_")
            name = obj.get("name", "").lower().replace(" ", "_")
            role = obj.get("role", "").lower()

            # Check if this object already exists in the scene.
            # - "tool" / "support" roles: skip if scene has it (e.g., fire extinguisher, cabinet)
            # - "target" role: ONLY skip if it's a REUSE_CATEGORY (fixed furniture/appliance).
            #   Small objects (food, medicine, books) should always be spawned.
            found_in_scene = False
            if hint:
                if hint in scene_cats:
                    found_in_scene = True
                else:
                    for sc in scene_cats:
                        # Only fuzzy-match if hint is long enough (>=4 chars)
                        # and the match is substantial — not just a prefix like "book" in "bookcase"
                        if len(hint) >= 4 and (hint in sc or sc in hint):
                            # But exclude false matches: "book" vs "bookcase", "table" vs "breakfast_table"
                            # Check if the hint is a standalone word or a substantial part
                            hint_words = set(hint.split("_"))
                            sc_words = set(sc.split("_"))
                            if hint_words & sc_words:  # at least one word in common
                                found_in_scene = True
                                break
            if name and not found_in_scene:
                if name in scene_cats:
                    found_in_scene = True
                else:
                    for sc in scene_cats:
                        if len(name) >= 4 and (name in sc or sc in name):
                            name_words = set(name.split("_"))
                            sc_words = set(sc.split("_"))
                            if name_words & sc_words:
                                found_in_scene = True
                                break

            if found_in_scene and role == "target":
                # Only skip if the matched scene category is large furniture
                matched_scene_cat = hint if hint in scene_cats else None
                if not matched_scene_cat:
                    for sc in scene_cats:
                        # Use word-level matching to find the actual scene category
                        hint_words = set((hint or "").replace("_", " ").split())
                        sc_words = set(sc.replace("_", " ").split())
                        if hint_words & sc_words:
                            matched_scene_cat = sc
                            break
                if matched_scene_cat and matched_scene_cat not in self.REUSE_CATEGORIES:
                    found_in_scene = False  # small object, should be spawned

            if found_in_scene:
                print(f"[match] {obj['name']}: already in scene (role={role}), skipping spawn")
                scene_native_count += 1
                continue

            # Search asset DB for matching categories
            best_record = None
            best_score = 0
            for task_name, records in self.asset_db.tasks.items():
                for r in self._usable_records(records):
                    score = 0
                    r_cats = {c.lower() for c in r.get("direct_categories", [])}
                    r_synset = r.get("synset", "").lower()
                    r_def = r.get("definition", "").lower()

                    # Exact category match
                    if hint and hint in r_cats:
                        score += 10
                    # Partial category match
                    if hint:
                        for c in r_cats:
                            if hint in c or c in hint:
                                score += 5
                                break
                    # Name match in synset
                    if name and name in r_synset:
                        score += 8
                    # Name words in synset
                    if name:
                        for word in name.split("_"):
                            if len(word) > 2 and word in r_synset:
                                score += 3
                    # Hint in definition
                    if hint and hint in r_def:
                        score += 4
                    # Name in definition
                    if name and len(name) > 3 and name in r_def:
                        score += 2

                    if score > best_score:
                        best_score = score
                        best_record = r

            if best_record and best_score >= 5:
                selected.append(best_record)
                print(f"[match] {obj['name']} → {best_record['synset']} (score={best_score})")
            else:
                # If no asset DB match, check if it's a structural fixture
                if hint and (hint in self.SPAWN_BLOCKLIST or hint in self.REUSE_CATEGORIES):
                    print(f"[match] {obj['name']}: structural fixture (hint={hint}), skipping spawn")
                    scene_native_count += 1
                elif hint:
                    print(f"[match] {obj['name']}: no match found in asset DB")
                else:
                    print(f"[match] {obj['name']}: no hint, skipping")

        # If ALL required objects were unmatched and none were scene-native,
        # the task can't be fulfilled — return empty to trigger rejection.
        if len(selected) == 0 and scene_native_count == 0:
            print(f"[match] all required objects unmatched, rejecting task", flush=True)
            return []

        return selected[:self.config.task_objects]

    def _select_context_records_for_objects(
        self, selected_records: list[dict], graph: dict | None = None
    ) -> list[dict]:
        """Select context objects based on the already-selected task records."""
        if not selected_records:
            return []
        selected_synsets = {r["synset"] for r in selected_records}
        graph = graph or self.snapshot()

        # Collect all usable records excluding selected
        candidates = []
        for task_name, records in self.asset_db.tasks.items():
            for r in self._usable_records(records):
                if r["synset"] not in selected_synsets:
                    candidates.append(r)

        if not candidates:
            return []

        # LLM selection if available
        if self._llm_client:
            result = self._llm_select_context(
                "custom_task", selected_records,
                candidates[:50], graph
            )
            if result:
                return result[:self.config.context_objects]

        return self._weighted_sample_records(candidates, self.config.context_objects)

    def _select_task_records(self, task, skip_tasks=None):
        skip = set(skip_tasks) if skip_tasks else set()
        # Also merge engine-level rejected task cache
        skip |= self._rejected_task_cache
        if task:
            pool = self._usable_records(self.asset_db.by_task(task))
            if len(pool) < self.config.task_objects:
                raise ValueError(f"Task {task!r} only has {len(pool)} usable records")
        else:
            task_pool = [
                name
                for name, records in self.asset_db.tasks.items()
                if name not in skip
                and len({record["synset"] for record in self._usable_records(records)}) >= self.config.task_objects
            ]
            if not task_pool:
                raise ValueError("No task has enough usable asset records")
            # LLM-enhanced task selection
            if self._llm_client:
                llm_task = self._llm_select_task(task_pool)
                if llm_task and llm_task in self.asset_db.tasks:
                    llm_pool = self._usable_records(self.asset_db.by_task(llm_task))
                    if len(llm_pool) >= self.config.task_objects:
                        task = llm_task
                        pool = llm_pool
                        return self._select_task_records_from_pool(task, pool)
            task = self.rng.choice(sorted(task_pool))
            pool = self._usable_records(self.asset_db.by_task(task))
        return self._select_task_records_from_pool(task, pool)

    def _select_task_records_from_pool(self, task, pool):
        """Select task records from pool, using LLM if available."""
        if self._llm_client and len(pool) > self.config.task_objects:
            llm_result = self._llm_select_task_objects(task, pool)
            if llm_result is not None:
                return llm_result
        return self._weighted_sample_records(pool, self.config.task_objects)

    def _llm_select_task_objects(self, task, pool):
        """Use LLM to select key task objects that form a linear chain."""
        candidates = [
            {
                "synset": r["synset"],
                "definition": r.get("definition", ""),
                "categories": r.get("direct_categories", [])[:4],
            }
            for r in pool
        ]
        result = llm_prompts.select_task_objects(
            client=self._llm_client,
            task_name=task,
            candidate_objects=candidates,
            num_to_select=self.config.task_objects,
        )
        if not result or not result.get("selected_synsets"):
            return None

        # Flatten nested lists (LLM sometimes returns [["a","b"], "c"])
        raw_synsets = result["selected_synsets"]
        flat_synsets = []
        for item in raw_synsets:
            if isinstance(item, list):
                flat_synsets.extend(item)
            else:
                flat_synsets.append(item)
        selected_synsets = set(flat_synsets)
        selected = [r for r in pool if r["synset"] in selected_synsets]

        # Fill remaining with weighted random if LLM selected too few
        if len(selected) < self.config.task_objects:
            remaining = [r for r in pool if r["synset"] not in selected_synsets]
            extra = self._weighted_sample_records(
                remaining, self.config.task_objects - len(selected)
            )
            selected.extend(extra)

        reasoning = result.get("reasoning", "")
        names = [r["synset"].split(".")[0] for r in selected]
        print(f"[llm] task objects ({len(selected)}): {', '.join(names)}")
        if reasoning:
            print(f"[llm] selection reasoning: {reasoning[:200]}")
        return selected

    def _select_context_records(self, task, selected_records, graph=None):
        selected = {record["synset"] for record in selected_records}
        candidates = [record for record in self._usable_records(self.asset_db.by_task(task)) if record["synset"] not in selected]
        # LLM-enhanced context object selection
        if self._llm_client and graph:
            llm_result = self._llm_select_context(task, selected_records, candidates, graph)
            if llm_result is not None:
                return llm_result
        return self._weighted_sample_records(candidates, self.config.context_objects)

    # Objects that should never be spawned as task/context objects.
    # These are structural elements or large fixed furniture that ARE
    # support surfaces, not things you place ON other surfaces.
    SPAWN_BLOCKLIST = {
        # True structural elements that can never be task objects
        "wall", "walls", "floor", "floors", "ceiling", "ceilings",
        "door", "window", "staircase", "stairs",
        # Outdoor objects — cannot be placed indoors, no stable support
        "lawn", "grass", "sidewalk", "driveway", "road", "street",
        "mailbox", "fence", "garage_door", "tree", "bush", "shrub",
        "flower_bed", "garden", "patio", "deck", "porch",
        "swing_set", "playground", "pool", "pond", "fountain",
        "fire_hydrant", "street_light", "traffic_light", "parking_meter",
        "outdoor", "exterior", "backyard", "front_yard",
    }

    def _usable_records(self, records):
        usable = []
        for record in records:
            if not record.get("direct_categories") or not record.get("tasks"):
                continue
            if record.get("object_type") in {"liquid", "microPhysicalSubstance", "visualSubstance"}:
                continue
            if record.get("object_type") == "cloth" and not self.config.allow_cloth:
                continue
            # Block true structural elements
            categories = {c.lower() for c in record.get("direct_categories", [])}
            if categories & self.SPAWN_BLOCKLIST:
                continue
            # For REUSE_CATEGORIES: only allow if the category already exists
            # in the scene. This prevents the LLM from selecting large furniture
            # that would need to be spawned (slow / error-prone). The LLM can
            # only pick furniture the scene already has.
            reuse_cats = categories & self.REUSE_CATEGORIES
            if reuse_cats:
                scene_cats = self._get_scene_categories()
                if not (reuse_cats & scene_cats):
                    continue  # none of this record's reuse categories exist in scene
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

    # ================================================================
    # LLM helper methods
    # ================================================================

    # Categories that are "destination" objects — if the instruction mentions
    # one of these, it must exist in the scene furniture inventory.
    DESTINATION_CATEGORIES = {
        "sink", "furniture_sink", "stove", "oven", "microwave",
        "dishwasher", "refrigerator", "electric_refrigerator", "fridge",
        "freezer", "wine_fridge", "mini_fridge", "beer_fridge",
        "trash_can", "public_trash_can", "recycling_bin",
        "toilet", "bathtub", "shower_stall", "washing_machine",
        "washer", "dryer", "coffee_maker", "toaster",
        "cabinet", "bottom_cabinet", "top_cabinet",
        "countertop", "counter", "kitchen_island",
        "table", "dining_table", "coffee_table", "breakfast_table",
        "desk", "dresser", "shelf", "bookcase",
        "bed", "sofa", "couch", "bench",
    }

    def _build_scene_furniture_dict(self, graph):
        """Build {room: [category1, category2, ...]} from scene graph."""
        furniture = {}
        for node in graph.get("nodes", []):
            if node.get("type") != "object":
                continue
            cat = node.get("category")
            if not cat:
                continue
            for room in node.get("rooms", []):
                if room not in furniture:
                    furniture[room] = []
                if cat not in furniture[room]:
                    furniture[room].append(cat)
        return furniture

    def _pre_validate_instruction(self, instruction, scene_furniture, target_room):
        """Check if instruction references objects not in the scene.

        If the instruction mentions a destination object (sink, stove, etc.)
        that doesn't exist in the scene, regenerate the instruction with
        explicit exclusion.

        Returns the (possibly modified) instruction.
        """
        if not instruction or not scene_furniture:
            return instruction

        # Get all scene categories across all rooms
        all_scene_cats = set()
        for cats in scene_furniture.values():
            all_scene_cats.update(c.lower() for c in cats)

        # Check if instruction mentions a destination that doesn't exist
        instruction_lower = instruction.lower()
        missing = set()
        for dest_cat in self.DESTINATION_CATEGORIES:
            # Check both the category name and its display name (with underscores replaced)
            dest_display = dest_cat.replace("_", " ")
            if (dest_cat in instruction_lower or dest_display in instruction_lower) \
                    and dest_cat not in all_scene_cats:
                # Also check partial matches (e.g., "sink" matches "furniture_sink")
                found = any(dest_cat in sc or sc in dest_cat for sc in all_scene_cats)
                if not found:
                    missing.add(dest_display)

        if missing:
            print(f"[pre-validate] instruction references missing objects: {missing}")
            print(f"[pre-validate] scene has: {sorted(all_scene_cats)}")
            if self._llm_client:
                # Regenerate instruction with explicit exclusion
                regen = llm_prompts.generate_instruction(
                    client=self._llm_client,
                    task_name=f"revision (without {', '.join(sorted(missing))})",
                    task_objects=[],  # will use fallback if empty
                    target_room=target_room,
                    scene_furniture=scene_furniture,
                    task_category=None,
                )
                if regen and regen.get("instruction"):
                    new_inst = regen["instruction"]
                    # Verify the new instruction doesn't reference the same missing objects
                    new_lower = new_inst.lower()
                    still_missing = {m for m in missing if m in new_lower}
                    if not still_missing:
                        print(f"[pre-validate] regenerated instruction: {new_inst}")
                        return new_inst
                    else:
                        print(f"[pre-validate] regenerated instruction still references: {still_missing}")
            # Fallback: just note the issue, don't block
            print(f"[pre-validate] WARNING: instruction may reference unavailable objects: {missing}")

        return instruction

    def _filter_tasks_by_scene_compatibility(self, task_pool):
        """Pre-filter: only remove tasks that need >50% reuse cats not in scene."""
        if not task_pool:
            return task_pool
        scene_cats = self._get_scene_categories()
        if not scene_cats:
            return task_pool

        compatible = []
        for name in task_pool:
            records = self._usable_records(self.asset_db.by_task(name))
            task_cats = set()
            for r in records:
                for c in r.get("direct_categories", []):
                    task_cats.add(c.lower().replace("__", "_"))

            # Only filter if >50% of task cats are REUSE_CATEGORIES not in scene
            missing_reuse = [
                c for c in task_cats
                if c in self.REUSE_CATEGORIES and c not in scene_cats
            ]
            if len(missing_reuse) > len(task_cats) * 0.5:
                continue
            compatible.append(name)

        if compatible:
            return compatible
        return task_pool

    def _llm_select_task(self, task_pool):
        """Use LLM to choose the best task from task_pool for the current scene."""
        graph = self.snapshot()
        rooms = [n["name"] for n in graph.get("nodes", []) if n.get("type") == "room"]
        category_counts = dict(Counter(
            n.get("category") for n in graph.get("nodes", [])
            if n.get("type") == "object" and n.get("category")
        ))

        # Pre-filter: only offer tasks compatible with scene objects
        filtered_pool = self._filter_tasks_by_scene_compatibility(task_pool)
        if len(filtered_pool) < len(task_pool):
            print(f"[llm] pre-filtered task pool: {len(task_pool)} → {len(filtered_pool)} "
                  f"(removed {len(task_pool) - len(filtered_pool)} scene-incompatible tasks)",
                  flush=True)

        # Build candidate list with brief info for LLM
        candidates = []
        task_list = list(filtered_pool)
        self.rng.shuffle(task_list)  # shuffle to avoid LLM always picking the same task
        for name in task_list:
            records = self._usable_records(self.asset_db.by_task(name))
            categories = sorted({
                cat for r in records for cat in r.get("direct_categories", [])
            })[:6]
            candidates.append({
                "task": name,
                "num_objects": len(records),
                "sample_categories": categories,
            })

        result = llm_prompts.select_task(
            client=self._llm_client,
            candidate_tasks=candidates,
            scene_context={"rooms": rooms, "category_counts": category_counts},
        )
        if result and result.get("selected_task"):
            chosen = result["selected_task"]
            reason = result.get("reason", "")
            print(f"[llm] task selected: {chosen} — {reason}")
            return chosen
        return None

    def _llm_select_context(self, task, selected_records, candidates, graph):
        """Use LLM to select context objects with meaningful associations."""
        rooms = [n["name"] for n in graph.get("nodes", []) if n.get("type") == "room"]

        task_objs = [
            {
                "synset": r["synset"],
                "definition": r.get("definition", ""),
                "categories": r.get("direct_categories", [])[:4],
            }
            for r in selected_records
        ]
        candidate_objs = [
            {
                "synset": r["synset"],
                "definition": r.get("definition", ""),
                "categories": r.get("direct_categories", [])[:4],
            }
            for r in candidates[:50]
        ]

        result = llm_prompts.select_context_objects(
            client=self._llm_client,
            task_name=task,
            task_objects=task_objs,
            candidate_objects=candidate_objs,
            num_to_select=self.config.context_objects,
            scene_context={"rooms": rooms},
        )
        if not result or not result.get("selected_synsets"):
            return None

        # Flatten nested lists (LLM sometimes returns [["a","b"], "c"])
        raw_synsets = result["selected_synsets"]
        flat_synsets = []
        for item in raw_synsets:
            if isinstance(item, list):
                flat_synsets.extend(item)
            else:
                flat_synsets.append(item)
        selected_synsets = set(flat_synsets)
        selected = [r for r in candidates if r["synset"] in selected_synsets]

        if len(selected) < self.config.context_objects:
            # Fill remaining with weighted random from unselected candidates
            remaining = [r for r in candidates if r["synset"] not in selected_synsets]
            extra = self._weighted_sample_records(
                remaining, self.config.context_objects - len(selected)
            )
            selected.extend(extra)

        reasoning = result.get("reasoning", [])
        # Deduplicate reasoning by synset
        seen_synsets = set()
        deduped_reasoning = []
        for r in reasoning:
            s = r.get("synset", "")
            if s and s not in seen_synsets:
                seen_synsets.add(s)
                deduped_reasoning.append(r)
        reasoning_str = "; ".join(
            f"{r.get('synset', '?')}: {r.get('reason', '?')[:80]}" for r in deduped_reasoning[:self.config.context_objects]
        )
        print(f"[llm] context objects ({len(selected)}): {reasoning_str}")
        return selected

    def _llm_choose_target_room(self, task, selected_records, rooms, graph):
        """Use LLM to choose the best room for the task."""
        # Build per-room furniture summary
        room_furniture = {}
        for node in graph.get("nodes", []):
            if node.get("type") != "object":
                continue
            for room in node.get("rooms", []):
                if room not in room_furniture:
                    room_furniture[room] = []
                cat = node.get("category")
                if cat and cat not in STRUCTURAL_CATEGORIES and len(room_furniture[room]) < 10:
                    room_furniture[room].append(cat)

        objects_to_place = [
            {
                "synset": r["synset"],
                "category": (r.get("direct_categories") or [""])[0],
                "definition": r.get("definition", ""),
                "role": "task_object",
            }
            for r in selected_records
        ]

        result = llm_prompts.assign_rooms(
            client=self._llm_client,
            rooms=rooms,
            objects_to_place=objects_to_place,
            task_name=task,
            scene_context={"room_furniture": room_furniture},
        )
        if not result or not result.get("assignments"):
            return None

        # Pick the room most frequently assigned
        room_counts = Counter()
        for assignment in result["assignments"]:
            room = assignment.get("room")
            if room and room in rooms:
                room_counts[room] += 1
        if not room_counts:
            return None

        chosen_room = room_counts.most_common(1)[0][0]
        reasons = [
            f"{a.get('synset', '?')}→{a.get('room', '?')}: {a.get('reason', '?')}"
            for a in result["assignments"][:3]
        ]
        print(f"[llm] target room: {chosen_room} — {'; '.join(reasons)}")
        return chosen_room

    def _llm_validate_task(self, task, selected_records, context_records, target_room, graph, task_category=None):
        """Use LLM to generate a natural instruction and validate task feasibility.

        Returns:
            (validation_result, generated_instruction)
            validation_result: dict or None from LLM validation
            generated_instruction: str — natural language instruction (always returned)
        """
        rooms = [n["name"] for n in graph.get("nodes", []) if n.get("type") == "room"]

        # Build scene furniture inventory: {room: [category1, category2, ...]}
        scene_furniture = self._build_scene_furniture_dict(graph)

        task_objects = [
            {
                "synset": r["synset"],
                "category": (r.get("direct_categories") or [""])[0],
                "definition": r.get("definition", ""),
                "role": "task_object",
            }
            for r in selected_records
        ]
        for r in context_records:
            task_objects.append({
                "synset": r["synset"],
                "category": (r.get("direct_categories") or [""])[0],
                "definition": r.get("definition", ""),
                "role": "context_object",
            })

        # Step 1: Generate natural language instruction (with scene context)
        fallback_instruction = f"Complete the {task.replace('_', ' ')} task in {target_room}."
        gen_result = llm_prompts.generate_instruction(
            client=self._llm_client,
            task_name=task,
            task_objects=[o for o in task_objects if o["role"] == "task_object"],
            target_room=target_room,
            scene_furniture=scene_furniture,
            task_category=task_category,
        )
        if gen_result and gen_result.get("instruction"):
            instruction = gen_result["instruction"]
            task_desc = gen_result.get("task_description", "")
            print(f"[llm] generated instruction: {instruction}")
            if task_desc:
                print(f"[llm] task description: {task_desc}")
        else:
            instruction = fallback_instruction

        # Step 2: Validate with the natural instruction (with scene context)
        result = llm_prompts.validate_task(
            client=self._llm_client,
            task_instruction=instruction,
            task_objects=task_objects,
            target_room=target_room,
            rooms=rooms,
            scene_furniture=scene_furniture,
        )
        if result:
            feasible = result.get("feasible", True)
            confidence = result.get("confidence", 0)
            issues = result.get("issues", [])
            print(f"[llm] validation: feasible={feasible} confidence={confidence} issues={issues}")
        return result, instruction

    # ---- Hard-reject keyword patterns ----
    # If a validator issue matches any of these, the task is irreparably unsuitable
    # for the current scene and should not be retried with the same task name.
    HARD_REJECT_PATTERNS: list[str] = [
        # Only truly unfixable patterns — things placement can never fix
        "requires cutting",
        "requires pouring",
        "requires stirring",
        "requires mixing",
        "requires measuring",
        "requires scooping",
        "requires peeling",
    ]

    @classmethod
    def _is_hard_reject(cls, issues: list[str]) -> bool:
        """Check if any validator issue matches a hard-reject pattern."""
        if not issues:
            return False
        issues_lower = " ".join(issues).lower()
        for pattern in cls.HARD_REJECT_PATTERNS:
            if pattern in issues_lower:
                return True
        return False

    def _build_llm_rejected_result(
        self, llm_validation, before_graph, primary_task, target_room,
        generated_instruction=None, hard_reject=False,
    ):
        """Build a result dict when LLM validation rejects the task setup.

        Does NOT increment _run_counter so the next successful run keeps
        sequential numbering.
        """
        run_id = f"online_env_a_rejected_{self._run_counter:04d}"
        issues = llm_validation.get("issues", [])
        improved = llm_validation.get("improved_instruction")
        fallback = generated_instruction or f"Complete the {primary_task} task in {target_room}."
        rejection_kind = "hard" if hard_reject else "soft"
        print(f"[llm] {rejection_kind.upper()}-REJECTED task setup: {issues}")
        if improved:
            print(f"[llm] suggested instruction: {improved}")
        return {
            "schema_version": "online_deltasg_env_a.v1",
            "run_id": run_id,
            "ok": False,
            "hard_reject": hard_reject,
            "rejection_kind": rejection_kind,
            "task_environment": None,
            "base_scene": {"scene_model": self._scene_model()},
            "task": {
                "task_id": f"{run_id}_task",
                "task_type": "Env-A",
                "primary_behavior_task": primary_task,
                "instruction": improved or fallback,
                "target_room": target_room,
            },
            "robot": None,
            "camera": None,
            "added_objects": [],
            "task_objects": [],
            "solution_plan": [],
            "before_graph": before_graph,
            "delta_sg": None,
            "validation": {
                "ok": False,
                "llm_rejected": True,
                "llm_validation": llm_validation,
            },
            "after_graph": before_graph,
            "task_instance": {
                "task_id": f"{run_id}_task",
                "instruction": improved or fallback,
                "target_room": target_room,
            },
            "debug": {
                "llm_rejected": True,
                "llm_validation": llm_validation,
            },
        }

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
        # LLM-enhanced room assignment
        if self._llm_client:
            llm_room = self._llm_choose_target_room(task, selected_records, rooms, graph)
            if llm_room and llm_room in rooms:
                return llm_room
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

    def _choose_live_placement(self, record, target_room, graph=None):
        graph = graph or self.snapshot()
        support_result = self._choose_support_node(record, target_room, graph)
        support_node = support_result["node"] if support_result else None
        mode = "on_top"
        if support_node:
            receptacle = ((support_node.get("semantic") or {}).get("receptacle") or {})
            wants_inside = (record.get("edit_metadata", {}).get("receptacle") or {}).get("supports_inside")
            if wants_inside and receptacle.get("supports_inside"):
                mode = "inside"
            elif receptacle.get("supports_on_top"):
                mode = "on_top"
            elif receptacle.get("supports_inside"):
                mode = "inside"
        count_on_support = self._placed_on_support.get(
            support_node["id"] if support_node else "__floor__", 0
        )
        pose = self._pose_near_support(support_node, target_room, graph,
                                       placed_on_support_count=count_on_support)
        return {
            "room_id": target_room,
            "mode": mode,
            "support_object_id": support_node["id"] if support_node else None,
            "support_category": support_node.get("category") if support_node else None,
            "support_candidates": support_result.get("candidates", []) if support_result else [],
            "pose": {
                "position": pose,
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "pose_source": "live_support_object",
        }

    # Object category → preferred support categories.
    # Maps object categories to the support surfaces they work best on.
    OBJECT_SUPPORT_AFFINITY: dict[str, set[str]] = {
        # Food items → kitchen surfaces
        "apple": {"countertop", "counter", "table", "dining_table", "plate", "bowl", "pot", "pan", "cutting_board"},
        "banana": {"countertop", "counter", "table", "dining_table", "plate", "bowl"},
        "bread": {"countertop", "counter", "table", "dining_table", "plate", "cutting_board"},
        "cheese": {"countertop", "counter", "table", "plate", "cutting_board", "refrigerator", "fridge"},
        "chicken": {"countertop", "counter", "table", "plate", "pot", "pan", "cutting_board", "refrigerator", "fridge"},
        "egg": {"countertop", "counter", "table", "plate", "bowl", "refrigerator", "fridge"},
        "fish": {"countertop", "counter", "table", "plate", "pan", "refrigerator", "fridge"},
        "meat": {"countertop", "counter", "table", "plate", "cutting_board", "refrigerator", "fridge"},
        "vegetable": {"countertop", "counter", "table", "plate", "cutting_board", "refrigerator", "fridge"},
        # Dishes/cutlery → kitchen/dining surfaces
        "plate": {"countertop", "counter", "table", "dining_table", "shelf", "dishwasher", "cabinet", "sink"},
        "bowl": {"countertop", "counter", "table", "dining_table", "shelf", "dishwasher", "cabinet", "sink"},
        "mug": {"countertop", "counter", "table", "dining_table", "shelf", "coffee_maker", "dishwasher", "cabinet"},
        "cup": {"countertop", "counter", "table", "dining_table", "shelf", "dishwasher", "cabinet"},
        "glass": {"countertop", "counter", "table", "dining_table", "shelf", "cabinet"},
        "fork": {"countertop", "counter", "table", "dining_table", "drawer", "dishwasher"},
        "knife": {"countertop", "counter", "table", "cutting_board", "drawer", "dishwasher"},
        "spoon": {"countertop", "counter", "table", "drawer", "dishwasher"},
        "cutting_board": {"countertop", "counter", "table", "dining_table"},
        "pot": {"stove", "countertop", "counter", "table", "sink", "cabinet", "oven", "dishwasher"},
        "pan": {"stove", "countertop", "counter", "table", "sink", "cabinet", "oven", "dishwasher"},
        "tray": {"countertop", "counter", "table", "dining_table", "shelf"},
        # Containers
        "basket": {"floor", "countertop", "counter", "table", "shelf", "cabinet"},
        "trash_can": {"floor"},
        "bucket": {"floor", "sink", "countertop", "counter"},
        "box": {"floor", "shelf", "countertop", "counter", "table", "cabinet"},
        "bag": {"floor", "countertop", "counter", "table", "chair"},
        "tupperware": {"countertop", "counter", "table", "cabinet", "refrigerator", "fridge", "shelf"},
        # Bottles/containers
        "bottle": {"countertop", "counter", "table", "shelf", "cabinet", "refrigerator", "fridge"},
        "jar": {"countertop", "counter", "table", "shelf", "cabinet", "refrigerator", "fridge"},
        "can": {"countertop", "counter", "table", "shelf", "cabinet"},
        "carton": {"countertop", "counter", "table", "refrigerator", "fridge", "shelf"},
        # Cleaning supplies
        "soap": {"countertop", "counter", "shelf", "sink", "cabinet"},
        "sponge": {"countertop", "counter", "sink", "shelf"},
        "towel": {"counter", "shelf", "rack", "cabinet", "table"},
        "disinfectant": {"countertop", "counter", "shelf", "cabinet", "floor"},
        "air_filter": {"floor", "table", "countertop", "counter", "desk"},
        # Personal items
        "shampoo": {"shelf", "countertop", "counter", "cabinet", "bathtub"},
        "soap_bar": {"shelf", "countertop", "counter", "sink", "bathtub"},
        "toothbrush": {"countertop", "counter", "shelf", "cabinet"},
        "book": {"table", "desk", "shelf", "bookcase", "nightstand", "bed"},
        "laptop": {"desk", "table", "countertop", "counter"},
        "phone": {"table", "desk", "nightstand", "countertop", "counter"},
        "remote": {"table", "desk", "nightstand", "sofa", "coffee_table"},
        # Clothing
        "cloth": {"bed", "chair", "sofa", "dresser", "shelf", "floor", "hamper", "basket"},
        "shoe": {"floor", "shelf", "rack"},
        "pillow": {"bed", "sofa", "chair", "floor"},
        "blanket": {"bed", "sofa", "chair", "floor"},
        # Decorative
        "vase": {"table", "countertop", "counter", "desk", "shelf"},
        "plant": {"floor", "table", "desk", "shelf", "countertop", "counter"},
        "frame": {"table", "desk", "shelf", "wall_shelf", "countertop"},
        "candle": {"table", "desk", "countertop", "counter", "shelf"},
        # Small appliances
        "coffee_maker": {"countertop", "counter", "table"},
        "toaster": {"countertop", "counter", "table"},
        "kettle": {"countertop", "counter", "table", "stove"},
        "blender": {"countertop", "counter", "table"},
        "microwave": {"countertop", "counter", "table"},
        "mixer": {"countertop", "counter", "table"},
    }

    @classmethod
    def _object_support_affinity_score(cls, obj_category: str, support_category: str) -> float:
        """Return a compatibility score (0.0–1.0) for placing obj on support."""
        obj_lower = obj_category.lower().replace("__", "_")
        support_lower = support_category.lower().replace("__", "_")
        food_keywords = {"food", "fruit", "vegetable", "meat", "fish", "egg", "cheese", "bread", "chicken", "beef", "pork", "bacon", "sausage", "lettuce", "tomato", "onion", "carrot", "strawberry", "blueberry", "peach", "apple", "banana", "orange", "lemon", "milk", "butter", "yogurt", "cream", "flour", "sugar", "salt", "pepper", "oil", "sauce", "coffee", "tea", "water", "juice", "wine", "beer", "soda", "chocolate", "candy", "snack", "cereal", "pasta", "rice", "noodle", "soup"}

        def generic_food_score() -> float:
            if support_lower in {"countertop", "counter", "table", "dining_table", "plate", "bowl", "pot", "pan", "cutting_board", "refrigerator", "fridge", "freezer", "electric_refrigerator"}:
                return 0.7
            if support_lower in {"cabinet", "bottom_cabinet", "shelf"}:
                return 0.45
            if support_lower in {"sink", "furniture_sink", "stove", "oven"}:
                return 0.25
            return 0.1

        # Direct match from affinity table
        for key, preferred in cls.OBJECT_SUPPORT_AFFINITY.items():
            if cls._category_matches_affinity_key(obj_lower, key):
                if support_lower in preferred:
                    return 0.8
                # Check partial matches
                for pref in preferred:
                    if pref in support_lower or support_lower in pref:
                        return 0.6
                if any(kw in obj_lower for kw in food_keywords):
                    return generic_food_score()
                return 0.1  # this object type, but wrong support

        # General heuristics
        # Food items prefer kitchen surfaces
        is_food = any(kw in obj_lower for kw in food_keywords)
        if is_food:
            return generic_food_score()

        # Default: prefer horizontal surfaces
        if support_lower in {"countertop", "counter", "table", "dining_table", "desk", "coffee_table", "breakfast_table", "nightstand", "dresser", "shelf", "bookcase", "cabinet", "stove", "oven", "dishwasher", "washer", "dryer"}:
            return 0.5
        if support_lower in {"floor", "bed", "sofa", "couch", "chair", "bench", "ottoman"}:
            return 0.3
        return 0.2

    def _choose_support_node(self, record, target_room, graph, top_n=4):
        """Choose the best support node for placing an object.

        Returns a dict with:
            - "node": the best support node (or None if no candidates).
            - "candidates": list of alternative support node dicts for fallback.

        Applies area-based crowding AND object-support affinity scoring.
        Supports >80% full are excluded entirely.
        """
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

            # Compute support surface area from bbox
            bbox = node.get("bbox") or {}
            extent = bbox.get("extent")
            if extent and len(extent) >= 2:
                support_area = float(extent[0]) * float(extent[1])
            else:
                support_area = 0.5  # fallback estimate

            # Check fill ratio — skip if support is nearly full
            occupied = self._support_occupied_area.get(node["id"], 0.0)
            fill_ratio = occupied / max(support_area, 0.01)
            if fill_ratio > 0.8:
                continue  # support is too full

            obj_cat = self._record_category_for_affinity(record)
            support_cat = node.get("category", "")
            affinity = self._object_support_affinity_score(obj_cat, support_cat)
            support_tokens = self._tokens(support_cat)
            if "floor" in support_tokens and not self._category_allows_floor(obj_cat):
                continue
            if self._has_explicit_support_affinity(obj_cat) and affinity <= 0.15:
                continue

            score = 0
            if wants_inside and receptacle.get("supports_inside"):
                score += 4
            if receptacle.get("supports_on_top"):
                score += 3
            if receptacle.get("supports_inside"):
                score += 2
            score += self._support_preference_score(node, target_room)

            # Object-support affinity score: prefer semantically compatible supports
            score += affinity * 10

            # Surface area bonus: prefer larger tables/counters (not floor)
            is_floor = node.get("category", "").lower() in {"floors", "floor"}
            if not is_floor:
                score += min(support_area * 3, 15)

            # Area-based crowding penalty
            score -= fill_ratio * 15

            candidates.append((score, node))
        if not candidates:
            return {"node": None, "candidates": []}
        candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
        best_score = candidates[0][0]
        top_group = [node for score, node in candidates if score == best_score]
        chosen = self.rng.choice(sorted(top_group, key=lambda n: n["id"]))
        alt_candidates = [node for _, node in candidates[:top_n]]
        alt_candidates = [n for n in alt_candidates if n["id"] != chosen["id"]]
        return {"node": chosen, "candidates": alt_candidates}

    def _build_placement_for_support(self, record, target_room, support_node, graph=None):
        """Build a placement dict for a specific fallback support node."""
        try:
            graph = graph or self.snapshot()
            mode = "on_top"
            receptacle = ((support_node.get("semantic") or {}).get("receptacle") or {})
            wants_inside = (record.get("edit_metadata", {}).get("receptacle") or {}).get("supports_inside")
            if wants_inside and receptacle.get("supports_inside"):
                mode = "inside"
            elif receptacle.get("supports_on_top"):
                mode = "on_top"
            elif receptacle.get("supports_inside"):
                mode = "inside"
            count_on_support = self._placed_on_support.get(support_node["id"], 0)
            pose = self._pose_near_support(support_node, target_room, graph,
                                           placed_on_support_count=count_on_support)
            return {
                "room_id": target_room,
                "mode": mode,
                "support_object_id": support_node["id"],
                "support_category": support_node.get("category"),
                "support_candidates": [],
                "pose": {
                    "position": pose,
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "pose_source": "fallback_support",
            }
        except Exception:
            return None

    def _build_floor_placement(self, record, target_room, graph=None):
        """Last-resort placement: on the floor in the target room."""
        try:
            graph = graph or self.snapshot()
            room_center = (graph.get("navigation") or {}).get("room_centers", {}).get(target_room)
            if room_center:
                pos = [
                    float(room_center[0]) + self.rng.gauss(0, 0.3),
                    float(room_center[1]) + self.rng.gauss(0, 0.3),
                    0.5,
                ]
            else:
                pos = [self.rng.gauss(0, 0.3), self.rng.gauss(0, 0.3), 0.5]
            return {
                "room_id": target_room,
                "mode": "on_top",
                "support_object_id": None,
                "support_category": "floor",
                "support_candidates": [],
                "pose": {
                    "position": pos,
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "pose_source": "floor_fallback",
            }
        except Exception:
            return None

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
            if state_cls is object_states.OnTop:
                support_aabb = support_obj.aabb
                sup_x_min = float(support_aabb[0][0])
                sup_x_max = float(support_aabb[1][0])
                sup_y_min = float(support_aabb[0][1])
                sup_y_max = float(support_aabb[1][1])
                sup_z_top = float(support_aabb[1][2])

                obj_aabb = obj.aabb
                obj_w = max(float(obj_aabb[1][0] - obj_aabb[0][0]), 0.06)
                obj_d = max(float(obj_aabb[1][1] - obj_aabb[0][1]), 0.06)
                margin = 0.03

                # Collect obstacles: objects ON the support + objects NEAR the support
                obstacles = []
                for other in self._scene_objects():
                    other_name = getattr(other, "name", "")
                    if other is obj or other_name == support_id:
                        continue
                    other_aabb = other.aabb
                    ox_min = float(other_aabb[0][0]); ox_max = float(other_aabb[1][0])
                    oy_min = float(other_aabb[0][1]); oy_max = float(other_aabb[1][1])
                    oz_min = float(other_aabb[0][2]); oz_max = float(other_aabb[1][2])
                    # Include if: on the support, OR AABB overlaps support XY and near support height
                    is_obstacle = False
                    try:
                        if other.states[state_cls].get_value(support_obj):
                            is_obstacle = True
                    except Exception:
                        pass
                    if not is_obstacle:
                        # Check if object is near the support (within 0.5m above or 0.2m below)
                        if (ox_min < sup_x_max and ox_max > sup_x_min and
                            oy_min < sup_y_max and oy_max > sup_y_min and
                            oz_min < sup_z_top + 0.5 and oz_max > sup_z_top - 0.2):
                            is_obstacle = True
                    if is_obstacle:
                        obstacles.append({
                            "x_min": ox_min, "x_max": ox_max,
                            "y_min": oy_min, "y_max": oy_max,
                        })

                # Grid-scan for a clear spot
                sw = sup_x_min + margin + obj_w/2
                ew = sup_x_max - margin - obj_w/2
                sd = sup_y_min + margin + obj_d/2
                ed = sup_y_max - margin - obj_d/2
                if ew <= sw or ed <= sd:
                    return {
                        "ok": False,
                        "mode": placement.get("mode"),
                        "support": support_id,
                        "state": state_cls.__name__,
                        "reason": "support_surface_too_small",
                        "object_footprint": [obj_w, obj_d],
                        "support_bounds": [sup_x_max - sup_x_min, sup_y_max - sup_y_min],
                    }
                w_range = ew - sw
                d_range = ed - sd

                n_steps = 10
                best_cx, best_cy = None, None
                for ix in range(n_steps):
                    for iy in range(n_steps):
                        cx = sw + w_range * (ix + 0.5) / n_steps
                        cy = sd + d_range * (iy + 0.5) / n_steps
                        ok_spot = True
                        for obs in obstacles:
                            if (cx - obj_w/2 - margin < obs["x_max"] and
                                cx + obj_w/2 + margin > obs["x_min"] and
                                cy - obj_d/2 - margin < obs["y_max"] and
                                cy + obj_d/2 + margin > obs["y_min"]):
                                ok_spot = False
                                break
                        if ok_spot:
                            best_cx, best_cy = cx, cy
                            break
                    if best_cx is not None:
                        break

                if best_cx is None:
                    return {
                        "ok": False,
                        "mode": placement.get("mode"),
                        "support": support_id,
                        "state": state_cls.__name__,
                        "reason": "support_surface_occupied",
                        "num_obstacles": len(obstacles),
                    }

                drop_z = sup_z_top + 0.02
                obj.set_position_orientation(
                    position=th.tensor([best_cx, best_cy, drop_z], dtype=th.float32),
                    orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
                )
                try:
                    obj.keep_still()
                except Exception:
                    pass
                self._step(5)
                try:
                    obj.wake()
                except Exception:
                    pass
                self._step(3)
                ok = bool(obj.states[state_cls].get_value(support_obj))
            else:
                # Inside: use set_value with sampling (only option for inside placement)
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

    def _validate_on_top_pose(self, obj, support_obj):
        """Validate that an OnTop sample is physically on the support surface."""
        try:
            obj_aabb = obj.aabb
            sup_aabb = support_obj.aabb
            ox_min, oy_min, oz_min = [float(v) for v in obj_aabb[0][:3]]
            ox_max, oy_max, oz_max = [float(v) for v in obj_aabb[1][:3]]
            sx_min, sy_min, _ = [float(v) for v in sup_aabb[0][:3]]
            sx_max, sy_max, sz_top = [float(v) for v in sup_aabb[1][:3]]

            obj_w = ox_max - ox_min
            obj_d = oy_max - oy_min
            sup_w = sx_max - sx_min
            sup_d = sy_max - sy_min
            clearance = 0.015

            if obj_w + 2 * clearance > sup_w or obj_d + 2 * clearance > sup_d:
                return {
                    "ok": False,
                    "reason": "object_larger_than_support_surface",
                    "object_footprint": [obj_w, obj_d],
                    "support_bounds": [sup_w, sup_d],
                }
            if (
                ox_min < sx_min - clearance
                or ox_max > sx_max + clearance
                or oy_min < sy_min - clearance
                or oy_max > sy_max + clearance
            ):
                return {
                    "ok": False,
                    "reason": "object_footprint_outside_support",
                    "object_bounds": [ox_min, oy_min, ox_max, oy_max],
                    "support_bounds": [sx_min, sy_min, sx_max, sy_max],
                }

            z_gap = oz_min - sz_top
            max_gap = max(0.18, min(0.35, (oz_max - oz_min) * 0.75))
            if z_gap < -0.04 or z_gap > max_gap:
                return {
                    "ok": False,
                    "reason": "object_not_on_support_height",
                    "z_gap": z_gap,
                    "allowed": [-0.04, max_gap],
                }
            return {"ok": True, "z_gap": z_gap}
        except Exception as exc:
            return {"ok": False, "reason": "support_pose_check_failed", "error": repr(exc)}

    def _collect_settling_report(self, object_names):
        before = {name: self._object_position(name) for name in object_names}
        self._step(self.config.settle_steps)

        # Build lookup for contact checking
        scene_objs_by_name = {}
        for name in object_names:
            obj = self.env.scene.object_registry("name", name, None)
            if obj is not None:
                scene_objs_by_name[name] = obj

        objects = []
        all_within = True
        contact_issues = []

        for name, start in before.items():
            end = self._object_position(name)
            if start is None or end is None:
                objects.append({"object_name": name, "ok": False, "error": "missing_pose"})
                all_within = False
                continue
            displacement = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
            within = displacement <= self.config.settle_threshold
            all_within = all_within and within

            entry = {
                "object_name": name,
                "start_position": start,
                "end_position": end,
                "displacement": displacement,
                "within_threshold": within,
            }

            # Check for unexpected contacts after settling
            obj = scene_objs_by_name.get(name)
            if obj is not None:
                support_obj = self._find_support_for_object(name)
                ignore_set = {obj}
                if support_obj is not None:
                    ignore_set.add(support_obj)
                try:
                    has_contact = RigidContactAPI.is_in_contact(
                        scene_idx=self.env.scene.idx,
                        query_set=[obj],
                        with_set=None,
                        ignore_set=ignore_set,
                        current_only=True,
                    )
                except Exception:
                    has_contact = False
                if has_contact:
                    entry["unexpected_contact"] = True
                    all_within = False
                    contact_issues.append(name)
                    contacting = []
                    for other_name, other_obj in scene_objs_by_name.items():
                        if other_name == name:
                            continue
                        try:
                            pair = RigidContactAPI.is_in_contact(
                                scene_idx=self.env.scene.idx,
                                query_set=[obj],
                                with_set=[other_obj],
                                ignore_set=None,
                                current_only=True,
                            )
                        except Exception:
                            pair = False
                        if pair:
                            contacting.append(other_name)
                    entry["contacting_objects"] = contacting
                else:
                    entry["unexpected_contact"] = False

            objects.append(entry)

        return {
            "settle_steps": self.config.settle_steps,
            "settle_threshold": self.config.settle_threshold,
            "all_within_threshold": all_within,
            "objects": objects,
            "contact_issues": contact_issues,
        }

    def _build_task_instance(self, run_id, primary_task, target_room, created_objects, instruction=None, required_objs=None, task_category=None):
        task_objects = [item for item in created_objects if item.get("semantic_role") == "task_object"]

        # Build object info for LLM plan generation — includes actual placement info
        plan_objects = []
        for item in task_objects:
            placement = item.get("placement") or {}
            obj_info = {
                "object_id": item["object_name"],
                "category": item["category"],
                "room": placement.get("room_id", target_room),
                "reused": item.get("reused", False),
            }
            # Add actual support surface info so the LLM generates accurate plans
            support_id = placement.get("support_object_id")
            if support_id:
                obj_info["placed_on"] = support_id
                obj_info["placement_mode"] = placement.get("mode", "on_top")
            elif placement.get("mode") == "reused":
                reused_name = item.get("reused_object_name")
                if reused_name:
                    obj_info["reused_object_name"] = reused_name
            plan_objects.append(obj_info)

        # Find scene objects referenced by the instruction but not in task_objects.
        # These are objects the LLM might need to reference in the plan (e.g., fridge,
        # countertop) that exist in the scene but weren't selected as task objects.
        if instruction and self._llm_client:
            task_cats = {o["category"].lower() for o in plan_objects}
            instruction_lower = instruction.lower()
            for obj in self._scene_objects():
                cat = getattr(obj, "category", None)
                if not cat or cat.lower() in task_cats:
                    continue
                # Check if this category is mentioned in the instruction
                if cat.lower().replace("_", " ") in instruction_lower or cat.lower() in instruction_lower:
                    rooms = self._rooms_for_obj(obj)
                    plan_objects.append({
                        "object_id": getattr(obj, "name", cat),
                        "category": cat,
                        "room": rooms[0] if rooms else target_room,
                        "reused": True,
                        "reference_only": True,  # not a task object, but available for interaction
                    })
                    task_cats.add(cat.lower())  # avoid duplicates

        # Also search for scene objects by required_objs hints.
        # This catches cases like "electric_switch" for "turn_on_light"
        # where the instruction doesn't mention the exact category name.
        if required_objs and instruction:
            for robj in required_objs:
                hint = robj.get("category_hint", "").lower().replace("_", " ")
                name = robj.get("name", "").lower().replace("_", " ")
                for obj in self._scene_objects():
                    cat = getattr(obj, "category", None)
                    if not cat or cat.lower() in task_cats:
                        continue
                    cat_display = cat.lower().replace("_", " ")
                    if hint and (hint in cat_display or cat_display in hint or hint.replace(" ", "_") == cat.lower()):
                        # Only match if words overlap (not just substring like "book" in "bookcase")
                        hint_words = set(hint.replace("_", " ").split())
                        cat_words = set(cat_display.split())
                        if hint_words & cat_words or hint.replace(" ", "_") == cat.lower():
                            rooms = self._rooms_for_obj(obj)
                            plan_objects.append({
                                "object_id": getattr(obj, "name", cat),
                                "category": cat,
                                "room": rooms[0] if rooms else target_room,
                                "reused": True,
                                "reference_only": True,
                            })
                            task_cats.add(cat.lower())
                            break
                    if name and (name in cat_display or cat_display in name):
                        rooms = self._rooms_for_obj(obj)
                        plan_objects.append({
                            "object_id": getattr(obj, "name", cat),
                            "category": cat,
                            "room": rooms[0] if rooms else target_room,
                            "reused": True,
                            "reference_only": True,
                        })
                        task_cats.add(cat.lower())
                        break

        # Refresh instruction with actual placement info
        if self._llm_client and instruction:
            placed_with_positions = []
            for po in plan_objects:
                if po.get("reference_only") or po.get("reused"):
                    continue
                # Resolve support object ID to category name
                support_name = po.get("placed_on", "")
                if support_name and "_" in support_name:
                    support_obj = self.env.scene.object_registry("name", support_name, None)
                    if support_obj is not None:
                        support_name = getattr(support_obj, "category", support_name)
                entry = {
                    "category": po.get("category", ""),
                    "placed_on": support_name or "floor",
                    "placement_mode": po.get("placement_mode", "on_top"),
                }
                placed_with_positions.append(entry)
            if placed_with_positions:
                refresh = llm_prompts.generate_instruction(
                    client=self._llm_client,
                    task_name=primary_task,
                    task_objects=placed_with_positions,
                    target_room=target_room,
                    task_category=task_category,
                )
                if refresh and refresh.get("instruction"):
                    instruction = refresh["instruction"]

        # LLM-driven solution plan
        plan = None
        rooms = list({obj.get("room", target_room) for obj in plan_objects} | {target_room})
        if self._llm_client and instruction:
            plan_result = llm_prompts.generate_solution_plan(
                client=self._llm_client,
                task_instruction=instruction,
                task_objects=plan_objects,
                target_room=target_room,
                rooms=rooms,
            )
            if plan_result and plan_result.get("solution_plan"):
                plan = plan_result["solution_plan"]
                reasoning = plan_result.get("reasoning", "")
                steps_str = " → ".join(
                    f"{s.get('primitive', '?')}({s.get('target_object', s.get('target_room', '?'))})"
                    for s in plan
                )
                print(f"[llm] solution plan ({len(plan)} steps): {steps_str}", flush=True)
                if reasoning:
                    print(f"[llm] plan reasoning: {reasoning[:200]}", flush=True)

        # Fallback: mechanical template
        if not plan:
            plan = [{"step_id": 1, "primitive": "MOVE", "nl": f"Move to {target_room}", "target_room": target_room}]
            step_id = 2
            for item in task_objects:
                plan.append({
                    "step_id": step_id,
                    "primitive": "MOVE",
                    "nl": f"Move to {item['category'].replace('_', ' ')}",
                    "target_object": item["object_name"],
                    "target_room": target_room,
                })
                step_id += 1
                plan.append({
                    "step_id": step_id,
                    "primitive": "INTERACT",
                    "nl": f"Interact with {item['category'].replace('_', ' ')}",
                    "target_object": item["object_name"],
                    "inventory": [],
                })
                step_id += 1

        return {
            "task_id": f"{run_id}_task",
            "task_type": "Env-A",
            "primary_behavior_task": primary_task,
            "instruction": instruction or f"Complete the {primary_task.replace('_', ' ')} task in {target_room}.",
            "target_room": target_room,
            "plan_objects": plan_objects,
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
                "primary_behavior_task": task_instance.get("primary_behavior_task", ""),
                "instruction": task_instance["instruction"],
                "target_room": target_room,
                "semantic_constraints": task_instance.get("semantic_constraints", []),
                "plan_objects": task_instance.get("plan_objects", []),
            },
            "robot": (robot_record := self._robot_record(target_room, graph)),
            "camera": self._camera_records(
                target_room, graph,
                initial_room=robot_record.get("initial_room") if robot_record else None,
            ),
            "added_objects": added_objects,
            "context_objects": [
                ao for ao in added_objects
                if "context_object" in ao.get("semantic_roles", [])
            ],
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
            object_name = item.get("object_name") or item.get("name") or item.get("object_id")
            pose = item.get("final_pose_before_warmup") or {}
            settled = settled_by_name.get(object_name, {})
            added.append(
                {
                    "object_id": object_name,
                    "object_name": object_name,
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
                    "final_pose_before_warmup": pose,
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

    def _camera_records(self, target_room, graph, initial_room=None):
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

        # Global cameras for robot's initial room and task target room
        room_centers = (graph.get("navigation") or {}).get("room_centers", {})
        rooms_to_cover = []
        if initial_room and initial_room != target_room:
            rooms_to_cover.append(initial_room)
        rooms_to_cover.append(target_room)

        for room_id in rooms_to_cover:
            if not room_id:
                continue
            center = room_centers.get(room_id)
            if not center:
                continue
            cam_pos, orientation = self._compute_global_camera_pose(room_id, center)
            camera_id = f"global_{room_id}"
            cameras.append(
                {
                    "camera_id": camera_id,
                    "camera_type": "global_camera",
                    "room_id": room_id,
                    "pose": {
                        "position": list(cam_pos),
                        "orientation_xyzw": list(orientation),
                    },
                    "modalities": ["rgb", "depth", "seg_semantic"],
                    "resolution": {"height": 480, "width": 640},
                    "status": "active",
                }
            )
            # Spawn the camera in the simulation
            self._spawn_global_camera(camera_id, room_id, list(cam_pos), list(orientation))

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

    def _spawn_global_camera(self, camera_id, room_id, position, orientation_xyzw,
                             resolution=None, modalities=None):
        """Spawn a standalone VisionSensor in the scene at the given pose."""
        from omnigibson.sensors import create_sensor
        resolution = resolution or {"height": 480, "width": 640}
        modalities = modalities or ["rgb", "depth", "seg_semantic"]
        try:
            sensor = create_sensor(
                sensor_type="Camera",
                relative_prim_path=f"/{camera_id}",
                name=camera_id,
                modalities=modalities,
                enabled=True,
                sensor_kwargs={
                    "image_height": resolution["height"],
                    "image_width": resolution["width"],
                },
            )
            # Load sensor directly into the scene (bypass add_object which is for USDObject)
            sensor.load(self.env.scene)
            sensor.initialize()
            sensor.set_position_orientation(
                position=np.array(position, dtype=np.float32),
                orientation=np.array(orientation_xyzw, dtype=np.float32),
            )
            return sensor
        except Exception:
            # If the camera already exists or spawn fails, skip silently
            return None

    # Per-room camera configs from camera_config_guide.md.
    # Rooms not listed use the official size rule: wall-center for small rooms,
    # corner camera for large rooms.
    ROOM_CAMERA_CONFIGS = {
        "living_room_0": ("SW", 30),
        "bedroom_0": ("SE", 45),
        "bathroom_0": ("NE", 45),
    }

    def _compute_global_camera_pose(self, room_id, room_center):
        """Compute camera position and orientation for a room overview.

        Uses corner-based placement per camera_config_guide.md:
        3D objects → seg_map pixel bbox → world corners.
        Large rooms use corner placement; small rooms use wall-center placement.
        h_offset=0. Falls back to room_center-based only if seg_map/corners fail.
        """
        try:
            corners = self._get_room_corners_from_objects(room_id)
            if corners is None:
                raise ValueError(f"No objects found in {room_id}")
            diag = float(np.linalg.norm(corners["NE"][:2] - corners["SW"][:2]))
            if diag <= 3.0:
                return self._compute_wall_center_camera(corners)
            if room_id in self.ROOM_CAMERA_CONFIGS:
                corner_name, v_angle = self.ROOM_CAMERA_CONFIGS[room_id]
                opposite_map = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}
                corner = corners[corner_name]
                opposite = corners[opposite_map[corner_name]]
                return self._compute_corner_camera(corner, opposite, v_angle=v_angle)
            return self._compute_corner_camera(corners["SW"], corners["NE"], v_angle=30)
        except Exception:
            pass

        # Fallback: room_center-based placement
        cam_pos = np.array([
            float(room_center[0]) + 0.5,
            float(room_center[1]) + 0.5,
            2.4,
        ], dtype=np.float32)
        look_at = np.array([float(room_center[0]), float(room_center[1]), 0.8], dtype=np.float32)
        return cam_pos, self._look_at_quat(cam_pos, look_at)

    def _get_room_corners_from_objects(self, room_name):
        """Get SW/NE/NW/SE world corners for a room via 3D objects → seg_map pixel mapping."""
        seg_map = self.env.scene.seg_map
        positions = []
        for obj in self._scene_objects():
            try:
                pos, _ = obj.get_position_orientation()
                pos = np.array(pos)
            except Exception:
                continue
            rooms = self._rooms_for_obj(obj)
            if room_name not in rooms:
                continue
            positions.append(pos)
        if not positions:
            return None
        pixels = []
        for pos in positions:
            px = seg_map.world_to_map(th.tensor([pos[0], pos[1]], dtype=th.float32))
            pixels.append(px.numpy())
        pixels = np.stack(pixels)
        px_min = pixels.min(axis=0).astype(int)
        px_max = pixels.max(axis=0).astype(int)
        map_h, map_w = seg_map.room_ins_map.shape
        px_min = np.clip(px_min, 0, [map_w - 1, map_h - 1])
        px_max = np.clip(px_max, 0, [map_w - 1, map_h - 1])
        sw = seg_map.map_to_world(th.tensor([px_min[0], px_min[1]], dtype=th.float32)).cpu().numpy()
        ne = seg_map.map_to_world(th.tensor([px_max[0], px_max[1]], dtype=th.float32)).cpu().numpy()
        return {
            "SW": sw,
            "NE": ne,
            "NW": np.array([sw[0], ne[1]]),
            "SE": np.array([ne[0], sw[1]]),
        }

    @staticmethod
    def _compute_corner_camera(corner, opposite, v_angle=30.0, inward=0.3, height=2.4):
        """Camera at room corner, looking inward along diagonal. h_offset=0."""
        diagonal = np.array([opposite[0] - corner[0], opposite[1] - corner[1]])
        diag_len = np.sqrt(diagonal[0]**2 + diagonal[1]**2)
        cam_pos = np.array([
            corner[0] + (diagonal[0] / diag_len) * inward,
            corner[1] + (diagonal[1] / diag_len) * inward,
            height,
        ], dtype=np.float32)
        diag_angle = np.degrees(np.arctan2(diagonal[1], diagonal[0]))
        yaw = np.radians(diag_angle - 90.0)  # h_offset=0
        pitch = np.radians(90.0 - v_angle)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
        q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
        orientation = np.array([
            q_yaw[3]*q_pitch[0] + q_yaw[0]*q_pitch[3] + q_yaw[1]*q_pitch[2] - q_yaw[2]*q_pitch[1],
            q_yaw[3]*q_pitch[1] - q_yaw[0]*q_pitch[2] + q_yaw[1]*q_pitch[3] + q_yaw[2]*q_pitch[0],
            q_yaw[3]*q_pitch[2] + q_yaw[0]*q_pitch[1] - q_yaw[1]*q_pitch[0] + q_yaw[2]*q_pitch[3],
            q_yaw[3]*q_pitch[3] - q_yaw[0]*q_pitch[0] - q_yaw[1]*q_pitch[1] - q_yaw[2]*q_pitch[2],
        ], dtype=np.float32)
        return cam_pos, orientation

    @staticmethod
    def _compute_wall_center_camera(corners, v_angle=45.0, inward=0.2, height=2.2):
        """Camera at the center of the shortest wall, looking inward. h_offset=0."""
        walls = [
            (corners["SW"], corners["SE"]),
            (corners["SE"], corners["NE"]),
            (corners["NE"], corners["NW"]),
            (corners["NW"], corners["SW"]),
        ]
        c1, c2 = min(walls, key=lambda wall: np.linalg.norm(wall[1][:2] - wall[0][:2]))
        wall_center = np.array([(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5], dtype=np.float32)
        wall_dir = np.array([c2[0] - c1[0], c2[1] - c1[1]], dtype=np.float32)
        wall_len = float(np.linalg.norm(wall_dir))
        if wall_len < 1e-6:
            raise ValueError("invalid wall length")
        wall_dir = wall_dir / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]], dtype=np.float32)
        cam_pos = np.array([
            wall_center[0] + normal[0] * inward,
            wall_center[1] + normal[1] * inward,
            height,
        ], dtype=np.float32)
        normal_angle = np.degrees(np.arctan2(normal[1], normal[0]))
        yaw = np.radians(normal_angle - 90.0)
        pitch = np.radians(90.0 - v_angle)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
        q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
        orientation = np.array([
            q_yaw[3]*q_pitch[0] + q_yaw[0]*q_pitch[3] + q_yaw[1]*q_pitch[2] - q_yaw[2]*q_pitch[1],
            q_yaw[3]*q_pitch[1] - q_yaw[0]*q_pitch[2] + q_yaw[1]*q_pitch[3] + q_yaw[2]*q_pitch[0],
            q_yaw[3]*q_pitch[2] + q_yaw[0]*q_pitch[1] - q_yaw[1]*q_pitch[0] + q_yaw[2]*q_pitch[3],
            q_yaw[3]*q_pitch[3] - q_yaw[0]*q_pitch[0] - q_yaw[1]*q_pitch[1] - q_yaw[2]*q_pitch[2],
        ], dtype=np.float32)
        return cam_pos, orientation

    @staticmethod
    def _look_at_quat(cam_pos, look_at):
        """Compute xyzw quaternion for camera at cam_pos looking at look_at.

        When the camera is looking mostly downward (|dz| is large), uses
        world +Y as the up vector to avoid the singularity and upside-down
        orientation that occurs when the look direction is parallel to world +Z.
        """
        d = np.array(look_at) - np.array(cam_pos)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-8:
            d = np.array([0.0, 0.0, -1.0])
        else:
            d = d / d_norm

        # If looking mostly down, use world +Y as up to avoid the Z-cross-Z singularity
        if abs(d[2]) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        else:
            up = np.array([0.0, 0.0, 1.0])

        right = np.cross(up, d)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        up = np.cross(d, right)
        up = up / np.linalg.norm(up)
        R = np.column_stack([right, up, -d])
        # Matrix to quaternion
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float32)

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
        categories = [c for c in record.get("direct_categories", []) if self._category_has_models(c)]
        if not categories:
            return record["synset"].split(".")[0].replace("__", "_")
        # For REUSE_CATEGORIES: prefer categories that exist in the scene.
        # This prevents picking "wine_fridge" when only "electric_refrigerator"
        # is in the scene, even though both come from the same record.
        scene_cats = self._get_scene_categories()
        reuse_in_scene = [
            c for c in categories
            if c.lower() in self.REUSE_CATEGORIES and c.lower() in scene_cats
        ]
        if reuse_in_scene:
            return self.rng.choice(sorted(reuse_in_scene))
        # For non-reuse categories or no scene match, pick from all valid categories
        non_reuse = [c for c in categories if c.lower() not in self.REUSE_CATEGORIES]
        if non_reuse:
            return self.rng.choice(sorted(non_reuse))
        return self.rng.choice(sorted(categories))

    def _pose_near_support(self, support, target_room, graph, placed_on_support_count=0):
        """Return a pose near the support surface with random XY jitter.

        The jitter prevents multiple objects placed on the same support from
        collapsing to identical coordinates.  Objects placed later get larger
        jitter radii (spiral outward).
        """
        if support:
            pos = (support.get("pose") or {}).get("position")
            bbox = support.get("bbox") or {}
            extent = bbox.get("extent") or [0.5, 0.5, 0.5]
            if pos and bbox.get("max"):
                ex = float(extent[0]) if len(extent) > 0 else 0.5
                ey = float(extent[1]) if len(extent) > 1 else 0.5
                jitter_x_max = max(0.03, ex * 0.3)
                jitter_y_max = max(0.03, ey * 0.3)
                angle = self.rng.uniform(0, 2 * 3.14159265)
                radius_scale = min(1.0, 0.3 + 0.25 * placed_on_support_count)
                jx = radius_scale * jitter_x_max * np.cos(angle) + self.rng.gauss(0, 0.02)
                jy = radius_scale * jitter_y_max * np.sin(angle) + self.rng.gauss(0, 0.02)
                jx = float(np.clip(jx, -jitter_x_max, jitter_x_max))
                jy = float(np.clip(jy, -jitter_y_max, jitter_y_max))
                return [float(pos[0]) + jx, float(pos[1]) + jy, float(bbox["max"][2]) + 0.15]
            if pos:
                return [
                    float(pos[0]) + self.rng.gauss(0, 0.05),
                    float(pos[1]) + self.rng.gauss(0, 0.05),
                    float(pos[2]) + 0.3,
                ]
        room_center = (graph.get("navigation") or {}).get("room_centers", {}).get(target_room)
        if room_center:
            return [
                float(room_center[0]) + self.rng.gauss(0, 0.1),
                float(room_center[1]) + self.rng.gauss(0, 0.1),
                0.8,
            ]
        return [self.rng.gauss(0, 0.1), self.rng.gauss(0, 0.1), 0.8]

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
        bbox_max = bbox.get("max")
        if bbox_max and len(bbox_max) >= 3 and float(bbox_max[2]) > 1.45:
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
