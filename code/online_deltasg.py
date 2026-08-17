"""
Online DeltaSG engine for live OmniGibson environments.

This module implements the layer-2 DeltaSG step described in intro.md as an
online process: it reads the current OmniGibson scene, decides task-oriented
scene edits, applies those edits directly to the running simulator, relaxes
physics, and returns an updated graph plus validation report.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
import time
import traceback
import importlib.util
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.utils import transform_utils as T
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_all_object_category_models, get_dataset_path
from omnigibson.utils.constants import PrimType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_client import create_llm_client
import llm_client as llm_prompts
from deltasg_expert import (
    DEFAULT_MAX_MANIPULATION_HEIGHT,
    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
    DEFAULT_MIN_MANIPULATION_HEIGHT,
    DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT,
    ExpertPlanError,
    SUPPORTED_APPLIANCE_TASKS,
    SUPPORTED_OPEN_CLOSE_TASKS,
    SUPPORTED_RETRIEVAL_DELIVERY_TASKS,
    direct_floor_primary_view_error,
    evaluate_manipulation_height,
    validate_env_a_plan_contract,
)

TASK_ASSET_DATABASE_PATH = REPO_ROOT / "OmniGibson" / "omnigibson" / "scene_graphs" / "task_asset_database.py"


def _load_task_asset_database_class():
    spec = importlib.util.spec_from_file_location("task_asset_database", TASK_ASSET_DATABASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskAssetDatabase


TaskAssetDatabase = _load_task_asset_database_class()


STRUCTURAL_CATEGORIES = {"agent", "ceilings", "ceiling", "floors", "floor", "walls", "wall"}
NON_BLOCKING_NAVIGATION_CATEGORIES = STRUCTURAL_CATEGORIES | {"carpet", "rug", "mat"}
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
    "retrieval_delivery": set(SUPPORTED_RETRIEVAL_DELIVERY_TASKS),
    # Open / Close
    "open_close": set(SUPPORTED_OPEN_CLOSE_TASKS),
    # Appliance (switch on/off)
    "appliance": set(SUPPORTED_APPLIANCE_TASKS),
}

# These taxonomy entries do not yet have a semantically exact asset and
# destination contract, so they must not be advertised as executable tasks.
PLANNED_RETRIEVAL_TASKS: set[str] = {
    "retrieve_remote",
    "put_object_on_table",
    "put_object_in_container",
}

# Flattened set of all task names for quick lookup
ALL_VALID_TASK_NAMES: set[str] = {t for tasks in VALID_TASKS.values() for t in tasks}
ENV_C_TYPES: tuple[str, ...] = ("retrieval_delivery", "open_close", "appliance", "fire")

# Retrieval tasks are intentionally limited to small, well-supported assets.
# The LLM still selects and validates the task, but it must not substitute a
# semantically unrelated or physically unsafe asset (e.g. a toy car for book).
SAFE_RETRIEVAL_ASSETS: dict[str, tuple[str, ...]] = {
    "retrieve_book": ("paperback_book",),
    "retrieve_medicine": ("bottle_of_medicine",),
    "retrieve_key": ("keys", "key_chain"),
    "retrieve_phone": ("cell_phone",),
    "retrieve_drink": ("bottle_of_water", "water_bottle"),
    "retrieve_food": ("canned_food",),
    "deliver_medicine": ("bottle_of_medicine",),
    "deliver_food": ("canned_food",),
    "deliver_drink": ("bottle_of_water", "water_bottle"),
}

assert VALID_TASKS["retrieval_delivery"] == set(SAFE_RETRIEVAL_ASSETS)

NATIVE_TASK_TARGET_TOKENS: dict[str, tuple[str, ...]] = {
    "open_door": ("door",), "close_door": ("door",),
    "open_window": ("window",), "close_window": ("window",),
    "open_fridge": ("fridge", "refrigerator"), "close_fridge": ("fridge", "refrigerator"),
    "open_cabinet": ("cabinet",), "close_cabinet": ("cabinet",),
    "turn_on_light": ("electric_switch", "light", "lamp"),
    "turn_off_light": ("electric_switch", "light", "lamp"),
    "turn_on_tv": ("tv", "television"), "turn_off_tv": ("tv", "television"),
    "turn_on_stove": ("stove",), "turn_off_stove": ("stove",),
}


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
    max_fallback_supports: int = 5
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
    max_failures_per_target_model: int = 2

    # ---- Context object behavior ----
    abort_on_task_object_failure: bool = True
    skip_context_on_failure: bool = True

    # ---- Batch generation safety ----
    # A full reset is required for dataset generation. Removing only spawned
    # objects can leave subtle rigid-body drift in the source scene.
    fast_env_a_cleanup: bool = False
    cache_base_graph: bool = True
    allow_repeat_tasks: bool = False

    # ---- Initial multi-camera coverage ----
    max_global_cameras: int = 3
    visibility_min_pixels: int = 8
    max_camera_pose_attempts_per_room: int = 6
    camera_pose_render_steps: int = 4

    # Generation must apply the same solvability contract as the downstream
    # expert. Symbolic experts still require a connected observation pose, but
    # do not require a Tiago arm-reachable 0.65 m grasp point.
    solvability_profile: str = "physical_control"

    # A stable placement is not useful for physical expert data when the task
    # object is outside the initial robot's traversable component or arm range.
    require_task_object_reachability: bool = True
    max_task_object_approach_distance: float = DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
    # The official traversability map is already eroded by the active robot
    # footprint. Additional clearance is opt-in because it can disconnect an
    # otherwise stable target-conditioned spawn from its own component.
    expert_base_clearance_margin: float = 0.20
    min_manipulation_height: float = DEFAULT_MIN_MANIPULATION_HEIGHT
    max_manipulation_height: float = DEFAULT_MAX_MANIPULATION_HEIGHT

    # Deterministic coverage controls. These remain unset for normal diverse
    # generation and are used by the coverage backfill scheduler.
    target_asset_category: str | None = None
    target_asset_model: str | None = None
    target_native_object_id: str | None = None
    target_placement_mode: str = "auto"


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
        self._floor_placed_objects: set[str] = set()
        self._anchored_native_fixtures: set[str] = set()
        self._robot_component_cache = {}
        self._reachable_room_pixels_cache = {}
        self._support_occupied_area = {}  # support_id → occupied XY area (m²)
        self._scene_categories_cache = None  # lazily computed set of categories in scene
        self._base_graph_cache = None
        # Keep an explicit native-object baseline. OG global state restore is
        # unsafe after dynamically adding/removing objects because its PhysX
        # tensor layout no longer matches the saved state.
        self._clean_scene_poses = {}
        for obj in self._scene_objects():
            name = getattr(obj, "name", "")
            if not name.startswith("online_env_"):
                try:
                    position, orientation = obj.get_position_orientation()
                    self._clean_scene_poses[name] = (position.clone(), orientation.clone())
                except Exception:
                    pass
        self._native_state_baselines = {}
        self._mutated_native_states = set()
        self._run_started = False
        self._pre_run_state = None
        # Retry / fail-fast state
        self._rejected_task_cache: set[str] = set()
        self._failed_placement_cache: set[tuple[str, str]] = set()  # (category, support_id) pairs
        self._rejected_rooms: set[str] = set()  # rooms where a task attempt failed; avoided on retry
        self._llm_retry_count: int = 0
        self._task_placement_start_time: float = 0.0
        self._scene_start_time: float = 0.0
        self._checkpoint: dict = {
            "attempted_tasks": [],
            "rejected_tasks": [],
            "failed_placements": [],
            "successful_samples": [],
        }
        self._used_target_categories: Counter[str] = Counter()
        self._used_target_models: Counter[tuple[str, str]] = Counter()
        self._failed_target_models: Counter[tuple[str, str]] = Counter()
        self._last_native_target_rejection = None
        self._rejected_native_target_cache: set[str] = set()
        self._prepared_native_target_id: str | None = None
        self._prepared_native_target: dict | None = None
        self._enabled_categories: set[str] | None = None  # None = all categories
        self._llm_client = create_llm_client(
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
        )
        if self._llm_client:
            print(f"[online-deltasg] LLM enabled: {self._llm_client.model}")
        else:
            raise RuntimeError("DeltaSG requires an available LLM client; refusing heuristic-only generation")

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
        self._floor_placed_objects = set()
        self._robot_component_cache = {}
        self._reachable_room_pixels_cache = {}
        self._support_occupied_area = {}
        self._scene_categories_cache = None  # refresh scene categories
        self._cleanup_spawned_objects(prefer_reset=not self.config.fast_env_a_cleanup)
        if self.config.cache_base_graph and self.config.fast_env_a_cleanup:
            before_graph = self._get_base_graph()
            self._pre_run_state = None
        else:
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
                    # This is a selection-state failure, not a task. Keeping
                    # it empty prevents the runner from poisoning its task
                    # skip set with a fake "unknown" task name.
                    primary_task="",
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

        generated_support_record = None

        # For retrieval/delivery, task semantics are simple enough to bind the
        # spawned object to a vetted category. This prevents LLM object
        # hallucinations from reaching the physics layer.
        if task_category == "retrieval_delivery":
            safe_records = self._safe_retrieval_records(primary_task)
            if self.config.target_asset_category and not safe_records:
                self._rejected_task_cache.add(primary_task)
                return self._build_llm_rejected_result(
                    llm_validation={
                        "issues": [
                            f"target_asset_category_incompatible:{self.config.target_asset_category}"
                        ]
                    },
                    before_graph=before_graph,
                    primary_task=primary_task,
                    target_room=target_room or "",
                    hard_reject=True,
                )
            if safe_records:
                selected_records = safe_records
                preferred_target_room = target_room
                native_room = self._choose_safe_target_room(
                    safe_records[0], before_graph, preferred_room=target_room,
                )
                target_room = native_room
                if target_room is None:
                    alternate_exclusions = set(self._rejected_rooms)
                    if preferred_target_room:
                        alternate_exclusions.add(preferred_target_room)
                    target_room = self._choose_support_bootstrap_room(
                        before_graph, excluded_rooms=alternate_exclusions,
                    ) or self._choose_support_bootstrap_room(
                        before_graph, excluded_rooms=self._rejected_rooms,
                    ) or self._choose_support_bootstrap_room(
                        before_graph,
                    )
                if target_room is None:
                    self._rejected_task_cache.add(primary_task)
                    return self._build_llm_rejected_result(
                        llm_validation={"issues": ["no_reachable_compatible_support_room"]},
                        before_graph=before_graph,
                        primary_task=primary_task,
                        target_room="",
                        hard_reject=True,
                    )
                if native_room is None:
                    generated_support_record = copy.deepcopy(
                        self._record_for_category("breakfast_table")
                    )
                    generated_support_record["_generated_support_fixture"] = True
                    print(
                        f"[retrieval-bootstrap] no native surface; placing breakfast_table in {target_room}",
                        flush=True,
                    )
                else:
                    print(
                        f"[retrieval-bootstrap] using vetted native support in {target_room}",
                        flush=True,
                    )
                print(f"[retrieval-profile] task={primary_task} asset="
                      f"{self._choose_category(safe_records[0])} room={target_room}", flush=True)

        native_target = None
        native_initial_state = None
        native_instruction = None
        if task_category in {"open_close", "appliance"}:
            # These tasks act on an existing fixture; never spawn whatever
            # auxiliary object the LLM happened to mention before binding it.
            selected_records = []
            rejected_state_targets = []
            while True:
                native_target = self._select_native_task_target(primary_task, before_graph)
                if native_target is None:
                    self._rejected_task_cache.add(primary_task)
                    return self._build_llm_rejected_result(
                        llm_validation={
                            "issues": ["no_officially_transitionable_native_target"],
                            "detail": {
                                "selection": self._last_native_target_rejection,
                                "rejected_targets": rejected_state_targets,
                            },
                        },
                        before_graph=before_graph,
                        primary_task=primary_task,
                        target_room=target_room,
                        hard_reject=True,
                    )
                native_initial_state = self._prepare_native_task_initial_state(
                    primary_task, native_target
                )
                if native_initial_state.get("ok"):
                    break
                rejected_state_targets.append(native_initial_state)
                self.reject_native_target(
                    native_target.get("object_id"),
                    native_initial_state.get("error", "official state transition preflight failed"),
                )
                if self.config.target_native_object_id:
                    self._rejected_task_cache.add(primary_task)
                    return self._build_llm_rejected_result(
                        llm_validation={
                            "issues": ["native_target_official_state_transition_failed"],
                            "detail": native_initial_state,
                        },
                        before_graph=before_graph,
                        primary_task=primary_task,
                        target_room=native_target["room_id"],
                        hard_reject=True,
                    )
            target_room = native_target["room_id"]
            required_objs = [{
                "name": native_target["category"],
                "category_hint": native_target["category"],
                "role": "target",
            }]
            native_instruction = self._native_task_instruction(primary_task, native_target)
            print(f"[native-target] task={primary_task} object={native_target['object_id']} "
                  f"room={target_room}", flush=True)

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
        if task_category in ("appliance", "open_close", "retrieval_delivery"):
            context_records = []
        else:
            context_records = self._select_context_records_for_objects(selected_records, before_graph)
        records = self._dedupe_records(
            ([generated_support_record] if generated_support_record else [])
            + selected_records
            + context_records
        )

        # LLM task feasibility validation
        generated_instruction = None
        llm_validation_result = None
        if self._llm_client:
            llm_validation, generated_instruction = self._llm_validate_task(
                primary_task, selected_records, context_records, target_room, before_graph,
                task_category=task_category,
            )
            if llm_validation is not None:
                llm_validation_result = llm_validation
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
                    improved = (llm_validation.get("improved_instruction") or "").strip()
                    if not hard_reject and improved:
                        print(f"[llm] accepting improved instruction: {improved}", flush=True)
                        llm_validation["accepted_improved_instruction"] = True
                        llm_validation["original_instruction"] = llm_validation.get(
                            "original_instruction", generated_instruction,
                        )
                        generated_instruction = improved
                    else:
                        if hard_reject or not self.config.allow_repeat_tasks:
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

        if native_instruction:
            generated_instruction = native_instruction

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
            "llm_validation": llm_validation_result,
        }

        created_names = []
        import time as _time
        self._task_placement_start_time = _time.time()
        abort_task = False
        abort_reason = None
        # An invalid physics view cannot be healed by retrying inside this
        # process. Escalate it so the runner starts a clean process.
        fatal_physics = False
        placement_graph = before_graph
        generated_support_id = None

        for idx, record in enumerate(records):
            if record.get("_generated_support_fixture"):
                role = "task_support"
            else:
                role = "task_object" if record in selected_records else "context_object"
            cat = self._choose_category(record)
            obj_name = f"{run_id}_{idx:03d}_{self._slug(cat)}"

            # Pre-check: an invalid physics view aborts every remaining record.
            if fatal_physics:
                print(f"[placement] ({idx + 1}/{len(records)}) SKIP {cat}: "
                      f"physics view invalid, clean process required", flush=True)
                validation["failed_objects"].append({
                    "ok": False,
                    "object_name": obj_name,
                    "category": cat,
                    "semantic_role": role,
                    "errors": [{"error": "skipped_after_fatal_physics_error"}],
                })
                continue

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
                cached_graph=placement_graph,
                preferred_support_id=(generated_support_id if role == "task_object" else None),
            )
            if (
                not add_result["ok"]
                and not fatal_physics
                and role == "task_object"
                and task_category == "retrieval_delivery"
                and generated_support_id is None
            ):
                bootstrap_room = self._choose_support_bootstrap_room(
                    placement_graph,
                    excluded_rooms={target_room} | self._rejected_rooms,
                ) or target_room
                if bootstrap_room != target_room:
                    print(
                        f"[retrieval-bootstrap] moving fallback support from cramped "
                        f"{target_room} to {bootstrap_room}",
                        flush=True,
                    )
                    target_room = bootstrap_room
                    delta["task"]["target_room"] = target_room
                support_record = copy.deepcopy(self._record_for_category("breakfast_table"))
                support_record["_generated_support_fixture"] = True
                support_name = f"{run_id}_bootstrap_support"
                print(
                    f"[retrieval-bootstrap] native surfaces failed; placing breakfast_table in {target_room}",
                    flush=True,
                )
                support_result = self.add_task_asset(
                    record=support_record,
                    object_name=support_name,
                    target_room=target_room,
                    semantic_role="task_support",
                    cached_graph=placement_graph,
                )
                if support_result.get("fatal_physics_error"):
                    fatal_physics = True
                elif support_result.get("ok"):
                    generated_support_id = support_result["object_name"]
                    created_names.append(generated_support_id)
                    validation["created_objects"].append(support_result)
                    delta["nodes"].append(support_result["delta_node"])
                    delta["edges"].extend(support_result["delta_edges"])
                    self._step(15)
                    placement_graph = self.snapshot()
                    add_result = self.add_task_asset(
                        record=record,
                        object_name=obj_name,
                        target_room=target_room,
                        semantic_role=role,
                        cached_graph=placement_graph,
                        preferred_support_id=generated_support_id,
                    )
                else:
                    add_result.setdefault("errors", []).append({
                        "error": "bootstrap_support_failed",
                        "detail": support_result.get("errors") or [],
                    })
            if add_result.get("fatal_physics_error"):
                fatal_physics = True
            if fatal_physics:
                abort_task = True
                abort_reason = "fatal_physics_error"
            elapsed = _time.time() - t0
            if add_result["ok"]:
                if not add_result.get("reused"):
                    created_names.append(add_result["object_name"])
                validation["created_objects"].append(add_result)
                delta["nodes"].append(add_result["delta_node"])
                delta["edges"].extend(add_result["delta_edges"])
                if role == "task_support":
                    generated_support_id = add_result.get("object_name")
                    self._step(15)
                    placement_graph = self.snapshot()
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
                    "category": cat,
                    "model": add_result.get("model"),
                    "support": support_id,
                    "errors": errors,
                })
                if add_result.get("model"):
                    self._failed_target_models[(cat, add_result["model"])] += 1
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
        # A room whose placements all failed is a dead end for this process;
        # avoid re-selecting it on the next attempt so retries explore fresh
        # rooms instead of re-poisoning the same one.
        if abort_task and target_room and abort_reason in {
            "all_task_objects_failed", "fatal_physics_error",
        }:
            self._rejected_rooms.add(target_room)
            print(f"[room-memory] rejecting room {target_room} for this process "
                  f"({abort_reason}); rejected={sorted(self._rejected_rooms)}", flush=True)
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
            validation["ok"] = bool(settling_ok)
            if failed_task_objects:
                print(f"[placement] PARTIAL: {len(placed_task_objects)} task + "
                      f"{len(placed_context_objects)} context placed, "
                      f"{len(failed_task_objects)} task objects skipped", flush=True)
            if not settling_ok:
                unstable = [o["object_name"] for o in validation["settling"]["objects"]
                           if not o.get("within_threshold", True)]
                print(f"[placement] FAILED: settling/contact issues on {unstable}", flush=True)
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

        delivery_destination = None
        if task_category == "retrieval_delivery" and placed_task_objects:
            if primary_task.startswith("deliver_"):
                delivery_destination = self._choose_delivery_destination(
                    after_graph, placed_task_objects[0]
                )
                if delivery_destination is None:
                    destination_support, delivery_destination = (
                        self._spawn_delivery_destination_support(
                            run_id,
                            after_graph,
                            source_room=(placed_task_objects[0].get("placement") or {}).get("room_id"),
                            source_item=placed_task_objects[0],
                        )
                    )
                    if destination_support.get("ok") and delivery_destination:
                        validation["created_objects"].append(destination_support)
                        delta["nodes"].append(destination_support["delta_node"])
                        delta["edges"].extend(destination_support["delta_edges"])
                        if not destination_support.get("reused"):
                            created_names.append(destination_support["object_name"])
                        self._step(15)
                        validation["settling"] = self._collect_settling_report(created_names)
                        validation["ok"] = bool(
                            validation["ok"]
                            and validation["settling"]["all_within_threshold"]
                        )
                        if not validation["settling"]["all_within_threshold"]:
                            failed_settling = [
                                {
                                    "object_name": item.get("object_name"),
                                    "displacement": item.get("displacement"),
                                    "unexpected_contact": item.get("unexpected_contact", False),
                                    "contacting_objects": item.get("contacting_objects", []),
                                }
                                for item in validation["settling"].get("objects", [])
                                if (
                                    not item.get("within_threshold", False)
                                    or item.get("unexpected_contact", False)
                                )
                            ]
                            print(
                                f"[delivery-destination] final settling rejected: "
                                f"{failed_settling}",
                                flush=True,
                            )
                        after_graph = self.snapshot()
                    else:
                        validation["ok"] = False
                        validation["failed_objects"].append(destination_support)
                if delivery_destination is not None:
                    target_room = delivery_destination["room_id"]
                    required_objs.append({
                        "name": delivery_destination["category"],
                        "category_hint": delivery_destination["category"],
                        "role": "support",
                    })
            generated_instruction = self._build_retrieval_instruction(
                primary_task, target_room, placed_task_objects[0], delivery_destination,
            )

        task_instance = self._build_task_instance(
            run_id, primary_task, target_room, validation["created_objects"],
            generated_instruction, required_objs=required_objs,
            task_category=task_category,
            native_target=native_target,
            delivery_destination=delivery_destination,
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
            state_changed_objects=[native_initial_state] if native_initial_state else [],
        )
        if validation["ok"]:
            for item in placed_task_objects:
                category = item.get("category")
                if category:
                    self._used_target_categories[category] += 1
        return {
            "schema_version": "online_deltasg_env_a.v1",
            "run_id": run_id,
            "ok": validation["ok"],
            "fatal_physics_error": fatal_physics,
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
        self._floor_placed_objects = set()
        self._cleanup_spawned_objects(prefer_reset=True)
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
        fire_position = (
            (fire_target.get("final_pose_before_warmup") or {}).get("position")
            or (fire_target.get("pose") or {}).get("position")
        )
        extinguisher = self.add_task_asset(
            record=extinguisher_record,
            object_name=f"{run_id}_extinguisher",
            target_room=target_room,
            semantic_role="interaction_tool",
            preferred_position=fire_position,
        )
        self._step(self.config.warmup_steps)
        settling_names = [extinguisher["object_name"]] if extinguisher.get("ok") and not extinguisher.get("reused") else []
        settling = self._collect_settling_report(settling_names)
        after_graph = self.snapshot()
        ok = bool(fire_state.get("ok") and extinguisher.get("ok") and settling.get("all_within_threshold"))
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
            "fire_target": fire_target,
            "solution_tool": extinguisher,
            "settling": settling,
        }
        task_instance = self._build_fire_task_instance(run_id, target_room, fire_target["name"], extinguisher)
        task_environment = self._build_task_environment_record(
            env_id=run_id,
            env_type="Env-B",
            task_instance=task_instance,
            target_room=target_room,
            created_objects=[
                item for item in (fire_target if fire_target.get("spawned") else None, extinguisher)
                if item and item.get("ok")
            ],
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
        # generate_env_b_fire performs the canonical clean-state restore. Doing
        # it here as well caused two consecutive resets before every Env-C fire
        # sample and destabilized several scenes.
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
        settling_names = [
            item.get("object_name") for item in (
                env_b.get("validation", {}).get("solution_tool", {}), bucket, toy
            )
            if item.get("ok") and not item.get("reused") and item.get("object_name")
        ]
        settling = self._collect_settling_report(settling_names)
        after_graph = self.snapshot()
        optimal = env_b["task_instance"]["task_objects"][0]["object_id"]
        ok = bool(env_b["ok"] and bucket["ok"] and toy["ok"] and settling.get("all_within_threshold"))
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
            "settling": settling,
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

    def generate_env_c(self, target_room=None, env_c_types=None, skip_tasks=None):
        """Create a full Env-C sample across all supported task themes."""
        choices = [item for item in (env_c_types or ENV_C_TYPES) if item in ENV_C_TYPES]
        if self._enabled_categories is not None:
            choices = [item for item in choices if item == "fire" or item in self._enabled_categories]
        if not choices:
            choices = ["fire"]

        used = Counter(
            ((sample.get("task") or "").split("::", 1)[0])
            for sample in self._checkpoint.get("successful_samples", [])
            if sample.get("task")
        )
        min_used = min(used.get(item, 0) for item in choices)
        type_name = self.rng.choice(sorted(item for item in choices if used.get(item, 0) == min_used))
        if type_name == "fire":
            return self.generate_env_c_fire_disambiguation(target_room=target_room)
        return self.generate_env_c_task_disambiguation(
            task_category=type_name,
            target_room=target_room,
            skip_tasks=skip_tasks,
        )

    def generate_env_c_task_disambiguation(self, task_category, target_room=None, skip_tasks=None):
        """Create Env-C disambiguation variants for Env-A task categories."""
        previous_categories = self._enabled_categories
        self.set_enabled_categories({task_category})
        try:
            base = self.generate_env_a(target_room=target_room, skip_tasks=skip_tasks)
        finally:
            self._enabled_categories = previous_categories

        if not base.get("ok"):
            return base

        run_id = f"online_env_c_{task_category}_{self._run_counter:04d}"
        self._run_counter += 1
        te = base.get("task_environment", {}) or {}
        task_instance = copy.deepcopy(base.get("task_instance") or {})
        target_room = target_room or task_instance.get("target_room") or (te.get("task") or {}).get("target_room")
        if task_category in {"open_close", "appliance"}:
            target_room = self._env_c_optimal_room(task_instance, target_room)
            task_instance["target_room"] = target_room
        before_graph = base.get("before_graph")
        delta_sg = copy.deepcopy(base.get("delta_sg") or {})
        delta_sg["delta_id"] = run_id
        delta_sg["operation"] = f"online_add_{task_category}_semantic_disambiguation"

        validation = copy.deepcopy(base.get("validation") or {})
        created_objects = list(validation.get("created_objects") or [])
        candidate = None
        distractor = None
        state_changed = []

        if task_category == "retrieval_delivery":
            if not any(item.get("semantic_role") == "task_object" for item in created_objects):
                base["ok"] = False
                base.setdefault("validation", {})["ok"] = False
                base["validation"].setdefault("failed_objects", []).append(
                    {
                        "ok": False,
                        "semantic_role": "task_object",
                        "errors": [{"error": "env_c_retrieval_requires_spawned_task_object"}],
                    }
                )
                return base
            candidate, distractor = self._add_env_c_retrieval_candidates(
                run_id=run_id,
                target_room=target_room,
                created_objects=created_objects,
                delta_sg=delta_sg,
            )
            for item in (candidate, distractor):
                if item and item.get("ok"):
                    created_objects.append(item)
        else:
            candidate, distractor = self._select_env_c_native_candidates(
                task_category=task_category,
                target_room=target_room,
                task_instance=task_instance,
                graph=base.get("after_graph") or self.snapshot(),
            )
            self._append_env_c_native_plan_objects(task_instance, [candidate, distractor])
            self._append_env_c_native_delta(delta_sg, target_room, [candidate, distractor])

        self._step(self.config.warmup_steps)
        settling_names = [
            item.get("object_name") for item in created_objects
            if item.get("ok") and not item.get("reused") and item.get("object_name")
        ]
        settling = self._collect_settling_report(settling_names)
        after_graph = self.snapshot()
        optimal = self._env_c_optimal_object(task_instance, created_objects)
        rejected = []
        for item, reason in (
            (candidate, "valid_candidate_but_not_requested_or_lower_utility"),
            (distractor, "invalid_affordance_or_wrong_object_class"),
        ):
            if item and (object_name := (item.get("object_name") or item.get("object_id"))):
                rejected.append({"object_id": object_name, "reason": reason})

        task_instance["task_id"] = f"{run_id}_task"
        task_instance["task_type"] = "Env-C"
        task_instance["semantic_constraints"] = self._env_c_constraints(task_category)
        task_instance["semantic_reasoning"] = {
            "reasoning_type": ["semantic_disambiguation", "affordance_grounding", "constraint_reasoning"],
            "task_family": task_category,
            "ground_truth": {
                "optimal_object": optimal,
                "rejected_candidates": rejected,
            },
        }
        task_instance["instruction"] = self._env_c_instruction(
            task_category,
            task_instance.get("instruction") or (te.get("task") or {}).get("instruction") or "",
        )

        validation.update(
            {
                "ok": bool(
                    validation.get("ok")
                    and optimal
                    and candidate
                    and candidate.get("ok")
                    and distractor
                    and distractor.get("ok")
                    and settling.get("all_within_threshold")
                ),
                "created_objects": created_objects,
                "candidate_solution": candidate,
                "semantic_distractor": distractor,
                "settling": settling,
            }
        )
        task_environment = self._build_task_environment_record(
            env_id=run_id,
            env_type="Env-C",
            task_instance=task_instance,
            target_room=target_room,
            created_objects=created_objects,
            validation=validation,
            delta_sg=delta_sg,
            graph=after_graph,
            state_changed_objects=state_changed,
        )
        return {
            "schema_version": f"online_deltasg_env_c_{task_category}_disambiguation.v1",
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
            "delta_sg": delta_sg,
            "validation": validation,
            "after_graph": after_graph,
            "task_instance": task_instance,
            "debug": {
                "before_graph": before_graph,
                "after_graph": after_graph,
                "base_run_id": base.get("run_id"),
                "env_c_type": task_category,
            },
        }

    def _add_env_c_retrieval_candidates(self, run_id, target_room, created_objects, delta_sg):
        task_objects = [item for item in created_objects if item.get("semantic_role") == "task_object"]
        if not task_objects:
            return None, None
        optimal = task_objects[0]
        optimal_category = optimal.get("category") or "book"
        graph = self.snapshot()
        alt_room = self._choose_different_room(graph, target_room) or target_room

        candidate_record = self._record_for_category(optimal_category, fallback_category="book")
        candidate = self.add_task_asset(
            record=candidate_record,
            object_name=f"{run_id}_candidate_{self._slug(optimal_category)}",
            target_room=alt_room,
            semantic_role="candidate_solution",
        )
        distractor_category = self._semantic_distractor_category(optimal_category)
        distractor = None
        distractor_categories = [distractor_category, "book", "notebook", "cup", "bowl"]
        tried = set()
        for idx, category in enumerate(distractor_categories):
            if not category or category == optimal_category or category in tried:
                continue
            tried.add(category)
            suffix = self._slug(category) if idx == 0 else f"{self._slug(category)}_{idx}"
            candidate_distractor = self.add_task_asset(
                record=self._record_for_category(category, fallback_category="book"),
                object_name=f"{run_id}_distractor_{suffix}",
                target_room=target_room,
                semantic_role="semantic_distractor",
            )
            if candidate_distractor and candidate_distractor.get("ok"):
                distractor = candidate_distractor
                break
            if distractor is None:
                distractor = candidate_distractor
        for item in (candidate, distractor):
            if item and item.get("ok"):
                delta_sg.setdefault("nodes", []).append(item.get("delta_node"))
                delta_sg.setdefault("edges", []).extend(item.get("delta_edges", []))
        return candidate, distractor

    def _semantic_distractor_category(self, category):
        cat = (category or "").lower()
        mapping = {
            "notebook": "book",
            "book": "notebook",
            "bottle_of_medicine": "soap",
            "medicine": "soap",
            "remote_control": "phone",
            "phone": "remote_control",
            "beer_bottle": "water_bottle",
            "beer_glass": "cup",
            "ice": "soap_bar",
            "apple": "ball",
            "food": "toy_food",
        }
        if cat in mapping:
            return mapping[cat]
        tokens = self._tokens(cat)
        if tokens & {"book", "notebook"}:
            return "book"
        if tokens & {"drink", "glass", "cup", "bottle"}:
            return "cup"
        if tokens & {"medicine", "pill"}:
            return "soap"
        if tokens & {"remote", "phone"}:
            return "phone"
        return "book"

    def _select_env_c_native_candidates(self, task_category, target_room, task_instance, graph):
        plan_objects = task_instance.get("plan_objects") or []
        optimal_id = self._env_c_optimal_object(task_instance, [])
        optimal_category = ""
        for item in plan_objects:
            if item.get("object_id") == optimal_id:
                optimal_category = (item.get("category") or "").lower()
                break
        target_tokens = self._env_c_native_category_tokens(task_category)
        nodes = []
        for node in graph.get("nodes", []):
            if node.get("type") != "object":
                continue
            category = (node.get("category") or "").lower()
            if not any(token in category for token in target_tokens):
                continue
            if node.get("id") == optimal_id:
                continue
            rooms = node.get("rooms") or []
            room_score = 0 if target_room in rooms else 1
            nodes.append((room_score, category, node.get("id"), node))
        nodes.sort()

        def native_record(node, role):
            if not node:
                return None
            rooms = node.get("rooms") or []
            return {
                "ok": True,
                "object_id": node.get("id"),
                "object_name": node.get("id"),
                "category": node.get("category"),
                "semantic_role": role,
                "semantic_roles": [role],
                "room_id": rooms[0] if rooms else target_room,
                "reused": True,
                "reference_only": True,
            }

        candidate_node = nodes[0][3] if nodes else None
        candidate_category = (candidate_node.get("category") or "").lower() if candidate_node else ""
        distractor_node = None
        for _, category, _, node in nodes[1:]:
            if category != candidate_category and category != optimal_category:
                distractor_node = node
                break
        if distractor_node is None:
            distractor_node = nodes[1][3] if len(nodes) > 1 else self._fallback_native_distractor(
                graph,
                target_room,
                {optimal_id, candidate_node.get("id") if candidate_node else None},
            )
        return native_record(candidate_node, "candidate_solution"), native_record(distractor_node, "semantic_distractor")

    def _fallback_native_distractor(self, graph, target_room, excluded):
        excluded = {item for item in excluded if item}
        nodes = []
        for node in graph.get("nodes", []):
            if node.get("type") != "object" or node.get("id") in excluded:
                continue
            rooms = node.get("rooms") or []
            room_score = 0 if target_room in rooms else 1
            nodes.append((room_score, node.get("category") or "", node.get("id"), node))
        nodes.sort()
        return nodes[0][3] if nodes else None

    @staticmethod
    def _env_c_native_category_tokens(task_category):
        if task_category == "open_close":
            return ("door", "window", "cabinet", "fridge", "refrigerator")
        if task_category == "appliance":
            return ("switch", "tv", "lamp", "stove", "oven", "microwave")
        return ()

    def _append_env_c_native_plan_objects(self, task_instance, candidates):
        plan_objects = task_instance.setdefault("plan_objects", [])
        existing = {item.get("object_id"): item for item in plan_objects}
        for item in candidates:
            if not item:
                continue
            object_id = item.get("object_id")
            if object_id in existing:
                # Native distractors are often already present as reference
                # context. Attach their Env-C role to that existing record.
                record = existing[object_id]
                role = item.get("semantic_role")
                roles = set(record.get("semantic_roles") or [])
                if role:
                    roles.add(role)
                    record["semantic_role"] = role
                roles.update(item.get("semantic_roles") or [])
                record["semantic_roles"] = sorted(roles)
                continue
            plan_objects.append(
                {
                    "object_id": object_id,
                    "category": item.get("category"),
                    "room": item.get("room_id"),
                    "reused": True,
                    "reference_only": True,
                    "semantic_role": item.get("semantic_role"),
                    "semantic_roles": item.get("semantic_roles"),
                }
            )
            existing[object_id] = plan_objects[-1]

    def _append_env_c_native_delta(self, delta_sg, target_room, candidates):
        for item in candidates:
            if not item:
                continue
            object_id = item.get("object_id")
            delta_sg.setdefault("nodes", []).append(
                {
                    "id": object_id,
                    "type": "semantic_candidate_object",
                    "category": item.get("category"),
                    "room_id": item.get("room_id") or target_room,
                    "semantic_roles": item.get("semantic_roles") or [item.get("semantic_role")],
                    "reused_from_scene": True,
                }
            )
            delta_sg.setdefault("edges", []).append(
                {"source": f"room::{item.get('room_id') or target_room}", "target": object_id, "relation": "contains"}
            )

    def _env_c_optimal_object(self, task_instance, created_objects):
        for item in task_instance.get("task_objects") or []:
            object_id = item.get("object_id")
            if object_id:
                return object_id
        # Native state-change tasks have no spawned task object. The INTERACT
        # target is authoritative; the first plan object may only be context.
        for step in reversed(task_instance.get("solution_plan") or []):
            if str(step.get("primitive") or "").upper() != "INTERACT":
                continue
            object_id = step.get("target_object")
            if object_id:
                return object_id
        for item in task_instance.get("plan_objects") or []:
            object_id = item.get("object_id")
            if object_id:
                return object_id
        for item in created_objects:
            if item.get("semantic_role") == "task_object":
                return item.get("object_name")
        return None

    @staticmethod
    def _env_c_optimal_room(task_instance, fallback_room):
        for key in ("task_objects", "plan_objects"):
            for item in task_instance.get(key) or []:
                room = item.get("room") or item.get("room_id")
                if room:
                    return room
        return fallback_room

    @staticmethod
    def _env_c_constraints(task_category):
        if task_category == "retrieval_delivery":
            return ["object_identity_disambiguation", "spatial_constraint", "instruction_grounding"]
        if task_category == "open_close":
            return ["articulation_affordance", "target_state_constraint", "object_identity_disambiguation"]
        if task_category == "appliance":
            return ["control_affordance", "device_state_constraint", "object_identity_disambiguation"]
        return ["semantic_disambiguation"]

    @staticmethod
    def _env_c_instruction(task_category, base_instruction):
        base = base_instruction.rstrip(".")
        if task_category == "retrieval_delivery":
            return f"{base}. Choose the object that exactly satisfies the instruction; ignore similar distractors."
        if task_category == "open_close":
            return f"{base}. Select the correct articulated object, not the other openable fixtures."
        if task_category == "appliance":
            return f"{base}. Select the correct controllable device or switch, not other appliances."
        return base_instruction

    def add_task_asset(
        self,
        record,
        object_name,
        target_room,
        semantic_role="task_object",
        cached_graph=None,
        preferred_position=None,
        avoid_position=None,
        preferred_support_id=None,
    ):
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
            existing = (
                None
                if record.get("_generated_support_fixture")
                else self._find_existing_in_room(category, target_room)
            )
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
            model = self._choose_target_model(
                category, preferred_models=record.get("_preferred_models")
            )
            result["model"] = model
            obj = DatasetObject(
                name=object_name,
                category=category,
                model=model,
                prim_type=prim_type,
                in_rooms=target_room,
                # NOTE: kinematic_only is intentionally NOT set here. It only takes
                # effect with fixed_base=True (see compute_kinematic_only in
                # usd_utils.py), and loading a fixed/kinematic body mid-play trips
                # the RigidContactAPI row-count assertion, so a generated support
                # cannot be anchored that way. The support stays dynamic; what keeps
                # it from falling through the floor is the stop/play physics rebuild
                # right below.
            )
            # Import the object while physics is STOPPED, then rebuild the physics
            # scene with play(). Importing a dynamic body mid-play corrupts PhysX's
            # incremental broadphase ("Illegal BroadPhaseUpdateData",
            # BpBroadPhaseABP.cpp) so the new body loses floor-contact pairs and
            # falls straight through the floor (observed: generated nightstands at
            # z=-2..-4 while the floor is z=0, in every 2026-08-10 run up to and
            # including enva_gen5_viewfix). The only historical Env-A sample that
            # stayed on the floor (enva_firstjob_officialontop, 2026-08-10) did so
            # by accident: a flood of PhysX errors made omni.physxui stop the
            # timeline, and the next _step() re-played the sim; that stop/play
            # rebuild re-registers every body with the broadphase from scratch and
            # restores floor contact. Do that rebuild deliberately here. The object
            # itself is initialized inside play() (_non_physics_step runs at the end
            # of play and calls obj.initialize() + keep_still() + update_handles()).
            was_playing = og.sim.is_playing()
            rebuild_pose_restore = []
            if was_playing:
                # stop()+play() re-initializes PhysX from USD and calls
                # Robot.reset(); capture the live poses of the robot and every
                # previously generated object first so the rebuild cannot
                # revert them (see _capture_physics_rebuild_poses).
                rebuild_pose_restore = self._capture_physics_rebuild_poses()
                og.sim.stop()
            self.env.scene.add_object(obj)
            self._clear_usd_selection()
            if was_playing:
                og.sim.play()
                self._restore_physics_rebuild_poses(rebuild_pose_restore)
                self._stabilize_rebuilt_native_fixtures(rebuild_pose_restore)
            else:
                self._step(1)

            # Build candidate placement list: primary + fallbacks + floor
            # Use cached graph from generate_env_a when available to avoid expensive re-scans
            placement_graph = cached_graph or self.snapshot()
            placement = self._choose_live_placement(
                record, target_room, graph=placement_graph, preferred_position=preferred_position,
            )
            if preferred_support_id:
                preferred_support = next(
                    (
                        node for node in placement_graph.get("nodes", [])
                        if node.get("id") == preferred_support_id
                    ),
                    None,
                )
                preferred_placement = (
                    self._build_placement_for_support(
                        record, target_room, preferred_support, graph=placement_graph,
                    )
                    if preferred_support is not None
                    else None
                )
                if preferred_placement:
                    # Pin the generated-support placement so the task object is
                    # tried on its intended support before any floor fallback,
                    # even after robot-distance ranking reorders candidates.
                    preferred_placement["_preferred_support_candidate"] = True
                    fallback_nodes = []
                    support_id = placement.get("support_object_id")
                    if support_id and support_id != preferred_support_id:
                        fallback = next(
                            (
                                node for node in placement_graph.get("nodes", [])
                                if node.get("id") == support_id
                            ),
                            None,
                        )
                        if fallback is not None:
                            fallback_nodes.append(fallback)
                    fallback_nodes.extend(placement.get("support_candidates", []))
                    preferred_placement["support_candidates"] = fallback_nodes
                    placement = preferred_placement
            floor_candidates = []
            generated_support_fixture = semantic_role == "task_support"
            # A task object may use floor as a last resort, but it is accepted
            # only after its live AABB proves that the grasp point is inside
            # the expert robot's manipulation-height band.
            task_floor_eligible = (
                semantic_role != "task_object"
                or self._floor_fallback_allowed(record, category)
            )
            if (
                generated_support_fixture
                or self._floor_fallback_allowed(record, category)
            ) and task_floor_eligible:
                for _ in range(3):
                    floor_placement = self._build_floor_placement(
                        record,
                        target_room,
                        graph=placement_graph,
                        preferred_position=preferred_position,
                        avoid_position=avoid_position,
                        spread_across_room=generated_support_fixture,
                        placement_obj=obj,
                        require_footprint_clear=generated_support_fixture,
                    )
                    if floor_placement:
                        floor_candidates.append(floor_placement)
            candidates = []
            primary_is_floor = (
                placement.get("mode") == "floor"
                or self._tokens(placement.get("support_category") or "") & {"floor", "floors"}
            )
            if generated_support_fixture:
                # Generated support furniture is itself the receptacle. Place
                # it directly on a free reachable floor pose instead of
                # attempting an OnTop relation with the structural floor node.
                candidates.extend(floor_candidates)
            elif self._category_prefers_floor(category):
                candidates.extend(floor_candidates)
                candidates.append(placement)
            elif not primary_is_floor or task_floor_eligible:
                candidates.append(placement)
            for alt_support in (
                [] if generated_support_fixture
                else placement.get("support_candidates", [])[:self.config.max_fallback_supports]
            ):
                alt_is_floor = self._tokens(alt_support.get("category") or "") & {"floor", "floors"}
                if semantic_role == "task_object" and alt_is_floor and not task_floor_eligible:
                    continue
                alt_placement = self._build_placement_for_support(record, target_room, alt_support, graph=placement_graph)
                if alt_placement:
                    candidates.append(alt_placement)
            if not generated_support_fixture and not self._category_prefers_floor(category):
                # Only allow floor placement for categories that reasonably go on the floor.
                # Do not use floor as a universal last resort: plates, food, knives, cups, etc.
                # should fail placement rather than becoming invalid task data.
                candidates.extend(floor_candidates)
            if semantic_role == "task_object" and self.config.target_placement_mode == "floor":
                candidates = floor_candidates

            if semantic_role == "task_object":
                for candidate in candidates:
                    candidate["prefer_robot_access"] = True

            # Relation placement is comparatively expensive and may move the
            # object to a random point on a support. Rank a wider candidate
            # pool by the proposed pose's distance to the robot's initial
            # traversable component, then retain the strict post-placement
            # reachability check below as the source of truth.
            if semantic_role == "task_object" and self.config.require_task_object_reachability:
                ranked_candidates = []
                for original_index, candidate in enumerate(candidates):
                    preflight = self._validate_task_approach_position(
                        candidate["pose"]["position"], {target_room},
                    )
                    candidate["preflight_robot_approach"] = preflight
                    distance = preflight.get("horizontal_distance")
                    ranked_candidates.append((
                        0 if preflight.get("ok") else 1,
                        float(distance) if distance is not None else float("inf"),
                        original_index,
                        candidate,
                    ))
                ranked_candidates.sort(key=lambda item: item[:3])
                candidates = [item[3] for item in ranked_candidates]
                # The distance ranking can move cheap floor candidates ahead of
                # the generated-support on-top candidate; restore the intended
                # support so it is tried first and floor is a true last resort.
                preferred_first = [c for c in candidates if c.get("_preferred_support_candidate")]
                if preferred_first:
                    remaining = [c for c in candidates if not c.get("_preferred_support_candidate")]
                    candidates = preferred_first + remaining

            placed = False
            chosen_placement = None
            relation_result = None

            # Cap attempts per object
            max_attempts = min(len(candidates), self.config.max_placement_attempts_per_object)
            placement_start = time.time()
            # Budget the blocking official relation sampler: at most one call
            # per object. The call is uninterruptible; repeated invocations in
            # the same process poison the physics view, so measure it here
            # instead of relying on the after-the-fact relation timeout log.
            official_fallback = {"calls": 0, "seconds": 0.0}

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
                if not is_floor and cache_key in self._failed_placement_cache:
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
                    # Ground the object from its live extent before checking
                    # collisions. This avoids treating a temporary spawn Z as
                    # its eventual manipulation height.
                    floor_height = self._ground_object_on_floor(obj, placement_attempt)
                    self._step(1)
                    overlapping = self._check_aabb_overlap(
                        obj,
                        exclude_names=None,
                        margin=0.02,
                        target_room=target_room,
                        ignore_floor_coverings=generated_support_fixture,
                    )
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
                        if semantic_role == "task_support":
                            lo, hi, _ = self._safe_aabb(obj)
                            target_aabb_xy = (
                                (lo[:2], hi[:2]) if lo and hi else None
                            )
                            position, _ = obj.get_position_orientation()
                            reachability = self._validate_task_approach_position(
                                position,
                                {target_room},
                                target_aabb_xy=target_aabb_xy,
                                # Keep the support in the blocker set. Otherwise a
                                # point under the table is falsely accepted.
                                target_object_id=None,
                            )
                            if not reachability["ok"]:
                                result["errors"].append({
                                    "error": "task_support_no_collision_free_operation_approach",
                                    "reachability": reachability,
                                    "support": "__floor__",
                                })
                                continue
                        else:
                            reachability = self._validate_task_object_approach(obj)
                        if semantic_role == "task_object" and not reachability["ok"]:
                            result["errors"].append({
                                "error": "task_object_unreachable",
                                "reachability": reachability,
                                "support": "__floor__",
                            })
                            continue
                        manipulation_height = self._validate_floor_manipulation_height(
                            obj, floor_height
                        )
                        if semantic_role == "task_object" and not manipulation_height["eligible"]:
                            result["errors"].append({
                                "error": "task_object_floor_height_out_of_range",
                                "manipulation_height": manipulation_height,
                                "support": "__floor__",
                            })
                            continue
                        primary_view_error = direct_floor_primary_view_error(
                            manipulation_height["relative_height"]
                        )
                        if semantic_role == "task_object" and primary_view_error:
                            result["errors"].append({
                                "error": "task_object_floor_primary_view_too_low",
                                "reason": primary_view_error,
                                "manipulation_height": manipulation_height,
                                "support": "__floor__",
                            })
                            continue
                        placed = True
                        chosen_placement = copy.deepcopy(placement_attempt)
                        chosen_placement["robot_approach"] = reachability
                        chosen_placement["manipulation_height"] = manipulation_height
                        chosen_placement["mode"] = "floor"
                        chosen_placement["support_category"] = "floor"
                        relation_result = {"ok": True, "mode": "floor", "reason": "floor_placement"}
                        break
                    else:
                        result["errors"].append({
                            "error": "aabb_overlap_floor",
                            "overlapping_objects": overlapping,
                        })
                        continue

                # Apply relation (OnTop/Inside) — with per-relation timeout
                support_pose_before = (
                    support_obj.get_position_orientation()
                    if support_obj is not None and not support_id.startswith("online_env_")
                    else None
                )
                rel_start = time.time()
                relation_result = self._apply_relation(
                    obj, placement_attempt, official_fallback=official_fallback
                )
                rel_elapsed = time.time() - rel_start
                self._step(2)  # reduced from 3+1 to 2: relation set_value already handles positioning

                if rel_elapsed > self.config.per_relation_attempt_timeout_sec:
                    print(f"\n[relation-timeout] {object_name} on {support_id}: "
                          f"{rel_elapsed:.1f}s > {self.config.per_relation_attempt_timeout_sec}s",
                          flush=True)

                if relation_result.get("fatal_physics_error"):
                    # An invalid physics view cannot be retried away inside this
                    # process. Abort the remaining candidates for this object so
                    # the task attempt fails fast and a clean process can start.
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: "
                          f"FATAL physics error in official sampler "
                          f"({relation_result.get('official_sampler', {}).get('error')})",
                          end="", flush=True)
                    result["errors"].append({
                        "error": "official_sampler_physics_error",
                        "relation": relation_result,
                        "support": support_id,
                    })
                    result["fatal_physics_error"] = True
                    break

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

                if semantic_role == "task_object":
                    floor_height = self._floor_height_for_position(
                        obj.get_position_orientation()[0]
                    )
                    aabb_min, aabb_max, _ = self._safe_aabb(obj)
                    min_portable_height = (
                        DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT
                        if self.config.solvability_profile == "physical_control"
                        else self.config.min_manipulation_height
                    )
                    portable_height = evaluate_manipulation_height(
                        "GRASP",
                        aabb_min[2],
                        aabb_max[2],
                        floor_height,
                        min_portable_height,
                        self.config.max_manipulation_height,
                    )
                    portable_height["solvability_profile"] = self.config.solvability_profile
                    if not portable_height["eligible"]:
                        result["errors"].append({
                            "error": "task_object_physical_grasp_height_out_of_range",
                            "manipulation_height": portable_height,
                            "support": support_id,
                        })
                        continue
                    placement_attempt["manipulation_height"] = portable_height

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

                support_tokens = self._tokens(placement_attempt.get("support_category") or "")
                if semantic_role == "task_object" and support_tokens & {"floor", "floors"}:
                    floor_height = self._floor_height_for_position(
                        obj.get_position_orientation()[0]
                    )
                    manipulation_height = self._validate_floor_manipulation_height(
                        obj, floor_height
                    )
                    if not manipulation_height["eligible"]:
                        result["errors"].append({
                            "error": "task_object_floor_height_out_of_range",
                            "manipulation_height": manipulation_height,
                            "support": support_id,
                        })
                        continue
                    placement_attempt["manipulation_height"] = manipulation_height

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

                if support_pose_before is not None:
                    # Native furniture can begin sliding only after several
                    # physics frames under load. Observe the same warmup
                    # window used by final sample validation before accepting
                    # this support.
                    self._step(20)
                    support_position, support_orientation = support_obj.get_position_orientation()
                    support_displacement = float(
                        th.linalg.norm(support_position - support_pose_before[0])
                    )
                    # Fail early before a slowly sliding support crosses the
                    # final 5 cm scene-integrity gate during later rendering.
                    if support_displacement > 0.01:
                        print(
                            f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: "
                            f"native support displaced {support_displacement:.3f}m",
                            end="",
                            flush=True,
                        )
                        result["errors"].append({
                            "error": "native_support_displaced_by_placement",
                            "displacement": support_displacement,
                            "support": support_id,
                        })
                        # Remove the payload before restoring the native fixture;
                        # otherwise its load can immediately move the fixture again.
                        object_position, object_orientation = obj.get_position_orientation()
                        object_position = object_position.clone()
                        object_position[2] += 2.0
                        obj.set_position_orientation(
                            position=object_position, orientation=object_orientation
                        )
                        obj.keep_still()
                        support_obj.set_position_orientation(
                            position=support_pose_before[0],
                            orientation=support_pose_before[1],
                        )
                        support_obj.keep_still()
                        self._step(2)
                        self._failed_placement_cache.add((category, support_id))
                        continue

                reachability = self._validate_task_object_approach(obj, target_room=target_room)
                if semantic_role == "task_object" and not reachability["ok"]:
                    print(
                        f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: "
                        f"robot approach FAILED ({reachability.get('reason')}, "
                        f"distance={reachability.get('horizontal_distance')}, "
                        f"rooms={reachability.get('target_rooms')}, "
                        f"component_rooms={reachability.get('component_rooms')})",
                        end="",
                        flush=True,
                    )
                    result["errors"].append({
                        "error": "task_object_unreachable",
                        "reachability": reachability,
                        "support": support_id,
                    })
                    continue

                # All checks passed
                if attempt_idx > 0:
                    print(f"\n  attempt {attempt_idx+1}/{max_attempts} [{attempt_label}]: OK",
                          flush=True)
                placed = True
                chosen_placement = copy.deepcopy(placement_attempt)
                chosen_placement["robot_approach"] = reachability
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
            else:
                self._floor_placed_objects.add(object_name)

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
        used_object_ids = Counter()
        for sample in self._checkpoint.get("successful_samples", []):
            diversity = sample.get("diversity") or {}
            for object_id in diversity.get("target_object_ids") or []:
                used_object_ids[object_id] += 1
        min_used = min(used_object_ids.get(node["id"], 0) for node in candidates)
        selected = self.rng.choice([
            node for node in candidates
            if used_object_ids.get(node["id"], 0) == min_used
        ])
        selected["spawned"] = False
        selected.setdefault("name", selected.get("id"))
        return selected

    def _spawn_fire_target(self, target_room):
        record = self._record_for_category("plywood", fallback_category="book")
        spawned = self.add_task_asset(
            record=record,
            object_name=f"online_env_b_fire_{self._run_counter:04d}_fire_target",
            target_room=target_room,
            semantic_role="anomaly_carrier",
        )
        if not spawned["ok"]:
            raise RuntimeError(f"Could not spawn a fire target: {spawned['errors']}")
        spawned["name"] = spawned["object_name"]
        spawned["spawned"] = True
        return spawned

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
                record = copy.deepcopy(self.rng.choice(records))
                record["_preferred_category"] = candidate
                return record
            if self._category_has_models(candidate):
                return {
                    "synset": f"{candidate}.n.01",
                    "direct_categories": [candidate],
                    "tasks": [],
                    "object_type": "rigid",
                    "edit_metadata": {
                        "interaction": {"kind": "manipulable", "confidence": "synthetic_category"},
                        "receptacle": {"supports_on_top": False, "supports_inside": False},
                    },
                }
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
            "failed_target_models": [
                {"category": category, "model": model, "count": count}
                for (category, model), count in sorted(self._failed_target_models.items())
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
        # In diversity mode a task name may be reused with a different pose.
        # Historical LLM/task-selection rejects must not exhaust the next
        # process's task pool; exact samples are rejected by the output-level
        # fingerprint instead. Retain the placement cache because it protects
        # the scene from known bad object/support combinations.
        self._rejected_task_cache = (
            set() if self.config.allow_repeat_tasks
            else set(state.get("rejected_task_cache", []))
        )
        self._failed_placement_cache = {
            (item["category"], item["support"])
            for item in state.get("failed_placement_cache", [])
        }
        self._failed_target_models = Counter({
            (item["category"], item["model"]): int(item.get("count", 1))
            for item in state.get("failed_target_models", [])
            if item.get("category") and item.get("model")
        })
        self._checkpoint["attempted_tasks"] = state.get("attempted_tasks", [])
        self._checkpoint["rejected_tasks"] = state.get("rejected_tasks", [])
        self._checkpoint["failed_placements"] = state.get("failed_placements", [])
        self._checkpoint["successful_samples"] = state.get("successful_samples", [])
        self._used_target_categories = Counter()
        self._used_target_models = Counter()
        for sample in self._checkpoint["successful_samples"]:
            diversity = sample.get("diversity") or {}
            for category in diversity.get("target_categories") or []:
                self._used_target_categories[category] += 1
            for item in diversity.get("target_models") or []:
                if isinstance(item, dict) and item.get("category") and item.get("model"):
                    self._used_target_models[(item["category"], item["model"])] += 1
        print(f"[checkpoint] loaded {checkpoint_path}: "
              f"resumed from run #{self._run_counter}, "
              f"{len(self._rejected_task_cache)} rejected tasks, "
              f"{len(self._failed_placement_cache)} failed placements, "
              f"{len(self._failed_target_models)} failed models", flush=True)
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
        "plywood",  # fire carrier; large flat stock is normally floor-staged
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
    def _category_prefers_floor(cls, category: str) -> bool:
        cat = (category or "").lower().replace("__", "_")
        if cat in {
            "fire_extinguisher",
            "bucket",
            "ice_bucket",
            "pail",
            "trash_can",
            "recycling_bin",
            "hamper",
            "laundry_basket",
            "mop",
            "broom",
            "dustpan",
            "vacuum_cleaner",
            "floor_lamp",
            "air_filter",
            "air_purifier",
            "space_heater",
            "plywood",
        }:
            return True
        tokens = set(re.split(r"[_\W]+", cat))
        return bool(tokens & {"bucket", "trash", "hamper", "mop", "broom", "vacuum"})

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

    def _floor_fallback_allowed(self, record: dict, category: str) -> bool:
        """Allow floor fallback for movable, non-structural assets only."""
        if self._category_allows_floor(category):
            return True
        tokens = self._tokens(category)
        blocked = {
            "car", "vehicle", "truck", "bus", "motorcycle", "wall", "ceiling",
            "floor", "sink", "toilet", "bathtub", "cabinet", "counter", "table",
        }
        if tokens & blocked:
            return False
        interaction = ((record.get("edit_metadata") or {}).get("interaction") or {}).get("kind")
        mass = self._record_min_mass(record)
        return interaction == "manipulable" and (mass is None or mass <= 20.0)

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
        preferred = record.get("_preferred_category")
        if preferred:
            return preferred
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

    def _check_aabb_overlap(
        self,
        obj,
        exclude_names=None,
        margin=0.02,
        target_room=None,
        ignore_floor_coverings=False,
    ):
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
        collision_objects = list(self._scene_objects())
        robots = list(getattr(self.env, "robots", None) or [])
        collision_objects.extend(robots)
        for other in collision_objects:
            other_name = getattr(other, "name", None)
            if other_name in exclude:
                continue
            other_category = getattr(other, "category", "") or ""
            other_tokens = self._tokens(other_category)
            if other_tokens & {"floor", "floors"}:
                continue
            if ignore_floor_coverings and other_tokens & {"carpet", "rug"}:
                continue
            # Room filter: only check objects in the same room (fast pre-filter)
            if target_room and other not in robots:
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

    def _pick_task_from_category(
        self,
        category: str,
        skip_tasks: set[str] | None = None,
        used_task_names: set[str] | None = None,
        task_counts: Counter[str] | None = None,
        allowed_tasks: set[str] | None = None,
    ) -> str | None:
        """Pick a task with inverse-frequency weighting for large batches."""
        skip = skip_tasks or set()
        pool = allowed_tasks if allowed_tasks is not None else self._get_active_categories().get(category, set())
        available = [t for t in pool if t not in skip]
        if not available:
            return None
        if category == "retrieval_delivery":
            model_gaps = {
                task: self._uncovered_model_count_for_task(task)
                for task in available
            }
            uncovered_tasks = [task for task in available if model_gaps[task] > 0]
            if uncovered_tasks:
                weights = [model_gaps[task] for task in uncovered_tasks]
                return self.rng.choices(uncovered_tasks, weights=weights, k=1)[0]
        # Strongly prefer unused tasks, then inverse-weight historical usage.
        # This preserves task diversity after the first pass through a family.
        used = used_task_names or set()
        counts = task_counts or Counter()
        weights = [(1 if t in used else 12) / (1 + counts.get(t, 0)) for t in available]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def _uncovered_model_count_for_task(self, task_name: str) -> int:
        count = 0
        for category in SAFE_RETRIEVAL_ASSETS.get(task_name, ()):
            try:
                models = get_all_object_category_models(category=category)
            except Exception:
                models = []
            count += sum(
                self._used_target_models.get((category, model), 0) == 0
                and self._failed_target_models.get((category, model), 0)
                < self.config.max_failures_per_target_model
                for model in models
            )
        return count

    def _get_base_graph(self):
        """Return a cached graph for the unchanged base scene."""
        if self._base_graph_cache is None:
            self._base_graph_cache = self.snapshot()
        return copy.deepcopy(self._base_graph_cache)

    def _remove_spawned_objects_by_prefix(self, step_after=True):
        removed = 0
        for obj in list(self._scene_objects()):
            name = getattr(obj, "name", "")
            if name.startswith("online_env_"):
                try:
                    self.env.scene.remove_object(obj)
                    removed += 1
                except Exception:
                    pass
            for state_cls in (object_states.OnFire, object_states.Burnt):
                try:
                    if state_cls in obj.states and obj.states[state_cls].get_value():
                        obj.states[state_cls].set_value(False)
                except Exception:
                    pass
        if removed:
            self._clear_usd_selection()
            if step_after:
                self._step(3)
            print(f"[cleanup] removed {removed} spawned objects", flush=True)
        return removed

    def _cleanup_spawned_objects(self, prefer_reset=True):
        """Restore scene before a run.

        Env-A only adds online_env_a_* objects, so batch generation can remove
        those objects directly. Env-B/C pass prefer_reset=True because they can
        change native object states.
        """
        # The environment is already clean and robot-stabilized before the first
        # run. Avoid any reset/state round-trip here.
        if not self._run_started:
            self._run_started = True
            return

        # Remove generated objects, then explicitly restore native poses. This
        # works across changed object topology and prevents cumulative drift.
        try:
            self._remove_spawned_objects_by_prefix(step_after=False)
            if not og.sim.is_playing():
                og.sim.play()
            scene_objects = {getattr(obj, "name", ""): obj for obj in self._scene_objects()}
            robot_names = {
                getattr(robot_obj, "name", "")
                for robot_obj in (getattr(self.env, "robots", None) or [])
            }
            for name, (position, orientation) in self._clean_scene_poses.items():
                # The robot pose is owned by task-scoped spawn preparation.
                # Restoring the pre-task robot pose here would silently undo a
                # target-conditioned spawn the moment generate_env_a() runs its
                # opening cleanup; the reverted robot then sinks through the
                # floor during the task warmup (observed as the ~0.62 m robot
                # scene-integrity drift on Beechwood_0 open_fridge). Scene
                # integrity still accounts for the robot; only this native
                # pose restore skips it.
                if name in robot_names:
                    continue
                obj = scene_objects.get(name)
                if obj is None:
                    continue
                obj.set_position_orientation(position=position, orientation=orientation)
                try:
                    obj.keep_still()
                except Exception:
                    pass
            for state_id in list(self._mutated_native_states):
                name, state_cls = state_id
                obj = scene_objects.get(name)
                if obj is None or state_cls not in getattr(obj, "states", {}):
                    continue
                baseline = self._native_state_baselines.get(state_id)
                if baseline is not None:
                    obj.states[state_cls].set_value(baseline)
            self._mutated_native_states.clear()
            self._base_graph_cache = None
            self._step(3)
            return
        except Exception as exc:
            print(f"[cleanup] native-pose restore failed: {exc}; falling back to reset", flush=True)

        # Fallback: full env reset (restores physics, object poses, states)
        if hasattr(self.env, 'reset'):
            try:
                self.env.reset()
                self._base_graph_cache = None
                self._step(5)
                return
            except Exception:
                pass

        # Fallback 1: load previously saved state
        if hasattr(self, '_pre_run_state') and self._pre_run_state is not None:
            try:
                og.sim.load_state(self._pre_run_state, serialized=False)
                self._pre_run_state = None
                self._base_graph_cache = None
                self._step(3)
                return
            except Exception as e:
                print(f"[cleanup] load_state failed: {e}", flush=True)

        # Fallback 2: remove spawned objects by name prefix
        self._remove_spawned_objects_by_prefix()

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
        # The runner clears cross-sample skips in diversity mode. Any skips it
        # does pass are therefore hard rejects or per-sample retry exclusions
        # and must always be honored.
        skip = (skip_tasks or set()) | self._rejected_task_cache
        graph = cached_graph or self.snapshot()
        scene_furniture = self._build_scene_furniture_dict(graph)
        active_tasks = {
            category: set(tasks) for category, tasks in self._get_active_categories().items()
        }
        if "retrieval_delivery" in active_tasks:
            active_tasks["retrieval_delivery"] &= set(SAFE_RETRIEVAL_ASSETS)

        # Track used task names to avoid repetition (at task level, not category)
        task_counts = Counter(
            s.get("task", "") for s in self._checkpoint.get("successful_samples", []) if s.get("task")
        )
        used_task_names = set(task_counts)
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
        task_name = self._pick_task_from_category(
            category, skip, used_task_names, task_counts, active_tasks.get(category),
        )
        if not task_name:
            # Try other categories
            for alt_cat in sorted(active_tasks):
                if alt_cat == category:
                    continue
                task_name = self._pick_task_from_category(
                    alt_cat, skip, used_task_names, task_counts, active_tasks.get(alt_cat),
                )
                if task_name:
                    category = alt_cat
                    break

        if task_name:
            print(f"[llm] selected task: {task_name} (category={category})")
        return category, task_name

    def _choose_diverse_required_record(self, records: list[dict]) -> dict:
        if not records:
            raise ValueError("records must be non-empty")

        def key(record):
            category = self._choose_category(record)
            return (
                self._used_target_categories.get(category, 0),
                record.get("synset", ""),
                category,
            )

        least_used = min(self._used_target_categories.get(self._choose_category(r), 0) for r in records)
        candidates = [
            r for r in records
            if self._used_target_categories.get(self._choose_category(r), 0) == least_used
        ]
        return self.rng.choice(sorted(candidates, key=key))

    def _safe_retrieval_records(self, task_name: str) -> list[dict]:
        """Return one vetted, small physical asset for a retrieval task."""
        categories = SAFE_RETRIEVAL_ASSETS.get(task_name, ())
        if self.config.target_asset_category:
            if self.config.target_asset_category not in categories:
                return []
            categories = (self.config.target_asset_category,)
        records = []
        for category in categories:
            try:
                record = self._record_for_category(category)
            except (KeyError, ValueError):
                continue
            chosen_category = self._choose_category(record)
            mass = self._record_min_mass(record)
            if chosen_category not in categories:
                continue
            if mass is not None and mass > 2.0:
                continue
            records.append(record)
        model_gaps = {}
        for record in records:
            category = self._choose_category(record)
            try:
                models = get_all_object_category_models(category=category)
            except Exception:
                models = []
            model_gaps[category] = sum(
                self._used_target_models.get((category, model), 0) == 0
                and self._failed_target_models.get((category, model), 0)
                < self.config.max_failures_per_target_model
                for model in models
            )
        if model_gaps and max(model_gaps.values()) > 0:
            max_gap = max(model_gaps.values())
            candidates = [
                record for record in records
                if model_gaps.get(self._choose_category(record), 0) == max_gap
            ]
            return [self.rng.choice(candidates)]
        return [self._choose_diverse_required_record(records)] if records else []

    def _build_retrieval_instruction(
        self,
        task_name: str,
        target_room: str,
        item: dict,
        destination: dict | None = None,
    ) -> str:
        """Use the actual accepted support, never an LLM-invented location."""
        category = (item.get("category") or "object").replace("_", " ")
        placement = item.get("placement") or {}
        support = (placement.get("support_category") or "support surface").replace("_", " ")
        if support == "floors":
            support = "floor"
        room = (placement.get("room_id") or target_room or "the room").replace("_", " ")
        if task_name.startswith("deliver_") and destination:
            destination_category = destination["category"].replace("_", " ")
            if destination_category == "floors":
                destination_category = "floor"
            destination_room = destination["room_id"].replace("_", " ")
            return (
                f"Move the {category} from the {support} in {room} "
                f"to the {destination_category} in {destination_room}."
            )
        return f"Retrieve the {category} from the {support} in {room}."

    def _choose_delivery_destination(self, graph: dict, item: dict) -> dict | None:
        """Choose an exact, reachable, manipulation-height-safe destination."""
        placement = item.get("placement") or {}
        source_support = placement.get("support_object_id")
        source_room = placement.get("room_id")
        destination_tokens = {
            "table", "counter", "countertop", "cabinet", "shelf", "desk",
            "dresser", "bookcase", "nightstand", "island",
        }
        candidates = []
        for node in graph.get("nodes", []):
            if node.get("type") != "object" or node.get("id") == source_support:
                continue
            if str(node.get("id", "")).startswith("online_env_"):
                continue
            category = str(node.get("category") or "")
            rooms = node.get("rooms") or []
            if not rooms:
                continue
            room = rooms[0]
            if self._tokens(category) & {"floor", "floors"}:
                continue
            if not (self._tokens(category) & destination_tokens):
                continue
            receptacle = ((node.get("semantic") or {}).get("receptacle") or {})
            if not receptacle.get("can_support"):
                continue
            destination = self._delivery_destination_from_node(node)
            if destination is None:
                continue
            different_room = 0 if room != source_room else 1
            candidates.append((
                different_room,
                category,
                node.get("id"),
                room,
                destination,
            ))
        if candidates:
            best_rank = min(item[0] for item in candidates)
            *_, destination = self.rng.choice(
                sorted(
                    (item for item in candidates if item[0] == best_rank),
                    key=lambda item: item[:4],
                )
            )
        else:
            return None
        print(
            f"[delivery-destination] using validated native support "
            f"id={destination['object_id']} room={destination['room_id']}",
            flush=True,
        )
        return destination

    def _delivery_destination_from_node(self, node, fallback_room=None):
        bbox = node.get("bbox") or {}
        aabb_min = bbox.get("min")
        aabb_max = bbox.get("max")
        position = (node.get("pose") or {}).get("position")
        rooms = node.get("rooms") or ([fallback_room] if fallback_room else [])
        if not node.get("id") or not aabb_min or not aabb_max or not position or not rooms:
            print(
                f"[delivery-destination] reject {node.get('id')}: incomplete geometry "
                f"aabb={bool(aabb_min and aabb_max)} pose={bool(position)} rooms={rooms}",
                flush=True,
            )
            return None
        floor_height = self._floor_height_for_position(aabb_min)
        manipulation_height = evaluate_manipulation_height(
            "PLACE_ON_TOP",
            aabb_min[2],
            aabb_max[2],
            floor_height,
            self.config.min_manipulation_height,
            self.config.max_manipulation_height,
        )
        if not manipulation_height.get("eligible"):
            print(
                f"[delivery-destination] reject {node['id']}: manipulation height "
                f"{manipulation_height}",
                flush=True,
            )
            return None
        if not self._delivery_top_surface_feasible(node):
            return None
        xy_diagonal = math.hypot(
            float(aabb_max[0]) - float(aabb_min[0]),
            float(aabb_max[1]) - float(aabb_min[1]),
        )
        preferred_center_distance = max(
            1.75,
            xy_diagonal / (2.0 * math.tan(math.radians(25.0))) + 0.25,
        )
        robot_approach = self._validate_task_approach_position(
            position,
            set(rooms),
            target_aabb_xy=(aabb_min[:2], aabb_max[:2]),
            # The operation pose must be outside the destination support. If
            # it is excluded here, a point under the table can be accepted.
            target_object_id=None,
            preferred_center_distance=preferred_center_distance,
        )
        if not robot_approach.get("ok"):
            print(
                f"[delivery-destination] reject {node['id']}: no external approach "
                f"{robot_approach}",
                flush=True,
            )
            return None
        if not self._primary_camera_operation_visible(node, robot_approach):
            print(
                f"[delivery-destination] reject {node['id']}: "
                "not fully visible from robot operation pose",
                flush=True,
            )
            return None
        return {
            "object_id": node["id"],
            "category": node.get("category"),
            "room_id": rooms[0],
            "placement_mode": "on_top",
            "robot_approach": robot_approach,
            "manipulation_height": manipulation_height,
        }

    def _primary_camera_operation_visible(self, node, approach):
        """Validate a fine PLACE target from its reserved robot approach."""
        candidate_xy = (approach or {}).get("candidate_position_xy")
        position = (node.get("pose") or {}).get("position")
        object_id = node.get("id")
        if not candidate_xy or not position or not object_id:
            return False
        robot = self.env.robots[0]
        sensors = getattr(robot, "sensors", None) or getattr(robot, "_sensors", None) or {}
        items = list(sensors.items()) if isinstance(sensors, dict) else [
            (getattr(sensor, "name", f"sensor_{index}"), sensor)
            for index, sensor in enumerate(sensors)
        ]
        cameras = [item for item in items if self._looks_like_camera_sensor(item[1])]
        primary = [item for item in cameras if self._is_primary_robot_camera(*item)]
        selected = primary[0] if primary else (cameras[0] if cameras else None)
        if selected is None:
            return False
        try:
            _, sensor = selected
            robot_position, robot_orientation = robot.get_position_orientation()
            camera_position, camera_orientation = sensor.get_position_orientation()
            robot_matrix = T.pose2mat((robot_position, robot_orientation))
            camera_matrix = T.pose2mat((camera_position, camera_orientation))
            relative_camera = th.linalg.inv(robot_matrix) @ camera_matrix
            yaw = math.atan2(
                float(position[1]) - float(candidate_xy[1]),
                float(position[0]) - float(candidate_xy[0]),
            )
            candidate_position = robot_position.clone()
            candidate_position[:2] = th.as_tensor(candidate_xy, dtype=th.float32)
            candidate_orientation = T.euler2quat(
                th.tensor([0.0, 0.0, yaw], dtype=th.float32)
            )
            candidate_camera = T.pose2mat(
                (candidate_position, candidate_orientation)
            ) @ relative_camera
            candidate_camera_position = candidate_camera[:3, 3]
            bbox = node.get("bbox") or {}
            lower = bbox.get("min") or position
            upper = bbox.get("max") or position
            size = np.asarray(upper, dtype=np.float32) - np.asarray(
                lower, dtype=np.float32
            )
            operation_point = np.asarray(
                [
                    0.5 * (float(lower[0]) + float(upper[0])),
                    0.5 * (float(lower[1]) + float(upper[1])),
                    0.5 * (float(lower[2]) + float(upper[2])),
                ],
                dtype=np.float32,
            )
            if float(np.linalg.norm(size[:2])) >= 1.0 or float(size[2]) >= 0.6:
                operation_point[2] = float(lower[2]) + 0.05 * float(size[2])
            # The expert performs an in-place base/head look-at after MOVE.
            # Test that reachable aimed view, not the unrelated spawn head pose.
            candidate_camera_orientation = self._look_at_quat(
                np.asarray(candidate_camera_position.cpu(), dtype=np.float32),
                operation_point,
            )
            resolution = self._sensor_resolution(sensor) or {}
            visibility = self._geometric_camera_visibility(
                candidate_camera_position,
                th.as_tensor(candidate_camera_orientation, dtype=th.float32),
                {object_id},
                width=int(resolution.get("width") or 320),
                height=int(resolution.get("height") or 240),
            ).get(object_id)
            if not visibility:
                return False
            x1, y1, x2, _ = visibility["bbox_xyxy"]
            width, height = visibility["image_size"]
            margin = max(2, int(round(min(width, height) * 0.01)))
            # PLACE needs a complete top and lateral extent; floor furniture
            # may continue below the image exactly as in expert validation.
            return x1 >= margin and y1 >= margin and x2 < width - margin
        except Exception as exc:
            print(
                f"[delivery-destination] primary camera validation failed for "
                f"{object_id}: {exc!r}",
                flush=True,
            )
            return False

    def _delivery_top_surface_feasible(self, node) -> bool:
        """Physically verify that a placeable top surface exists.

        The lexical supports_on_top rule accepts open-top fixtures
        (Merom_0 bottom_cabinet_no_top_vdedzt_0: deliver_food PLACE
        SAMPLING_ERROR on both expert attempts — the AABB top is empty air,
        so the OG onTop sampler finds no pose). Cast a downward raycast fan
        from just above the live AABB top and require a majority of hits on
        the destination itself near that height: a real top panel answers on
        nearly every ray, an open box answers only on its rim. Fail-closed:
        any exception makes the destination ineligible and the existing
        resample / spawned-support contract takes over.
        """
        object_id = node.get("id")
        if not object_id:
            return False
        try:
            obj = self.env.scene.object_registry("name", object_id, None)
            if obj is None:
                return False
            target_path = str(getattr(obj, "prim_path", "")).rstrip("/")
            if not target_path:
                return False
            lower, upper = obj.aabb
            lower = np.asarray(lower.cpu(), dtype=float)
            upper = np.asarray(upper.cpu(), dtype=float)
            top_z = float(upper[2])
            origin_z = top_z + 0.10
            # Inset the fan 10% so edge rays cannot graze neighboring
            # fixtures or the fixture's own side walls.
            x0 = float(lower[0]) + 0.10 * float(upper[0] - lower[0])
            x1 = float(upper[0]) - 0.10 * float(upper[0] - lower[0])
            y0 = float(lower[1]) + 0.10 * float(upper[1] - lower[1])
            y1 = float(upper[1]) - 0.10 * float(upper[1] - lower[1])
            grid = 5
            confirmed = 0
            total = 0
            for row in range(grid):
                for col in range(grid):
                    fx = row / (grid - 1)
                    fy = col / (grid - 1)
                    origin = [x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), origin_z]
                    ray = og.sim.psqi.raycast_closest(
                        origin=origin,
                        dir=[0.0, 0.0, -1.0],
                        distance=3.0,
                    )
                    total += 1
                    if not ray.get("hit"):
                        continue
                    hit_path = str(ray.get("rigidBody") or ray.get("collision") or "")
                    hit_z = origin_z - float(ray.get("distance", 3.0))
                    if target_path in hit_path and hit_z >= top_z - 0.05:
                        confirmed += 1
            feasible = confirmed * 2 > total
            if not feasible:
                print(
                    f"[delivery-destination] physical reject {object_id}: "
                    f"top-surface raycasts confirmed {confirmed}/{total}",
                    flush=True,
                )
            return feasible
        except Exception as exc:
            print(
                f"[delivery-destination] physical reject {object_id}: "
                f"feasibility check error {exc!r}",
                flush=True,
            )
            return False

    def _approach_pose_clears_object(self, approach, target_position, blocker):
        """Check that a previously reserved manipulation pose remains usable."""
        candidate_xy = (approach or {}).get("candidate_position_xy")
        if not candidate_xy or blocker is None or not getattr(self.env, "robots", None):
            return False
        try:
            robot = self.env.robots[0]
            robot_position, robot_orientation = robot.get_position_orientation()
            robot_position = np.asarray(robot_position.cpu(), dtype=float)
            live_points = np.asarray(robot.collision_points_world.cpu(), dtype=float)
            live_rotation = np.asarray(T.quat2mat(robot_orientation).cpu(), dtype=float)
            local_points = (live_points - robot_position) @ live_rotation
            yaw = math.atan2(
                float(target_position[1]) - float(candidate_xy[1]),
                float(target_position[0]) - float(candidate_xy[0]),
            )
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                dtype=float,
            )
            candidate_position = np.asarray(
                [float(candidate_xy[0]), float(candidate_xy[1]), robot_position[2]],
                dtype=float,
            )
            candidate_points = local_points @ rotation.T + candidate_position
            lower, upper = blocker.aabb
            lower = np.asarray(lower.cpu(), dtype=float) - 0.10
            upper = np.asarray(upper.cpu(), dtype=float) + 0.10
            return not np.any(
                np.all(candidate_points >= lower, axis=1)
                & np.all(candidate_points <= upper, axis=1)
            )
        except Exception as exc:
            print(
                f"[delivery-support] reserved approach check failed: {exc!r}",
                flush=True,
            )
            return False

    def _spawn_delivery_destination_support(self, run_id, graph, source_room, source_item):
        """Place a real support fixture when a scene has no legal destination."""
        candidate_rooms = []
        excluded_rooms = {source_room} if source_room else set()
        for _ in range(3):
            room = self._choose_support_bootstrap_room(
                graph, excluded_rooms=excluded_rooms,
            )
            if room is None:
                break
            candidate_rooms.append(room)
            excluded_rooms.add(room)
        if source_room and source_room not in candidate_rooms:
            candidate_rooms.append(source_room)
        if not candidate_rooms:
            fallback_room = self._choose_support_bootstrap_room(graph)
            if fallback_room:
                candidate_rooms.append(fallback_room)
        if not candidate_rooms:
            return ({
                "ok": False,
                "semantic_role": "delivery_destination",
                "errors": [{"error": "no_reachable_delivery_support_room"}],
            }, None)

        # A delivery support is independent of the already accepted source
        # placement. Retry it locally instead of repeating LLM generation,
        # source placement, settling, and camera selection. If only one room is
        # reachable, repeated entries deliberately resample both model and pose.
        max_local_attempts = 6
        compact_models = self._compact_support_models("coffee_table")
        source_placement = (source_item or {}).get("placement") or {}
        source_pose = (
            (source_item or {}).get("pose")
            or (source_item or {}).get("final_pose_before_warmup")
            or source_placement.get("pose")
            or {}
        )
        attempt_candidates = [
            (room, model)
            for room in candidate_rooms
            for model in compact_models
            if self._support_model_has_floor_pose(
                "coffee_table",
                model,
                room,
                graph,
                avoid_position=source_pose.get("position"),
            )
        ][:max_local_attempts]
        if not attempt_candidates:
            print(
                f"[delivery-support] preflight found no footprint-clear model/room "
                f"candidate rooms={candidate_rooms}",
                flush=True,
            )
            return ({
                "ok": False,
                "semantic_role": "delivery_destination",
                "errors": [{"error": "no_footprint_clear_delivery_support_pose"}],
            }, None)
        failures = []
        for attempt_index, (target_room, preferred_model) in enumerate(
            attempt_candidates, 1
        ):
            record = copy.deepcopy(self._record_for_category("coffee_table"))
            record["_generated_support_fixture"] = True
            record["_preferred_models"] = [preferred_model]
            object_name = f"{run_id}_delivery_support"
            if attempt_index > 1:
                object_name = f"{object_name}_{attempt_index:02d}"
            result = self.add_task_asset(
                record=record,
                object_name=object_name,
                target_room=target_room,
                semantic_role="task_support",
                cached_graph=graph,
                avoid_position=source_pose.get("position"),
            )
            if not result.get("ok"):
                failures.append({
                    "attempt": attempt_index,
                    "room": target_room,
                    "model": result.get("model"),
                    "errors": result.get("errors") or [],
                })
                print(
                    f"[delivery-support] attempt {attempt_index}/{max_local_attempts} failed "
                    f"room={target_room} model={result.get('model')} "
                    f"errors={[item.get('error') for item in result.get('errors') or []]}",
                    flush=True,
                )
                continue

            self._step(10)
            live_graph = self.snapshot()
            node = next(
                (
                    item for item in live_graph.get("nodes") or []
                    if item.get("id") == result.get("object_name")
                ),
                None,
            )
            if node is None:
                live_object = self.env.scene.object_registry(
                    "name", result.get("object_name"), None
                )
                if live_object is not None:
                    info = self._collect_object_info(live_object)
                    node = {
                        "id": info["name"],
                        "category": info["category"],
                        "pose": {"position": info["position"]},
                        "bbox": {"min": info["aabb_min"], "max": info["aabb_max"]},
                        "rooms": info["rooms"],
                    }
            destination = self._delivery_destination_from_node(
                node or {}, fallback_room=target_room
            )
            live_support = self.env.scene.object_registry(
                "name", result.get("object_name"), None
            )
            if destination is not None and not self._approach_pose_clears_object(
                source_placement.get("robot_approach")
                or ((source_item or {}).get("validation") or {}).get("robot_approach"),
                source_pose.get("position"),
                live_support,
            ):
                print(
                    f"[delivery-support] attempt {attempt_index}/{max_local_attempts} rejected: "
                    "occupies source manipulation pose",
                    flush=True,
                )
                destination = None
            if destination is not None:
                destination["generated_support"] = True
                destination["generation_attempt"] = attempt_index
                return result, destination

            failures.append({
                "attempt": attempt_index,
                "room": target_room,
                "model": result.get("model"),
                "errors": [{"error": "generated_delivery_support_not_manipulable"}],
            })
            self._remove_object_safe_by_name(result.get("object_name"))
            self._forget_generated_placement(result)
            self._step(3)

        return ({
            "ok": False,
            "semantic_role": "delivery_destination",
            "errors": [{
                "error": "delivery_support_attempts_exhausted",
                "attempts": failures,
            }],
        }, None)

    def _compact_support_models(self, category):
        """Return installed support models that leave room for robot operation."""
        installed = set(get_all_object_category_models(category=category))
        metadata_root = Path(get_dataset_path("behavior-1k-assets")) / "objects" / category
        ranked = []
        for model in installed:
            metadata_path = metadata_root / model / "misc" / "metadata.json"
            try:
                size = json.loads(metadata_path.read_text(encoding="utf-8"))["bbox_size"]
                width, depth, height = (float(value) for value in size[:3])
            except (OSError, KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (width, depth, height)):
                continue
            if min(width, depth) < 0.30 or max(width, depth) > 0.75:
                continue
            if not (
                self.config.min_manipulation_height
                <= height
                <= self.config.max_manipulation_height
            ):
                continue
            ranked.append((width * depth, model))
        return [model for _, model in sorted(ranked)[:12]]

    def _support_model_has_floor_pose(
        self, category, model, target_room, graph, avoid_position=None
    ):
        """Preflight a support footprint before importing its USD asset."""
        metadata_path = (
            Path(get_dataset_path("behavior-1k-assets"))
            / "objects"
            / category
            / model
            / "misc"
            / "metadata.json"
        )
        try:
            size = json.loads(metadata_path.read_text(encoding="utf-8"))["bbox_size"]
            half_extent = np.asarray(size[:2], dtype=float) * 0.5 + 0.02
            if not np.all(np.isfinite(half_extent)):
                return False
            room_pixels = self._robot_reachable_room_pixels().get(target_room) or []
            if not room_pixels:
                return False
            blockers = []
            for node in graph.get("nodes", []):
                if node.get("type") != "object":
                    continue
                rooms = node.get("rooms") or []
                if rooms and target_room not in rooms:
                    continue
                if self._tokens(node.get("category") or "") & NON_BLOCKING_NAVIGATION_CATEGORIES:
                    continue
                bbox = node.get("bbox") or {}
                lower = bbox.get("min")
                upper = bbox.get("max")
                if lower and upper:
                    blockers.append((
                        np.asarray(lower[:2], dtype=float),
                        np.asarray(upper[:2], dtype=float),
                    ))
            for robot in getattr(self.env, "robots", None) or []:
                lower, upper, _ = self._safe_aabb(robot)
                if lower and upper:
                    blockers.append((
                        np.asarray(lower[:2], dtype=float),
                        np.asarray(upper[:2], dtype=float),
                    ))
            trav_map = self.env.scene.trav_map
            avoid_xy = (
                np.asarray(avoid_position[:2], dtype=float)
                if avoid_position is not None
                else None
            )
            for pixel in room_pixels:
                center = np.asarray(trav_map.map_to_world(pixel).cpu(), dtype=float)[:2]
                if avoid_xy is not None and float(np.linalg.norm(center - avoid_xy)) < 1.0:
                    continue
                proposed_lo = center - half_extent
                proposed_hi = center + half_extent
                if any(
                    np.all(proposed_lo < blocker_hi)
                    and np.all(proposed_hi > blocker_lo)
                    for blocker_lo, blocker_hi in blockers
                ):
                    continue
                if self._hypothetical_support_has_operation_approach(
                    room_pixels, center, proposed_lo, proposed_hi
                ):
                    return True
            return False
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def _forget_generated_placement(self, result):
        """Remove placement accounting for an object rejected after spawning."""
        object_name = result.get("object_name")
        placement = result.get("placement") or {}
        support_id = placement.get("support_object_id")
        support_key = support_id or "__floor__"
        if self._placed_on_support.get(support_key, 0) > 0:
            self._placed_on_support[support_key] -= 1
            if self._placed_on_support[support_key] == 0:
                self._placed_on_support.pop(support_key, None)
        self._placement_support_map.pop(object_name, None)
        self._floor_placed_objects.discard(object_name)

    def prepare_native_task_robot_spawn(self, task_name: str) -> str | None:
        """Choose the official native target before a serial task is generated."""
        if task_name not in NATIVE_TASK_TARGET_TOKENS:
            return None
        self._cleanup_spawned_objects(prefer_reset=True)
        self._robot_component_cache = {}
        self._reachable_room_pixels_cache = {}
        self._base_graph_cache = None
        target = self._select_native_task_target(
            task_name, self.snapshot(), require_robot_approach=False,
        )
        if target is not None:
            target["task_name"] = task_name
        self._prepared_native_target = target
        self._prepared_native_target_id = target.get("object_id") if target else None
        return self._prepared_native_target_id

    def bind_prepared_native_task_spawn(self, task_name: str) -> bool:
        """Bind the physically validated target-conditioned robot pose."""
        target = self._prepared_native_target
        if not target or target.get("task_name") != task_name:
            return False
        obj = self.env.scene.object_registry("name", target["object_id"], None)
        if obj is None:
            return False
        robot_position, _ = self.env.robots[0].get_position_orientation()
        robot_room = self.env.scene.seg_map.get_room_instance_by_point(robot_position[:2])
        lower, upper = obj.aabb
        nearest_xy = th.minimum(th.maximum(robot_position[:2], lower[:2]), upper[:2])
        distance = float(th.linalg.norm(robot_position[:2] - nearest_xy))
        if robot_room != target.get("room_id") or distance > 1.35:
            print(
                f"[robot-spawn] target binding rejected task={task_name} "
                f"target={target['object_id']} room={robot_room} distance={distance:.3f}",
                flush=True,
            )
            return False
        target["robot_approach"] = {
            "ok": True,
            "reason": "target_conditioned_spawn",
            "horizontal_distance": distance,
            "distance_reference": "aabb_edge",
            "max_horizontal_distance": 1.35,
            "candidate_position_xy": self._to_list(robot_position[:2]),
            "candidate_room": robot_room,
            "target_rooms": [target["room_id"]],
            "native_occupancy_rejections": 0,
        }
        print(
            f"[robot-spawn] target binding accepted task={task_name} "
            f"target={target['object_id']} distance={distance:.3f}",
            flush=True,
        )
        return True

    def invalidate_robot_reachability(self) -> None:
        """Refresh component-dependent caches after an in-process robot move."""
        self._robot_component_cache = {}
        self._reachable_room_pixels_cache = {}
        self._base_graph_cache = None

    def _select_native_task_target(
        self, task_name: str, graph: dict, require_robot_approach: bool = True,
    ) -> dict | None:
        """Find a real stateful scene object compatible with the task verb."""
        if (
            require_robot_approach
            and self._prepared_native_target
            and self._prepared_native_target.get("task_name") == task_name
            and (self._prepared_native_target.get("robot_approach") or {}).get("ok")
        ):
            target = self._prepared_native_target
            self._prepared_native_target = None
            self._prepared_native_target_id = None
            return target
        self._last_native_target_rejection = None
        tokens = NATIVE_TASK_TARGET_TOKENS.get(task_name, ())
        required_state = "Open" if task_name.startswith(("open_", "close_")) else "ToggledOn"
        candidates = []
        exact_seen = False
        for node in graph.get("nodes", []):
            if node.get("type") != "object":
                continue
            if (
                self.config.target_native_object_id
                and node.get("id") != self.config.target_native_object_id
            ):
                continue
            if (
                require_robot_approach
                and self._prepared_native_target_id
                and node.get("id") != self._prepared_native_target_id
            ):
                continue
            if (
                not self.config.target_native_object_id
                and node.get("id") in self._rejected_native_target_cache
            ):
                continue
            if self.config.target_native_object_id:
                exact_seen = True
            category = str(node.get("category") or "").lower()
            if not any(token in category for token in tokens):
                if self.config.target_native_object_id:
                    self._last_native_target_rejection = {
                        "stage": "identity",
                        "object_id": node.get("id"),
                        "category": category,
                        "task": task_name,
                        "reason": "category is incompatible with the requested task",
                    }
                continue
            if required_state not in set(node.get("available_states") or []):
                if self.config.target_native_object_id:
                    self._last_native_target_rejection = {
                        "stage": "state_capability",
                        "object_id": node.get("id"),
                        "required_state": required_state,
                        "reason": "required state is unavailable",
                    }
                continue
            rooms = node.get("rooms") or []
            if not rooms:
                if self.config.target_native_object_id:
                    self._last_native_target_rejection = {
                        "stage": "room",
                        "object_id": node.get("id"),
                        "reason": "target has no official room assignment",
                    }
                continue
            position = (node.get("pose") or {}).get("position")
            if not position:
                if self.config.target_native_object_id:
                    self._last_native_target_rejection = {
                        "stage": "pose",
                        "object_id": node.get("id"),
                        "reason": "target pose is missing",
                    }
                continue
            manipulation_height = self._native_target_manipulation_height(task_name, node)
            if not manipulation_height.get("eligible"):
                if self.config.target_native_object_id:
                    print(
                        f"[native-target] rejected exact object={node.get('id')} "
                        f"height={manipulation_height}",
                        flush=True,
                    )
                    self._last_native_target_rejection = {
                        "stage": "manipulation_height",
                        "object_id": node.get("id"),
                        "detail": manipulation_height,
                    }
                continue
            bbox = node.get("bbox") or {}
            aabb_min = bbox.get("min")
            aabb_max = bbox.get("max")
            target_aabb_xy = (
                (aabb_min[:2], aabb_max[:2]) if aabb_min and aabb_max else None
            )
            robot_approach = None
            prepared_current_pose = (
                require_robot_approach
                and node.get("id") == self._prepared_native_target_id
                and target_aabb_xy is not None
            )
            if prepared_current_pose:
                robot_position, _ = self.env.robots[0].get_position_orientation()
                robot_room = self.env.scene.seg_map.get_room_instance_by_point(
                    robot_position[:2]
                )
                nearest_xy = th.minimum(
                    th.maximum(robot_position[:2].cpu(), th.as_tensor(target_aabb_xy[0])),
                    th.as_tensor(target_aabb_xy[1]),
                )
                distance = float(th.linalg.norm(robot_position[:2].cpu() - nearest_xy))
                if robot_room in set(rooms) and distance <= 1.35:
                    robot_approach = {
                        "ok": True,
                        "reason": "target_conditioned_spawn",
                        "horizontal_distance": distance,
                        "distance_reference": "aabb_edge",
                        "max_horizontal_distance": 1.35,
                        "candidate_position_xy": self._to_list(robot_position[:2]),
                        "candidate_room": robot_room,
                        "target_rooms": sorted(set(rooms)),
                        "native_occupancy_rejections": 0,
                    }
            if require_robot_approach and robot_approach is None:
                robot_approach = self._validate_task_approach_position(
                    position,
                    set(rooms),
                    target_aabb_xy=target_aabb_xy,
                    max_horizontal_distance=1.35,
                    # Keep the fixture in the blocker set so the saved pose
                    # cannot intersect the object it will actuate.
                    target_object_id=None,
                )
            if require_robot_approach and not robot_approach.get("ok"):
                if self.config.target_native_object_id:
                    print(
                        f"[native-target] rejected exact object={node.get('id')} "
                        f"approach={robot_approach}",
                        flush=True,
                    )
                    self._last_native_target_rejection = {
                        "stage": "robot_approach",
                        "object_id": node.get("id"),
                        "detail": robot_approach,
                    }
                continue
            exact = 0 if category in tokens else 1
            candidates.append((
                exact, category, node.get("id"), rooms[0], robot_approach,
                manipulation_height,
            ))
        if not candidates:
            if self.config.target_native_object_id and not exact_seen:
                self._last_native_target_rejection = {
                    "stage": "identity",
                    "object_id": self.config.target_native_object_id,
                    "reason": "exact target is absent from the live scene graph",
                }
            return None
        used_object_ids = Counter()
        for sample in self._checkpoint.get("successful_samples", []):
            if sample.get("task") != task_name:
                continue
            diversity = sample.get("diversity") or {}
            for object_id in diversity.get("target_object_ids") or []:
                used_object_ids[object_id] += 1
        min_used = min(used_object_ids.get(item[2], 0) for item in candidates)
        least_used = [item for item in candidates if used_object_ids.get(item[2], 0) == min_used]
        _, category, object_id, room_id, robot_approach, manipulation_height = self.rng.choice(
            sorted(least_used, key=lambda item: item[:4])
        )
        if require_robot_approach:
            self._prepared_native_target_id = None
        return {
            "object_id": object_id,
            "category": category,
            "room_id": room_id,
            "robot_approach": robot_approach,
            "manipulation_height": manipulation_height,
        }

    def reject_native_target(self, object_id: str | None, reason: str) -> None:
        """Avoid reselecting a failed automatic fixture in the same scene process."""
        if not object_id or self.config.target_native_object_id:
            return
        self._rejected_native_target_cache.add(object_id)
        print(
            f"[native-target] cached rejection object={object_id} reason={reason} "
            f"count={len(self._rejected_native_target_cache)}",
            flush=True,
        )

    @staticmethod
    def _native_task_state_spec(task_name):
        if task_name.startswith(("open_", "close_")):
            state_cls = object_states.Open
            state_key = "open"
            desired_final = task_name.startswith("open_")
        else:
            state_cls = object_states.ToggledOn
            state_key = "toggled_on"
            desired_final = task_name.startswith("turn_on_")
        return state_cls, state_key, desired_final

    def _prepare_native_task_initial_state(self, task_name, target):
        """Preflight the official state setter both ways, ending at task initial state."""
        object_id = target["object_id"]
        obj = self.env.scene.object_registry("name", object_id, None)
        state_cls, state_key, desired_final = self._native_task_state_spec(task_name)
        result = {
            "ok": False,
            "object_id": object_id,
            "category": target.get("category"),
            "room_id": target.get("room_id"),
            "states": {state_key: not desired_final},
            "expected_task_final_states": {state_key: desired_final},
            "semantic_roles": ["task_target", "task_initial_state"],
        }
        if obj is None or state_cls not in getattr(obj, "states", {}):
            result["error"] = "native state target is unavailable"
            return result
        state_id = (object_id, state_cls)
        try:
            current = bool(obj.states[state_cls].get_value())
            self._native_state_baselines.setdefault(state_id, current)
            required_initial = not desired_final
            transitions = []

            def apply_and_observe(label, requested):
                before = bool(obj.states[state_cls].get_value())
                setter_returned = True
                if before != requested:
                    setter_returned = bool(obj.states[state_cls].set_value(requested))
                    self._mutated_native_states.add(state_id)
                immediate = bool(obj.states[state_cls].get_value())
                self._clear_usd_selection()
                self._step(5)
                settled = bool(obj.states[state_cls].get_value())
                transition = {
                    "phase": label,
                    "requested": requested,
                    "before": before,
                    "setter_returned": setter_returned,
                    "immediate": immediate,
                    "settled": settled,
                    "ok": setter_returned and immediate == requested and settled == requested,
                }
                transitions.append(transition)
                return transition["ok"]

            initial_ok = apply_and_observe("task_initial", required_initial)
            final_ok = initial_ok and apply_and_observe("task_final_preflight", desired_final)
            restore_ok = apply_and_observe("restore_task_initial", required_initial)
            observed = bool(obj.states[state_cls].get_value())
            result["official_state_transition_preflight"] = transitions
            result["observed_initial_state"] = observed
            result["ok"] = initial_ok and final_ok and restore_ok and observed == required_initial
            if not result["ok"]:
                result["error"] = "official native state transition is not stable and reversible"
        except Exception as exc:
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()
        return result

    def _native_target_manipulation_height(self, task_name, node):
        bbox = node.get("bbox") or {}
        aabb_min = bbox.get("min")
        aabb_max = bbox.get("max")
        if not aabb_min or not aabb_max:
            return {"required": True, "eligible": False, "reason": "native target AABB missing"}
        heights = [float(value) for value in self.env.scene.trav_map.floor_heights]
        lower_z = float(aabb_min[2])
        below = [height for height in heights if height <= lower_z + 0.10]
        floor_height = max(below) if below else min(
            heights, key=lambda height: abs(height - lower_z)
        )
        if task_name.startswith("open_"):
            primitive = "OPEN"
        elif task_name.startswith("close_"):
            primitive = "CLOSE"
        elif task_name.startswith("turn_on_"):
            primitive = "TOGGLE_ON"
        else:
            primitive = "TOGGLE_OFF"
        return evaluate_manipulation_height(
            primitive,
            aabb_min[2],
            aabb_max[2],
            floor_height,
            self.config.min_manipulation_height,
            self.config.max_manipulation_height,
        )

    @staticmethod
    def _native_task_instruction(task_name: str, target: dict) -> str:
        verb = "Open" if task_name.startswith("open_") else "Close" if task_name.startswith("close_") else "Turn on" if "turn_on" in task_name else "Turn off"
        category = target["category"].replace("_", " ")
        room = target["room_id"].replace("_", " ")
        return f"{verb} the {category} in {room}."

    def _room_camera_priority(self, room: str, graph: dict) -> int:
        """Lower is better for the documented fixed room-camera strategies."""
        room_lower = str(room or "").lower()
        if any(token in room_lower for token in ("storage", "closet", "empty")):
            priority = 3
        elif any(token in room_lower for token in ("corridor", "entryway")):
            priority = 2
        elif any(token in room_lower for token in ("utility", "garage")):
            priority = 1
        else:
            priority = 0
        visible_anchors = 0
        for node in graph.get("nodes", []):
            if node.get("type") != "object" or room not in (node.get("rooms") or []):
                continue
            if not (self._tokens(node.get("category") or "") & STRUCTURAL_CATEGORIES):
                visible_anchors += 1
        if visible_anchors < 2:
            priority += 2
        if not ((graph.get("navigation") or {}).get("room_centers") or {}).get(room):
            priority += 2
        return priority

    def _robot_component_pixels(self, floor: int):
        """Return the initial robot's eroded traversability component."""
        if not getattr(self.env, "robots", None):
            return None
        robot = self.env.robots[0]
        cache_key = (floor, getattr(robot, "name", "robot"))
        cached = self._robot_component_cache.get(cache_key)
        if cached is not None:
            return cached
        trav_map = self.env.scene.trav_map
        traversable = trav_map._erode_trav_map(
            th.clone(trav_map.floor_map[floor]), robot=robot
        )
        extra_clearance = float(self.config.expert_base_clearance_margin)
        if extra_clearance > 0:
            clearance_pixels = int(
                math.ceil(extra_clearance / float(trav_map.map_resolution))
            )
            kernel_size = max(1, 2 * clearance_pixels + 1)
            traversable = th.as_tensor(
                cv2.erode(
                    traversable.cpu().numpy(),
                    np.ones((kernel_size, kernel_size), dtype=np.uint8),
                ),
                device=traversable.device,
            )
        free = traversable.cpu().numpy() != 0
        robot_position, _ = robot.get_position_orientation()
        source_pixel = trav_map.world_to_map(robot_position[:2]).to(traversable.device).long()
        source_pixel[0].clamp_(0, traversable.shape[0] - 1)
        source_pixel[1].clamp_(0, traversable.shape[1] - 1)
        start = (int(source_pixel[0]), int(source_pixel[1]))
        if not free[start]:
            traversable_pixels = np.argwhere(free)
            if not len(traversable_pixels):
                return None
            nearest = int(
                np.argmin(
                    np.linalg.norm(
                        traversable_pixels.astype(float) - np.asarray(start, dtype=float),
                        axis=1,
                    )
                )
            )
            start = tuple(int(value) for value in traversable_pixels[nearest])

        # Match the physical expert's path graph exactly: diagonal motion is
        # valid only when both adjacent orthogonal cells are free. Generic
        # 8-connected components can cross the corner of two obstacles and
        # accept placements that the physical base can never reach.
        reachable = np.zeros(free.shape, dtype=bool)
        reachable[start] = True
        queue = deque([start])
        neighbors = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        while queue:
            row, col = queue.popleft()
            for drow, dcol in neighbors:
                nrow, ncol = row + drow, col + dcol
                if not (0 <= nrow < free.shape[0] and 0 <= ncol < free.shape[1]):
                    continue
                if not free[nrow, ncol] or reachable[nrow, ncol]:
                    continue
                if drow and dcol and (
                    not free[row + drow, col] or not free[row, col + dcol]
                ):
                    continue
                reachable[nrow, ncol] = True
                queue.append((nrow, ncol))
        cached = {
            "source_label": 1,
            "pixels": th.nonzero(th.as_tensor(reachable, device=traversable.device)),
        }
        self._robot_component_cache[cache_key] = cached
        return cached

    def _robot_reachable_room_pixels(self, floor: int | None = None) -> dict:
        """Group connected robot navigation pixels by official room instance."""
        if not getattr(self.env, "robots", None):
            return {}
        trav_map = self.env.scene.trav_map
        if floor is None:
            robot_position, _ = self.env.robots[0].get_position_orientation()
            floor = min(
                range(trav_map.n_floors),
                key=lambda index: abs(
                    float(trav_map.floor_heights[index]) - float(robot_position[2])
                ),
            )
        if floor in self._reachable_room_pixels_cache:
            return self._reachable_room_pixels_cache[floor]
        component = self._robot_component_pixels(floor)
        room_pixels = defaultdict(list)
        if component is not None:
            for pixel in component["pixels"]:
                xy = trav_map.map_to_world(pixel)
                room = self.env.scene.seg_map.get_room_instance_by_point(xy[:2])
                if room:
                    room_pixels[room].append(pixel)
        result = dict(room_pixels)
        self._reachable_room_pixels_cache[floor] = result
        return result

    def _choose_safe_target_room(self, record: dict, graph: dict, preferred_room: str | None = None) -> str | None:
        """Balance source rooms while retaining a physically valid fallback."""
        rooms = [node["name"] for node in graph.get("nodes", []) if node.get("type") == "room"]
        reachable_rooms = set(self._robot_reachable_room_pixels())
        if not reachable_rooms:
            return None
        room_counts = Counter()
        for sample in self._checkpoint.get("successful_samples", []):
            diversity = sample.get("diversity") or {}
            source_rooms = diversity.get("source_rooms") or []
            if source_rooms:
                for room in source_rooms:
                    room_counts[room] += 1
            elif diversity.get("target_room"):
                room_counts[diversity["target_room"]] += 1
        candidates = []
        category = self._choose_category(record)
        for room in rooms:
            if reachable_rooms and room not in reachable_rooms:
                continue
            if room in self._rejected_rooms:
                continue
            support_result = self._choose_support_node(record, room, graph)
            support_nodes = [
                node
                for node in [support_result.get("node"), *support_result.get("candidates", [])]
                if node is not None
                and (
                    self._category_prefers_floor(category)
                    or not self._tokens(node.get("category", "")) & {"floor", "floors"}
                )
            ]
            support = support_nodes[0] if support_nodes else None
            if support:
                approach_quality = 1
                for support_node in support_nodes:
                    bbox = support_node.get("bbox") or {}
                    position = (support_node.get("pose") or {}).get("position")
                    if not position:
                        continue
                    support_aabb_min = bbox.get("min")
                    support_aabb_max = bbox.get("max")
                    support_aabb_xy = (
                        (support_aabb_min[:2], support_aabb_max[:2])
                        if support_aabb_min and support_aabb_max
                        else None
                    )
                    proposed = [
                        float(position[0]),
                        float(position[1]),
                        float((bbox.get("max") or position)[2]) + 0.15,
                    ]
                    if self._validate_task_approach_position(
                        proposed,
                        {room},
                        target_aabb_xy=support_aabb_xy,
                        # The robot must stand outside the support footprint.
                        target_object_id=None,
                    ).get("ok"):
                        approach_quality = 0
                        break
                if approach_quality:
                    continue
                quality = 1 if self._tokens(support.get("category", "")) & {"floor", "floors"} else 0
            elif self._category_prefers_floor(category) and self._floor_fallback_allowed(record, category):
                approach_quality = 0
                quality = 2
            else:
                continue
            camera_priority = self._room_camera_priority(room, graph)
            preferred_penalty = 0 if room == preferred_room else 1
            support_capacity = -min(len(support_nodes), 6)
            candidates.append((
                approach_quality,
                support_capacity,
                camera_priority,
                quality,
                room_counts.get(room, 0),
                preferred_penalty,
                room,
            ))
        if not candidates:
            return None
        best_approach = min(item[0] for item in candidates)
        approach_safe = [item for item in candidates if item[0] == best_approach]
        best_capacity = min(item[1] for item in approach_safe)
        capacity_safe = [item for item in approach_safe if item[1] == best_capacity]
        best_camera = min(item[2] for item in capacity_safe)
        camera_safe = [item for item in capacity_safe if item[2] == best_camera]
        best_quality = min(item[3] for item in camera_safe)
        quality_safe = [item for item in camera_safe if item[3] == best_quality]
        best_usage = min(item[4] for item in quality_safe)
        best = [item for item in quality_safe if item[4] == best_usage]
        return self.rng.choice(sorted(best))[6]

    def _choose_support_bootstrap_room(
        self, graph: dict, excluded_rooms: set[str] | None = None,
    ) -> str | None:
        """Choose an open reachable room for a generated support fixture."""
        excluded_rooms = set(excluded_rooms or [])
        room_pixels = self._robot_reachable_room_pixels()
        rooms = {
            node["name"]
            for node in graph.get("nodes", [])
            if node.get("type") == "room"
            and node.get("name") in room_pixels
            and node.get("name") not in excluded_rooms
        }
        candidates = []
        for room in rooms:
            pixel_count = len(room_pixels.get(room) or [])
            # Sparse corridors can still fit a compact generated support. A
            # large arbitrary pixel-count gate previously removed them before
            # the actual fixture footprint and operation-pose checks ran,
            # forcing two delivery supports into the same small room.
            if pixel_count < 8:
                continue
            room_lower = room.lower()
            if "empty" in room_lower:
                semantic_penalty = 0
            elif any(token in room_lower for token in ("living", "bedroom", "kitchen", "dining")):
                semantic_penalty = 1
            elif "entryway" in room_lower:
                semantic_penalty = 2
            elif any(token in room_lower for token in ("bathroom", "corridor", "closet")):
                semantic_penalty = 4
            else:
                semantic_penalty = 3
            candidates.append((
                semantic_penalty,
                self._room_camera_priority(room, graph),
                -pixel_count,
                room,
            ))
        return min(candidates)[3] if candidates else None

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

        def normalize(value):
            return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

        scene_cats = {normalize(category) for category in self._get_scene_categories()}
        selected = []
        scene_native_count = 0  # track objects skipped because they're already in scene
        for obj in required_objs:
            hint = normalize(obj.get("category_hint", ""))
            name = normalize(obj.get("name", ""))
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
                        name_words = set(name.split("_"))
                        sc_words = set(sc.split("_"))
                        exact_word_match = bool(name_words) and name_words <= sc_words
                        if exact_word_match or (
                            len(name) >= 4 and (name in sc or sc in name)
                        ):
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

            # Search asset DB for matching categories. Keep all reasonable
            # matches so repeated batch runs can prefer less-used categories.
            scored_records = []
            for task_name, records in self.asset_db.tasks.items():
                for r in self._usable_records(records):
                    score = 0
                    r_cats = {normalize(c) for c in r.get("direct_categories", [])}
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

                    if score >= 5:
                        scored_records.append((score, r))

            if scored_records:
                best_score = max(score for score, _ in scored_records)
                close_matches = [
                    r for score, r in scored_records
                    if score >= max(5, best_score - 2)
                ]
                best_record = self._choose_diverse_required_record(close_matches)
                selected.append(best_record)
                used = self._used_target_categories.get(self._choose_category(best_record), 0)
                print(
                    f"[match] {obj['name']} → {best_record['synset']} "
                    f"(score={best_score}, prior_category_uses={used})"
                )
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
        candidates = self._rank_context_candidates(candidates)

        # LLM selection if available
        if self._llm_client:
            result = self._llm_select_context(
                "custom_task", selected_records,
                candidates[:50], graph
            )
            if result:
                return result[:self.config.context_objects]

        return self._weighted_sample_records(candidates, self.config.context_objects)

    def _record_min_mass(self, record: dict) -> float | None:
        masses = record.get("mass_estimates") or {}
        values = []
        for value in masses.values():
            try:
                if value is not None:
                    values.append(float(value))
            except Exception:
                pass
        return min(values) if values else None

    def _rank_context_candidates(self, candidates: list[dict]) -> list[dict]:
        def score(record):
            category = self._choose_category(record)
            tokens = self._tokens(category)
            mass = self._record_min_mass(record)
            interaction = (record.get("edit_metadata", {}).get("interaction") or {}).get("kind")
            receptacle = record.get("edit_metadata", {}).get("receptacle") or {}
            value = 0.0
            if interaction == "manipulable":
                value += 3.0
            if mass is not None:
                if mass <= 0.5:
                    value += 2.0
                elif mass <= 1.0:
                    value += 0.5
                else:
                    value -= 2.0
            if receptacle.get("supports_inside"):
                value += 0.3
            if receptacle.get("supports_on_top") and mass is not None and mass > 0.5:
                value -= 0.8
            if tokens & {"tray", "pan", "pot", "basket", "bucket", "box"}:
                value -= 0.6
            return (-value, record.get("synset", ""))

        return sorted(candidates, key=score)

    def _select_task_records(self, task, skip_tasks=None):
        skip = set(skip_tasks) if skip_tasks else set()
        # Also merge engine-level rejected task cache
        if not self.config.allow_repeat_tasks:
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

    def _choose_live_placement(self, record, target_room, graph=None, preferred_position=None):
        graph = graph or self.snapshot()
        support_result = self._choose_support_node(
            record, target_room, graph, preferred_position=preferred_position,
        )
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
        # Bathroom medicine retrieval can legitimately use a sink basin in
        # sparse scenes that have no counter or cabinet.
        "bottle_of_medicine": {"countertop", "counter", "table", "shelf", "cabinet", "sink", "wall_mounted_sink"},
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
        "key_chain": {"table", "desk", "nightstand", "countertop", "counter", "shelf", "dresser", "cabinet"},
        "keys": {"table", "desk", "nightstand", "countertop", "counter", "shelf", "dresser", "cabinet"},
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

    def _choose_support_node(self, record, target_room, graph, top_n=8, preferred_position=None):
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
            if support_tokens & {"carpet", "rug", "mat", "doormat"}:
                continue
            # Scene categories are commonly named "floors" (plural). Treat
            # both spellings as controlled low-priority floor supports.
            if support_tokens & {"floor", "floors"} and not self._floor_fallback_allowed(record, obj_cat):
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

            # Fixed room cameras see elevated, open horizontal surfaces much
            # more reliably than floors or deep storage furniture.
            if support_tokens & {"coffee", "breakfast", "dining", "table", "desk", "counter", "countertop", "island"}:
                score += 8
            elif support_tokens & {"bookcase", "shelf", "cabinet", "dresser", "nightstand"}:
                score += 2
            if is_floor:
                score -= 12
            if receptacle.get("supports_inside") and not receptacle.get("supports_on_top"):
                score -= 5

            # Area-based crowding penalty
            score -= fill_ratio * 15

            if preferred_position is not None:
                center = self._center(node)
                if center is not None:
                    distance = float(np.linalg.norm(
                        np.asarray(center[:2], dtype=float)
                        - np.asarray(preferred_position[:2], dtype=float)
                    ))
                    score += max(0.0, 18.0 - distance * 4.0)

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

    def _validate_task_approach_position(
        self,
        target_position,
        target_rooms,
        target_aabb_xy=None,
        max_horizontal_distance=None,
        target_object_id=None,
        preferred_center_distance=None,
    ):
        """Estimate a manipulation approach from an object or proposed pose."""
        if not self.config.require_task_object_reachability:
            return {"ok": True, "reason": "disabled"}
        if not getattr(self.env, "robots", None):
            return {"ok": False, "reason": "missing_robot"}
        try:
            target_position = th.as_tensor(target_position, dtype=th.float32)
            trav_map = self.env.scene.trav_map
            floor = min(
                range(trav_map.n_floors),
                key=lambda index: abs(
                    float(trav_map.floor_heights[index]) - float(target_position[2])
                ),
            )
            cached = self._robot_component_pixels(floor)
            if cached is None:
                return {"ok": False, "reason": "no_traversable_pixels", "floor": floor}

            pixels = cached["pixels"]
            if pixels.numel() == 0:
                return {
                    "ok": False,
                    "reason": "empty_robot_component",
                    "floor": floor,
                    "source_label": cached["source_label"],
                }
            target_rooms = (
                {target_rooms} if isinstance(target_rooms, str) else set(target_rooms or [])
            )
            room_pixels = self._robot_reachable_room_pixels(floor)
            if target_rooms:
                allowed_pixels = [
                    pixel
                    for room in sorted(target_rooms)
                    for pixel in room_pixels.get(room, [])
                ]
                candidate_pixels = (
                    th.stack(allowed_pixels).to(pixels.device)
                    if allowed_pixels
                    else None
                )
            else:
                candidate_pixels = pixels
            if candidate_pixels is None or candidate_pixels.numel() == 0:
                return {
                    "ok": False,
                    "reason": "no_same_room_approach",
                    "floor": floor,
                    "source_label": cached["source_label"],
                    "target_rooms": sorted(target_rooms),
                    "component_rooms": sorted(room_pixels),
                }
            threshold = float(
                self.config.max_task_object_approach_distance
                if max_horizontal_distance is None
                else max_horizontal_distance
            )
            target_pixel = trav_map.world_to_map(target_position[:2]).to(candidate_pixels.device)
            distances = th.linalg.norm(candidate_pixels.float() - target_pixel.float(), dim=1)
            if target_aabb_xy is not None:
                candidate_world_xy = trav_map.map_to_world(candidate_pixels)
                aabb_min_device = th.as_tensor(
                    target_aabb_xy[0], dtype=th.float32, device=candidate_world_xy.device
                )
                aabb_max_device = th.as_tensor(
                    target_aabb_xy[1], dtype=th.float32, device=candidate_world_xy.device
                )
                nearest_world_xy = th.minimum(
                    th.maximum(candidate_world_xy, aabb_min_device), aabb_max_device
                )
                edge_distances = th.linalg.norm(
                    candidate_world_xy - nearest_world_xy, dim=1
                )
                eligible_distance_mask = edge_distances <= threshold
                if not bool(th.any(eligible_distance_mask)):
                    nearest_index = int(th.argmin(edge_distances))
                    nearest_candidate = candidate_world_xy[nearest_index]
                    return {
                        "ok": False,
                        "reason": "approach_too_far",
                        "floor": floor,
                        "source_label": cached["source_label"],
                        "horizontal_distance": float(edge_distances[nearest_index]),
                        "distance_reference": "aabb_edge",
                        "max_horizontal_distance": threshold,
                        "candidate_position_xy": self._to_list(nearest_candidate),
                        "candidate_room": self.env.scene.seg_map.get_room_instance_by_point(
                            nearest_candidate
                        ),
                        "target_rooms": sorted(target_rooms),
                        "native_occupancy_rejections": 0,
                    }
                candidate_pixels = candidate_pixels[eligible_distance_mask]
                distances = distances[eligible_distance_mask]
            if preferred_center_distance is not None:
                preferred_pixels = float(preferred_center_distance) / float(
                    trav_map.map_resolution
                )
                candidate_order = th.abs(distances - preferred_pixels)
            else:
                candidate_order = distances
            candidate_xy, occupancy_rejections = self._collision_free_approach_candidate(
                candidate_pixels,
                candidate_order,
                target_position,
                target_object_id,
            )
            if candidate_xy is None:
                return {
                    "ok": False,
                    "reason": "no_collision_free_approach",
                    "floor": floor,
                    "source_label": cached["source_label"],
                    "target_rooms": sorted(target_rooms),
                    "native_occupancy_rejections": occupancy_rejections,
                }
            candidate_room = self.env.scene.seg_map.get_room_instance_by_point(candidate_xy[:2])
            if target_aabb_xy is None:
                horizontal_distance = float(
                    th.linalg.norm(target_position[:2].cpu() - candidate_xy[:2].cpu())
                )
                distance_reference = "object_center"
            else:
                aabb_min_xy = th.as_tensor(target_aabb_xy[0], dtype=th.float32)
                aabb_max_xy = th.as_tensor(target_aabb_xy[1], dtype=th.float32)
                nearest_xy = th.minimum(
                    th.maximum(candidate_xy[:2].cpu(), aabb_min_xy), aabb_max_xy
                )
                horizontal_distance = float(
                    th.linalg.norm(candidate_xy[:2].cpu() - nearest_xy)
                )
                distance_reference = "aabb_edge"
            return {
                "ok": horizontal_distance <= threshold,
                "reason": "reachable" if horizontal_distance <= threshold else "approach_too_far",
                "floor": floor,
                "source_label": cached["source_label"],
                "horizontal_distance": horizontal_distance,
                "distance_reference": distance_reference,
                "max_horizontal_distance": threshold,
                "candidate_position_xy": self._to_list(candidate_xy[:2]),
                "candidate_room": candidate_room,
                "target_rooms": sorted(target_rooms),
                "native_occupancy_rejections": occupancy_rejections,
            }
        except Exception as exc:
            return {
                "ok": False,
                "reason": "reachability_check_error",
                "error": repr(exc),
            }

    def _collision_free_approach_candidate(
        self, candidate_pixels, distances, target_position, target_object_id,
    ):
        """Choose the nearest candidate whose real robot boundary clears scene objects."""
        robot = self.env.robots[0]
        robot_position, robot_orientation = robot.get_position_orientation()
        robot_position = np.asarray(robot_position.cpu(), dtype=float)
        live_points = np.asarray(robot.collision_points_world.cpu(), dtype=float)
        live_rotation = np.asarray(T.quat2mat(robot_orientation).cpu(), dtype=float)
        local_points = (live_points - robot_position) @ live_rotation
        target = self.env.scene.object_registry("name", target_object_id, None)
        blockers = []
        for scene_object in self._scene_objects():
            if scene_object is robot or scene_object is target:
                continue
            category = str(getattr(scene_object, "category", "") or "").lower()
            if category in NON_BLOCKING_NAVIGATION_CATEGORIES:
                continue
            try:
                lower, upper = scene_object.aabb
                lower = np.asarray(lower.cpu(), dtype=float)
                upper = np.asarray(upper.cpu(), dtype=float)
            except Exception:
                continue
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                blockers.append((lower, upper))

        trav_map = self.env.scene.trav_map
        occupancy_rejections = 0
        for index in th.argsort(distances).cpu().tolist():
            candidate_xy = trav_map.map_to_world(candidate_pixels[index])
            yaw = math.atan2(
                float(target_position[1] - candidate_xy[1]),
                float(target_position[0] - candidate_xy[0]),
            )
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = np.asarray(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                dtype=float,
            )
            candidate_position = np.asarray(
                [float(candidate_xy[0]), float(candidate_xy[1]), robot_position[2]],
                dtype=float,
            )
            candidate_points = local_points @ rotation.T + candidate_position
            occupied = any(
                np.any(
                    np.all(candidate_points >= lower - 0.10, axis=1)
                    & np.all(candidate_points <= upper + 0.10, axis=1)
                )
                for lower, upper in blockers
            )
            if not occupied:
                return candidate_xy, occupancy_rejections
            occupancy_rejections += 1
        return None, occupancy_rejections

    def _validate_task_object_approach(self, obj, target_room=None):
        """Validate a manipulation approach in the initial robot component."""
        try:
            target_position, _ = obj.get_position_orientation()
            target_rooms = {target_room} if target_room else set(
                getattr(obj, "in_rooms", None) or self._rooms_for_obj(obj)
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "reachability_check_error",
                "error": repr(exc),
            }
        aabb_min, aabb_max, _ = self._safe_aabb(obj)
        target_aabb_xy = (
            (aabb_min[:2], aabb_max[:2]) if aabb_min and aabb_max else None
        )
        return self._validate_task_approach_position(
            target_position,
            target_rooms,
            target_aabb_xy=target_aabb_xy,
            target_object_id=obj.name,
        )

    def _floor_height_for_position(self, position):
        """Return the closest official traversability floor height."""
        heights = [float(value) for value in self.env.scene.trav_map.floor_heights]
        if not heights:
            return 0.0
        z = float(position[2])
        below = [height for height in heights if height <= z + 0.10]
        return max(below) if below else min(heights, key=lambda height: abs(height - z))

    def _ground_object_on_floor(self, obj, placement):
        position = list(placement["pose"]["position"])
        orientation = placement["pose"]["orientation_xyzw"]
        floor_height = self._floor_height_for_position(position)
        try:
            live_lo, live_hi = obj.aabb
            lo = self._to_list(live_lo)
            hi = self._to_list(live_hi)
        except Exception:
            lo, hi, _ = self._safe_aabb(obj)
        extent_z = float(hi[2] - lo[2]) if lo and hi else 0.10
        current_position, _ = obj.get_position_orientation()
        if lo and hi:
            aabb_center = th.tensor(
                [(lo[index] + hi[index]) / 2.0 for index in range(3)],
                dtype=th.float32,
            )
            center_offset = aabb_center - current_position.to(dtype=th.float32)
        else:
            center_offset = th.zeros(3, dtype=th.float32)
        target_center = th.tensor(
            [
                float(position[0]),
                float(position[1]),
                floor_height + max(extent_z, 0.01) / 2.0 + 0.005,
            ],
            dtype=th.float32,
        )
        position = self._to_list(target_center - center_offset)
        obj.set_position_orientation(
            position=th.tensor(position, dtype=th.float32),
            orientation=th.tensor(orientation, dtype=th.float32),
        )
        placement["pose"]["position"] = position
        return floor_height

    def _validate_floor_manipulation_height(self, obj, floor_height):
        lo, hi, _ = self._safe_aabb(obj)
        if not lo or not hi:
            return {
                "required": True,
                "eligible": False,
                "primitive": "GRASP",
                "reason": "object AABB unavailable after floor placement",
                "min_height": float(self.config.min_manipulation_height),
                "max_height": float(self.config.max_manipulation_height),
            }
        return evaluate_manipulation_height(
            "GRASP",
            lo[2],
            hi[2],
            floor_height,
            self.config.min_manipulation_height,
            self.config.max_manipulation_height,
        )

    def _build_floor_placement(
        self,
        record,
        target_room,
        graph=None,
        preferred_position=None,
        avoid_position=None,
        spread_across_room=False,
        placement_obj=None,
        require_footprint_clear=False,
    ):
        """Last-resort placement: on the floor in the target room."""
        try:
            graph = graph or self.snapshot()
            room_center = (graph.get("navigation") or {}).get("room_centers", {}).get(target_room)
            room_pixels = self._robot_reachable_room_pixels()
            pixels = room_pixels.get(target_room) or []
            if pixels and placement_obj is not None:
                try:
                    operation_pixels = list(pixels)
                    trav_map = self.env.scene.trav_map
                    reference = room_center or preferred_position
                    ranked_pixels = list(pixels)
                    if reference is not None:
                        reference_pixel = trav_map.world_to_map(
                            th.as_tensor(reference[:2], dtype=th.float32)
                        )
                        ranked_pixels.sort(
                            key=lambda pixel: float(
                                th.linalg.norm(
                                    pixel.float().cpu() - reference_pixel.float().cpu()
                                )
                            )
                        )
                    if avoid_position is not None:
                        avoid_pixel = trav_map.world_to_map(
                            th.as_tensor(avoid_position[:2], dtype=th.float32)
                        )
                        min_separation_pixels = 1.0 / float(trav_map.map_resolution)
                        separated = [
                            pixel
                            for pixel in ranked_pixels
                            if float(
                                th.linalg.norm(
                                    pixel.float().cpu() - avoid_pixel.float().cpu()
                                )
                            )
                            >= min_separation_pixels
                        ]
                        if separated:
                            ranked_pixels = separated
                    # Full-room footprint x operation-pose scans become
                    # quadratic. The best central, source-separated poses are
                    # sufficient; live post-placement validation remains final.
                    pixels = ranked_pixels[:96]
                    live_lo, live_hi = placement_obj.aabb
                    live_lo = np.asarray(live_lo.cpu(), dtype=float)
                    live_hi = np.asarray(live_hi.cpu(), dtype=float)
                    live_position = np.asarray(
                        placement_obj.get_position_orientation()[0].cpu(), dtype=float
                    )
                    center_offset = (live_lo + live_hi) * 0.5 - live_position
                    half_extent = (live_hi - live_lo)[:2] * 0.5 + 0.02
                    blockers = []
                    seen = set()
                    robots = list(getattr(self.env, "robots", None) or [])
                    for other in [
                        *self._scene_objects(),
                        *robots,
                    ]:
                        if other is placement_obj or id(other) in seen:
                            continue
                        seen.add(id(other))
                        is_robot = other in robots
                        tokens = self._tokens(getattr(other, "category", "") or "")
                        if not is_robot and tokens & NON_BLOCKING_NAVIGATION_CATEGORIES:
                            continue
                        if not is_robot:
                            rooms = self._rooms_for_obj(other)
                            if rooms and target_room not in rooms:
                                continue
                        other_lo, other_hi, _ = self._safe_aabb(other)
                        if other_lo is None or other_hi is None:
                            continue
                        blockers.append((
                            np.asarray(other_lo[:2], dtype=float),
                            np.asarray(other_hi[:2], dtype=float),
                        ))

                    footprint_clear_pixels = []
                    for pixel in pixels:
                        xy = np.asarray(trav_map.map_to_world(pixel).cpu(), dtype=float)
                        center = xy + center_offset[:2]
                        proposed_lo = center - half_extent
                        proposed_hi = center + half_extent
                        if not any(
                            np.all(proposed_lo < other_hi)
                            and np.all(proposed_hi > other_lo)
                            for other_lo, other_hi in blockers
                        ) and self._hypothetical_support_has_operation_approach(
                            operation_pixels,
                            center,
                            proposed_lo,
                            proposed_hi,
                        ):
                            footprint_clear_pixels.append(pixel)
                    if footprint_clear_pixels:
                        pixels = footprint_clear_pixels
                    elif require_footprint_clear:
                        print(
                            f"[floor-placement] no footprint-clear pose for "
                            f"{getattr(placement_obj, 'name', '?')}",
                            flush=True,
                        )
                        return None
                    else:
                        # The conservative 2-D footprint / operation preflight
                        # can reject every pixel in a small room even when the
                        # live 3-D AABB and robot-approach checks accept a pose.
                        # Stay on official reachable pixels and let those
                        # existing final gates decide; do not fall back to an
                        # unvetted Gaussian point near the room center.
                        pixels = ranked_pixels
                        print(
                            f"[floor-placement] conservative footprint preflight "
                            f"found no pose for {getattr(placement_obj, 'name', '?')}; "
                            f"using {len(pixels)} reachable candidates",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[floor-placement] footprint preflight failed for "
                        f"{getattr(placement_obj, 'name', '?')}: {exc!r}",
                        flush=True,
                    )
            if pixels:
                trav_map = self.env.scene.trav_map
                reference = preferred_position if preferred_position is not None else room_center
                if reference is not None:
                    reference_pixel = trav_map.world_to_map(
                        th.as_tensor(reference[:2], dtype=th.float32)
                    )
                    ordered = sorted(
                        pixels,
                        key=lambda pixel: float(
                            th.linalg.norm(pixel.float().cpu() - reference_pixel.float().cpu())
                        ),
                    )
                    if avoid_position is not None:
                        avoid_pixel = trav_map.world_to_map(
                            th.as_tensor(avoid_position[:2], dtype=th.float32)
                        )
                        min_separation_pixels = 1.0 / float(trav_map.map_resolution)
                        separated = [
                            pixel
                            for pixel in ordered
                            if float(
                                th.linalg.norm(
                                    pixel.float().cpu() - avoid_pixel.float().cpu()
                                )
                            )
                            >= min_separation_pixels
                        ]
                        if separated:
                            ordered = separated
                        pool = ordered[: min(25, len(ordered))]
                    else:
                        pool = ordered if spread_across_room else ordered[: min(25, len(ordered))]
                    pixel = self.rng.choice(pool)
                else:
                    pixel = self.rng.choice(pixels)
                xy = trav_map.map_to_world(pixel)
                pos = [float(xy[0]), float(xy[1]), 0.5]
                pose_source = "robot_reachable_floor"
            elif preferred_position is not None:
                angle = self.rng.uniform(-math.pi, math.pi)
                radius = self.rng.uniform(0.8, 1.4)
                pos = [
                    float(preferred_position[0]) + math.cos(angle) * radius,
                    float(preferred_position[1]) + math.sin(angle) * radius,
                    0.5,
                ]
                pose_source = "floor_fallback"
            elif room_center:
                pos = [
                    float(room_center[0]) + self.rng.gauss(0, 0.3),
                    float(room_center[1]) + self.rng.gauss(0, 0.3),
                    0.5,
                ]
                pose_source = "floor_fallback"
            else:
                pos = [self.rng.gauss(0, 0.3), self.rng.gauss(0, 0.3), 0.5]
                pose_source = "floor_fallback"
            return {
                "room_id": target_room,
                "mode": "floor",
                "support_object_id": None,
                "support_category": "floor",
                "support_candidates": [],
                "pose": {
                    "position": pos,
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "pose_source": pose_source,
            }
        except Exception:
            return None

    def _hypothetical_support_has_operation_approach(
        self, room_pixels, target_center, proposed_lo, proposed_hi
    ):
        """Reserve a Tiago operation pose before a generated support is spawned."""
        if not room_pixels or not getattr(self.env, "robots", None):
            return False
        try:
            robot = self.env.robots[0]
            robot_position, robot_orientation = robot.get_position_orientation()
            robot_position = np.asarray(robot_position.cpu(), dtype=float)
            live_points = np.asarray(robot.collision_points_world.cpu(), dtype=float)
            live_rotation = np.asarray(T.quat2mat(robot_orientation).cpu(), dtype=float)
            local_points = (live_points - robot_position) @ live_rotation
            target_center = np.asarray(target_center[:2], dtype=float)
            proposed_lo = np.asarray(proposed_lo[:2], dtype=float)
            proposed_hi = np.asarray(proposed_hi[:2], dtype=float)
            trav_map = self.env.scene.trav_map
            candidates = []
            for pixel in room_pixels:
                xy = np.asarray(trav_map.map_to_world(pixel).cpu(), dtype=float)[:2]
                nearest = np.minimum(np.maximum(xy, proposed_lo), proposed_hi)
                distance = float(np.linalg.norm(xy - nearest))
                if distance <= self.config.max_task_object_approach_distance:
                    candidates.append((distance, xy))
            for _, xy in sorted(candidates, key=lambda item: item[0])[:64]:
                yaw = math.atan2(target_center[1] - xy[1], target_center[0] - xy[0])
                cosine, sine = math.cos(yaw), math.sin(yaw)
                rotation = np.asarray(
                    [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                )
                candidate_position = np.asarray(
                    [xy[0], xy[1], robot_position[2]], dtype=float
                )
                candidate_points = local_points @ rotation.T + candidate_position
                occupied = np.any(
                    np.all(candidate_points[:, :2] >= proposed_lo - 0.10, axis=1)
                    & np.all(candidate_points[:, :2] <= proposed_hi + 0.10, axis=1)
                )
                if not occupied:
                    return True
            return False
        except Exception as exc:
            print(
                f"[floor-placement] operation preflight failed: {exc!r}",
                flush=True,
            )
            return False

    def _apply_relation(self, obj, placement, official_fallback=None):
        support_id = placement.get("support_object_id")
        if not support_id:
            return {"ok": False, "mode": placement.get("mode"), "reason": "no_support_object"}
        support_obj = self.env.scene.object_registry("name", support_id, None)
        if support_obj is None:
            return {"ok": False, "mode": placement.get("mode"), "support": support_id, "reason": "support_missing"}
        support_tokens = self._tokens(getattr(support_obj, "category", "") or "")
        if placement.get("mode") == "on_top" and support_tokens & {
            "floor", "floors", "carpet", "rug", "mat", "doormat",
        }:
            return {
                "ok": False,
                "mode": placement.get("mode"),
                "support": support_id,
                "reason": "structural_surface_requires_floor_mode",
            }
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
            official_sampler = None
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
                obj_h = max(float(obj_aabb[1][2] - obj_aabb[0][2]), 0.02)
                margin = max(0.03, min(0.08, 0.5 * min(obj_w, obj_d)))

                # Collect obstacles: objects ON the support + objects NEAR the support
                obstacles = []
                for other in self._scene_objects():
                    other_name = getattr(other, "name", "")
                    if (
                        other is obj
                        or other_name == support_id
                        or other in (getattr(self.env, "robots", None) or [])
                    ):
                        continue
                    other_tokens = self._tokens(getattr(other, "category", "") or "")
                    if other_tokens & {
                        "agent",
                        "floor", "floors", "carpet", "rug",
                        "wall", "walls", "ceiling", "ceilings",
                    }:
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
                            "name": other_name,
                            "category": getattr(other, "category", None),
                            "x_min": ox_min, "x_max": ox_max,
                            "y_min": oy_min, "y_max": oy_max,
                            "z_min": oz_min, "z_max": oz_max,
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
                grid_indices = sorted(
                    range(n_steps), key=lambda index: abs(index + 0.5 - n_steps / 2.0)
                )
                access_xy = None
                if placement.get("prefer_robot_access"):
                    support_position, _ = support_obj.get_position_orientation()
                    support_rooms = set(getattr(support_obj, "in_rooms", None) or [])
                    access = self._validate_task_approach_position(
                        support_position,
                        support_rooms,
                        target_aabb_xy=((sup_x_min, sup_y_min), (sup_x_max, sup_y_max)),
                        target_object_id=None,
                    )
                    if access.get("ok"):
                        access_xy = access.get("candidate_position_xy")
                best_cx, best_cy = None, None
                grid_points = [
                    (
                        sw + w_range * (ix + 0.5) / n_steps,
                        sd + d_range * (iy + 0.5) / n_steps,
                    )
                    for ix in grid_indices
                    for iy in grid_indices
                ]
                if access_xy is not None:
                    grid_points.sort(
                        key=lambda point: math.hypot(
                            point[0] - float(access_xy[0]),
                            point[1] - float(access_xy[1]),
                        )
                    )
                for cx, cy in grid_points:
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

                if best_cx is None:
                    return {
                        "ok": False,
                        "mode": placement.get("mode"),
                        "support": support_id,
                        "state": state_cls.__name__,
                        "reason": "support_surface_occupied",
                        "num_obstacles": len(obstacles),
                        "obstacles": obstacles,
                        "support_bounds": {
                            "x_min": sup_x_min, "x_max": sup_x_max,
                            "y_min": sup_y_min, "y_max": sup_y_max,
                            "z_top": sup_z_top,
                        },
                    }

                # Asset origins need not coincide with their AABB centers.
                # Preserve the live offset so the requested bottom height and
                # XY footprint are correct for every USD model.
                obj_position, obj_orientation = obj.get_position_orientation()
                obj_center = (obj_aabb[0] + obj_aabb[1]) / 2.0
                center_offset = obj_center - obj_position
                target_center = th.tensor(
                    [best_cx, best_cy, sup_z_top + obj_h / 2.0 + 0.01],
                    dtype=th.float32,
                )
                target_position = target_center - center_offset.to(dtype=th.float32)
                obj.set_position_orientation(
                    position=target_position,
                    orientation=obj_orientation,
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
                try:
                    _diag_pos = obj.get_position_orientation()[0]
                    _diag_pose = self._validate_on_top_pose(obj, support_obj)
                    _diag_sup_pos = support_obj.get_position_orientation()[0]
                    print(
                        f"[ontop-diag] support={support_id} ontop_state={ok} "
                        f"sup_aabb_top_z={sup_z_top:.3f} "
                        f"sup_xy=({sup_x_min:.2f},{sup_y_min:.2f})-({sup_x_max:.2f},{sup_y_max:.2f}) "
                        f"sup_pos={[round(float(v), 3) for v in _diag_sup_pos]} "
                        f"drop_xy=({best_cx:.3f},{best_cy:.3f}) obj_h={obj_h:.3f} "
                        f"obj_pos={[round(float(v), 3) for v in _diag_pos]} "
                        f"pose_check={_diag_pose.get('ok')}({_diag_pose.get('reason', '')})",
                        flush=True,
                    )
                except Exception as _diag_exc:
                    print(f"[ontop-diag] diagnostic error: {_diag_exc!r}", flush=True)
                used_official_resample = False
                if not ok:
                    # The direct geometric drop is the fast path. When an
                    # articulated or offset-origin support rejects it,
                    # OmniGibson's official relation sampler may resample at
                    # most once per object: the blocking call is
                    # uninterruptible, so it is budgeted and measured here
                    # instead of being retried behind a timeout that only logs
                    # after the call returns.
                    #
                    # If the object clearly missed (it settled well below the
                    # intended support-top drop height), the support's AABB top
                    # was not a usable surface. The blocking official sampler
                    # cannot correct a wrong surface and would only burn the
                    # whole placement budget (~100s), so fail fast and let the
                    # outer loop try another candidate/support/room.
                    clear_miss = False
                    settled_bottom_z = None
                    try:
                        settled_bottom_z = float(obj.aabb[0][2])
                        if settled_bottom_z < (sup_z_top + 0.01) - 0.06:
                            clear_miss = True
                    except Exception:
                        clear_miss = False
                    if clear_miss:
                        official_sampler = {
                            "attempted": False,
                            "reason": "geometric_drop_clear_miss",
                            "settled_bottom_z": settled_bottom_z,
                            "intended_bottom_z": sup_z_top + 0.01,
                        }
                    elif official_fallback is not None and official_fallback["calls"] >= 1:
                        official_sampler = {
                            "attempted": False,
                            "reason": "official_sampler_budget_exhausted",
                            "calls": official_fallback["calls"],
                            "total_seconds": official_fallback["seconds"],
                        }
                    else:
                        sampler_start = time.time()
                        sampler_error = None
                        try:
                            ok = bool(
                                obj.states[state_cls].set_value(
                                    support_obj, True, reset_before_sampling=True,
                                )
                            )
                        except Exception as exc:
                            sampler_error = repr(exc)
                            ok = False
                        sampler_seconds = time.time() - sampler_start
                        if official_fallback is not None:
                            official_fallback["calls"] += 1
                            official_fallback["seconds"] += sampler_seconds
                        official_sampler = {
                            "attempted": True,
                            "calls": (official_fallback or {}).get("calls", 1),
                            "seconds": sampler_seconds,
                            "total_seconds": (official_fallback or {}).get("seconds", sampler_seconds),
                            "exceeded_relation_budget": (
                                sampler_seconds > self.config.per_relation_attempt_timeout_sec
                            ),
                            "error": sampler_error,
                        }
                    used_official_resample = ok
                    if official_sampler.get("error") is not None:
                        # Illegal BroadPhaseUpdateData / invalid physics views
                        # surface here; retrying inside this process poisons it.
                        official_sampler["fatal"] = True
            else:
                # Inside: use set_value with sampling (only option for inside placement)
                ok = bool(obj.states[state_cls].set_value(support_obj, True, reset_before_sampling=True))
                used_official_resample = False
            result = {
                "ok": ok,
                "mode": placement.get("mode"),
                "support": support_id,
                "state": state_cls.__name__,
            }
            if used_official_resample:
                result["reason"] = "official_on_top_resample"
            if official_sampler is not None:
                result["official_sampler"] = official_sampler
                if official_sampler.get("fatal"):
                    result["fatal_physics_error"] = True
            if not ok and state_cls is object_states.OnTop:
                pose_check = self._validate_on_top_pose(obj, support_obj)
                result["pose_check"] = pose_check
                result["object_aabb"] = self._to_list(obj.aabb)
                result["support_aabb"] = self._to_list(support_obj.aabb)
                if pose_check.get("ok"):
                    result["ok"] = True
                    result["reason"] = "geometric_on_top_fallback"
                    result["state_reported"] = False
            return result
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
            clearance = max(0.03, min(0.08, 0.5 * min(obj_w, obj_d)))

            if obj_w + 2 * clearance > sup_w or obj_d + 2 * clearance > sup_d:
                return {
                    "ok": False,
                    "reason": "object_larger_than_support_surface",
                    "object_footprint": [obj_w, obj_d],
                    "support_bounds": [sup_w, sup_d],
                }
            if (
                ox_min < sx_min + clearance
                or ox_max > sx_max - clearance
                or oy_min < sy_min + clearance
                or oy_max > sy_max - clearance
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
                # A validated floor placement is expected to contact the
                # floor. Only contacts with another generated object remain a
                # failure in that case.
                floor_placement = name in self._floor_placed_objects
                contacting = []
                if floor_placement:
                    expected_children = {
                        child_name
                        for child_name, support_name in self._placement_support_map.items()
                        if support_name == name
                    }
                    for other_name, other_obj in scene_objs_by_name.items():
                        if other_name == name or other_name in expected_children:
                            continue
                        try:
                            if RigidContactAPI.is_in_contact(
                                scene_idx=self.env.scene.idx,
                                query_set=[obj], with_set=[other_obj],
                                ignore_set=None, current_only=True,
                            ):
                                contacting.append(other_name)
                        except Exception:
                            pass
                if (has_contact and not floor_placement) or contacting:
                    entry["unexpected_contact"] = True
                    all_within = False
                    contact_issues.append(name)
                    if not contacting:
                        for other_name, other_obj in scene_objs_by_name.items():
                            if other_name == name:
                                continue
                            try:
                                pair = RigidContactAPI.is_in_contact(
                                    scene_idx=self.env.scene.idx,
                                    query_set=[obj], with_set=[other_obj],
                                    ignore_set=None, current_only=True,
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

    @staticmethod
    def _fallback_enva_solution_plan(
        primary_task,
        target_room,
        task_objects,
        native_target=None,
        delivery_destination=None,
    ):
        """Return the minimal exact-ID plan for a validated Env-A contract."""
        if native_target:
            object_id = native_target["object_id"]
            room = native_target["room_id"]
            return [
                {
                    "step_id": 1,
                    "primitive": "MOVE",
                    "nl": f"Move to {object_id}",
                    "target_object": object_id,
                    "target_room": room,
                },
                {
                    "step_id": 2,
                    "primitive": "INTERACT",
                    "nl": primary_task.replace("_", " "),
                    "target_object": object_id,
                    "target_room": room,
                },
            ]
        if not task_objects:
            return []
        task_object = task_objects[0]
        object_id = task_object["object_name"]
        source_room = (task_object.get("placement") or {}).get("room_id") or target_room
        plan = [
            {
                "step_id": 1,
                "primitive": "MOVE",
                "nl": f"Move to {object_id}",
                "target_object": object_id,
                "target_room": source_room,
            },
            {
                "step_id": 2,
                "primitive": "PICK",
                "nl": f"Pick up {object_id}",
                "target_object": object_id,
                "target_room": source_room,
            },
        ]
        if primary_task.startswith("deliver_") and delivery_destination:
            destination_id = delivery_destination["object_id"]
            destination_room = delivery_destination["room_id"]
            plan.extend([
                {
                    "step_id": 3,
                    "primitive": "MOVE",
                    "nl": f"Move to {destination_id}",
                    "target_object": destination_id,
                    "target_room": destination_room,
                },
                {
                    "step_id": 4,
                    "primitive": "PLACE",
                    "nl": f"Place {object_id} on {destination_id}",
                    "target_object": destination_id,
                    "target_room": destination_room,
                    "placement_mode": delivery_destination.get("placement_mode", "on_top"),
                },
            ])
        return plan

    def _enforce_enva_solution_plan(
        self,
        plan,
        primary_task,
        task_category,
        target_room,
        plan_objects,
        task_objects,
        native_target=None,
        delivery_destination=None,
    ):
        """Keep a valid LLM plan, otherwise replace it with an exact-ID plan."""
        fallback = self._fallback_enva_solution_plan(
            primary_task,
            target_room,
            task_objects,
            native_target=native_target,
            delivery_destination=delivery_destination,
        )
        task_object_id = task_objects[0]["object_name"] if task_objects else None
        destination_id = (
            delivery_destination.get("object_id") if delivery_destination else None
        )
        native_target_id = native_target.get("object_id") if native_target else None

        if task_category == "retrieval_delivery" and (
            not task_object_id
            or (primary_task.startswith("deliver_") and not destination_id)
        ):
            # The enclosing generation result is already marked rejected. Keep
            # it serializable instead of turning a structured placement failure
            # into a process-level exception.
            return fallback

        def validate(candidate):
            run = {
                "ok": True,
                "task_environment": {
                    "task": {
                        "primary_behavior_task": primary_task,
                        "plan_objects": plan_objects,
                    },
                    "solution_plan": candidate,
                },
            }
            return validate_env_a_plan_contract(
                run,
                task_object_id=task_object_id,
                destination_object_id=destination_id,
                native_target_id=native_target_id,
            )

        try:
            validate(plan)
            return plan
        except ExpertPlanError as exc:
            print(
                f"[plan-contract] replacing invalid LLM plan for {primary_task}: {exc}",
                flush=True,
            )
        validate(fallback)
        return fallback

    def _build_task_instance(
        self,
        run_id,
        primary_task,
        target_room,
        created_objects,
        instruction=None,
        required_objs=None,
        task_category=None,
        native_target=None,
        delivery_destination=None,
    ):
        task_objects = [item for item in created_objects if item.get("semantic_role") == "task_object"]

        # Build object info for LLM plan generation — includes actual placement info
        plan_objects = []
        for item in task_objects:
            placement = item.get("placement") or {}
            obj_info = {
                "object_id": item["object_name"],
                "category": item["category"],
                "model": item.get("model"),
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

        if native_target:
            plan_objects.append({
                "object_id": native_target["object_id"],
                "category": native_target["category"],
                "room": native_target["room_id"],
                "reused": True,
                "reference_only": True,
                "semantic_role": "target",
                "robot_approach": native_target.get("robot_approach"),
                "manipulation_height": native_target.get("manipulation_height"),
            })

        if delivery_destination:
            plan_objects.append({
                "object_id": delivery_destination["object_id"],
                "category": delivery_destination["category"],
                "room": delivery_destination["room_id"],
                "reused": True,
                "reference_only": True,
                "semantic_role": "delivery_destination",
                "placement_mode": delivery_destination.get("placement_mode", "on_top"),
                "robot_approach": delivery_destination.get("robot_approach"),
                "manipulation_height": delivery_destination.get("manipulation_height"),
            })

        # Find scene objects referenced by the instruction but not in task_objects.
        # These are objects the LLM might need to reference in the plan (e.g., fridge,
        # countertop) that exist in the scene but weren't selected as task objects.
        if instruction and self._llm_client and task_category not in {
            "retrieval_delivery", "open_close", "appliance",
        }:
            task_cats = {o["category"].lower() for o in plan_objects}
            instruction_lower = instruction.lower()
            for obj in self._scene_objects_prefer_room(target_room):
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
        if required_objs and instruction and task_category not in {
            "retrieval_delivery", "open_close", "appliance",
        }:
            task_cats = {o["category"].lower() for o in plan_objects}
            for robj in required_objs:
                hint = robj.get("category_hint", "").lower().replace("_", " ")
                name = robj.get("name", "").lower().replace("_", " ")
                for obj in self._scene_objects_prefer_room(target_room):
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

        if task_category in {"open_close", "appliance"} and not task_objects and plan_objects:
            actual_room = plan_objects[0].get("room")
            if actual_room and actual_room != target_room:
                instruction = self._rewrite_instruction_room(instruction, target_room, actual_room)
                target_room = actual_room

        # Refresh instruction with actual placement info
        # Retrieval instructions are built from the accepted physical pose
        # above. Do not let the LLM overwrite that ground truth with an
        # invented destination during the generic instruction-refresh step.
        if self._llm_client and instruction and task_category != "retrieval_delivery":
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

        if task_category in {"retrieval_delivery", "open_close", "appliance"}:
            plan = self._enforce_enva_solution_plan(
                plan,
                primary_task,
                task_category,
                target_room,
                plan_objects,
                task_objects,
                native_target=native_target,
                delivery_destination=delivery_destination,
            )
        # Legacy fallback for non-Env-A contracts.
        elif not plan:
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

    def _scene_objects_prefer_room(self, target_room):
        def sort_key(obj):
            rooms = self._rooms_for_obj(obj)
            return (0 if target_room in rooms else 1, getattr(obj, "name", ""))

        return sorted(self._scene_objects(), key=sort_key)

    @staticmethod
    def _rewrite_instruction_room(instruction, old_room, new_room):
        if not instruction:
            return instruction
        old_label = str(old_room or "").rsplit("_", 1)[0].replace("_", " ")
        new_label = str(new_room or "").rsplit("_", 1)[0].replace("_", " ")
        if not old_label or not new_label or old_label == new_label:
            return instruction
        for prefix in ("in the ", "in "):
            needle = f"{prefix}{old_label}"
            if needle in instruction:
                return instruction.replace(needle, f"{prefix}{new_label}", 1)
        return instruction

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
        robot_record = self._robot_record(target_room, graph)
        cameras, camera_coverage = self._camera_records(
            target_room=target_room,
            graph=graph,
            initial_room=robot_record.get("initial_room") if robot_record else None,
            task_instance=task_instance,
            added_objects=added_objects,
            state_changed_objects=state_changed_objects,
        )
        validation["camera_coverage"] = camera_coverage
        if not camera_coverage.get("ok"):
            validation["ok"] = False
            task_name = task_instance.get("primary_behavior_task", "")
            if (
                target_room
                and self._category_for_task(task_name) == "retrieval_delivery"
            ):
                self._rejected_rooms.add(target_room)
                print(
                    f"[room-memory] rejecting room {target_room} for this process "
                    f"(initial_camera_coverage); rejected={sorted(self._rejected_rooms)}",
                    flush=True,
                )
        return {
            "schema_version": "task_environment.v1",
            "env_id": env_id,
            "env_type": env_type,
            "base_scene": {
                "scene_model": scene_model,
                "base_env_usd": f"{scene_model}.usd" if scene_model else None,
                "source": "live_omnigibson_env",
            },
            "generation": {
                "llm_enabled": self._llm_client is not None,
                "llm_model": self.config.llm_model,
                "solution_plan_policy": "llm_with_exact_env_a_contract_fallback",
                "solvability_profile": self.config.solvability_profile,
            },
            "task": {
                "task_id": task_instance["task_id"],
                "task_type": task_instance["task_type"],
                "primary_behavior_task": task_instance.get("primary_behavior_task", ""),
                "instruction": task_instance["instruction"],
                "target_room": target_room,
                "semantic_constraints": task_instance.get("semantic_constraints", []),
                "semantic_reasoning": task_instance.get("semantic_reasoning"),
                "plan_objects": task_instance.get("plan_objects", []),
            },
            "robot": robot_record,
            "camera": cameras,
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
                        "robot_approach": (item.get("placement") or {}).get("robot_approach"),
                        "manipulation_height": (item.get("placement") or {}).get(
                            "manipulation_height"
                        ),
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
                        "robot_approach": (item.get("placement") or {}).get("robot_approach"),
                        "manipulation_height": (item.get("placement") or {}).get(
                            "manipulation_height"
                        ),
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
            "camera_coverage": validation.get("camera_coverage"),
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

    def _camera_records(
        self,
        target_room,
        graph,
        initial_room=None,
        task_instance=None,
        added_objects=None,
        state_changed_objects=None,
    ):
        cameras = []
        targets = self._camera_target_objects(
            task_instance or {}, graph, added_objects or [], state_changed_objects or [],
        )
        target_names = set(targets)
        robot_visible = {}
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
                is_primary = self._is_primary_robot_camera(name, sensor)
                cameras.append(
                    {
                        "camera_id": str(name),
                        "camera_type": "robot_camera",
                        "attached_to": getattr(robot, "name", "robot_0"),
                        "room_id": None,
                        "pose": self._object_pose_record(sensor),
                        "modalities": self._sensor_modalities(sensor),
                        "resolution": self._sensor_resolution(sensor),
                        "is_primary": is_primary,
                    }
                )
            robot_visible = self._robot_camera_visibility(robot, target_names)

        for camera in cameras:
            if camera.get("is_primary"):
                camera["visible_task_objects"] = sorted(robot_visible)
                camera["visibility"] = robot_visible

        room_centers = (graph.get("navigation") or {}).get("room_centers", {})
        primary_task = str((task_instance or {}).get("primary_behavior_task") or "")
        # A robot view that covers delivery targets only at reset is not
        # persistent: after navigating to the source, the destination can
        # disappear even when both objects are in the same room.
        persistent_global_targets = (
            set(target_names)
            if primary_task.startswith("deliver_")
            else set()
        )
        uncovered = (target_names - set(robot_visible)) | persistent_global_targets
        placed_rooms = set()
        global_visible = {}
        errors = []

        # When the robot already sees every task object, keep one randomized
        # official room camera so the instance still has a global viewpoint.
        pending_rooms = []
        if not uncovered:
            available_rooms = sorted(room_centers)
            if available_rooms:
                pending_rooms.append(self.rng.choice(available_rooms))

        while uncovered or pending_rooms:
            if len(placed_rooms) >= self.config.max_global_cameras:
                errors.append({
                    "error": "max_global_cameras_reached",
                    "limit": self.config.max_global_cameras,
                    "uncovered_objects": sorted(uncovered),
                })
                break
            if pending_rooms:
                room_id = pending_rooms.pop(0)
            else:
                object_id = sorted(uncovered)[0]
                target_rooms = targets[object_id].get("room_ids") or [
                    targets[object_id].get("room_id")
                ]
                target_rooms = [room for room in target_rooms if room]
                room_id = next((room for room in target_rooms if room not in placed_rooms), None)
                if not target_rooms:
                    errors.append({"error": "target_room_unknown", "object_id": object_id})
                    break
                if room_id is None:
                    errors.append({
                        "error": "target_not_visible_from_room_camera",
                        "object_id": object_id,
                        "room_ids": target_rooms,
                    })
                    break
            center = room_centers.get(room_id)
            if room_id in placed_rooms:
                continue
            if not center:
                errors.append({"error": "room_camera_pose_unavailable", "room_id": room_id})
                break
            best_camera = None
            room_targets = {
                name for name in uncovered
                if room_id in (targets[name].get("room_ids") or [targets[name].get("room_id")])
            }
            camera_candidates = self._global_camera_candidates(room_id, center)
            if primary_task.startswith("deliver_") and room_targets:
                graph_nodes = {
                    node.get("id"): node
                    for node in graph.get("nodes", [])
                    if node.get("type") == "object"
                }
                focus_points = []
                for name in room_targets:
                    node = graph_nodes.get(name) or {}
                    bbox = node.get("bbox") or {}
                    lower = bbox.get("min")
                    upper = bbox.get("max")
                    if lower and upper:
                        focus_points.append(
                            0.5
                            * (
                                np.asarray(lower[:3], dtype=np.float32)
                                + np.asarray(upper[:3], dtype=np.float32)
                            )
                        )
                    else:
                        position = (node.get("pose") or {}).get("position")
                        if position:
                            focus_points.append(np.asarray(position[:3], dtype=np.float32))
                if focus_points:
                    task_focus = np.mean(focus_points, axis=0)
                    camera_candidates = [
                        (
                            position,
                            self._look_at_quat(position, task_focus),
                            f"{method}_task_aim",
                        )
                        for position, _, method in camera_candidates
                    ]
                    print(
                        f"[camera-coverage] task-aimed delivery candidates "
                        f"room={room_id} targets={sorted(room_targets)}",
                        flush=True,
                    )
            full_height_fixture = any(
                self._tokens(targets[name].get("category")) & {"door", "doors", "window", "windows"}
                for name in room_targets
            )
            if full_height_fixture:
                graph_nodes = {
                    node.get("id"): node for node in graph.get("nodes", [])
                    if node.get("type") == "object"
                }
                focus_points = [
                    (graph_nodes.get(name, {}).get("pose") or {}).get("position")
                    for name in room_targets
                ]
                focus_points = [point for point in focus_points if point]
                focus_xy = np.mean(
                    [np.asarray(point[:2], dtype=np.float32) for point in focus_points], axis=0,
                ) if focus_points else None

                def full_height_rank(item):
                    if not item[2].endswith("_20"):
                        return (1, 0.0, item[2])
                    distance = (
                        float(np.linalg.norm(np.asarray(item[0][:2]) - focus_xy))
                        if focus_xy is not None else 0.0
                    )
                    return (0, -distance, item[2])

                camera_candidates.sort(
                    key=full_height_rank
                )
            if not uncovered:
                camera_candidates = camera_candidates[:1]
            else:
                camera_candidates = camera_candidates[
                    : self.config.max_camera_pose_attempts_per_room
                ]
            for cam_pos, orientation, method in camera_candidates:
                print(
                    f"[camera-coverage] testing room={room_id} pose={method}",
                    flush=True,
                )
                visibility = self._global_camera_visibility(cam_pos, orientation, target_names)
                print(
                    f"[camera-coverage] tested room={room_id} pose={method} "
                    f"visible={sorted(visibility)}",
                    flush=True,
                )
                score = (
                    len(set(visibility) & uncovered),
                    sum(item.get("pixel_count", 0) for item in visibility.values()),
                )
                if best_camera is None or score > best_camera[0]:
                    best_camera = (score, cam_pos, orientation, method, visibility)
                if room_targets and room_targets <= set(visibility):
                    break
            _, cam_pos, orientation, camera_method, visibility = best_camera
            camera_id = f"global_{room_id}"
            cameras.append(
                {
                    "camera_id": camera_id,
                    "camera_type": "global_camera",
                    "room_id": room_id,
                    "pose": {
                        "position": [float(value) for value in cam_pos],
                        "orientation_xyzw": [float(value) for value in orientation],
                    },
                    "modalities": ["rgb", "seg_instance"],
                    "resolution": {"height": 720, "width": 1280},
                    "status": "visibility_validated",
                    "placement_method": camera_method,
                    "visible_task_objects": sorted(visibility),
                    "visibility": visibility,
                }
            )
            placed_rooms.add(room_id)
            global_visible.update(visibility)
            uncovered -= set(visibility)

        visible = set(robot_visible) | set(global_visible)
        coverage = {
            "ok": bool(target_names) and not errors and visible >= target_names
            and set(global_visible) >= persistent_global_targets,
            "policy": "robot_first_iterative_room_camera_v2_persistent_delivery",
            "max_global_cameras": self.config.max_global_cameras,
            "target_objects": sorted(target_names),
            "robot_visible_objects": sorted(robot_visible),
            "global_visible_objects": sorted(global_visible),
            "persistent_global_targets": sorted(persistent_global_targets),
            "visible_objects": sorted(visible),
            "uncovered_objects": sorted(target_names - visible),
            "global_camera_rooms": sorted(placed_rooms),
            "errors": errors,
        }
        if not target_names:
            coverage["errors"].append({"error": "no_task_objects_for_visibility"})
        print(
            "[camera-coverage] "
            f"ok={coverage['ok']} targets={coverage['target_objects']} "
            f"robot={coverage['robot_visible_objects']} "
            f"global={coverage['global_visible_objects']} "
            f"rooms={coverage['global_camera_rooms']} "
            f"uncovered={coverage['uncovered_objects']} errors={coverage['errors']}",
            flush=True,
        )
        return cameras, coverage

    def _global_camera_candidates(self, room_id, room_center):
        candidates = []
        preferred_pos, preferred_ori = self._compute_global_camera_pose(room_id, room_center)
        candidates.append((preferred_pos, preferred_ori, "official_preferred"))
        corners = self._get_room_corners_from_objects(room_id)
        if corners is not None:
            opposite = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}
            room_xy = np.asarray(room_center[:2], dtype=np.float32)
            room_diagonal = float(
                np.linalg.norm(np.asarray(corners["NE"][:2]) - np.asarray(corners["SW"][:2]))
            )
            walls = (
                ("SW_SE", corners["SW"], corners["SE"]),
                ("SE_NE", corners["SE"], corners["NE"]),
                ("NE_NW", corners["NE"], corners["NW"]),
                ("NW_SW", corners["NW"], corners["SW"]),
            )

            # Spend the bounded visibility budget on spatial coverage first.
            # The four object-derived corners are stable across scenes. Extra
            # wall midpoints remain fallbacks because an inferred wall point
            # can land inside geometry in cluttered rooms.
            corner_names = ("SW", "SE", "NW", "NE") if np.isfinite(room_diagonal) and room_diagonal >= 1e-6 else ()
            for corner_name in corner_names:
                pos, ori = self._compute_corner_camera(
                    corners[corner_name], corners[opposite[corner_name]], v_angle=20,
                )
                candidates.append((pos, ori, f"official_corner_{corner_name}_20"))
            for corner_name in corner_names:
                pos, ori = self._compute_corner_camera(
                    corners[corner_name], corners[opposite[corner_name]], v_angle=45,
                )
                candidates.append((pos, ori, f"official_corner_{corner_name}_45"))

            for angle in (30, 60):
                for corner_name in corner_names:
                    pos, ori = self._compute_corner_camera(
                        corners[corner_name], corners[opposite[corner_name]], v_angle=angle,
                    )
                    candidates.append((pos, ori, f"official_corner_{corner_name}_{angle}"))
            for angle in (45, 30, 60):
                for wall_name, c1, c2 in walls:
                    pose = self._compute_inward_wall_camera(c1, c2, room_xy, v_angle=angle)
                    if pose is not None:
                        candidates.append((*pose, f"official_wall_{wall_name}_{angle}"))
        unique = []
        seen = set()
        for pos, ori, method in candidates:
            key = tuple(np.round(np.concatenate((pos, ori)), 3))
            if key in seen:
                continue
            seen.add(key)
            unique.append((pos, ori, method))
        return unique

    @classmethod
    def _compute_inward_wall_camera(cls, c1, c2, room_center, v_angle=45.0, inward=0.2, height=2.2):
        wall_center = np.asarray([(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5], dtype=np.float32)
        wall_dir = np.asarray([c2[0] - c1[0], c2[1] - c1[1]], dtype=np.float32)
        norm = float(np.linalg.norm(wall_dir))
        if norm < 1e-6:
            return None
        wall_dir /= norm
        normals = (
            np.asarray([-wall_dir[1], wall_dir[0]], dtype=np.float32),
            np.asarray([wall_dir[1], -wall_dir[0]], dtype=np.float32),
        )
        normal = max(normals, key=lambda value: float(np.dot(value, room_center - wall_center)))
        cam_pos = np.asarray([
            wall_center[0] + normal[0] * inward,
            wall_center[1] + normal[1] * inward,
            height,
        ], dtype=np.float32)
        look_at = np.asarray([room_center[0], room_center[1], 0.6], dtype=np.float32)
        return cam_pos, cls._look_at_quat(cam_pos, look_at)

    def _camera_target_objects(self, task_instance, graph, added_objects, state_changed_objects):
        by_id = {}
        for item in added_objects:
            object_id = item.get("object_id") or item.get("object_name")
            if object_id:
                by_id[object_id] = {"room_id": item.get("room_id"), "category": item.get("category")}
        for item in state_changed_objects:
            object_id = item.get("object_id") or item.get("object_name")
            if object_id:
                by_id[object_id] = {"room_id": item.get("room_id"), "category": item.get("category")}
        for item in task_instance.get("plan_objects") or []:
            object_id = item.get("object_id")
            if object_id:
                by_id.setdefault(object_id, {
                    "room_id": item.get("room") or item.get("room_id"),
                    "category": item.get("category"),
                })
        graph_nodes = {
            node.get("id"): node for node in graph.get("nodes", [])
            if node.get("type") == "object"
        }
        actionable = {
            step.get(key)
            for step in task_instance.get("solution_plan") or []
            for key in ("target_object", "tool_object")
            if step.get(key)
        }
        actionable.update(
            item.get("object_id") for item in task_instance.get("task_objects") or []
            if item.get("object_id")
        )
        targets = {}
        for object_id in sorted(actionable):
            node = graph_nodes.get(object_id) or {}
            record = dict(by_id.get(object_id) or {})
            rooms = node.get("rooms") or []
            recorded_rooms = record.get("room_ids") or record.get("room_id") or []
            if isinstance(recorded_rooms, str):
                recorded_rooms = [recorded_rooms]
            all_rooms = list(dict.fromkeys([*recorded_rooms, *rooms]))
            record["room_ids"] = all_rooms
            record["room_id"] = all_rooms[0] if all_rooms else None
            record["category"] = record.get("category") or node.get("category")
            targets[object_id] = record
        return targets

    @staticmethod
    def _is_primary_robot_camera(name, sensor):
        label = f"{name} {getattr(sensor, 'name', '')}".lower()
        return any(token in label for token in ("eyes", "head")) and not any(
            token in label for token in ("eef", "wrist")
        )

    def _robot_camera_visibility(self, robot, target_names):
        sensors = getattr(robot, "sensors", None) or getattr(robot, "_sensors", None) or {}
        items = list(sensors.items()) if isinstance(sensors, dict) else [
            (getattr(sensor, "name", f"sensor_{index}"), sensor)
            for index, sensor in enumerate(sensors)
        ]
        cameras = [item for item in items if self._looks_like_camera_sensor(item[1])]
        primary = [item for item in cameras if self._is_primary_robot_camera(*item)]
        selected = primary[0] if primary else (cameras[0] if cameras else None)
        if selected is None:
            print("[camera-coverage] robot has no primary camera", flush=True)
            return {}
        name, sensor = selected
        try:
            position, orientation = sensor.get_position_orientation()
            resolution = self._sensor_resolution(sensor) or {}
            width = int(resolution.get("width") or 1280)
            height = int(resolution.get("height") or 720)
            visibility = self._geometric_camera_visibility(
                position, orientation, target_names, width=width, height=height
            )
            print(
                f"[camera-coverage] geometric robot probe on {name} "
                f"visible={sorted(visibility)}",
                flush=True,
            )
            return visibility
        except Exception as exc:
            print(f"[camera-coverage] robot geometric probe failed: {exc}", flush=True)
            traceback.print_exc()
            return {}

    def _global_camera_visibility(self, position, orientation, target_names):
        try:
            return self._geometric_camera_visibility(
                position, orientation, target_names, width=1280, height=720
            )
        except Exception as exc:
            print(f"[camera-coverage] global geometric probe failed: {exc}", flush=True)
            traceback.print_exc()
            return {}

    def _geometric_camera_visibility(
        self, position, orientation, target_names, width, height, horizontal_fov_deg=65.0
    ):
        camera_position = np.asarray(position, dtype=np.float64)
        rotation = T.quat2mat(th.as_tensor(orientation, dtype=th.float32)).cpu().numpy()
        focal = 0.5 * float(width) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
        visibility = {}
        for name in sorted(target_names):
            obj = self.env.scene.object_registry("name", name, None)
            if obj is None:
                continue
            lower, upper = obj.aabb
            lower = np.asarray(lower.cpu() if hasattr(lower, "cpu") else lower, dtype=np.float64)
            upper = np.asarray(upper.cpu() if hasattr(upper, "cpu") else upper, dtype=np.float64)
            corners = np.asarray([
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ])
            camera_points = (corners - camera_position) @ rotation
            in_front = camera_points[:, 2] < -0.05
            if not np.any(in_front):
                print(f"[camera-coverage] geometric reject {name}: behind_camera", flush=True)
                continue
            points = camera_points[in_front]
            depth = -points[:, 2]
            pixels_x = focal * points[:, 0] / depth + width * 0.5
            pixels_y = height * 0.5 - focal * points[:, 1] / depth
            x1 = max(0, int(math.floor(float(np.min(pixels_x)))))
            y1 = max(0, int(math.floor(float(np.min(pixels_y)))))
            x2 = min(width - 1, int(math.ceil(float(np.max(pixels_x)))))
            y2 = min(height - 1, int(math.ceil(float(np.max(pixels_y)))))
            if x2 <= x1 or y2 <= y1:
                print(
                    f"[camera-coverage] geometric reject {name}: outside_frame "
                    f"raw_bbox={[float(np.min(pixels_x)), float(np.min(pixels_y)), float(np.max(pixels_x)), float(np.max(pixels_y))]}",
                    flush=True,
                )
                continue
            pixel_count = (x2 - x1 + 1) * (y2 - y1 + 1)
            if pixel_count < self.config.visibility_min_pixels:
                print(
                    f"[camera-coverage] geometric reject {name}: pixels={pixel_count}",
                    flush=True,
                )
                continue
            center = (lower + upper) * 0.5
            target_path = str(getattr(obj, "prim_path", ""))
            ray_points = [center, np.asarray([center[0], center[1], upper[2]]), *corners]
            visible_hit_path = None
            surface_visible = False
            blocked_paths = []
            for ray_point in ray_points:
                distance = float(np.linalg.norm(ray_point - camera_position))
                if distance < 1e-6:
                    visible_hit_path = "camera_inside_target"
                    surface_visible = True
                    break
                ray = og.sim.psqi.raycast_closest(
                    origin=camera_position.tolist(),
                    dir=((ray_point - camera_position) / distance).tolist(),
                    distance=distance,
                )
                hit_path = str(ray.get("rigidBody") or ray.get("collision") or "")
                hit_distance = float(ray.get("distance", distance))
                blocked = bool(
                    ray.get("hit")
                    and target_path not in hit_path
                    and hit_distance > 0.35
                    and hit_distance < distance - 0.03
                )
                if not blocked:
                    visible_hit_path = hit_path or None
                    surface_visible = True
                    break
                blocked_paths.append(f"{hit_path}@{hit_distance:.2f}m")
            if not surface_visible:
                print(
                    f"[camera-coverage] geometric reject {name}: occluded_by="
                    f"{sorted(set(blocked_paths))[:5]}",
                    flush=True,
                )
                continue
            margin = max(2, int(round(min(width, height) * 0.01)))
            visibility[name] = {
                "pixel_count": int(pixel_count),
                "bbox_xyxy": [x1, y1, x2, y2],
                "image_size": [int(width), int(height)],
                "bbox_clipped": bool(
                    x1 < margin or y1 < margin
                    or x2 >= width - margin or y2 >= height - margin
                ),
                "visibility_source": "official_camera_frustum_physx_raycast",
                "occlusion_hit_path": visible_hit_path,
            }
        return visibility

    @classmethod
    def _iter_instance_observations(cls, obs, info, path=()):
        if not isinstance(obs, dict):
            return
        if "seg_instance" in obs:
            label_map = info.get("seg_instance", {}) if isinstance(info, dict) else {}
            yield path, obs["seg_instance"], label_map
        for key, value in obs.items():
            if isinstance(value, dict):
                child_info = info.get(key, {}) if isinstance(info, dict) else {}
                yield from cls._iter_instance_observations(value, child_info, path + (str(key),))

    @staticmethod
    def _instance_target_stats(seg, label_map, target_names):
        try:
            if hasattr(seg, "detach"):
                seg = seg.detach()
            if hasattr(seg, "cpu"):
                seg = seg.cpu()
            array = np.asarray(seg)
            array = np.squeeze(array)
        except Exception:
            return {}
        if array.ndim != 2 or not isinstance(label_map, dict):
            return {}
        target_instance_ids = defaultdict(list)
        for raw_id, raw_label in label_map.items():
            label = str(raw_label)
            path_parts = set(label.rstrip("/").split("/"))
            name = next(
                (target for target in target_names if target == label or target in path_parts),
                None,
            )
            if name is None:
                continue
            try:
                instance_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            target_instance_ids[name].append(instance_id)

        stats = {}
        for name, instance_ids in target_instance_ids.items():
            ys, xs = np.where(np.isin(array, instance_ids))
            if not len(xs):
                continue
            width, height = int(array.shape[1]), int(array.shape[0])
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            stats[name] = {
                "pixel_count": int(len(xs)),
                "bbox_xyxy": [x1, y1, x2, y2],
                "image_size": [width, height],
            }
        return stats

    def _visibility_from_instance(self, seg, label_map, target_names):
        visibility = {}
        for name, stats in self._instance_target_stats(seg, label_map, target_names).items():
            if stats["pixel_count"] < self.config.visibility_min_pixels:
                continue
            x1, y1, x2, y2 = stats["bbox_xyxy"]
            width, height = stats["image_size"]
            margin = max(2, int(round(min(width, height) * 0.01)))
            if x1 < margin or y1 < margin or x2 >= width - margin or y2 >= height - margin:
                continue
            visibility[name] = stats
        return visibility

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
        if not np.isfinite(diag_len) or diag_len < 1e-6:
            raise ValueError("invalid room diagonal")
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

        right = np.cross(d, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        # Camera local -Z is the view direction. Build an upright right-handed
        # basis: right x up = -view and the projected up axis follows world up.
        up = np.cross(-d, right)
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
        preferred = record.get("_preferred_category")
        if preferred in categories:
            return preferred
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

    def _choose_target_model(self, category, preferred_models=None):
        """Choose a least-used model so large batches cover task assets evenly."""
        try:
            models = sorted(get_all_object_category_models(category=category))
        except Exception:
            models = []
        if not models:
            raise ValueError(f"No available models found for category {category}")
        if self.config.target_asset_model and category == self.config.target_asset_category:
            if self.config.target_asset_model not in models:
                raise ValueError(
                    f"Model {self.config.target_asset_model!r} is not installed for {category!r}"
                )
            return self.config.target_asset_model
        if preferred_models is not None:
            preferred = [model for model in preferred_models if model in models]
            if not preferred:
                raise ValueError(
                    f"No preferred models are installed for category {category}: "
                    f"{preferred_models}"
                )
            models = preferred
        eligible = [
            model for model in models
            if self._failed_target_models.get((category, model), 0)
            < self.config.max_failures_per_target_model
        ]
        if not eligible:
            raise ValueError(
                f"All models for category {category} reached the failure limit "
                f"({self.config.max_failures_per_target_model})"
            )
        min_used = min(self._used_target_models.get((category, model), 0) for model in eligible)
        candidates = [
            model for model in eligible
            if self._used_target_models.get((category, model), 0) == min_used
        ]
        return self.rng.choice(candidates)

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
        if not og.sim.is_playing():
            og.sim.play()
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

    def _capture_physics_rebuild_poses(self):
        """Capture live poses that a stop/play physics rebuild would revert.

        ``og.sim.stop()`` followed by ``og.sim.play()`` re-initializes PhysX
        from the USD stage and ``play()`` additionally calls ``Robot.reset()``
        on every robot. Poses written only to the GPU tensor view during play
        (a generated object's confirmed placement, the stabilized robot spawn)
        are NOT persisted to USD, so the rebuild silently reverts them:
        observed in enva_gen5_pbfix2 as a placed nightstand teleporting back
        to its import pose at the scene origin (under a door prim, which then
        read as ``support_surface_occupied``) and the robot teleporting 7.7 m
        back to its initial pose (tripping scene_integrity on every sample).
        Record the live poses here so they can be re-applied right after the
        rebuild. Native movable furniture must be included: Benevolence_1's
        ``bookcase_owvfik_3`` shifts 15 cm on rebuild even when the generated
        object is placed on a different bookcase.
        """
        saved = []
        try:
            robots = list(getattr(self.env, "robots", None) or [])
        except Exception:
            robots = []
        targets = list(robots)
        try:
            targets.extend(self._scene_objects())
        except Exception:
            pass
        seen = set()
        for obj in targets:
            key = getattr(obj, "prim_path", None) or getattr(obj, "name", None) or id(obj)
            if key in seen:
                continue
            seen.add(key)
            try:
                position, orientation = obj.get_position_orientation()
                saved.append((obj, position.clone(), orientation.clone()))
            except Exception:
                continue
        return saved

    def _restore_physics_rebuild_poses(self, saved):
        """Re-apply poses displaced by a stop/play physics rebuild."""
        for obj, position, orientation in saved or []:
            try:
                current_position, current_orientation = obj.get_position_orientation()
                position_changed = float(((current_position - position) ** 2).sum() ** 0.5) > 1e-4
                orientation_delta = min(
                    float(((current_orientation - orientation) ** 2).sum() ** 0.5),
                    float(((current_orientation + orientation) ** 2).sum() ** 0.5),
                )
                if not position_changed and orientation_delta <= 1e-4:
                    continue
                obj.set_position_orientation(position=position, orientation=orientation)
            except Exception:
                continue
            try:
                if getattr(obj, "name", "") not in self._anchored_native_fixtures:
                    obj.keep_still()
            except Exception:
                pass

    def _stabilize_rebuilt_native_fixtures(self, saved):
        """Anchor only native, jointless fixtures destabilized by rebuild."""
        candidates = []
        robots = set(getattr(self.env, "robots", None) or [])
        for obj, position, orientation in saved or []:
            name = str(getattr(obj, "name", "") or "")
            tokens = self._tokens(getattr(obj, "category", "") or "")
            try:
                has_joints = bool(obj.joints)
            except Exception:
                has_joints = True
            if (
                obj in robots
                or name.startswith("online_env_")
                or has_joints
                or tokens & STRUCTURAL_CATEGORIES
            ):
                continue
            candidates.append((obj, position, orientation))
        if not candidates:
            return
        self._step(20)
        anchored = []
        for obj, position, orientation in candidates:
            try:
                current_position, _ = obj.get_position_orientation()
                displacement = float(th.linalg.norm(current_position - position))
            except Exception:
                continue
            if displacement <= 0.01:
                continue
            print(
                f"[physics-rebuild] anchoring unstable native fixture "
                f"{obj.name} displacement={displacement:.3f}m",
                flush=True,
            )
            try:
                obj.set_position_orientation(position=position, orientation=orientation)
                obj.root_link.set_attribute("physics:kinematicEnabled", True)
                self._anchored_native_fixtures.add(obj.name)
                anchored.append(obj.name)
            except Exception as exc:
                print(
                    f"[physics-rebuild] failed to anchor {obj.name}: {exc!r}",
                    flush=True,
                )
        if not anchored:
            return

        # A generated object already resting on a fixture follows that fixture
        # during the diagnostic settle window.  Once the fixture is restored
        # and anchored, restore generated objects from the same rebuild
        # transaction as well so their support relation is unchanged.
        for obj, position, orientation in saved or []:
            name = str(getattr(obj, "name", "") or "")
            if not name.startswith("online_env_"):
                continue
            try:
                obj.set_position_orientation(position=position, orientation=orientation)
                obj.keep_still()
            except Exception:
                continue
        self._step(2)

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
