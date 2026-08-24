"""
Run the online DeltaSG pipeline on a live OmniGibson scene.

This script is intended to be launched on the remote server where OmniGibson is
installed, for example:

conda run -n behavior python code/run_online_deltasg.py --scene Rs_int --robot fetch --num-envs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True
gm.GUI_VIEWPORT_ONLY = True

import omnigibson as og
import omnigibson.lazy as lazy

from api import RobotSpawnError, create_env, ensure_dir, stabilize_robot_spawn
from deltasg_expert import DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE, SUPPORTED_ENV_A_TASKS
from online_deltasg import OnlineDeltaSGConfig, OnlineDeltaSGEngine


def _round_pose(position):
    if not isinstance(position, (list, tuple)):
        return None
    return [round(float(value), 3) for value in position[:3]]


def sample_fingerprint(run):
    """Stable identity excluding generated run IDs, but retaining placement."""
    te = run.get("task_environment") or {}
    task = te.get("task") or run.get("task") or {}
    objects = []
    for item in te.get("added_objects") or []:
        placement = item.get("placement") or {}
        pose = item.get("pose") or placement.get("pose") or {}
        objects.append({
            "category": item.get("category"),
            "model": item.get("model"),
            "roles": sorted(item.get("semantic_roles") or []),
            "room": item.get("room_id"),
            "mode": placement.get("mode"),
            "support": placement.get("support_object_id"),
            "position": _round_pose(pose.get("position")),
        })
    plan_objects = [
        {
            "id": item.get("object_id"), "category": item.get("category"),
            "roles": sorted(item.get("semantic_roles") or [item.get("semantic_role")]),
            "room": item.get("room") or item.get("room_id"),
        }
        for item in task.get("plan_objects") or []
    ]
    payload = {
        "scene": ((te.get("base_scene") or {}).get("scene_model")),
        "env_type": te.get("env_type"),
        "primary_task": task.get("primary_behavior_task"),
        "task_type": task.get("task_type"),
        "target_room": task.get("target_room"),
        "objects": sorted(objects, key=lambda item: json.dumps(item, sort_keys=True)),
        "plan_objects": sorted(plan_objects, key=lambda item: json.dumps(item, sort_keys=True)),
        "state_changes": sorted(
            (item.get("object_id"), sorted((item.get("states") or {}).items()))
            for item in te.get("state_changed_objects") or []
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def sample_diversity_record(run):
    """Persist the sampling dimensions used to balance future batch rounds."""
    te = run.get("task_environment") or {}
    task = te.get("task") or {}
    objects = te.get("added_objects") or []
    plan_objects = task.get("plan_objects") or []
    task_objects = [
        item for item in objects
        if "task_object" in set(item.get("semantic_roles") or [])
    ]
    categories = [item.get("category") for item in task_objects if item.get("category")]
    if not categories:
        categories = [
            item.get("category") for item in plan_objects
            if item.get("category")
            and item.get("semantic_role") in {"target", "task_object"}
        ]
    supports = [
        (item.get("placement") or {}).get("support_category")
        for item in objects if (item.get("placement") or {}).get("support_category")
    ]
    source_rooms = sorted(
        set(item.get("room_id") for item in task_objects if item.get("room_id"))
    )
    position_bins = []
    target_models = set()
    for item in task_objects:
        if item.get("category") and item.get("model"):
            target_models.add((item["category"], item["model"]))
        position = ((item.get("pose") or {}).get("position"))
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            position_bins.append([round(float(position[0]) / 0.25), round(float(position[1]) / 0.25)])
    return {
        "env_type": te.get("env_type"),
        "task_family": task.get("task_type"),
        "primary_task": task.get("primary_behavior_task"),
        "target_room": task.get("target_room"),
        "source_rooms": source_rooms,
        "target_categories": sorted(set(categories)),
        "target_models": [
            {"category": category, "model": model}
            for category, model in sorted(target_models)
        ],
        "target_object_ids": sorted(set(
            item.get("object_id")
            for item in plan_objects
            if item.get("object_id")
        )),
        "support_categories": sorted(set(supports)),
        "position_bins_25cm": sorted(position_bins),
    }


def load_existing_fingerprints(output_dir):
    fingerprints = set()
    for path in Path(output_dir).glob("online_env*.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("ok"):
            fingerprints.add(sample_fingerprint(run)[0])
    return fingerprints


def load_existing_runs(output_dir):
    """Load accepted runs retained by an interrupted checkpointed process."""
    runs = []
    for path in sorted(Path(output_dir).glob("online_env*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("ok"):
            runs.append((run, path))
    return runs


def summary_item(run, path):
    validation = run.get("validation", {})
    created = validation.get("created_objects")
    failed = validation.get("failed_objects")
    return {
        "run_id": run["run_id"],
        "ok": run["ok"],
        "task": run["task_instance"]["instruction"],
        "target_room": run["task_instance"]["target_room"],
        "created": len(created) if isinstance(created, list) else None,
        "failed": len(failed) if isinstance(failed, list) else None,
        "path": str(path),
    }


def interaction_target(run):
    """Return the single fixture manipulated by a formal Env-A plan."""
    targets = {
        step.get("target_object")
        for step in ((run.get("task_environment") or {}).get("solution_plan") or [])
        if step.get("primitive") == "INTERACT" and step.get("target_object")
    }
    return next(iter(targets)) if len(targets) == 1 else None


def scene_integrity(before_graph, after_graph, max_displacement=0.05):
    """Reject a sample when a source-scene object is missing or displaced."""
    before = {
        node.get("id"): node for node in (before_graph or {}).get("nodes", [])
        if node.get("type") == "object"
        and node.get("category") != "agent"
        and not str(node.get("id", "")).startswith("online_env_")
    }
    after = {
        node.get("id"): node for node in (after_graph or {}).get("nodes", [])
        if node.get("type") == "object" and node.get("category") != "agent"
    }
    missing = []
    moved = []
    for object_id, node in before.items():
        after_node = after.get(object_id)
        if after_node is None:
            missing.append(object_id)
            continue
        start = ((node.get("pose") or {}).get("position"))
        end = ((after_node.get("pose") or {}).get("position"))
        if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)):
            continue
        displacement = sum((float(a) - float(b)) ** 2 for a, b in zip(start[:3], end[:3])) ** 0.5
        if displacement > max_displacement:
            moved.append({"object_id": object_id, "displacement": round(displacement, 6)})
    return {
        "ok": not missing and not moved,
        "max_displacement": max_displacement,
        "missing_source_objects": missing,
        "moved_source_objects": moved,
    }


def enforce_run_quality(run):
    validation = run.setdefault("validation", {})
    integrity = scene_integrity(run.get("before_graph"), run.get("after_graph"))
    validation["scene_integrity"] = integrity
    te_validation = ((run.get("task_environment") or {}).setdefault("validation", {}))
    te_validation["scene_integrity"] = integrity
    if not integrity["ok"]:
        validation["ok"] = False
        te_validation["ok"] = False
        run["ok"] = False
    return integrity


def enforce_expert_plan_quality(run):
    """Require a deterministic expert-plan compilation for successful Env-A samples."""
    te = run.get("task_environment") or {}
    if not run.get("ok") or te.get("env_type") != "Env-A":
        return None
    from deltasg_expert import ExpertPlanError, compile_expert_plan

    validation = run.setdefault("validation", {})
    te_validation = te.setdefault("validation", {})
    try:
        compiled = compile_expert_plan(run)
    except ExpertPlanError as exc:
        result = {"ok": False, "error": str(exc)}
        run["ok"] = False
        validation["ok"] = False
        te_validation["ok"] = False
    else:
        result = {
            "ok": True,
            "num_steps": len(compiled.steps),
            "warnings": list(compiled.warnings),
        }
        te["compiled_expert_plan"] = compiled.to_dict()
        run["compiled_expert_plan"] = compiled.to_dict()
    validation["expert_plan_preflight"] = result
    te_validation["expert_plan_preflight"] = result
    return result


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def harden_headless_kit():
    """Apply safe headless cleanup without unloading Kit extensions at runtime."""
    clear_usd_selection()


def clear_usd_selection():
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


def build_dataset_outputs(output_dir, summary, runs):
    output_dir = Path(output_dir)
    index_items = []
    task_environments = []
    for run, item in zip(runs, summary["runs"]):
        task_environment = run.get("task_environment")
        index_items.append(
            {
                "env_id": run.get("run_id"),
                "env_type": (task_environment or {}).get("env_type"),
                "ok": run.get("ok", False),
                "path": item["path"],
                "task_id": ((task_environment or {}).get("task") or {}).get("task_id"),
                "instruction": ((task_environment or {}).get("task") or {}).get("instruction")
                or run.get("task_instance", {}).get("instruction"),
                "target_room": ((task_environment or {}).get("task") or {}).get("target_room")
                or run.get("task_instance", {}).get("target_room"),
                "base_scene": ((task_environment or {}).get("base_scene") or {}).get("scene_model"),
                "num_added_objects": len((task_environment or {}).get("added_objects", [])),
                "num_task_objects": len((task_environment or {}).get("task_objects", [])),
                "num_solution_steps": len((task_environment or {}).get("solution_plan", [])),
                "validation_ok": ((task_environment or {}).get("validation") or {}).get("ok"),
            }
        )
        if task_environment is not None:
            task_environments.append(task_environment)

    dataset_index = {
        "schema_version": "deltasg_dataset_index.v1",
        "scene": summary.get("scene"),
        "robot": summary.get("robot"),
        "num_environments": len(index_items),
        "num_ok": sum(1 for item in index_items if item["ok"]),
        "items": index_items,
    }
    dataset = {
        "schema_version": "deltasg_dataset.v1",
        "scene": summary.get("scene"),
        "robot": summary.get("robot"),
        "num_environments": len(task_environments),
        "task_environments": task_environments,
    }

    index_path = output_dir / "dataset_index.json"
    dataset_path = output_dir / "dataset.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_index, f, ensure_ascii=False, indent=2, default=str)
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)
    return index_path, dataset_path


def main():
    parser = argparse.ArgumentParser(description="Run online DeltaSG Env-A generation in a live OmniGibson env.")
    parser.add_argument("--scene", default="Rs_int", help="OmniGibson scene model, e.g. Rs_int")
    parser.add_argument("--robot", default="fetch", help="Robot model. Use none to skip robot creation.")
    parser.add_argument(
        "--env-type",
        choices=["A", "B", "C"],
        default="A",
        help="DeltaSG environment type: A=basic task, B=fire anomaly, C=constraint semantic tasks.",
    )
    parser.add_argument("--task", default=None, help="Optional BEHAVIOR task name for Env-A, e.g. cook_eggplant-0")
    parser.add_argument(
        "--task-sequence",
        default=None,
        help="Comma-separated Env-A tasks to generate in order within one simulator process.",
    )
    parser.add_argument("--target-asset-category", default=None)
    parser.add_argument("--target-asset-model", default=None)
    parser.add_argument("--target-native-object-id", default=None)
    parser.add_argument(
        "--target-placement-mode",
        choices=("auto", "floor"),
        default="auto",
        help="Coverage diagnostic override; normal generation should use auto.",
    )
    parser.add_argument("--target-room", default=None, help="Optional room id, e.g. kitchen_0")
    parser.add_argument(
        "--task-categories",
        default=None,
        help="Comma-separated task categories to use. Default: all. "
             "Options: retrieval_delivery, open_close, appliance",
    )
    parser.add_argument(
        "--env-c-types",
        default=None,
        help="Comma-separated Env-C themes. Default: retrieval_delivery,open_close,appliance,fire.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of online DeltaSG edits to apply sequentially.")
    parser.add_argument("--task-objects", type=int, default=2)
    parser.add_argument("--context-objects", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--settle-threshold", type=float, default=0.25)
    parser.add_argument("--min-manipulation-height", type=float, default=0.10)
    parser.add_argument("--max-manipulation-height", type=float, default=1.55)
    parser.add_argument(
        "--min-global-cameras",
        type=int,
        default=2,
        help="Minimum official global cameras saved for every Env-A/B/C sample.",
    )
    parser.add_argument(
        "--max-global-cameras",
        type=int,
        default=3,
        help="Maximum official room cameras used to cover task objects at initialization.",
    )
    parser.add_argument("--max-camera-pose-attempts", type=int, default=6)
    parser.add_argument("--camera-pose-render-steps", type=int, default=4)
    parser.add_argument(
        "--solvability-profile",
        choices=("oracle_symbolic", "physical_control"),
        default="physical_control",
        help="Apply generation eligibility gates for the downstream expert backend.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. None = random each run.")
    parser.add_argument(
        "--allow-cloth",
        action="store_true",
        help="Allow cloth assets. Requires a working GPU dynamics setup.",
    )
    parser.add_argument(
        "--enable-transition-rules",
        action="store_true",
        help="Enable OmniGibson transition rules. Off by default for online DeltaSG scene editing.",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Optional asset metadata dir. Defaults to asset_pipeline/metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="code/outputs/online_deltasg",
        help="Directory where online run logs are written.",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3.8-max",
        help="Required LLM model name.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM API base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM API key (default: reads DASHSCOPE_API_KEY env var).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries per env when LLM validation rejects. 0 = unlimited.",
    )
    parser.add_argument(
        "--single-attempt",
        action="store_true",
        help="Run one generation attempt and return failure to an outer clean-process runner.",
    )
    # ---- Retry control ----
    parser.add_argument(
        "--max-llm-retries",
        type=int,
        default=3,
        help="Max LLM retries per scene before giving up.",
    )
    parser.add_argument(
        "--max-retries-per-task",
        type=int,
        default=1,
        help="Max retries for the same task name.",
    )
    parser.add_argument(
        "--max-generation-time",
        type=float,
        default=900.0,
        help="Max total generation time (seconds) per scene.",
    )
    # ---- Placement fail-fast ----
    parser.add_argument(
        "--placement-timeout",
        type=float,
        default=60.0,
        help="Per-object placement timeout (seconds).",
    )
    parser.add_argument(
        "--relation-timeout",
        type=float,
        default=10.0,
        help="Per-relation-attempt timeout (seconds).",
    )
    parser.add_argument(
        "--max-placement-attempts",
        type=int,
        default=4,
        help="Max placement attempts per object.",
    )
    parser.add_argument(
        "--max-total-placement-time",
        type=float,
        default=120.0,
        help="Max total placement time (seconds) per environment.",
    )
    parser.add_argument(
        "--max-model-failures",
        type=int,
        default=2,
        help="Skip a specific asset model after this many failed placements.",
    )
    parser.add_argument(
        "--no-abort-on-task-failure",
        action="store_true",
        help="Continue placing context objects even if task objects fail.",
    )
    parser.add_argument(
        "--no-skip-context-on-failure",
        action="store_true",
        help="Continue placing context objects even after task abort.",
    )
    parser.add_argument(
        "--unsafe-fast-env-a-cleanup",
        action="store_true",
        help="Unsafe optimization: remove spawned objects instead of resetting. Do not use for dataset generation.",
    )
    parser.add_argument(
        "--no-cache-base-graph",
        action="store_true",
        help="Rebuild the base scene graph for every Env-A sample.",
    )
    parser.add_argument(
        "--allow-repeat-tasks",
        action="store_true",
        help="Allow repeated primary task names across samples. Useful for large Env-A batches.",
    )
    # ---- Checkpoint ----
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N runs (0 = only at end).",
    )
    args = parser.parse_args()

    task_sequence = None
    if args.task_sequence:
        if args.env_type != "A":
            parser.error("--task-sequence is only supported for Env-A")
        if args.task:
            parser.error("--task and --task-sequence are mutually exclusive")
        task_sequence = [item.strip() for item in args.task_sequence.split(",") if item.strip()]
        invalid_tasks = sorted(set(task_sequence) - set(SUPPORTED_ENV_A_TASKS))
        if not task_sequence or invalid_tasks:
            parser.error(f"invalid --task-sequence tasks: {invalid_tasks or task_sequence}")
        args.num_envs = len(task_sequence)

    if args.min_manipulation_height < 0 or args.max_manipulation_height <= args.min_manipulation_height:
        parser.error("manipulation height bounds must satisfy 0 <= min < max")
    if args.max_camera_pose_attempts < 1 or args.camera_pose_render_steps < 1:
        parser.error("camera pose attempts and render steps must both be positive")
    if args.target_asset_model and not args.target_asset_category:
        parser.error("--target-asset-model requires --target-asset-category")
    if args.target_native_object_id and (args.target_asset_category or args.target_asset_model):
        parser.error("native-object and spawned-asset coverage controls are mutually exclusive")
    if args.target_placement_mode == "floor" and (
        args.env_type != "A" or not args.target_asset_category or args.target_native_object_id
    ):
        parser.error("--target-placement-mode floor requires an Env-A spawned target asset")

    normalized_robot = str(args.robot).lower()
    robot_model = None if normalized_robot in {"none", "null", ""} else normalized_robot
    ensure_dir(args.output_dir)
    env = None
    try:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = args.enable_transition_rules
        # Creating multiple Replicator annotators during Environment startup
        # can segfault in Isaac Sim 5.1. Start with RGB, stabilize the scene,
        # then attach instance segmentation to the primary camera only.
        env = create_env(
            scene_model=args.scene,
            robot_model=robot_model,
            robot_obs_modalities=["rgb"],
        )
        harden_headless_kit()
        stabilize_robot_spawn(
            env,
            seed=args.seed,
            preferred_target_name=args.target_native_object_id,
        )
        config = OnlineDeltaSGConfig(
            task_objects=args.task_objects,
            context_objects=args.context_objects,
            warmup_steps=args.warmup_steps,
            settle_steps=args.settle_steps,
            settle_threshold=args.settle_threshold,
            seed=args.seed,
            allow_cloth=args.allow_cloth,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            max_llm_retries_per_scene=args.max_llm_retries,
            max_retries_per_task=args.max_retries_per_task,
            max_total_generation_time_sec=args.max_generation_time,
            per_object_placement_timeout_sec=args.placement_timeout,
            per_relation_attempt_timeout_sec=args.relation_timeout,
            max_placement_attempts_per_object=args.max_placement_attempts,
            max_total_placement_time_sec=args.max_total_placement_time,
            max_failures_per_target_model=args.max_model_failures,
            abort_on_task_object_failure=not args.no_abort_on_task_failure,
            skip_context_on_failure=not args.no_skip_context_on_failure,
            fast_env_a_cleanup=args.unsafe_fast_env_a_cleanup,
            cache_base_graph=not args.no_cache_base_graph,
            allow_repeat_tasks=args.allow_repeat_tasks,
            min_global_cameras=args.min_global_cameras,
            max_global_cameras=args.max_global_cameras,
            max_camera_pose_attempts_per_room=args.max_camera_pose_attempts,
            camera_pose_render_steps=args.camera_pose_render_steps,
            solvability_profile=args.solvability_profile,
            max_task_object_approach_distance=DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
            expert_base_clearance_margin=(
                0.0 if args.solvability_profile == "oracle_symbolic" else 0.20
            ),
            min_manipulation_height=args.min_manipulation_height,
            max_manipulation_height=args.max_manipulation_height,
            target_asset_category=args.target_asset_category,
            target_asset_model=args.target_asset_model,
            target_native_object_id=args.target_native_object_id,
            target_placement_mode=args.target_placement_mode,
        )
        engine = OnlineDeltaSGEngine(env=env, metadata_dir=args.metadata_dir, config=config)

        # Parse task category filter
        if args.task_categories:
            enabled_categories = set(c.strip() for c in args.task_categories.split(","))
            from online_deltasg import VALID_TASKS
            invalid = enabled_categories - set(VALID_TASKS.keys())
            if invalid:
                print(f"ERROR: unknown categories: {invalid}. Valid: {sorted(VALID_TASKS.keys())}")
                hard_exit(1)
            engine.set_enabled_categories(enabled_categories)
            print(f"[online-deltasg] task categories: {sorted(enabled_categories)}")
        else:
            from online_deltasg import VALID_TASKS
            print(f"[online-deltasg] task categories: all {sorted(VALID_TASKS.keys())}")

        summary = {
            "ok": True,
            "scene": args.scene,
            "robot": robot_model,
            "num_envs": args.num_envs,
            "runs": [],
        }
        runs = []
        existing_run_count = 0
        skip_tasks = set()
        fingerprints = load_existing_fingerprints(args.output_dir)
        print(f"[quality] loaded {len(fingerprints)} existing sample fingerprints", flush=True)
        # Resume from checkpoint if requested
        if args.resume:
            loaded_skip_tasks = engine.load_checkpoint(args.output_dir)
            skip_tasks = set() if args.allow_repeat_tasks else loaded_skip_tasks
            existing = load_existing_runs(args.output_dir)
            runs.extend(run for run, _ in existing)
            summary["runs"].extend(summary_item(run, path) for run, path in existing)
            existing_run_count = len(existing)
            summary["num_envs"] = existing_run_count + args.num_envs
            existing_indices = [
                int(path.stem.rsplit("_", 1)[-1])
                for _, path in existing
                if path.stem.rsplit("_", 1)[-1].isdigit()
            ]
            if existing_indices:
                engine._run_counter = max(engine._run_counter, max(existing_indices) + 1)
            print(
                f"[checkpoint] retained {existing_run_count} accepted runs in resumed outputs",
                flush=True,
            )

        for idx in range(args.num_envs):
            current_task = task_sequence[idx] if task_sequence else args.task
            task_label = f" task={current_task}" if current_task else ""
            print(
                f"[online-deltasg] run {idx + 1}/{args.num_envs}{task_label}",
                flush=True,
            )
            run_start_time = time.time()
            run_llm_retries = 0
            run = None
            attempt = 0
            task_retry_count = {}  # task_name -> retry count
            # Diversity mode permits a task name in later samples, but a task
            # that repeatedly fails must still be excluded for this sample so
            # it cannot consume the entire retry budget.
            run_skip_tasks = set(skip_tasks)

            while True:
                # Check per-run generation time budget
                elapsed_run = time.time() - run_start_time
                if elapsed_run > config.max_total_generation_time_sec:
                    print(f"[retry-control] run generation time {elapsed_run:.1f}s "
                          f"> max {config.max_total_generation_time_sec}s, stopping", flush=True)
                    break

                # Check LLM retry budget for this run
                if run_llm_retries >= config.max_llm_retries_per_scene:
                    print(f"[retry-control] LLM retries {run_llm_retries} "
                          f">= max {config.max_llm_retries_per_scene}, stopping", flush=True)
                    break

                # Retry-scoped target-conditioned spawn: the first attempt and
                # every recovery retry re-clean the sample, re-select the
                # native target, re-stabilize the robot spawn over the full
                # generation warmup horizon, and re-bind the target. A binding
                # consumed by a failed attempt is never reused, so a
                # recoverable physical failure cannot degrade into
                # no_officially_transitionable_native_target.
                spawn_failed = False
                if task_sequence and current_task:
                    # Sample transaction boundary: remove prior generated objects
                    # before selecting a target or stabilizing Tiago. The generator
                    # consumes this prepared state and never resets the robot after
                    # its spawn has been validated.
                    engine.begin_env_a_attempt()
                    preferred_target = engine.prepare_native_task_robot_spawn(current_task)
                    if not preferred_target:
                        # retrieval/delivery have no official native appliance;
                        # recover a vetted source support surface so the robot
                        # spawns within the 1.35 m bind gate instead of an
                        # unconditional pose far from every surface.
                        preferred_target = engine.prepare_retrieval_delivery_robot_spawn(current_task)
                    if preferred_target:
                        print(
                            f"[robot-spawn] preparing task={current_task} "
                            f"target={preferred_target} attempt={attempt + 1}",
                            flush=True,
                        )
                        try:
                            stabilize_robot_spawn(
                                env,
                                seed=(args.seed or 0) + idx + 1 + attempt,
                                # Validate each candidate for at least the same
                                # effective warmup horizon generation applies
                                # after binding; the old 10-step window missed
                                # the 0.613 m robot drift that the later
                                # generation warmup exposed.
                                warmup_steps=max(args.warmup_steps, 10),
                                preferred_target_name=preferred_target,
                                preferred_max_distance=DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
                                settle_scene=False,
                            )
                        except RobotSpawnError as exc:
                            print(
                                f"[robot-spawn] target-conditioned spawn exhausted "
                                f"task={current_task} attempt={attempt + 1} "
                                f"reason={getattr(exc, 'reason', repr(exc))}",
                                flush=True,
                            )
                            spawn_failed = True
                        if not spawn_failed and not engine.bind_prepared_native_task_spawn(current_task):
                            print(
                                f"[robot-spawn] target-conditioned spawn could not bind "
                                f"task={current_task} attempt={attempt + 1}",
                                flush=True,
                            )
                            engine.reject_native_target(
                                preferred_target, "robot_spawn_binding"
                            )
                            spawn_failed = True
                        if not spawn_failed:
                            engine.invalidate_robot_reachability()
                    else:
                        # No official native target in the scene: either the task
                        # is non-native (retrieval/delivery) or it is a native
                        # appliance task whose target is *generated*
                        # (appliance-poor scene). Neither path re-uprights the
                        # robot after a prior task, so a tilt introduced by an
                        # object-placement collision in one task cascades into a
                        # robot-stability rejection in every later task (observed
                        # tilt up to 0.67 across the serial 23-task run). A fresh
                        # unconditional spawn each attempt breaks that cascade
                        # without disturbing any target: the robot is re-sampled,
                        # upright, and warmup-validated on the current scene.
                        try:
                            stabilize_robot_spawn(
                                env,
                                seed=(args.seed or 0) + idx + 1 + attempt,
                                warmup_steps=max(args.warmup_steps, 10),
                                settle_scene=False,
                            )
                        except RobotSpawnError as exc:
                            print(
                                f"[robot-spawn] unconditional re-stabilize exhausted "
                                f"task={current_task} attempt={attempt + 1} "
                                f"reason={getattr(exc, 'reason', repr(exc))}",
                                flush=True,
                            )
                            spawn_failed = True
                        if not spawn_failed:
                            engine.invalidate_robot_reachability()
                if spawn_failed:
                    attempt += 1
                    limit = args.max_retries
                    if limit > 0 and attempt > limit:
                        print(
                            f"[robot-spawn] max retries ({limit}) reached for "
                            f"task={current_task}",
                            flush=True,
                        )
                        break
                    print(
                        f"[robot-spawn] retry {attempt}"
                        f"{f'/{limit}' if limit > 0 else ''} task={current_task}",
                        flush=True,
                    )
                    continue

                if args.env_type == "A":
                    run = engine.generate_env_a(
                        task=current_task,
                        target_room=args.target_room,
                        skip_tasks=run_skip_tasks,
                    )
                elif args.env_type == "B":
                    run = engine.generate_env_b_fire(target_room=args.target_room)
                else:
                    env_c_types = None
                    if args.env_c_types:
                        env_c_types = [item.strip() for item in args.env_c_types.split(",") if item.strip()]
                    run = engine.generate_env_c(
                        target_room=args.target_room,
                        env_c_types=env_c_types,
                        skip_tasks=run_skip_tasks,
                    )

                integrity = enforce_run_quality(run)
                if not integrity["ok"]:
                    print(f"[quality] rejected scene integrity failure: "
                          f"missing={len(integrity['missing_source_objects'])} "
                          f"moved={len(integrity['moved_source_objects'])} "
                          f"missing_objects={integrity['missing_source_objects']} "
                          f"moved_objects={integrity['moved_source_objects']}", flush=True)
                expert_preflight = enforce_expert_plan_quality(run)
                if expert_preflight is not None and not expert_preflight["ok"]:
                    print(
                        f"[quality] rejected invalid expert plan: {expert_preflight['error']}",
                        flush=True,
                    )
                if run.get("ok"):
                    fingerprint, _ = sample_fingerprint(run)
                    if fingerprint in fingerprints:
                        run["ok"] = False
                        run.setdefault("validation", {})["ok"] = False
                        run["validation"]["duplicate_sample"] = True
                        te_validation = ((run.get("task_environment") or {}).setdefault("validation", {}))
                        te_validation["ok"] = False
                        te_validation["duplicate_sample"] = True
                        print("[quality] rejected exact duplicate sample", flush=True)
                    else:
                        run.setdefault("validation", {})["sample_fingerprint"] = fingerprint
                        ((run.get("task_environment") or {}).setdefault("validation", {}))["sample_fingerprint"] = fingerprint
                        diversity = sample_diversity_record(run)
                        run["diversity"] = diversity
                        (run.get("task_environment") or {})["diversity"] = diversity

                # Retry conditions:
                # 1. LLM rejected the task setup (not feasible / not linear)
                # 2. All task_objects failed to place (0 valid data)
                validation = run.get("validation", {})
                llm_rejected = validation.get("llm_rejected", False)
                hard_reject = run.get("hard_reject", False)
                no_task_objects = (
                    not run.get("ok", False) and not llm_rejected
                )
                should_retry = llm_rejected or no_task_objects
                camera_failed = not (validation.get("camera_coverage") or {}).get("ok", True)

                # Hard rejects should not retry the same task, but the sample slot
                # can still be recovered by asking the LLM for a different task.
                if should_retry and hard_reject:
                    rejected_task = (run.get("task") or {}).get("primary_behavior_task", "")
                    if rejected_task and rejected_task != "unknown":
                        run_skip_tasks.add(rejected_task)
                        if not args.allow_repeat_tasks:
                            skip_tasks.add(rejected_task)
                    print(f"[hard-reject] task is fundamentally unsuitable, "
                          f"skipping it and selecting another task", flush=True)

                # Check if we should stop (success, retry limit reached, or a
                # fatal physics view that in-process retries cannot heal)
                if run.get("fatal_physics_error"):
                    print("[fatal-physics] invalid physics view during placement; "
                          "stopping retries in this scene process", flush=True)
                    break
                if should_retry and hard_reject and current_task:
                    print(
                        "[hard-reject] explicit task cannot change in this process; "
                        "returning control to the outer runner",
                        flush=True,
                    )
                    break
                if should_retry and camera_failed and args.target_native_object_id:
                    print(
                        "[camera-coverage] explicit native target cannot change in this "
                        "process; returning control to the outer candidate runner",
                        flush=True,
                    )
                    break
                if not should_retry:
                    break
                if args.single_attempt:
                    print(
                        "[retry-control] single-attempt mode: returning failure "
                        "to the clean-process runner",
                        flush=True,
                    )
                    break

                run_llm_retries += 1
                attempt += 1
                limit = args.max_retries
                if limit > 0 and attempt > limit:
                    print(f"[online-deltasg] max retries ({limit}) reached, giving up", flush=True)
                    break

                rejected_task = (run.get("task") or {}).get("primary_behavior_task", "")
                if rejected_task and rejected_task != "unknown":
                    if hard_reject or not args.allow_repeat_tasks:
                        skip_tasks.add(rejected_task)
                    # Track per-task retry count
                    task_retry_count[rejected_task] = task_retry_count.get(rejected_task, 0) + 1
                    if task_retry_count[rejected_task] > config.max_retries_per_task:
                        run_skip_tasks.add(rejected_task)
                        print(f"[retry-control] task '{rejected_task}' retried "
                              f"{task_retry_count[rejected_task]}x > max {config.max_retries_per_task}, "
                              f"permanently skipping", flush=True)
                if camera_failed and args.env_type == "A":
                    engine.reject_native_target(
                        interaction_target(run), "initial_camera_coverage",
                    )
                if llm_rejected:
                    reason = "LLM rejected"
                elif camera_failed:
                    reason = "initial camera coverage failed"
                else:
                    reason = "task generation or placement failed"
                limit_str = f"/{limit}" if limit > 0 else ""
                print(f"[retry-control] retry {attempt}{limit_str} (LLM: {run_llm_retries}/{config.max_llm_retries_per_scene}): "
                      f"{reason}, task={rejected_task}, run skip set: {len(run_skip_tasks)}", flush=True)
                if len(run_skip_tasks) >= 50:
                    print(f"[online-deltasg] WARNING: 50+ tasks skipped, "
                          f"scene may not support enough linear tasks", flush=True)

            # Handle case where retry budget exhausted without a run
            if run is None:
                print(f"[online-deltasg] run {idx + 1} failed: retry budget exhausted", flush=True)
                summary["ok"] = False
                continue

            run_path = Path(args.output_dir) / f"{run['run_id']}.json"
            engine.save_run(run, run_path)
            # Track successfully generated tasks to prevent duplicates (even if failed)
            task_name = (run.get("task") or {}).get("primary_behavior_task", "")
            if task_name and task_name != "unknown" and not args.allow_repeat_tasks:
                skip_tasks.add(task_name)
                print(f"[online-deltasg] added '{task_name}' to skip_tasks (now {len(skip_tasks)} total)", flush=True)
            # Only include successful runs in dataset
            if run.get("ok"):
                runs.append(run)
                fingerprints.add(sample_fingerprint(run)[0])
                diversity = run.get("diversity") or sample_diversity_record(run)
                engine._checkpoint["successful_samples"].append({
                    "run_id": run["run_id"],
                    "task": task_name,
                    "diversity": diversity,
                })
                for item in diversity.get("target_models") or []:
                    if item.get("category") and item.get("model"):
                        engine._used_target_models[(item["category"], item["model"])] += 1
            else:
                print(f"[online-deltasg] run {idx + 1} excluded from dataset (ok=False)", flush=True)

            item = summary_item(run, run_path)

            # Save checkpoint at interval
            if args.checkpoint_interval > 0 and (idx + 1) % args.checkpoint_interval == 0:
                engine.save_checkpoint(args.output_dir)
            # Only include OK runs in summary
            if run.get("ok"):
                summary["runs"].append(item)
                summary["ok"] = summary["ok"] and run["ok"]
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

            if run.get("fatal_physics_error"):
                summary["aborted_reason"] = "fatal_physics_error"
                print(f"[fatal-physics] aborting remaining samples in this process "
                      f"({args.num_envs - idx - 1} skipped); a clean process is required",
                      flush=True)
                break

        # A requested sample slot is successful only if it produced an
        # accepted dataset entry. Outer coverage runners use this status to
        # start a fresh OG process instead of reusing contaminated physics.
        if len(runs) - existing_run_count != args.num_envs:
            summary["ok"] = False
            summary["num_generated"] = len(runs)

        # Final checkpoint save
        engine.save_checkpoint(args.output_dir)

        summary_path = Path(args.output_dir) / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        index_path, dataset_path = build_dataset_outputs(args.output_dir, summary, runs)
        print(f"[online-deltasg] saved {summary_path}", flush=True)
        print(f"[online-deltasg] saved {index_path}", flush=True)
        print(f"[online-deltasg] saved {dataset_path}", flush=True)
        hard_exit(0 if summary["ok"] else 2)
    except Exception as exc:
        error = {
            "ok": False,
            "error": repr(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        if hasattr(exc, "reason"):
            error["reason"] = exc.reason
        if hasattr(exc, "detail"):
            error["detail"] = exc.detail
        error_path = Path(args.output_dir) / "error.json"
        with error_path.open("w", encoding="utf-8") as f:
            json.dump(error, f, ensure_ascii=False, indent=2)
        print(json.dumps(error, ensure_ascii=False, indent=2), flush=True)
        hard_exit(1)
    finally:
        if env is not None:
            try:
                og.clear()
            except Exception:
                pass


if __name__ == "__main__":
    main()
