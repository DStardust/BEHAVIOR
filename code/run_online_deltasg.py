"""
Run the online DeltaSG pipeline on a live OmniGibson scene.

This script is intended to be launched on the remote server where OmniGibson is
installed, for example:

conda run -n behavior python code/run_online_deltasg.py --scene Rs_int --robot fetch --num-envs 1
"""

from __future__ import annotations

import argparse
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
gm.RENDER_VIEWER_CAMERA = False
gm.GUI_VIEWPORT_ONLY = True

import omnigibson as og
import omnigibson.lazy as lazy

from api import create_env, ensure_dir
from online_deltasg import OnlineDeltaSGConfig, OnlineDeltaSGEngine


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
        help="DeltaSG environment type: A=basic task, B=fire anomaly, C=fire disambiguation.",
    )
    parser.add_argument("--task", default=None, help="Optional BEHAVIOR task name for Env-A, e.g. cook_eggplant-0")
    parser.add_argument("--target-room", default=None, help="Optional room id, e.g. kitchen_0")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of online DeltaSG edits to apply sequentially.")
    parser.add_argument("--task-objects", type=int, default=2)
    parser.add_argument("--context-objects", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--settle-threshold", type=float, default=0.25)
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
        default=None,
        help="LLM model name (default: qwen-plus via DASHSCOPE_API_KEY env var).",
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
        "--no-abort-on-task-failure",
        action="store_true",
        help="Continue placing context objects even if task objects fail.",
    )
    parser.add_argument(
        "--no-skip-context-on-failure",
        action="store_true",
        help="Continue placing context objects even after task abort.",
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

    robot_model = None if str(args.robot).lower() in {"none", "null", ""} else args.robot
    ensure_dir(args.output_dir)
    env = None
    try:
        with gm.unlocked():
            gm.ENABLE_TRANSITION_RULES = args.enable_transition_rules
        env = create_env(scene_model=args.scene, robot_model=robot_model)
        env.reset()
        harden_headless_kit()
        for _ in range(30):
            og.sim.step()

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
            abort_on_task_object_failure=not args.no_abort_on_task_failure,
            skip_context_on_failure=not args.no_skip_context_on_failure,
        )
        engine = OnlineDeltaSGEngine(env=env, metadata_dir=args.metadata_dir, config=config)

        summary = {
            "ok": True,
            "scene": args.scene,
            "robot": robot_model,
            "num_envs": args.num_envs,
            "runs": [],
        }
        runs = []
        skip_tasks = set()
        # Resume from checkpoint if requested
        if args.resume:
            skip_tasks = engine.load_checkpoint(args.output_dir)

        for idx in range(args.num_envs):
            print(f"[online-deltasg] run {idx + 1}/{args.num_envs}", flush=True)
            run_start_time = time.time()
            run_llm_retries = 0
            run = None
            attempt = 0
            task_retry_count = {}  # task_name -> retry count

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

                if args.env_type == "A":
                    run = engine.generate_env_a(
                        task=args.task, target_room=args.target_room, skip_tasks=skip_tasks,
                    )
                elif args.env_type == "B":
                    run = engine.generate_env_b_fire(target_room=args.target_room)
                else:
                    run = engine.generate_env_c_fire_disambiguation(target_room=args.target_room)

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

                # Don't retry hard rejects
                if should_retry and hard_reject:
                    print(f"[hard-reject] task is fundamentally unsuitable, "
                          f"not retrying the same task", flush=True)
                    should_retry = False

                # Check if we should stop (success or retry limit reached)
                if not should_retry:
                    break  # success or hard reject

                run_llm_retries += 1
                attempt += 1
                limit = args.max_retries
                if limit > 0 and attempt > limit:
                    print(f"[online-deltasg] max retries ({limit}) reached, giving up", flush=True)
                    break

                rejected_task = (run.get("task") or {}).get("primary_behavior_task", "")
                if rejected_task:
                    skip_tasks.add(rejected_task)
                    # Track per-task retry count
                    task_retry_count[rejected_task] = task_retry_count.get(rejected_task, 0) + 1
                    if task_retry_count[rejected_task] > config.max_retries_per_task:
                        print(f"[retry-control] task '{rejected_task}' retried "
                              f"{task_retry_count[rejected_task]}x > max {config.max_retries_per_task}, "
                              f"permanently skipping", flush=True)
                reason = "LLM rejected" if llm_rejected else "all task objects failed to place"
                limit_str = f"/{limit}" if limit > 0 else ""
                print(f"[retry-control] retry {attempt}{limit_str} (LLM: {run_llm_retries}/{config.max_llm_retries_per_scene}): "
                      f"{reason}, task={rejected_task}, skip set: {len(skip_tasks)}", flush=True)
                if len(skip_tasks) >= 50:
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
            if task_name:
                skip_tasks.add(task_name)
                print(f"[online-deltasg] added '{task_name}' to skip_tasks (now {len(skip_tasks)} total)", flush=True)
            # Only include successful runs in dataset
            if run.get("ok"):
                runs.append(run)
                engine._checkpoint["successful_samples"].append({
                    "run_id": run["run_id"],
                    "task": task_name,
                })
            else:
                print(f"[online-deltasg] run {idx + 1} excluded from dataset (ok=False)", flush=True)

            validation = run.get("validation", {})
            created = validation.get("created_objects")
            failed = validation.get("failed_objects")
            item = {
                "run_id": run["run_id"],
                "ok": run["ok"],
                "task": run["task_instance"]["instruction"],
                "target_room": run["task_instance"]["target_room"],
                "created": len(created) if isinstance(created, list) else None,
                "failed": len(failed) if isinstance(failed, list) else None,
                "path": str(run_path),
            }

            # Save checkpoint at interval
            if args.checkpoint_interval > 0 and (idx + 1) % args.checkpoint_interval == 0:
                engine.save_checkpoint(args.output_dir)
            # Only include OK runs in summary
            if run.get("ok"):
                summary["runs"].append(item)
                summary["ok"] = summary["ok"] and run["ok"]
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

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
        error = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
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
