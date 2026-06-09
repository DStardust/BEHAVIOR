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
        json.dump(dataset_index, f, ensure_ascii=False, indent=2)
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
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
    parser.add_argument("--task-objects", type=int, default=5)
    parser.add_argument("--context-objects", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--settle-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
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
        for idx in range(args.num_envs):
            print(f"[online-deltasg] run {idx + 1}/{args.num_envs}", flush=True)
            if args.env_type == "A":
                run = engine.generate_env_a(task=args.task, target_room=args.target_room)
            elif args.env_type == "B":
                run = engine.generate_env_b_fire(target_room=args.target_room)
            else:
                run = engine.generate_env_c_fire_disambiguation(target_room=args.target_room)
            run_path = Path(args.output_dir) / f"{run['run_id']}.json"
            engine.save_run(run, run_path)
            runs.append(run)
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
            summary["runs"].append(item)
            summary["ok"] = summary["ok"] and run["ok"]
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

        summary_path = Path(args.output_dir) / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
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
