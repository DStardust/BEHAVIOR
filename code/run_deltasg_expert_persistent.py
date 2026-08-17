"""Execute multiple DeltaSG expert samples in one OmniGibson process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import run_deltasg_expert as expert

og = expert.og


def _entries(args):
    if args.manifest:
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("persistent expert manifest must be a JSON list")
        rows = data
    elif args.input_root:
        input_root = Path(args.input_root).resolve()
        rows = []
        cell_counts = {}
        for input_path in sorted(input_root.rglob("online_env*.json")):
            run = json.loads(input_path.read_text(encoding="utf-8"))
            if run.get("ok") is not True:
                continue
            relative = input_path.relative_to(input_root)
            label = relative.parts[0] if len(relative.parts) > 1 else "unlabeled"
            task = (
                ((run.get("task_environment") or {}).get("task") or run.get("task") or {}).get(
                    "primary_behavior_task"
                )
                or ""
            )
            scene = ((run.get("task_environment") or {}).get("base_scene") or {}).get(
                "scene_model"
            ) or "unknown"
            if not _contains(args.labels, label) or not _contains(args.tasks, task):
                continue
            generated_robot = str((run.get("robot") or {}).get("model") or "")
            if args.robot and generated_robot.casefold() != args.robot.casefold():
                continue
            cell = (label, scene)
            if args.max_per_cell > 0 and cell_counts.get(cell, 0) >= args.max_per_cell:
                continue
            if args.limit > 0 and len(rows) >= args.limit:
                break
            output_dir = Path(args.output_root).resolve() / relative.with_suffix("")
            result_path = output_dir / "expert_result.json"
            if _reusable_result(result_path, input_path, args.backend):
                continue
            rows.append({"input": str(input_path), "output": str(output_dir)})
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    else:
        rows = [
            {
                "input": value,
                "output": str(Path(args.output_root) / Path(value).stem),
            }
            for value in args.input_json
        ]
    result = []
    for row in rows:
        input_path = Path(row["input"]).resolve()
        output_dir = Path(row["output"]).resolve()
        result.append((input_path, output_dir))
    return result


def _contains(requested, value):
    values = set(str(requested).replace(",", " ").split())
    return "all" in values or value in values


def _reusable_result(result_path, input_path, backend):
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        recorded_input = Path(str(result.get("input") or "")).resolve()
    except (OSError, ValueError, TypeError):
        return False
    profile = result.get("backend") or {}
    return (
        result.get("accepted") is True
        and result.get("qa_eligible") is True
        and recorded_input == input_path.resolve()
        and profile.get("name") == backend
        and profile.get("generation_profile_verified") is True
    )


def _sample_identity(run, backend):
    task_environment = run.get("task_environment") or {}
    scene = (task_environment.get("base_scene") or {}).get("scene_model")
    robot = str((run.get("robot") or {}).get("model") or "")
    profile = (task_environment.get("generation") or {}).get("solvability_profile")
    if not scene or not robot:
        raise ValueError("expert input is missing scene or robot identity")
    if profile != backend:
        raise ValueError(
            f"generation solvability profile {profile!r} does not match backend {backend!r}"
        )
    robot = {"tiago": "Tiago", "r1": "R1", "fetch": "fetch"}.get(robot.casefold(), robot)
    return scene, robot


def _has_added_objects(run):
    return bool((run.get("task_environment") or {}).get("added_objects"))


def _preloaded_object_configs(group_rows, backend):
    """Build one parked, collision-safe object set for a same-scene worker."""
    configs = {}
    for _, _, _, _, run in group_rows:
        for config in expert._physical_added_object_configs(run, backend=backend):
            name = config["name"]
            previous = configs.get(name)
            if previous is not None and (
                previous.get("category") != config.get("category")
                or previous.get("model") != config.get("model")
            ):
                raise ValueError(f"inconsistent preloaded object identity {name!r}")
            configs[name] = config
    result = []
    for index, config in enumerate(configs.values()):
        parked = dict(config)
        parked["position"] = [float(index % 16) * 2.0, float(index // 16) * 2.0, -50.0]
        result.append(parked)
    return result


def _write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cleanup_remaining_global_streams():
    streams = []
    for sensor in tuple(expert.VisionSensor.SENSORS.values()):
        if str(getattr(sensor, "name", "")).startswith("deltasg_global_"):
            streams.append(({}, sensor))
    expert.cleanup_persistent_camera_streams({"globals": streams})


def main():
    parser = argparse.ArgumentParser(
        description="Execute same-scene DeltaSG expert samples without repeated Kit startup"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--input-json", action="append")
    source.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--labels", default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--robot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-cell", type=int, default=0)
    parser.add_argument(
        "--backend", choices=["oracle_symbolic", "physical_control"], default="oracle_symbolic"
    )
    parser.add_argument("--llm-model", default="qwen3.8-max")
    parser.add_argument("--primitive-attempts", type=int, default=2)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--min-bbox-pixels", type=int, default=8)
    parser.add_argument("--view-width", type=int, default=640)
    parser.add_argument("--view-height", type=int, default=480)
    parser.add_argument("--max-native-displacement", type=float, default=0.05)
    parser.add_argument("--max-task-object-displacement", type=float, default=0.05)
    parser.add_argument("--min-manipulation-height", type=float, default=0.10)
    parser.add_argument("--max-manipulation-height", type=float, default=1.55)
    args = parser.parse_args()
    if (args.input_json or args.input_root) and not args.output_root:
        parser.error("--output-root is required with --input-json or --input-root")
    rows = []
    for input_path, output_dir in _entries(args):
        run = json.loads(input_path.read_text(encoding="utf-8"))
        scene, robot = _sample_identity(run, args.backend)
        rows.append((scene, robot, input_path, output_dir, run))
    rows.sort(key=lambda row: (row[0], row[1], str(row[2])))

    env = None
    environment_key = None
    environment_loads = 0
    accepted = 0
    started = time.monotonic()
    for sample_index, (scene, robot, input_path, output_dir, run) in enumerate(rows, 1):
        sample_started = time.monotonic()
        key = (scene, robot)
        try:
            fresh_environment = key != environment_key or env is None
            if fresh_environment:
                if env is not None:
                    _cleanup_remaining_global_streams()
                    og.clear()
                print(
                    f"[expert-persistent] loading environment scene={scene} robot={robot}",
                    flush=True,
                )
                group_rows = [row for row in rows if (row[0], row[1]) == key]
                preloaded_configs = _preloaded_object_configs(group_rows, args.backend)
                env = expert._create_expert_env(
                    scene,
                    robot,
                    args.backend,
                    robot_pose=(run.get("task_environment") or {}).get("robot", {}).get("pose"),
                    added_objects=preloaded_configs,
                    camera_resolution=(args.view_width, args.view_height),
                )
                preloaded_names = tuple(config["name"] for config in preloaded_configs)
                environment_loads += 1
            execute_args = SimpleNamespace(**vars(args))
            execute_args.robot = robot
            execute_args.preloaded_delta_names = preloaded_names
            output_dir.mkdir(parents=True, exist_ok=True)
            env, result = expert.execute(
                run,
                input_path,
                output_dir,
                execute_args,
                env=env,
                persistent=True,
            )
            environment_key = key
        except Exception as exc:
            try:
                _cleanup_remaining_global_streams()
            except Exception:
                pass
            result = {
                "schema_version": "deltasg_expert_result.v1",
                "accepted": False,
                "qa_eligible": False,
                "input": str(input_path),
                "llm_model": args.llm_model,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        elapsed = time.monotonic() - sample_started
        result["persistent_worker"] = {
            "enabled": True,
            "sample_index": sample_index,
            "environment_loads": environment_loads,
            "sample_elapsed_seconds": elapsed,
        }
        _write_result(output_dir / "expert_result.json", result)
        if result.get("accepted") is True:
            accepted += 1
        print(
            f"[expert-persistent] sample={sample_index}/{len(rows)} "
            f"accepted={result.get('accepted')} elapsed={elapsed:.1f}s input={input_path}",
            flush=True,
        )

    summary = {
        "total": len(rows),
        "accepted": accepted,
        "failed": len(rows) - accepted,
        "environment_loads": environment_loads,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Individual expert rejections are valid completed outcomes. Coverage and
    # acceptance thresholds are enforced by the batch audit / scene E2E gate;
    # a nonzero worker exit is reserved for an interrupted worker process.
    os._exit(0)


if __name__ == "__main__":
    main()
