#!/usr/bin/env python3
"""Rebuild versioned Env-A native target eligibility from saved scene graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_deltasg_coverage import eligible_native_task_pairs
from audit_deltasg_outputs import iter_run_files, load_json


def _scene_model(run):
    te = run.get("task_environment") or {}
    return (
        (te.get("base_scene") or {}).get("scene_model")
        or (run.get("base_scene") or {}).get("scene_model")
    )


def build_native_eligibility(scenes, runs, min_height=0.10, max_height=1.55):
    best_graphs = {}
    for path, run in runs:
        scene = _scene_model(run)
        graph = run.get("before_graph") or (run.get("debug") or {}).get("before_graph") or {}
        if scene not in scenes or not graph.get("nodes"):
            continue
        score = (
            sum(node.get("type") == "object" for node in graph["nodes"]),
            sum(bool(node.get("available_states")) for node in graph["nodes"]),
        )
        if scene not in best_graphs or score > best_graphs[scene][0]:
            best_graphs[scene] = (score, path, graph)

    missing = sorted(set(scenes) - set(best_graphs))
    if missing:
        raise ValueError(f"no saved before_graph for scenes: {missing}")

    payload = {"schema_version": "deltasg_enva_scene_eligibility.v1", "scenes": {}}
    for scene in scenes:
        _, _, graph = best_graphs[scene]
        probe = {"before_graph": graph}
        pairs = set()
        for label in ("envA_open_close", "envA_appliance"):
            pairs.update(
                eligible_native_task_pairs(label, probe, min_height, max_height)
            )
        eligible_tasks = {}
        target_ids = set()
        for task, object_id in sorted(pairs):
            eligible_tasks.setdefault(task, []).append(object_id)
            target_ids.add(object_id)
        has_open_close = any(task.startswith(("open_", "close_")) for task in eligible_tasks)
        has_appliance = any(task.startswith(("turn_on_", "turn_off_")) for task in eligible_tasks)
        if not has_open_close or not has_appliance:
            raise ValueError(
                f"scene {scene!r} lacks an eligible native task family: "
                f"open_close={has_open_close} appliance={has_appliance}"
            )
        node_index = {
            node.get("id"): node
            for node in graph.get("nodes") or []
            if node.get("id") in target_ids
        }
        payload["scenes"][scene] = {
            "graph_nodes": len(graph.get("nodes") or []),
            "eligible_tasks": eligible_tasks,
            "eligible_target_objects": {
                object_id: {
                    "category": node_index[object_id].get("category"),
                    "rooms": sorted(node_index[object_id].get("rooms") or []),
                }
                for object_id in sorted(target_ids)
            },
        }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--scenes-file",
        default=str(Path(__file__).resolve().parent / "configs" / "env_a_scenes.txt"),
    )
    parser.add_argument("--min-manipulation-height", type=float, default=0.10)
    parser.add_argument("--max-manipulation-height", type=float, default=1.55)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.min_manipulation_height < 0 or args.max_manipulation_height <= args.min_manipulation_height:
        parser.error("manipulation height bounds must satisfy 0 <= min < max")

    scenes = [
        line.strip()
        for line in Path(args.scenes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = Path(args.input_root)
    runs = []
    for path in iter_run_files(root):
        try:
            runs.append((str(path), load_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
    payload = build_native_eligibility(
        scenes,
        runs,
        args.min_manipulation_height,
        args.max_manipulation_height,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scenes": len(payload["scenes"]), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
