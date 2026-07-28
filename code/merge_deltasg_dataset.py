"""
Merge per-environment online DeltaSG JSON files into dataset-level files.

This script does not launch OmniGibson. It is useful after copying or generating
many `online_env_*.json` files in an output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def merge_deltasg_dataset(input_dir, pattern="online_env_*.json", output_index=None, output_dataset=None):
    input_dir = Path(input_dir)
    files = sorted(path for path in input_dir.glob(pattern) if path.name not in {"dataset.json", "dataset_index.json"})
    items = []
    task_environments = []

    for path in files:
        run = load_json(path)
        task_environment = run.get("task_environment")
        if task_environment is not None:
            task_environments.append(task_environment)

        task = (task_environment or {}).get("task") or run.get("task_instance", {})
        validation = (task_environment or {}).get("validation") or run.get("validation", {})
        items.append(
            {
                "env_id": run.get("run_id") or (task_environment or {}).get("env_id"),
                "env_type": (task_environment or {}).get("env_type"),
                "ok": run.get("ok", validation.get("ok", False)),
                "path": str(path),
                "task_id": task.get("task_id"),
                "instruction": task.get("instruction"),
                "target_room": task.get("target_room"),
                "base_scene": ((task_environment or {}).get("base_scene") or {}).get("scene_model"),
                "num_added_objects": len((task_environment or {}).get("added_objects", [])),
                "num_task_objects": len((task_environment or {}).get("task_objects", [])),
                "num_solution_steps": len((task_environment or {}).get("solution_plan", [])),
                "validation_ok": validation.get("ok"),
            }
        )

    scene = next((item["base_scene"] for item in items if item.get("base_scene")), None)
    dataset_index = {
        "schema_version": "deltasg_dataset_index.v1",
        "source_dir": str(input_dir),
        "scene": scene,
        "num_environments": len(items),
        "num_ok": sum(1 for item in items if item["ok"]),
        "items": items,
    }
    dataset = {
        "schema_version": "deltasg_dataset.v1",
        "source_dir": str(input_dir),
        "scene": scene,
        "num_environments": len(task_environments),
        "task_environments": task_environments,
    }

    output_index = Path(output_index) if output_index else input_dir / "dataset_index.json"
    output_dataset = Path(output_dataset) if output_dataset else input_dir / "dataset.json"
    write_json(output_index, dataset_index)
    write_json(output_dataset, dataset)
    return output_index, output_dataset, dataset_index


def main():
    parser = argparse.ArgumentParser(description="Merge online DeltaSG per-env JSON files into dataset files.")
    parser.add_argument("--input-dir", default="code/outputs/deltasg")
    parser.add_argument("--pattern", default="online_env_*.json")
    parser.add_argument("--output-index", default=None)
    parser.add_argument("--output-dataset", default=None)
    args = parser.parse_args()

    output_index, output_dataset, dataset_index = merge_deltasg_dataset(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_index=args.output_index,
        output_dataset=args.output_dataset,
    )
    print(f"saved {output_index}")
    print(f"saved {output_dataset}")
    print(json.dumps({"num_environments": dataset_index["num_environments"], "num_ok": dataset_index["num_ok"]}, indent=2))


if __name__ == "__main__":
    main()
