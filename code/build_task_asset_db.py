"""
Export the layer-2 task asset metadata database.

The output is a scene-editing catalog used by DeltaSG / task generation. It
normalizes BEHAVIOR-1K metadata into records that can answer questions such as:
which task-relevant assets exist, which categories instantiate them, which
rooms and paths are available in the current 3DSG, and which objects are good
candidates for placement, interaction, or abnormal-state generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ASSET_DB_PATH = REPO_ROOT / "OmniGibson" / "omnigibson" / "scene_graphs" / "task_asset_database.py"


def load_task_asset_database_class():
    spec = importlib.util.spec_from_file_location("task_asset_database", TASK_ASSET_DB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskAssetDatabase


def build_database(metadata_dir, scene_graph_path=None):
    task_asset_database = load_task_asset_database_class().load(metadata_dir=metadata_dir)
    data = database_to_dict(task_asset_database)
    data["schema"] = {
        "record_key": "synset",
        "purpose": "task_asset_catalog_for_3dsg_scene_editing",
        "core_fields": [
            "synset",
            "state",
            "definition",
            "tasks",
            "direct_categories",
            "properties",
            "object_type",
            "mass_estimates",
            "sampled_placements",
            "edit_metadata",
        ],
        "edit_metadata": {
            "receptacle": "whether the asset can support placement on top / inside",
            "interaction": "none | manipulable | articulable | controllable",
            "abnormal": "potential abnormal states supported by metadata",
        },
    }

    if scene_graph_path:
        scene_graph = _load_json(scene_graph_path)
        data["scene_overlay"] = _build_scene_overlay(scene_graph, task_asset_database)

    return data


def database_to_dict(task_asset_database):
    if hasattr(task_asset_database, "to_dict"):
        return task_asset_database.to_dict()

    return {
        "source": "behavior_1k_asset_metadata",
        "metadata_dir": str(task_asset_database.metadata_dir),
        "num_records": len(task_asset_database.records),
        "num_categories": len(task_asset_database.categories),
        "num_tasks": len(task_asset_database.tasks),
        "records": list(task_asset_database.records.values()),
        "indices": {
            "categories": {
                category: [record["synset"] for record in records]
                for category, records in task_asset_database.categories.items()
            },
            "tasks": {
                task: [record["synset"] for record in records]
                for task, records in task_asset_database.tasks.items()
            },
        },
    }


def _build_scene_overlay(scene_graph, task_asset_database):
    object_nodes = [node for node in scene_graph.get("nodes", []) if node.get("type") == "object"]
    room_nodes = [node for node in scene_graph.get("nodes", []) if node.get("type") == "room"]
    category_counts = Counter(node.get("category") for node in object_nodes if node.get("category"))

    category_records = {}
    for category in sorted(category_counts):
        records = task_asset_database.by_category(category)
        category_records[category] = [record["synset"] for record in records]

    return {
        "scene_model": scene_graph.get("scene_model"),
        "rooms": [node["name"] for node in room_nodes],
        "category_counts": dict(category_counts),
        "category_to_synsets": category_records,
        "navigation": scene_graph.get("navigation", {}),
        "object_instances": [
            {
                "object_id": node.get("id"),
                "category": node.get("category"),
                "rooms": node.get("rooms", []),
                "pose": node.get("pose"),
                "semantic": node.get("semantic"),
            }
            for node in object_nodes
        ],
    }


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build task asset metadata database JSON.")
    parser.add_argument(
        "--metadata-dir",
        default=str(REPO_ROOT / "asset_pipeline" / "metadata"),
        help="Path to asset_pipeline/metadata",
    )
    parser.add_argument(
        "--scene-graph",
        default=None,
        help="Optional level2_3dsg.json. If provided, adds scene categories, rooms, and room paths.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "code" / "outputs" / "task_asset_database.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    data = build_database(args.metadata_dir, args.scene_graph)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"saved {output_path} with {data['num_records']} synset records, "
        f"{data['num_categories']} categories, {data['num_tasks']} tasks"
    )
    if "scene_overlay" in data:
        overlay = data["scene_overlay"]
        print(
            f"attached scene overlay for {overlay.get('scene_model')} with "
            f"{len(overlay.get('rooms', []))} rooms and {len(overlay.get('object_instances', []))} objects"
        )


if __name__ == "__main__":
    main()
