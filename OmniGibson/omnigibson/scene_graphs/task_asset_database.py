"""
Task asset metadata database for graph-driven scene editing.

This module normalizes the scattered BEHAVIOR-1K metadata used by layer 2:
task-required synsets, synset abilities/properties, category mappings, masses,
and optional sampled task-object placements. It does not launch OmniGibson and
can be used by planning / DeltaSG code before physics validation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_METADATA_DIR = REPO_ROOT / "asset_pipeline" / "metadata"


PROPERTY_COLUMNS = (
    "breakable",
    "fillable",
    "flammable",
    "openable",
    "toggleable",
    "cookable",
    "heatSource",
    "coldSource",
    "sliceable",
    "diceable",
    "slicer",
    "assembleable",
    "meltable",
    "particleRemover",
    "particleApplier",
    "particleSource",
    "particleSink",
    "sceneObject",
    "waterCook",
    "mixingTool",
)


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _split_cell(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TaskAssetDatabase:
    """
    Normalized view over task-relevant BEHAVIOR-1K assets.

    Records are keyed by synset and include direct categories, direct objects,
    tasks, abilities/properties, object type, mass estimate, and sampled
    placements when available.
    """

    def __init__(self, metadata_dir=ASSET_METADATA_DIR):
        self.metadata_dir = Path(metadata_dir)
        self.records = {}
        self.categories = defaultdict(list)
        self.tasks = defaultdict(list)

    @classmethod
    def load(cls, metadata_dir=ASSET_METADATA_DIR):
        db = cls(metadata_dir=metadata_dir)
        db._load()
        return db

    def _load(self):
        required_synsets = _read_csv(self.metadata_dir / "task_required_synsets.csv")
        synset_properties = {
            row["synset"]: row for row in _read_csv(self.metadata_dir / "synset_property.csv") if row.get("synset")
        }
        category_mapping = _read_csv(self.metadata_dir / "category_mapping.csv")
        masses = _load_json(self.metadata_dir / "gpt4_masses.json", {})
        sampled_infos = _load_json(self.metadata_dir / "task_object_infos.json", {})

        categories_by_synset = defaultdict(list)
        mass_by_category = {}
        for row in category_mapping:
            category = row.get("category")
            synset = row.get("synset")
            if category and synset:
                categories_by_synset[synset].append(category)
                mass_by_category[category] = _coerce_float(row.get("mass (auto)") or row.get("mass estimate ig2"))

        all_categories = sorted(mass_by_category, key=len, reverse=True)
        sampled_by_category = defaultdict(list)
        for env_task, objects in sampled_infos.items():
            for object_name, placement in objects.items():
                category = _infer_category_from_object_name(object_name, all_categories)
                sampled_by_category[category].append({"env_task": env_task, "object_name": object_name, **placement})

        for row in required_synsets:
            synset = row["Name"]
            prop_row = synset_properties.get(synset, {})
            direct_categories = _split_cell(row.get("Direct Categories"))
            if not direct_categories:
                direct_categories = sorted(categories_by_synset.get(synset, []))
            direct_objects = _split_cell(row.get("Direct Objects"))
            tasks = _split_cell(row.get("Tasks"))
            properties = {name: _truthy(prop_row.get(name, False)) for name in PROPERTY_COLUMNS}
            categories = sorted(set(direct_categories))

            record = {
                "synset": synset,
                "state": row.get("Synset State"),
                "definition": row.get("Definition"),
                "parents": _split_cell(row.get("Parents")),
                "children": _split_cell(row.get("Children")),
                "tasks": tasks,
                "direct_categories": categories,
                "direct_objects": direct_objects,
                "properties": properties,
                "object_type": prop_row.get("objectType"),
                "mass_estimates": {
                    category: masses.get(category, mass_by_category.get(category)) for category in categories
                },
                "sampled_placements": {
                    category: sampled_by_category.get(category, [])[:20] for category in categories
                },
            }
            record["edit_metadata"] = self._infer_edit_metadata(record)
            self.records[synset] = record
            for category in categories:
                self.categories[category].append(record)
            for task in tasks:
                self.tasks[task].append(record)

    def _infer_edit_metadata(self, record):
        properties = record["properties"]
        categories = set(record["direct_categories"])
        category_tokens = set()
        for category in categories:
            category_tokens.update(category.lower().split("_"))

        container_tokens = {"basket", "bin", "bowl", "box", "cabinet", "drawer", "jar", "pot", "sink"}
        supports_inside = any(properties[name] for name in ("fillable", "openable")) or bool(
            category_tokens & container_tokens
        )
        supports_on_top = bool(
            category_tokens
            & {"bed", "bench", "cabinet", "counter", "desk", "floor", "plate", "shelf", "sofa", "table", "tray"}
        )

        if any(
            properties[name]
            for name in (
                "toggleable",
                "heatSource",
                "coldSource",
                "particleApplier",
                "particleRemover",
                "particleSource",
            )
        ):
            interaction = "controllable"
        elif properties["openable"]:
            interaction = "articulable"
        elif properties["sceneObject"]:
            interaction = "none"
        else:
            interaction = "manipulable"

        abnormal = []
        if properties["breakable"]:
            abnormal.append("broken")
        if properties["flammable"]:
            abnormal.append("on_fire")
        if properties["cookable"]:
            abnormal.append("burnt")

        return {
            "receptacle": {
                "can_support": supports_on_top or supports_inside,
                "supports_on_top": supports_on_top,
                "supports_inside": supports_inside,
                "confidence": "metadata_inferred",
            },
            "interaction": {"kind": interaction, "confidence": "metadata_inferred"},
            "abnormal": {"potential": abnormal, "current": [], "confidence": "metadata_inferred"},
        }

    def by_synset(self, synset):
        return self.records[synset]

    def by_category(self, category):
        return list(self.categories.get(category, []))

    def by_task(self, task):
        return list(self.tasks.get(task, []))

    def sampleable_records(self, require_interaction=None, require_receptacle=False):
        records = list(self.records.values())
        if require_interaction is not None:
            records = [
                record
                for record in records
                if record["edit_metadata"]["interaction"]["kind"] == require_interaction
            ]
        if require_receptacle:
            records = [record for record in records if record["edit_metadata"]["receptacle"]["can_support"]]
        return records

    def to_dict(self):
        return {
            "source": "behavior_1k_asset_metadata",
            "metadata_dir": str(self.metadata_dir),
            "num_records": len(self.records),
            "num_categories": len(self.categories),
            "num_tasks": len(self.tasks),
            "records": list(self.records.values()),
            "indices": {
                "categories": {category: [record["synset"] for record in records] for category, records in self.categories.items()},
                "tasks": {task: [record["synset"] for record in records] for task, records in self.tasks.items()},
            },
        }

    def export_json(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path


def _coerce_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _infer_category_from_object_name(object_name, categories):
    for category in categories:
        if object_name == category or object_name.startswith(f"{category}_"):
            return category
    return object_name.rsplit("_", 1)[0]
