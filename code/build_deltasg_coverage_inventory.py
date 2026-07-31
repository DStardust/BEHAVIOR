#!/usr/bin/env python3
"""Build the installed task-asset model inventory used by coverage audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnigibson.utils.asset_utils import get_all_object_category_models


TASK_ASSET_GROUPS = {
    "retrieval_delivery": [
        "paperback_book",
        "bottle_of_medicine",
        "keys",
        "key_chain",
        "cell_phone",
        "bottle_of_water",
        "water_bottle",
        "canned_food",
    ],
    "fire_common": [
        "fire_extinguisher",
    ],
    "fire_env_c": [
        "bucket",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups = {}
    for group, categories in TASK_ASSET_GROUPS.items():
        groups[group] = {}
        for category in categories:
            models = sorted(get_all_object_category_models(category=category))
            if models:
                groups[group][category] = models

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "deltasg_coverage_inventory.v1",
        "groups": groups,
        "num_categories": sum(len(categories) for categories in groups.values()),
        "num_models": sum(
            len(models)
            for categories in groups.values()
            for models in categories.values()
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
