"""Regenerate generate_dataset/0907/qra.jsonl for the 10-instance subset.

Uses the DashScope key from ../jd_api, runs the full pipeline (translate →
attach_mcq with LLM option phrasing → LLM reasoning → batch export with images).
"""
from __future__ import annotations

import logging
import os
import sys

from llm_client import create_llm_client
from translator import discover_task_instances, generate_batch_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_ROOT = "/home2/jiaodian/BEHAVIOR-main/task_instance/data_0901"
OUTPUT_DIR = "/home2/jiaodian/BEHAVIOR-main/generate_dataset/0907"
API_KEY_PATH = "/home2/jiaodian/BEHAVIOR-main/jd_api"

TARGETS = {
    ("Beechwood_0_int", "online_env_b_fire_0001"),
    ("Beechwood_0_int", "online_env_c_fire_disambiguation_0017"),
    ("Beechwood_0_int", "online_env_c_fire_disambiguation_0021"),
    ("Beechwood_0_int", "online_env_c_open_close_0013"),
    ("Beechwood_1_int", "online_env_b_fire_0005"),
    ("Ihlen_0_int", "online_env_b_fire_0002"),
    ("Pomaria_0_int", "online_env_b_fire_0004"),
    ("Pomaria_0_int", "online_env_c_appliance_0025"),
    ("Pomaria_0_int", "online_env_c_open_close_0005"),
    ("Pomaria_2_int", "online_env_b_fire_0001"),
}


def main() -> int:
    key = open(API_KEY_PATH, encoding="utf-8").read().strip()
    client = create_llm_client(api_key=key)
    if client is None:
        print("ERROR: LLM client unavailable", file=sys.stderr)
        return 1

    instances = [i for i in discover_task_instances(DATA_ROOT)
                 if (i["scene"], i["sample_id"]) in TARGETS]
    instances.sort(key=lambda i: (i["scene"], i["sample_id"]))
    print(f"regenerating {len(instances)} instances -> {OUTPUT_DIR}")

    out = generate_batch_dataset(instances, OUTPUT_DIR, llm=client)
    print("done:", out)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
