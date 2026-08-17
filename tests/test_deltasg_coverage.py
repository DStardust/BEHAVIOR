"""Pure-data regression tests for the DeltaSG coverage gate."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "code" / "audit_deltasg_coverage.py"
sys.path.insert(0, str(ROOT / "code"))

from audit_deltasg_coverage import eligible_native_task_pairs


def sample(scene):
    return {
        "ok": True,
        "task_environment": {
            "env_type": "Env-A",
            "base_scene": {"scene_model": scene},
            "task": {
                "primary_behavior_task": "retrieve_book",
                "instruction": "Retrieve the paperback book from the desk.",
                "target_room": "office_0",
                "plan_objects": [{
                    "object_id": "book_0",
                    "category": "paperback_book",
                }],
            },
            "robot": {"robot_id": "fetch"},
            "camera": [{"camera_id": "robot_camera"}],
            "solution_plan": [{"primitive": "PICK", "target_object": "book_0"}],
            "task_objects": [{"object_id": "book_0"}],
            "added_objects": [{
                "object_id": "book_0",
                "category": "paperback_book",
                "model": "model_a",
                "semantic_roles": ["task_object"],
                "room_id": "office_0",
                "placement": {
                    "support_category": "desk",
                    "robot_approach": {
                        "ok": True,
                        "horizontal_distance": 0.5,
                        "max_horizontal_distance": 1.0,
                    },
                },
            }],
            "validation": {
                "ok": True,
                "scene_integrity": {"ok": True},
                "settling": {"all_within_threshold": True},
                "camera_coverage": {
                    "ok": True,
                    "target_objects": ["book_0"],
                    "visible_objects": ["book_0"],
                },
            },
        },
        "validation": {
            "ok": True,
            "scene_integrity": {"ok": True},
            "settling": {"all_within_threshold": True},
            "camera_coverage": {
                "ok": True,
                "target_objects": ["book_0"],
                "visible_objects": ["book_0"],
            },
            "sample_fingerprint": "unique",
        },
        "diversity": {
            "primary_task": "retrieve_book",
            "target_categories": ["paperback_book"],
            "target_models": [{"category": "paperback_book", "model": "model_a"}],
            "target_object_ids": ["book_0"],
        },
    }


def run_audit(root):
    return subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--root",
            str(root),
            "--labels",
            "envA_retrieval_delivery",
            "--min-clean-per-cell",
            "1",
            "--fail-on-gaps",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    pairs = eligible_native_task_pairs("envA_open_close", {
        "before_graph": {
            "nodes": [
                {
                    "type": "object",
                    "id": "door_0",
                    "category": "door",
                    "available_states": ["Open"],
                    "rooms": ["room_0"],
                    "bbox": {"min": [0, 0, 0], "max": [1, 1, 2.2]},
                },
                {
                    "type": "object",
                    "id": "floors_0",
                    "category": "floors",
                    "rooms": ["room_0"],
                    "bbox": {"min": [0, 0, -0.3], "max": [1, 1, 0]},
                },
                {
                    "type": "object",
                    "id": "top_cabinet_0",
                    "category": "top_cabinet",
                    "available_states": ["Open"],
                    "rooms": ["room_0"],
                    "bbox": {"min": [0, 0, 1.6], "max": [1, 1, 2.4]},
                },
            ],
        },
    })
    assert pairs == {("open_door", "door_0"), ("close_door", "door_0")}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scenes.txt").write_text("Scene_0\n", encoding="utf-8")
        out = root / "envA_retrieval_delivery" / "Scene_0"
        out.mkdir(parents=True)
        (out / "online_env_a_0001.json").write_text(
            json.dumps(sample("Scene_0")),
            encoding="utf-8",
        )

        result = run_audit(root)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert report["covered_cells"] == 1
        assert report["label_summary"]["envA_retrieval_delivery"]["unique_target_models"] == 1

        (root / "scenes.txt").write_text("Scene_0\nScene_1\n", encoding="utf-8")
        result = run_audit(root)
        assert result.returncode == 2
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["gaps"]["missing_cells"] == [{
            "label": "envA_retrieval_delivery",
            "scene": "Scene_1",
            "clean": 0,
            "required": 1,
        }]

    print("PASS: DeltaSG coverage gate enforces every requested scene/label cell")


if __name__ == "__main__":
    main()
