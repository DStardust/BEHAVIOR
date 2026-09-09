import base64
import json
from pathlib import Path

from prepare_deltasg_modelscope import prepare, prepare_sources


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_generation(path: Path, task: str = "respond_to_smoke_warning") -> None:
    _write_json(
        path,
        {
            "ok": True,
            "task_environment": {
                "task": {
                    "primary_behavior_task": task,
                    "instruction": "Complete the task.",
                },
                "generation": {
                    "llm_model": "qwen3.8-max",
                    "solvability_profile": "oracle_symbolic",
                },
            },
            "validation": {
                "camera_coverage": {
                    "num_global_cameras": 2,
                    "global_camera_rooms": ["living_room_0", "bedroom_0"],
                }
            },
        },
    )


def _write_expert(expert: Path, generation: Path, accepted: bool = True) -> None:
    event_dir = expert / "frames" / "step_001_pre"
    robot = event_dir / "robot_primary"
    robot.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("rgb", "seg_semantic_preview", "seg_instance_preview"):
        path = robot / f"{name}.png"
        path.write_bytes(_PNG)
        paths[name] = str(path)
    for name in ("seg_semantic", "seg_instance"):
        path = robot / f"{name}.npy"
        path.write_bytes(b"npy")
        paths[name] = str(path)

    cameras = []
    for index in (1, 2):
        path = event_dir / "global" / f"camera_{index}" / "rgb.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_PNG)
        cameras.append(
            {
                "camera_id": f"camera_{index}",
                "paths": {"rgb": str(path)},
            }
        )
    action = expert / "actions" / "step_001.npy"
    action.parent.mkdir(parents=True, exist_ok=True)
    action.write_bytes(b"action")
    _write_json(
        expert / "expert_result.json",
        {
            "accepted": accepted,
            "qa_eligible": accepted,
            "input": str(generation),
            "backend": {
                "name": "oracle_symbolic",
                "low_level_vla_actions_eligible": False,
            },
            "observation_events": [
                {
                    "event_id": "step_001_pre",
                    "robot_primary": {"paths": paths},
                    "global_cameras": cameras,
                }
            ],
            "steps": [
                {
                    "step": {"step_id": 1},
                    "pre_observation": "step_001_pre",
                    "post_observation": "step_001_pre",
                    "actions_path": str(action),
                }
            ],
        },
    )


def test_prepare_stages_only_generation_and_expert_accepted_samples(tmp_path):
    root = tmp_path / "batch"
    scene = "Beechwood_0_int"
    for index, accepted in enumerate((True, False)):
        sample = f"online_env_b_fire_{index:04d}"
        generation = root / scene / "generation" / "envB_fire" / f"{sample}.json"
        _write_generation(generation)
        expert = root / scene / "expert" / "envB_fire" / sample
        _write_expert(expert, generation, accepted)

    output = root / "modelscope_ready"
    summary = prepare(root, output)

    assert summary["accepted_samples"] == 1
    assert summary["exclusions"] == {"Env-B:expert_missing_or_rejected": 1}
    rows = [json.loads(line) for line in (output / "accepted_manifest.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [
        f"Env-B/{scene}/online_env_b_fire_0000"
    ]
    assert rows[0]["visualization_complete"] is True
    assert rows[0]["low_level_vla_actions_eligible"] is False
    staged = output / rows[0]["sample_dir"]
    assert (staged / "generation.json").is_file()
    assert (
        staged
        / "expert"
        / "frames"
        / "step_001_pre"
        / "robot_primary"
        / "rgb.png"
    ).read_bytes() == _PNG
    portable = json.loads((staged / "expert" / "expert_result.json").read_text())
    assert portable["input"] == "generation.json"
    assert portable["steps"][0]["actions_path"] == "expert/actions/step_001.npy"
    assert not (
        output / "data" / "Env-B" / scene / "online_env_b_fire_0001"
    ).exists()


def test_prepare_sources_combines_env_a_b_c_layouts(tmp_path):
    scene = "Beechwood_0_int"
    roots = {name: tmp_path / name for name in ("enva", "envb", "envc")}
    layouts = {
        "Env-A": (
            roots["enva"] / scene / "generation" / "online_env_a_0000.json",
            None,
        ),
        "Env-B": (
            roots["envb"]
            / scene
            / "generation"
            / "envB_fire"
            / "online_env_b_fire_0000.json",
            "envB_fire",
        ),
        "Env-C": (
            roots["envc"]
            / scene
            / "generation"
            / "envC_all"
            / "online_env_c_appliance_0000.json",
            "envC_all",
        ),
    }
    root_for_env = {"Env-A": roots["enva"], "Env-B": roots["envb"], "Env-C": roots["envc"]}
    for env_type, (generation, family_dir) in layouts.items():
        _write_generation(generation, task=f"task_{env_type}")
        expert = root_for_env[env_type] / scene / "expert"
        if family_dir:
            expert /= family_dir
        _write_expert(expert / generation.stem, generation)

    output = tmp_path / "release"
    summary = prepare_sources(
        [
            ("Env-A", roots["enva"]),
            ("Env-B", roots["envb"]),
            ("Env-C", roots["envc"]),
        ],
        output,
    )

    assert summary["accepted_samples"] == 3
    assert summary["accepted_by_env"] == {"Env-A": 1, "Env-B": 1, "Env-C": 1}
    assert len((output / "accepted_manifest.jsonl").read_text().splitlines()) == 3


def test_prepare_discovers_new_env_b_all_layout(tmp_path):
    root = tmp_path / "batch"
    scene = "Rs_int"
    sample = "online_env_b_dirty_clothes_0000"
    generation = root / scene / "generation" / "envB_all" / f"{sample}.json"
    _write_generation(generation, task="collect_dirty_clothes")
    expert = root / scene / "expert" / "envB_all" / sample
    _write_expert(expert, generation)

    output = tmp_path / "release"
    summary = prepare(root, output)

    assert summary["accepted_samples"] == 1
    assert summary["accepted_by_task"] == {"Env-B/collect_dirty_clothes": 1}
