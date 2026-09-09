"""Stage accepted DeltaSG Env-A/B/C artifacts for ModelScope upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any


_LAYOUTS = {
    "Env-A": (("*/generation/online_env_a_*.json", None),),
    "Env-B": (
        ("*/generation/envB_all/online_env_b_*.json", "envB_all"),
        ("*/generation/envB_fire/online_env_b_fire_*.json", "envB_fire"),
    ),
    "Env-C": (("*/generation/envC_all/online_env_c_*.json", "envC_all"),),
}
_ROBOT_PATHS = (
    "rgb",
    "seg_semantic",
    "seg_semantic_preview",
    "seg_instance",
    "seg_instance_preview",
)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return (width, height) if width > 0 and height > 0 else None


def _source_artifact(value: Any, expert_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = expert_dir / path
    try:
        path = path.resolve()
        path.relative_to(expert_dir.resolve())
    except (OSError, ValueError):
        return None
    return path


def _audit_visualization(expert_dir: Path, expert_result: dict) -> dict:
    errors: list[str] = []
    events = expert_result.get("observation_events") or []
    event_ids = {event.get("event_id") for event in events}
    resolutions: Counter[str] = Counter()
    global_frames = 0

    steps = expert_result.get("steps") or []
    if not events:
        errors.append("no_observation_events")
    if not steps:
        errors.append("no_expert_steps")
    for step in steps:
        step_id = step.get("step", {}).get("step_id")
        for key in ("pre_observation", "post_observation"):
            if step.get(key) not in event_ids:
                errors.append(f"step_{step_id}_{key}_missing")
        action_path = _source_artifact(step.get("actions_path"), expert_dir)
        if action_path is None or not action_path.is_file() or action_path.stat().st_size == 0:
            errors.append(f"step_{step_id}_actions_missing")

    for event in events:
        event_id = event.get("event_id") or "unknown"
        robot_paths = (event.get("robot_primary") or {}).get("paths") or {}
        for key in _ROBOT_PATHS:
            artifact = _source_artifact(robot_paths.get(key), expert_dir)
            if artifact is None or not artifact.is_file() or artifact.stat().st_size == 0:
                errors.append(f"{event_id}_robot_{key}_missing")
                continue
            if key == "rgb" or key.endswith("preview"):
                size = _png_size(artifact)
                if size is None:
                    errors.append(f"{event_id}_robot_{key}_invalid_png")
                else:
                    resolutions[f"{size[0]}x{size[1]}"] += 1

        global_cameras = event.get("global_cameras") or []
        if len(global_cameras) < 2:
            errors.append(f"{event_id}_fewer_than_two_global_cameras")
        for camera in global_cameras:
            camera_id = camera.get("camera_id") or "unknown"
            artifact = _source_artifact((camera.get("paths") or {}).get("rgb"), expert_dir)
            if artifact is None or not artifact.is_file() or artifact.stat().st_size == 0:
                errors.append(f"{event_id}_{camera_id}_rgb_missing")
                continue
            size = _png_size(artifact)
            if size is None:
                errors.append(f"{event_id}_{camera_id}_invalid_png")
            else:
                resolutions[f"{size[0]}x{size[1]}"] += 1
                global_frames += 1

    return {
        "complete": not errors,
        "errors": errors,
        "observation_events": len(events),
        "global_rgb_frames": global_frames,
        "resolutions": dict(sorted(resolutions.items())),
    }


def _portable_result(value: Any, expert_dir: Path, generation_path: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable_result(item, expert_dir, generation_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_result(item, expert_dir, generation_path) for item in value]
    if not isinstance(value, str) or not value.startswith("/"):
        return value

    path = Path(value)
    try:
        resolved = path.resolve()
        if resolved == generation_path.resolve():
            return "generation.json"
        relative = resolved.relative_to(expert_dir.resolve())
    except (OSError, ValueError):
        return value
    return str(Path("expert") / relative)


def _link_tree(source: Path, destination: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        if source_file.name == "expert_result.json":
            continue
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_file, destination_file)
        file_count += 1
        total_bytes += source_file.stat().st_size
    return file_count, total_bytes


def _sample_paths(root: Path, env_type: str):
    for pattern, family_dir in _LAYOUTS[env_type]:
        for generation_path in sorted(root.glob(pattern)):
            if family_dir is None:
                scene_root = generation_path.parent.parent
                expert_dir = scene_root / "expert" / generation_path.stem
            else:
                scene_root = generation_path.parent.parent.parent
                expert_dir = scene_root / "expert" / family_dir / generation_path.stem
            yield scene_root.name, generation_path, expert_dir


def prepare_sources(sources: list[tuple[str, Path]], output: Path) -> dict:
    sources = [(env_type, root.resolve()) for env_type, root in sources]
    output = output.resolve()
    for _, root in sources:
        if output == root or output in root.parents:
            raise ValueError("output must not be a source root or its parent")
    if output.exists():
        shutil.rmtree(output)
    data_root = output / "data"
    data_root.mkdir(parents=True)

    rows = []
    exclusions: Counter[str] = Counter()
    accepted_by_env: Counter[str] = Counter()
    accepted_by_scene: Counter[str] = Counter()
    accepted_by_task: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    needs_flame_asset = False

    for env_type, root in sources:
        if env_type not in _LAYOUTS:
            raise ValueError(f"unsupported environment type: {env_type}")
        for scene, generation_path, expert_dir in _sample_paths(root, env_type):
            generation = _load_json(generation_path)
            if generation.get("ok") is not True:
                exclusions[f"{env_type}:generation_failed"] += 1
                continue
            expert_result_path = expert_dir / "expert_result.json"
            expert_result = _load_json(expert_result_path)
            if expert_result.get("accepted") is not True:
                exclusions[f"{env_type}:expert_missing_or_rejected"] += 1
                continue
            visual = _audit_visualization(expert_dir, expert_result)
            if not visual["complete"]:
                exclusions[f"{env_type}:visualization_incomplete"] += 1
                continue

            sample_dir = data_root / env_type / scene / generation_path.stem
            sample_dir.mkdir(parents=True)
            staged_generation = sample_dir / "generation.json"
            os.link(generation_path, staged_generation)
            expert_files, expert_bytes = _link_tree(expert_dir, sample_dir / "expert")
            portable_result = _portable_result(expert_result, expert_dir, generation_path)
            staged_result = sample_dir / "expert" / "expert_result.json"
            staged_result.parent.mkdir(parents=True, exist_ok=True)
            staged_result.write_text(
                json.dumps(portable_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            sample_files = expert_files + 2
            sample_bytes = (
                expert_bytes + generation_path.stat().st_size + staged_result.stat().st_size
            )
            total_files += sample_files
            total_bytes += sample_bytes

            task = (generation.get("task_environment") or {}).get("task") or generation.get("task") or {}
            generation_config = (generation.get("task_environment") or {}).get("generation") or {}
            if any(
                (item.get("visual_effect") or {}).get("mode") == "deltasg_usdz_flame_v1"
                for item in (generation.get("task_environment") or {}).get("state_changed_objects") or []
            ):
                needs_flame_asset = True
            camera = (generation.get("validation") or {}).get("camera_coverage") or {}
            backend = expert_result.get("backend") or {}
            task_name = task.get("primary_behavior_task") or expert_result.get("task_name")
            accepted_by_env[env_type] += 1
            accepted_by_scene[f"{env_type}/{scene}"] += 1
            accepted_by_task[f"{env_type}/{task_name}"] += 1
            rows.append(
                {
                    "sample_id": f"{env_type}/{scene}/{generation_path.stem}",
                    "env_type": env_type,
                    "scene": scene,
                    "task": task_name,
                    "task_family": expert_result.get("task_family"),
                    "instruction": task.get("instruction"),
                    "llm_model": generation_config.get("llm_model"),
                    "expert_backend": backend.get("name"),
                    "generation_profile": generation_config.get("solvability_profile"),
                    "accepted": True,
                    "qa_eligible": bool(expert_result.get("qa_eligible")),
                    "low_level_vla_actions_eligible": bool(
                        backend.get("low_level_vla_actions_eligible", False)
                    ),
                    "num_steps": len(expert_result.get("steps") or []),
                    "num_observation_events": visual["observation_events"],
                    "num_global_rgb_frames": visual["global_rgb_frames"],
                    "visualization_complete": True,
                    "visualization_resolutions": visual["resolutions"],
                    "robot_modalities": ["rgb", "seg_semantic", "seg_instance"],
                    "global_modalities": ["rgb"],
                    "num_global_cameras": camera.get("num_global_cameras"),
                    "global_camera_rooms": camera.get("global_camera_rooms") or [],
                    "source_batch": root.name,
                    "generation_json": str(staged_generation.relative_to(output)),
                    "expert_result": str(staged_result.relative_to(output)),
                    "sample_dir": str(sample_dir.relative_to(output)),
                    "file_count": sample_files,
                    "bytes": sample_bytes,
                }
            )

    shared_assets = []
    if needs_flame_asset:
        source_asset = Path(__file__).resolve().parent / "assets" / "Flame_Animation.usdz"
        if not source_asset.is_file():
            raise FileNotFoundError(f"packaged flame asset is missing: {source_asset}")
        staged_asset = output / "assets" / source_asset.name
        staged_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, staged_asset)
        shared_assets.append(str(staged_asset.relative_to(output)))
        total_files += 1
        total_bytes += staged_asset.stat().st_size

    manifest = output / "accepted_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "deltasg_modelscope_staging.v2",
        "source_batches": [root.name for _, root in sources],
        "accepted_samples": len(rows),
        "accepted_by_env": dict(sorted(accepted_by_env.items())),
        "accepted_by_scene": dict(sorted(accepted_by_scene.items())),
        "accepted_by_task": dict(sorted(accepted_by_task.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "files": total_files,
        "bytes": total_bytes,
        "shared_assets": shared_assets,
        "visualization_complete": True,
        "expert_backend": "oracle_symbolic",
        "low_level_vla_actions_eligible": False,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    env_counts = ", ".join(
        f"{env_type}: {count}" for env_type, count in sorted(accepted_by_env.items())
    )
    (output / "README.md").write_text(
        "# DeltaSG Env-A/B/C Accepted Samples\n\n"
        f"This release contains {len(rows)} accepted samples ({env_counts}). Every "
        "listed expert observation has robot-primary RGB, semantic and instance "
        "segmentation (raw NPY plus PNG previews), and 2-3 global-camera RGB views.\n\n"
        "Only samples that passed generation and the complete oracle_symbolic expert "
        "plan are included. Expert JSON paths are relative to each sample directory.\n\n"
        "The oracle_symbolic backend uses audited OmniGibson state transitions but may "
        "teleport during high-level navigation or manipulation. The data provide visual "
        "and state supervision and are not low-level physical VLA action trajectories. "
        "Global cameras contain RGB only; segmentation is supplied by the robot-primary "
        "view.\n",
        encoding="utf-8",
    )
    return summary


def prepare(root: Path, output: Path) -> dict:
    """Backward-compatible Env-B staging entry point."""
    return prepare_sources([("Env-B", root)], output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", nargs="?", type=Path, help="legacy Env-B batch root")
    parser.add_argument("--env-a-root", type=Path)
    parser.add_argument("--env-b-root", type=Path)
    parser.add_argument("--env-c-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    explicit = [
        (env_type, root)
        for env_type, root in (
            ("Env-A", args.env_a_root),
            ("Env-B", args.env_b_root),
            ("Env-C", args.env_c_root),
        )
        if root is not None
    ]
    if explicit:
        if args.batch_root is not None or args.output is None:
            parser.error("Env-A/B/C roots require --output and cannot use batch_root")
        sources = explicit
        output = args.output
    else:
        if args.batch_root is None:
            parser.error("provide batch_root or at least one --env-*-root")
        sources = [("Env-B", args.batch_root)]
        output = args.output or args.batch_root / "modelscope_ready"
    print(json.dumps(prepare_sources(sources, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
