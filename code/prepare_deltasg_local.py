"""Stage accepted DeltaSG Env-A/B/C artifacts for local distribution.

Reads one or more batch roots whose scenes follow the unified
``*/generation/<env>_all/online_<env>_*.json`` + ``*/expert/<env>_all/<sample>/``
layout and produces a self-contained, publish-ready directory (no ModelScope
dependency) containing copied samples, a portable expert result per sample,
and dataset-level manifest/summary/README files.

Each sample is placed under ``data/<Env>/<scene>/<unique_id>/``, where
``<unique_id>`` is a short ``<a|b|c>_<hash>`` id derived deterministically from
the source batch, environment, scene and original run id. The same id replaces
the original run id (and every id derived from it: task_id, delta_id, env_id,
object names, plus any cross-referenced Env-B fire run id) inside the staged
generation.json and expert_result.json, so samples from different batches never
collide after merging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Any


_LAYOUTS = {
    "Env-A": (("*/generation/envA_all/online_env_a_*.json", "envA_all"),),
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
_RUN_ID_RE = re.compile(r"online_env_[abc](?:_[a-z]+)*_\d{4}")
_ENV_PREFIX = {"Env-A": "a", "Env-B": "b", "Env-C": "c"}


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


def _uses_flame_asset(item: dict) -> bool:
    """True if a state-changed object applies the bundled Flame_Animation.usdz effect."""
    effect = item.get("visual_effect") or {}
    return (
        effect.get("source") == "Flame_Animation.usdz"
        or str(effect.get("mode", "")).startswith("deltasg_usdz_flame")
    )


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


def _stage_file(source: Path, destination: Path) -> None:
    """Copy ``source`` to ``destination``, creating parent directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        if source_file.name == "expert_result.json":
            continue
        relative = source_file.relative_to(source)
        _stage_file(source_file, destination / relative)
        file_count += 1
        total_bytes += source_file.stat().st_size
    return file_count, total_bytes


def _sample_paths(root: Path, env_type: str):
    for pattern, family_dir in _LAYOUTS[env_type]:
        for generation_path in sorted(root.glob(pattern)):
            scene_root = generation_path.parent.parent.parent
            expert_dir = scene_root / "expert" / family_dir / generation_path.stem
            yield scene_root.name, generation_path, expert_dir


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "batch"


def _assign_batch_tokens(sources: list[tuple[str, Path]]) -> dict[Path, str]:
    """Map each distinct source root to a short, collision-free batch token."""
    roots = list(dict.fromkeys(root for _, root in sources))
    tokens = [_slug(root.name) for root in roots]
    counts = Counter(tokens)
    assigned: dict[Path, str] = {}
    for root, token in zip(roots, tokens):
        if counts[token] > 1:
            digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
            token = f"{token}__{digest}"
        assigned[root] = token
    return assigned


def _env_type_from_run_id(run_id: str) -> str:
    if run_id.startswith("online_env_a_"):
        return "Env-A"
    if run_id.startswith("online_env_b_"):
        return "Env-B"
    return "Env-C"


def _new_run_id(env_type: str, batch_token: str, scene: str, old_run_id: str) -> str:
    digest = hashlib.sha1(
        f"{batch_token}:{env_type}:{scene}:{old_run_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{_ENV_PREFIX[env_type]}_{digest}"


def _collect_run_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found.update(_RUN_ID_RE.findall(key))
            _collect_run_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_run_ids(item, found)
    elif isinstance(value, str):
        found.update(_RUN_ID_RE.findall(value))


def _replace_run_ids(text: str, replacements: dict[str, str]) -> str:
    return _RUN_ID_RE.sub(lambda m: replacements.get(m.group(0), m.group(0)), text)


def _rewrite_run_ids(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            (_replace_run_ids(key, replacements) if isinstance(key, str) else key): (
                _rewrite_run_ids(item, replacements)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_run_ids(item, replacements) for item in value]
    if isinstance(value, str):
        return _replace_run_ids(value, replacements)
    return value


def prepare_sources(sources: list[tuple[str, Path]], output: Path) -> dict:
    sources = [(env_type, root.resolve()) for env_type, root in sources]
    batch_tokens = _assign_batch_tokens(sources)
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

            leaf_id = _new_run_id(env_type, batch_tokens[root], scene, generation_path.stem)
            run_id_bases: set[str] = set()
            _collect_run_ids(generation, run_id_bases)
            _collect_run_ids(expert_result, run_id_bases)
            replacements = {
                base: _new_run_id(_env_type_from_run_id(base), batch_tokens[root], scene, base)
                for base in run_id_bases
            }

            sample_dir = data_root / env_type / scene / leaf_id
            sample_dir.mkdir(parents=True)
            staged_generation = sample_dir / "generation.json"
            staged_generation.write_text(
                json.dumps(_rewrite_run_ids(generation, replacements), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            expert_files, expert_bytes = _copy_tree(expert_dir, sample_dir / "expert")
            portable_result = _portable_result(expert_result, expert_dir, generation_path)
            portable_result = _rewrite_run_ids(portable_result, replacements)
            staged_result = sample_dir / "expert" / "expert_result.json"
            staged_result.write_text(
                json.dumps(portable_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            sample_files = expert_files + 2
            sample_bytes = (
                expert_bytes + staged_generation.stat().st_size + staged_result.stat().st_size
            )
            total_files += sample_files
            total_bytes += sample_bytes

            task = (generation.get("task_environment") or {}).get("task") or generation.get("task") or {}
            generation_config = (generation.get("task_environment") or {}).get("generation") or {}
            if any(
                _uses_flame_asset(item)
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
                    "sample_id": f"{env_type}/{scene}/{leaf_id}",
                    "source_run_id": generation_path.stem,
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
        "schema_version": "deltasg_staging.v2",
        "source_batches": [root.name for root in dict.fromkeys(root for _, root in sources)],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batch_root",
        type=Path,
        nargs="+",
        help="input batch root(s) containing */generation and */expert trees",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--env-types",
        nargs="+",
        choices=["Env-A", "Env-B", "Env-C"],
        help="restrict to these environment types (default: auto-detect all present)",
    )
    args = parser.parse_args()

    roots = [root.resolve() for root in args.batch_root]
    for root in roots:
        if not root.is_dir():
            parser.error(f"batch root does not exist: {root}")

    sources: list[tuple[str, Path]] = []
    for root in roots:
        env_types = args.env_types or [
            env_type
            for env_type in _LAYOUTS
            if any(root.glob(pattern) for pattern, _ in _LAYOUTS[env_type])
        ]
        if not env_types:
            parser.error(f"no Env-A/B/C generation files found under {root}")
        sources.extend((env_type, root) for env_type in env_types)
    output = args.output
    print(json.dumps(prepare_sources(sources, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
