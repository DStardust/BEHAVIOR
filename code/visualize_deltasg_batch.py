"""
Batch visualize online DeltaSG samples with one OmniGibson scene load.

Outputs the legacy before / after pair plus every recorded global camera and the
primary robot camera. No debug markers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True
gm.GUI_VIEWPORT_ONLY = True

import numpy as np
import torch as th
import omnigibson as og
from PIL import Image, ImageDraw, ImageFont
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from omnigibson.utils.constants import PrimType

from api import create_env, get_all_scene_objects, stabilize_robot_spawn
from deltasg_visual_effects import (
    SMOKE_FLOW_RENDER_WARMUP_FRAMES,
    SMOKE_FLOW_WARMUP_STEPS,
    SMOKE_ONLY_ON_FIRE_MODE,
    configure_on_fire_smoke_only,
)
from visualize_env_a import corner_camera, float_list, official_camera_pose, room_corners, yaw_pitch_quat


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run_files(input_dir: Path):
    files = []
    for path in sorted(input_dir.glob("online_env*.json")):
        if path.name.startswith("checkpoint"):
            continue
        try:
            with path.open() as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if run.get("ok") is True:
            files.append(path)
    return files


def object_name(record):
    return record.get("object_name") or record.get("object_id") or record.get("name")


def to_numpy(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def jsonable(value):
    if isinstance(value, th.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def enrich_final_poses(run):
    validation_by_name = {
        object_name(item): item
        for item in (run.get("validation", {}) or {}).get("created_objects", [])
    }
    for ao in (run.get("task_environment", {}) or {}).get("added_objects", []):
        name = object_name(ao)
        if name and not ao.get("final_pose_before_warmup"):
            final_pose = (validation_by_name.get(name) or {}).get("final_pose_before_warmup")
            if final_pose:
                ao["final_pose_before_warmup"] = final_pose


def cleanup_online_objects(env):
    for obj in list(get_all_scene_objects(env.scene)):
        name = getattr(obj, "name", "")
        if name.startswith("online_env_"):
            try:
                env.scene.remove_object(obj)
            except Exception:
                pass
        for state_cls in (object_states.OnFire, object_states.Burnt):
            try:
                if state_cls in obj.states and obj.states[state_cls].get_value():
                    obj.states[state_cls].set_value(False)
            except Exception:
                pass
    for _ in range(3):
        og.sim.render()


def hide_online_objects(env):
    """Hide prior visual reconstructions without invalidating Replicator mappings."""
    for obj in list(env.scene.objects):
        if getattr(obj, "name", "").startswith("online_env_"):
            try:
                obj.visible = False
            except Exception:
                pass
        for state_cls in (object_states.OnFire, object_states.Burnt):
            try:
                if state_cls in obj.states and obj.states[state_cls].get_value():
                    obj.states[state_cls].set_value(False)
            except Exception:
                pass
    for _ in range(3):
        og.sim.render()


def reset_scene(env):
    env.reset()
    for _ in range(30):
        og.sim.render()


def env_type(run):
    return ((run.get("task_environment", {}) or {}).get("env_type") or "").strip()


def camera_room_for_run(run):
    te = run.get("task_environment", {}) or {}
    for item in te.get("added_objects", []) or []:
        room = item.get("room_id") or item.get("room")
        if room:
            return room
    for item in te.get("state_changed_objects", []) or []:
        room = item.get("room_id") or item.get("room")
        if room:
            return room
    task = te.get("task", {}) or {}
    primary_task = str(task.get("primary_behavior_task") or "")
    native_action_task = primary_task.startswith(("open_", "close_", "turn_on_", "turn_off_"))
    for item in task.get("plan_objects", []) or []:
        # Retrieval support furniture is only context. For native state-change
        # tasks, however, the reference-only fixture is the actual target.
        if item.get("reference_only") and not native_action_task:
            continue
        room = item.get("room_id") or item.get("room")
        if room:
            return room
    return task.get("target_room")


def camera_for_run(env, run):
    te = run.get("task_environment", {}) or {}
    camera_room = camera_room_for_run(run)
    if camera_room:
        cam_pos, cam_ori, method = official_camera_pose(env, camera_room)
        if cam_pos is not None:
            return cam_pos, cam_ori, method

    positions = []
    for ao in te.get("added_objects", []):
        pose = ao.get("final_pose_before_warmup") or ao.get("pose") or (ao.get("placement") or {}).get("pose") or {}
        if pose.get("position"):
            positions.append(float_list(pose["position"]))
    if positions:
        center = np.mean(positions, axis=0)
    else:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return (
        np.array([float(center[0]) + 2.0, float(center[1]) + 2.0, 2.5], dtype=np.float32),
        yaw_pitch_quat(45.0, 45.0),
        "fallback",
    )


def wall_center_camera_from_wall(c1, c2, room_center, v_angle=45.0, inward=0.2, height=2.2):
    wall_center = np.array([(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5], dtype=np.float32)
    wall_dir = np.array([c2[0] - c1[0], c2[1] - c1[1]], dtype=np.float32)
    norm = float(np.linalg.norm(wall_dir))
    if norm < 1e-6:
        return None
    wall_dir = wall_dir / norm
    normals = [np.array([-wall_dir[1], wall_dir[0]], dtype=np.float32), np.array([wall_dir[1], -wall_dir[0]], dtype=np.float32)]
    normal = max(normals, key=lambda n: float(np.dot(n, room_center[:2] - wall_center)))
    cam_pos = np.array([wall_center[0] + normal[0] * inward, wall_center[1] + normal[1] * inward, height], dtype=np.float32)
    normal_angle = math.degrees(math.atan2(normal[1], normal[0]))
    return cam_pos, yaw_pitch_quat(normal_angle - 90.0, v_angle)


def official_camera_candidates(env, room_name):
    candidates = []
    if room_name:
        cam_pos, cam_ori, method = official_camera_pose(env, room_name)
        if cam_pos is not None:
            candidates.append((cam_pos, cam_ori, method))

    corners = room_corners(env, room_name) if room_name else None
    if corners is None:
        return candidates

    center_xy = np.mean(np.stack([corners[name][:2] for name in ("SW", "SE", "NE", "NW")]), axis=0)
    room_center = np.array([center_xy[0], center_xy[1]], dtype=np.float32)
    diag = float(np.linalg.norm(corners["NE"][:2] - corners["SW"][:2]))
    opposite = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}

    view_angles = (30, 45, 60)
    if diag <= 3.0:
        walls = [
            ("SW-SE", corners["SW"], corners["SE"]),
            ("SE-NE", corners["SE"], corners["NE"]),
            ("NE-NW", corners["NE"], corners["NW"]),
            ("NW-SW", corners["NW"], corners["SW"]),
        ]
        for wall_name, c1, c2 in walls:
            for v_angle in view_angles:
                result = wall_center_camera_from_wall(c1, c2, room_center, v_angle=v_angle)
                if result is not None:
                    cam_pos, cam_ori = result
                    candidates.append(
                        (cam_pos, cam_ori, f"wall-center {wall_name} diag={diag:.2f} v={v_angle}")
                    )
    else:
        room_configs = {"living_room_0": [("SW", 30), ("SE", 30), ("NW", 30), ("NE", 30)],
                        "bedroom_0": [("SE", 45), ("SW", 45), ("NE", 45), ("NW", 45)],
                        "bathroom_0": [("NE", 45), ("NW", 45), ("SE", 45), ("SW", 45)]}
        configs = room_configs.get(room_name, [("SW", 30), ("SE", 30), ("NW", 30), ("NE", 30)])
        for corner_name, preferred_angle in configs:
            angles = (preferred_angle,) + tuple(angle for angle in view_angles if angle != preferred_angle)
            for v_angle in angles:
                cam_pos, cam_ori = corner_camera(
                    corners[corner_name], corners[opposite[corner_name]], v_angle=v_angle
                )
                candidates.append((cam_pos, cam_ori, f"corner {corner_name} diag={diag:.2f} v={v_angle}"))

    unique = []
    seen = set()
    for cam_pos, cam_ori, method in candidates:
        key = tuple(np.round(np.asarray(cam_pos, dtype=float), 3).tolist() + np.round(np.asarray(cam_ori, dtype=float), 3).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append((cam_pos, cam_ori, method))
    return unique


def nearby_camera_rooms(env, run, primary_room, limit=3):
    te = run.get("task_environment", {}) or {}
    target_positions = []
    for item in te.get("added_objects", []) or []:
        pose = item.get("pose") or item.get("final_pose_before_warmup") or {}
        if pose.get("position"):
            target_positions.append(np.asarray(pose["position"][:2], dtype=float))
    if not target_positions:
        return []
    target_xy = np.mean(target_positions, axis=0)
    room_distances = {}
    for obj in get_all_scene_objects(env.scene):
        rooms = getattr(obj, "in_rooms", None) or []
        if isinstance(rooms, str):
            rooms = [rooms]
        if not rooms:
            continue
        try:
            position, _ = obj.get_position_orientation()
            distance = float(np.linalg.norm(np.asarray(position[:2], dtype=float) - target_xy))
        except Exception:
            continue
        for room in rooms:
            if room and room != primary_room:
                room_distances[str(room)] = min(distance, room_distances.get(str(room), float("inf")))
    return [room for room, _ in sorted(room_distances.items(), key=lambda item: (item[1], item[0]))[:limit]]


def target_object_metadata(run):
    metadata = {}
    te = run.get("task_environment", {}) or {}
    role_by_name = {}
    validation = run.get("validation", {}) or {}
    for key in ("solution_tool", "candidate_solution", "semantic_distractor"):
        item = validation.get(key)
        if isinstance(item, dict):
            name = object_name(item)
            role = item.get("semantic_role") or key
            if name:
                role_by_name[name] = role
    reasoning = te.get("semantic_reasoning", {}) or run.get("semantic_reasoning", {}) or {}
    optimal = ((reasoning.get("ground_truth") or {}).get("optimal_object"))
    if optimal:
        role_by_name.setdefault(optimal, "optimal_solution")
    for rejected in (reasoning.get("ground_truth") or {}).get("rejected_candidates", []):
        name = object_name(rejected)
        if name:
            role_by_name.setdefault(name, rejected.get("reason") or "rejected_candidate")

    for ao in te.get("added_objects", []):
        name = object_name(ao)
        if not name:
            continue
        metadata[name] = {
            "object_name": name,
            "category": ao.get("category"),
            "role": ao.get("role") or ao.get("semantic_role") or role_by_name.get(name),
            "placement": ao.get("placement"),
            "target_kind": "added_object",
            "room": ao.get("room_id") or ao.get("room"),
        }
    for item in te.get("state_changed_objects", []):
        name = object_name(item)
        if not name:
            continue
        metadata[name] = {
            "object_name": name,
            "category": item.get("category"),
            "role": next(iter(item.get("semantic_roles") or []), None),
            "states": item.get("states"),
            "target_kind": "state_changed_object",
            "room": item.get("room_id") or item.get("room"),
        }
    task = te.get("task", {}) or {}
    solution_plan = te.get("solution_plan") or run.get("solution_plan") or []
    action_targets = {
        step.get(key)
        for step in solution_plan
        for key in ("target_object", "tool_object")
        if step.get(key)
    }
    for item in task.get("plan_objects", []) or []:
        name = object_name(item)
        semantic_role = item.get("semantic_role") or role_by_name.get(name)
        if item.get("reference_only") and not semantic_role and name not in action_targets:
            continue
        if not name or name in metadata:
            continue
        metadata[name] = {
            "object_name": name,
            "category": item.get("category"),
            "role": semantic_role or item.get("role") or "action_target",
            "target_kind": "plan_object",
            "room": item.get("room_id") or item.get("room"),
        }
    return metadata


def target_camera_rooms(run, target_metadata):
    rooms = {meta.get("room") for meta in target_metadata.values() if meta.get("room")}
    primary = camera_room_for_run(run)
    if primary:
        rooms.add(primary)
    return sorted(rooms)


def get_camera_obs(viewer, cam_pos, cam_ori):
    viewer.set_position_orientation(
        position=np.array(cam_pos, dtype=np.float32),
        orientation=np.array(cam_ori, dtype=np.float32),
    )
    last_error = None
    for retry in range(3):
        for _ in range(10 * (retry + 1)):
            og.sim.render()
        try:
            return viewer.get_obs()
        except RuntimeError as exc:
            last_error = exc
            if "input.numel() == 0" not in str(exc):
                raise
            print(f"[vis] empty segmentation frame; retry {retry + 1}/3", flush=True)
    raise last_error


def capture_obs(viewer, cam_pos, cam_ori, path: Path | str):
    obs, info = get_camera_obs(viewer, cam_pos, cam_ori)
    if isinstance(obs, dict) and "rgb" in obs:
        arr = np.asarray(obs["rgb"])
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        Image.fromarray(arr.astype(np.uint8)).save(path)
        print(f"  Saved: {path}")
    return obs, info


def _safe_view_name(value):
    text = str(value or "camera")
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def recorded_camera_views(run):
    """Return the generation-time global views and primary robot view."""
    cameras = run.get("camera") or (run.get("task_environment", {}) or {}).get("camera") or []
    globals_ = [camera for camera in cameras if camera.get("camera_type") == "global_camera"]
    primary_robot = next(
        (
            camera
            for camera in cameras
            if camera.get("camera_type") == "robot_camera" and camera.get("is_primary") is True
        ),
        None,
    )
    views = [(f"global_{idx:02d}_{_safe_view_name(camera.get('camera_id'))}", camera) for idx, camera in enumerate(globals_, 1)]
    if primary_robot is not None:
        views.append((f"robot_primary_{_safe_view_name(primary_robot.get('camera_id'))}", primary_robot))
    return views


def save_segmentation_visualization(obs, info, modality, path):
    if not isinstance(obs, dict) or modality not in obs:
        return None
    segmentation = to_numpy(obs[modality])
    if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
        segmentation = segmentation[..., 0]
    if segmentation.ndim != 2:
        return None
    ids = segmentation.astype(np.uint64, copy=False)
    color = np.stack(
        [
            (ids * 37 + 17) % 251,
            (ids * 67 + 53) % 251,
            (ids * 97 + 89) % 251,
        ],
        axis=-1,
    ).astype(np.uint8)
    color[ids == 0] = 0
    Image.fromarray(color).save(path)
    labels = jsonable((info or {}).get(modality) or {})
    label_path = Path(path).with_suffix(".json")
    with label_path.open("w", encoding="utf-8") as stream:
        json.dump(labels, stream, ensure_ascii=False, indent=2)
    return str(path)


def save_recorded_camera_views(run, stem, output_dir, viewer, *, draw_bboxes):
    sample_dir = output_dir / stem / "recorded_cameras"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for view_name, camera in recorded_camera_views(run):
        pose = camera.get("pose") or {}
        position = pose.get("position")
        orientation = pose.get("orientation_xyzw")
        if not position or not orientation:
            print(f"[vis] warning: {view_name} has no recorded pose", flush=True)
            continue
        rgb_path = sample_dir / f"{view_name}_after.png"
        obs, info = capture_obs(viewer, position, orientation, rgb_path)
        save_bbox_outputs(
            run,
            view_name,
            sample_dir,
            position,
            orientation,
            obs,
            info,
            draw_overlay=draw_bboxes,
        )
        instance_path = save_segmentation_visualization(
            obs, info, "seg_instance", sample_dir / f"{view_name}_seg_instance.png"
        )
        semantic_path = save_segmentation_visualization(
            obs, info, "seg_semantic", sample_dir / f"{view_name}_seg_semantic.png"
        )
        rgb = np.asarray((obs or {}).get("rgb")) if isinstance(obs, dict) else np.empty(0)
        rendered.append(
            {
                "view_name": view_name,
                "camera_id": camera.get("camera_id"),
                "camera_type": camera.get("camera_type"),
                "room_id": camera.get("room_id"),
                "recorded_resolution": camera.get("resolution"),
                "rendered_resolution": (
                    {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])} if rgb.ndim >= 2 else None
                ),
                "rgb": str(rgb_path),
                "seg_instance": instance_path,
                "seg_semantic": semantic_path,
            }
        )
    manifest = {
        "run_id": run.get("run_id"),
        "num_recorded_global_cameras": sum(
            camera.get("camera_type") == "global_camera" for _, camera in recorded_camera_views(run)
        ),
        "num_rendered_views": len(rendered),
        "views": rendered,
    }
    with (sample_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def compute_instance_bboxes(obs, info, target_metadata, min_pixels=8):
    if not isinstance(obs, dict) or "seg_instance" not in obs:
        return []
    seg = to_numpy(obs["seg_instance"])
    if seg.ndim != 2:
        return []
    id_to_name = (info or {}).get("seg_instance") or {}
    bboxes = []
    for raw_id, label in id_to_name.items():
        name = str(label)
        if name not in target_metadata:
            continue
        try:
            instance_id = int(raw_id)
        except Exception:
            continue
        ys, xs = np.where(seg == instance_id)
        if len(xs) < min_pixels:
            continue
        meta = target_metadata[name]
        bboxes.append(
            {
                "object_name": name,
                "category": meta.get("category"),
                "role": meta.get("role"),
                "target_kind": meta.get("target_kind"),
                "source": "seg_instance_pixels",
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "pixel_count": int(len(xs)),
                "image_size": [int(seg.shape[1]), int(seg.shape[0])],
            }
        )
    return sorted(bboxes, key=lambda item: item["object_name"])


def visible_instance_summary(obs, info, max_items=200):
    if not isinstance(obs, dict) or "seg_instance" not in obs:
        return []
    seg = to_numpy(obs["seg_instance"])
    if seg.ndim != 2:
        return []
    id_to_name = (info or {}).get("seg_instance") or {}
    items = []
    for raw_id, label in id_to_name.items():
        try:
            instance_id = int(raw_id)
        except Exception:
            continue
        pixel_count = int(np.count_nonzero(seg == instance_id))
        if pixel_count <= 0:
            continue
        items.append({"id": instance_id, "name": str(label), "pixel_count": pixel_count})
    items.sort(key=lambda item: item["pixel_count"], reverse=True)
    return items[:max_items]


def extract_native_bboxes(obs, info):
    native = {}
    if not isinstance(obs, dict):
        return native
    for modality in ("bbox_2d_tight", "bbox_2d_loose"):
        if modality in obs:
            native[modality] = {
                "boxes": jsonable(obs[modality]),
                "semantic_id_to_category": jsonable((info or {}).get(modality, {})),
            }
    return native


def draw_bbox_overlay(rgb, bboxes, output_path: Path | str):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    colors = [(255, 60, 60), (40, 180, 80), (60, 140, 255), (240, 190, 40), (210, 80, 220), (40, 210, 210)]
    for idx, item in enumerate(bboxes):
        x1, y1, x2, y2 = item["bbox_xyxy"]
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = item.get("category") or item.get("object_name")
        label = str(label).replace("_", " ")
        text_box = draw.textbbox((0, 0), label, font=font)
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=(0, 0, 0))
        draw.text((x1 + 3, ty + 2), label, fill=color, font=font)
    img.save(output_path)


def save_bbox_outputs(run, stem, output_dir, cam_pos, cam_ori, obs, info, draw_overlay=True):
    target_metadata = target_object_metadata(run)
    instance_bboxes = compute_instance_bboxes(obs, info, target_metadata)
    visible_instances = visible_instance_summary(obs, info)
    target_names = sorted(target_metadata)
    visible_target_names = {item["object_name"] for item in instance_bboxes}
    payload = {
        "run_id": stem,
        "camera": {
            "position": [float(v) for v in cam_pos],
            "orientation_xyzw": [float(v) for v in cam_ori],
        },
        "target_objects": target_names,
        "missing_target_objects": [name for name in target_names if name not in visible_target_names],
        "objects": instance_bboxes,
        "visible_instances": visible_instances,
        "native_omnigibson": extract_native_bboxes(obs, info),
    }
    json_path = output_dir / f"{stem}_after_bboxes.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {json_path} ({len(instance_bboxes)} visible target objects)")

    if draw_overlay and isinstance(obs, dict) and "rgb" in obs:
        overlay_path = output_dir / f"{stem}_after_bboxes.png"
        draw_bbox_overlay(obs["rgb"], instance_bboxes, overlay_path)
        print(f"  Saved: {overlay_path}")


def optimize_camera_for_visible_target_objects(env, viewer, run, initial_camera, min_pixels=8):
    target_metadata = target_object_metadata(run)
    if not target_metadata:
        return (*initial_camera, {"visible_count": 0, "target_count": 0, "candidates": []})

    camera_room = camera_room_for_run(run)
    candidates = official_camera_candidates(env, camera_room)
    for target_room in target_camera_rooms(run, target_metadata):
        if target_room == camera_room:
            continue
        for cam_pos, cam_ori, method in official_camera_candidates(env, target_room):
            candidates.append((cam_pos, cam_ori, f"{target_room}: {method}"))
    for nearby_room in nearby_camera_rooms(env, run, camera_room):
        for cam_pos, cam_ori, method in official_camera_candidates(env, nearby_room):
            candidates.append((cam_pos, cam_ori, f"{nearby_room}: {method}"))
    if not candidates:
        candidates = [initial_camera]

    best = None
    viable = []
    results = []
    for cam_pos, cam_ori, method in candidates:
        try:
            obs, info = get_camera_obs(viewer, cam_pos, cam_ori)
        except Exception as exc:
            results.append({"method": method, "error": repr(exc)})
            print(f"[vis] camera candidate failed ({method}): {exc}", flush=True)
            continue
        bboxes = compute_instance_bboxes(obs, info, target_metadata, min_pixels=min_pixels)
        visible_count = len({item["object_name"] for item in bboxes})
        visible_names = sorted({item["object_name"] for item in bboxes})
        pixel_count = sum(int(item.get("pixel_count", 0)) for item in bboxes)
        native_count = len(extract_native_bboxes(obs, info).get("bbox_2d_tight", {}).get("boxes", []))
        score = (visible_count, pixel_count, native_count)
        result = {
            "method": method,
            "visible_count": visible_count,
            "pixel_count": pixel_count,
            "native_bbox_count": native_count,
            "visible_objects": visible_names,
            "camera": {
                "position": [float(v) for v in cam_pos],
                "orientation_xyzw": [float(v) for v in cam_ori],
            },
        }
        results.append(result)
        viable.append((result, cam_pos, cam_ori, method))
        if best is None or score > best[0]:
            best = (score, cam_pos, cam_ori, method)

    if best is None:
        return (*initial_camera, {"visible_count": 0, "target_count": len(target_metadata), "candidates": results})
    _, cam_pos, cam_ori, method = best
    primary = next(
        item for item in viable
        if item[3] == method and np.allclose(item[1], cam_pos) and np.allclose(item[2], cam_ori)
    )
    selected = [primary[0]]
    covered = set(primary[0]["visible_objects"])
    remaining = [item for item in viable if item is not primary]
    while remaining and len(covered) < len(target_metadata):
        choice = max(
            remaining,
            key=lambda item: (
                len(set(item[0]["visible_objects"]) - covered),
                item[0]["pixel_count"],
            ),
        )
        new_names = set(choice[0]["visible_objects"]) - covered
        if not new_names:
            break
        selected.append(choice[0])
        covered.update(choice[0]["visible_objects"])
        remaining.remove(choice)
    diagnostics = {
        "visible_count": int(best[0][0]),
        "target_count": len(target_metadata),
        "aggregate_visible_count": len(covered),
        "selected_views": selected,
        "candidates": results,
    }
    return cam_pos, cam_ori, f"{method} optimized_visible={diagnostics['visible_count']}/{diagnostics['target_count']}", diagnostics


def save_additional_bbox_views(run, stem, output_dir, viewer, diagnostics, draw_overlay=True):
    selected = diagnostics.get("selected_views") or []
    if len(selected) <= 1:
        return
    bbox_path = output_dir / f"{stem}_after_bboxes.json"
    with bbox_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metadata = target_object_metadata(run)
    views = []
    aggregate = {item["object_name"]: item for item in payload.get("objects") or []}
    for idx, selected_view in enumerate(selected[1:], 1):
        camera = selected_view["camera"]
        cam_pos = camera["position"]
        cam_ori = camera["orientation_xyzw"]
        obs, info = capture_obs(viewer, cam_pos, cam_ori, output_dir / f"{stem}_after_view_{idx}.png")
        boxes = compute_instance_bboxes(obs, info, metadata)
        for item in boxes:
            aggregate.setdefault(item["object_name"], item)
        if draw_overlay and isinstance(obs, dict) and "rgb" in obs:
            draw_bbox_overlay(obs["rgb"], boxes, output_dir / f"{stem}_after_view_{idx}_bboxes.png")
        views.append({
            "index": idx,
            "method": selected_view.get("method"),
            "camera": camera,
            "objects": boxes,
        })
    payload["additional_views"] = views
    payload["all_visible_target_objects"] = sorted(aggregate)
    payload["missing_target_objects"] = sorted(set(payload.get("target_objects") or []) - set(aggregate))
    with bbox_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def apply_state_changes(env, run):
    te = run.get("task_environment", {}) or {}
    smoke_effect_count = 0
    for item in te.get("state_changed_objects", []):
        name = item.get("object_id") or item.get("object_name")
        if not name:
            continue
        obj = env.scene.object_registry("name", name, None)
        if obj is None:
            continue
        states = item.get("states") or {}
        if states.get("on_fire"):
            if object_states.OnFire not in obj.states:
                raise RuntimeError(f"{name!r} does not expose official OnFire state")
            if not obj.states[object_states.OnFire].set_value(True):
                raise RuntimeError(f"failed to restore official OnFire state for {name!r}")
            visual_effect = item.get("visual_effect") or {}
            if visual_effect.get("mode") == SMOKE_ONLY_ON_FIRE_MODE:
                configured = configure_on_fire_smoke_only(obj)
                if not configured.get("ok"):
                    raise RuntimeError(f"failed to restore smoke-only effect for {name!r}: {configured}")
                smoke_effect_count += 1
    return smoke_effect_count


def spawn_added_objects(env, run):
    spawned = []
    te = run.get("task_environment", {}) or {}
    for ao in te.get("added_objects", []):
        placement = ao.get("placement", {}) or {}
        if placement.get("mode") == "reused":
            continue
        name = object_name(ao)
        category = ao.get("category")
        if not name or not category:
            continue
        final_pose = ao.get("final_pose_before_warmup") or {}
        pos = (
            (ao.get("pose") or {}).get("position")
            or final_pose.get("position")
            or (placement.get("pose") or {}).get("position")
        )
        ori = (
            (ao.get("pose") or {}).get("orientation_xyzw")
            or final_pose.get("orientation_xyzw")
            or (placement.get("pose") or {}).get("orientation_xyzw")
            or [0, 0, 0, 1]
        )
        if not pos:
            continue
        obj = env.scene.object_registry("name", name, None)
        if obj is None:
            obj = DatasetObject(
                name=name,
                category=category,
                model=ao.get("model"),
                visual_only=True,
                prim_type=PrimType.RIGID,
            )
            env.scene.add_object(obj)
        else:
            # Env-C may intentionally retain an Env-B task object with the same
            # stable object id. Reuse that live object instead of duplicating it.
            obj.visible = True
        obj.set_position_orientation(
            position=th.tensor(pos, dtype=th.float32),
            orientation=th.tensor(ori, dtype=th.float32),
        )
        spawned.append(obj)
    for _ in range(20):
        og.sim.render()
    return spawned


def visualize_one(
    env,
    viewer,
    run_path: Path,
    output_dir: Path,
    reset_non_env_a: bool,
    save_bboxes: bool,
    draw_bboxes: bool,
    optimize_camera: bool,
    render_recorded_cameras: bool,
):
    with run_path.open() as f:
        run = json.load(f)
    enrich_final_poses(run)
    hide_online_objects(env)

    initial_camera = camera_for_run(env, run)
    cam_pos, cam_ori, method = initial_camera
    stem = run_path.stem
    camera_diagnostics = None

    spawn_added_objects(env, run)
    # Spawned fire carriers must exist before their OnFire state is restored.
    smoke_effect_count = apply_state_changes(env, run)
    if smoke_effect_count:
        # Flow density evolves with simulation time, not render-only frames.
        for _ in range(SMOKE_FLOW_WARMUP_STEPS):
            og.sim.step()
        for _ in range(SMOKE_FLOW_RENDER_WARMUP_FRAMES):
            og.sim.render()
    if optimize_camera and save_bboxes:
        cam_pos, cam_ori, method, camera_diagnostics = optimize_camera_for_visible_target_objects(
            env, viewer, run, initial_camera,
        )

    if render_recorded_cameras:
        manifest = save_recorded_camera_views(
            run, stem, output_dir, viewer, draw_bboxes=draw_bboxes
        )
        print(
            f"[vis] {stem}: rendered {manifest['num_rendered_views']} recorded camera views",
            flush=True,
        )

    print(f"[vis] {stem}: camera({method})", flush=True)
    if save_bboxes:
        obs, info = capture_obs(viewer, cam_pos, cam_ori, output_dir / f"{stem}_after.png")
        save_bbox_outputs(
            run,
            stem,
            output_dir,
            cam_pos,
            cam_ori,
            obs,
            info,
            draw_overlay=draw_bboxes,
        )
        if camera_diagnostics is not None:
            bbox_json_path = output_dir / f"{stem}_after_bboxes.json"
            with bbox_json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["camera_selection"] = camera_diagnostics
            with bbox_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            save_additional_bbox_views(
                run,
                stem,
                output_dir,
                viewer,
                camera_diagnostics,
                draw_overlay=draw_bboxes,
            )
    else:
        capture(viewer, cam_pos, cam_ori, str(output_dir / f"{stem}_after.png"))

    hide_online_objects(env)
    capture_obs(viewer, cam_pos, cam_ori, output_dir / f"{stem}_before.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--robot", default="fetch")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-reset-non-env-a",
        action="store_true",
        help="Reuse the scene for Env-B/C visualization. Faster, but can hide reset bugs.",
    )
    parser.add_argument(
        "--no-bboxes",
        action="store_true",
        help="Only save RGB images. By default, also save per-object 2D bbox JSON from OmniGibson camera observations.",
    )
    parser.add_argument(
        "--no-bbox-overlay",
        action="store_true",
        help="Skip saving *_after_bboxes.png overlay images.",
    )
    parser.add_argument(
        "--no-camera-optimize",
        action="store_true",
        help="Use the first official room camera without checking whether target objects are visible.",
    )
    parser.add_argument(
        "--single-camera-only",
        action="store_true",
        help="Skip the recorded global and primary robot camera views and only render the legacy pair.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = run_files(input_dir)
    if args.limit is not None:
        files = files[: args.limit]
    print(f"[vis] {len(files)} files from {input_dir}", flush=True)
    if not files:
        hard_exit(0)

    robot_name = str(args.robot).lower()
    robot_model = None if robot_name in {"none", "null", ""} else robot_name
    # This renderer captures through OmniGibson's official viewer camera. Robot
    # vision sensors are unused and can overflow Replicator graphs in large scenes.
    env = create_env(
        scene_model=args.scene,
        robot_model=robot_model,
        robot_obs_modalities=[],
    )
    if robot_model is not None:
        stabilize_robot_spawn(env, seed=0)
    reset_scene(env)

    viewer = og.sim.viewer_camera
    modalities = ["rgb"]
    if not args.no_bboxes:
        # Bboxes are derived from instance pixels. Native bbox annotators add
        # redundant Replicator graphs and crash consistently in Ihlen_1_int.
        modalities.extend(["seg_instance", "seg_semantic"])
    if hasattr(viewer, "add_modality"):
        for modality in modalities:
            try:
                viewer.add_modality(modality)
            except Exception as exc:
                print(f"[vis] warning: failed to add camera modality {modality}: {exc}", flush=True)

    for idx, path in enumerate(files, 1):
        print(f"[vis] {idx}/{len(files)} {path.name}", flush=True)
        visualize_one(
            env,
            viewer,
            path,
            output_dir,
            reset_non_env_a=not args.no_reset_non_env_a,
            save_bboxes=not args.no_bboxes,
            draw_bboxes=not args.no_bbox_overlay,
            optimize_camera=not args.no_camera_optimize,
            render_recorded_cameras=not args.single_camera_only,
        )

    # Kit can segfault while tearing down SyntheticData graphs in og.clear().
    # The process is dedicated to one scene, so exit directly after flushing.
    hard_exit(0)


if __name__ == "__main__":
    main()
