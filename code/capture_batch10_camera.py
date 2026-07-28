"""Capture robot + global camera views for each DeltaSG sample.
Camera placement per camera_config_guide.md:
- 3D objects → seg_map pixel bbox → world corners
- Corner-based placement with per-room config (corner, v_angle)
- h_offset=0, inward=0.3m, height=2.4m
"""
import os, sys, json, math
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True
gm.GUI_VIEWPORT_ONLY = True

import numpy as np
from PIL import Image
import omnigibson as og
import torch as th
from omnigibson.objects import DatasetObject
from omnigibson.utils.constants import PrimType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import create_env, get_all_scene_objects
from online_deltasg import OnlineDeltaSGEngine


# Per-room config from camera_config_guide.md
# Per-room config from camera_config_guide.md (for Rs_int)
# For other scenes, corners are computed dynamically from 3D objects → seg_map
ROOM_CAMERA_CONFIGS = {
    "living_room_0": ("SW", 30),
    "bedroom_0": ("SE", 45),
    "bathroom_0": ("NE", 45),
    # kitchen_0 excluded — top cabinets block all angles
}
# Default fallback for rooms not in ROOM_CAMERA_CONFIGS
DEFAULT_CORNER = "SW"
DEFAULT_V_ANGLE = 30
OPPOSITE_MAP = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}
INWARD, HEIGHT = 0.3, 2.4


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def save_rgb(rgb, path):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


def compute_room_corners(env, room_name):
    """Get SW/NE/NW/SE world corners via 3D objects → seg_map pixel mapping."""
    seg_map = env.scene.seg_map
    positions = []
    for obj in get_all_scene_objects(env.scene):
        try:
            pos, _ = obj.get_position_orientation()
            pos = np.array(pos)
        except Exception:
            continue
        rooms = getattr(obj, "in_rooms", None)
        if not rooms:
            continue
        if isinstance(rooms, str):
            rooms = [rooms]
        if room_name not in rooms:
            continue
        positions.append(pos)
    if not positions:
        return None
    pixels = []
    for pos in positions:
        px = seg_map.world_to_map(th.tensor([pos[0], pos[1]], dtype=th.float32))
        pixels.append(px.numpy())
    pixels = np.stack(pixels)
    px_min = pixels.min(axis=0).astype(int)
    px_max = pixels.max(axis=0).astype(int)
    map_h, map_w = seg_map.room_ins_map.shape
    px_min = np.clip(px_min, 0, [map_w - 1, map_h - 1])
    px_max = np.clip(px_max, 0, [map_w - 1, map_h - 1])
    sw = seg_map.map_to_world(th.tensor([px_min[0], px_min[1]], dtype=th.float32)).cpu().numpy()
    ne = seg_map.map_to_world(th.tensor([px_max[0], px_max[1]], dtype=th.float32)).cpu().numpy()
    return {"SW": sw, "NE": ne, "NW": np.array([sw[0], ne[1]]), "SE": np.array([ne[0], sw[1]])}


def compute_corner_camera(corner, opposite, v_angle=30.0):
    """Camera at room corner, looking inward along diagonal. h_offset=0."""
    diagonal = np.array([opposite[0] - corner[0], opposite[1] - corner[1]])
    diag_len = np.sqrt(diagonal[0]**2 + diagonal[1]**2)
    # Use smaller inward for small rooms
    inward = min(INWARD, diag_len * 0.1)
    cam_pos = np.array([
        corner[0] + (diagonal[0] / diag_len) * inward,
        corner[1] + (diagonal[1] / diag_len) * inward,
        HEIGHT,
    ], dtype=np.float32)
    diag_angle = math.degrees(math.atan2(diagonal[1], diagonal[0]))
    yaw = math.radians(diag_angle - 90.0)  # h_offset=0
    pitch = math.radians(90.0 - v_angle)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
    q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
    orientation = quat_multiply(q_yaw, q_pitch)
    return cam_pos, orientation


def capture_at_pose(viewer, position, orientation, path):
    viewer.set_position_orientation(
        position=np.array(position, dtype=np.float32),
        orientation=np.array(orientation, dtype=np.float32),
    )
    for _ in range(5):
        og.sim.step()
    obs, _ = viewer.get_obs()
    if isinstance(obs, dict) and "rgb" in obs:
        save_rgb(obs["rgb"], path)
        return True
    return False


def main():
    data_dir = Path("/home2/daiyang/BEHAVIOR/code/outputs/Rs_int_test")
    out_dir = Path("/home2/daiyang/BEHAVIOR/code/outputs/Rs_int_camera")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect scene_model from dataset
    scene_model = "Rs_int"
    dataset_path = data_dir / "dataset.json"
    if dataset_path.exists():
        import json
        with open(dataset_path) as f:
            ds = json.load(f)
        scene_model = ds.get("scene", scene_model)

    print(f"=== Camera Capture for {scene_model} ===")
    env = create_env(scene_model=scene_model, robot_model="fetch")
    env.reset()
    for _ in range(30):
        og.sim.step()

    viewer = og.sim.viewer_camera
    try:
        viewer.add_modality("rgb")
    except Exception:
        pass

    robot = env.robots[0]

    # Pre-compute room corners for all rooms in the scene
    room_corners_cache = {}
    all_rooms = set()
    for obj in get_all_scene_objects(env.scene):
        rooms = getattr(obj, "in_rooms", None)
        if not rooms:
            continue
        if isinstance(rooms, str):
            rooms = [rooms]
        all_rooms.update(rooms)

    for room_name in sorted(all_rooms):
        corners = compute_room_corners(env, room_name)
        if corners is None:
            continue
        if np.allclose(corners["SW"][:2], corners["NE"][:2]):
            continue  # single-point room, skip
        room_corners_cache[room_name] = corners
        # Use known config or default
        corner_name, v_angle = ROOM_CAMERA_CONFIGS.get(room_name, (DEFAULT_CORNER, DEFAULT_V_ANGLE))
        cam_pos, _ = compute_corner_camera(corners[corner_name], corners[OPPOSITE_MAP[corner_name]], v_angle)
        print(f"  {room_name}: {corner_name} v={v_angle}° cam={cam_pos.round(2)}")

    samples = sorted(data_dir.glob("online_env_a_*.json"))
    for fpath in samples:
        with open(fpath) as f:
            data = json.load(f)
        te = data.get("task_environment", {})
        added = te.get("added_objects", [])
        task = te.get("task", {})
        target_room = task.get("target_room", "")
        robot_room = te.get("robot", {}).get("initial_room", "")
        rid = fpath.stem
        sample_dir = out_dir / rid
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Determine rooms to cover (only those with camera configs)
        rooms_to_cover = []
        for room_id in (target_room, robot_room):
            if room_id and room_id in room_corners_cache and room_id not in rooms_to_cover:
                rooms_to_cover.append(room_id)

        # 1. Robot cameras (before)
        robot_obs, _ = robot.get_obs()
        if isinstance(robot_obs, dict):
            for cam_key, cam_obs in robot_obs.items():
                if isinstance(cam_obs, dict) and "rgb" in cam_obs:
                    if "eyes" in cam_key.lower():
                        prefix = "robot_eyes"
                    elif "eef" in cam_key.lower():
                        prefix = "robot_eef"
                    else:
                        prefix = "robot_" + cam_key.replace(":", "_").replace("/", "_")
                    save_rgb(cam_obs["rgb"], str(sample_dir / f"{prefix}_before.png"))

        # 2. Global cameras (before)
        for room_id in rooms_to_cover:
            corners = room_corners_cache.get(room_id)
            if corners is None:
                continue
            corner_name, v_angle = ROOM_CAMERA_CONFIGS.get(room_id, (DEFAULT_CORNER, DEFAULT_V_ANGLE))
            cam_pos, cam_ori = compute_corner_camera(
                corners[corner_name], corners[OPPOSITE_MAP[corner_name]], v_angle
            )
            cam_id = f"global_{room_id}"
            capture_at_pose(viewer, cam_pos, cam_ori, str(sample_dir / f"{cam_id}_before.png"))

        # 3. Spawn added objects
        spawned = []
        for ao in added:
            pl = ao.get("placement", {})
            if pl.get("mode") == "reused":
                continue
            pos = pl.get("pose", {}).get("position") or ao.get("pose", {}).get("position", [0, 0, 0])
            ori = pl.get("pose", {}).get("orientation_xyzw") or ao.get("pose", {}).get("orientation_xyzw", [0, 0, 0, 1])
            try:
                obj = DatasetObject(name=ao["object_name"], category=ao["category"], prim_type=PrimType.RIGID)
                env.scene.add_object(obj)
                obj.set_position_orientation(
                    position=th.tensor(pos, dtype=th.float32),
                    orientation=th.tensor(ori, dtype=th.float32),
                )
                spawned.append(obj)
            except Exception as e:
                print(f"  Spawn failed: {ao['object_name']} - {e}")

        for _ in range(10):
            og.sim.step()

        # 4. Robot cameras (after)
        robot_obs, _ = robot.get_obs()
        if isinstance(robot_obs, dict):
            for cam_key, cam_obs in robot_obs.items():
                if isinstance(cam_obs, dict) and "rgb" in cam_obs:
                    if "eyes" in cam_key.lower():
                        prefix = "robot_eyes"
                    elif "eef" in cam_key.lower():
                        prefix = "robot_eef"
                    else:
                        prefix = "robot_" + cam_key.replace(":", "_").replace("/", "_")
                    save_rgb(cam_obs["rgb"], str(sample_dir / f"{prefix}_after.png"))

        # 5. Global cameras (after)
        for room_id in rooms_to_cover:
            corners = room_corners_cache.get(room_id)
            if corners is None:
                continue
            corner_name, v_angle = ROOM_CAMERA_CONFIGS.get(room_id, (DEFAULT_CORNER, DEFAULT_V_ANGLE))
            cam_pos, cam_ori = compute_corner_camera(
                corners[corner_name], corners[OPPOSITE_MAP[corner_name]], v_angle
            )
            cam_id = f"global_{room_id}"
            capture_at_pose(viewer, cam_pos, cam_ori, str(sample_dir / f"{cam_id}_after.png"))

        for obj in spawned:
            try:
                env.scene.remove_object(obj)
            except Exception:
                pass

        rooms_str = ", ".join(rooms_to_cover)
        print(f"  {rid}: {task.get('instruction', '?')[:80]} [{rooms_str}] ({len(spawned)} spawned)")

    print(f"Done: {out_dir}")
    og.clear()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()