"""
Generate before/after room-level camera images for Env-A sample.
Uses og.sim.viewer_camera at global camera position, looking at room center.

Usage (after conda activate behavior):
    export CUDA_VISIBLE_DEVICES=0
    python code/visualize_env_a.py \
        --run-file code/outputs/online_deltasg_2/online_env_a_0002.json \
        --output-dir code/outputs/vis_captures
"""
import argparse, json, math, os, sys
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True
gm.GUI_VIEWPORT_ONLY = True

import numpy as np
from PIL import Image
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.utils.constants import PrimType
import torch as th

from api import get_all_scene_objects


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def save_rgb(rgb, path):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")


def float_list(values):
    return [float(v) for v in values]


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def room_corners(env, room_name):
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
        if room_name in rooms:
            positions.append(pos)
    if len(positions) < 2:
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


def corner_camera(corner, opposite, v_angle=30.0, inward=0.3, height=2.4):
    diag = np.array([opposite[0] - corner[0], opposite[1] - corner[1]], dtype=np.float32)
    diag_len = float(np.linalg.norm(diag))
    inward = min(inward, diag_len * 0.1)
    cam_pos = np.array([
        corner[0] + (diag[0] / diag_len) * inward,
        corner[1] + (diag[1] / diag_len) * inward,
        height,
    ], dtype=np.float32)
    diag_angle = math.degrees(math.atan2(diag[1], diag[0]))
    return cam_pos, yaw_pitch_quat(diag_angle - 90.0, v_angle)


def wall_center_camera(corners, v_angle=45.0, inward=0.2, height=2.2):
    walls = [
        (corners["SW"], corners["SE"]),
        (corners["SE"], corners["NE"]),
        (corners["NE"], corners["NW"]),
        (corners["NW"], corners["SW"]),
    ]
    c1, c2 = min(walls, key=lambda wall: np.linalg.norm(wall[1][:2] - wall[0][:2]))
    wall_center = np.array([(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5], dtype=np.float32)
    wall_dir = np.array([c2[0] - c1[0], c2[1] - c1[1]], dtype=np.float32)
    wall_dir = wall_dir / np.linalg.norm(wall_dir)
    normal = np.array([-wall_dir[1], wall_dir[0]], dtype=np.float32)
    cam_pos = np.array([wall_center[0] + normal[0] * inward, wall_center[1] + normal[1] * inward, height], dtype=np.float32)
    normal_angle = math.degrees(math.atan2(normal[1], normal[0]))
    return cam_pos, yaw_pitch_quat(normal_angle - 90.0, v_angle)


def yaw_pitch_quat(yaw_deg, v_angle):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(90.0 - v_angle)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
    q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
    return quat_multiply(q_yaw, q_pitch)


def official_camera_pose(env, room_name):
    configs = {"living_room_0": ("SW", 30), "bedroom_0": ("SE", 45), "bathroom_0": ("NE", 45)}
    opposite = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}
    corners = room_corners(env, room_name)
    if corners is None:
        return None, None, None
    diag = float(np.linalg.norm(corners["NE"][:2] - corners["SW"][:2]))
    if diag <= 3.0:
        cam_pos, ori = wall_center_camera(corners)
        return cam_pos, ori, f"wall-center diag={diag:.2f}"
    corner_name, v_angle = configs.get(room_name, ("SW", 30))
    cam_pos, ori = corner_camera(corners[corner_name], corners[opposite[corner_name]], v_angle=v_angle)
    return cam_pos, ori, f"corner {corner_name} diag={diag:.2f} v={v_angle}"


def capture(viewer, cam_pos, cam_ori, path):
    """Move viewer camera and capture."""
    viewer.set_position_orientation(
        position=np.array(cam_pos, dtype=np.float32),
        orientation=np.array(cam_ori, dtype=np.float32),
    )
    for _ in range(10):
        og.sim.step()
    obs, _ = viewer.get_obs()
    if isinstance(obs, dict) and 'rgb' in obs:
        save_rgb(obs['rgb'], path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-file", required=True)
    parser.add_argument("--output-dir", default="code/outputs/vis_captures")
    parser.add_argument("--scene", default="Rs_int")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.run_file) as f:
        run = json.load(f)

    te = run.get("task_environment", {})
    added_objects = te.get("added_objects", [])
    task = te.get("task", {})
    instruction = task.get("instruction", "")
    validation_by_name = {
        item.get("object_name") or item.get("object_id") or item.get("name"): item
        for item in (run.get("validation", {}) or {}).get("created_objects", [])
    }
    for ao in added_objects:
        object_name = ao.get("object_name") or ao.get("object_id") or ao.get("name")
        if object_name and not ao.get("final_pose_before_warmup"):
            final_pose = (validation_by_name.get(object_name) or {}).get("final_pose_before_warmup")
            if final_pose:
                ao["final_pose_before_warmup"] = final_pose

    if not added_objects:
        print("No spawned objects")
        return

    print(f"Task: {instruction}")

    from api import create_env
    env = create_env(scene_model=args.scene, robot_model=None)
    env.reset()
    for _ in range(30):
        og.sim.step()

    target_room = task.get("target_room")
    cam_pos, cam_ori, method = official_camera_pose(env, target_room)
    if cam_pos is None:
        positions = [float_list(ao.get("pose",{}).get("position",[0,0,0])) for ao in added_objects]
        center = np.mean(positions, axis=0)
        cam_pos = np.array([float(center[0]) + 2.0, float(center[1]) + 2.0, 2.5], dtype=np.float32)
        cam_ori = yaw_pitch_quat(45.0, 45.0)
        method = "fallback"

    print(f"  camera({method}): {[round(float(v),2) for v in cam_pos]}")

    # Setup viewer camera
    viewer = og.sim.viewer_camera
    if hasattr(viewer, "add_modality"):
        try:
            viewer.add_modality("rgb")
        except Exception:
            pass

    fname = Path(args.run_file).stem

    # BEFORE
    capture(viewer, cam_pos, cam_ori, f"{args.output_dir}/{fname}_before.png")
    print(f"  BEFORE captured")

    spawned = []

    # Spawn at the final physics-settled pose when available. The planned
    # placement pose can differ from the actual OnTop/Inside sampled pose.
    for ao in added_objects:
        placement = ao.get("placement", {}) or {}
        object_name = ao.get("object_name") or ao.get("object_id") or ao.get("name")
        if placement.get("mode") == "reused":
            print(f"  Skip reused: {object_name} ({ao['category']})")
            continue
        final_pose = ao.get("final_pose_before_warmup") or {}
        pos = final_pose.get("position") or placement.get("pose", {}).get("position") or ao.get("pose", {}).get("position", [0, 0, 0])
        ori = final_pose.get("orientation_xyzw") or placement.get("pose", {}).get("orientation_xyzw") or ao.get("pose", {}).get("orientation_xyzw", [0, 0, 0, 1])
        obj = DatasetObject(name=object_name, category=ao["category"],
                            prim_type=PrimType.RIGID)
        env.scene.add_object(obj)
        obj.set_position_orientation(
            position=th.tensor(pos, dtype=th.float32),
            orientation=th.tensor(ori, dtype=th.float32),
        )
        try:
            obj.keep_still()
        except Exception:
            pass
        spawned.append(obj)
        print(f"  Spawned: {object_name} ({ao['category']}) at z={pos[2]:.2f}")

    for _ in range(20):
        og.sim.step()

    # AFTER
    capture(viewer, cam_pos, cam_ori, f"{args.output_dir}/{fname}_after.png")
    print(f"  AFTER captured")

    print(f"Done. {fname}_before.png vs {fname}_after.png")
    og.clear()
    hard_exit(0)


if __name__ == "__main__":
    main()
