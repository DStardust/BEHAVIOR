"""Capture all room cameras for a scene, trying all 4 corners.
Usage: python code/capture_all_rooms.py --scene Benevolence_0_int
Output: code/outputs/{scene}_all_rooms/{room_name}_{corner}.png
"""
import os, sys, math, argparse
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True; gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True; gm.RENDER_VIEWER_CAMERA = True; gm.GUI_VIEWPORT_ONLY = True

import numpy as np
from PIL import Image
import omnigibson as og
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import create_env, get_all_scene_objects


def quat_multiply(q1, q2):
    x1,y1,z1,w1=q1; x2,y2,z2,w2=q2
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2], dtype=np.float32)


def compute_room_corners(env, room_name):
    """Get SW/NE/NW/SE world corners via 3D objects → seg_map pixel mapping."""
    seg_map = env.scene.seg_map
    positions = []
    for obj in get_all_scene_objects(env.scene):
        try: pos, _ = obj.get_position_orientation(); pos = np.array(pos)
        except: continue
        rooms = getattr(obj, "in_rooms", None)
        if not rooms: continue
        if isinstance(rooms, str): rooms = [rooms]
        if room_name not in rooms: continue
        positions.append(pos)
    if len(positions) < 2: return None
    pixels = []
    for pos in positions:
        px = seg_map.world_to_map(th.tensor([pos[0], pos[1]], dtype=th.float32))
        pixels.append(px.numpy())
    pixels = np.stack(pixels)
    px_min = pixels.min(axis=0).astype(int); px_max = pixels.max(axis=0).astype(int)
    map_h, map_w = seg_map.room_ins_map.shape
    px_min = np.clip(px_min, 0, [map_w-1, map_h-1]); px_max = np.clip(px_max, 0, [map_w-1, map_h-1])
    sw = seg_map.map_to_world(th.tensor([px_min[0], px_min[1]], dtype=th.float32)).cpu().numpy()
    ne = seg_map.map_to_world(th.tensor([px_max[0], px_max[1]], dtype=th.float32)).cpu().numpy()
    if np.allclose(sw[:2], ne[:2]): return None
    return {"SW": sw, "NE": ne, "NW": np.array([sw[0], ne[1]]), "SE": np.array([ne[0], sw[1]])}


def cam_at_corner(corner, opposite, v_angle=30.0, inward=0.3, height=2.4):
    """Camera at corner, looking inward along diagonal. h_offset=0. For large rooms."""
    diag = np.array([opposite[0]-corner[0], opposite[1]-corner[1]])
    diag_len = np.sqrt(diag[0]**2+diag[1]**2)
    inward = min(inward, diag_len * 0.1)
    cam_pos = np.array([corner[0]+(diag[0]/diag_len)*inward, corner[1]+(diag[1]/diag_len)*inward, height], dtype=np.float32)
    diag_angle = math.degrees(math.atan2(diag[1], diag[0]))
    yaw = math.radians(diag_angle - 90.0)
    pitch = math.radians(90.0 - v_angle)
    cp,sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy,sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    q_pitch = np.array([sp,0,0,cp], dtype=np.float32); q_yaw = np.array([0,0,sy,cy], dtype=np.float32)
    return cam_pos, quat_multiply(q_yaw, q_pitch)


def cam_at_wall_center(c1, c2, v_angle=45.0, inward=0.2, height=2.2):
    """Camera at center of short wall, looking inward. For small rooms."""
    wall_center = np.array([(c1[0]+c2[0])/2, (c1[1]+c2[1])/2])
    wall_dir = np.array([c2[0]-c1[0], c2[1]-c1[1]])
    wall_len = np.sqrt(wall_dir[0]**2+wall_dir[1]**2)
    if wall_len < 1e-6: return None, None
    wall_dir = wall_dir / wall_len
    normal = np.array([-wall_dir[1], wall_dir[0]])
    cam_pos = np.array([wall_center[0] + normal[0]*inward, wall_center[1] + normal[1]*inward, height], dtype=np.float32)
    normal_angle = math.degrees(math.atan2(normal[1], normal[0]))
    yaw = math.radians(normal_angle - 90.0)
    pitch = math.radians(90.0 - v_angle)
    cp,sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy,sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    q_pitch = np.array([sp,0,0,cp], dtype=np.float32); q_yaw = np.array([0,0,sy,cy], dtype=np.float32)
    return cam_pos, quat_multiply(q_yaw, q_pitch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Rs_int")
    args = parser.parse_args()

    out_dir = Path(f"code/outputs/{args.scene}_all_rooms")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {args.scene}: capturing all rooms ===")
    env = create_env(scene_model=args.scene, robot_model="fetch")
    env.reset()
    for _ in range(30): og.sim.step()

    viewer = og.sim.viewer_camera
    try: viewer.add_modality("rgb")
    except: pass

    # Find all rooms
    all_rooms = set()
    for obj in get_all_scene_objects(env.scene):
        rooms = getattr(obj, "in_rooms", None)
        if not rooms: continue
        if isinstance(rooms, str): rooms = [rooms]
        all_rooms.update(rooms)

    OPPOSITE = {"SW": "NE", "SE": "NW", "NW": "SE", "NE": "SW"}

    for room_name in sorted(all_rooms):
        corners = compute_room_corners(env, room_name)
        if corners is None:
            print(f"  {room_name}: skip (not enough objects)")
            continue

        diag = np.linalg.norm(corners["NE"][:2] - corners["SW"][:2])
        is_small = diag <= 3.0
        method = "wall" if is_small else "corner"

        if is_small:
            # Small room: wall-center cameras (v_angle=45°, height=2.2m)
            walls = [
                ("SW", "SE", corners["SW"], corners["SE"]),
                ("SE", "NE", corners["SE"], corners["NE"]),
                ("NE", "NW", corners["NE"], corners["NW"]),
                ("NW", "SW", corners["NW"], corners["SW"]),
            ]
            shortest = min(walls, key=lambda w: np.linalg.norm(w[3][:2] - w[2][:2]))
            w1_name, w2_name, c1, c2 = shortest
            cam_pos, ori = cam_at_wall_center(c1, c2)
            if cam_pos is not None:
                viewer.set_position_orientation(position=cam_pos, orientation=ori)
                for _ in range(5): og.sim.step()
                obs, _ = viewer.get_obs()
                if isinstance(obs, dict) and 'rgb' in obs:
                    arr = np.asarray(obs['rgb'])[...,:3].astype(np.uint8)
                    path = out_dir / f"{room_name}_wall_{w1_name}-{w2_name}.png"
                    Image.fromarray(arr).save(str(path))
            print(f"  {room_name}: {method} (diag={diag:.1f}m, v=45°, h=2.2m)")
        else:
            # Large room: corner cameras (v_angle=30°, height=2.4m)
            for corner_name in ["SW", "SE", "NW", "NE"]:
                corner = corners[corner_name]
                opposite = corners[OPPOSITE[corner_name]]
                cam_pos, ori = cam_at_corner(corner, opposite)
                viewer.set_position_orientation(position=cam_pos, orientation=ori)
                for _ in range(5): og.sim.step()
                obs, _ = viewer.get_obs()
                if isinstance(obs, dict) and 'rgb' in obs:
                    arr = np.asarray(obs['rgb'])[...,:3].astype(np.uint8)
                    path = out_dir / f"{room_name}_{corner_name}.png"
                    Image.fromarray(arr).save(str(path))
            print(f"  {room_name}: {method} (diag={diag:.1f}m, v=30°, h=2.4m)")

    print(f"Done: {out_dir}")
    og.clear()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()