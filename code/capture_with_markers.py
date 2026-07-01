"""Capture camera with colored markers above spawned objects."""
import os, sys, math, json
os.environ['OMNIGIBSON_HEADLESS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True; gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True; gm.RENDER_VIEWER_CAMERA = True; gm.GUI_VIEWPORT_ONLY = True
import omnigibson as og
sys.path.insert(0, '/home2/daiyang/BEHAVIOR/code')
from api import create_env
import numpy as np
from PIL import Image
import torch as th

V_DEG = 20

# Bright marker colors (RGB) for different objects
MARKER_COLORS = [
    (1.0, 0.0, 0.0),   # red
    (0.0, 1.0, 0.0),   # green
    (0.0, 0.0, 1.0),   # blue
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 1.0),   # magenta
    (0.0, 1.0, 1.0),   # cyan
    (1.0, 0.5, 0.0),   # orange
    (0.5, 0.0, 1.0),   # purple
]

def cam_pose(env):
    pos_list = []
    for obj in env.scene.objects:
        try:
            rooms = getattr(obj, "in_rooms", None)
            if rooms and "living_room_0" in (rooms if isinstance(rooms, list) else [rooms]):
                p, _ = obj.get_position_orientation()
                pos_list.append(np.array(p))
        except: pass
    s = np.stack(pos_list); bmin = s.min(axis=0); bmax = s.max(axis=0)
    sw = np.array([bmin[0], bmin[1]]); ne = np.array([bmax[0], bmax[1]])
    diagonal = ne - sw; da = math.degrees(math.atan2(diagonal[1], diagonal[0]))
    base_yaw = da - 90
    dl = math.sqrt(diagonal[0]**2 + diagonal[1]**2)
    cam_pos = np.array([sw[0] + diagonal[0]/dl*0.3, sw[1] + diagonal[1]/dl*0.3, 2.4], dtype=np.float32)
    yaw = math.radians(base_yaw); pitch = math.radians(90 - V_DEG)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
    q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
    x1,y1,z1,w1=q_yaw; x2,y2,z2,w2=q_pitch
    ori = np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                    w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2], dtype=np.float32)
    return cam_pos, ori

def capture(viewer, cam_pos, cam_ori, path):
    viewer.set_position_orientation(position=cam_pos, orientation=cam_ori)
    for _ in range(5): og.sim.step()
    obs, _ = viewer.get_obs()
    if isinstance(obs, dict) and 'rgb' in obs:
        Image.fromarray(np.asarray(obs['rgb'])[..., :3].astype(np.uint8)).save(path)
        return True
    return False

batch = '/home2/daiyang/BEHAVIOR/code/outputs/batch10'
out = '/home2/daiyang/BEHAVIOR/code/outputs/batch10_camera'
os.makedirs(out, exist_ok=True)

from omnigibson.objects import DatasetObject, PrimitiveObject
from omnigibson.utils.constants import PrimType

env = create_env(scene_model="Rs_int", robot_model="fetch")
env.reset()
for _ in range(30): og.sim.step()

viewer = og.sim.viewer_camera
try: viewer.add_modality("rgb")
except: pass

for obj in env.scene.objects:
    try: obj.keep_still()
    except: pass

for fname in sorted(os.listdir(batch)):
    if not fname.startswith('online_env_a_') or 'rejected' in fname:
        continue
    with open(os.path.join(batch, fname)) as fh:
        data = json.load(fh)
    te = data.get('task_environment', {})
    added = te.get('added_objects', [])
    if not added:
        continue
    task = te.get('task', {})
    rid = fname.replace('.json', '')
    print(f"{rid}: {task.get('instruction', '?')}", flush=True)
    tdir = f'{out}/{rid}'
    os.makedirs(tdir, exist_ok=True)

    cp, co = cam_pose(env)
    capture(viewer, cp, co, f'{tdir}/before.png')

    spawned = []
    markers = []
    color_idx = 0
    for ao in added:
        pl = ao.get('placement', {})
        if pl.get('mode') == 'reused':
            continue
        pos = pl.get('pose', {}).get('position') or ao.get('pose', {}).get('position', [0, 0, 0])
        ori = pl.get('pose', {}).get('orientation_xyzw') or ao.get('pose', {}).get('orientation_xyzw', [0, 0, 0, 1])

        # Spawn the object
        obj = DatasetObject(name=ao['object_name'], category=ao['category'], prim_type=PrimType.RIGID)
        env.scene.add_object(obj)
        obj.set_position_orientation(
            position=th.tensor(pos, dtype=th.float32),
            orientation=th.tensor(ori, dtype=th.float32),
        )
        obj.keep_still()
        spawned.append(obj)

        # Spawn a colored marker cube above the object
        marker_name = f"marker_{ao['object_name']}"
        marker_z = float(pos[2]) + 0.3  # 30cm above object
        marker_pos = [float(pos[0]), float(pos[1]), marker_z]

        color = MARKER_COLORS[color_idx % len(MARKER_COLORS)]
        color_idx += 1

        marker = PrimitiveObject(
            name=marker_name,
            primitive_type="Cube",
            prim_path=f"/World/{marker_name}",
            category="marker",
            scale=[0.08, 0.08, 0.08],  # 8cm cube
            rgba=color + (1.0,),  # RGBA
        )
        env.scene.add_object(marker)
        marker.set_position_orientation(
            position=th.tensor(marker_pos, dtype=th.float32),
            orientation=th.tensor([0, 0, 0, 1], dtype=th.float32),
        )
        marker.keep_still()
        markers.append(marker)

        print(f"  {ao['object_name']}: z={pos[2]:.2f} marker={color} at z={marker_z:.2f}", flush=True)

    for _ in range(5): og.sim.step()

    cp, co = cam_pose(env)
    capture(viewer, cp, co, f'{tdir}/after.png')

    # Also capture a marked-only version (without the objects, just markers)
    # Remove objects, keep markers
    for obj in spawned:
        try: env.scene.remove_object(obj)
        except: pass
    for _ in range(3): og.sim.step()
    cp, co = cam_pose(env)
    capture(viewer, cp, co, f'{tdir}/marked.png')

    # Clean up markers
    for m in markers:
        try: env.scene.remove_object(m)
        except: pass

    print(f"  done", flush=True)

print(f"Output: {out}", flush=True)
sys.stdout.flush()
os._exit(0)