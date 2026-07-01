"""Capture camera for any scene/room - uses target_room from task data."""
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
import torch as th
from PIL import Image, ImageDraw, ImageFont

V_DEG = 20
FOCAL_LENGTH = 17.0
HORIZONTAL_APERTURE = 20.995
IMG_W, IMG_H = 1280, 720

def cam_pose_for_room(env, target_room):
    """Find target room's object bbox SW corner and compute camera pose."""
    positions = []
    for obj in env.scene.objects:
        try:
            rooms = getattr(obj, "in_rooms", None)
            if rooms and target_room in (rooms if isinstance(rooms, list) else [rooms]):
                p, _ = obj.get_position_orientation()
                positions.append(np.array(p))
        except: pass
    if not positions:
        return None, None
    s = np.stack(positions); bmin = s.min(axis=0); bmax = s.max(axis=0)
    sw = np.array([bmin[0], bmin[1]]); ne = np.array([bmax[0], bmax[1]])
    diagonal = ne - sw
    diag_angle = math.degrees(math.atan2(diagonal[1], diagonal[0]))
    base_yaw = diag_angle - 90.0
    diag_len = math.sqrt(diagonal[0]**2 + diagonal[1]**2)
    cam_pos = np.array([sw[0] + diagonal[0]/diag_len*0.3, sw[1] + diagonal[1]/diag_len*0.3, 2.4], dtype=np.float32)
    yaw = math.radians(base_yaw); pitch = math.radians(90 - V_DEG)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)
    q_yaw = np.array([0, 0, sy, cy], dtype=np.float32)
    x1,y1,z1,w1=q_yaw; x2,y2,z2,w2=q_pitch
    ori = np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                    w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2], dtype=np.float32)
    return cam_pos, ori

def build_view_matrix(cam_pos, cam_ori):
    def quat_to_matrix(q):
        x, y, z, w = q; xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z; wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
            [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)],
        ])
    R = quat_to_matrix(cam_ori); R_t = R.T
    V = np.eye(4); V[:3, :3] = R_t; V[:3, 3] = -R_t @ cam_pos
    return V

def build_proj(focal, aperture, w, h):
    fov = 2.0 * math.atan(aperture / (2.0 * focal)); aspect = w / h
    f = 1.0 / math.tan(fov / 2.0)
    P = np.zeros((4, 4))
    P[0, 0] = f / aspect; P[1, 1] = f
    P[2, 2] = (1000 + 0.01) / (0.01 - 1000); P[2, 3] = (2*1000*0.01) / (0.01 - 1000)
    P[3, 2] = -1.0
    return P

def world_to_screen(world_pos, V, P, w, h):
    pos = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)
    eye = V @ pos; clip = P @ eye
    if abs(clip[3]) < 1e-8: return None
    ndc = clip[:3] / clip[3]
    sx = int((ndc[0] * 0.5 + 0.5) * w); sy = int((1.0 - (ndc[1] * 0.5 + 0.5)) * h)
    if sx < 0 or sx >= w or sy < 0 or sy >= h: return None
    return (sx, sy)

def capture_labeled(viewer, cam_pos, cam_ori, path, obj_positions, obj_labels):
    viewer.set_position_orientation(position=cam_pos, orientation=cam_ori)
    for _ in range(5): og.sim.step()
    obs, info = viewer.get_obs()
    if not isinstance(obs, dict) or 'rgb' not in obs: return False
    rgb = np.asarray(obs['rgb'])[..., :3].astype(np.uint8)
    img = Image.fromarray(rgb); draw = ImageDraw.Draw(img)
    V = build_view_matrix(cam_pos, cam_ori); P = build_proj(FOCAL_LENGTH, HORIZONTAL_APERTURE, IMG_W, IMG_H)
    colors = [(255,0,0), (0,200,0), (0,120,255), (220,180,0), (220,0,200), (0,200,200)]
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except: font = ImageFont.load_default()
    for i, (pos, label) in enumerate(zip(obj_positions, obj_labels)):
        lp = np.array([pos[0], pos[1], float(pos[2]) + 0.3], dtype=np.float32)
        screen = world_to_screen(lp, V, P, IMG_W, IMG_H)
        if screen is None: continue
        sx, sy = screen; color = colors[i % len(colors)]
        r = 10; draw.ellipse([sx-r, sy-r, sx+r, sy+r], outline=color, width=3)
        bbox = draw.textbbox((0, 0), label, font=font); tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = sx + 15, sy - th // 2
        draw.rectangle([tx-3, ty-3, tx+tw+3, ty+th+3], fill=(0,0,0,200))
        draw.text((tx, ty), label, fill=color, font=font)
    img.save(path); return True

def process_scene(data_dir, scene_model, out_dir):
    from omnigibson.objects import DatasetObject
    from omnigibson.utils.constants import PrimType

    env = create_env(scene_model=scene_model, robot_model="fetch")
    env.reset()
    for _ in range(30): og.sim.step()
    viewer = og.sim.viewer_camera
    try: viewer.add_modality("rgb")
    except: pass
    for obj in env.scene.objects:
        try: obj.keep_still()
        except: pass

    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith('online_env_a_') or 'rejected' in fname: continue
        with open(os.path.join(data_dir, fname)) as fh: data = json.load(fh)
        te = data.get('task_environment', {})
        added = te.get('added_objects', [])
        if not added: continue
        task = te.get('task', {})
        target_room = task.get('target_room', '')
        if not target_room:
            robot = te.get('robot', {})
            target_room = robot.get('target_room', '')
        rid = fname.replace('.json', '')
        print(f"{rid}: {task.get('instruction', '?')} [room={target_room}]", flush=True)
        tdir = f'{out_dir}/{rid}'; os.makedirs(tdir, exist_ok=True)

        cp, co = cam_pose_for_room(env, target_room)
        if cp is None:
            print(f"  SKIP: no objects in room '{target_room}'", flush=True)
            continue
        capture_labeled(viewer, cp, co, f'{tdir}/before.png', [], [])

        spawned = []; obj_positions = []; obj_labels = []
        for ao in added:
            pl = ao.get('placement', {})
            if pl.get('mode') == 'reused': continue
            pos = pl.get('pose', {}).get('position') or ao.get('pose', {}).get('position', [0,0,0])
            ori = pl.get('pose', {}).get('orientation_xyzw') or ao.get('pose', {}).get('orientation_xyzw', [0,0,0,1])
            obj = DatasetObject(name=ao['object_name'], category=ao['category'], prim_type=PrimType.RIGID)
            env.scene.add_object(obj)
            obj.set_position_orientation(position=th.tensor(pos, dtype=th.float32), orientation=th.tensor(ori, dtype=th.float32))
            obj.keep_still(); spawned.append(obj)
            obj_positions.append(pos); obj_labels.append(ao.get('category', ao['object_name']).replace('_', ' '))
        for _ in range(5): og.sim.step()

        cp, co = cam_pose_for_room(env, target_room)
        capture_labeled(viewer, cp, co, f'{tdir}/after_labeled.png', obj_positions, obj_labels)
        for obj in spawned:
            try: env.scene.remove_object(obj)
            except: pass
        print(f"  done ({len(spawned)} labeled)", flush=True)
    print(f"Done: {out_dir}", flush=True)

if __name__ == '__main__':
    import sys
    scenes = [
        ('/home2/daiyang/BEHAVIOR/code/outputs/test_Pomaria', 'Pomaria_0_int', '/home2/daiyang/BEHAVIOR/code/outputs/test_Pomaria_camera'),
        ('/home2/daiyang/BEHAVIOR/code/outputs/test_Benevolence', 'Benevolence_0_int', '/home2/daiyang/BEHAVIOR/code/outputs/test_Benevolence_camera'),
    ]
    for data_dir, scene_model, out_dir in scenes:
        if os.path.isdir(data_dir):
            process_scene(data_dir, scene_model, out_dir)
    sys.stdout.flush()
    os._exit(0)