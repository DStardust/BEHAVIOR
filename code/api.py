import os
import json
import math
import random
import traceback

import numpy as np
from PIL import Image

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True

import omnigibson as og


# =========================
# Common utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def safe_name(s: str) -> str:
    return str(s).replace(":", "_").replace("/", "_").replace(" ", "_")


# =========================
# Step2 utilities
# =========================
def save_rgb(rgb, path):
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_depth_vis(depth, path):
    arr = np.asarray(depth).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_segmentation_vis(seg, path):
    seg = np.asarray(seg).astype(np.uint32)
    ids = np.unique(seg)

    vis = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for sid in ids:
        if sid == 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = np.array([
                (int(sid) * 37) % 256,
                (int(sid) * 67) % 256,
                (int(sid) * 97) % 256,
            ], dtype=np.uint8)
        vis[seg == sid] = color

    Image.fromarray(vis).save(path)


def save_modalities(obs_dict, info_dict, prefix):
    saved = {
        "prefix": prefix,
        "obs_keys": list(obs_dict.keys()) if isinstance(obs_dict, dict) else [],
        "files": {},
        "seg_semantic_label_map": None,
    }

    if "rgb" in obs_dict:
        rgb_path = f"{prefix}_rgb.png"
        save_rgb(obs_dict["rgb"], rgb_path)
        saved["files"]["rgb_png"] = rgb_path

    if "depth" in obs_dict:
        depth_raw_path = f"{prefix}_depth.npy"
        depth_vis_path = f"{prefix}_depth.png"
        np.save(depth_raw_path, np.asarray(obs_dict["depth"]))
        save_depth_vis(obs_dict["depth"], depth_vis_path)
        saved["files"]["depth_raw_npy"] = depth_raw_path
        saved["files"]["depth_vis_png"] = depth_vis_path

    if "seg_semantic" in obs_dict:
        seg = np.asarray(obs_dict["seg_semantic"]).astype(np.uint32)

        seg_raw_path = f"{prefix}_seg_semantic.npy"
        seg_vis_path = f"{prefix}_seg_semantic_vis.png"
        np.save(seg_raw_path, seg)
        save_segmentation_vis(seg, seg_vis_path)

        saved["files"]["seg_semantic_raw_npy"] = seg_raw_path
        saved["files"]["seg_semantic_vis_png"] = seg_vis_path

        if isinstance(info_dict, dict) and "seg_semantic" in info_dict:
            saved["seg_semantic_label_map"] = info_dict["seg_semantic"]

    return saved


def find_robot_cameras(robot_obs):
    cams = {}
    if not isinstance(robot_obs, dict):
        return cams

    for k, v in robot_obs.items():
        if isinstance(v, dict) and any(mod in v for mod in ["rgb", "depth", "seg_semantic"]):
            cams[k] = v
    return cams


def build_prefix_for_robot_cam(cam_key):
    lower = cam_key.lower()
    if "eyes" in lower:
        return "robot_eyes"
    elif "eef_link" in lower or "eef" in lower:
        return "robot_eef"
    else:
        return f"robot_{safe_name(cam_key)}"


# =========================
# Step3 utilities
# =========================
def to_list(x):
    try:
        return x.tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return x


def get_state_by_class_name(obj, class_name):
    for cls, state in getattr(obj, "states", {}).items():
        if cls.__name__ == class_name:
            return state
    return None


def get_room_info_fallback(obj):
    room = {}
    for attr in ["room_type", "room_instance", "in_rooms", "rooms"]:
        if hasattr(obj, attr):
            try:
                room[attr] = to_list(getattr(obj, attr))
            except Exception:
                room[attr] = None
    return room


def collect_obj_info(obj):
    info = {
        "name": getattr(obj, "name", None),
        "category": getattr(obj, "category", None),
        "prim_path": getattr(obj, "prim_path", None),
        "room_info": get_room_info_fallback(obj),
        "available_states": [],
    }

    try:
        pos, quat = obj.get_position_orientation()
        info["position"] = to_list(pos)
        info["orientation_xyzw"] = to_list(quat)
    except Exception:
        info["position"] = None
        info["orientation_xyzw"] = None

    try:
        info["available_states"] = sorted([cls.__name__ for cls in obj.states.keys()])
    except Exception:
        info["available_states"] = []

    aabb_state = get_state_by_class_name(obj, "AABB")
    if aabb_state is not None:
        try:
            aabb_min, aabb_max = aabb_state.get_value()
            info["aabb_min"] = to_list(aabb_min)
            info["aabb_max"] = to_list(aabb_max)
            try:
                aabb_min_arr = np.asarray(aabb_min, dtype=np.float32)
                aabb_max_arr = np.asarray(aabb_max, dtype=np.float32)
                info["aabb_extent"] = to_list(aabb_max_arr - aabb_min_arr)
            except Exception:
                info["aabb_extent"] = None
        except Exception:
            info["aabb_min"] = None
            info["aabb_max"] = None
            info["aabb_extent"] = None
    else:
        info["aabb_min"] = None
        info["aabb_max"] = None
        info["aabb_extent"] = None

    contact_state = get_state_by_class_name(obj, "ContactBodies")
    if contact_state is not None:
        try:
            contacts = contact_state.get_value()
            info["contact_body_count"] = len(contacts)
        except Exception:
            info["contact_body_count"] = None
    else:
        info["contact_body_count"] = None

    return info


def get_all_scene_objects(scene):
    objs = []

    if hasattr(scene, "objects"):
        try:
            scene_objects = getattr(scene, "objects")
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    if not objs and hasattr(scene, "_objects"):
        try:
            scene_objects = getattr(scene, "_objects")
            if isinstance(scene_objects, dict):
                objs.extend(scene_objects.values())
            else:
                objs.extend(list(scene_objects))
        except Exception:
            pass

    dedup = {}
    for obj in objs:
        key = getattr(obj, "prim_path", None) or getattr(obj, "name", None) or str(id(obj))
        dedup[key] = obj

    return list(dedup.values())


# =========================
# Config builders
# =========================
def build_base_cfg(scene_model: str):
    return {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
        "objects": [],
        "robots": [],
    }


def build_robot_cfg(robot_model: str, obs_modalities=None):
    cfg = {
        "model": robot_model,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "obs_modalities": ["rgb", "depth", "seg_semantic"],
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": 240,
                    "image_width": 320,
                }
            }
        },
    }
    if obs_modalities is not None:
        cfg["obs_modalities"] = list(obs_modalities)
    return cfg


def create_env(
    scene_model: str = "Rs_int",
    robot_model: str | None = "fetch",
    robot_obs_modalities=None,
):
    cfg = build_base_cfg(scene_model)
    if robot_model is not None:
        cfg["robots"] = [build_robot_cfg(robot_model, obs_modalities=robot_obs_modalities)]
    return og.Environment(configs=cfg)


def stabilize_robot_spawn(env, seed, max_attempts=16, warmup_steps=30, native_displacement_limit=0.05):
    """Place the robot on the official traversability map and persist that reset pose."""
    if not getattr(env, "robots", None):
        for _ in range(warmup_steps):
            og.sim.step()
        return None

    robot = env.robots[0]
    rng = random.Random(seed)
    pinned_native_names = set()
    jointed_native_names = set()
    furniture_tokens = {
        "armchair", "bed", "bookcase", "cabinet", "chair", "counter", "countertop",
        "desk", "dresser", "island", "lamp", "ottoman", "shelf", "sofa", "table",
        "wardrobe",
    }
    # Environment.reset() restores the scene RNG in some OG scenes. Sampling
    # after every reset therefore returns the same bad traversability point and
    # makes retries ineffective. Build a genuinely diverse candidate list first.
    spawn_candidates = []
    for _ in range(max_attempts):
        floor = rng.randrange(int(env.scene.n_floors))
        _, position = env.scene.get_random_point(floor=floor, robot=robot)
        yaw = rng.uniform(-math.pi, math.pi)
        spawn_candidates.append((position.clone(), yaw))
    scene_objects = {getattr(obj, "name", ""): obj for obj in get_all_scene_objects(env.scene)}
    native_baseline = {}
    for name, obj in scene_objects.items():
        if obj is robot or name.startswith("online_env_"):
            continue
        try:
            position, orientation = obj.get_position_orientation()
            native_baseline[name] = (position.clone(), orientation.clone())
        except Exception:
            continue
    for attempt in range(1, max_attempts + 1):
        # Raw env.reset() destabilizes native rigid bodies in several scenes.
        # Restore their original poses through PhysX while it is playing. OG's
        # stopped EntityPrim path asserts that an articulated entity prim and
        # its root link have exactly matching orientations, which is not true
        # for every official scene object.
        if attempt > 1:
            if not og.sim.is_playing():
                og.sim.play()
            for name, (position, orientation) in native_baseline.items():
                obj = scene_objects.get(name)
                if obj is None:
                    continue
                obj.set_position_orientation(position=position, orientation=orientation)
                try:
                    obj.keep_still()
                except Exception:
                    pass
        # Fetch is a holonomic-base robot. Setting its pose while physics is
        # running teleports six base joints and can invalidate PhysX broadphase.
        # Pose only the robot while stopped, then rebuild the view via play().
        if og.sim.is_playing():
            og.sim.stop()
        for name in pinned_native_names:
            if name in jointed_native_names:
                continue
            obj = scene_objects.get(name)
            if obj is None:
                continue
            try:
                from omnigibson.utils.usd_utils import create_joint

                create_joint(
                    prim_path=f"{obj.prim_path}/deltasg_root_joint",
                    joint_type="FixedJoint",
                    body1=obj.root_link.prim_path,
                )
                jointed_native_names.add(name)
            except Exception as exc:
                raise RuntimeError(f"Failed to pin unstable native furniture {name}: {exc}") from exc
        position, yaw = spawn_candidates[attempt - 1]
        orientation = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        robot.set_position_orientation(position=position, orientation=orientation, frame="scene")
        og.sim.play()
        robot.keep_still()
        start, _ = robot.get_position_orientation()
        native_start = {name: pose[0] for name, pose in native_baseline.items()}
        for _ in range(warmup_steps):
            og.sim.step()
        end, end_quat = robot.get_position_orientation()
        displacement = float(((end - start) ** 2).sum() ** 0.5)
        tilt = float((end_quat[0] ** 2 + end_quat[1] ** 2) ** 0.5)
        moved_native = []
        moved_objects = {}
        for obj in get_all_scene_objects(env.scene):
            name = getattr(obj, "name", "")
            if name not in native_start:
                continue
            try:
                current = obj.get_position_orientation()[0]
                moved = float(((current - native_start[name]) ** 2).sum() ** 0.5)
            except Exception:
                continue
            if moved > native_displacement_limit:
                moved_native.append((name, moved))
                moved_objects[name] = obj
        native_max = max((moved for _, moved in moved_native), default=0.0)
        if displacement <= 0.10 and tilt <= 0.15 and not moved_native:
            print(
                f"[robot-spawn] stable traversable pose attempt={attempt} "
                f"displacement={displacement:.3f} tilt={tilt:.3f} native_max={native_max:.3f}",
                flush=True,
            )
            return {
                "attempt": attempt,
                "displacement": displacement,
                "tilt": tilt,
                "native_max_displacement": native_max,
                "pinned_native_objects": sorted(pinned_native_names),
            }
        newly_pinned = []
        for name, _ in moved_native:
            obj = moved_objects.get(name)
            category = str(getattr(obj, "category", "") or "").lower()
            is_furniture = any(token in category for token in furniture_tokens)
            if is_furniture and name not in pinned_native_names:
                pinned_native_names.add(name)
                newly_pinned.append(name)
        moved_preview = ",".join(f"{name}:{moved:.3f}" for name, moved in moved_native[:3]) or "none"
        print(
            f"[robot-spawn] rejected pose attempt={attempt} "
            f"displacement={displacement:.3f} tilt={tilt:.3f} "
            f"native_max={native_max:.3f} moved={moved_preview}",
            flush=True,
        )
        if newly_pinned:
            print(
                f"[robot-spawn] pinning unstable native furniture at original pose: {sorted(newly_pinned)}",
                flush=True,
            )
    raise RuntimeError(f"Could not find a stable robot spawn after {max_attempts} attempts")


# =========================
# Step2 on existing env
# =========================
def run_camera_capture_on_env(env, scene_model: str, robot_model: str, output_dir: str = "."):
    ensure_dir(output_dir)

    cwd = os.getcwd()
    try:
        os.chdir(output_dir)

        print("[step2 1/5] reset")
        env.reset()

        print("[step2 2/5] warmup")
        for _ in range(20):
            og.sim.step()

        print("[step2 3/5] get robot camera observations")
        if not getattr(env, "robots", None):
            raise RuntimeError("No robots found in env. Step2 requires a robot-enabled environment.")

        robot = env.robots[0]
        robot_obs, robot_info = robot.get_obs()

        robot_cam_dict = find_robot_cameras(robot_obs)
        if not robot_cam_dict:
            raise RuntimeError(
                f"No robot cameras found. robot_obs keys={list(robot_obs.keys()) if isinstance(robot_obs, dict) else type(robot_obs)}"
            )

        robot_results = {}
        for cam_key, cam_obs in robot_cam_dict.items():
            prefix = build_prefix_for_robot_cam(cam_key)
            cam_info = robot_info.get(cam_key, {}) if isinstance(robot_info, dict) else {}
            robot_results[cam_key] = save_modalities(cam_obs, cam_info, prefix)
            print(f"saved robot camera: {cam_key} -> prefix {prefix}")

        print("[step2 4/5] get default viewer camera observations")
        viewer = og.sim.viewer_camera
        if hasattr(viewer, "add_modality"):
            for m in ["rgb", "depth", "seg_semantic"]:
                try:
                    viewer.add_modality(m)
                except Exception:
                    pass

        for _ in range(5):
            og.sim.step()

        viewer_obs, viewer_info = viewer.get_obs()
        if not isinstance(viewer_obs, dict):
            raise RuntimeError("viewer_obs is not a dict")

        viewer_result = save_modalities(viewer_obs, viewer_info, "viewer")
        print("saved default viewer camera outputs")

        print("[step2 5/5] save result json")
        result = {
            "ok": True,
            "scene_model": scene_model,
            "robot_name": robot.name,
            "robot_model": robot_model,
            "robot_action_dim": int(robot.action_dim),
            "robot_camera_keys": list(robot_cam_dict.keys()),
            "robot_results": robot_results,
            "viewer_result": viewer_result,
            "notes": {
                "viewer_pose": "default",
                "robot_main_camera_expected": "robot_eyes",
                "robot_aux_camera_expected": "robot_eef",
            },
        }

        write_json("step2_all_in_one_result.json", result)
        print("saved step2_all_in_one_result.json")
        return result
    finally:
        os.chdir(cwd)


# =========================
# Step3 on existing env
# =========================
def export_all_scene_objects_on_env(env, scene_model: str = "Rs_int", output_dir: str = "."):
    ensure_dir(output_dir)

    print("[step3 1/3] warmup")
    for _ in range(30):
        og.sim.step()

    print("[step3 2/3] export all scene objects")
    all_objs = get_all_scene_objects(env.scene)
    print(f"total scene objects found = {len(all_objs)}")

    exported = []
    category_counts = {}

    for obj in all_objs:
        info = collect_obj_info(obj)
        exported.append(info)

        cat = info["category"] if info["category"] is not None else "__NONE__"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    result = {
        "ok": True,
        "scene_model": scene_model,
        "num_scene_objects": len(all_objs),
        "category_counts": dict(sorted(category_counts.items(), key=lambda x: x[0])),
        "objects": exported,
    }

    write_json(os.path.join(output_dir, "scene_all_objects.json"), result)
    print(f"saved scene_all_objects.json with {len(exported)} objects")
    print("[step3 3/3] done")
    return result


# =========================
# Backward-compatible wrappers
# =========================
def run_camera_capture(scene_model: str = "Rs_int", robot_model: str = "fetch", output_dir: str = "."):
    env = create_env(scene_model=scene_model, robot_model=robot_model)
    return run_camera_capture_on_env(env=env, scene_model=scene_model, robot_model=robot_model, output_dir=output_dir)


def export_all_scene_objects(scene_model: str = "Rs_int", output_dir: str = "."):
    env = create_env(scene_model=scene_model, robot_model=None)
    env.reset()
    for _ in range(30):
        og.sim.step()
    return export_all_scene_objects_on_env(env=env, scene_model=scene_model, output_dir=output_dir)


# =========================
# Main merged pipeline
# =========================
def run_level1(scene_model: str = "Rs_int", robot_model: str = "fetch", output_dir: str = "."):
    ensure_dir(output_dir)

    summary = {
        "scene_model": scene_model,
        "robot_model": robot_model,
        "output_dir": output_dir,
        "ok": False,
        "steps": {},
    }

    try:
        print("[level1 1/3] create env once for step2 + step3")
        env = create_env(scene_model=scene_model, robot_model=robot_model)

        print("[level1 2/3] run step2 on shared env")
        summary["steps"]["step2_camera"] = run_camera_capture_on_env(
            env=env,
            scene_model=scene_model,
            robot_model=robot_model,
            output_dir=output_dir,
        )

        print("[level1 3/3] run step3 on shared env")
        summary["steps"]["step3_scene_objects"] = export_all_scene_objects_on_env(
            env=env,
            scene_model=scene_model,
            output_dir=output_dir,
        )

        summary["ok"] = True
        write_json(os.path.join(output_dir, "level1_summary.json"), summary)
        return summary
    except Exception as e:
        summary["error"] = repr(e)
        summary["traceback"] = traceback.format_exc()
        write_json(os.path.join(output_dir, "level1_summary.json"), summary)
        raise
