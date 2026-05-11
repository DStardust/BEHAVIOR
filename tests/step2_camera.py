import os
import sys
import json
import traceback
import numpy as np
from PIL import Image

# =========================
# Basic env
# =========================
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import omnigibson as og


# =========================
# Utilities
# =========================
def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def safe_name(s: str) -> str:
    return s.replace(":", "_").replace("/", "_").replace(" ", "_")


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
    """
    Save rgb / depth / seg_semantic if available.
    Returns a dict describing saved files.
    """
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
    """
    Return all camera-like entries in robot_obs.
    """
    cams = {}
    if not isinstance(robot_obs, dict):
        return cams

    for k, v in robot_obs.items():
        if isinstance(v, dict):
            if any(mod in v for mod in ["rgb", "depth", "seg_semantic"]):
                cams[k] = v
    return cams


def build_prefix_for_robot_cam(cam_key):
    """
    Prefer readable short names.
    """
    lower = cam_key.lower()
    if "eyes" in lower:
        return "robot_eyes"
    elif "eef_link" in lower or "eef" in lower:
        return "robot_eef"
    else:
        return f"robot_{safe_name(cam_key)}"


# =========================
# Main
# =========================
def main():
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "objects": [],
        "robots": [
            {
                "model": "fetch",
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
        ],
    }

    try:
        print("[1/6] create env")
        env = og.Environment(configs=cfg)

        print("[2/6] reset")
        env.reset()

        print("[3/6] warmup")
        for _ in range(20):
            og.sim.step()

        # =========================
        # Robot cameras
        # =========================
        print("[4/6] get robot camera observations")
        robot = env.robots[0]
        robot_obs, robot_info = robot.get_obs()

        robot_cam_dict = find_robot_cameras(robot_obs)
        if not robot_cam_dict:
            raise RuntimeError(f"No robot cameras found. robot_obs keys={list(robot_obs.keys()) if isinstance(robot_obs, dict) else type(robot_obs)}")

        robot_results = {}
        for cam_key, cam_obs in robot_cam_dict.items():
            prefix = build_prefix_for_robot_cam(cam_key)
            cam_info = robot_info.get(cam_key, {}) if isinstance(robot_info, dict) else {}
            robot_results[cam_key] = save_modalities(cam_obs, cam_info, prefix)
            print(f"saved robot camera: {cam_key} -> prefix {prefix}")

        # =========================
        # Viewer camera
        # =========================
        print("[5/6] get default viewer camera observations")
        viewer = og.sim.viewer_camera

        # Use default viewer pose; do NOT manually change it
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

        # =========================
        # Save summary
        # =========================
        print("[6/6] save result json")
        result = {
            "ok": True,
            "scene_model": "Rs_int",
            "robot_name": robot.name,
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

        with open("step2_all_in_one_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print("saved step2_all_in_one_result.json")

        print("\n=== Step2 all-in-one finished successfully ===")
        for cam_key, cam_res in robot_results.items():
            print(f"[robot camera] {cam_key}")
            for k, v in cam_res["files"].items():
                print(f"  - {k}: {v}")

        print("[viewer camera]")
        for k, v in viewer_result["files"].items():
            print(f"  - {k}: {v}")

        hard_exit(0)

    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        hard_exit(1)


if __name__ == "__main__":
    main()