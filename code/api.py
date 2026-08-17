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
from omnigibson.utils.sampling_utils import raytest


# =========================
# Common utilities
# =========================
class RobotSpawnError(RuntimeError):
    """A scene-level failure to establish a physically grounded robot pose."""

    def __init__(self, reason, detail=None):
        self.reason = str(reason)
        self.detail = dict(detail or {})
        super().__init__(self.reason)


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


def build_robot_cfg(robot_model: str, obs_modalities=None, camera_resolution=None):
    image_width, image_height = camera_resolution or (320, 240)
    cfg = {
        "model": robot_model,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "obs_modalities": ["rgb", "depth", "seg_semantic"],
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": int(image_height),
                    "image_width": int(image_width),
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
    added_objects=None,
    camera_resolution=None,
):
    cfg = build_base_cfg(scene_model)
    if robot_model is not None:
        cfg["robots"] = [
            build_robot_cfg(
                robot_model,
                obs_modalities=robot_obs_modalities,
                camera_resolution=camera_resolution,
            )
        ]
    # Preload generated objects during initial construction so their
    # kinematic/dynamic flags are fixed before physics/contact views exist.
    cfg["objects"] = list(added_objects or [])
    return og.Environment(configs=cfg)


def stabilize_robot_spawn(
    env,
    seed,
    max_attempts=16,
    warmup_steps=30,
    native_displacement_limit=0.05,
    preferred_target_name=None,
    preferred_max_distance=1.0,
    settle_scene=True,
):
    """Place the robot on the official traversability map and persist that reset pose."""
    if not getattr(env, "robots", None):
        for _ in range(warmup_steps):
            og.sim.step()
        return None

    robot = env.robots[0]
    robot_body_paths = {
        str(link.prim_path)
        for link in getattr(robot, "links", {}).values()
        if getattr(link, "prim_path", None)
    }
    rng = random.Random(seed)
    def geometry_center(obj):
        aabb_state = get_state_by_class_name(obj, "AABB")
        if aabb_state is None:
            return None
        try:
            lower, upper = aabb_state.get_value()
            return (lower.clone() + upper.clone()) * 0.5
        except Exception:
            return None
    # Measure the upright robot root-to-ground offset before any scene settling.
    # A traversability map supplies XY topology and a nominal floor height, but
    # its Z value is not a physical support guarantee across all scene assets.
    try:
        robot_root_position, _ = robot.get_position_orientation()
        robot_aabb_min, _ = robot.aabb
        robot_ground_offset = float(robot_root_position[2] - robot_aabb_min[2])
    except Exception as exc:
        raise RobotSpawnError(
            "robot_ground_geometry_unavailable", {"error": repr(exc)}
        ) from exc
    if not math.isfinite(robot_ground_offset) or not 0.0 <= robot_ground_offset <= 1.0:
        raise RobotSpawnError(
            "robot_ground_geometry_invalid",
            {"root_to_aabb_bottom": robot_ground_offset},
        )
    print(
        f"[robot-spawn] measured root_to_aabb_bottom={robot_ground_offset:.3f}",
        flush=True,
    )

    # Some official scenes contain movable furniture that drops into its
    # physically valid pose during the first simulation frames. Record the
    # integrity baseline only after this one-time scene settling; otherwise the
    # deterministic drop is incorrectly attributed to every robot spawn.
    robot.keep_still()
    # Let loose official-scene clutter finish its one-time gravity settling
    # before recording the integrity baseline. This is cheaper than restoring
    # a slightly pre-settled pose on every spawn retry.
    # Benevolence_2 contains loose articulated doors that can continue their
    # one-time gravity settling after 60 frames. Keep the longer window local
    # to that scene so other scenes retain their established seeded spawn.
    scene_model = str(getattr(env.scene, "scene_model", ""))
    settle_steps = (
        max(90, warmup_steps * 3)
        if scene_model == "Benevolence_2_int"
        else max(60, warmup_steps * 2)
    )
    for _ in range(settle_steps if settle_scene else 0):
        og.sim.step()
    # Convergence settle: the fixed windows above were still mid-relaxation for
    # slow fixtures (Benevolence_0 door_ohagsq_0 plateaued at 0.186 m across
    # all 16 spawn attempts after the 90-step window; Merom_1
    # bottom_cabinet_jhymlr_0 moved 0.092 m per attempt), so every
    # restore-to-baseline replayed the same A->B displacement and exhausted the
    # spawn gate. Keep stepping in windows until every native object's geometry
    # center moves less than the convergence threshold within a whole window;
    # capped so pathological jitter cannot hang the run. No gate is relaxed:
    # the per-attempt 0.05 m native-movement limit stays untouched.
    settle_window_steps = 15
    settle_convergence_limit = 0.005
    settle_max_extra_windows = 40
    settle_targets = []
    for obj in get_all_scene_objects(env.scene):
        name = getattr(obj, "name", "")
        if obj is robot or not name or name.startswith("online_env_"):
            continue
        settle_targets.append(obj)
    for window_index in range(settle_max_extra_windows if settle_scene else 0):
        pre_centers = {}
        for obj in settle_targets:
            center = geometry_center(obj)
            if center is not None:
                pre_centers[id(obj)] = (obj, center)
        for _ in range(settle_window_steps):
            og.sim.step()
        max_native_move = 0.0
        for obj, pre_center in pre_centers.values():
            post_center = geometry_center(obj)
            if post_center is None:
                continue
            moved = float(((post_center - pre_center) ** 2).sum() ** 0.5)
            max_native_move = max(max_native_move, moved)
        print(
            f"[robot-spawn] settle window={window_index + 1} "
            f"max_native_move={max_native_move:.5f}",
            flush=True,
        )
        if max_native_move < settle_convergence_limit:
            break
    # Environment.reset() restores the scene RNG in some OG scenes. Sampling
    # after every reset therefore returns the same bad traversability point and
    # makes retries ineffective. Build a genuinely diverse candidate list first.
    spawn_candidates = []
    if preferred_target_name:
        target = env.scene.object_registry("name", preferred_target_name, None)
        if target is None:
            print(
                f"[robot-spawn] preferred native target {preferred_target_name!r} is missing; "
                "falling back to ordinary stable spawn",
                flush=True,
            )
        else:
            try:
                import torch as th

                target_position, _ = target.get_position_orientation()
                target_xy_radius = 0.0
                target_aabb = get_state_by_class_name(target, "AABB")
                if target_aabb is not None:
                    try:
                        target_lower, target_upper = target_aabb.get_value()
                        target_xy_radius = float(
                            th.linalg.norm((target_upper[:2] - target_lower[:2]) * 0.5)
                        )
                    except Exception:
                        target_xy_radius = 0.0
                effective_max_distance = preferred_max_distance + target_xy_radius
                target_rooms = set(getattr(target, "in_rooms", None) or [])
                trav_map = env.scene.trav_map
                floor = min(
                    range(trav_map.n_floors),
                    key=lambda index: abs(
                        float(trav_map.floor_heights[index]) - float(target_position[2])
                    ),
                )
                traversable = trav_map._erode_trav_map(
                    th.clone(trav_map.floor_map[floor]), robot=robot
                )
                target_pixel = trav_map.world_to_map(target_position[:2]).to(traversable.device)
                pixels = th.nonzero(traversable == 255)
                if pixels.numel():
                    distances = th.linalg.norm(pixels.float() - target_pixel.float(), dim=1)
                    for index in th.argsort(distances).cpu().tolist():
                        xy = trav_map.map_to_world(pixels[index])
                        room = env.scene.seg_map.get_room_instance_by_point(xy[:2])
                        distance = float(th.linalg.norm(target_position[:2].cpu() - xy[:2].cpu()))
                        if target_rooms and room not in target_rooms:
                            continue
                        if distance > effective_max_distance:
                            continue
                        position = th.tensor(
                            [float(xy[0]), float(xy[1]), float(trav_map.floor_heights[floor])],
                            dtype=th.float32,
                        )
                        yaw = math.atan2(
                            float(target_position[1] - position[1]),
                            float(target_position[0] - position[0]),
                        )
                        if any(
                            float(th.linalg.norm(position[:2] - candidate[0][:2])) < 0.25
                            for candidate in spawn_candidates
                            if candidate[2] == 0
                        ):
                            continue
                        spawn_candidates.append((position, yaw, 0))
                        # Cover the operation ring instead of spending every
                        # attempt on adjacent 10 cm pixels at the fixture edge.
                        if len(spawn_candidates) >= max_attempts:
                            break
            except Exception as exc:
                print(
                    f"[robot-spawn] target-conditioned spawn for {preferred_target_name!r} "
                    f"failed ({exc!r}); falling back to ordinary stable spawn",
                    flush=True,
                )
            if not spawn_candidates:
                print(
                    f"[robot-spawn] no same-room traversable spawn within "
                    f"{preferred_max_distance}m of native target {preferred_target_name!r}; "
                    "falling back to ordinary stable spawn",
                    flush=True,
                )
    for _ in range(max_attempts * 3):
        floor = rng.randrange(int(env.scene.n_floors))
        _, position = env.scene.get_random_point(floor=floor, robot=robot)
        yaw = rng.uniform(-math.pi, math.pi)
        spawn_candidates.append((position.clone(), yaw, 1))
    scene_objects = {getattr(obj, "name", ""): obj for obj in get_all_scene_objects(env.scene)}
    ground_support_paths = []
    for obj in scene_objects.values():
        category = str(getattr(obj, "category", "") or "").lower()
        if category not in {"floor", "floors", "carpet", "rug"}:
            continue
        prim_path = str(getattr(obj, "prim_path", "") or "")
        if prim_path:
            ground_support_paths.append(prim_path.rstrip("/"))
    if not ground_support_paths:
        raise RobotSpawnError("scene_has_no_physical_ground_objects")

    def robot_native_link_overlaps(min_penetration=0.005):
        """Return native rigid links penetrated by the robot's current links."""
        robot_links = []
        for robot_link_name, robot_link in (getattr(robot, "links", None) or {}).items():
            try:
                lower, upper = robot_link.aabb
                robot_links.append((
                    str(robot_link_name),
                    np.asarray(lower.cpu(), dtype=float),
                    np.asarray(upper.cpu(), dtype=float),
                ))
            except Exception:
                continue
        overlaps = []
        for object_name, obj in scene_objects.items():
            if obj is robot or object_name.startswith("online_env_"):
                continue
            category = str(getattr(obj, "category", "") or "").lower()
            if category in {
                "floor", "floors", "carpet", "rug",
                "wall", "walls", "ceiling", "ceilings",
            }:
                continue
            for object_link_name, object_link in (getattr(obj, "links", None) or {}).items():
                try:
                    lower, upper = object_link.aabb
                    lower = np.asarray(lower.cpu(), dtype=float)
                    upper = np.asarray(upper.cpu(), dtype=float)
                except Exception:
                    continue
                for robot_link_name, robot_lower, robot_upper in robot_links:
                    penetration = np.minimum(robot_upper, upper) - np.maximum(robot_lower, lower)
                    if np.all(penetration > min_penetration):
                        overlaps.append({
                            "object": object_name,
                            "object_link": str(object_link_name),
                            "robot_link": robot_link_name,
                            "penetration": [round(float(value), 4) for value in penetration],
                        })
                        if len(overlaps) >= 8:
                            return overlaps
        return overlaps

    native_baseline = {}
    for name, obj in scene_objects.items():
        if obj is robot or name.startswith("online_env_"):
            continue
        try:
            position, orientation = obj.get_position_orientation()
            native_baseline[name] = (
                position.clone(),
                orientation.clone(),
                geometry_center(obj),
            )
        except Exception:
            continue
    # deltasg #20 (Benevolence_0 door_ohagsq_0, 2026-08-12): spawn-restore
    # diagnostics showed was_restored=True with pre-warmup deviation stuck at
    # 0.061-0.185 m on every attempt. set_position_orientation only restores
    # an articulated fixture's root link; the joint state perturbed by an
    # earlier spawn warmup (e.g. a door hinge angle) stays displaced, so every
    # later attempt re-measures the same native deviation and exhausts the
    # spawn gate. Capture joint positions/velocities alongside the root-pose
    # baseline so restore can return articulated natives completely.
    native_joint_baseline = {}
    for name, obj in scene_objects.items():
        if name not in native_baseline:
            continue
        try:
            if int(getattr(obj, "n_joints", 0) or 0) <= 0:
                continue
            native_joint_baseline[name] = (
                obj.get_joint_positions().clone(),
                obj.get_joint_velocities().clone(),
            )
        except Exception:
            continue
    # enva_gen10_pbfix3_20260810_234131: a random spawn landed in a small
    # traversability pocket whose only reachable room was dining_room_0; one
    # failed placement then excluded that room and every later sample
    # hard-rejected with no_reachable_compatible_support_room. Prefer
    # candidates on the largest connected traversability component so the
    # robot keeps multi-room access. Soft preference only (stable reorder, no
    # exclusion); best-effort: any failure keeps the official candidate order.
    try:
        import cv2
        import torch as th

        trav_map = env.scene.trav_map
        floor_components = {}
        for floor_index in range(int(trav_map.n_floors)):
            traversable = trav_map._erode_trav_map(
                th.clone(trav_map.floor_map[floor_index]), robot=robot
            )
            traversable_np = traversable.cpu().numpy().astype(np.uint8)
            _, labels_np = cv2.connectedComponents(traversable_np, connectivity=8)
            component_labels, component_counts = np.unique(
                labels_np[traversable_np != 0], return_counts=True
            )
            if component_labels.size:
                floor_components[floor_index] = (
                    labels_np,
                    int(component_labels[int(np.argmax(component_counts))]),
                )

        def largest_component_rank(candidate):
            position = candidate[0]
            floor_index = min(
                range(int(trav_map.n_floors)),
                key=lambda index: abs(
                    float(trav_map.floor_heights[index]) - float(position[2])
                ),
            )
            component = floor_components.get(floor_index)
            if component is None:
                return 1
            labels_np, best_label = component
            pixel = trav_map.world_to_map(position[:2]).long()
            pixel[0].clamp_(0, labels_np.shape[0] - 1)
            pixel[1].clamp_(0, labels_np.shape[1] - 1)
            return 0 if int(labels_np[int(pixel[0]), int(pixel[1])]) == best_label else 1

        def semantic_room_rank(candidate):
            try:
                room = env.scene.seg_map.get_room_instance_by_point(candidate[0][:2])
            except Exception:
                room = None
            return 0 if room else 1

        spawn_candidates.sort(
            key=lambda candidate: (
                candidate[2],
                semantic_room_rank(candidate),
                largest_component_rank(candidate),
            )
        )
        semantic = sum(
            1 for candidate in spawn_candidates if semantic_room_rank(candidate) == 0
        )
        preferred = sum(
            1 for candidate in spawn_candidates if largest_component_rank(candidate) == 0
        )
        print(
            f"[robot-spawn] semantic-room candidates={semantic}/{len(spawn_candidates)} "
            f"largest-component candidates={preferred}/{len(spawn_candidates)} "
            f"target-conditioned candidates="
            f"{sum(1 for candidate in spawn_candidates if candidate[2] == 0)}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[robot-spawn] largest-component preference skipped ({exc!r})",
            flush=True,
        )
    # get_random_point(..., robot=robot) already samples the robot-eroded
    # official traversability map. Scene-object AABBs are too coarse for a
    # second collision filter (e.g. cabinets include open clearance space),
    # so retain official-map order apart from the largest-component
    # preference above and let the short physics check validate it.
    spawn_candidates = spawn_candidates[:max_attempts]
    print(
        f"[robot-spawn] official-map candidates={len(spawn_candidates)}",
        flush=True,
    )
    if not spawn_candidates:
        raise RobotSpawnError("no_traversable_robot_spawn_candidates")
    def grounded_candidate(candidate_position):
        floor_hint = float(candidate_position[2])
        ray_start = [float(candidate_position[0]), float(candidate_position[1]), floor_hint + 2.0]
        hit = raytest(
            start_point=ray_start,
            end_point=[ray_start[0], ray_start[1], floor_hint - 8.0],
            only_closest=True,
            ignore_bodies=robot_body_paths,
        )
        valid_hits = []
        raw_hits = []
        if hit.get("hit"):
            normal = np.asarray(hit.get("normal"), dtype=float)
            hit_position = np.asarray(hit.get("position"), dtype=float)
            hit_path = str(hit.get("rigidBody") or hit.get("collision") or "")
            raw_hits.append({
                "path": hit_path,
                "z": float(hit_position[2]) if hit_position.shape == (3,) else None,
                "normal_z": float(normal[2]) if normal.shape == (3,) else None,
            })
            is_ground_path = any(
                hit_path == path or hit_path.startswith(f"{path}/")
                for path in ground_support_paths
            )
            if (
                normal.shape == (3,)
                and hit_position.shape == (3,)
                and normal[2] >= 0.7
                and is_ground_path
            ):
                valid_hits.append(
                    (abs(float(hit_position[2]) - floor_hint), hit_position, hit_path)
                )
        if not valid_hits:
            return None, raw_hits[:5]
        _, hit_position, hit_path = min(valid_hits, key=lambda item: item[0])
        grounded = candidate_position.clone()
        grounded[2] = float(hit_position[2]) + robot_ground_offset + 0.015
        return (grounded, float(hit_position[2]), hit_path), raw_hits[:5]

    physics_rebuild_attempted = False
    official_ground_plane_added = False
    if not any(
        grounded_candidate(position)[0] is not None
        for position, _, _ in spawn_candidates
    ):
        # A loaded scene can expose its traversability map before its static
        # colliders have entered the PhysX scene-query view. Rebuild that view
        # once and restore the live, post-settle poses so the repair cannot
        # reset the robot or displace native furniture.
        physics_rebuild_attempted = True
        floor_heights = [float(height) for height in env.scene.trav_map.floor_heights]
        if len(floor_heights) != 1 or abs(floor_heights[0]) > 0.02:
            raise RobotSpawnError(
                "physical_ground_query_unavailable",
                {
                    "scene_model": scene_model,
                    "candidates": len(spawn_candidates),
                    "physics_rebuild_attempted": False,
                    "floor_heights": floor_heights,
                },
            )
        print(
            "[robot-spawn] zero physical ground hits; installing official invisible "
            "ground plane during one physics rebuild",
            flush=True,
        )
        saved_sim_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        og.sim.add_ground_plane(floor_plane_visible=False)
        ground_support_paths.append(str(og.sim.floor_plane.prim_path).rstrip("/"))
        official_ground_plane_added = True
        og.sim.play()
        og.sim.load_state(saved_sim_state, serialized=False)
        for _ in range(2):
            og.sim.step()
        rebuild_moved_native = []
        for name, baseline in native_baseline.items():
            obj = scene_objects.get(name)
            if obj is None or baseline[2] is None:
                continue
            current_center = geometry_center(obj)
            if current_center is None:
                continue
            moved = float(((current_center - baseline[2]) ** 2).sum() ** 0.5)
            if moved > native_displacement_limit:
                rebuild_moved_native.append((name, moved))
        if rebuild_moved_native:
            raise RobotSpawnError(
                "physics_rebuild_changed_native_scene",
                {
                    "scene_model": scene_model,
                    "physics_rebuild_attempted": True,
                    "moved_native_objects": rebuild_moved_native,
                },
            )
        robot.keep_still()
        if not any(
            grounded_candidate(position)[0] is not None
            for position, _, _ in spawn_candidates
        ):
            raise RobotSpawnError(
                "physical_ground_query_unavailable",
                {
                    "scene_model": scene_model,
                    "candidates": len(spawn_candidates),
                    "physics_rebuild_attempted": True,
                    "ground_support_paths": ground_support_paths,
                },
            )

    native_to_restore = set()
    ground_rejections = 0
    for attempt, (position, yaw, candidate_kind) in enumerate(spawn_candidates, 1):
        # Raw env.reset() destabilizes native rigid bodies in several scenes.
        # Restore their original poses through PhysX while it is playing. OG's
        # stopped EntityPrim path asserts that an articulated entity prim and
        # its root link have exactly matching orientations, which is not true
        # for every official scene object.
        if native_to_restore:
            if not og.sim.is_playing():
                og.sim.play()
            for name in native_to_restore:
                native_position, native_orientation, _ = native_baseline[name]
                obj = scene_objects.get(name)
                if obj is None:
                    continue
                obj.set_position_orientation(
                    position=native_position,
                    orientation=native_orientation,
                )
                # deltasg #20: also restore articulated joint state (root-pose
                # restore alone leaves perturbed hinge angles in place, which
                # keeps the native displaced and poisons every later attempt).
                joint_state = native_joint_baseline.get(name)
                if joint_state is not None:
                    try:
                        obj.set_joint_positions(joint_state[0])
                        obj.set_joint_velocities(joint_state[1])
                    except Exception:
                        pass
                try:
                    obj.keep_still()
                except Exception:
                    pass
        # Keep PhysX running while moving Fetch. Several official scenes contain
        # unanchored furniture that re-enters gravity settling after stop/play;
        # restarting physics for every candidate corrupts both the scene and all
        # later retries. The following warmup still rejects real spawn collisions.
        if not og.sim.is_playing():
            og.sim.play()
        grounded, raw_ground_hits = grounded_candidate(position)
        if grounded is None:
            ground_rejections += 1
            print(
                f"[robot-spawn] rejected ground candidate attempt={attempt} "
                f"xy=({float(position[0]):.3f},{float(position[1]):.3f}) "
                f"floor_hint={float(position[2]):.3f} reason=no_floor_raycast_hit "
                f"raw_hits={json.dumps(raw_ground_hits, separators=(',', ':'))}",
                flush=True,
            )
            continue
        position, ground_z, ground_path = grounded
        orientation = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        # Restore the canonical upright joint state before every candidate so
        # joint flail left over from a previously rejected candidate or an
        # earlier failed sample cannot contaminate this candidate's stability
        # validation. Robot.reset() only writes joint state (no sim rebuild).
        robot.reset()
        robot.set_position_orientation(position=position, orientation=orientation, frame="scene")
        robot.keep_still()
        link_overlaps = robot_native_link_overlaps()
        if link_overlaps:
            preview = json.dumps(link_overlaps[:3], separators=(",", ":"))
            print(
                f"[robot-spawn] rejected link collision attempt={attempt} "
                f"xy=({float(position[0]):.3f},{float(position[1]):.3f}) "
                f"overlaps={preview}",
                flush=True,
            )
            continue
        start, _ = robot.get_position_orientation()
        native_start = {name: pose[0] for name, pose in native_baseline.items()}
        if os.environ.get("DELTASG_SPAWN_RESTORE_DIAG") == "1":
            for name, obj in scene_objects.items():
                if name not in native_baseline:
                    continue
                baseline_center = native_baseline[name][2]
                if baseline_center is None:
                    continue
                center = geometry_center(obj)
                if center is None:
                    continue
                dev = float(((center - baseline_center) ** 2).sum() ** 0.5)
                if dev > 0.01:
                    print(
                        f"[spawn-restore-diag] attempt={attempt} pre-warmup "
                        f"{name} dev_from_baseline={dev:.4f} "
                        f"was_restored={name in native_to_restore}",
                        flush=True,
                    )
        for _ in range(warmup_steps):
            og.sim.step()
        end, end_quat = robot.get_position_orientation()
        displacement_vector = end - start
        displacement = float((displacement_vector**2).sum() ** 0.5)
        tilt = float((end_quat[0] ** 2 + end_quat[1] ** 2) ** 0.5)
        try:
            end_aabb_min, _ = robot.aabb
            ground_gap = float(end_aabb_min[2] - ground_z)
        except Exception:
            ground_gap = float("nan")
        grounded_ok = math.isfinite(ground_gap) and -0.05 <= ground_gap <= 0.15
        moved_native = []
        for obj in get_all_scene_objects(env.scene):
            name = getattr(obj, "name", "")
            if name not in native_start:
                continue
            try:
                current = obj.get_position_orientation()[0]
                pose_moved = float(((current - native_start[name]) ** 2).sum() ** 0.5)
            except Exception:
                continue
            baseline_center = native_baseline[name][2]
            current_center = geometry_center(obj)
            geometry_moved = (
                float(((current_center - baseline_center) ** 2).sum() ** 0.5)
                if baseline_center is not None and current_center is not None
                else pose_moved
            )
            # Articulated scene assets can report a shifted entity root after a
            # stop/play cycle even though their rendered geometry did not move.
            moved = max(pose_moved, geometry_moved) if baseline_center is None else geometry_moved
            if moved > native_displacement_limit:
                moved_native.append((name, moved))
        native_max = max((moved for _, moved in moved_native), default=0.0)
        # Target-conditioned ring candidates (candidate_kind == 0) are bound
        # immediately after this gate and then feed the scene-integrity
        # baseline that accounts the robot at the 0.05 m gate. Accept them
        # only within that same limit so a drifting candidate is detected and
        # resampled BEFORE binding instead of failing the run after the
        # binding was consumed. Ordinary random candidates keep the
        # historical 0.10 m because their integrity baseline is recorded
        # after this warmup.
        robot_displacement_limit = (
            native_displacement_limit if candidate_kind == 0 else 0.10
        )
        if (
            displacement <= robot_displacement_limit
            and tilt <= 0.15
            and grounded_ok
            and not moved_native
        ):
            spawn_room = env.scene.seg_map.get_room_instance_by_point(end[:2])
            print(
                f"[robot-spawn] stable traversable pose attempt={attempt} "
                f"displacement={displacement:.3f} tilt={tilt:.3f} "
                f"ground_z={ground_z:.3f} ground_gap={ground_gap:.3f} "
                f"native_max={native_max:.3f} room={spawn_room} "
                f"validated_steps={warmup_steps} "
                f"robot_limit={robot_displacement_limit:.2f} "
                f"xy=({float(end[0]):.3f},{float(end[1]):.3f})",
                flush=True,
            )
            return {
                "attempt": attempt,
                "displacement": displacement,
                "tilt": tilt,
                "ground_z": ground_z,
                "ground_gap": ground_gap,
                "ground_path": ground_path,
                "room": spawn_room,
                "native_max_displacement": native_max,
                "pinned_native_objects": [],
                "physics_rebuild_attempted": physics_rebuild_attempted,
                "official_ground_plane_added": official_ground_plane_added,
            }
        native_to_restore = {name for name, _ in moved_native}
        moved_preview = ",".join(f"{name}:{moved:.3f}" for name, moved in moved_native[:3]) or "none"
        print(
            f"[robot-spawn] rejected pose attempt={attempt} "
            f"displacement={displacement:.3f} tilt={tilt:.3f} "
            f"delta=({float(displacement_vector[0]):.3f},"
            f"{float(displacement_vector[1]):.3f},{float(displacement_vector[2]):.3f}) "
            f"ground_z={ground_z:.3f} ground_gap={ground_gap:.3f} "
            f"native_max={native_max:.3f} moved={moved_preview} "
            f"validated_steps={warmup_steps} "
            f"robot_limit={robot_displacement_limit:.2f}",
            flush=True,
        )
    raise RobotSpawnError(
        "no_physically_stable_robot_spawn",
        {
            "scene_model": scene_model,
            "attempts": len(spawn_candidates),
            "ground_raycast_rejections": ground_rejections,
            "physics_rebuild_attempted": physics_rebuild_attempted,
            "official_ground_plane_added": official_ground_plane_added,
            "largest_component_preferred": preferred if "preferred" in locals() else None,
        },
    )


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
