"""Replay and validate a DeltaSG task with OmniGibson expert primitives.

The current ``oracle_symbolic`` backend is a task-solvability verifier.  It
uses OmniGibson object states and assisted grasping, but is not a source of
low-level VLA control actions because navigation / manipulation may teleport.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True
gm.GUI_VIEWPORT_ONLY = True

import numpy as np
import cv2
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from PIL import Image
from omnigibson import object_states
from omnigibson.object_states import toggle as toggle_state
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitiveSet,
    SymbolicSemanticActionPrimitives,
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet,
    StarterSemanticActionPrimitives,
)
from omnigibson.action_primitives import starter_semantic_action_primitives as starter_primitives
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
)
from omnigibson.objects import DatasetObject
from omnigibson.sensors import VisionSensor
from omnigibson.utils.constants import PrimType
from omnigibson.utils import transform_utils as T

from api import create_env, get_all_scene_objects, validate_robot_stability
from deltasg_visual_effects import (
    SMOKE_FLOW_RENDER_WARMUP_FRAMES,
    SMOKE_FLOW_WARMUP_STEPS,
    SMOKE_ONLY_ON_FIRE_MODE,
    configure_on_fire_smoke_only,
)
from deltasg_expert import (
    DEFAULT_MAX_MANIPULATION_HEIGHT,
    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
    DEFAULT_MIN_MANIPULATION_HEIGHT,
    DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT,
    ExpertPlanError,
    FINE_MANIPULATION_PRIMITIVES,
    MANIPULATION_PRIMITIVES,
    PLACE_ACCESS_CORRIDOR_MARGIN,
    PLACE_ACCESS_HOVER_CLEARANCE,
    PLACE_ACCESS_MAX_CANDIDATES,
    PLACE_ACCESS_REACH_MARGIN,
    PLACE_NATIVE_MAX_ATTEMPTS,
    PLACE_NATIVE_MAX_WALL_SECONDS,
    PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION,
    PLACE_SAMPLING_SURFACE_XY_MARGIN,
    PLACE_SAMPLING_SURFACE_Z_EPSILON,
    PLACE_SUPPORT_EMBED_TOLERANCE,
    PLACE_SUPPORT_FLOAT_TOLERANCE,
    PLACE_RELEASE_CONTACT_CLEARANCE,
    PLACE_SUPPORT_PROBE_MARGIN,
    PLACE_SUPPORT_PROBE_RADIUS_FRACTION,
    PLACE_SUPPORT_REFERENCE_SURFACE_TOLERANCE,
    PLACE_SUPPORT_REFERENCE_TOP_EPSILON,
    PLACE_SUPPORT_REFERENCE_XY_RADIUS,
    compile_expert_plan,
    evaluate_manipulation_height,
    place_descent_corridor_blockers,
    plan_object_index,
    validate_visibility_snapshot,
)


SYMBOLIC_PRIMITIVE_MAP = {
    name: getattr(SymbolicSemanticActionPrimitiveSet, name)
    for name in (
        "NAVIGATE_TO",
        "GRASP",
        "PLACE_ON_TOP",
        "PLACE_INSIDE",
        "OPEN",
        "CLOSE",
        "TOGGLE_ON",
        "TOGGLE_OFF",
    )
}
PHYSICAL_PRIMITIVE_MAP = {
    name: getattr(StarterSemanticActionPrimitiveSet, name)
    for name in (
        "NAVIGATE_TO",
        "GRASP",
        "PLACE_ON_TOP",
        "PLACE_INSIDE",
        "OPEN",
        "CLOSE",
        "TOGGLE_ON",
        "TOGGLE_OFF",
    )
}
NON_BLOCKING_NAVIGATION_CATEGORIES = {
    "agent",
    "ceiling",
    "ceilings",
    "floor",
    "floors",
    "carpet",
    "rug",
    "mat",
    "walls",
    "wall",
}


def _camera_relative_to_robot(robot):
    _, camera = _primary_robot_camera(robot)
    robot_pose = robot.get_position_orientation()
    camera_pose = camera.get_position_orientation()
    robot_matrix = np.asarray(T.pose2mat(robot_pose).cpu(), dtype=float)
    camera_matrix = np.asarray(T.pose2mat(camera_pose).cpu(), dtype=float)
    return np.linalg.inv(robot_matrix) @ camera_matrix


def _project_candidate_target_bbox(relative_camera, candidate_xy, yaw, obj, width=320, height=240):
    candidate = np.eye(4, dtype=float)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    candidate[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    candidate[:3, 3] = [float(candidate_xy[0]), float(candidate_xy[1]), 0.0]
    camera = candidate @ relative_camera
    lower, upper = obj.aabb
    lower = np.asarray(lower.cpu(), dtype=float)
    upper = np.asarray(upper.cpu(), dtype=float)
    corners = np.asarray([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ])
    camera_points = (corners - camera[:3, 3]) @ camera[:3, :3]
    points = camera_points[camera_points[:, 2] < -0.05]
    if not len(points):
        return None
    focal = 0.5 * width / math.tan(math.radians(65.0) * 0.5)
    depth = -points[:, 2]
    pixels_x = focal * points[:, 0] / depth + width * 0.5
    pixels_y = height * 0.5 - focal * points[:, 1] / depth
    raw_bbox = [
        int(math.floor(float(np.min(pixels_x)))),
        int(math.floor(float(np.min(pixels_y)))),
        int(math.ceil(float(np.max(pixels_x)))),
        int(math.ceil(float(np.max(pixels_y)))),
    ]
    margin = max(3, int(round(min(width, height) * 0.02)))
    bbox = [
        max(margin, raw_bbox[0]),
        max(margin, raw_bbox[1]),
        min(width - margin - 1, raw_bbox[2]),
        min(height - margin - 1, raw_bbox[3]),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 16:
        return None
    return bbox


def _robot_collision_geometry(robot):
    """Cache the live robot collision boundary in its local frame."""
    robot_position, robot_orientation = robot.get_position_orientation()
    robot_position = np.asarray(robot_position.cpu(), dtype=float)
    live_points = np.asarray(robot.collision_points_world.cpu(), dtype=float)
    live_rotation = np.asarray(T.quat2mat(robot_orientation).cpu(), dtype=float)
    local_points = (live_points - robot_position) @ live_rotation
    return local_points, float(robot_position[2])


def _robot_collision_points_at_pose(robot, xy, yaw, collision_geometry=None):
    """Transform the live robot collision boundary to a candidate planar pose."""
    local_points, robot_z = collision_geometry or _robot_collision_geometry(robot)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    candidate_rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    candidate_position = np.asarray(
        [float(xy[0]), float(xy[1]), robot_z], dtype=float
    )
    return local_points @ candidate_rotation.T + candidate_position


def _navigation_blocker_aabbs(env, robot, target):
    # Objects in the robot's hand move rigidly with it; they are part of the
    # robot system, not obstacles (Beechwood_0 deliver_drink 2026-08-14: the
    # carried bottle's robot_radius-inflated AABB parked in front of the base
    # and severed the clearance-safe BFS corridor to the destination table).
    held_objects = set()
    for held_object in (getattr(robot, "_ag_obj_in_hand", None) or {}).values():
        if held_object is not None:
            held_objects.add(held_object)
    blockers = []
    for scene_object in _scene_objects(env).values():
        if scene_object is robot or scene_object is target:
            continue
        if scene_object in held_objects:
            continue
        category = str(getattr(scene_object, "category", "")).lower()
        if category in NON_BLOCKING_NAVIGATION_CATEGORIES:
            continue
        try:
            lower, upper = scene_object.aabb
            lower = np.asarray(lower.cpu(), dtype=float)
            upper = np.asarray(upper.cpu(), dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            continue
        blockers.append((str(getattr(scene_object, "name", "")) or category, lower, upper))
    return blockers


def _native_occupant_at_pose(
    env, robot, target, xy, yaw, margin=0.10, blockers=None, collision_geometry=None,
):
    """Return an unrelated object intersecting the candidate robot boundary.

    margin history: 0.01 m allowed poses ~1 cm off furniture AABBs
    (Beechwood_1 c1d45e183b10, bottom_cabinet_jhymlr_0): post-teleport settle
    drift closed the gap, and every observation-capture render loop (Isaac
    advances physics on each update tick while playing) then pushed the
    cabinet 0.056 m per capture until the 0.1132 m integrity rejection.
    0.10 m keeps a physical clearance buffer without blanket map erosion.
    """
    candidate_points = _robot_collision_points_at_pose(
        robot, xy, yaw, collision_geometry=collision_geometry,
    )
    blockers = blockers if blockers is not None else _navigation_blocker_aabbs(env, robot, target)
    for name, lower, upper in blockers:
        inside = np.all(candidate_points >= lower - margin, axis=1) & np.all(
            candidate_points <= upper + margin, axis=1
        )
        if np.any(inside):
            return name
    return None


def _connected_observation_pose(
    env,
    robot,
    obj,
    preferred_distance,
    clearance_margin=0.0,
    fallback_rank=0,
    fallback_yaw_offset=-10.0,
    require_route=False,
    route_clearance_margin=0.20,
    route_target=None,
    max_target_aabb_distance=None,
):
    """Choose a traversable pose in the robot's current connected component."""
    target_position, _ = obj.get_position_orientation()
    robot_position, _ = robot.get_position_orientation()
    trav_map = env.scene.trav_map
    floor = min(
        range(trav_map.n_floors),
        key=lambda index: abs(float(trav_map.floor_heights[index]) - float(target_position[2])),
    )
    traversable = trav_map._erode_trav_map(th.clone(trav_map.floor_map[floor]), robot=robot)
    if clearance_margin > 0:
        clearance_pixels = int(math.ceil(clearance_margin / float(trav_map.map_resolution)))
        kernel_size = max(1, 2 * clearance_pixels + 1)
        traversable = th.as_tensor(
            cv2.erode(
                traversable.cpu().numpy(),
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            ),
            device=traversable.device,
        )
    free = traversable.cpu().numpy() != 0
    source_pixel = trav_map.world_to_map(robot_position[:2]).to(traversable.device).long()
    source_pixel[0].clamp_(0, traversable.shape[0] - 1)
    source_pixel[1].clamp_(0, traversable.shape[1] - 1)
    start = (int(source_pixel[0]), int(source_pixel[1]))
    if not free[start]:
        all_free = np.argwhere(free)
        if not len(all_free):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                f"No robot-traversable pixels on floor {floor}",
                {"object": obj.name},
            )
        nearest = int(
            np.argmin(
                np.linalg.norm(all_free.astype(float) - np.asarray(start, dtype=float), axis=1)
            )
        )
        start = (int(all_free[nearest][0]), int(all_free[nearest][1]))
    # Flood-fill the robot's component under the SAME move rule as the
    # navigation BFS in _connected_navigation_waypoints (diagonals allowed
    # only where both orthogonal neighbors are free). cv2.connectedComponents
    # (connectivity=8) corner-cuts through blocked diagonals and labeled
    # stand-offs reachable that the clearance-safe BFS could not reach
    # (Benevolence_1 retrieve_book/close_window: PLANNING_ERROR 'No
    # clearance-safe map path to manipulation stand-off', 2/2 byte-identical).
    reachable = np.zeros(free.shape, dtype=bool)
    reachable[start] = True
    queue = deque([start])
    bfs_neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue:
        row, col = queue.popleft()
        for drow, dcol in bfs_neighbors:
            nrow, ncol = row + drow, col + dcol
            if not (0 <= nrow < free.shape[0] and 0 <= ncol < free.shape[1]):
                continue
            if not free[nrow, ncol] or reachable[nrow, ncol]:
                continue
            if drow and dcol and (not free[row + drow, col] or not free[row, col + dcol]):
                continue
            reachable[nrow, ncol] = True
            queue.append((nrow, ncol))
    pixels = th.nonzero(th.as_tensor(reachable, device=traversable.device))
    if pixels.numel() == 0:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            f"Robot traversable component is empty on floor {floor}",
            {"object": obj.name},
        )
    target_pixel = trav_map.world_to_map(target_position[:2]).to(pixels.device)
    preferred_pixels = preferred_distance / float(trav_map.map_resolution)
    distances = th.linalg.norm(pixels.float() - target_pixel.float(), dim=1)
    robot_pixel = trav_map.world_to_map(robot_position[:2]).to(pixels.device).float()
    target_to_robot = robot_pixel - target_pixel.float()
    target_to_robot = target_to_robot / th.clamp(th.linalg.norm(target_to_robot), min=1e-6)
    target_to_candidates = pixels.float() - target_pixel.float()
    candidate_directions = target_to_candidates / th.clamp(
        th.linalg.norm(target_to_candidates, dim=1, keepdim=True), min=1e-6
    )
    alignment = candidate_directions @ target_to_robot
    # A distance-only choice can put the stand-off point behind a small floor
    # target, making the robot drive through and displace it. Approach from the
    # robot-facing side while retaining the requested manipulation stand-off.
    candidate_scores = th.abs(distances - preferred_pixels) + preferred_pixels * (1.0 - alignment)
    target_rooms = set(getattr(obj, "in_rooms", None) or [])
    enforce_target_room_visibility = str(getattr(obj, "name", "")).startswith(
        "online_env_"
    )
    candidate_xy = None
    target_lower, target_upper = obj.aabb
    target_lower = np.asarray(target_lower.cpu(), dtype=float)
    target_upper = np.asarray(target_upper.cpu(), dtype=float)
    target_center = (target_lower + target_upper) * 0.5
    target_top_center = target_center.copy()
    target_top_center[2] = target_upper[2]
    target_path = str(getattr(obj, "prim_path", ""))
    robot_path = str(getattr(robot, "prim_path", ""))
    relative_camera = (
        _camera_relative_to_robot(robot)
        if str(getattr(obj, "name", "")).startswith("online_env_")
        else None
    )
    blockers = _navigation_blocker_aabbs(env, robot, obj)
    collision_geometry = _robot_collision_geometry(robot)
    candidate_yaw = None
    framing_fallbacks = []
    valid_candidates_seen = 0
    rejected = {
        "target_room": 0,
        "room_boundary": 0,
        "native_occupancy": 0,
        "line_of_sight": 0,
        "projection": 0,
        "no_route": 0,
        "operation_distance": 0,
    }

    def _route_exists(candidate_xy, candidate_yaw):
        # Exact-footprint occupancy is validated against raw blocker AABBs, but
        # the navigation BFS runs on the stricter map (robot erosion + clearance
        # erosion + robot-radius-inflated blockers). A stand-off can therefore
        # pass every gate and still sit in a pocket that its own footprint carve
        # cannot connect to the robot's component (Beechwood_0 deliver_drink
        # 2026-08-14: candidate (-2.15, 0.05) was accepted but the
        # ottoman/armchair inflated AABBs severed every corridor, while
        # (-1.75, 0.25) routed cleanly). Gate each accepted candidate with the
        # real clearance-safe BFS so selection never commits to an unroutable
        # stand-off. No erosion, inflation, or margin is relaxed.
        try:
            _connected_navigation_waypoints(
                env,
                robot,
                th.tensor(
                    [
                        float(candidate_xy[0]),
                        float(candidate_xy[1]),
                        float(candidate_yaw),
                    ],
                    dtype=th.float32,
                ),
                clearance_margin=route_clearance_margin,
                target=route_target,
            )
        except ActionPrimitiveError:
            return False
        return True
    # Preserve the preferred stand-off while forbidding geometrically close
    # points across a wall, behind a support, or in another occluded room.
    for candidate_index in th.argsort(candidate_scores).cpu().tolist():
        xy = trav_map.map_to_world(pixels[candidate_index])
        candidate_room = env.scene.seg_map.get_room_instance_by_point(xy[:2])
        if enforce_target_room_visibility and target_rooms and candidate_room not in target_rooms:
            rejected["target_room"] += 1
            continue
        if enforce_target_room_visibility and target_rooms:
            crosses_room_boundary = any(
                env.scene.seg_map.get_room_instance_by_point(
                    (1.0 - alpha) * xy[:2] + alpha * target_position[:2]
                )
                not in target_rooms
                for alpha in np.linspace(0.05, 0.95, 19)
            )
            if crosses_room_boundary:
                rejected["room_boundary"] += 1
                continue
        nearest_target_xy = np.minimum(
            np.maximum(np.asarray(xy[:2].cpu(), dtype=float), target_lower[:2]),
            target_upper[:2],
        )
        target_aabb_distance = float(
            np.linalg.norm(np.asarray(xy[:2].cpu(), dtype=float) - nearest_target_xy)
        )
        if (
            max_target_aabb_distance is not None
            and target_aabb_distance > max_target_aabb_distance
        ):
            rejected["operation_distance"] += 1
            continue
        camera_origin = np.asarray(
            [float(xy[0]), float(xy[1]), float(robot_position[2]) + 1.25],
            dtype=float,
        )
        line_of_sight = False
        for ray_point in (target_center, target_top_center):
            distance = float(np.linalg.norm(ray_point - camera_origin))
            if distance < 1e-6:
                line_of_sight = True
                break
            ray = og.sim.psqi.raycast_closest(
                origin=camera_origin.tolist(),
                dir=((ray_point - camera_origin) / distance).tolist(),
                distance=distance,
            )
            hit_path = str(ray.get("rigidBody") or ray.get("collision") or "")
            hit_distance = float(ray.get("distance", distance))
            blocked = bool(
                ray.get("hit")
                and target_path not in hit_path
                and (not robot_path or robot_path not in hit_path)
                and hit_distance < distance - 0.03
            )
            if not blocked:
                line_of_sight = True
                break
        if not line_of_sight:
            rejected["line_of_sight"] += 1
            continue
        direct_yaw = math.atan2(
            float(target_position[1] - xy[1]),
            float(target_position[0] - xy[0]),
        )
        if relative_camera is not None:
            fallback_yaw = direct_yaw + math.radians(fallback_yaw_offset)
            if _native_occupant_at_pose(
                env, robot, obj, xy, fallback_yaw,
                blockers=blockers, collision_geometry=collision_geometry,
            ) is None:
                framing_fallbacks.append((xy, fallback_yaw))
            projected_yaws = []
            for offset_degrees in (0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0):
                biased_yaw = direct_yaw + math.radians(offset_degrees)
                bbox = _project_candidate_target_bbox(
                    relative_camera, xy, biased_yaw, obj
                )
                if bbox is not None:
                    # Prefer the left half of the image because Fetch's arm
                    # occupies the centre-right, but do not turn that visual
                    # preference into a false navigation failure.
                    projected_yaws.append((max(0, bbox[2] - int(320 * 0.55)), biased_yaw))
            if not projected_yaws:
                rejected["projection"] += 1
                continue
            candidate_yaw = next(
                (
                    yaw
                    for _, yaw in sorted(projected_yaws, key=lambda item: item[0])
                    if _native_occupant_at_pose(
                        env, robot, obj, xy, yaw,
                        blockers=blockers, collision_geometry=collision_geometry,
                    ) is None
                ),
                None,
            )
            if candidate_yaw is None:
                rejected["native_occupancy"] += 1
                continue
        else:
            candidate_yaw = direct_yaw
            if _native_occupant_at_pose(
                env, robot, obj, xy, candidate_yaw,
                blockers=blockers, collision_geometry=collision_geometry,
            ) is not None:
                rejected["native_occupancy"] += 1
                continue
        if valid_candidates_seen < fallback_rank:
            valid_candidates_seen += 1
            continue
        if require_route and not _route_exists(xy, candidate_yaw):
            rejected["no_route"] += 1
            continue
        candidate_xy = xy
        break
    if candidate_xy is None and framing_fallbacks:
        # The fixed Fetch camera's Replicator intrinsics are not exposed by the
        # sensor API used here. Projection is therefore a ranking signal only;
        # the official post-navigation instance mask remains the hard bbox gate.
        ranked_fallbacks = framing_fallbacks[
            min(fallback_rank, len(framing_fallbacks) - 1) :
        ]
        if not require_route:
            candidate_xy, candidate_yaw = ranked_fallbacks[0]
        else:
            for fallback_xy, fallback_yaw in ranked_fallbacks:
                if _route_exists(fallback_xy, fallback_yaw):
                    candidate_xy, candidate_yaw = fallback_xy, fallback_yaw
                    break
    if candidate_xy is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            "No connected observation pose satisfies traversability, clearance, visibility, and framing",
            {
                "object": obj.name,
                "target_rooms": sorted(target_rooms),
                "connected_candidates": int(pixels.shape[0]),
                "rejected": rejected,
            },
        )
    return th.tensor([candidate_xy[0], candidate_xy[1], candidate_yaw], dtype=th.float32)


def _target_framing_distance(obj, robot=None, minimum=1.25):
    """Choose a stand-off that can frame large manipulation supports."""
    try:
        lower, upper = obj.aabb
        size = upper - lower
        xy_diagonal = float(th.linalg.norm(size[:2]))
        tiago = str(getattr(robot, "model", "")).casefold() == "tiago"
        if obj.name.startswith("online_env_") and tiago:
            # Tiago can aim its head, so portable targets should stay near the
            # centre of its manipulation envelope instead of the far boundary.
            minimum = 0.85
        elif obj.name.startswith("online_env_"):
            minimum = max(minimum, 1.75)
        # Fetch's fixed primary-camera stream cannot be re-aimed after its
        # instance annotator is attached in Isaac Sim 5.1. Stand farther from
        # small ground-level targets so they remain inside the default view.
        if (
            obj.name.startswith("online_env_")
            and not tiago
            and float(upper[2]) < 0.5
            and float(size[2]) < 0.4
        ):
            minimum = max(minimum, 2.25)
    except Exception:
        return minimum
    # A conservative 50-degree horizontal framing cone plus base clearance.
    return max(minimum, xy_diagonal / (2.0 * math.tan(math.radians(25.0))) + 0.25)


def _floor_height_below(env, obj):
    """Return the scene floor supporting an object's world-space AABB."""
    aabb_min, aabb_max = obj.aabb
    lower_z = float(aabb_min[2])
    center_z = (lower_z + float(aabb_max[2])) / 2.0
    heights = [float(value) for value in env.scene.trav_map.floor_heights]
    below = [height for height in heights if height <= lower_z + 0.10]
    floor_height = max(below) if below else min(heights, key=lambda value: abs(value - center_z))
    return floor_height, aabb_min, aabb_max


def _horizontal_target_aabb_distance(robot, obj):
    robot_position, _ = robot.get_position_orientation()
    lower, upper = obj.aabb
    robot_xy = np.asarray(robot_position[:2].cpu(), dtype=float)
    lower_xy = np.asarray(lower[:2].cpu(), dtype=float)
    upper_xy = np.asarray(upper[:2].cpu(), dtype=float)
    nearest_xy = np.minimum(np.maximum(robot_xy, lower_xy), upper_xy)
    return float(np.linalg.norm(robot_xy - nearest_xy))


def _manipulation_height_gate(env, step, target, args):
    if step.primitive not in MANIPULATION_PRIMITIVES:
        return None
    if target is None:
        return {
            "required": True,
            "eligible": False,
            "primitive": step.primitive,
            "reason": "manipulation target is missing",
        }
    floor_height, aabb_min, aabb_max = _floor_height_below(env, target)
    min_height = (
        max(args.min_manipulation_height, DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT)
        if args.backend == "physical_control" and step.primitive == "GRASP"
        else args.min_manipulation_height
    )
    result = evaluate_manipulation_height(
        step.primitive,
        float(aabb_min[2]),
        float(aabb_max[2]),
        floor_height,
        min_height=min_height,
        max_height=args.max_manipulation_height,
    )
    result["object_id"] = target.name
    return result


def _connected_navigation_waypoints(
    env, robot, goal_pose, clearance_margin=0.20, spacing=0.10, target=None
):
    """Build collision-clear map waypoints for CuRobo's local base plans."""
    robot_position, robot_orientation = robot.get_position_orientation()
    trav_map = env.scene.trav_map
    floor = min(
        range(trav_map.n_floors),
        key=lambda index: abs(float(trav_map.floor_heights[index]) - float(robot_position[2])),
    )
    traversable = trav_map._erode_trav_map(th.clone(trav_map.floor_map[floor]), robot=robot)
    if clearance_margin > 0:
        clearance_pixels = int(math.ceil(clearance_margin / float(trav_map.map_resolution)))
        kernel_size = max(1, 2 * clearance_pixels + 1)
        traversable = th.as_tensor(
            cv2.erode(
                traversable.cpu().numpy(),
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            ),
            device=traversable.device,
        )
    free = traversable.cpu().numpy() != 0
    collision_geometry = _robot_collision_geometry(robot)
    local_points, robot_z = collision_geometry
    robot_top = robot_z + float(np.max(local_points[:, 2]))
    robot_bottom = robot_z + float(np.min(local_points[:, 2]))
    boundary_horizontal_reach = np.linalg.norm(local_points[:, :2], axis=1)
    for _, lower, upper in _navigation_blocker_aabbs(env, robot, target):
        if upper[2] < robot_bottom + 0.02 or lower[2] > robot_top - 0.02:
            continue
        # Z-aware inflation: a blocker can only be touched by boundary points
        # whose height overlaps the blocker's z range, so inflate by the
        # horizontal reach of just that height band, not the full
        # arm-inclusive radius. Beechwood_0 deliver_drink attempt 5
        # (2026-08-15): full-radius inflation made the 0.40 m ottoman sever
        # the entire corridor to the destination table even though the
        # carried bottle rides ~0.85 m above it and only the 0.405 m
        # base/torso boundary can physically touch it; every valid stand-off
        # candidate failed the route gate and NAVIGATE_TO degenerated into a
        # 0.14 m no-op. Height band margin 0.05 m and horizontal margin
        # 0.05 m; erosion, start/goal carve, and BFS are untouched.
        band_z_min = float(lower[2]) - robot_z - 0.05
        band_z_max = float(upper[2]) - robot_z + 0.05
        band_reach = boundary_horizontal_reach[
            (local_points[:, 2] >= band_z_min) & (local_points[:, 2] <= band_z_max)
        ]
        if not len(band_reach):
            continue
        inflation = float(band_reach.max()) + 0.05
        expanded_lower = lower[:2] - inflation
        expanded_upper = upper[:2] + inflation
        corner_pixels = np.asarray(
            [
                trav_map.world_to_map(th.as_tensor(expanded_lower)).cpu().numpy(),
                trav_map.world_to_map(th.as_tensor(expanded_upper)).cpu().numpy(),
            ],
            dtype=int,
        )
        row_min, col_min = np.maximum(corner_pixels.min(axis=0), 0)
        row_max, col_max = np.minimum(corner_pixels.max(axis=0), np.asarray(free.shape) - 1)
        free[row_min : row_max + 1, col_min : col_max + 1] = False
    # The disk-inflated blocker map above is a yaw-independent worst case:
    # inflating every AABB by its height-band collision reach (even z-aware)
    # can mark the robot's own standing pose AND the stand-off goal as
    # blocked, so snap() sends start and goal to one shared distant free
    # pixel and BFS degenerates into a route through furniture
    # (Beechwood_0 open_fridge 2026-08-14: waypoints
    # [(-4.75, -8.25), goal] from the kitchen pose, robot jammed after 1.4 m).
    # Carve the exact robot footprint at the two demonstrably collision-free
    # poses: the robot physically occupies the start pose, and the goal pose
    # passed the exact-footprint occupant check (0.10 m margin) or the
    # CuRobo-validated saved approach. Mid-route erosion and inflation are
    # untouched.
    current_yaw = math.atan2(
        2.0 * (float(robot_orientation[3]) * float(robot_orientation[2])
               + float(robot_orientation[0]) * float(robot_orientation[1])),
        1.0 - 2.0 * (float(robot_orientation[1]) ** 2 + float(robot_orientation[2]) ** 2),
    )
    carve_poses = (
        (np.asarray(robot_position[:2].cpu(), dtype=float), current_yaw),
        (np.asarray(goal_pose[:2].cpu(), dtype=float), float(goal_pose[2])),
    )
    for carve_xy, carve_yaw in carve_poses:
        carve_points = _robot_collision_points_at_pose(
            robot, carve_xy, carve_yaw, collision_geometry=collision_geometry
        )
        center = carve_xy
        for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
            ring = center + (carve_points[:, :2] - center) * scale
            batched = trav_map.world_to_map(th.as_tensor(ring, dtype=th.float32))
            # world_to_map flips every tensor dim; for (N, 2) input that also
            # reverses point order, which is undone here (the per-point
            # component swap into (row, col) is intended).
            pixels = np.asarray(batched.cpu(), dtype=int)[::-1]
            pixels = np.clip(pixels, [0, 0], np.asarray(free.shape) - 1)
            free[pixels[:, 0], pixels[:, 1]] = True
    free_pixels = np.argwhere(free)
    if not len(free_pixels):
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            "No clearance-safe traversable pixels for navigation",
            {"floor": floor, "clearance_margin": clearance_margin},
        )

    def snap(world_xy):
        pixel = np.asarray(trav_map.world_to_map(world_xy).cpu(), dtype=int)
        pixel = np.clip(pixel, [0, 0], np.asarray(free.shape) - 1)
        if free[tuple(pixel)]:
            return tuple(int(value) for value in pixel)
        nearest = np.argmin(np.linalg.norm(free_pixels - pixel, axis=1))
        return tuple(int(value) for value in free_pixels[nearest])

    start = snap(robot_position[:2])
    goal = snap(goal_pose[:2])
    parents = np.full((*free.shape, 2), -2, dtype=np.int32)
    parents[start] = start
    queue = deque([start])
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while queue and tuple(parents[goal]) == (-2, -2):
        row, col = queue.popleft()
        for drow, dcol in neighbors:
            nxt = (row + drow, col + dcol)
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]):
                continue
            if not free[nxt] or tuple(parents[nxt]) != (-2, -2):
                continue
            if drow and dcol and (not free[row + drow, col] or not free[row, col + dcol]):
                continue
            parents[nxt] = (row, col)
            queue.append(nxt)
    if tuple(parents[goal]) == (-2, -2):
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PLANNING_ERROR,
            "No clearance-safe map path to manipulation stand-off",
            {"floor": floor, "clearance_margin": clearance_margin},
        )
    path = [goal]
    while path[-1] != start:
        path.append(tuple(int(value) for value in parents[path[-1]]))
    path.reverse()
    stride = max(1, int(math.ceil(spacing / float(trav_map.map_resolution))))
    selected = path[stride::stride]
    if not selected or selected[-1] != goal:
        selected.append(goal)
    world_points = [trav_map.map_to_world(th.tensor(pixel, device=traversable.device)) for pixel in selected]
    exact_goal = th.as_tensor(goal_pose[:2], device=traversable.device)
    if float(th.linalg.norm(world_points[-1][:2] - exact_goal)) > 0.05:
        # The clearance map may snap the manipulation stand-off slightly away
        # from a table. Keep the safe route, then make one short local approach
        # to the robot-footprint-valid pose selected above.
        world_points.append(exact_goal)
    waypoints = []
    for index, point in enumerate(world_points):
        if index + 1 < len(world_points):
            following = world_points[index + 1]
            yaw = math.atan2(float(following[1] - point[1]), float(following[0] - point[0]))
        else:
            yaw = float(goal_pose[2])
        waypoints.append(th.tensor([point[0], point[1], yaw], dtype=th.float32))
    return waypoints


def _closed_doors_on_route(env, robot, target, waypoints, route_margin=0.35):
    """Return closed doors intersected by a planned map route, in route order."""
    robot_position, _ = robot.get_position_orientation()
    route = [np.asarray(robot_position[:2].cpu(), dtype=float)]
    route.extend(np.asarray(waypoint[:2].cpu(), dtype=float) for waypoint in waypoints)
    prerequisites = []
    for door in _scene_objects(env).values():
        if door is target or str(getattr(door, "category", "")).lower() != "door":
            continue
        if object_states.Open not in getattr(door, "states", {}):
            continue
        if bool(door.states[object_states.Open].get_value()):
            continue
        lower, upper = door.aabb
        lower = np.asarray(lower[:2].cpu(), dtype=float)
        upper = np.asarray(upper[:2].cpu(), dtype=float)
        route_distances = [
            float(np.linalg.norm(np.maximum(np.maximum(lower - point, point - upper), 0.0)))
            for point in route
        ]
        hit_indices = [
            index for index, distance in enumerate(route_distances) if distance <= route_margin
        ]
        if not hit_indices:
            continue
        hit_index = hit_indices[0]
        approach_index = hit_index
        while approach_index > 0 and route_distances[approach_index] < 0.60:
            approach_index -= 1
        door_position, _ = door.get_position_orientation()
        approach = route[approach_index]
        yaw = math.atan2(
            float(door_position[1]) - approach[1],
            float(door_position[0]) - approach[0],
        )
        prerequisites.append(
            {
                "door": door,
                "route_index": hit_index,
                "approach_pose": th.tensor(
                    [approach[0], approach[1], yaw], dtype=th.float32
                ),
                "approach_distance": route_distances[approach_index],
            }
        )
    return sorted(prerequisites, key=lambda item: item["route_index"])


class DeltaSGOraclePrimitives(SymbolicSemanticActionPrimitives):
    """Symbolic primitives with traversability-map navigation for Fetch.

    OmniGibson's symbolic class skips CuRobo construction but inherits a
    NAVIGATE_TO sampler that still dereferences CuRobo.  This implementation
    chooses a robot-eroded traversable point near the object, verifies map
    connectivity, faces the target, and then uses the symbolic pose executor.
    """

    def __init__(self, *args, **kwargs):
        self._deltasg_inventory_object = None
        self._deltasg_inventory_gravity_disabled = False
        self.last_navigation_prerequisites = []
        super().__init__(*args, **kwargs)

    def _get_obj_in_hand(self):
        return self._deltasg_inventory_object

    def _set_inventory_gravity(self, enabled):
        held = self._get_obj_in_hand()
        if held is None:
            return
        for link in held.links.values():
            if enabled:
                link.enable_gravity()
            else:
                link.disable_gravity()
        self._deltasg_inventory_gravity_disabled = not enabled

    def _set_inventory_collisions(self, enabled):
        held = self._get_obj_in_hand()
        if held is None:
            return
        for link in held.links.values():
            if enabled:
                link.enable_collisions()
            else:
                link.disable_collisions()

    def _sync_inventory_to_eef(self):
        held = self._get_obj_in_hand()
        if held is None:
            return
        held.set_position_orientation(position=self.robot.get_eef_position(self.arm))
        held.keep_still()

    def _grasp(self, obj):
        held = self._get_obj_in_hand()
        if held is not None:
            if held is obj:
                return
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot grasp when the oracle inventory is already occupied",
                {"target object": obj.name, "object currently in hand": held.name},
            )

        # The upstream symbolic primitive creates an assisted-grasp FixedJoint.
        # Isaac Sim 5.1 can segfault in that native path for runtime-added floor
        # objects. Oracle validation only needs deterministic inventory and
        # state transitions; physical_control retains the official real grasp.
        eef_position = self.robot.get_eef_position(self.arm)
        obj.set_position_orientation(position=eef_position)
        obj.keep_still()
        self._deltasg_inventory_object = obj
        self._set_inventory_gravity(False)
        self._set_inventory_collisions(False)
        if False:
            yield None

    def _release(self):
        held = self._get_obj_in_hand()
        if held is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot release when the oracle inventory is empty",
            )
        held.keep_still()
        self._set_inventory_collisions(True)
        self._set_inventory_gravity(True)
        self._deltasg_inventory_object = None
        if False:
            yield None

    def _open_or_close(self, obj, should_open):
        """Symbolic open/close with a deterministic OPEN and CLOSE.

        Mirrors OmniGibson's SymbolicSemanticActionPrimitives._open_or_close,
        except both directions drive the joint to the hard end (``fully=True``).
        The default ``set_value(...)`` samples a random position in the 5%
        band next to the target end; for horizontal-hinge doors (e.g. fridge
        dszchb, axis=X) a sample near the OPEN side drifts back closed while a
        sample near the CLOSED side drifts back open within ~20 sim steps, so
        the post-settle assertion would spuriously fail. Driving to the hard
        closed end (0) or hard open end (upper limit) is stable in both
        directions. This matches the generator's own deterministic transitions.
        """
        if self._get_obj_in_hand():
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand().name},
            )
        if object_states.Open not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not openable.",
                {"target object": obj.name},
            )
        if should_open == obj.states[object_states.Open].get_value():
            return
        obj.states[object_states.Open].set_value(should_open, fully=True)
        yield from self._settle_robot()
        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The object did not open or close as expected. Maybe try again",
                {"target object": obj.name, "is it currently open": obj.states[object_states.Open].get_value()},
            )

    def _place_with_predicate(self, obj, predicate, near_poses=None, near_poses_threshold=None):
        """Apply the official symbolic relation before clearing inventory."""
        held = self._get_obj_in_hand()
        if held is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "You need to be grasping an object first to place it somewhere.",
            )
        state = held.states.get(predicate)
        if state is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Held object does not expose {predicate.__name__}",
                {"held object": held.name, "target object": obj.name},
            )
        # Unified wall-bounded official placement. Both native and generated
        # supports use OmniGibson's official OnTop/Inside setter, which runs its
        # own kinematic sample + settle + state verification. The previous
        # native-support branch instead looped 40x around the cuboid sampler
        # and, on every in-envelope candidate, dumped/settled/reloaded the scene
        # while the held object was in half-grasped inventory state; that
        # emitted repeated `Illegal BroadPhaseUpdateData` and stalled the
        # persistent scene on deliver_medicine (~7 minutes, no result). The
        # setter's high/low-level sampling attempts are clamped here so a single
        # set_value call cannot spend minutes, and a wall-clock deadline plus a
        # bounded retry count make a failed support reject this sample instead
        # of hanging the process. The official predicate and the strict 1.15 m
        # AABB-edge distance gate remain the only acceptance for the final state.
        from omnigibson.utils.object_state_utils import m as object_state_macros

        old_high = object_state_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS
        old_low = object_state_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS
        changed = False
        reached = False
        placed_object_distance = math.inf
        deadline = time.monotonic() + PLACE_NATIVE_MAX_WALL_SECONDS
        attempts_tried = 0
        try:
            with object_state_macros.unlocked():
                object_state_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 2
                object_state_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 2
            for _ in range(PLACE_NATIVE_MAX_ATTEMPTS):
                if time.monotonic() > deadline:
                    break
                attempts_tried += 1
                self._set_inventory_collisions(True)
                self._set_inventory_gravity(True)
                changed = bool(state.set_value(obj, True))
                reached = bool(state.get_value(obj))
                if changed and reached:
                    placed_object_distance = _horizontal_target_aabb_distance(
                        self.robot, held
                    )
                    if placed_object_distance <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE:
                        break
                held.set_position_orientation(position=self.robot.get_eef_position(self.arm))
                held.keep_still()
                self._set_inventory_gravity(False)
                self._set_inventory_collisions(False)
        finally:
            with object_state_macros.unlocked():
                object_state_macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = old_high
                object_state_macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = old_low
        if (
            not changed
            or not reached
            or placed_object_distance > DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Official symbolic relation setter did not reach the requested state inside the operation envelope",
                {
                    "held object": held.name,
                    "target object": obj.name,
                    "predicate": predicate.__name__,
                    "state_set_succeeded": changed,
                    "state_reached": reached,
                    "placed_object_horizontal_distance": placed_object_distance,
                    "attempts_tried": attempts_tried,
                    "wall_deadline_seconds": PLACE_NATIVE_MAX_WALL_SECONDS,
                    "max_placed_object_horizontal_distance": (
                        DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                    ),
                },
            )
        yield from self._release()

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        # Fetch's default arm posture extends in front of the base. A 0.85 m
        # oracle approach can strike small tabletop targets before the grasp;
        # keep a wider observation pose and let symbolic GRASP bridge reach.
        restore_target_pose = None
        if obj.name.startswith("online_env_") and self._get_obj_in_hand() is not obj:
            position, orientation = obj.get_position_orientation()
            restore_target_pose = (position.clone(), orientation.clone())
        saved_xy = None
        if (
            not obj.name.startswith("online_env_")
            and int(getattr(self, "_deltasg_navigation_fallback_rank", 0)) == 0
        ):
            saved_xy = getattr(self, "_deltasg_saved_robot_approaches", {}).get(obj.name)
        if saved_xy is not None:
            lower, upper = obj.aabb
            saved_xy_array = np.asarray(saved_xy, dtype=float)
            nearest_xy = np.minimum(
                np.maximum(saved_xy_array, np.asarray(lower[:2].cpu(), dtype=float)),
                np.asarray(upper[:2].cpu(), dtype=float),
            )
            candidate_room = self.env.scene.seg_map.get_room_instance_by_point(saved_xy_array)
            target_rooms = set(getattr(obj, "in_rooms", None) or [])
            if (
                float(np.linalg.norm(saved_xy_array - nearest_xy))
                > DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                or (target_rooms and candidate_room not in target_rooms)
            ):
                saved_xy = None
        if saved_xy is not None:
            target_position, _ = obj.get_position_orientation()
            candidate_pose = th.tensor(
                [
                    float(saved_xy[0]),
                    float(saved_xy[1]),
                    math.atan2(
                        float(target_position[1]) - float(saved_xy[1]),
                        float(target_position[0]) - float(saved_xy[0]),
                    ),
                ],
                dtype=th.float32,
            )
            route = [candidate_pose]
        else:
            # Generated task objects must be revalidated against the live map
            # and official post-navigation segmentation on every replay.
            candidate_pose = _connected_observation_pose(
                self.env,
                self.robot,
                obj,
                preferred_distance=_target_framing_distance(obj, robot=self.robot),
                fallback_rank=int(getattr(self, "_deltasg_navigation_fallback_rank", 0)),
                max_target_aabb_distance=DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
            )
            route = _connected_navigation_waypoints(
                self.env, self.robot, candidate_pose, clearance_margin=0.0, spacing=0.10
            )
        self.last_navigation_prerequisites = []
        for prerequisite in _closed_doors_on_route(
            self.env, self.robot, obj, route
        ):
            door = prerequisite["door"]
            yield from self._navigate_to_pose(prerequisite["approach_pose"])
            before = bool(door.states[object_states.Open].get_value())
            changed = bool(door.states[object_states.Open].set_value(True))
            yield from self._settle_robot()
            after = bool(door.states[object_states.Open].get_value())
            if not changed or not after:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Could not open a closed door on the navigation route",
                    {"door": door.name, "before": before, "after": after},
                )
            self.last_navigation_prerequisites.append(
                {
                    "primitive": "OPEN",
                    "object_id": door.name,
                    "state_before": before,
                    "state_after": after,
                    "approach_pose_xyyaw": _jsonable(prerequisite["approach_pose"]),
                    "approach_distance": prerequisite["approach_distance"],
                }
            )
        yield from self._navigate_to_pose(candidate_pose)
        if restore_target_pose is not None:
            obj.set_position_orientation(
                position=restore_target_pose[0], orientation=restore_target_pose[1]
            )
            obj.keep_still()

    def _navigate_to_pose(self, pose_2d):
        current_position, _ = self.robot.get_position_orientation()
        yaw = float(pose_2d[2])
        robot_pose = (
            th.tensor([float(pose_2d[0]), float(pose_2d[1]), float(current_position[2])], dtype=th.float32),
            th.tensor([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=th.float32),
        )
        nav_diag = os.environ.get("DELTASG_NAV_DIAG") == "1"
        pre_centers = {}
        if nav_diag:
            for name, obj in _scene_objects(self.env).items():
                if not name or name.startswith("online_env_") or obj is self.robot:
                    continue
                try:
                    lower, upper = obj.aabb
                    pre_centers[name] = (
                        (lower + upper) * 0.5,
                        np.asarray(lower.cpu(), dtype=float),
                        np.asarray(upper.cpu(), dtype=float),
                    )
                except Exception:
                    continue
        _teleport_robot_preserving_delta_objects(self.env, self.robot, *robot_pose)
        if nav_diag:
            for name, (pre_center, lower, upper) in pre_centers.items():
                points = _robot_collision_points_at_pose(
                    self.robot, (float(pose_2d[0]), float(pose_2d[1])), float(pose_2d[2])
                )
                inside = np.all(points >= lower - 0.01, axis=1) & np.all(points <= upper + 0.01, axis=1)
                if np.any(inside):
                    print(
                        f"[nav-diag] teleport target=({float(pose_2d[0]):.3f},{float(pose_2d[1]):.3f}) "
                        f"robot boundary OVERLAPS {name} AABB at margin 0.01",
                        flush=True,
                    )
        yield from self._settle_robot()
        self._sync_inventory_to_eef()
        if nav_diag:
            for name, (pre_center, _, _) in pre_centers.items():
                obj = self.env.scene.object_registry("name", name, None)
                if obj is None:
                    continue
                try:
                    lower, upper = obj.aabb
                    moved = float(th.linalg.norm(((lower + upper) * 0.5) - pre_center))
                except Exception:
                    continue
                if moved > 0.01:
                    print(
                        f"[nav-diag] teleport target=({float(pose_2d[0]):.3f},{float(pose_2d[1]):.3f}) "
                        f"{name} moved={moved:.4f} during NAVIGATE_TO settle",
                        flush=True,
                    )

    def _settle_robot(self):
        # Fetch's default base controller cannot be converted through
        # Robot.q_to_action(), which the starter primitive assumes. Oracle
        # validation does not export low-level actions, so settle directly.
        for _ in range(5):
            self.robot.keep_still()
            with og.sim.render_on_step(False):
                og.sim.step()
        if False:
            yield None


class DeltaSGPhysicalPrimitives(StarterSemanticActionPrimitives):
    """Starter primitives with deterministic connected-component navigation."""

    def _sample_grasp_pose(self, obj):
        attempts = getattr(self, "_deltasg_grasp_pose_attempts", defaultdict(int))
        self._deltasg_grasp_pose_attempts = attempts
        attempt = attempts[obj.name]
        attempts[obj.name] += 1
        if not obj.name.startswith("online_env_") or attempt % 2 == 1:
            return super()._sample_grasp_pose(obj)

        lower, upper = obj.aabb
        center = (lower + upper) * 0.5
        robot_position, _ = self.robot.get_position_orientation()
        approach = center - robot_position
        approach[2] = 0.0
        approach = approach / th.clamp(th.linalg.norm(approach), min=1e-6)

        # Keep the gripper clear of the support while approaching the object
        # horizontally from the robot-facing side.
        grasp_position = center.clone()
        grasp_position[2] = upper[2] - min(0.02, float((upper[2] - lower[2]) * 0.2))
        local_z = -approach
        local_x = th.tensor([0.0, 0.0, 1.0], dtype=th.float32, device=center.device)
        local_y = th.linalg.cross(local_z, local_x)
        local_y = local_y / th.clamp(th.linalg.norm(local_y), min=1e-6)
        grasp_frame = th.stack((local_x, local_y, local_z), dim=1)
        grasp_quaternion = T.quat_multiply(
            T.mat2quat(grasp_frame),
            th.tensor([1.0, 0.0, 0.0, 0.0], dtype=th.float32, device=center.device),
        )
        fingertip_offset = th.mean(
            th.tensor(
                list(self.robot.eef_to_fingertip_lengths[self.arm].values()),
                dtype=th.float32,
                device=center.device,
            )
        )
        pregrasp_position = grasp_position - approach * (
            fingertip_offset + starter_primitives.m.GRASP_APPROACH_DISTANCE
        )
        return (pregrasp_position, grasp_quaternion), (grasp_position, grasp_quaternion)

    def _sample_pose_near_object(
        self,
        obj,
        eef_pose=None,
        plan_with_open_gripper=False,
        sampling_attempts=starter_primitives.m.MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT,
        skip_obstacle_update=False,
    ):
        saved_xy = getattr(self, "_deltasg_saved_robot_approaches", {}).get(obj.name)
        if saved_xy is not None and eef_pose is not None:
            target_position, _ = obj.get_position_orientation()
            yaw = math.atan2(
                float(target_position[1]) - float(saved_xy[1]),
                float(target_position[0]) - float(saved_xy[0]),
            )
            candidate = th.tensor(
                [float(saved_xy[0]), float(saved_xy[1]), yaw], dtype=th.float32
            )
            valid = bool(
                self._validate_poses(
                    [candidate],
                    eef_pose=eef_pose,
                    plan_with_open_gripper=plan_with_open_gripper,
                    skip_obstacle_update=skip_obstacle_update,
                )[0]
            )
            hover_valid = None
            hover_pose = self._place_hover_pose(eef_pose)
            if valid and hover_pose is not None:
                hover_valid = bool(
                    self._validate_poses(
                        [candidate],
                        eef_pose=hover_pose,
                        plan_with_open_gripper=plan_with_open_gripper,
                        skip_obstacle_update=skip_obstacle_update,
                    )[0]
                )
            diagnostics = getattr(self, "physical_approach_diagnostics", {})
            self.physical_approach_diagnostics = diagnostics
            diagnostics.setdefault(obj.name, {})["saved_generation_pose_validation"] = {
                "candidate_pose_xyyaw": _jsonable(candidate),
                "arm_reachable_and_collision_free": valid,
                "hover_reachable_and_collision_free": hover_valid,
            }
            if valid and hover_valid is not False:
                return candidate
        pose = super()._sample_pose_near_object(
            obj,
            eef_pose=eef_pose,
            plan_with_open_gripper=plan_with_open_gripper,
            sampling_attempts=sampling_attempts,
            skip_obstacle_update=skip_obstacle_update,
        )
        if pose is not None and eef_pose is not None:
            diagnostics = getattr(self, "physical_approach_diagnostics", {})
            self.physical_approach_diagnostics = diagnostics
            validation = {
                "candidate_pose_xyyaw": _jsonable(pose),
            }
            hover_pose = self._place_hover_pose(eef_pose)
            if hover_pose is not None:
                hover_valid = bool(
                    self._validate_poses(
                        [pose],
                        eef_pose=hover_pose,
                        plan_with_open_gripper=plan_with_open_gripper,
                        skip_obstacle_update=skip_obstacle_update,
                    )[0]
                )
                validation["hover_reachable_and_collision_free"] = hover_valid
                if not hover_valid:
                    validation["rejected_reason"] = "place_hover_not_reachable"
                    diagnostics.setdefault(obj.name, {})[
                        "official_validated_navigation_pose"
                    ] = validation
                    return None
            diagnostics.setdefault(obj.name, {})[
                "official_validated_navigation_pose"
            ] = validation
        return pose

    def _place_hover_pose(self, hand_pose):
        if getattr(self, "_deltasg_place_hover", None) is None:
            return None
        hover_position = th.as_tensor(hand_pose[0], dtype=th.float32).clone()
        hover_position[2] += PLACE_ACCESS_HOVER_CLEARANCE
        return hover_position, hand_pose[1]

    def _base_joint_values_for_world_pose(self, pose_2d):
        """Convert an absolute-world (x, y, yaw) base candidate to holonomic base joint values.

        Tiago's holonomic base pins the root link (base_footprint_x) at the spawn
        pose with a fixed joint; the six virtual base joints encode base_footprint
        RELATIVE to that root link, and CuRobo reads them in that frame (its
        obstacles and IK targets are transformed by the inverse live root-link
        pose). The Starter convention of writing absolute world coordinates into
        the base joints therefore places the CuRobo model base at
        spawn_pose @ candidate_pose instead of candidate_pose, which silently
        breaks every collision and reachability gate (verified: diag9 P2/P3).
        This helper applies the same convention as Robot.set_position_orientation.
        """
        root_pos, root_orn = self.robot.root_link.get_position_orientation()
        body_pos, body_orn = self.robot.base_footprint_link.get_position_orientation()
        body_euler = T.quat2euler(body_orn)
        cand_pos = th.tensor(
            [float(pose_2d[0]), float(pose_2d[1]), float(body_pos[2])], dtype=th.float32
        )
        cand_orn = T.euler2quat(
            th.tensor(
                [float(body_euler[0]), float(body_euler[1]), float(pose_2d[2])],
                dtype=th.float32,
            )
        )
        rel_pos, rel_orn = T.relative_pose_transform(cand_pos, cand_orn, root_pos, root_orn)
        rel_euler = T.mat2euler_intrinsic(T.quat2mat(rel_orn))
        return th.cat([rel_pos, rel_euler]).float()

    def _validate_poses(self, candidate_poses, eef_pose=None, plan_with_open_gripper=False, skip_obstacle_update=False):
        """Starter validation with holonomic base candidates written root-relative.

        Identical gates to the official implementation (batched collision with the
        held object attached, then world-collision IK reachability per surviving
        candidate); only the base joint encoding of each candidate changes.
        """
        base_idx = getattr(self.robot, "base_idx", None)
        if base_idx is None or len(base_idx) != 6:
            return super()._validate_poses(
                candidate_poses,
                eef_pose=eef_pose,
                plan_with_open_gripper=plan_with_open_gripper,
                skip_obstacle_update=skip_obstacle_update,
            )
        if plan_with_open_gripper:
            current_joint_pos = self._get_joint_position_with_fingers_at_limit("upper")
        else:
            current_joint_pos = self.robot.get_joint_positions()
        candidate_joint_positions = []
        for pose in candidate_poses:
            joint_pos = current_joint_pos.clone()
            joint_pos[base_idx] = self._base_joint_values_for_world_pose(pose)
            candidate_joint_positions.append(joint_pos)
        candidate_joint_positions = th.stack(candidate_joint_positions)

        obj_in_hand = self._get_obj_in_hand()
        attached_obj = (
            {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link}
            if obj_in_hand is not None
            else None
        )
        invalid_results = self._motion_generator.check_collisions(
            candidate_joint_positions,
            self_collision_check=False,
            skip_obstacle_update=skip_obstacle_update,
            attached_obj=attached_obj,
        ).cpu()
        for i in range(len(candidate_poses)):
            if invalid_results[i].item():
                continue
            if eef_pose is not None:
                if not self._target_in_reach_of_robot(
                    eef_pose,
                    initial_joint_pos=candidate_joint_positions[i],
                    skip_obstacle_update=skip_obstacle_update,
                ):
                    invalid_results[i] = True
        return ~invalid_results

    def _move_hand(self, target_pose, *args, ignore_objects=None, **kwargs):
        held = self._get_obj_in_hand()
        if held is not None and held in (ignore_objects or []):
            # Sticky grasp approaches intentionally remove the just-grasped
            # object from world collision checking. Attaching that same object
            # to CuRobo in this state can dereference its removed mesh cache.
            # Solve the short endpoint with collision-aware IK, then actuate the
            # resulting official joint target through the normal controller.
            self._motion_generator.update_obstacles(ignore_objects=ignore_objects)
            joint_position = self._convert_cartesian_to_joint_space(target_pose)
            yield from self._move_hand_direct_joint(joint_position)
            return
        hover_target = getattr(self, "_deltasg_place_hover", None)
        if hover_target is not None and held is not None:
            # Fix N (attempt-7 / diag27, Beechwood_0 deliver_drink): the stock
            # single CuRobo plan from the folded carry pose to an edge place
            # pose swept the arm around the support at carry height and made
            # real contact the sphere model missed (stall residuals
            # 0.005-0.019 rad). Consume the armed flag once and approach via a
            # hover waypoint directly above the place pose instead; the final
            # leg is then a short near-vertical descent. Both legs keep the
            # full official CuRobo plan + execution monitor (no standard is
            # relaxed).
            self._deltasg_place_hover = None
            hand_pos, hand_orn = target_pose
            hover_pos = th.as_tensor(hand_pos, dtype=th.float32).clone()
            hover_pos[2] = float(hover_pos[2]) + PLACE_ACCESS_HOVER_CLEARANCE
            self.physical_approach_diagnostics = getattr(
                self, "physical_approach_diagnostics", {}
            )
            self.physical_approach_diagnostics.setdefault(hover_target.name, {})[
                "place_hover"
            ] = {
                "hand_pose_position": _jsonable(hand_pos),
                "hover_pose_position": _jsonable(hover_pos),
                "hover_clearance_m": PLACE_ACCESS_HOVER_CLEARANCE,
                "intermediate_low_precision": True,
            }
            hover_kwargs = dict(kwargs)
            hover_kwargs["low_precision"] = True
            yield from super()._move_hand(
                (hover_pos, hand_orn),
                *args,
                ignore_objects=ignore_objects,
                **hover_kwargs,
            )
        yield from super()._move_hand(
            target_pose, *args, ignore_objects=ignore_objects, **kwargs
        )
        place_target = getattr(self, "_deltasg_place_target", None)
        held_after_move = self._get_obj_in_hand()
        if place_target is not None and held_after_move is not None:
            actual_hand_pose = self.robot.eef_links[
                self.arm
            ].get_position_orientation()
            diagnostics = getattr(self, "physical_approach_diagnostics", {})
            self.physical_approach_diagnostics = diagnostics
            commanded_position = th.as_tensor(target_pose[0], dtype=th.float32)
            actual_position = th.as_tensor(actual_hand_pose[0], dtype=th.float32)
            held_low, held_high = held_after_move.aabb
            diagnostics.setdefault(place_target.name, {})[
                "place_final_hand"
            ] = {
                "commanded_position": _jsonable(commanded_position),
                "actual_position": _jsonable(actual_position),
                "position_error_m": round(
                    float(th.linalg.norm(commanded_position - actual_position)), 4
                ),
                "held_object_aabb": [_jsonable(held_low), _jsonable(held_high)],
            }

    def _place_with_predicate(self, obj, predicate):
        """Official place flow armed with the Fix N access gates.

        The official flow runs unchanged; for its duration two DeltaSG gates
        are armed: access-aware OnTop candidate validation inside
        ``_sample_pose_with_object_and_predicate`` (clear descent corridor +
        manipulability reach margin) and the hover-then-descend approach in
        ``_move_hand``. Every official collision/reachability gate still
        applies on top.
        """
        self._deltasg_place_hover = obj if predicate is object_states.OnTop else None
        self._deltasg_place_target = obj if predicate is object_states.OnTop else None
        self._deltasg_place_access = None
        try:
            yield from super()._place_with_predicate(obj, predicate)
        finally:
            self._deltasg_place_hover = None
            self._deltasg_place_target = None

    def _place_corridor_obstacle_aabbs(self, held_obj, target_obj):
        """Scene AABBs that can block a place descent corridor.

        Excludes the placement target, the held object and the robot; the
        intersection test itself discards everything below the corridor.
        """
        aabbs = []
        for name, obj in _scene_objects(self.env).items():
            if obj is target_obj or obj is held_obj or obj is self.robot:
                continue
            try:
                low, high = obj.aabb
            except Exception:
                continue
            aabbs.append((name, (_to_numpy(low), _to_numpy(high))))
        return aabbs

    def _record_place_access_diagnostics(self, target_obj, accepted_pose, fallback_used):
        state = getattr(self, "_deltasg_place_access", None) or {}
        self.physical_approach_diagnostics = getattr(
            self, "physical_approach_diagnostics", {}
        )
        self.physical_approach_diagnostics.setdefault(target_obj.name, {})[
            "place_access"
        ] = {
            "candidates_examined": state.get("examined", 0),
            "accepted_pose_position": (
                None if accepted_pose is None else _jsonable(accepted_pose[0])
            ),
            "accepted_pose_orientation": (
                None if accepted_pose is None else _jsonable(accepted_pose[1])
            ),
            "fallback_used": bool(fallback_used),
            "rejections": state.get("rejections", []),
            "reach_margin_m": PLACE_ACCESS_REACH_MARGIN,
            "hover_clearance_m": PLACE_ACCESS_HOVER_CLEARANCE,
            "corridor_margin_m": PLACE_ACCESS_CORRIDOR_MARGIN,
            "surface_rejections": state.get("corrupted", 0),
            "support_rejections": state.get("support_rejected", 0),
            "accepted_support_check": state.get("accepted_support"),
        }

    def _place_sampling_surface_corruption(self, target_obj, pose, held_obj):
        """Fix P/P2/P3: reject draws the live sampling surface cannot legitimately produce.

        diag30 verified that clean OnTop draws on this scene land strictly inside
        the target's base-aligned footprint at z ~= support top + half held
        height + PREDICATE_SAMPLING_Z_OFFSET (0.673/0.683 for the Beechwood_0
        breakfast table). attempt-9 instead drew a ring 0.14-0.60 m outside the
        footprint at z=0.780 because get_base_aligned_bbox transiently reported
        an inflated box after the grasp/stop-play sequence. attempt-10 showed a
        second corruption class: ALL draws uniformly inflated to z=0.7626 vs
        z_expected 0.672 while xy stayed near the footprint. The reference
        geometry is snapshotted pre-grasp (_capture_reference_scene_bboxes);
        legitimate drift is bounded by the replay-integrity displacement budget
        (<= 0.05 m) plus the sampler's 2% ray offset, so the margins below
        exclude no legitimate candidate.

        Fix P3 (adversarial review of P2): every quantity the gate compares is
        taken from the pre-grasp reference snapshot, never from live geometry
        that may itself be corrupted (attempt-10 inflated the live held
        base-aligned extent to ~0.47 m; a live held read would have inflated
        z_expected with it and masked the deviation), and the z tolerances are
        derived from the sampler's ray geometry: rays start/end at the target
        base-aligned bbox expanded by PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION
        of its height, so a legitimate candidate deviates from
        (reference aabb top + half reference held height) by at most
        f*h + Z_OFFSET upward and (1+f)*h + Z_OFFSET downward. This admits
        tiered supports (a sofa seat sits far below the backrest-top AABB top)
        while still excluding both observed corruption signatures. Returns None
        when the draw is consistent with the reference surface, else a detail
        dict that also captures the live sampler inputs so the corruption
        source can be adjudicated.
        """
        references = getattr(self, "_deltasg_reference_bboxes", None) or {}
        reference = references.get(getattr(target_obj, "name", None))
        if reference is None:
            return None
        held_reference = references.get(getattr(held_obj, "name", None))
        try:
            center = np.asarray(reference["bbox_center"], dtype=float)
            extent = np.asarray(reference["bbox_extent"], dtype=float)
            position = np.asarray(
                [float(v) for v in pose[0]], dtype=float
            )
            if held_reference is not None:
                held_extent_z = float(
                    np.asarray(held_reference["bbox_extent"], dtype=float)[2]
                )
            else:
                held_extent_z = float(_to_numpy(held_obj.aabb_extent)[2])
        except Exception:
            return None
        half_xy = extent[:2] * 0.5 + PLACE_SAMPLING_SURFACE_XY_MARGIN
        overshoot = np.maximum(
            (center[:2] - half_xy) - position[:2],
            position[:2] - (center[:2] + half_xy),
        )
        xy_outside = float(max(0.0, float(np.max(overshoot))))
        z_expected = float(reference["aabb_top"]) + held_extent_z * 0.5
        z_deviation = position[2] - z_expected
        ref_height = float(extent[2])
        z_offset = getattr(
            starter_primitives.m, "PREDICATE_SAMPLING_Z_OFFSET", 0.02
        )
        z_up_tolerance = (
            PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION * ref_height
            + z_offset
            + PLACE_SAMPLING_SURFACE_Z_EPSILON
        )
        z_down_tolerance = (
            (1.0 + PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION) * ref_height
            + z_offset
            + PLACE_SAMPLING_SURFACE_Z_EPSILON
        )
        if (
            xy_outside <= 0.0
            and -z_down_tolerance <= z_deviation <= z_up_tolerance
        ):
            return None
        sampler_inputs = {}
        try:
            _, _, held_base_extent, _ = held_obj.get_base_aligned_bbox()
            sampler_inputs["held_base_aligned_extent"] = [
                round(float(v), 3) for v in _to_numpy(held_base_extent)
            ]
        except Exception:
            pass
        try:
            ray_center, _, ray_extent, _ = target_obj.get_base_aligned_bbox(
                xy_aligned=True
            )
            sampler_inputs["target_ray_box_center"] = [
                round(float(v), 3) for v in _to_numpy(ray_center)
            ]
            sampler_inputs["target_ray_box_extent"] = [
                round(float(v), 3) for v in _to_numpy(ray_extent)
            ]
        except Exception:
            pass
        sampler_inputs["held_aabb_extent"] = [
            round(float(v), 3) for v in _to_numpy(held_obj.aabb_extent)
        ]
        return {
            "reference_bbox_center": [round(float(v), 3) for v in center],
            "reference_bbox_extent": [round(float(v), 3) for v in extent],
            "reference_aabb_top": float(reference["aabb_top"]),
            "xy_outside_m": round(xy_outside, 3),
            "z_expected": round(z_expected, 3),
            "z_deviation_m": round(z_deviation, 3),
            "z_up_tolerance_m": round(z_up_tolerance, 3),
            "z_down_tolerance_m": round(z_down_tolerance, 3),
            "held_extent_source": (
                "reference" if held_reference is not None else "live_aabb"
            ),
            **sampler_inputs,
        }

    def _place_support_contact_check(
        self, target_obj, held_obj, pose, world_aligned=False
    ):
        """Fix P4: verify an accepted OnTop candidate rests on the live surface.

        Attempt 11 showed the class the reference gate (Fix P/P2/P3) cannot
        catch: a draw whose implied ray hit sits below a solid support top —
        the sampling ray passed through the tabletop near its center and the
        candidate bottom ended up ~4 cm embedded in solid geometry, which
        CuRobo could neither plan to nor reach. The P3 downward tolerance
        must stay generous enough for tiered supports (a sofa seat sits far
        below its AABB top), so instead of comparing against the reference
        AABB this probe compares the candidate against the live surface
        itself: a downward ray fan around the candidate (center + ring at
        PLACE_SUPPORT_PROBE_RADIUS_FRACTION of the held footprint radius).
        Because both sides of every comparison are live ray hits, mesh-vs-
        AABB offsets cancel and tolerances stay tight. Verdicts:

        * no_support_within_probe — the center ray found nothing under the
          candidate within the probe window (release would drop the object);
        * support_not_target — the highest surface under the candidate
          belongs to another body (candidate rests in/on the wrong object);
        * embedded_below_local_surface — the candidate bottom is more than
          PLACE_SUPPORT_EMBED_TOLERANCE below the local surface directly
          under it (the center ray hit; review R1 — NOT the max over the
          whole fan, which false-rejects clean draws on uneven surfaces);
          attempt 11: bottom 0.466 vs tabletop 0.506;
        * floating_above_support — the bottom is more than Z_OFFSET +
          PLACE_SUPPORT_FLOAT_TOLERANCE above the center hit (a release
          would drop the object, the attempt-10 signature).

        Returns (None, detail) when the candidate is consistent, else
        (reason, detail). The detail dict is kept in both cases so the
        accepted path is instrumented too — attempt 11's accepted candidate
        left no forensic data, which this closes. Held extents prefer the
        live world-aligned AABB because that is exactly what the stock
        sampler used to build the candidate z (review R2: mixing in the
        pre-grasp reference half_z inflates float_gap for tilted held
        objects); the reference snapshot is the fallback when the live
        query fails. Corrupted live extents stay caught upstream by the
        Fix P3 z gate, which checks the candidate z itself.
        """
        position = np.asarray([float(v) for v in pose[0]], dtype=float)
        references = getattr(self, "_deltasg_reference_bboxes", None) or {}
        held_reference = references.get(getattr(held_obj, "name", None))
        held_extent = None
        held_extent_source = None
        bottom_z = None
        # Match the stock sampler exactly. Its returned pose is the object
        # base-link pose, not the sampled bbox-center pose, so the bbox offset
        # and orientation must be transformed before computing the bottom.
        try:
            if world_aligned:
                bbox_extent = held_obj.aabb_extent
                bbox_center = held_obj.aabb_center
                bbox_pos_in_base, bbox_orn_in_base = T.relative_pose_transform(
                    bbox_center,
                    th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
                    *held_obj.get_position_orientation(),
                )
                held_extent_source = "world_aligned_bbox"
            else:
                _, _, bbox_extent, bbox_pos_in_base = (
                    held_obj.get_base_aligned_bbox()
                )
                bbox_orn_in_base = th.tensor(
                    [0.0, 0.0, 0.0, 1.0], dtype=th.float32
                )
                held_extent_source = "base_aligned_bbox"
            bbox_extent = np.asarray(_to_numpy(bbox_extent), dtype=float)
            bbox_center, bbox_orientation = T.pose_transform(
                th.as_tensor(pose[0], dtype=th.float32),
                th.as_tensor(pose[1], dtype=th.float32),
                th.as_tensor(bbox_pos_in_base, dtype=th.float32),
                th.as_tensor(bbox_orn_in_base, dtype=th.float32),
            )
            rotation = np.asarray(_to_numpy(T.quat2mat(bbox_orientation)), dtype=float)
            held_extent = bbox_extent
            bottom_z = float(_to_numpy(bbox_center)[2]) - 0.5 * float(
                np.abs(rotation[2]) @ bbox_extent
            )
        except Exception:
            held_extent = None
        if bottom_z is None and held_reference is not None:
            try:
                held_extent = np.asarray(
                    held_reference["bbox_extent"], dtype=float
                )
                held_extent_source = "reference"
                bottom_z = float(position[2]) - float(held_extent[2]) * 0.5
            except Exception:
                held_extent = None
                bottom_z = None
        if bottom_z is None:
            return None, {"reason_note": "held_extent_unavailable"}
        footprint_radius = float(np.max(held_extent[:2])) * 0.5
        candidate_height = max(0.0, float(position[2]) - bottom_z)
        z_start = float(position[2]) + candidate_height + PLACE_SUPPORT_PROBE_MARGIN
        z_end = bottom_z - PLACE_SUPPORT_PROBE_MARGIN
        ring_radius = footprint_radius * PLACE_SUPPORT_PROBE_RADIUS_FRACTION
        offsets = [(0.0, 0.0)]
        offsets += [
            (ring_radius * dx, ring_radius * dy)
            for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        ]
        ignore_bodies = [link.prim_path for link in held_obj.links.values()]
        try:
            ignore_bodies += [
                link.prim_path for link in self.robot.links.values()
            ]
        except Exception:
            pass
        try:
            from omnigibson.utils import sampling_utils

            rays = []
            for dx, dy in offsets:
                result = sampling_utils.raytest(
                    start_point=th.tensor(
                        [position[0] + dx, position[1] + dy, z_start]
                    ),
                    end_point=th.tensor(
                        [position[0] + dx, position[1] + dy, z_end]
                    ),
                    only_closest=True,
                    ignore_bodies=ignore_bodies,
                )
                rays.append(result)
        except Exception as exc:  # never block sampling on a failed probe
            return None, {"reason_note": f"probe_failed: {exc}"}
        target_paths = {link.prim_path for link in target_obj.links.values()}
        ray_hits = []
        for (dx, dy), result in zip(offsets, rays):
            if result.get("hit"):
                ray_hits.append({
                    "offset_xy": [round(dx, 4), round(dy, 4)],
                    "hit_z": round(float(result["position"][2]), 4),
                    "rigid_body": result.get("rigidBody"),
                    "is_target": result.get("rigidBody") in target_paths,
                })
            else:
                ray_hits.append({
                    "offset_xy": [round(dx, 4), round(dy, 4)],
                    "hit_z": None,
                    "rigid_body": None,
                    "is_target": False,
                })
        detail = {
            "candidate_bottom_z": round(bottom_z, 4),
            "probe_z_start": round(z_start, 4),
            "probe_z_end": round(z_end, 4),
            "ring_radius_m": round(ring_radius, 4),
            "held_extent_source": held_extent_source,
            "rays": ray_hits,
        }
        center = ray_hits[0]
        if center["hit_z"] is None:
            return "no_support_within_probe", detail
        if not center["is_target"]:
            return "support_not_target", detail
        hit_zs = [r["hit_z"] for r in ray_hits if r["hit_z"] is not None]
        h_max = max(hit_zs)
        # Review R1: embed/float verdicts compare against the LOCAL surface
        # under the candidate (center hit), not the max over the fan — a ring
        # ray on a higher part of an uneven tabletop or on a neighbor object
        # would inflate h_max and false-reject clean draws. The P5 reference
        # comparison likewise counts TARGET hits only as the live surface.
        target_hit_zs = [
            r["hit_z"]
            for r in ray_hits
            if r["hit_z"] is not None and r["is_target"]
        ]
        live_top_z = max(target_hit_zs) if target_hit_zs else center["hit_z"]
        z_offset = getattr(
            starter_primitives.m, "PREDICATE_SAMPLING_Z_OFFSET", 0.02
        )
        embed_deficit = center["hit_z"] - bottom_z
        float_gap = bottom_z - center["hit_z"]
        detail["support_h_max_z"] = round(h_max, 4)
        detail["support_live_top_z"] = round(live_top_z, 4)
        detail["embed_deficit_m"] = round(embed_deficit, 4)
        detail["float_gap_m"] = round(float_gap, 4)
        if embed_deficit > PLACE_SUPPORT_EMBED_TOLERANCE:
            return "embedded_below_local_surface", detail
        if float_gap > z_offset + PLACE_SUPPORT_FLOAT_TOLERANCE:
            return "floating_above_support", detail
        # Fix P5: cross-check the live surface against the pre-grasp
        # reference raycast map. The live probe and the sampler share one
        # scene-query layer, so a uniform loss of the support's top collision
        # mesh reads as a consistent (lower) surface to both; only the
        # pre-grasp capture still knows the true top. Static supports do not
        # move between captures, so a shortfall beyond the noise tolerance is
        # the through-table signature (attempt 11).
        reference = references.get(getattr(target_obj, "name", None)) or {}
        ref_hits = reference.get("support_surface_hits")
        if ref_hits:
            # Review R3: hits above the captured AABB top belong to objects
            # resting ON the support at capture time (cutting board, tray).
            # If a plan step removes such an object before this place, its
            # top must not serve as the reference surface or every legit
            # draw trips support_surface_lost_vs_reference.
            ref_top = reference.get("aabb_top")
            if ref_top is not None:
                ref_hits = [
                    h
                    for h in ref_hits
                    if h[2] <= ref_top + PLACE_SUPPORT_REFERENCE_TOP_EPSILON
                ]
            near = [
                h for h in ref_hits
                if math.hypot(h[0] - position[0], h[1] - position[1])
                <= PLACE_SUPPORT_REFERENCE_XY_RADIUS
            ]
            if near:
                ref_min = min(h[2] for h in near)
                detail["reference_surface_min_z"] = round(ref_min, 4)
                detail["reference_surface_deficit_m"] = round(
                    ref_min - live_top_z, 4
                )
                if (
                    ref_min - live_top_z
                    > PLACE_SUPPORT_REFERENCE_SURFACE_TOLERANCE
                ):
                    return "support_surface_lost_vs_reference", detail
        return None, detail

    def _sample_pose_with_object_and_predicate(
        self,
        predicate,
        held_obj,
        target_obj,
        world_aligned=False,
        near_poses=None,
        near_poses_threshold=None,
    ):
        """Fix O access-aware OnTop place candidate gate.

        The official sampler accepts ANY predicate-satisfying point on the
        target surface. attempt-7 / diag27 showed the failure class this
        creates: a far-edge table candidate (~0.82 m from the base, beside an
        armchair) passed the official reachability gate but forced an arm
        sweep that contacted the support region and stalled. attempt 8 showed
        the complementary trap: when the base parks far from the target, a
        blanket horizontal margin rejects every candidate and a naive
        fallback voids the gate entirely. Candidates are therefore gated on

        * a clear vertical descent corridor (held-object swept volume vs every
          margin-inflated non-target scene AABB), always; and
        * a horizontal reach margin from the current base pose ONLY when the
          candidate's hand pose is IK-reachable from that base — i.e. exactly
          when the stock flow would skip re-navigation and execute a long
          horizontal sweep at carry height. Far candidates that are NOT
          IK-reachable are accepted: the stock contract then re-navigates via
          _sample_pose_near_object/_navigate_to_pose and approaches with a
          short descent.

        If no candidate survives within PLACE_ACCESS_MAX_CANDIDATES samples,
        the closest corridor-clear candidate is returned (minimal sweep); the
        unconditional stock behavior (last sampled pose) is retained only when
        even the corridor blocks everything, so completeness is unchanged.
        The official collision/reachability standards still apply to whichever
        candidate survives. On top of these access gates, Fix P/P2 validates
        every draw against the pre-grasp reference surface
        (_place_sampling_surface_corruption) and voids the whole batch —
        including gate-passing survivors — when any draw trips it, forcing the
        apply_ref stop/play rebuild and a resample. Fix P4 additionally
        validates every surviving draw against the live support surface
        (_place_support_contact_check): a downward ray fan around the
        candidate must confirm the target is under it, the candidate bottom
        is not embedded in surrounding geometry (the attempt-11 through-table
        class, z_dev -0.04 inside the reference gate's tiered-support
        tolerance) and not floating above the support (attempt-10 class).
        """
        if predicate is not object_states.OnTop or held_obj is None or target_obj is None:
            return super()._sample_pose_with_object_and_predicate(
                predicate,
                held_obj,
                target_obj,
                world_aligned=world_aligned,
                near_poses=near_poses,
                near_poses_threshold=near_poses_threshold,
            )
        state = getattr(self, "_deltasg_place_access", None)
        if state is None:
            state = {
                "examined": 0,
                "rejections": [],
                "fallback_pose": None,
                "fallback_best": None,
                "fallback_best_distance": None,
                "corrupted": 0,
                "support_rejected": 0,
                "hover_rejected": 0,
                "accepted_support": None,
                "fallback_best_support": None,
            }
            self._deltasg_place_access = state
        held_lower, held_upper = held_obj.aabb
        held_half_extents = (held_upper - held_lower) * 0.5
        obstacle_aabbs = self._place_corridor_obstacle_aabbs(held_obj, target_obj)
        robot_xy = self.robot.get_position_orientation()[0][:2]
        initial_joint_pos = self._get_joint_position_with_fingers_at_limit("upper")
        for _ in range(PLACE_ACCESS_MAX_CANDIDATES):
            pose = super()._sample_pose_with_object_and_predicate(
                predicate,
                held_obj,
                target_obj,
                world_aligned=world_aligned,
                near_poses=near_poses,
                near_poses_threshold=near_poses_threshold,
            )
            if world_aligned:
                position = th.as_tensor(pose[0], dtype=th.float32).clone()
                stock_offset = float(getattr(
                    starter_primitives.m, "PREDICATE_SAMPLING_Z_OFFSET", 0.02
                ))
                contact_lowering = max(
                    0.0, stock_offset - PLACE_RELEASE_CONTACT_CLEARANCE
                )
                position[2] -= contact_lowering
                pose = position, pose[1]
            state["examined"] += 1
            state["fallback_pose"] = pose
            corruption = self._place_sampling_surface_corruption(
                target_obj, pose, held_obj
            )
            if corruption is not None:
                # Fix P: this draw lies outside the pre-grasp reference
                # surface; the live sampler inputs are corrupted. Never let it
                # seed a fallback or reach the corridor/reach gates.
                state["corrupted"] += 1
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "sampling_surface_corrupted",
                    **corruption,
                })
                continue
            blockers = place_descent_corridor_blockers(
                pose[0], held_half_extents, obstacle_aabbs
            )
            if blockers:
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "descent_corridor_blocked",
                    "blockers": blockers,
                })
                continue
            # Fix P4: confirm the candidate actually rests on the live
            # surface (center ray hits the target; bottom is not embedded in
            # surrounding geometry and not floating above the support). This
            # catches the through-table draw class from attempt 11 that the
            # reference-surface gate cannot, because the comparison is made
            # against live ray hits on both sides.
            support_reason, support_detail = self._place_support_contact_check(
                target_obj, held_obj, pose, world_aligned=world_aligned
            )
            if support_reason is not None:
                state["support_rejected"] += 1
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "support_contact_inconsistent",
                    "support_reason": support_reason,
                    **support_detail,
                })
                continue
            state["accepted_support"] = support_detail
            distance = float(th.linalg.norm(
                th.as_tensor(pose[0][:2], dtype=th.float32)
                - th.as_tensor(robot_xy, dtype=th.float32)
            ))
            hand_pose = self._get_hand_pose_for_object_pose(pose)
            in_reach = bool(self._target_in_reach_of_robot(
                hand_pose,
                initial_joint_pos=initial_joint_pos,
                skip_obstacle_update=True,
            ))
            hover_pose = self._place_hover_pose(hand_pose)
            if in_reach and hover_pose is not None and not bool(
                self._target_in_reach_of_robot(
                    hover_pose,
                    initial_joint_pos=initial_joint_pos,
                    skip_obstacle_update=True,
                )
            ):
                state["hover_rejected"] += 1
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "place_hover_not_reachable_from_current_base",
                })
                continue
            if (
                state["fallback_best_distance"] is None
                or distance < state["fallback_best_distance"]
            ):
                state["fallback_best_distance"] = distance
                state["fallback_best"] = pose
                state["fallback_best_support"] = support_detail
            if distance <= PLACE_ACCESS_REACH_MARGIN:
                if state["corrupted"] > 0:
                    # Fix P2: this draw passed the geometric gates, but an
                    # earlier draw in the same batch tripped the reference
                    # surface gate, so the live sampling surface is suspect and
                    # no survivor of this batch can be trusted. Void it and
                    # exhaust the batch so apply_ref rebuilds (stop/play, the
                    # verified repair) and resamples from a clean surface.
                    state["rejections"].append({
                        "candidate_pose_position": _jsonable(pose[0]),
                        "reason": "clean_candidate_voided_corrupted_batch",
                        "surface_rejections_so_far": state["corrupted"],
                    })
                    continue
                self._record_place_access_diagnostics(
                    target_obj, accepted_pose=pose, fallback_used=False
                )
                return pose
            # Far candidates are only dangerous when the stock flow would
            # place from the current base (IK-reachable => no re-navigation,
            # long horizontal sweep). If the pose is NOT IK-reachable from
            # here the stock contract re-navigates via
            # _sample_pose_near_object/_navigate_to_pose, so the sweep is
            # short and the candidate is accepted.
            if in_reach:
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "reach_margin",
                    "horizontal_distance_m": round(distance, 3),
                    "reach_margin_m": PLACE_ACCESS_REACH_MARGIN,
                    "in_reach": True,
                })
                continue
            if state["corrupted"] > 0:
                # Fix P2: same batch-void rule as the near accept path.
                state["rejections"].append({
                    "candidate_pose_position": _jsonable(pose[0]),
                    "reason": "clean_candidate_voided_corrupted_batch",
                    "surface_rejections_so_far": state["corrupted"],
                })
                continue
            self._record_place_access_diagnostics(
                target_obj, accepted_pose=pose, fallback_used=False
            )
            return pose
        if state["corrupted"] > 0:
            # Fix P/P2: at least one draw tripped the reference-surface gate
            # and none was accepted (P2 voids clean survivors of a corrupted
            # batch). Every draw of this batch came from the same corrupted
            # sampling surface, so no fallback candidate can be trusted either.
            # Fail the primitive: the apply_ref retry rebuilds the physics
            # scene (stop/play, the verified repair) and resamples.
            self._deltasg_place_sampling_corrupted = True
            self._record_place_access_diagnostics(
                target_obj, accepted_pose=None, fallback_used=False
            )
            print(
                f"[expert] Fix P: place sampling surface corrupted "
                f"(target={getattr(target_obj, 'name', target_obj)} "
                f"examined={state['examined']} surface_rejections={state['corrupted']}); "
                f"forcing physics rebuild and resample",
                flush=True,
            )
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Fix P: OnTop place sampling surface corrupted; "
                "forcing physics rebuild and resample",
                {
                    "target object": getattr(target_obj, "name", str(target_obj)),
                    "candidates_examined": state["examined"],
                    "surface_rejections": state["corrupted"],
                    "support_rejections": state["support_rejected"],
                },
            )
        if state["fallback_best"] is not None:
            state["accepted_support"] = state.get("fallback_best_support")
            self._record_place_access_diagnostics(
                target_obj, accepted_pose=state["fallback_best"], fallback_used=True
            )
            return state["fallback_best"]
        if state["support_rejected"] > 0:
            # Fix P4: every draw failed support-contact consistency and no
            # gate-clean fallback exists. A whole batch of through-table /
            # floating draws is the scene-query failure signature (the
            # sampler's rays no longer see the support surface correctly),
            # so force the verified repair: apply_ref rebuilds the physics
            # scene (stop/play) and resamples from a clean surface. If the
            # support really is unusable the second attempt fails the step
            # honestly instead of embedding the object in it.
            self._deltasg_place_sampling_corrupted = True
            self._record_place_access_diagnostics(
                target_obj, accepted_pose=None, fallback_used=False
            )
            print(
                f"[expert] Fix P4: OnTop place candidates failed support-contact "
                f"consistency (target={getattr(target_obj, 'name', target_obj)} "
                f"examined={state['examined']} "
                f"support_rejections={state['support_rejected']}); "
                f"forcing physics rebuild and resample",
                flush=True,
            )
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Fix P4: OnTop place candidates failed support-contact "
                "consistency; forcing physics rebuild and resample",
                {
                    "target object": getattr(target_obj, "name", str(target_obj)),
                    "candidates_examined": state["examined"],
                    "support_rejections": state["support_rejected"],
                    "support_rejection_reasons": [
                        rejection.get("support_reason")
                        for rejection in state["rejections"]
                        if rejection.get("reason") == "support_contact_inconsistent"
                    ],
                },
            )
        if state["hover_rejected"] > 0:
            self._record_place_access_diagnostics(
                target_obj, accepted_pose=None, fallback_used=False
            )
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.SAMPLING_ERROR,
                "Could not sample an OnTop pose with a reachable hover approach",
                {
                    "target object": getattr(target_obj, "name", str(target_obj)),
                    "candidates_examined": state["examined"],
                    "hover_rejections": state["hover_rejected"],
                },
            )
        self._record_place_access_diagnostics(
            target_obj, accepted_pose=None, fallback_used=True
        )
        return state["fallback_pose"]

    def _move_fingers_to_limit(self, limit_type):
        target_joint_positions = self._get_joint_position_with_fingers_at_limit(limit_type)
        gripper_indices = self.robot.gripper_control_idx[self.arm]
        action = self.robot.q_to_action(target_joint_positions)
        for _ in range(starter_primitives.m.MAX_STEPS_FOR_GRASP_OR_RELEASE):
            current_gripper = self.robot.get_joint_positions()[gripper_indices]
            target_gripper = target_joint_positions[gripper_indices]
            if th.allclose(
                current_gripper,
                target_gripper,
                atol=starter_primitives.m.JOINT_POS_DIFF_THRESHOLD,
            ):
                break
            yield self._postprocess_action(action)
            if limit_type == "lower" and self._get_obj_in_hand() is not None:
                break

    def _execute_release(self):
        """Release physically, then let an OnTop placement settle on-policy."""
        released_obj = self._get_obj_in_hand()
        place_target = getattr(self, "_deltasg_place_target", None)
        diagnostics = getattr(self, "physical_approach_diagnostics", {})
        self.physical_approach_diagnostics = diagnostics
        release_diagnostic = {}

        if released_obj is not None and place_target is not None:
            stable_steps = 0
            previous_center = None
            for pre_release_steps in range(1, 61):
                hold_action = self.robot.q_to_action(self.robot.get_joint_positions())
                yield self._postprocess_action(hold_action)
                center = _to_numpy(released_obj.aabb_center)
                if previous_center is not None and float(
                    np.linalg.norm(center - previous_center)
                ) <= 0.001:
                    stable_steps += 1
                else:
                    stable_steps = 0
                previous_center = center
                if stable_steps >= 5:
                    live_low, live_high = released_obj.aabb
                    target_low, target_high = place_target.aabb
                    live_center = (live_low + live_high) * 0.5
                    live_xy_on_target = bool(th.all(
                        (live_center[:2] >= target_low[:2])
                        & (live_center[:2] <= target_high[:2])
                    ))
                    live_gap = float(live_low[2] - target_high[2])
                    if live_xy_on_target and -0.005 <= live_gap <= 0.03:
                        break

            released_low, released_high = released_obj.aabb
            target_low, target_high = place_target.aabb
            released_center = (released_low + released_high) * 0.5
            xy_on_target = bool(th.all(
                (released_center[:2] >= target_low[:2])
                & (released_center[:2] <= target_high[:2])
            ))
            vertical_gap = float(released_low[2] - target_high[2])
            ready_to_release = xy_on_target and -0.005 <= vertical_gap <= 0.03
            release_diagnostic["pre_release"] = {
                "steps": pre_release_steps,
                "stable_steps": stable_steps,
                "xy_on_target": xy_on_target,
                "vertical_gap_m": round(vertical_gap, 4),
                "ready": ready_to_release,
                "released_object_aabb": [
                    _jsonable(released_low),
                    _jsonable(released_high),
                ],
                "target_aabb": [_jsonable(target_low), _jsonable(target_high)],
            }
            diagnostics.setdefault(place_target.name, {})[
                "place_release_settle"
            ] = release_diagnostic
            if not ready_to_release:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Held object did not physically converge above the placement target",
                    release_diagnostic["pre_release"],
                )

        yield from super()._execute_release()
        if released_obj is None or place_target is None:
            return

        stable_steps = 0
        steps = 0
        for steps in range(1, 61):
            hold_action = self.robot.q_to_action(self.robot.get_joint_positions())
            yield self._postprocess_action(hold_action)
            if bool(released_obj.states[object_states.OnTop].get_value(place_target)):
                stable_steps += 1
                if stable_steps >= 5:
                    break
            else:
                stable_steps = 0

        release_diagnostic.update({
            "steps": steps,
            "stable_on_top_steps": stable_steps,
            "on_top": bool(
                released_obj.states[object_states.OnTop].get_value(place_target)
            ),
            "touching": bool(
                released_obj.states[object_states.Touching].get_value(place_target)
            ),
        })
        try:
            vertical = released_obj.states[object_states.VerticalAdjacency].get_value()
            release_diagnostic["target_below"] = place_target in vertical.negative_neighbors
            release_diagnostic["target_above"] = place_target in vertical.positive_neighbors
        except Exception as exc:
            release_diagnostic["adjacency_error"] = repr(exc)
        try:
            released_low, released_high = released_obj.aabb
            target_low, target_high = place_target.aabb
            release_diagnostic.update({
                "released_object_pose": _jsonable(
                    released_obj.get_position_orientation()[0]
                ),
                "released_object_aabb": [
                    _jsonable(released_low),
                    _jsonable(released_high),
                ],
                "target_aabb": [_jsonable(target_low), _jsonable(target_high)],
            })
        except Exception as exc:
            release_diagnostic["geometry_error"] = repr(exc)
        diagnostics.setdefault(place_target.name, {})[
            "place_release_settle"
        ] = release_diagnostic

    def _open_or_close(self, obj, should_open):
        """Execute the disabled Starter open/close trajectory with state checks."""
        state = obj.states.get(object_states.Open)
        if state is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Target object has no Open state",
                {"target object": obj.name},
            )
        if bool(state.get_value()) == should_open:
            return
        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot open or close an object while holding an object",
                {"object in hand": self._get_obj_in_hand().name},
            )

        self._tracking_object = self.robot
        yield from self._execute_release()
        for _ in range(starter_primitives.m.MAX_ATTEMPTS_FOR_OPEN_CLOSE):
            grasp_data = starter_primitives.get_grasp_position_for_open(
                self.robot,
                obj,
                should_open,
                None,
                **({} if should_open else {"num_waypoints": 3}),
            )
            if grasp_data is None:
                continue
            _, grasp_pose, target_poses, object_direction, _, position_change = grasp_data
            if abs(position_change) < 0.1:
                return

            approach_pose = (
                grasp_pose[0]
                + object_direction * starter_primitives.m.OPEN_GRASP_APPROACH_DISTANCE,
                grasp_pose[1],
            )
            try:
                yield from self._navigate_if_needed(obj, eef_pose=grasp_pose)
                yield from self._move_hand(grasp_pose, stop_if_stuck=True)
                if should_open:
                    yield from self._execute_grasp()
                yield from self._navigate_if_needed(obj, eef_pose=approach_pose)
                yield from self._move_hand_linearly_cartesian(
                    approach_pose,
                    ignore_failure=False,
                    stop_on_contact=should_open,
                    stop_if_stuck=True,
                )
                yield self._postprocess_action(self._empty_action())
                for target_pose in target_poses:
                    yield from self._move_hand_linearly_cartesian(
                        target_pose, ignore_failure=False, stop_if_stuck=True
                    )
                yield from self._move_hand_linearly_cartesian(
                    self.robot.eef_links[self.arm].get_position_orientation(),
                    ignore_failure=True,
                    stop_if_stuck=True,
                )
                if should_open:
                    yield from self._execute_release()
                    yield from self._move_base_backward()
            except ActionPrimitiveError:
                if should_open:
                    yield from self._execute_release()
                    yield from self._move_base_backward()
                else:
                    yield from self._move_hand_backward()
            if bool(state.get_value()) == should_open:
                return

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
            "Physical trajectory did not set the requested Open state",
            {"target object": obj.name, "actual": bool(state.get_value())},
        )

    def _toggle(self, obj, value):
        """Move the physical gripper into the object's official toggle link."""
        state = obj.states.get(object_states.ToggledOn)
        if state is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Target object has no ToggledOn state",
                {"target object": obj.name},
            )
        if bool(state.get_value()) == value:
            return

        self._tracking_object = obj
        toggle_position = state.link.get_position_orientation()[0]
        robot_position, _ = self.robot.get_position_orientation()
        approach_direction = robot_position - toggle_position
        approach_direction[2] = 0.0
        approach_direction = approach_direction / th.clamp(
            th.linalg.norm(approach_direction), min=1e-6
        )
        approach_position = toggle_position + approach_direction * 0.10
        hand_orientation = self.robot.eef_links[self.arm].get_position_orientation()[1]
        approach_hand_pose = (approach_position, hand_orientation)
        desired_hand_pose = (toggle_position, hand_orientation)
        yield from self._navigate_if_needed(obj, eef_pose=approach_hand_pose)
        yield from self._move_hand(approach_hand_pose)
        yield from self._move_hand_linearly_cartesian(
            desired_hand_pose,
            ignore_failure=True,
            stop_on_contact=True,
            stop_if_stuck=True,
        )
        for _ in range(toggle_state.m.CAN_TOGGLE_STEPS + 2):
            yield self._postprocess_action(self._empty_action())
            if bool(state.get_value()) == value:
                return

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
            "Physical contact did not set the requested ToggledOn state",
            {"target object": obj.name, "actual": bool(state.get_value())},
        )

    def _get_head_goal_q(self, target_obj_pose):
        """Track low targets by clamping, rather than discarding, head goals."""
        target_position = target_obj_pose[0]
        robot_position, robot_orientation = self.robot.get_position_orientation()
        head_position, _ = self.robot.links["head_2_link"].get_position_orientation()
        delta = target_position - robot_position
        target_yaw = math.atan2(float(delta[1]), float(delta[0]))
        robot_yaw = self._quat_to_yaw(robot_orientation)
        pan = math.atan2(math.sin(target_yaw - robot_yaw), math.cos(target_yaw - robot_yaw))
        horizontal = max(
            float(th.linalg.norm(target_position[:2] - head_position[:2])), 1e-6
        )
        tilt = math.atan2(float(target_position[2] - head_position[2]), horizontal)
        head1 = self.robot.joints["head_1_joint"]
        head2 = self.robot.joints["head_2_joint"]
        margin = 1e-3
        pan = min(max(pan, float(head1.lower_limit) + margin), float(head1.upper_limit) - margin)
        tilt = min(max(tilt, float(head2.lower_limit) + margin), float(head2.upper_limit) - margin)
        return th.tensor([pan, tilt], dtype=th.float32)

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        # The generic Starter NAVIGATE_TO samples a grasp pose even when the
        # requested operation is only navigation. Small tabletop targets make
        # that sampler unnecessarily brittle. Planning and emitted controls
        # remain the official CuRobo Starter implementation.
        current_position, current_orientation = self.robot.get_position_orientation()
        target_position, _ = obj.get_position_orientation()
        target_aabb_min, target_aabb_max = obj.aabb
        nearest_target_to_robot_xy = th.minimum(
            th.maximum(current_position[:2], target_aabb_min[:2]), target_aabb_max[:2]
        )
        current_target_distance = float(
            th.linalg.norm(nearest_target_to_robot_xy - current_position[:2])
        )
        if eef_pose is None and current_target_distance <= 0.75:
            target_yaw = math.atan2(
                float(target_position[1] - current_position[1]),
                float(target_position[0] - current_position[0]),
            )
            current_yaw = self._quat_to_yaw(current_orientation)
            yaw_error = abs(
                math.atan2(
                    math.sin(target_yaw - current_yaw),
                    math.cos(target_yaw - current_yaw),
                )
            )
            current_xy = current_position[:2]
            pose_source = None
            if yaw_error <= 0.2:
                pose_yaw = current_yaw
                pose_source = "current_satisfied_pose"
            elif _native_occupant_at_pose(
                self.env, self.robot, obj, current_xy, target_yaw
            ) is None:
                pose_yaw = target_yaw
                pose_source = "current_reachable_pose"
            if pose_source is not None:
                pose = th.tensor(
                    [float(current_xy[0]), float(current_xy[1]), pose_yaw],
                    dtype=th.float32,
                )
                self.physical_approach_diagnostics = getattr(
                    self, "physical_approach_diagnostics", {}
                )
                self.physical_approach_diagnostics[obj.name] = {
                    "candidate_pose_xyyaw": _jsonable(pose),
                    "horizontal_target_distance": current_target_distance,
                    "horizontal_target_center_distance": float(
                        th.linalg.norm(target_position[:2] - current_position[:2])
                    ),
                    "current_horizontal_target_distance": current_target_distance,
                    "distance_reference": "aabb_edge",
                    "pose_source": pose_source,
                    "yaw_error_before": yaw_error,
                    "navigation_waypoints": {
                        "count": 1,
                        "poses_xyyaw": [_jsonable(pose)],
                    },
                }
                if pose_source == "current_satisfied_pose":
                    yield self._postprocess_action(self._empty_action())
                else:
                    yield from self._navigate_to_pose(pose, skip_obstacle_update=False)
                return
        if eef_pose is None:
            pose = _connected_observation_pose(
                self.env,
                self.robot,
                obj,
                preferred_distance=0.65,
                require_route=True,
                route_target=obj,
            )
            pose_source = "connected_observation_pose"
        else:
            # The explicit NAVIGATE step only needs a visible connected stance.
            # Fine manipulation is stricter: choose a base pose that the
            # official primitive has validated against the sampled EEF target.
            # A base-distance shortcut previously skipped this test and made
            # every retry repeat the same arm-infeasible grasp stance.
            pose = self._sample_pose_near_object(
                obj,
                eef_pose=eef_pose,
                skip_obstacle_update=skip_obstacle_update,
            )
            if pose is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "Could not find a connected arm-reachable pose near the object",
                    {"object": obj.name},
                )
            pose_source = "official_eef_reachable_pose"
        nearest_target_xy = th.minimum(
            th.maximum(pose[:2], target_aabb_min[:2]), target_aabb_max[:2]
        )
        target_distance = float(th.linalg.norm(nearest_target_xy - pose[:2]))
        target_center_distance = float(th.linalg.norm(target_position[:2] - pose[:2]))
        if not hasattr(self, "physical_approach_diagnostics"):
            self.physical_approach_diagnostics = {}
        approach_diagnostic = self.physical_approach_diagnostics.setdefault(obj.name, {})
        approach_diagnostic.update({
            "candidate_pose_xyyaw": _jsonable(pose),
            "horizontal_target_distance": target_distance,
            "horizontal_target_center_distance": target_center_distance,
            "current_horizontal_target_distance": current_target_distance,
            "distance_reference": "aabb_edge",
            "pose_source": pose_source,
        })
        if eef_pose is None and target_distance > DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE:
            # Honesty gate: a NAVIGATE_TO whose selected stand-off stays more
            # than 1.15 m away from the target AABB edge is a no-op shuffle
            # (Beechwood_0 deliver_drink attempt 5, 2026-08-15: observation
            # pose selection degenerated to a robot-adjacent candidate 0.14 m
            # away / 1.56 m from the table, and the displacement-only
            # postcondition silently accepted it). Refuse it as a planning
            # failure instead of pretending to approach. Generalizes the
            # previous online_env_-only reachability gate to every target.
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Navigation stand-off does not approach the target; refusing a no-op NAVIGATE_TO",
                {
                    "object": obj.name,
                    "pose_source": pose_source,
                    "horizontal_target_distance": target_distance,
                    "current_horizontal_target_distance": current_target_distance,
                },
            )
        waypoints = _connected_navigation_waypoints(
            self.env, self.robot, pose, target=obj
        )
        self.physical_approach_diagnostics[obj.name]["navigation_waypoints"] = {
            "count": len(waypoints),
            "poses_xyyaw": [_jsonable(waypoint) for waypoint in waypoints],
        }
        for waypoint in waypoints:
            yield from self._navigate_to_pose(waypoint, skip_obstacle_update=False)

    def _navigate_to_pose(self, pose_2d, skip_obstacle_update=False):
        # The Starter place flow calls this directly with a validated stand-off
        # pose that can be meters away. Driving that as one straight segment
        # crosses furniture and stalls (Beechwood_0 deliver_drink attempt 4:
        # stopped 0.1135 m short of the validated place stance, position
        # tolerance 0.10 m). Route far goals through the same clearance-map
        # BFS that NAVIGATE_TO uses; the exact-footprint goal carve applies.
        # Short goals (per-waypoint drives from _navigate_to_obj, in-place
        # rotations) keep the direct drive: waypoint spacing is ~0.10-0.14 m.
        goals = [pose_2d]
        current_position, _ = self.robot.get_position_orientation()
        horizontal_distance = float(
            th.linalg.norm(th.as_tensor(pose_2d, dtype=th.float32)[:2] - current_position[:2])
        )
        if horizontal_distance > 0.5:
            try:
                goals = _connected_navigation_waypoints(
                    self.env,
                    self.robot,
                    pose_2d,
                    target=getattr(self, "_tracking_object", None),
                )
            except ActionPrimitiveError:
                goals = [pose_2d]
        # Primitive configs use a position-mode holonomic base because CuRobo
        # trajectory playback requires robot.q_to_action(). The clearance-map
        # planner already supplies a collision-safe global route, so drive each
        # local waypoint as bounded position increments through that controller.
        for goal in goals:
            pose_3d = self._get_robot_pose_from_2d_pose(goal)
            for _ in range(500):
                relative_position, relative_orientation = self._world_pose_to_robot_pose(pose_3d)
                position_error = float(th.linalg.norm(relative_position[:2]))
                relative_yaw = float(self._quat_to_yaw(relative_orientation))
                yaw_error = abs(relative_yaw)
                if position_error <= 0.1 and yaw_error <= 0.2:
                    break
                action = self._empty_action()
                base_action = action[self.robot.controller_action_idx["base"]]
                if position_error > 0.1:
                    step_distance = min(position_error, 0.02)
                    base_action[:2] = relative_position[:2] / position_error * step_distance
                base_action[2] = max(-0.04, min(0.04, relative_yaw))
                action[self.robot.controller_action_idx["base"]] = base_action
                yield self._postprocess_action(action)
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Could not reach the planned navigation pose",
                    {"position_error": position_error, "yaw_error": yaw_error},
                )
        yield self._postprocess_action(self._empty_action())

    def _rotate_base_to_yaw(self, target_yaw):
        """Rotate at the current XY through the official joint action path."""
        start_position, start_orientation = self.robot.get_position_orientation()
        current_yaw = self._quat_to_yaw(start_orientation)
        yaw_delta = math.atan2(
            math.sin(float(target_yaw) - current_yaw),
            math.cos(float(target_yaw) - current_yaw),
        )
        if abs(yaw_delta) < 0.05:
            return

        joint_goal = self.robot.get_joint_positions().clone()
        base_indices = self.robot.base_control_idx
        joint_goal[base_indices[2]] += yaw_delta
        yield from self._execute_motion_plan(joint_goal.unsqueeze(0), ignore_failure=True)

        end_position, end_orientation = self.robot.get_position_orientation()
        position_drift = float(th.linalg.norm(end_position[:2] - start_position[:2]))
        yaw_error = abs(
            math.atan2(
                math.sin(float(target_yaw) - self._quat_to_yaw(end_orientation)),
                math.cos(float(target_yaw) - self._quat_to_yaw(end_orientation)),
            )
        )
        if position_drift > 0.08 or yaw_error > 0.35:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not rotate in place toward the manipulation target",
                {"position_drift": position_drift, "yaw_error": yaw_error},
            )

    @staticmethod
    def _quat_to_yaw(quaternion):
        x, y, z, w = (float(value) for value in quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _scene_query_visible(self, obj):
        """Whether PhysX scene queries currently see ``obj``.

        diag10-12: after render-off physics stepping, psqi raycasts can pass
        through every dynamic rigid body (static geometry stays visible), so
        predicate sampling refuses every draw with missed_object even though
        the support surface is physically present. Probe with the same
        raytest path the sampler uses. A probe ray counts the object visible
        if it hits one of the object's links; it counts scene queries alive
        (but the probe column occluded, e.g. by the held object) if the
        closest hit is another body inside the object's vertical extent.
        Only rays that pass through the object down to/below its bottom face
        (or miss entirely) are evidence that scene queries lost the object.
        """
        try:
            from omnigibson.utils import sampling_utils

            low, high = obj.aabb
            link_paths = {link.prim_path for link in obj.links.values()}
            cx = float((low[0] + high[0]) / 2.0)
            cy = float((low[1] + high[1]) / 2.0)
            z_start = float(high[2]) + 0.30
            z_end = float(low[2]) - 0.20
            occluded = False
            for dx, dy in ((0.0, 0.0), (0.10, 0.0), (0.0, 0.10)):
                ray = sampling_utils.raytest(
                    start_point=th.tensor([cx + dx, cy + dy, z_start]),
                    end_point=th.tensor([cx + dx, cy + dy, z_end]),
                    only_closest=True,
                )
                if not ray.get("hit"):
                    continue
                if ray.get("rigidBody") in link_paths:
                    return True
                if float(ray["position"][2]) >= float(low[2]) + 0.02:
                    occluded = True
            return occluded
        except Exception:
            # Never block sampling on a failed probe.
            return True

    def _prime_scene_query_visibility(self, obj, max_steps=1200):
        """Step physics until psqi sees ``obj``, yielding the no-op actions.

        diag12 verdict: after replaying 818 render-off actions the table and
        the bottle were invisible to scene queries; 20 more render-off physics
        steps restored them. Rendering is neither required nor the trigger
        (diag11's render-on "repair" was the extra physics steps); continued
        physics stepping is. Actions are yielded instead of stepping directly
        so the priming steps stay in the recorded/replayed trace. Sampling
        itself performs no physics steps, so visibility at the probe moment
        is what the subsequent draws see.
        """
        for _ in range(max_steps):
            if self._scene_query_visible(obj):
                return
            empty_action = self.robot.q_to_action(self.robot.get_joint_positions())
            yield self._postprocess_action(empty_action)

    def _rebuild_physics_scene_queries(self, obj, settle_steps=20):
        """Rebuild the PhysX scene via stop/play to recover blind scene queries.

        diag22-24: the GRASP step's FixedJoint creation can corrupt the PhysX
        broadphase (burst of "Illegal BroadPhaseUpdateData"), after which psqi
        raycasts pass through every dynamic rigid body and the ray-cast OnTop
        sampler fails 100% no matter how long physics steps (1200+ tried).
        og.sim.stop() + play() rebuilds the PhysX scene and broadphase from
        scratch and restores visibility — the same mechanism VERIFIED for the
        generation path in online_deltasg.py (deliberate stop/play physics
        rebuild) and for this expert state by diag23/diag24.

        play() calls robot.reset() and its internal _non_physics_step runs
        assisted-grasp handling, which would release the held object; guard it
        with robot._disable_grasp_handling (robot.py post_step gate) so the AG
        joint prim (which survives in USD) and the grasp state are kept, then
        restore the exact joint state and every pose the rebuild reverts.
        Settle steps are yielded as no-op actions so they stay in the
        recorded/replayed trace.
        """
        if not og.sim.is_playing():
            return
        robot = self.robot
        saved_joints = robot.get_joint_positions().clone()
        saved_poses = []
        seen = set()
        for target in [robot, *_scene_objects(self.env).values()]:
            key = getattr(target, "prim_path", None) or getattr(target, "name", None) or id(target)
            if key in seen:
                continue
            seen.add(key)
            try:
                position, orientation = target.get_position_orientation()
                saved_poses.append((target, position.clone(), orientation.clone()))
            except Exception:
                continue
        prior_disable = getattr(robot, "_disable_grasp_handling", False)
        robot._disable_grasp_handling = True
        try:
            og.sim.stop()
            og.sim.play()
        finally:
            robot._disable_grasp_handling = prior_disable
        # play() ran robot.reset(); restore the exact pre-rebuild joint state.
        robot.set_joint_positions(saved_joints, drive=False)
        for target, position, orientation in saved_poses:
            if target is robot:
                continue
            try:
                current_position, _ = target.get_position_orientation()
                if float(((current_position - position) ** 2).sum() ** 0.5) <= 1e-4:
                    continue
                target.set_position_orientation(position=position, orientation=orientation)
                target.keep_still()
            except Exception:
                continue
        # Refresh the AG constraint prim references so the later release works.
        for arm in getattr(robot, "arm_names", []):
            constraint = (getattr(robot, "_ag_obj_constraints", None) or {}).get(arm)
            if constraint is None:
                continue
            try:
                path = constraint.GetPath().pathString
                refreshed = og.sim.stage.GetPrimAtPath(og.lazy.pxr.Sdf.Path(path))
                if refreshed is not None and refreshed.IsValid():
                    robot._ag_obj_constraints[arm] = refreshed
            except Exception:
                continue
        visible = self._scene_query_visible(obj)
        print(
            f"[expert] physics rebuild (stop/play) for scene queries on "
            f"{getattr(obj, 'name', obj)} visible_after={visible}",
            flush=True,
        )
        for _ in range(settle_steps):
            empty_action = robot.q_to_action(robot.get_joint_positions())
            yield self._postprocess_action(empty_action)

    def apply_ref(self, primitive, *args, attempts=5):
        """Preserve primary failures while reporting cleanup as diagnostics."""
        ctrl = self.controller_functions[primitive]
        errors = []
        self.last_cleanup_errors = []
        place_target = (
            args[0]
            if args
            and primitive
            in (
                StarterSemanticActionPrimitiveSet.PLACE_ON_TOP,
                StarterSemanticActionPrimitiveSet.PLACE_INSIDE,
            )
            else None
        )
        for _ in range(attempts):
            if place_target is not None:
                rebuilt = False
                if getattr(self, "_deltasg_place_sampling_corrupted", False):
                    # Fix P: a corrupted sampling surface is invisible to the
                    # visibility probe (it still hits the target's own links),
                    # so force the stop/play rebuild — the verified repair
                    # (diag23-24; diag30 sampled clean after it).
                    self._deltasg_place_sampling_corrupted = False
                    yield from self._rebuild_physics_scene_queries(place_target)
                    rebuilt = True
                yield from self._prime_scene_query_visibility(place_target)
                if not rebuilt and not self._scene_query_visible(place_target):
                    # Grasp-corrupted broadphase: stepping alone never heals it
                    # (diag20-22); rebuild the physics scene (diag23-24).
                    yield from self._rebuild_physics_scene_queries(place_target)
            primary_error = None
            try:
                yield from ctrl(*args)
            except ActionPrimitiveError as exc:
                primary_error = exc
                errors.append(exc)
            except Exception:
                raise

            cleanup_errors = []
            # Navigation leaves the arm and gripper unchanged. Avoiding an arm
            # reset here also prevents a failed reset from masking navigation.
            cleanup_generators = [self._settle_robot]
            if primitive != StarterSemanticActionPrimitiveSet.NAVIGATE_TO:
                if not self._get_obj_in_hand():
                    cleanup_generators.insert(0, self._execute_release)
                cleanup_generators.insert(-1, self._reset_robot)
            for cleanup in cleanup_generators:
                try:
                    yield from cleanup()
                except Exception as exc:
                    cleanup_errors.append({"operation": cleanup.__name__, "error": repr(exc)})
            self.last_cleanup_errors.extend(cleanup_errors)

            # Cleanup is best effort and is not the primitive postcondition.
            # In particular, R1 may retain a sub-centimeter joint residual
            # while securely holding a newly grasped object. The caller checks
            # object state and scene integrity after this generator returns.
            if primary_error is None:
                return

        raise ActionPrimitiveErrorGroup(errors)


def _jsonable(value):
    if isinstance(value, th.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _to_numpy(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _name(record):
    return record.get("object_name") or record.get("object_id") or record.get("name")


def _iter_modalities(obs, info, path=()):
    if not isinstance(obs, dict):
        return
    if any(key in obs for key in ("rgb", "seg_instance", "seg_semantic")):
        yield path, obs, info if isinstance(info, dict) else {}
    for key, value in obs.items():
        if isinstance(value, dict):
            child_info = info.get(key, {}) if isinstance(info, dict) else {}
            yield from _iter_modalities(value, child_info, path + (str(key),))


def _is_primary_camera(path):
    label = "/".join(path).lower()
    return any(token in label for token in ("eyes", "head")) and not any(
        token in label for token in ("eef", "wrist")
    )


def _instance_bboxes(obs, info, target_ids, min_pixels):
    seg = np.squeeze(_to_numpy(obs.get("seg_instance")))
    labels = info.get("seg_instance") or {}
    if seg.ndim != 2 or not isinstance(labels, dict):
        return {}
    target_instance_ids = defaultdict(list)
    for raw_id, raw_label in labels.items():
        label = str(raw_label)
        path_parts = set(label.rstrip("/").split("/"))
        target = next(
            (target_id for target_id in target_ids if target_id == label or target_id in path_parts),
            None,
        )
        if target is None:
            continue
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        target_instance_ids[target].append(instance_id)

    result = {}
    for target, instance_ids in target_instance_ids.items():
        ys, xs = np.where(np.isin(seg, instance_ids))
        if len(xs) < min_pixels:
            continue
        result[target] = {
            "pixel_count": int(len(xs)),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "image_size": [int(seg.shape[1]), int(seg.shape[0])],
        }
    return result


def _semantic_from_instance(obs, info, instance_categories):
    """Build a stable category mask from the official instance render."""
    if "seg_instance" not in obs:
        return None, {}
    instance = np.squeeze(_to_numpy(obs["seg_instance"]))
    if instance.ndim != 2 or not np.issubdtype(instance.dtype, np.integer):
        return None, {}
    instance_labels = info.get("seg_instance") or {}
    categories = ["background"] + sorted(set(instance_categories.values()) - {"background"})
    category_ids = {category: index for index, category in enumerate(categories)}
    semantic = np.zeros(instance.shape, dtype=np.uint16)
    labels = {category_ids[category]: category for category in categories}
    for raw_id, raw_label in instance_labels.items():
        label = str(raw_label)
        category = "background" if label == "background" else instance_categories.get(label)
        if category is None:
            category = label
            if category not in category_ids:
                category_ids[category] = len(category_ids)
                labels[category_ids[category]] = category
        try:
            semantic[instance == int(raw_id)] = category_ids[category]
        except (TypeError, ValueError):
            continue
    return semantic, labels


def _segmentation_preview(array):
    """Colorize integer labels without changing the raw training mask."""
    labels = np.asarray(array, dtype=np.uint64)
    preview = np.zeros((*labels.shape, 3), dtype=np.uint8)
    non_background = labels != 0
    preview[..., 0] = np.where(non_background, 64 + (labels * 37) % 192, 0)
    preview[..., 1] = np.where(non_background, 64 + (labels * 67) % 192, 0)
    preview[..., 2] = np.where(non_background, 64 + (labels * 97) % 192, 0)
    return preview


def _save_camera_sample(obs, info, directory, target_ids, min_pixels, instance_categories):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    if "rgb" in obs:
        rgb = _to_numpy(obs["rgb"])[..., :3].astype(np.uint8)
        path = directory / "rgb.png"
        Image.fromarray(rgb).save(path)
        paths["rgb"] = str(path)
    derived_semantic = False
    semantic_labels = info.get("seg_semantic") or {}
    if "seg_semantic" not in obs:
        semantic, semantic_labels = _semantic_from_instance(obs, info, instance_categories)
        if semantic is not None:
            obs = dict(obs)
            obs["seg_semantic"] = semantic
            derived_semantic = True
    for modality in ("seg_semantic", "seg_instance"):
        if modality not in obs:
            continue
        array = np.squeeze(_to_numpy(obs[modality]))
        npy_path = directory / f"{modality}.npy"
        np.save(npy_path, array)
        paths[modality] = str(npy_path)
        if array.ndim == 2 and array.size and np.issubdtype(array.dtype, np.integer):
            png_path = directory / f"{modality}.png"
            Image.fromarray(_segmentation_preview(array), mode="RGB").save(png_path)
            paths[f"{modality}_preview"] = str(png_path)
    return {
        "paths": paths,
        "bboxes": _instance_bboxes(obs, info, target_ids, min_pixels),
        "labels": {
            "seg_instance": _jsonable(info.get("seg_instance") or {}),
            "seg_semantic": _jsonable(semantic_labels),
        },
        "semantic_source": (
            "derived_from_official_instance_segmentation"
            if derived_semantic
            else "official_semantic_segmentation"
        ),
    }


def _primary_robot_camera(robot):
    candidates = [
        (name, sensor)
        for name, sensor in robot.sensors.items()
        if hasattr(sensor, "add_modality") and hasattr(sensor, "get_position_orientation")
    ]
    primary = [item for item in candidates if _is_primary_camera((item[0],))]
    selected = primary[0] if primary else (candidates[0] if candidates else None)
    if selected is None:
        raise RuntimeError("robot has no primary camera sensor")
    return selected


def _capture_robot(
    robot, sensor, directory, target_ids, min_pixels, instance_categories, focus_object=None
):
    # Keep the primary sensor rigidly attached to Tiago. Target framing is done
    # exclusively through the robot's official pan/tilt joints.
    camera_obs, camera_info = sensor.get_obs()
    result = _save_camera_sample(
        camera_obs, camera_info, directory, target_ids, min_pixels, instance_categories
    )
    result["camera_path"] = [str(getattr(sensor, "name", "primary"))]
    result["capture_camera"] = "robot_primary_sensor"
    return result


def _capture_globals(camera_streams, directory, target_ids, min_pixels, instance_categories):
    results = []
    for camera, sensor in camera_streams:
        # A fixed Replicator RenderProduct can retain the previous pose of a
        # teleported object even with RTX temporal effects disabled. Force a
        # camera cut, then restore the exact approved generation camera pose,
        # so the saved frame contains only the current simulator state.
        position, orientation = sensor.get_position_orientation()
        cut_position = position.clone()
        cut_position[2] += 0.01
        sensor.set_position_orientation(position=cut_position, orientation=orientation)
        og.sim.render()
        sensor.set_position_orientation(position=position, orientation=orientation)
        og.sim.render()
        obs, info = sensor.get_obs()
        camera_id = str(camera.get("camera_id") or f"global_{len(results)}")
        result = _save_camera_sample(
            obs,
            info,
            directory / camera_id.replace("/", "_"),
            target_ids,
            min_pixels,
            instance_categories,
        )
        rgb = _to_numpy(obs.get("rgb")) if "rgb" in obs else None
        if rgb is not None and rgb.ndim >= 2:
            output_height, output_width = int(rgb.shape[0]), int(rgb.shape[1])
            geometric_bboxes = {}
            for target_id, visibility in (camera.get("visibility") or {}).items():
                if target_id not in target_ids:
                    continue
                bbox = visibility.get("bbox_xyxy")
                image_size = visibility.get("image_size")
                if not bbox or not image_size or len(bbox) != 4 or len(image_size) != 2:
                    continue
                source_width, source_height = (float(value) for value in image_size)
                if source_width <= 0 or source_height <= 0:
                    continue
                scale_x = output_width / source_width
                scale_y = output_height / source_height
                scaled_bbox = [
                    int(round(bbox[0] * scale_x)),
                    int(round(bbox[1] * scale_y)),
                    int(round(bbox[2] * scale_x)),
                    int(round(bbox[3] * scale_y)),
                ]
                scaled_pixels = int(round(float(visibility.get("pixel_count") or 0) * scale_x * scale_y))
                if scaled_pixels < min_pixels:
                    continue
                geometric_bboxes[target_id] = {
                    "pixel_count": scaled_pixels,
                    "bbox_xyxy": scaled_bbox,
                    "image_size": [output_width, output_height],
                    "bbox_clipped": bool(visibility.get("bbox_clipped")),
                    "visibility_source": visibility.get("visibility_source"),
                }
            result["bboxes"] = geometric_bboxes
        result["camera_id"] = camera_id
        result["room_id"] = camera.get("room_id")
        result["capture_camera"] = "fixed_official_global_rgb_sensor"
        result["visibility_evidence"] = "generation_frustum_physx_raycast"
        results.append(result)
    return results


def _capture_event_unprotected(
    env,
    run,
    output_dir,
    event_id,
    target_ids,
    min_pixels,
    focus_object_id=None,
    camera_streams=None,
):
    event_dir = output_dir / "frames" / event_id
    camera_streams = camera_streams or {}
    sensor = camera_streams.get("primary")
    if sensor is None:
        _, sensor = _primary_robot_camera(env.robots[0])
    og.sim.render()
    _nav_diag_r6_state(env, env.robots[0], f"capture:{event_id}:render_1")
    # Defect #19 round-5 characterization (additive, DELTASG_NAV_DIAG-gated):
    # bracket the capture sub-operations so a native mover can be pinned to the
    # bare renders vs the robot-camera ops vs the global-camera reads.
    _nav_diag_checkpoint(env, env.robots[0], f"cap_bare:{event_id}")
    objects = _scene_objects(env)
    instance_categories = {
        name: str(getattr(obj, "category", None) or "unknown")
        for name, obj in objects.items()
    }
    for robot in env.robots:
        instance_categories[str(getattr(robot, "name", "robot"))] = "robot"
    robot_camera = _capture_robot(
        env.robots[0],
        sensor,
        event_dir / "robot_primary",
        target_ids,
        min_pixels,
        instance_categories,
        objects.get(focus_object_id),
    )
    _nav_diag_checkpoint(env, env.robots[0], f"cap_robotcam:{event_id}")
    global_views = _capture_globals(
        camera_streams.get("globals") or [],
        event_dir / "global",
        target_ids,
        min_pixels,
        instance_categories,
    )
    _nav_diag_checkpoint(env, env.robots[0], f"cap_globals:{event_id}")
    global_bboxes = {}
    for view in global_views:
        global_bboxes.update(view["bboxes"])
    object_poses = {}
    for object_id in sorted(target_ids):
        obj = objects.get(object_id)
        if obj is None:
            continue
        try:
            position, orientation = obj.get_position_orientation()
            object_poses[object_id] = {
                "position": _jsonable(position),
                "orientation_xyzw": _jsonable(orientation),
            }
        except Exception:
            pass
    robot = env.robots[0]
    robot_position, robot_orientation = robot.get_position_orientation()
    camera_joint_positions = robot.get_joint_positions()[robot.camera_control_idx]
    return {
        "event_id": event_id,
        "robot_primary": robot_camera,
        "global_cameras": global_views,
        "robot_visible": sorted(robot_camera["bboxes"]),
        "global_visible": sorted(global_bboxes),
        "object_poses": object_poses,
        "robot_pose": {
            "position": _jsonable(robot_position),
            "orientation_xyzw": _jsonable(robot_orientation),
            "camera_joint_positions": _jsonable(camera_joint_positions),
        },
        "robot_stability": validate_robot_stability(env),
    }


def _capture_event(
    env,
    run,
    output_dir,
    event_id,
    target_ids,
    min_pixels,
    focus_object_id=None,
    camera_streams=None,
):
    return _capture_event_unprotected(
        env,
        run,
        output_dir,
        event_id,
        target_ids,
        min_pixels,
        focus_object_id,
        camera_streams,
    )


def _sink_diag_enabled():
    return os.environ.get("DELTASG_SINK_DIAG") == "1"


def _sink_diag_objects(env):
    names = [
        name.strip()
        for name in os.environ.get("DELTASG_SINK_DIAG_OBJECTS", "").split(",")
        if name.strip()
    ]
    found = []
    for name in names:
        obj = env.scene.object_registry("name", name, None)
        if obj is not None:
            found.append((name, obj))
    return found


_SINK_DIAG_STEP = [0]


def _sink_diag_trace(env, label):
    """Additive characterization probe (DELTASG_SINK_DIAG=1): log per-step
    origin z + AABB z of the objects named in DELTASG_SINK_DIAG_OBJECTS so a
    replayed-object sink can be discriminated between a relaxing native
    support and a contact-pair loss."""
    if not _sink_diag_enabled():
        return
    for name, obj in _sink_diag_objects(env):
        try:
            position, _ = obj.get_position_orientation()
            lower, upper = obj.aabb
            print(
                f"[sink-diag] step={_SINK_DIAG_STEP[0]} label={label} object={name} "
                f"origin_z={float(position[2]):.5f} "
                f"aabb_z=[{float(lower[2]):.5f},{float(upper[2]):.5f}]",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[sink-diag] step={_SINK_DIAG_STEP[0]} label={label} object={name} "
                f"error={exc!r}",
                flush=True,
            )


def _set_robot_pose(robot, pose, env=None, diag_env=None):
    position = pose.get("position")
    orientation = pose.get("orientation_xyzw")
    if not position or not orientation:
        raise RuntimeError("saved robot pose is incomplete")
    if not og.sim.is_playing():
        og.sim.play()
    robot.set_position_orientation(
        position=th.tensor(position, dtype=th.float32),
        orientation=th.tensor(orientation, dtype=th.float32),
        frame="scene",
    )
    robot.keep_still()
    # Match generation's scene-settling horizon before the integrity baseline.
    for _ in range(30):
        og.sim.step()
        if diag_env is not None:
            _SINK_DIAG_STEP[0] += 1
            _sink_diag_trace(diag_env, "set_robot_pose")
    if env is None:
        return
    # Convergence settle (mirrors api.stabilize_robot_spawn): the fixed 30-step
    # window was still mid-relaxation for slow fixtures (Merom_1 table lamp
    # moved 0.167 m after it and tripped the step-1 integrity recheck 2/2).
    # Keep stepping in windows until every native object's geometry center is
    # stable within the threshold; capped so pathological jitter cannot hang
    # the run. Added (online_env_*) objects are excluded: their replay sink is
    # a separate defect and must not stretch the native settle.
    settle_targets = []
    for name, obj in _scene_objects(env).items():
        if not name or name.startswith("online_env_") or obj in env.robots:
            continue
        settle_targets.append(obj)
    for window_index in range(20):
        pre_centers = {}
        for obj in settle_targets:
            try:
                center = _integrity_pose_record(obj)["geometry_center"]
            except Exception:
                center = None
            if center is not None:
                pre_centers[id(obj)] = (obj, center)
        for _ in range(15):
            og.sim.step()
            if diag_env is not None:
                _SINK_DIAG_STEP[0] += 1
                _sink_diag_trace(diag_env, "set_robot_pose")
        max_native_move = 0.0
        for obj, pre_center in pre_centers.values():
            try:
                post_center = _integrity_pose_record(obj)["geometry_center"]
            except Exception:
                post_center = None
            if post_center is None:
                continue
            moved = float(np.linalg.norm(post_center - pre_center))
            max_native_move = max(max_native_move, moved)
        print(
            f"[expert-settle] window={window_index + 1} "
            f"max_native_move={max_native_move:.5f}",
            flush=True,
        )
        if max_native_move < 0.005:
            break
    camera_joint_positions = pose.get("camera_joint_positions") or []
    if len(camera_joint_positions) == len(robot.camera_control_idx):
        robot.set_joint_positions(
            th.tensor(camera_joint_positions, dtype=th.float32),
            indices=robot.camera_control_idx,
            drive=False,
        )


def _saved_robot_pose_stability(pose, max_tilt=0.15):
    position = pose.get("position") or []
    orientation = pose.get("orientation_xyzw") or []
    if len(position) != 3 or len(orientation) != 4:
        return {"ok": False, "reason": "saved_robot_pose_incomplete"}
    values = [float(value) for value in position + orientation]
    if not all(math.isfinite(value) for value in values):
        return {"ok": False, "reason": "saved_robot_pose_nonfinite"}
    qx, qy, qz, qw = values[3:]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-6:
        return {"ok": False, "reason": "saved_robot_orientation_invalid"}
    tilt = math.sqrt((qx / norm) ** 2 + (qy / norm) ** 2)
    return {
        "ok": tilt <= max_tilt,
        "reason": None if tilt <= max_tilt else "saved_robot_pose_not_upright",
        "tilt": tilt,
        "max_tilt": max_tilt,
        "position": values[:3],
        "orientation_xyzw": values[3:],
    }


def _warm_natives_to_rest(env):
    """Settle natives under the render regime before the integrity baseline.

    Defect #19 (Beechwood_1 c1d45e183b10), mechanism round 4 verdict:
    the round-3 render-warmup fix was REFUTED — it converged at window 1 with
    max_native_move=0.00000 (nothing moved), yet bottom_cabinet_jhymlr_0 still
    moved 0.0561 / 0.1696 m in the two task capture windows and scene_integrity
    rejected with the same 0.1132 m root-link displacement. Plain render
    windows therefore do NOT surface the drift. The discriminating fact: a
    SLEEPING rigid body is excluded from PhysX simulation, so no render/step
    window can move it — it reads perfectly "at rest" (0.00000) until something
    wakes it. The later robot teleport + camera/segmentation setup perturbs the
    PhysX scene and wakes the metastable fixture, which then slides/tips
    mid-task relative to the baseline. Fix: explicitly wake every native before
    the settle windows so metastable USD-default poses converge to rest, THEN
    snapshot the baseline. Same shape as generation's post-warmup baselines and
    the #17/#20 convergence settles. The integrity gate is unchanged:
    post-baseline displacement during the task still rejects. Added
    (online_env_*) objects are excluded: they are reset to saved poses and
    gated by the delta replay integrity separately.
    """
    settle_targets = []
    for name, obj in _scene_objects(env).items():
        if not name or name.startswith("online_env_") or obj in env.robots:
            continue
        settle_targets.append(obj)
    # Wake sleeping natives so the settle windows below actually simulate them.
    # A sleeping body never moves under render()/step(), which is why the
    # round-3 warmup converged instantly (0.00000) while the same fixtures
    # drifted once the task-time teleport/camera setup woke them mid-run.
    asleep_names = []
    for obj in settle_targets:
        try:
            if bool(getattr(obj, "is_asleep", False)):
                asleep_names.append(str(getattr(obj, "name", "?")))
        except Exception:
            pass
        try:
            obj.wake()
        except Exception:
            pass
    print(
        f"[expert-warmup] woke natives asleep={len(asleep_names)} "
        f"sample={asleep_names[:8]}",
        flush=True,
    )
    for window_index in range(40):
        pre_centers = {}
        for obj in settle_targets:
            try:
                center = _integrity_pose_record(obj)["geometry_center"]
            except Exception:
                center = None
            if center is not None:
                pre_centers[id(obj)] = (obj, center)
        for _ in range(20):
            og.sim.render()
        max_native_move = 0.0
        for obj, pre_center in pre_centers.values():
            try:
                post_center = _integrity_pose_record(obj)["geometry_center"]
            except Exception:
                post_center = None
            if post_center is None:
                continue
            moved = float(np.linalg.norm(post_center - pre_center))
            max_native_move = max(max_native_move, moved)
        print(
            f"[expert-warmup] window={window_index + 1} "
            f"max_native_move={max_native_move:.5f}",
            flush=True,
        )
        if max_native_move < 0.005:
            break


def _create_expert_env(
    scene,
    robot_model,
    backend,
    robot_pose=None,
    added_objects=None,
    camera_resolution=None,
):
    if backend == "oracle_symbolic":
        # Match the physical backend's stable Replicator lifecycle: create the
        # environment with RGB and attach segmentation after the scene is live.
        # Preload generated delta objects in the initial environment config so
        # kinematic task supports are anchored before physics/contact views
        # initialize; dynamic post-construction spawning fell through the floor.
        return create_env(
            scene_model=scene,
            robot_model=str(robot_model).lower(),
            robot_obs_modalities=["rgb"],
            added_objects=added_objects,
            camera_resolution=camera_resolution,
        )
    if robot_model not in {"R1", "Tiago"}:
        raise ValueError("physical_control requires --robot R1 or --robot Tiago")
    import yaml

    config_path = Path(og.example_config_path) / f"{robot_model.lower()}_primitives.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot_config = dict(config["robots"][0])
    # The Isaac 5.1 Replicator graph can crash while Tiago initializes several
    # segmentation annotators. Start the official config with RGB, then attach
    # segmentation to the single primary camera after the environment is live.
    robot_config["obs_modalities"] = ["rgb"]
    if camera_resolution is not None:
        image_width, image_height = camera_resolution
        sensor_kwargs = robot_config.setdefault("sensor_config", {}).setdefault(
            "VisionSensor", {}
        ).setdefault("sensor_kwargs", {})
        sensor_kwargs["image_width"] = int(image_width)
        sensor_kwargs["image_height"] = int(image_height)
    if robot_model == "Tiago":
        robot_config["include_sensor_names"] = ["eyes"]
    robot_pose = robot_pose or {}
    if robot_pose.get("position") and robot_pose.get("orientation_xyzw"):
        position = [float(value) for value in robot_pose["position"]]
        qx, qy, qz, qw = (float(value) for value in robot_pose["orientation_xyzw"])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        robot_config["position"] = position
        robot_config["orientation"] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        robot_config["pose_frame"] = "scene"
    return og.Environment(
        configs={
            "env": {"action_frequency": 30, "physics_frequency": 60},
            "scene": {"type": "InteractiveTraversableScene", "scene_model": scene},
            "robots": [robot_config],
            "objects": added_objects or [],
        }
    )


def _initialize_segmentation_streams(env, run, camera_resolution=(640, 480)):
    """Attach fixed official instance streams without moving an active camera."""
    # DLSS / TAA blends previous object positions into the current frame after
    # symbolic navigation and placement. Dataset supervision must represent a
    # single simulator state, so use non-temporal AA at the higher resolution.
    settings = lazy.carb.settings.get_settings()
    settings.set_int("/rtx/post/aa/op", 0)
    settings.set_int("/rtx-defaults/post/aa/op", 0)
    settings.set_bool("/omni/replicator/captureMotionBlur", False)
    settings.set_bool("/rtx/post/motionblur/enabled", False)
    settings.set_bool("/rtx-defaults/post/motionblur/enabled", False)
    settings.set_bool("/rtx/raytracing/enableAccumulation", False)
    settings.set_bool("/rtx-defaults/raytracing/enableAccumulation", False)
    image_width, image_height = camera_resolution
    _, sensor = _primary_robot_camera(env.robots[0])
    sensor._deltasg_primary_local_pose = sensor.get_position_orientation(frame="parent")
    sensor.add_modality("seg_instance")
    for _ in range(5):
        og.sim.render()
    sensor.get_obs()
    print("[expert] robot instance stream initialized", flush=True)
    cameras = [
        camera
        for camera in ((run.get("task_environment") or {}).get("camera") or run.get("camera") or [])
        if camera.get("camera_type") == "global_camera"
    ]
    global_streams = []
    for index, camera in enumerate(cameras):
        pose = camera.get("pose") or {}
        position = pose.get("position")
        orientation = pose.get("orientation_xyzw")
        if not position or not orientation:
            continue
        camera_id = str(camera.get("camera_id") or f"global_{index}")
        safe_id = "".join(character if character.isalnum() else "_" for character in camera_id)
        global_sensor = VisionSensor(
            relative_prim_path=f"/deltasg_global_cameras/camera_{index}_{safe_id}",
            name=f"deltasg_global_{index}_{safe_id}",
            modalities=["rgb"],
            image_height=int(image_height),
            image_width=int(image_width),
        )
        global_sensor.load(env.scene)
        global_sensor.set_position_orientation(
            position=np.asarray(position, dtype=np.float32),
            orientation=np.asarray(orientation, dtype=np.float32),
        )
        global_sensor.initialize()
        global_streams.append((camera, global_sensor))
    for _ in range(5):
        og.sim.render()
    for _, global_sensor in global_streams:
        global_sensor.get_obs()
    print(f"[expert] fixed global RGB streams initialized count={len(global_streams)}", flush=True)
    return {"primary": sensor, "globals": global_streams}


def cleanup_persistent_camera_streams(camera_streams):
    """Remove sample-specific global sensors while keeping the robot stream alive."""
    if not camera_streams:
        return
    for _, sensor in camera_streams.get("globals") or []:
        try:
            sensor.remove()
        except Exception as exc:
            print(
                f"[expert-persistent] failed to remove global sensor "
                f"{getattr(sensor, 'name', '?')}: {exc!r}",
                flush=True,
            )
    for _ in range(2):
        og.sim.render()


def prepare_persistent_scene_reset(env):
    """Refresh PhysX views before hard-reset removes the previous Delta objects."""
    if og.sim.is_playing():
        og.sim.stop()
    og.sim.play()
    for _ in range(3):
        og.sim.step()
    env.robots[0].keep_still()


def configure_preloaded_delta_objects(env, run, preloaded_names):
    """Activate only this sample's objects in a shared symbolic scene."""
    active_names = {
        _name(record)
        for record in ((run.get("task_environment") or {}).get("added_objects") or [])
        if (record.get("placement") or {}).get("mode") != "reused"
    }
    objects = _scene_objects(env)
    missing = sorted(active_names - set(objects))
    if missing:
        raise RuntimeError(f"preloaded Delta objects missing: {missing}")
    for name in preloaded_names:
        obj = objects.get(name)
        if obj is None:
            continue
        active = name in active_names
        obj.visible = active
        for link in obj.links.values():
            if active:
                link.enable_collisions()
            else:
                link.disable_collisions()
        if active:
            obj.enable_gravity()
        else:
            obj.disable_gravity()
        obj.keep_still()
    print(
        f"[expert-persistent] activated delta objects={len(active_names)}/"
        f"{len(preloaded_names)}",
        flush=True,
    )


def _physical_added_object_configs(run, backend="physical_control"):
    configs = []
    te = run.get("task_environment") or {}
    for record in te.get("added_objects") or []:
        placement = record.get("placement") or {}
        if placement.get("mode") == "reused":
            continue
        name = _name(record)
        pose = record.get("pose") or record.get("final_pose_before_warmup") or placement.get("pose") or {}
        if not name or not record.get("category") or not pose.get("position"):
            raise RuntimeError(f"cannot configure added object {name!r}")
        task_support = "task_support" in set(record.get("semantic_roles") or [])
        anchored_for_replay = task_support
        config = {
            "type": "DatasetObject",
            "name": name,
            "category": record["category"],
            "position": pose["position"],
            "orientation": pose.get("orientation_xyzw") or [0, 0, 0, 1],
            # Delta supports are furniture, not task-actuated objects. Anchor
            # them in the reconstructed environment so a robot contact cannot
            # turn an otherwise valid manipulation episode into scene drift.
            "fixed_base": anchored_for_replay,
            "kinematic_only": anchored_for_replay,
        }
        if record.get("room_id"):
            config["in_rooms"] = record["room_id"]
        if record.get("model"):
            config["model"] = record["model"]
        configs.append(config)
    return configs


def _saved_robot_approaches(run):
    te = run.get("task_environment") or {}
    records = [
        *(te.get("added_objects") or []),
        *((te.get("task") or run.get("task") or {}).get("plan_objects") or []),
    ]
    approaches = {}
    approach_distances = {}
    for record in records:
        object_id = _name(record)
        placement = record.get("placement") or {}
        validation = record.get("validation") or {}
        approach = (
            placement.get("robot_approach")
            or validation.get("robot_approach")
            or record.get("robot_approach")
            or {}
        )
        xy = approach.get("candidate_position_xy")
        if object_id and approach.get("ok") is True and isinstance(xy, list) and len(xy) == 2:
            distance = float(approach.get("horizontal_distance", math.inf))
            if object_id not in approaches or distance < approach_distances[object_id]:
                approaches[object_id] = [float(xy[0]), float(xy[1])]
                approach_distances[object_id] = distance
    return approaches


def _spawn_added_objects(env, run):
    spawned = []
    te = run.get("task_environment") or {}
    for record in te.get("added_objects") or []:
        placement = record.get("placement") or {}
        if placement.get("mode") == "reused":
            continue
        name = _name(record)
        pose = record.get("pose") or record.get("final_pose_before_warmup") or placement.get("pose") or {}
        if not name or not record.get("category") or not pose.get("position"):
            raise RuntimeError(f"cannot reconstruct added object {name!r}")
        task_support = "task_support" in set(record.get("semantic_roles") or [])
        obj = DatasetObject(
            name=name,
            category=record["category"],
            model=record.get("model"),
            prim_type=PrimType.RIGID,
            in_rooms=record.get("room_id"),
            fixed_base=task_support,
            kinematic_only=task_support,
        )
        env.scene.add_object(obj)
        obj.set_position_orientation(
            position=th.tensor(pose["position"], dtype=th.float32),
            orientation=th.tensor(pose.get("orientation_xyzw") or [0, 0, 0, 1], dtype=th.float32),
        )
        try:
            obj.keep_still()
        except Exception:
            pass
        spawned.append(obj)
    for _ in range(20):
        og.sim.step()
    return spawned


def _hold_symbolic_grasp_targets(env, grasp_target_ids):
    """Keep generated portable targets at their validated pose until GRASP."""
    objects = _scene_objects(env)
    held = []
    for object_id in grasp_target_ids:
        obj = objects.get(object_id)
        if obj is None:
            continue
        obj.disable_gravity()
        for link in obj.links.values():
            link.disable_collisions()
        obj.keep_still()
        held.append(object_id)
    if held:
        print(f"[expert] holding symbolic grasp targets={sorted(held)}", flush=True)
    return held


def _apply_saved_initial_states(env, run):
    """Replay task setup states before executing the saved expert plan."""
    state_types = {
        "open": object_states.Open,
        "toggled_on": object_states.ToggledOn,
        "on_fire": object_states.OnFire,
    }
    objects = _scene_objects(env)
    applied = []
    smoke_effect_count = 0
    te = run.get("task_environment") or {}
    for record in te.get("state_changed_objects") or []:
        object_id = _name(record)
        obj = objects.get(object_id)
        if obj is None:
            raise RuntimeError(f"state target {object_id!r} is missing during replay")
        for state_key, desired in (record.get("states") or {}).items():
            state_cls = state_types.get(str(state_key).lower())
            if state_cls is None:
                continue
            if state_cls not in obj.states:
                raise RuntimeError(f"{object_id!r} does not expose state {state_key!r}")
            current = bool(obj.states[state_cls].get_value())
            if current != bool(desired):
                # #25: mirror the generator's deterministic transitions (fully=True
                # in BOTH directions) so the replayed initial state for
                # open_fridge / close_cabinet / close_fridge stays put instead of
                # drifting across the 5% threshold on the same random-sampling bug
                # that made an Open door read the wrong value after settle.
                if state_cls is object_states.Open:
                    applied_ok = bool(obj.states[state_cls].set_value(bool(desired), fully=True))
                else:
                    applied_ok = bool(obj.states[state_cls].set_value(bool(desired)))
                if not applied_ok:
                    raise RuntimeError(
                        f"failed to replay state {state_key}={bool(desired)} for {object_id!r}"
                    )
            applied.append({"object_id": object_id, "state": state_key, "value": bool(desired)})
        visual_effect = record.get("visual_effect") or {}
        if visual_effect.get("mode") == SMOKE_ONLY_ON_FIRE_MODE:
            configured = configure_on_fire_smoke_only(obj)
            if not configured.get("ok"):
                raise RuntimeError(
                    f"failed to replay smoke-only OnFire effect for {object_id!r}: {configured}"
                )
            smoke_effect_count += 1
    if applied:
        settle_steps = SMOKE_FLOW_WARMUP_STEPS if smoke_effect_count else 5
        for _ in range(settle_steps):
            og.sim.step()
        if smoke_effect_count:
            for _ in range(SMOKE_FLOW_RENDER_WARMUP_FRAMES):
                og.sim.render()
            print(
                f"[expert] warmed official smoke-only Flow effects="
                f"{smoke_effect_count} steps={settle_steps} "
                f"render_frames={SMOKE_FLOW_RENDER_WARMUP_FRAMES}"
            )
    return applied


def _delta_replay_integrity(env, run, max_displacement, settle_steps=20, reset_aabb_tops=None):
    """Assert preloaded delta objects stay anchored during replay settling.

    Runs immediately after replay and before camera/segmentation setup so a
    fallen support rejects fast instead of wasting a render setup and
    producing misleading images.
    """
    for _ in range(settle_steps):
        og.sim.step()
        _SINK_DIAG_STEP[0] += 1
        _sink_diag_trace(env, "delta_replay_settle")
    objects = _scene_objects(env)
    entries = []
    ok = True
    te = run.get("task_environment") or {}
    for record in te.get("added_objects") or []:
        placement = record.get("placement") or {}
        if placement.get("mode") == "reused":
            continue
        object_id = _name(record)
        pose = record.get("pose") or record.get("final_pose_before_warmup") or placement.get("pose") or {}
        kinematic_only = "task_support" in set(record.get("semantic_roles") or [])
        stationary = "task_object" not in set(record.get("semantic_roles") or [])
        obj = objects.get(object_id)
        saved_position = pose.get("position")
        if obj is None or not saved_position:
            ok = False
            entries.append({
                "object_id": object_id,
                "saved_pose": _jsonable(saved_position),
                "replayed_pose": None,
                "displacement": None,
                "kinematic_only": kinematic_only,
            })
            continue
        replayed_position = np.asarray(obj.get_position_orientation()[0], dtype=float)
        saved_position_array = np.asarray(saved_position, dtype=float)
        displacement = float(np.linalg.norm(replayed_position - saved_position_array))
        vertical_displacement = float(replayed_position[2] - saved_position_array[2])
        if stationary and displacement > max_displacement:
            ok = False
        # Self-referential sink gate: the reset teleported the object to its
        # saved pose while physics was playing; if its AABB top drops by more
        # than the tolerance during the replay settle, it tunneled into its
        # support (Rs_int book: 0.053 m sink, displacement 0.0439 slipped
        # under the 0.05 displacement gate and hid the object from every
        # robot-primary segmentation). No external geometry assumptions.
        sunk = False
        replayed_aabb_top = None
        reset_aabb_top = (reset_aabb_tops or {}).get(object_id)
        manipulation_height = (
            (record.get("validation") or {}).get("manipulation_height")
            or placement.get("manipulation_height")
            or {}
        )
        generation_aabb_top = manipulation_height.get("aabb_max_z")
        comparison_aabb_top = (
            float(generation_aabb_top)
            if generation_aabb_top is not None
            else reset_aabb_top
        )
        if comparison_aabb_top is not None:
            try:
                _, upper = obj.aabb
                replayed_aabb_top = float(upper[2])
                if (
                    vertical_displacement < -0.01
                    and replayed_aabb_top < comparison_aabb_top - 0.015
                ):
                    sunk = True
                    ok = False
            except Exception:
                replayed_aabb_top = None
        entries.append({
            "object_id": object_id,
            "saved_pose": _jsonable(saved_position),
            "replayed_pose": _jsonable(replayed_position),
            "displacement": displacement,
            "vertical_displacement": vertical_displacement,
            "kinematic_only": kinematic_only,
            "reset_aabb_top": reset_aabb_top,
            "generation_aabb_top": generation_aabb_top,
            "comparison_aabb_top": comparison_aabb_top,
            "replayed_aabb_top": replayed_aabb_top,
            "sunk": sunk,
        })
    return {
        "ok": ok,
        "max_displacement": max_displacement,
        "settle_steps": settle_steps,
        "objects": entries,
    }


def _scene_objects(env):
    return {getattr(obj, "name", ""): obj for obj in get_all_scene_objects(env.scene)}


def _capture_reference_scene_bboxes(env, support_surface_ids=None):
    """Fix P: snapshot the sampler-input geometry before it can be corrupted.

    attempt-9/diag30: OnTop place sampling reads
    ``target.get_base_aligned_bbox(xy_aligned=True)`` at draw time
    (sampling_utils ray start points). After the GRASP FixedJoint + stop/play
    rebuild that quantity can transiently report an inflated box (observed:
    dining-set-union ring candidates at inflated z) even while ``obj.aabb``,
    scene integrity and global frames all look normal. Capture the exact
    sampler inputs once, pre-grasp, so the place sampler override can reject
    corrupted draws against this reference.

    Fix P5 (attempt 11): also snapshot, for the plan objects, a pre-grasp
    downward-raycast map of each target's support column (20 rays: a 4x4
    interior grid plus 4 corner rays at +-0.45 of the extents — review R4,
    so P3-corridor corner candidates stay covered by the cross-check). The live Fix P4 probe shares the sampler's scene-query
    layer, so a uniform loss of a tabletop collision mesh (rays pass through
    the top and hit the underside) is invisible to any same-instant live
    check; the pre-grasp map is the independent healthy-geometry source that
    flags it (support_surface_lost_vs_reference).
    """
    reference = {}
    for name, obj in _scene_objects(env).items():
        try:
            center, _, extent, _ = obj.get_base_aligned_bbox(xy_aligned=True)
            low, high = obj.aabb
        except Exception:
            continue
        reference[name] = {
            "bbox_center": [round(float(v), 4) for v in _to_numpy(center)],
            "bbox_extent": [round(float(v), 4) for v in _to_numpy(extent)],
            "aabb_top": round(float(_to_numpy(high)[2]), 4),
        }
        if support_surface_ids is not None and name not in support_surface_ids:
            continue
        if support_surface_ids is None:
            continue
        try:
            from omnigibson.utils import sampling_utils

            cx, cy = float(_to_numpy(center)[0]), float(_to_numpy(center)[1])
            ex, ey = float(_to_numpy(extent)[0]), float(_to_numpy(extent)[1])
            z_start = float(_to_numpy(high)[2]) + 0.05
            z_end = float(_to_numpy(low)[2]) - 0.05
            hits = []
            grid = [
                (fx, fy)
                for fx in (-0.375, -0.125, 0.125, 0.375)
                for fy in (-0.375, -0.125, 0.125, 0.375)
            ]
            # Review R4: the interior 4x4 grid leaves P3-corridor corner
            # candidates (footprint + XY_MARGIN) farther than
            # PLACE_SUPPORT_REFERENCE_XY_RADIUS from any reference ray, so a
            # through-table draw in a corner would escape both P4 (the live
            # probe shares the broken query layer) and P5 (no nearby
            # reference hit -> silently skipped). Corner rays at +-0.45 of
            # the extents close the gap for extents up to ~1 m.
            grid += [
                (fx, fy)
                for fx in (-0.45, 0.45)
                for fy in (-0.45, 0.45)
            ]
            for fx, fy in grid:
                rx = cx + fx * ex
                ry = cy + fy * ey
                ray = sampling_utils.raytest(
                    start_point=th.tensor([rx, ry, z_start]),
                    end_point=th.tensor([rx, ry, z_end]),
                    only_closest=True,
                )
                if ray.get("hit"):
                    hits.append([
                        round(rx, 4),
                        round(ry, 4),
                        round(float(ray["position"][2]), 4),
                    ])
            reference[name]["support_surface_hits"] = hits
        except Exception:
            continue
    return reference


_NAV_DIAG_LAST = {}


def _nav_diag_r6_state(env, robot, tag):
    """Log the #19 fixture dynamics and closest Tiago link AABB."""
    if os.environ.get("DELTASG_NAV_DIAG_R6") != "1":
        return
    object_id = os.environ.get("DELTASG_NAV_DIAG_OBJECT", "bottom_cabinet_jhymlr_0")
    obj = _scene_objects(env).get(object_id)
    if obj is None:
        print(f"[nav-diag-r6] {tag} object={object_id} missing", flush=True)
        return

    try:
        lower, upper = obj.aabb
        lower = np.asarray(lower.cpu(), dtype=float)
        upper = np.asarray(upper.cpu(), dtype=float)
        center = (lower + upper) * 0.5
    except Exception as exc:
        print(f"[nav-diag-r6] {tag} object={object_id} aabb_error={exc!r}", flush=True)
        return

    def vector(getter):
        try:
            return np.asarray(getter().cpu(), dtype=float)
        except Exception:
            return np.full(3, np.nan, dtype=float)

    linear_velocity = vector(obj.get_linear_velocity)
    angular_velocity = vector(obj.get_angular_velocity)
    try:
        asleep = bool(obj.is_asleep)
    except Exception:
        asleep = None

    closest = None
    overlaps = []
    for link_name, link in (getattr(robot, "links", None) or {}).items():
        try:
            link_lower, link_upper = link.aabb
            link_lower = np.asarray(link_lower.cpu(), dtype=float)
            link_upper = np.asarray(link_upper.cpu(), dtype=float)
        except Exception:
            continue
        axis_gap = np.maximum(np.maximum(lower - link_upper, link_lower - upper), 0.0)
        distance = float(np.linalg.norm(axis_gap))
        record = (distance, str(link_name), link_lower, link_upper)
        if closest is None or record[0] < closest[0]:
            closest = record
        if distance == 0.0:
            overlaps.append(str(link_name))

    closest_text = "none"
    if closest is not None:
        distance, link_name, link_lower, link_upper = closest
        closest_text = (
            f"{link_name}:{distance:.5f}:"
            f"[{','.join(f'{value:.3f}' for value in link_lower)}]-"
            f"[{','.join(f'{value:.3f}' for value in link_upper)}]"
        )
    print(
        f"[nav-diag-r6] {tag} object={object_id} asleep={asleep} "
        f"center=({center[0]:.5f},{center[1]:.5f},{center[2]:.5f}) "
        f"linear=({linear_velocity[0]:.6f},{linear_velocity[1]:.6f},{linear_velocity[2]:.6f}) "
        f"angular=({angular_velocity[0]:.6f},{angular_velocity[1]:.6f},{angular_velocity[2]:.6f}) "
        f"cabinet_aabb=[{','.join(f'{value:.3f}' for value in lower)}]-"
        f"[{','.join(f'{value:.3f}' for value in upper)}] "
        f"closest_robot_link={closest_text} overlaps={overlaps}",
        flush=True,
    )


def _nav_diag_checkpoint(env, robot, tag):
    """DELTASG_NAV_DIAG: attribute native displacement to execution intervals.

    Additive characterization only (defect #19, Beechwood_1 c1d45e183b10):
    prints every native object whose AABB centre moved more than 0.01 m since
    the previous checkpoint, so a displacement can be pinned to the exact
    teleport/settle/action interval instead of the whole step. Mover lines
    also report root-link / max-link / joint deltas and the robot distance to
    the mover AABB so contact-driven pushing can be told apart from
    scene-intrinsic settling.
    """
    if os.environ.get("DELTASG_NAV_DIAG") != "1":
        return
    centers = {}
    roots = {}
    link_sets = {}
    joint_set = {}
    for name, obj in _scene_objects(env).items():
        if not name or name.startswith("online_env_") or obj is robot:
            continue
        try:
            lower, upper = obj.aabb
            centers[name] = np.asarray(((lower + upper) * 0.5).cpu(), dtype=float)
        except Exception:
            continue
        try:
            roots[name] = np.asarray(obj.root_link.get_position_orientation()[0].cpu(), dtype=float)
        except Exception:
            pass
        try:
            links = getattr(obj, "links", None) or {}
            if links:
                positions = {}
                for link_name, link in links.items():
                    positions[link_name] = np.asarray(
                        link.get_position_orientation()[0].cpu(), dtype=float
                    )
                link_sets[name] = positions
        except Exception:
            pass
        try:
            if int(getattr(obj, "n_joints", 0) or 0) > 0:
                joint_set[name] = np.asarray(obj.get_joint_positions().cpu(), dtype=float)
        except Exception:
            pass
    previous = _NAV_DIAG_LAST.get("centers")
    if previous:
        prev_roots = _NAV_DIAG_LAST.get("roots") or {}
        prev_links = _NAV_DIAG_LAST.get("links") or {}
        prev_joints = _NAV_DIAG_LAST.get("joints") or {}
        try:
            robot_position = np.asarray(robot.get_position_orientation()[0].cpu(), dtype=float)
        except Exception:
            robot_position = None
        for name, center in centers.items():
            pre = previous.get(name)
            if pre is None:
                continue
            moved = float(np.linalg.norm(center - pre))
            if moved <= 0.01:
                continue
            detail = []
            detail.append(f"acenter=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})")
            pre_root = prev_roots.get(name)
            root = roots.get(name)
            if pre_root is not None and root is not None:
                detail.append(f"root_moved={float(np.linalg.norm(root - pre_root)):.4f}")
                detail.append(f"aroot=({root[0]:.3f},{root[1]:.3f},{root[2]:.3f})")
            pre_link_set = prev_links.get(name) or {}
            cur_link_set = link_sets.get(name) or {}
            link_max, link_max_name = 0.0, ""
            for link_name, cur_pos in cur_link_set.items():
                pre_pos = pre_link_set.get(link_name)
                if pre_pos is None:
                    continue
                link_moved = float(np.linalg.norm(cur_pos - pre_pos))
                if link_moved > link_max:
                    link_max, link_max_name = link_moved, link_name
            if pre_link_set:
                detail.append(f"link_max={link_max:.4f}({link_max_name})")
            pre_joint = prev_joints.get(name)
            cur_joint = joint_set.get(name)
            if pre_joint is not None and cur_joint is not None and pre_joint.shape == cur_joint.shape:
                detail.append(f"joint_max_delta={float(np.max(np.abs(cur_joint - pre_joint))):.4f}")
            if robot_position is not None:
                try:
                    lower, upper = _scene_objects(env)[name].aabb
                    lower_np = np.asarray(lower.cpu(), dtype=float)
                    upper_np = np.asarray(upper.cpu(), dtype=float)
                    nearest = np.clip(robot_position, lower_np, upper_np)
                    detail.append(f"robot_dist_aabb={float(np.linalg.norm(robot_position - nearest)):.3f}")
                    detail.append(f"robot_xy=({robot_position[0]:.3f},{robot_position[1]:.3f})")
                except Exception:
                    pass
            suffix = (" " + " ".join(detail)) if detail else ""
            print(
                f"[nav-diag] {tag}: {name} moved={moved:.4f} since previous checkpoint{suffix}",
                flush=True,
            )
    _NAV_DIAG_LAST["centers"] = centers
    _NAV_DIAG_LAST["roots"] = roots
    _NAV_DIAG_LAST["links"] = link_sets
    _NAV_DIAG_LAST["joints"] = joint_set


def _teleport_robot_preserving_delta_objects(env, robot, position, orientation):
    """Move the robot without restarting physics for native or delta objects."""
    if not og.sim.is_playing():
        og.sim.play()
    _nav_diag_checkpoint(env, robot, "teleport_pre")
    _nav_diag_r6_state(env, robot, "teleport_pre")
    robot.set_position_orientation(position=position, orientation=orientation, frame="scene")
    robot.keep_still()
    _nav_diag_r6_state(env, robot, "teleport_post_before_render")
    if os.environ.get("DELTASG_NAV_DIAG_R6") == "1":
        for render_index in range(3):
            og.sim.render()
            _nav_diag_r6_state(env, robot, f"teleport_only_render_{render_index + 1}")
    _nav_diag_checkpoint(env, robot, "teleport_post")


def _reset_delta_objects_to_saved_poses(env, run):
    """Teleport preloaded delta objects back to their saved generation poses.

    Objects present in the initial config from play start can spawn in exact
    contact with their supports; PhysX then resolves the construction-time
    overlap by pushing them through thin-shell collision. Rs_int paperback
    book: sank 0.053 m into coffee_table_fqluyq_0 during env load while the
    table stayed perfectly still (sink-diag step 0) — contact-pair loss at
    the play transition, not fixture relaxation. Generation settled the same
    pose at 3.4e-6 m displacement, and runtime teleports while physics is
    playing are the regime where contact pairs rebuild correctly, so reset
    every added object after the natives have converged and before the
    baseline/replay gates. Returns the post-reset AABB top per object for the
    replay sink gate.
    """
    objects = _scene_objects(env)
    reset_aabb_tops = {}
    reset_ids = []
    records = [
        record
        for record in (run.get("task_environment") or {}).get("added_objects") or []
        if (record.get("placement") or {}).get("mode") != "reused"
    ]
    records_by_name = {_name(record): record for record in records}

    def saved_pose(record):
        placement = record.get("placement") or {}
        return (
            record.get("pose")
            or record.get("final_pose_before_warmup")
            or placement.get("pose")
            or {}
        )

    def reset_record(record, position_override=None):
        object_id = _name(record)
        obj = objects.get(object_id)
        pose = saved_pose(record)
        position = pose.get("position")
        orientation = pose.get("orientation_xyzw")
        if obj is None or not position or not orientation:
            return None
        position = position_override if position_override is not None else position
        try:
            obj.set_position_orientation(
                position=th.tensor(position, dtype=th.float32),
                orientation=th.tensor(orientation, dtype=th.float32),
                frame="scene",
            )
            obj.keep_still()
        except Exception as exc:
            print(f"[delta-reset] failed to reset {object_id}: {exc!r}", flush=True)
            return None
        reset_ids.append(object_id)
        return obj

    support_records = [
        record
        for record in records
        if "task_support" in set(record.get("semantic_roles") or [])
    ]
    task_records = [record for record in records if record not in support_records]
    for record in support_records:
        reset_record(record)

    # A portable object saved OnTop depends on the support's settled contact
    # pose. Let preloaded supports converge first, then preserve the saved
    # object-to-support translation when restoring the portable object.
    if support_records:
        for _ in range(20):
            og.sim.step()
    for record in task_records:
        placement = record.get("placement") or {}
        support_id = placement.get("support_object_id")
        position_override = None
        support_record = records_by_name.get(support_id)
        support_obj = objects.get(support_id)
        object_position = saved_pose(record).get("position")
        support_position = saved_pose(support_record).get("position") if support_record else None
        if support_obj is not None and object_position and support_position:
            live_support_position = np.asarray(
                support_obj.get_position_orientation()[0], dtype=float
            )
            position_override = (
                np.asarray(object_position, dtype=float)
                + live_support_position
                - np.asarray(support_position, dtype=float)
            ).tolist()
        obj = reset_record(record, position_override=position_override)
        if (
            obj is not None
            and support_obj is not None
            and placement.get("mode") == "on_top"
        ):
            try:
                object_lower, _ = obj.aabb
                _, support_upper = support_obj.aabb
                clearance = 0.005
                z_correction = float(support_upper[2] + clearance - object_lower[2])
                if abs(z_correction) > 1e-4:
                    position, orientation = obj.get_position_orientation()
                    position = position.clone()
                    position[2] += z_correction
                    obj.set_position_orientation(
                        position=position,
                        orientation=orientation,
                        frame="scene",
                    )
                    obj.keep_still()
                print(
                    f"[delta-reset] aligned {obj.name} on {support_obj.name}: "
                    f"z_correction={z_correction:.5f} clearance={clearance:.3f}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[delta-reset] failed to align {obj.name} on {support_obj.name}: "
                    f"{exc!r}",
                    flush=True,
                )

    for record in records:
        object_id = _name(record)
        obj = objects.get(object_id)
        if obj is None:
            continue
        try:
            _, upper = obj.aabb
            reset_aabb_tops[object_id] = float(upper[2])
        except Exception:
            pass
    if reset_ids:
        print(
            f"[delta-reset] support-first teleported {len(reset_ids)} delta objects: "
            f"{reset_ids}",
            flush=True,
        )
    return reset_aabb_tops


def _native_pose_snapshot(env):
    result = {}
    for name, obj in _scene_objects(env).items():
        if not name or name.startswith("online_env_") or obj in env.robots:
            continue
        try:
            result[name] = _integrity_pose_record(obj)
        except Exception:
            pass
    return result


def _integrity_pose_record(obj):
    position = np.asarray(obj.get_position_orientation()[0], dtype=float)
    root_link_position = None
    try:
        root_link_position = np.asarray(
            obj.root_link.get_position_orientation()[0], dtype=float
        )
    except Exception:
        pass
    link_positions = {}
    for name, link in getattr(obj, "links", {}).items():
        try:
            link_positions[name] = np.asarray(
                link.get_position_orientation()[0], dtype=float
            )
        except Exception:
            pass
    geometry_center = None
    aabb_state = next(
        (state for state in obj.states.values() if state.__class__.__name__ == "AABB"),
        None,
    )
    if aabb_state is not None:
        try:
            lower, upper = aabb_state.get_value()
            geometry_center = (
                np.asarray(lower, dtype=float) + np.asarray(upper, dtype=float)
            ) * 0.5
        except Exception:
            geometry_center = None
    return {
        "position": position,
        "fixed_base": bool(getattr(obj, "fixed_base", False)),
        "root_link_position": root_link_position,
        "geometry_center": geometry_center,
        "link_positions": link_positions,
    }


def _scene_integrity(env, baseline, max_displacement):
    objects = _scene_objects(env)
    missing = sorted(set(baseline) - set(objects))
    moved = []
    for name, start in baseline.items():
        obj = objects.get(name)
        if obj is None:
            continue
        try:
            end = _integrity_pose_record(obj)
        except Exception:
            continue
        if isinstance(start, dict):
            start_position = start["position"]
            start_center = start.get("geometry_center")
        else:
            start_position = start
            start_center = None
        start_links = start.get("link_positions") if isinstance(start, dict) else None
        common_links = set(start_links or {}) & set(end["link_positions"])
        start_root = start.get("root_link_position") if isinstance(start, dict) else None
        if (
            isinstance(start, dict)
            and start.get("fixed_base")
            and end.get("fixed_base")
            and not name.startswith("online_env_")
        ):
            displacement = 0.0
        elif start_root is not None and end["root_link_position"] is not None:
            displacement = float(np.linalg.norm(end["root_link_position"] - start_root))
        elif common_links:
            displacement = max(
                float(np.linalg.norm(end["link_positions"][name] - start_links[name]))
                for name in common_links
            )
        elif start_center is not None and end["geometry_center"] is not None:
            displacement = float(np.linalg.norm(end["geometry_center"] - start_center))
        else:
            displacement = float(np.linalg.norm(end["position"][:3] - start_position[:3]))
        if displacement > max_displacement:
            moved.append({"object_id": name, "displacement": displacement})
    return {"ok": not missing and not moved, "missing": missing, "moved": moved}


def _combined_scene_integrity(env, native_baseline, stationary_delta_baseline, max_displacement):
    native = _scene_integrity(env, native_baseline, max_displacement)
    stationary_delta = _scene_integrity(env, stationary_delta_baseline, max_displacement)
    missing_detail = [
        *({"object_id": name, "group": "native"} for name in native["missing"]),
        *({"object_id": name, "group": "stationary_delta"} for name in stationary_delta["missing"]),
    ]
    missing = [item["object_id"] for item in missing_detail]
    moved = [
        *({**item, "group": "native"} for item in native["moved"]),
        *({**item, "group": "stationary_delta"} for item in stationary_delta["moved"]),
    ]
    return {
        "ok": not missing and not moved,
        "missing": missing,
        "missing_detail": missing_detail,
        "moved": moved,
        "native": native,
        "stationary_delta": stationary_delta,
    }


def _rotate_toward(env, robot, obj, fallback_rank=0, fallback_yaw_offset=-10.0):
    robot_pos, _ = robot.get_position_orientation()
    pose = None
    if _horizontal_target_aabb_distance(robot, obj) <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE:
        target_position, _ = obj.get_position_orientation()
        current_yaw = math.atan2(
            float(target_position[1] - robot_pos[1]),
            float(target_position[0] - robot_pos[0]),
        )
        if _native_occupant_at_pose(
            env, robot, obj, robot_pos[:2], current_yaw
        ) is None:
            pose = th.tensor(
                [float(robot_pos[0]), float(robot_pos[1]), current_yaw],
                dtype=th.float32,
            )
    if pose is None:
        try:
            pose = _connected_observation_pose(
                env,
                robot,
                obj,
                preferred_distance=0.85 + 0.25 * fallback_rank,
                fallback_rank=fallback_rank,
                fallback_yaw_offset=fallback_yaw_offset,
                max_target_aabb_distance=DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
            )
        except Exception:
            return False
    yaw = float(pose[2])
    orientation = th.tensor([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=th.float32)
    position = th.tensor([float(pose[0]), float(pose[1]), float(robot_pos[2])], dtype=th.float32)
    if os.environ.get("DELTASG_NAV_DIAG") == "1":
        print(
            f"[nav-diag] rotate_toward target={getattr(obj, 'name', '?')} "
            f"framing_pose=({float(pose[0]):.3f},{float(pose[1]):.3f},{yaw:.3f}) "
            f"fallback_rank={fallback_rank}",
            flush=True,
        )
    _teleport_robot_preserving_delta_objects(env, robot, position, orientation)
    head_aim = _aim_tiago_head(robot, obj)
    for _ in range(10):
        robot.keep_still()
        with og.sim.render_on_step(False):
            og.sim.step()
    # SIGSEGV hardening (Ihlen_1 8cf022fc134d / Merom_0 8c1fd6ae00c0): the
    # teleport above moved the robot-primary camera prim and the loop ran
    # render-off. Drain the post-process graph with a few render-only ticks
    # before the caller's next real render burst.
    for _ in range(3):
        og.sim.render()
    _nav_diag_checkpoint(env, robot, "rotate_toward_settle_post")
    return {
        "base_aimed": True,
        "head_aimed": head_aim["aimed"],
        "head_aim": head_aim,
        "fallback_rank": fallback_rank,
        "fallback_yaw_offset": fallback_yaw_offset,
        "pose_xyyaw": _jsonable(pose),
    }


def _aim_tiago_head(robot, obj):
    """Point Tiago's official pan/tilt joints at the target AABB centre."""
    if str(getattr(robot, "model", "")).lower() != "tiago":
        return {"aimed": False, "reason": "robot_has_no_tiago_head"}
    try:
        lower, upper = obj.aabb
        target_position = (lower + upper) * 0.5
        size = upper - lower
        if float(th.linalg.norm(size[:2])) >= 1.0 or float(size[2]) >= 0.6:
            target_position[2] = lower[2] + 0.05 * size[2]
        robot_pose = robot.get_position_orientation()
        target_in_base = T.relative_pose_transform(
            target_position,
            th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            *robot_pose,
        )[0]
        head_position, head_orientation = robot.links["head_2_link"].get_position_orientation()
        head_in_base = T.relative_pose_transform(
            head_position, head_orientation, *robot_pose
        )[0]
        pan = math.atan2(float(target_in_base[1]), float(target_in_base[0]))
        horizontal = max(float(th.linalg.norm(target_in_base[:2])), 1e-6)
        tilt = math.atan2(float(target_in_base[2] - head_in_base[2]), horizontal)
        head_joints = (robot.joints["head_1_joint"], robot.joints["head_2_joint"])
        margin = 1e-3
        requested = (pan, tilt)
        clamped = [
            min(
                max(value, float(joint.lower_limit) + margin),
                float(joint.upper_limit) - margin,
            )
            for value, joint in zip(requested, head_joints)
        ]
        robot.set_joint_positions(
            th.tensor(clamped, dtype=th.float32),
            indices=robot.camera_control_idx,
            drive=False,
        )
        # The next official capture performs the render. Rendering here after
        # changing the camera-bearing head joints can crash SyntheticData while
        # it rebuilds the post-process graph.
        return {
            "aimed": True,
            "requested_pan_tilt": list(requested),
            "applied_pan_tilt": clamped,
            "clamped": any(abs(a - b) > 1e-6 for a, b in zip(requested, clamped)),
        }
    except Exception as exc:
        return {"aimed": False, "reason": "head_aim_error", "error": repr(exc)}


def _restore_visible_observation_pose(env, robot, event):
    pose = event.get("robot_pose") or {}
    position = pose.get("position")
    orientation = pose.get("orientation_xyzw")
    if len(position or []) != 3 or len(orientation or []) != 4:
        return False
    _teleport_robot_preserving_delta_objects(
        env,
        robot,
        th.tensor(position, dtype=th.float32),
        th.tensor(orientation, dtype=th.float32),
    )
    head_position = pose.get("camera_joint_positions")
    if len(head_position or []) == len(robot.camera_control_idx):
        robot.set_joint_positions(
            th.tensor(head_position, dtype=th.float32),
            indices=robot.camera_control_idx,
            drive=False,
        )
    for _ in range(10):
        robot.keep_still()
        with og.sim.render_on_step(False):
            og.sim.step()
    # SIGSEGV hardening (Ihlen_1 8cf022fc134d / Merom_0 8c1fd6ae00c0): the
    # teleport + head-joint restore above moved the robot-primary camera prim,
    # and the loop ran render-off. Drain the post-process graph with a few
    # render-only ticks before the caller's next real render burst.
    for _ in range(3):
        og.sim.render()
    return True


def _step_control_without_observation(env, action):
    """Apply one real control step without allocating camera observations."""
    env._pre_step(action)
    with og.sim.render_on_step(False):
        og.sim.step()


def _release_motion_generator(controller):
    """Free CuRobo world buffers before allocating keyframe render targets."""
    motion_generator = getattr(controller, "_motion_generator", None)
    if motion_generator is None:
        return [], {}
    skipped = list(getattr(motion_generator, "skipped_singular_collision_meshes", []))
    diagnostics = dict(getattr(controller, "physical_approach_diagnostics", {}))
    controller._motion_generator = None
    del motion_generator
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()
    return skipped, diagnostics


def _route_room_obstacle_predicate(env, robot, target):
    """Keep collision objects in rooms traversed between robot and target."""
    robot_position, _ = robot.get_position_orientation()
    target_position, _ = (
        target.get_position_orientation() if target is not None else robot.get_position_orientation()
    )
    route_rooms = set(getattr(target, "in_rooms", None) or []) if target is not None else set()
    for alpha in np.linspace(0.0, 1.0, 41):
        xy = (1.0 - alpha) * robot_position[:2] + alpha * target_position[:2]
        room = env.scene.seg_map.get_room_instance_by_point(xy)
        if room:
            route_rooms.add(room)
    start = np.asarray(robot_position[:2].cpu(), dtype=float)
    end = np.asarray(target_position[:2].cpu(), dtype=float)
    segment = end - start
    segment_norm_sq = max(float(np.dot(segment, segment)), 1e-9)

    def predicate(obj):
        if obj is target:
            return True
        rooms = set(getattr(obj, "in_rooms", None) or [])
        if rooms:
            return bool(rooms & route_rooms)
        try:
            position, _ = obj.get_position_orientation()
            point = np.asarray(position[:2].cpu(), dtype=float)
            alpha = float(np.clip(np.dot(point - start, segment) / segment_norm_sq, 0.0, 1.0))
            return float(np.linalg.norm(point - (start + alpha * segment))) <= 2.0
        except Exception:
            return False

    return predicate, sorted(route_rooms)


def _physical_look_at(env, controller, obj):
    """Face and track an object using only official physical controls."""
    robot = controller.robot
    robot_position, _ = robot.get_position_orientation()
    target_position, _ = obj.get_position_orientation()
    yaw = math.atan2(
        float(target_position[1] - robot_position[1]),
        float(target_position[0] - robot_position[0]),
    )
    controller._tracking_object = obj
    actions = []
    # A same-XY CuRobo navigation plan can fail in tight spaces even though a
    # pure turn is valid. Drive only the absolute base yaw joint through the
    # official robot action conversion and Starter motion executor.
    for action in controller._rotate_base_to_yaw(yaw):
        _step_control_without_observation(env, action)
        actions.append(_to_numpy(action))

    if robot.model != "tiago":
        return actions

    for _ in range(60):
        action = controller._postprocess_action(controller._empty_action())
        _step_control_without_observation(env, action)
        actions.append(_to_numpy(action))
        goal = controller._get_head_goal_q(obj.get_position_orientation())
        actual = robot.get_joint_positions()[robot.camera_control_idx]
        if float(th.max(th.abs(goal - actual))) < 0.02:
            break
    return actions


def _check_postcondition(primitive, target, carried, controller):
    if primitive == "GRASP":
        actual = controller._get_obj_in_hand()
        return actual is target, {"object_in_hand": getattr(actual, "name", None)}
    if primitive in {"PLACE_ON_TOP", "PLACE_INSIDE"}:
        state = object_states.OnTop if primitive == "PLACE_ON_TOP" else object_states.Inside
        ok = carried is not None and state in carried.states and bool(carried.states[state].get_value(target))
        return ok, {"relation": state.__name__, "subject": getattr(carried, "name", None), "object": target.name}
    if primitive in {"OPEN", "CLOSE"}:
        value = primitive == "OPEN"
        actual = bool(target.states[object_states.Open].get_value()) if object_states.Open in target.states else None
        return actual == value, {"state": "Open", "expected": value, "actual": actual}
    if primitive in {"TOGGLE_ON", "TOGGLE_OFF"}:
        value = primitive == "TOGGLE_ON"
        actual = bool(target.states[object_states.ToggledOn].get_value()) if object_states.ToggledOn in target.states else None
        return actual == value, {"state": "ToggledOn", "expected": value, "actual": actual}
    if primitive == "EXTINGUISH":
        actual = (
            bool(target.states[object_states.OnFire].get_value())
            if object_states.OnFire in target.states
            else None
        )
        return actual is False, {"state": "OnFire", "expected": False, "actual": actual}
    return True, {}


def execute(run, input_path, output_dir, args, env=None, persistent=False):
    plan = compile_expert_plan(run)
    if plan.task_family not in {"retrieval_delivery", "open_close", "appliance", "fire"}:
        raise ExpertPlanError(f"expert v1 does not execute task family {plan.task_family!r}")
    te = run.get("task_environment") or {}
    generation_profile = (te.get("generation") or {}).get("solvability_profile")
    if generation_profile is not None and generation_profile != args.backend:
        raise ExpertPlanError(
            f"generation solvability profile {generation_profile!r} does not match "
            f"expert backend {args.backend!r}"
        )
    generation_profile_verified = generation_profile == args.backend
    symbolic_grasp_targets = {
        step.target_object for step in plan.steps if step.primitive == "GRASP"
    }
    scene = (te.get("base_scene") or {}).get("scene_model")
    if not scene:
        raise ExpertPlanError("base scene model is missing")
    generation_robot = str((run.get("robot") or {}).get("model") or "")
    if not generation_robot:
        raise ExpertPlanError("generation robot model is missing")
    if generation_robot.casefold() != str(args.robot).casefold():
        raise ExpertPlanError(
            f"expert robot {args.robot!r} does not match generation robot "
            f"{generation_robot!r}"
        )

    print(
        f"[expert] loading scene={scene} robot={args.robot} backend={args.backend} "
        f"task={plan.task_name} steps={len(plan.steps)}",
        flush=True,
    )
    robot_pose = te.get("robot", {}).get("pose") or run.get("robot", {}).get("pose") or {}
    saved_robot_stability = _saved_robot_pose_stability(robot_pose)
    if not saved_robot_stability["ok"]:
        raise ExpertPlanError(
            f"saved robot pose is not a stable upright pose: {saved_robot_stability}"
        )
    # Preload generated delta objects during initial environment construction
    # for both backends so kinematic task supports are anchored before the
    # physics/contact views initialize (dynamic spawning fell through floors).
    if env is None:
        added_object_configs = _physical_added_object_configs(run, backend=args.backend)
        env = _create_expert_env(
            scene,
            args.robot,
            args.backend,
            robot_pose=robot_pose,
            added_objects=added_object_configs,
            camera_resolution=(args.view_width, args.view_height),
        )
        print("[expert] environment loaded", flush=True)
    else:
        if not persistent:
            raise RuntimeError("an existing expert environment requires persistent=True")
        print(f"[expert-persistent] resetting scene={scene}", flush=True)
        prepare_persistent_scene_reset(env)
        env.scene.reset(hard=True)
        preloaded_names = getattr(args, "preloaded_delta_names", None)
        if preloaded_names is None:
            _spawn_added_objects(env, run)
            print("[expert-persistent] delta objects loaded", flush=True)
        else:
            configure_preloaded_delta_objects(env, run, preloaded_names)
    if args.backend == "oracle_symbolic":
        _hold_symbolic_grasp_targets(env, symbolic_grasp_targets)
    _sink_diag_trace(env, "post_load")
    if args.backend == "oracle_symbolic":
        _set_robot_pose(env.robots[0], robot_pose, env=env, diag_env=env)
    # Defect #19 fix: settle natives under the render regime (capture windows)
    # before the baseline; step-only settling under-samples simulation time
    # and leaves USD-default fixtures mid-relaxation.
    _warm_natives_to_rest(env)
    objects = _scene_objects(env)
    missing = sorted(set(plan.object_ids) - set(objects))
    if missing:
        raise RuntimeError(f"plan objects missing after replay: {missing}")
    reset_aabb_tops = _reset_delta_objects_to_saved_poses(env, run)
    _sink_diag_trace(env, "post_delta_reset")
    replay_integrity = _delta_replay_integrity(
        env,
        run,
        args.max_native_displacement,
        settle_steps=0 if args.backend == "oracle_symbolic" else 20,
        reset_aabb_tops=reset_aabb_tops,
    )
    initial_states = _apply_saved_initial_states(env, run)
    print("[expert] delta objects replayed", flush=True)
    initial_robot_stability = validate_robot_stability(env)
    accepted = replay_integrity["ok"] and initial_robot_stability["ok"]
    rejection = (
        None
        if accepted
        else {
            "stage": (
                "delta_replay_integrity"
                if not replay_integrity["ok"]
                else "initial_robot_stability"
            ),
            "detail": (
                replay_integrity if not replay_integrity["ok"] else initial_robot_stability
            ),
        }
    )
    camera_streams = None
    if not accepted:
        # Reject before initializing segmentation streams; otherwise each bad
        # sample wastes a render setup and produces misleading images.
        print(
            "[expert] initial replay or robot stability failed; skipping camera init and plan: "
            f"{json.dumps(rejection, ensure_ascii=False)}",
            flush=True,
        )
    else:
        camera_streams = _initialize_segmentation_streams(
            env, run, camera_resolution=(args.view_width, args.view_height)
        )
        print("[expert] segmentation streams initialized", flush=True)
        # Sensor initialization rebuilds Replicator / PhysX views and can wake
        # an otherwise sleeping native fixture. Let that one-time setup effect
        # converge before defining the task-time scene-integrity baseline.
        _warm_natives_to_rest(env)
        if args.backend == "oracle_symbolic":
            reset_aabb_tops = _reset_delta_objects_to_saved_poses(env, run)
            replay_integrity = _delta_replay_integrity(
                env,
                run,
                args.max_native_displacement,
                settle_steps=0,
                reset_aabb_tops=reset_aabb_tops,
            )
            if not replay_integrity["ok"]:
                accepted = False
                rejection = {
                    "stage": "post_camera_delta_replay_integrity",
                    "detail": replay_integrity,
                }
    baseline = _native_pose_snapshot(env)
    _nav_diag_checkpoint(env, env.robots[0], "post_baseline")
    task_object_baseline = {
        object_id: np.asarray(objects[object_id].get_position_orientation()[0], dtype=float)
        for object_id in plan.object_ids
    }
    # Fix P: reference sampler-input geometry, captured pre-grasp (before any
    # FixedJoint/stop-play can transiently corrupt get_base_aligned_bbox).
    # Fix P5: for every object a plan step may place on, also snapshot the
    # pre-grasp support-column raycast map (independent healthy-geometry
    # source for the Fix P4 support-contact probe).
    support_surface_ids = set(plan.object_ids)
    for step in getattr(plan, "steps", []) or []:
        if getattr(step, "target_object", None):
            support_surface_ids.add(step.target_object)
        support_surface_ids.update(getattr(step, "useful_objects", ()) or ())
    reference_scene_bboxes = _capture_reference_scene_bboxes(
        env, support_surface_ids=support_surface_ids
    )
    intentionally_moved_delta_ids = {
        object_id
        for step in plan.steps
        for object_id in (step.carried_object,)
        if object_id
    }
    intentionally_moved_delta_ids.update(
        step.target_object
        for step in plan.steps
        if step.primitive == "GRASP" and step.target_object
    )
    stationary_delta_ids = {
        _name(record)
        for record in ((run.get("task_environment") or {}).get("added_objects") or [])
        if "task_object" not in set(record.get("semantic_roles") or [])
        and _name(record) not in intentionally_moved_delta_ids
    }
    stationary_delta_baseline = {
        object_id: _integrity_pose_record(objects[object_id])
        for object_id in stationary_delta_ids
        if object_id in objects
    }

    controller = None
    physical_controller_key = None
    primitive_map = SYMBOLIC_PRIMITIVE_MAP if args.backend == "oracle_symbolic" else PHYSICAL_PRIMITIVE_MAP
    if accepted and args.backend == "oracle_symbolic":
        controller = DeltaSGOraclePrimitives(env, env.robots[0])
    records = []
    target_ids = set(plan.object_ids)
    events = []
    last_post = None
    skipped_collision_meshes = []
    approach_diagnostics = {}
    assisted_interactions = []
    saved_robot_approaches = _saved_robot_approaches(run)
    if controller is not None:
        controller._deltasg_saved_robot_approaches = saved_robot_approaches

    for step_index, step in enumerate(plan.steps if accepted else []):
        print(
            f"[expert] step={step.step_id}/{len(plan.steps)} primitive={step.primitive} "
            f"target={step.target_object}",
            flush=True,
        )
        target = objects.get(step.target_object) if step.target_object else None
        carried = objects.get(step.carried_object) if step.carried_object else None
        height_gate = _manipulation_height_gate(env, step, target, args)
        # No simulator step occurs between one primitive's postcondition and
        # the next primitive. Reuse that exact observation instead of asking
        # Replicator to render the same state twice; repeated back-to-back
        # segmentation renders are unstable in Isaac Sim 5.1.
        if last_post is None:
            # Before NAVIGATE_TO the target may intentionally be visible only
            # from a global camera. Aim the robot head only for a fine operation;
            # navigation itself moves and faces the robot before its post frame.
            if target is not None and step.primitive in MANIPULATION_PRIMITIVES:
                _aim_tiago_head(env.robots[0], target)
            pre = _capture_event(
                env, run, output_dir, f"step_{step.step_id:03d}_pre", target_ids,
                args.min_bbox_pixels, step.target_object, camera_streams
            )
            events.append(pre)
        else:
            pre = last_post
            print(
                f"[expert] step={step.step_id} reusing observation={pre['event_id']}",
                flush=True,
            )
        visibility_errors = validate_visibility_snapshot(
            step,
            pre["robot_visible"],
            pre["global_visible"],
            pre["robot_primary"]["bboxes"],
            args.min_bbox_pixels,
        )
        if args.backend == "physical_control":
            obstacle_predicate, route_rooms = _route_room_obstacle_predicate(
                env, env.robots[0], target
            )
            print(
                f"[expert] step={step.step_id} collision_rooms={route_rooms}",
                flush=True,
            )
            controller_key = (step.target_object, tuple(route_rooms))
            assisted_primitive = step.primitive in {
                "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"
            }
            reuse_controller = (
                assisted_primitive
                and controller is not None
                and controller_key == physical_controller_key
            )
            if not reuse_controller:
                if controller is not None:
                    skipped, diagnostics = _release_motion_generator(controller)
                    skipped_collision_meshes.extend(skipped)
                    approach_diagnostics.update(diagnostics)
                controller = DeltaSGPhysicalPrimitives(
                    env,
                    env.robots[0],
                    enable_head_tracking=args.robot == "Tiago",
                    curobo_batch_size=1,
                    curobo_obstacle_predicate=obstacle_predicate,
                )
                controller._deltasg_saved_robot_approaches = saved_robot_approaches
                controller._deltasg_reference_bboxes = reference_scene_bboxes
                physical_controller_key = controller_key
        recovery = None
        recovery_action_rows = []
        if (
            step.primitive in MANIPULATION_PRIMITIVES or step.primitive == "NAVIGATE_TO"
        ) and visibility_errors and target is not None:
            recovery_trigger = (
                "post_navigation_target_not_visible"
                if step_index > 0 and plan.steps[step_index - 1].primitive == "NAVIGATE_TO"
                else "fine_manipulation_target_not_visible"
            )
            if args.backend == "physical_control":
                try:
                    recovery_actions = _physical_look_at(env, controller, target)
                    recovery_action_rows = list(recovery_actions)
                    recovery_dir = output_dir / "actions"
                    recovery_dir.mkdir(parents=True, exist_ok=True)
                    recovery_path = recovery_dir / f"step_{step.step_id:03d}_look_at.npy"
                    np.save(
                        recovery_path,
                        np.stack(recovery_actions)
                        if recovery_actions
                        else np.empty((0, env.robots[0].action_dim)),
                    )
                    recovery = {
                        "type": "physical_base_and_head_look_at",
                        "trigger": recovery_trigger,
                        "attempted": True,
                        "actions_executed": len(recovery_actions),
                        "actions_path": str(recovery_path),
                    }
                except Exception as exc:
                    recovery = {
                        "type": "physical_base_and_head_look_at",
                        "trigger": recovery_trigger,
                        "attempted": True,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                recovered = _capture_event(
                    env, run, output_dir, f"step_{step.step_id:03d}_look_at", target_ids,
                    args.min_bbox_pixels, step.target_object, camera_streams
                )
                events.append(recovered)
                visibility_errors = validate_visibility_snapshot(
                    step,
                    recovered["robot_visible"],
                    recovered["global_visible"],
                    recovered["robot_primary"]["bboxes"],
                    args.min_bbox_pixels,
                )
                recovery["succeeded"] = not visibility_errors
            else:
                head_aim = _aim_tiago_head(env.robots[0], target)
                recovered = _capture_event(
                    env,
                    run,
                    output_dir,
                    f"step_{step.step_id:03d}_pre_head_aim",
                    target_ids,
                    args.min_bbox_pixels,
                    step.target_object,
                    camera_streams,
                )
                events.append(recovered)
                visibility_errors = validate_visibility_snapshot(
                    step,
                    recovered["robot_visible"],
                    recovered["global_visible"],
                    recovered["robot_primary"]["bboxes"],
                    args.min_bbox_pixels,
                )
                pre = recovered
                recovery = {
                    "type": "oracle_head_only_look_at",
                    "trigger": recovery_trigger,
                    "attempted": True,
                    "head_aim": head_aim,
                    "base_motion_commanded": False,
                    "succeeded": not visibility_errors,
                }
        record = {
            "step": step.to_dict(),
            "pre_observation": pre["event_id"],
            "visibility_errors": visibility_errors,
            "manipulation_height": height_gate,
            "recovery": recovery,
            "actions_executed": len(recovery_action_rows),
            "robot_stability": pre.get("robot_stability"),
        }
        manipulation_approach = None
        if step.primitive in MANIPULATION_PRIMITIVES and target is not None:
            approach_distance = _horizontal_target_aabb_distance(env.robots[0], target)
            manipulation_approach = {
                "distance_reference": "aabb_edge",
                "horizontal_distance": approach_distance,
                "max_horizontal_distance": DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
                "eligible": approach_distance <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE,
            }
            record["manipulation_approach"] = manipulation_approach
        if not (pre.get("robot_stability") or {}).get("ok", False):
            accepted = False
            rejection = {
                "step_id": step.step_id,
                "stage": "robot_stability",
                "detail": pre.get("robot_stability"),
            }
            records.append(record)
            break
        if height_gate is not None and not height_gate["eligible"]:
            accepted = False
            rejection = {
                "step_id": step.step_id,
                "stage": "manipulation_height",
                "detail": height_gate,
            }
            records.append(record)
            break
        if visibility_errors:
            accepted = False
            rejection = {"step_id": step.step_id, "stage": "visibility", "errors": visibility_errors}
            records.append(record)
            break
        if manipulation_approach is not None and not manipulation_approach["eligible"]:
            accepted = False
            rejection = {
                "step_id": step.step_id,
                "stage": "manipulation_approach",
                "detail": manipulation_approach,
            }
            records.append(record)
            break
        # A physical visibility recovery changes the robot state before the
        # primitive. Keep those controls at the front of the exported step
        # trace so replay and VLA supervision see the complete trajectory.
        action_rows = list(recovery_action_rows)
        try:
            _nav_diag_checkpoint(env, env.robots[0], f"step{step.step_id}_pre:{step.primitive}")
            if step.primitive == "WAIT":
                for _ in range(args.wait_steps):
                    action = th.zeros(env.robots[0].action_dim, dtype=th.float32)
                    _step_control_without_observation(env, action)
                    action_rows.append(_to_numpy(action))
                    record["actions_executed"] += 1
            elif (
                args.backend == "physical_control"
                and step.primitive in {"OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}
            ):
                state_type = (
                    object_states.Open
                    if step.primitive in {"OPEN", "CLOSE"}
                    else object_states.ToggledOn
                )
                requested_value = step.primitive in {"OPEN", "TOGGLE_ON"}
                state = target.states.get(state_type) if target is not None else None
                if state is None:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                        f"Target has no {state_type.__name__} state",
                        {"target object": getattr(target, "name", None)},
                    )
                for _ in range(5):
                    action = controller._postprocess_action(controller._empty_action())
                    _step_control_without_observation(env, action)
                    action_rows.append(_to_numpy(action))
                    record["actions_executed"] += 1
                changed = bool(state.set_value(requested_value))
                actual_value = bool(state.get_value())
                interaction = {
                    "step_id": step.step_id,
                    "primitive": step.primitive,
                    "target_object": target.name,
                    "requested_value": requested_value,
                    "actual_value": actual_value,
                    "state_set_succeeded": changed,
                    "mode": "omnigibson_assisted_state_transition",
                }
                record["assisted_interaction"] = interaction
                assisted_interactions.append(interaction)
                if not changed or actual_value != requested_value:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                        "Assisted state transition did not reach the requested state",
                        interaction,
                    )
            elif step.primitive == "EXTINGUISH":
                state = target.states.get(object_states.OnFire) if target is not None else None
                if state is None:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                        "Target has no OnFire state",
                        {"target object": getattr(target, "name", None)},
                    )
                changed = bool(state.set_value(False))
                actual_value = bool(state.get_value())
                interaction = {
                    "step_id": step.step_id,
                    "primitive": step.primitive,
                    "target_object": target.name,
                    "requested_value": False,
                    "actual_value": actual_value,
                    "state_set_succeeded": changed,
                    "mode": "omnigibson_official_on_fire_transition",
                }
                record["assisted_interaction"] = interaction
                assisted_interactions.append(interaction)
                if not changed or actual_value:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                        "Official OnFire transition did not extinguish the target",
                        interaction,
                    )
            else:
                if args.backend == "oracle_symbolic" and step.primitive == "GRASP":
                    for link in target.links.values():
                        link.enable_collisions()
                    target.enable_gravity()
                    target.keep_still()
                for action in controller.apply_ref(primitive_map[step.primitive], target, attempts=args.primitive_attempts):
                    _step_control_without_observation(env, action)
                    action_rows.append(_to_numpy(action))
                    record["actions_executed"] += 1
                    if record["actions_executed"] % 25 == 0:
                        _nav_diag_checkpoint(
                            env, env.robots[0], f"step{step.step_id}_action{record['actions_executed']}"
                        )
                    if args.sample_every > 0 and record["actions_executed"] % args.sample_every == 0:
                        event = _capture_event(
                            env,
                            run,
                            output_dir,
                            f"step_{step.step_id:03d}_action_{record['actions_executed']:05d}",
                            target_ids,
                            args.min_bbox_pixels,
                            step.target_object,
                            camera_streams,
                        )
                        events.append(event)
            _nav_diag_checkpoint(env, env.robots[0], f"step{step.step_id}_post:{step.primitive}")
            if step.primitive == "NAVIGATE_TO":
                prerequisites = list(
                    getattr(controller, "last_navigation_prerequisites", [])
                )
                record["navigation_prerequisites"] = prerequisites
                for prerequisite in prerequisites:
                    door_id = prerequisite["object_id"]
                    target_ids.add(door_id)
                    if door_id in objects:
                        # Opening this articulated route door is intentional;
                        # keep integrity checks anchored to its verified open pose.
                        baseline[door_id] = _integrity_pose_record(objects[door_id])
            post_ok, postcondition = _check_postcondition(step.primitive, target, carried, controller)
            if step.primitive == "NAVIGATE_TO" and target is not None:
                navigation_approach_distance = _horizontal_target_aabb_distance(
                    env.robots[0], target
                )
                postcondition["horizontal_target_distance"] = navigation_approach_distance
                postcondition["max_horizontal_target_distance"] = (
                    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                postcondition["distance_reference"] = "aabb_edge"
                post_ok = (
                    post_ok
                    and navigation_approach_distance
                    <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                current_target_position = np.asarray(
                    target.get_position_orientation()[0], dtype=float
                )
                target_displacement = float(
                    np.linalg.norm(current_target_position - task_object_baseline[target.name])
                )
                postcondition["target_displacement"] = target_displacement
                postcondition["max_target_displacement"] = args.max_task_object_displacement
                post_ok = post_ok and target_displacement <= args.max_task_object_displacement
            if step.primitive in {"PLACE_ON_TOP", "PLACE_INSIDE"} and carried is not None:
                placed_object_distance = _horizontal_target_aabb_distance(
                    env.robots[0], carried
                )
                postcondition["placed_object_horizontal_distance"] = placed_object_distance
                postcondition["max_placed_object_horizontal_distance"] = (
                    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                postcondition["distance_reference"] = "aabb_edge"
                post_ok = (
                    post_ok
                    and placed_object_distance
                    <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
            record["postcondition"] = postcondition
            record["postcondition_ok"] = post_ok
            if args.backend == "physical_control":
                skipped, diagnostics = _release_motion_generator(controller)
                skipped_collision_meshes.extend(skipped)
                approach_diagnostics.update(diagnostics)
            post_head_aim = _aim_tiago_head(env.robots[0], target) if target is not None else None
            post = _capture_event(
                env, run, output_dir, f"step_{step.step_id:03d}_post", target_ids,
                args.min_bbox_pixels, step.target_object, camera_streams
            )
            events.append(post)
            last_post = post
            record["post_observation"] = post["event_id"]
            record["post_head_aim"] = post_head_aim
            post_visibility_step = None
            # A successfully grasped object is now in inventory. Per the
            # dataset visibility contract, inventory objects are not required
            # to remain framed in the post-grasp camera observation.
            if step.primitive in MANIPULATION_PRIMITIVES and step.primitive != "GRASP":
                post_visibility_step = step
            elif step.primitive == "NAVIGATE_TO" and step_index + 1 < len(plan.steps):
                next_step = plan.steps[step_index + 1]
                if (
                    next_step.primitive in MANIPULATION_PRIMITIVES
                    and next_step.target_object == step.target_object
                ):
                    post_visibility_step = next_step
            post_visibility_errors = (
                validate_visibility_snapshot(
                    post_visibility_step,
                    post["robot_visible"],
                    post["global_visible"],
                    post["robot_primary"]["bboxes"],
                    args.min_bbox_pixels,
                )
                if post_visibility_step is not None
                else []
            )
            navigation_visibility_recovery = []
            if (
                args.backend == "oracle_symbolic"
                and step.primitive == "NAVIGATE_TO"
                and target is not None
                and post_visibility_errors
            ):
                for fallback_rank in (1, 2):
                    controller._deltasg_navigation_fallback_rank = fallback_rank
                    try:
                        for _ in controller.apply_ref(
                            primitive_map["NAVIGATE_TO"], target, attempts=1
                        ):
                            pass
                        head_aim = _aim_tiago_head(env.robots[0], target)
                        recovered_post = _capture_event(
                            env,
                            run,
                            output_dir,
                            f"step_{step.step_id:03d}_post_nav_retry_{fallback_rank}",
                            target_ids,
                            args.min_bbox_pixels,
                            step.target_object,
                            camera_streams,
                        )
                        events.append(recovered_post)
                        retry_errors = validate_visibility_snapshot(
                            post_visibility_step,
                            recovered_post["robot_visible"],
                            recovered_post["global_visible"],
                            recovered_post["robot_primary"]["bboxes"],
                            args.min_bbox_pixels,
                        )
                        navigation_visibility_recovery.append(
                            {
                                "fallback_rank": fallback_rank,
                                "head_aim": head_aim,
                                "observation": recovered_post["event_id"],
                                "errors": retry_errors,
                            }
                        )
                        post = recovered_post
                        last_post = recovered_post
                        post_visibility_errors = retry_errors
                        if not retry_errors:
                            break
                    except Exception as exc:
                        navigation_visibility_recovery.append(
                            {"fallback_rank": fallback_rank, "error": repr(exc)}
                        )
                controller._deltasg_navigation_fallback_rank = 0
            record["navigation_visibility_recovery"] = navigation_visibility_recovery
            record["post_observation"] = post["event_id"]
            record["post_visibility_errors"] = post_visibility_errors
            record["post_robot_stability"] = post.get("robot_stability")
            _nav_diag_checkpoint(env, env.robots[0], f"step{step.step_id}_post_capture:{step.primitive}")
            # Capturing the official RGB / segmentation observation advances
            # simulation. A symbolic relation can be true immediately after
            # set_value() and then fall apart before the recorded post frame.
            # The visual supervision is valid only if the same official
            # postcondition still holds in the state represented by that frame.
            stable_post_ok, stable_postcondition = _check_postcondition(
                step.primitive, target, carried, controller
            )
            if step.primitive == "NAVIGATE_TO" and target is not None:
                stable_navigation_approach_distance = _horizontal_target_aabb_distance(
                    env.robots[0], target
                )
                stable_postcondition["horizontal_target_distance"] = (
                    stable_navigation_approach_distance
                )
                stable_postcondition["max_horizontal_target_distance"] = (
                    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                stable_postcondition["distance_reference"] = "aabb_edge"
                stable_post_ok = (
                    stable_post_ok
                    and stable_navigation_approach_distance
                    <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                stable_target_position = np.asarray(
                    target.get_position_orientation()[0], dtype=float
                )
                stable_target_displacement = float(
                    np.linalg.norm(
                        stable_target_position - task_object_baseline[target.name]
                    )
                )
                stable_postcondition["target_displacement"] = stable_target_displacement
                stable_postcondition["max_target_displacement"] = (
                    args.max_task_object_displacement
                )
                stable_post_ok = (
                    stable_post_ok
                    and stable_target_displacement <= args.max_task_object_displacement
                )
            if step.primitive in {"PLACE_ON_TOP", "PLACE_INSIDE"} and carried is not None:
                stable_placed_object_distance = _horizontal_target_aabb_distance(
                    env.robots[0], carried
                )
                stable_postcondition["placed_object_horizontal_distance"] = (
                    stable_placed_object_distance
                )
                stable_postcondition["max_placed_object_horizontal_distance"] = (
                    DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
                stable_postcondition["distance_reference"] = "aabb_edge"
                stable_post_ok = (
                    stable_post_ok
                    and stable_placed_object_distance
                    <= DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE
                )
            record["postcondition_after_capture"] = stable_postcondition
            post_ok = post_ok and stable_post_ok
            record["postcondition_ok"] = post_ok
            step_integrity = _combined_scene_integrity(
                env, baseline, stationary_delta_baseline, args.max_native_displacement
            )
            record["scene_integrity"] = step_integrity
            if not (post.get("robot_stability") or {}).get("ok", False):
                accepted = False
                rejection = {
                    "step_id": step.step_id,
                    "stage": "post_robot_stability",
                    "detail": post.get("robot_stability"),
                }
            elif post_visibility_errors:
                accepted = False
                rejection = {
                    "step_id": step.step_id,
                    "stage": "post_visibility",
                    "errors": post_visibility_errors,
                }
            elif not post_ok:
                accepted = False
                rejection = {"step_id": step.step_id, "stage": "postcondition", "detail": postcondition}
            elif not step_integrity["ok"]:
                accepted = False
                rejection = {
                    "step_id": step.step_id,
                    "stage": "scene_integrity",
                    "detail": step_integrity,
                }
        except Exception as exc:
            accepted = False
            record["execution_error"] = repr(exc)
            record["traceback"] = traceback.format_exc()
            rejection = {"step_id": step.step_id, "stage": "execution", "error": repr(exc)}
        finally:
            if args.backend == "physical_control" and getattr(controller, "_motion_generator", None) is not None:
                skipped, diagnostics = _release_motion_generator(controller)
                skipped_collision_meshes.extend(skipped)
                approach_diagnostics.update(diagnostics)
            actions_dir = output_dir / "actions"
            actions_dir.mkdir(parents=True, exist_ok=True)
            action_path = actions_dir / f"step_{step.step_id:03d}.npy"
            np.save(
                action_path,
                np.stack(action_rows) if action_rows else np.empty((0, env.robots[0].action_dim)),
            )
            record["actions_path"] = str(action_path)
            cleanup_errors = getattr(controller, "last_cleanup_errors", None)
            if cleanup_errors:
                record["cleanup_errors"] = cleanup_errors
        records.append(record)
        print(
            f"[expert] step={step.step_id} accepted={accepted} "
            f"actions={record.get('actions_executed', 0)}",
            flush=True,
        )
        if not accepted:
            break

    _nav_diag_checkpoint(env, env.robots[0], "final_integrity")
    integrity = _combined_scene_integrity(
        env, baseline, stationary_delta_baseline, args.max_native_displacement
    )
    final_robot_stability = validate_robot_stability(env)
    if not final_robot_stability["ok"]:
        accepted = False
        rejection = rejection or {
            "stage": "final_robot_stability",
            "detail": final_robot_stability,
        }
    if not integrity["ok"]:
        accepted = False
        rejection = rejection or {"stage": "scene_integrity", "detail": integrity}
    skipped_collision_meshes = sorted(set(skipped_collision_meshes))
    used_assisted_interaction = bool(assisted_interactions)
    complete_action_trace = all(
        not (
            (record.get("recovery") or {}).get("type")
            == "physical_base_and_head_look_at"
        )
        or record.get("actions_executed", 0)
        >= (record.get("recovery") or {}).get("actions_executed", 0)
        for record in records
    )
    physical_action_count = sum(
        int(record.get("actions_executed") or 0) for record in records
    )
    physical_nonzero_action_count = 0
    if args.backend == "physical_control":
        for record in records:
            action_path = Path(str(record.get("actions_path") or ""))
            if not action_path.is_file():
                continue
            actions = np.load(action_path, allow_pickle=False)
            if actions.ndim == 2 and len(actions):
                physical_nonzero_action_count += int(
                    np.count_nonzero(np.any(np.abs(actions) > 1e-6, axis=1))
                )
    physical_trajectory_available = (
        accepted
        and args.backend == "physical_control"
        and generation_profile_verified
        and complete_action_trace
        and physical_action_count > 0
        and physical_nonzero_action_count > 0
    )
    supervision_level = (
        "hybrid_interaction"
        if physical_trajectory_available and used_assisted_interaction
        else "strict_physical"
        if physical_trajectory_available
        else "oracle_symbolic"
        if args.backend == "oracle_symbolic"
        else "rejected_physical"
    )
    result = {
        "schema_version": "deltasg_expert_result.v1",
        "accepted": accepted,
        "qa_eligible": accepted,
        "input": str(input_path),
        "run_id": run.get("run_id"),
        "scene": scene,
        "robot": args.robot,
        "task_name": plan.task_name,
        "task_family": plan.task_family,
        "llm_model": args.llm_model,
        "backend": {
            "name": args.backend,
            "generation_solvability_profile": generation_profile,
            "generation_profile_verified": generation_profile_verified,
            "physical_solubility_validation": (
                accepted
                and args.backend == "physical_control"
                and generation_profile_verified
                and not used_assisted_interaction
                and complete_action_trace
            ),
            "low_level_vla_actions_eligible": (
                accepted
                and args.backend == "physical_control"
                and generation_profile_verified
                and not used_assisted_interaction
                and complete_action_trace
            ),
            "assisted_interaction": used_assisted_interaction,
            "complete_action_trace": complete_action_trace,
            "physical_trajectory_available": physical_trajectory_available,
            "physical_action_count": physical_action_count,
            "physical_nonzero_action_count": physical_nonzero_action_count,
            "supervision_level": supervision_level,
            "reason": (
                "Open/toggle state was completed through an audited OmniGibson assisted transition"
                if used_assisted_interaction
                else "official StarterSemanticActionPrimitives emitted the saved actions"
                if args.backend == "physical_control"
                else "OmniGibson symbolic primitives may teleport during navigation/manipulation"
            ),
        },
        "compiled_plan": plan.to_dict(),
        "steps": records,
        "observation_events": events,
        "scene_integrity": integrity,
        "robot_stability": {
            "saved": saved_robot_stability,
            "initial": initial_robot_stability,
            "final": final_robot_stability,
        },
        "physical_diagnostics": {
            "skipped_singular_collision_meshes": skipped_collision_meshes,
            "approach_candidates": approach_diagnostics,
            "assisted_interactions": assisted_interactions,
        },
        "replayed_initial_states": initial_states,
        "delta_replay_integrity": replay_integrity,
        "rejection": rejection,
    }
    if persistent:
        cleanup_persistent_camera_streams(camera_streams)
    return env, result


def main():
    parser = argparse.ArgumentParser(description="Execute and validate one DeltaSG expert plan")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--robot",
        default=None,
        help="Expert robot. Defaults to the robot recorded by generation.",
    )
    parser.add_argument("--backend", choices=["oracle_symbolic", "physical_control"], default="physical_control")
    parser.add_argument("--llm-model", default="qwen3.8-max")
    parser.add_argument("--primitive-attempts", type=int, default=2)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=0, help="Also sample every N yielded actions; 0=keyframes only")
    parser.add_argument("--min-bbox-pixels", type=int, default=8)
    parser.add_argument("--view-width", type=int, default=640)
    parser.add_argument("--view-height", type=int, default=480)
    parser.add_argument("--max-native-displacement", type=float, default=0.05)
    parser.add_argument("--max-task-object-displacement", type=float, default=0.05)
    parser.add_argument(
        "--min-manipulation-height",
        type=float,
        default=DEFAULT_MIN_MANIPULATION_HEIGHT,
        help="Minimum fine-operation point height above the supporting floor in metres.",
    )
    parser.add_argument(
        "--max-manipulation-height",
        type=float,
        default=DEFAULT_MAX_MANIPULATION_HEIGHT,
        help="Maximum fine-operation point height above the supporting floor in metres.",
    )
    args = parser.parse_args()
    if args.min_manipulation_height < 0 or args.max_manipulation_height <= args.min_manipulation_height:
        parser.error("manipulation height bounds must satisfy 0 <= min < max")

    input_path = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "expert_result.json"
    env = None
    try:
        run = json.loads(input_path.read_text(encoding="utf-8"))
        if args.robot is None:
            generated_robot = str((run.get("robot") or {}).get("model") or "")
            args.robot = {
                "tiago": "Tiago",
                "r1": "R1",
                "fetch": "fetch",
            }.get(generated_robot.casefold(), generated_robot)
        env, result = execute(run, input_path, output_dir, args)
    except Exception as exc:
        result = {
            "schema_version": "deltasg_expert_result.v1",
            "accepted": False,
            "qa_eligible": False,
            "input": str(input_path),
            "llm_model": args.llm_model,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps({"accepted": result.get("accepted"), "result": str(result_path), "error": result.get("error")}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Replicator teardown is unstable after segmentation annotators.  This is a
    # dedicated one-sample process, so preserve the result and exit directly.
    os._exit(0 if result.get("accepted") else 2)


if __name__ == "__main__":
    main()
