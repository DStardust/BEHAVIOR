"""
Build a layer-2 3DSG-like topology from layer-1 exported OmniGibson outputs.

This module intentionally depends only on the Python standard library so it can
run on machines without an OmniGibson / Isaac Sim environment. It consumes the
`scene_all_objects.json` files produced by `code/api.py`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


SUPPORT_SURFACE_TOKENS = {
    "bar",
    "bed",
    "bench",
    "cabinet",
    "cart",
    "chair",
    "counter",
    "countertop",
    "desk",
    "dresser",
    "island",
    "mat",
    "nightstand",
    "ottoman",
    "plate",
    "rack",
    "shelf",
    "sofa",
    "stool",
    "table",
    "tray",
}
INSIDE_RECEPTACLE_TOKENS = {
    "basket",
    "bin",
    "bowl",
    "box",
    "cabinet",
    "can",
    "container",
    "cup",
    "drawer",
    "fridge",
    "jar",
    "mug",
    "oven",
    "pot",
    "sink",
    "washer",
}
STRUCTURAL_TOKENS = {"ceilings", "ceiling", "floor", "floors", "room", "wall", "walls"}
CONTROLLABLE_STATE_NAMES = {"ToggledOn", "HeatSourceOrSink"}
CONTROLLABLE_CATEGORY_TOKENS = {
    "dishwasher",
    "fridge",
    "lamp",
    "laptop",
    "microwave",
    "oven",
    "sink",
    "speaker",
    "switch",
    "tv",
}
ARTICULABLE_STATE_NAMES = {"Open"}
ARTICULABLE_CATEGORY_TOKENS = {
    "cabinet",
    "drawer",
    "door",
    "fridge",
    "dishwasher",
    "microwave",
    "oven",
    "toilet",
    "window",
}
ABNORMAL_STATE_NAMES = {"OnFire": "on_fire", "Burnt": "burnt", "Covered": "covered"}


def load_scene_objects(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "objects" not in data:
        raise ValueError(f"{path} does not look like a scene_all_objects.json file")
    return data


def build_graph_from_scene_objects(data, near_distance=1.25, support_z_tolerance=0.08):
    objects = data["objects"]
    nodes = []
    edges = []

    rooms = sorted(
        {
            room
            for obj in objects
            for room in _rooms_for(obj)
            if room
        }
    )
    for room in rooms:
        nodes.append(
            {
                "id": f"room::{room}",
                "type": "room",
                "name": room,
                "category": "room",
                "semantic": {"interaction": {"kind": "none", "confidence": "explicit"}},
            }
        )

    object_by_name = {}
    for obj in objects:
        node = {
            "id": obj["name"],
            "type": "object",
            "name": obj["name"],
            "category": obj.get("category"),
            "prim_path": obj.get("prim_path"),
            "pose": {
                "position": obj.get("position"),
                "orientation_xyzw": obj.get("orientation_xyzw"),
            },
            "bbox": {
                "min": obj.get("aabb_min"),
                "max": obj.get("aabb_max"),
                "extent": obj.get("aabb_extent"),
            },
            "rooms": _rooms_for(obj),
            "available_states": obj.get("available_states", []),
            "semantic": infer_semantic_from_level1_object(obj),
        }
        nodes.append(node)
        object_by_name[obj["name"]] = node

        for room in node["rooms"]:
            if room:
                edges.append({"source": f"room::{room}", "target": node["id"], "relation": "contains"})

    room_groups = defaultdict(list)
    for node in object_by_name.values():
        for room in node["rooms"]:
            room_groups[room].append(node)

    for room, room_nodes in room_groups.items():
        for idx, src in enumerate(room_nodes):
            for dst in room_nodes[idx + 1 :]:
                relation = _spatial_relation(src, dst, near_distance, support_z_tolerance)
                if relation is None:
                    continue
                relation["room"] = room
                edges.append({"source": src["id"], "target": dst["id"], **relation})

    room_edges, navigation = _build_room_navigation(rooms, object_by_name.values())
    edges.extend(room_edges)

    return {
        "scene_model": data.get("scene_model"),
        "source": "level1_scene_all_objects",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "num_objects": data.get("num_scene_objects", len(objects)),
            "num_rooms": len(rooms),
            "category_counts": data.get("category_counts", {}),
        },
        "navigation": navigation,
    }


def infer_semantic_from_level1_object(obj):
    category = obj.get("category") or ""
    tokens = _tokens(category)
    states = set(obj.get("available_states", []))
    extent = obj.get("aabb_extent")
    reasons = []

    supports_on_top = bool(tokens & SUPPORT_SURFACE_TOKENS)
    if supports_on_top:
        reasons.append("category_surface_token")
    elif _is_large_horizontal_surface(extent):
        supports_on_top = True
        reasons.append("large_horizontal_bbox")

    supports_inside = bool(tokens & INSIDE_RECEPTACLE_TOKENS)
    if supports_inside:
        reasons.append("category_container_token")
    if states & {"Contains", "Filled"}:
        supports_inside = True
        reasons.append("container_state_available")

    if category == "agent":
        interaction = {"kind": "agent", "confidence": "explicit", "reasons": ["agent_category"]}
    elif states & CONTROLLABLE_STATE_NAMES or tokens & CONTROLLABLE_CATEGORY_TOKENS:
        interaction = {"kind": "controllable", "confidence": "inferred", "reasons": ["state_or_category"]}
    elif (states & ARTICULABLE_STATE_NAMES or tokens & ARTICULABLE_CATEGORY_TOKENS) and not tokens & STRUCTURAL_TOKENS:
        interaction = {"kind": "articulable", "confidence": "inferred", "reasons": ["state_or_category"]}
    elif tokens & STRUCTURAL_TOKENS:
        interaction = {"kind": "none", "confidence": "inferred", "reasons": ["structural_category"]}
    else:
        interaction = {"kind": "manipulable", "confidence": "inferred", "reasons": ["non_structural_object"]}

    potential_abnormal = sorted({name for state, name in ABNORMAL_STATE_NAMES.items() if state in states})
    return {
        "receptacle": {
            "can_support": supports_on_top or supports_inside,
            "supports_on_top": supports_on_top,
            "supports_inside": supports_inside,
            "confidence": "inferred" if reasons else "unknown",
            "reasons": reasons,
        },
        "interaction": interaction,
        "abnormal": {
            "potential": potential_abnormal,
            "current": [],
            "confidence": "state_type_inferred" if potential_abnormal else "none",
        },
    }


def export_graph(scene_objects_path, output_path=None):
    data = load_scene_objects(scene_objects_path)
    graph = build_graph_from_scene_objects(data)
    output_path = Path(output_path) if output_path else Path(scene_objects_path).with_name("level2_3dsg.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return output_path, graph


def _rooms_for(obj):
    room_info = obj.get("room_info") or {}
    rooms = room_info.get("in_rooms") if isinstance(room_info, dict) else []
    if isinstance(rooms, str):
        rooms = [rooms]
    return [room for room in rooms or [] if room]


def _tokens(category):
    return {token for token in category.replace("-", "_").lower().split("_") if token}


def _is_large_horizontal_surface(extent):
    if not extent or len(extent) < 3:
        return False
    x, y, z = [float(v) for v in extent[:3]]
    return x >= 0.35 and y >= 0.35 and z <= max(x, y) * 0.6


def _center(node):
    pos = node["pose"].get("position")
    if pos and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    bbox_min = node["bbox"].get("min")
    bbox_max = node["bbox"].get("max")
    if bbox_min and bbox_max:
        return [(float(lo) + float(hi)) * 0.5 for lo, hi in zip(bbox_min[:3], bbox_max[:3])]
    return None


def _xy_overlap(src, dst):
    src_min, src_max = src["bbox"].get("min"), src["bbox"].get("max")
    dst_min, dst_max = dst["bbox"].get("min"), dst["bbox"].get("max")
    if not src_min or not src_max or not dst_min or not dst_max:
        return False
    return (
        min(float(src_max[0]), float(dst_max[0])) > max(float(src_min[0]), float(dst_min[0]))
        and min(float(src_max[1]), float(dst_max[1])) > max(float(src_min[1]), float(dst_min[1]))
    )


def _spatial_relation(src, dst, near_distance, support_z_tolerance):
    src_center, dst_center = _center(src), _center(dst)
    if src_center is None or dst_center is None:
        return None

    dx = src_center[0] - dst_center[0]
    dy = src_center[1] - dst_center[1]
    xy_dist = math.sqrt(dx * dx + dy * dy)

    src_min, src_max = src["bbox"].get("min"), src["bbox"].get("max")
    dst_min, dst_max = dst["bbox"].get("min"), dst["bbox"].get("max")
    if src_min and src_max and dst_min and dst_max and _xy_overlap(src, dst):
        src_top = float(src_max[2])
        dst_bottom = float(dst_min[2])
        dst_top = float(dst_max[2])
        src_bottom = float(src_min[2])
        if abs(dst_bottom - src_top) <= support_z_tolerance:
            return {"relation": "supports_candidate", "mode": "on_top", "distance": xy_dist}
        if abs(src_bottom - dst_top) <= support_z_tolerance:
            return {"relation": "supported_by_candidate", "mode": "on_top", "distance": xy_dist}

    if xy_dist <= near_distance:
        return {"relation": "near", "distance": xy_dist}
    return None


def _build_room_navigation(rooms, object_nodes):
    room_set = set(rooms)
    room_centers = _estimate_room_centers(rooms, object_nodes)
    adjacency = defaultdict(dict)
    edges = []

    for node in object_nodes:
        category = node.get("category") or ""
        node_rooms = [room for room in node.get("rooms", []) if room in room_set]
        if len(node_rooms) < 2 or "door" not in _tokens(category):
            continue
        for idx, src in enumerate(sorted(set(node_rooms))):
            for dst in sorted(set(node_rooms))[idx + 1 :]:
                distance = _room_distance(room_centers, src, dst)
                _add_room_edge(adjacency, src, dst, distance, "door_connection", node["id"])
                edges.append(
                    {
                        "source": f"room::{src}",
                        "target": f"room::{dst}",
                        "relation": "room_adjacent",
                        "mode": "door_connection",
                        "via_object": node["id"],
                        "distance": distance,
                    }
                )

    for src, dst, distance in _minimum_room_connectors(rooms, room_centers, adjacency):
        _add_room_edge(adjacency, src, dst, distance, "centroid_route_candidate", None)
        edges.append(
            {
                "source": f"room::{src}",
                "target": f"room::{dst}",
                "relation": "room_route_candidate",
                "mode": "centroid_route_candidate",
                "distance": distance,
            }
        )

    navigation = {
        "room_centers": room_centers,
        "room_edges": [
            {"source": src, "target": dst, **meta}
            for src, dsts in sorted(adjacency.items())
            for dst, meta in sorted(dsts.items())
            if src < dst
        ],
        "shortest_room_paths": _all_pairs_room_paths(rooms, adjacency),
    }
    return edges, navigation


def _estimate_room_centers(rooms, object_nodes):
    centers = defaultdict(list)
    for node in object_nodes:
        if _tokens(node.get("category") or "") & STRUCTURAL_TOKENS:
            continue
        center = _center(node)
        if center is None:
            continue
        for room in node.get("rooms", []):
            if room in rooms:
                centers[room].append(center)

    result = {}
    for room in rooms:
        points = centers.get(room, [])
        if not points:
            result[room] = None
            continue
        result[room] = [
            sum(point[axis] for point in points) / len(points)
            for axis in range(3)
        ]
    return result


def _room_distance(room_centers, src, dst):
    src_center = room_centers.get(src)
    dst_center = room_centers.get(dst)
    if src_center is None or dst_center is None:
        return None
    dx = src_center[0] - dst_center[0]
    dy = src_center[1] - dst_center[1]
    return math.sqrt(dx * dx + dy * dy)


def _add_room_edge(adjacency, src, dst, distance, mode, via_object):
    meta = {"distance": distance, "mode": mode}
    if via_object:
        meta["via_object"] = via_object
    adjacency[src][dst] = meta
    adjacency[dst][src] = meta


def _minimum_room_connectors(rooms, room_centers, adjacency):
    rooms = list(rooms)
    if len(rooms) < 2:
        return []

    remaining = set(rooms)
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = [start]
        while queue:
            room = queue.pop()
            for neighbor in adjacency.get(room, {}):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    connectors = []
    while len(components) > 1:
        best = None
        for left_idx, left in enumerate(components):
            for right_idx, right in enumerate(components[left_idx + 1 :], start=left_idx + 1):
                for src in left:
                    for dst in right:
                        distance = _room_distance(room_centers, src, dst)
                        if distance is None:
                            continue
                        candidate = (distance, left_idx, right_idx, src, dst)
                        if best is None or candidate < best:
                            best = candidate
        if best is None:
            break
        distance, left_idx, right_idx, src, dst = best
        connectors.append((src, dst, distance))
        merged = components[left_idx] | components[right_idx]
        components = [
            component
            for idx, component in enumerate(components)
            if idx not in {left_idx, right_idx}
        ]
        components.append(merged)
    return connectors


def _all_pairs_room_paths(rooms, adjacency):
    return {
        src: {
            dst: _shortest_room_path(src, dst, adjacency)
            for dst in rooms
            if dst != src
        }
        for src in rooms
    }


def _shortest_room_path(src, dst, adjacency):
    frontier = [(0.0, src, [])]
    seen = set()
    while frontier:
        frontier.sort(key=lambda item: item[0])
        cost, room, path = frontier.pop(0)
        if room in seen:
            continue
        seen.add(room)
        next_path = path + [room]
        if room == dst:
            return {"rooms": next_path, "distance": cost}
        for neighbor, meta in adjacency.get(room, {}).items():
            if neighbor in seen:
                continue
            step_cost = meta.get("distance")
            frontier.append((cost + (step_cost if step_cost is not None else 1.0), neighbor, next_path))
    return None


def main():
    parser = argparse.ArgumentParser(description="Build layer-2 3DSG JSON from layer-1 scene_all_objects.json.")
    parser.add_argument("scene_objects_json", help="Path to scene_all_objects.json")
    parser.add_argument("--output", default=None, help="Output path. Defaults to level2_3dsg.json next to input.")
    args = parser.parse_args()

    output_path, graph = export_graph(args.scene_objects_json, args.output)
    print(f"saved {output_path} with {graph['num_nodes']} nodes and {graph['num_edges']} edges")


if __name__ == "__main__":
    main()
