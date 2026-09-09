"""Spatial evidence for household tool storage, independent of manipulation reach."""

import math


def rigid_bbox_fits(subject_size, container_size, clearance=0.04):
    """Conservative axis-permutation fit prefilter; official Inside remains required."""
    return all(a + clearance <= b for a, b in zip(sorted(subject_size), sorted(container_size)))


def horizontal_bbox_gap(first, second):
    return math.hypot(*(
        max(float(first[0][axis]) - float(second[1][axis]),
            float(second[0][axis]) - float(first[1][axis]), 0.0)
        for axis in (0, 1)
    ))


def evaluate_tool_layout(tool_bbox, target_bbox, policy, anchor_bbox=None):
    gap = horizontal_bbox_gap(tool_bbox, target_bbox)
    anchor_gap = horizontal_bbox_gap(tool_bbox, anchor_bbox) if anchor_bbox is not None else None
    minimum = policy["min_target_gap"]
    maximum = policy.get("max_anchor_gap")
    return {
        "ok": gap >= minimum and (maximum is None or anchor_gap is not None and anchor_gap <= maximum),
        "policy": policy,
        "target_horizontal_gap": gap,
        "anchor_horizontal_gap": anchor_gap,
        "distance_reference": "horizontal_aabb_edge",
    }
