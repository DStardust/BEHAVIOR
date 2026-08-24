"""DeltaSG expert-plan compilation and execution-independent validation.

This module deliberately has no OmniGibson imports.  It is used both by the
online expert runner and by fast dataset audits / unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SUPPORTED_SOURCE_PRIMITIVES = {"MOVE", "PICK", "PLACE", "INTERACT", "WAIT"}
SUPPORTED_EXPERT_PRIMITIVES = {
    "NAVIGATE_TO",
    "GRASP",
    "PLACE_ON_TOP",
    "PLACE_INSIDE",
    "OPEN",
    "CLOSE",
    "TOGGLE_ON",
    "TOGGLE_OFF",
    "EXTINGUISH",
    "WAIT",
}
MANIPULATION_PRIMITIVES = {
    "GRASP",
    "PLACE_ON_TOP",
    "PLACE_INSIDE",
    "OPEN",
    "CLOSE",
    "TOGGLE_ON",
    "TOGGLE_OFF",
}
FINE_MANIPULATION_PRIMITIVES = {"GRASP", "PLACE_ON_TOP", "PLACE_INSIDE"}
DEFAULT_MIN_MANIPULATION_HEIGHT = 0.10
DEFAULT_MAX_MANIPULATION_HEIGHT = 1.55
DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT = 0.65
DEFAULT_MIN_DIRECT_FLOOR_PRIMARY_VIEW_HEIGHT = 0.18
DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE = 1.15
# Fix N (2026-08-15): access-aware OnTop place contract. attempt-7 / diag27 on
# Beechwood_0 deliver_drink showed the official place sampler accepting a
# far-edge table candidate (~0.82 m from the base, beside an armchair); the
# resulting single CuRobo approach swept the arm into the support region and
# stalled 0.005-0.019 rad short. These gates reject such candidates; they add
# criteria and never relax a collision or reachability standard.
PLACE_ACCESS_REACH_MARGIN = 0.75
PLACE_ACCESS_HOVER_CLEARANCE = 0.25
PLACE_ACCESS_CORRIDOR_MARGIN = 0.02
PLACE_ACCESS_MAX_CANDIDATES = 12
# Fix P (2026-08-15): reference-surface sanity gate for OnTop place sampling.
# attempt-9 on Beechwood_0 deliver_drink drew place candidates in a ring
# 0.14-0.60 m outside the table footprint at inflated z (0.780 vs the clean
# 0.673) even though scene integrity and obj.aabb looked normal: the sampler
# reads target.get_base_aligned_bbox(xy_aligned=True) at draw time (ray start
# points, sampling_utils.py) and that quantity was transiently inflated after
# the GRASP FixedJoint + stop/play rebuild ("Illegal BroadPhaseUpdateData"
# burst class). diag30 verified clean draws land strictly inside the reference
# footprint at z ~= support top + half held height + PREDICATE_SAMPLING_Z_OFFSET.
# The margins below bound legitimate drift (replay-integrity displacement
# budget <= 0.05 m plus the sampler's 2% ray offset) and exclude no legitimate
# candidate; when a draw trips the gate the primitive fails so apply_ref
# rebuilds the physics scene (the verified repair, diag23-24 + diag30) and
# resamples from a clean surface. These gates add criteria; they never relax a
# collision or reachability standard.
PLACE_SAMPLING_SURFACE_XY_MARGIN = 0.05
# Fix P3 (2026-08-15, after adversarial review of Fix P2): the z tolerances
# are derived from the sampler's own ray geometry instead of fixed scalars.
# The OnTop sampler casts rays from the target's base-aligned bbox expanded
# by DEFAULT_AABB_OFFSET_FRACTION=0.02 of the extent per axis
# (sampling_utils.py), so a legitimate hit z lies within
# [bbox_bottom - f*h, bbox_top + f*h] and a legitimate candidate z deviates
# from (reference top + half held height) by at most f*h + Z_OFFSET upward
# and (1+f)*h + Z_OFFSET downward. Fixed P2 scalars were wrong in both
# directions: the 0.04 upward tolerance only suited table-height targets (a
# clean top-placement on a 1.5 m target deviates up to ~0.05), and the 0.15
# downward tolerance falsely rejected legitimate tiered placements (a sofa
# seat sits ~0.35 m below the backrest-top AABB top used for z_expected) —
# which the batch-void rule would then amplify into whole-batch rejection.
# Corruption stays detectable because the observed signatures (+0.09 m on
# attempt 10 / +0.108 m on attempt 9 for the 0.506 m table) far exceed the
# derived bounds (0.030/0.040 m) at every target height. Z_EPSILON absorbs
# raycast/float noise on top of the analytic bounds.
PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION = 0.02
PLACE_SAMPLING_SURFACE_Z_EPSILON = 0.01
# Fix P4 (2026-08-16, attempt 11): support-contact consistency probe for
# ACCEPTED OnTop candidates. Attempt 11 drew a geometrically plausible
# center-table candidate (z_dev -0.0395 vs the reference surface, inside the
# P3 downward tolerance that must stay generous for tiered supports) whose
# implied ray hit sat ~6 cm BELOW the tabletop: the sampling ray passed
# through the table surface near its center and the bottle bottom ended up
# ~4 cm embedded in solid tabletop, which CuRobo could neither plan to
# (attempt 0: "no accessible path") nor reach (attempt 1: articulation
# stall). The P3 reference gate cannot catch this class because a tiered
# support legitimately sits far below its AABB top. Instead, verify each
# accepted candidate against the live surface itself: cast a small downward
# ray fan (center + ring at PLACE_SUPPORT_PROBE_RADIUS_FRACTION of the held
# footprint radius) around the candidate and require
#   * the center ray to hit the target within the probe window
#     (else no_support_within_probe / support_not_target),
#   * the candidate bottom to sit no more than PLACE_SUPPORT_EMBED_TOLERANCE
#     below the LOCAL surface under it — the center ray hit
#     (embedded_below_local_surface: the candidate occupies space taken by
#     support geometry). Review R1: the comparison must use the center hit,
#     not the max over the whole fan — a ring ray landing on a higher part of
#     an uneven tabletop (or on a neighbor object) made h_max exceed the
#     local surface and falsely embedded clean draws (diag30 z=0.673:
#     bottom 0.4965 vs ring hit at the 0.5058 aabb_top -> deficit 0.0093 >
#     0.005), and
#   * the candidate bottom to sit no more than Z_OFFSET +
#     PLACE_SUPPORT_FLOAT_TOLERANCE above the center hit
#     (floating_above_support: the release would drop the object).
# The probe compares live surface hits to live surface hits, so mesh-vs-AABB
# offsets cancel and the embed tolerance can stay tight; the observed -0.0395
# embed exceeds it ~2x while clean draws sit at +Z_OFFSET. The probe starts
# PLACE_SUPPORT_PROBE_MARGIN above the candidate top so a ray over an
# embedded candidate still originates above the true surrounding surface.
# Review R2: bottom_z must use the SAME extent the stock sampler used to
# build the candidate (the live world-aligned held AABB). A tilted held
# object's live half_z exceeds the pre-grasp reference half_z, and mixing
# the two inflates float_gap by the delta (a 45-degree tray: 56 mm > the
# tolerance -> false reject). Corrupted live extents stay caught upstream by
# the Fix P3 z gate, which compares the candidate z itself against the
# reference surface.
PLACE_SUPPORT_PROBE_MARGIN = 0.10
PLACE_SUPPORT_PROBE_RADIUS_FRACTION = 0.7
PLACE_SUPPORT_EMBED_TOLERANCE = 0.005
PLACE_SUPPORT_FLOAT_TOLERANCE = 0.02
# The stock OnTop sampler leaves 20 mm for a gravity drop. Small rigid objects
# can come to rest at PhysX contact offset without producing the current-frame
# Touching state required by OnTop. Physical placement keeps 5 mm clearance
# and executes the remaining descent through the real arm trajectory.
PLACE_RELEASE_CONTACT_CLEARANCE = 0.005
# Fix P5 (2026-08-16): the live Fix P4 probe shares the sampler's scene-query
# layer, so a uniform loss of a support's top collision mesh (rays pass
# through the top and hit the underside) is invisible to any same-instant
# live check. The pre-grasp reference support-column raycast map
# (_capture_reference_scene_bboxes) is the independent healthy source: if the
# live probe's highest TARGET hit is more than this below the lowest
# pre-grasp hit within PLACE_SUPPORT_REFERENCE_XY_RADIUS of the candidate,
# the top surface was lost post-grasp (attempt-11 class) and the draw is
# rejected with support_surface_lost_vs_reference, feeding the same
# batch-void/rebuild machinery. Static supports do not move between the two
# captures, so the tolerance only absorbs raycast noise.
# Review R4: the reference grid also carries 4 corner rays at +-0.45 of the
# extents; with the interior 4x4 grid alone, P3-corridor corner candidates
# (footprint + XY_MARGIN) sat up to 0.139 m from the nearest reference ray
# on the 0.37x0.40 m breakfast table, silently skipping this cross-check.
# 0.15 m covers corners for extents up to ~1 m.
# Review R3: reference hits ABOVE the captured AABB top belong to objects
# resting ON the support (cutting board, tray). If such an object is removed
# by an earlier plan step, comparing against its top would falsely trip
# support_surface_lost_vs_reference; hits above aabb_top + this epsilon are
# discarded before the comparison.
PLACE_SUPPORT_REFERENCE_SURFACE_TOLERANCE = 0.015
PLACE_SUPPORT_REFERENCE_XY_RADIUS = 0.15
PLACE_SUPPORT_REFERENCE_TOP_EPSILON = 0.005
# Fix W (2026-08-18): the native-support placement path nested an outer
# 40-attempt loop around the official cuboid sampler and, on every in-envelope
# candidate, dumped/settled/reloaded the physics scene while the held object
# was in half-grasped inventory state. deliver_medicine (bottle ->
# breakfast_table_skczfi_2) burned ~7 minutes in that loop emitting repeated
# `Illegal BroadPhaseUpdateData` and produced no expert_result.json. Unify
# native and generated supports on the bounded official OnTop/Inside setter:
# a wall-clock deadline plus a small retry count bound every sample, so a
# failed support rejects one sample instead of stalling the persistent scene.
# These are ceilings, not success criteria: the 1.15 m AABB-edge distance gate
# and the official predicate stay unchanged and never relax.
PLACE_NATIVE_MAX_ATTEMPTS = 6
PLACE_NATIVE_MAX_WALL_SECONDS = 45.0
SUPPORTED_RETRIEVAL_DELIVERY_TASKS = frozenset({
    "retrieve_medicine", "retrieve_key", "retrieve_phone", "retrieve_book",
    "retrieve_drink", "retrieve_food", "deliver_medicine", "deliver_food",
    "deliver_drink",
})
SUPPORTED_OPEN_CLOSE_TASKS = frozenset({
    "open_door", "close_door", "open_window", "close_window",
    "open_fridge", "close_fridge", "open_cabinet", "close_cabinet",
})
SUPPORTED_APPLIANCE_TASKS = frozenset({
    "turn_on_light", "turn_off_light", "turn_on_tv", "turn_off_tv",
    "turn_on_stove", "turn_off_stove",
})
SUPPORTED_FIRE_TASKS = frozenset({
    "respond_to_smoke_warning",
    "select_fire_suppression_tool",
})
SUPPORTED_ENV_A_TASKS = (
    SUPPORTED_RETRIEVAL_DELIVERY_TASKS
    | SUPPORTED_OPEN_CLOSE_TASKS
    | SUPPORTED_APPLIANCE_TASKS
)


class ExpertPlanError(ValueError):
    """Raised when a DeltaSG sample cannot be compiled without guessing IDs."""


def direct_floor_primary_view_error(
    relative_height: float,
    min_height: float = DEFAULT_MIN_DIRECT_FLOOR_PRIMARY_VIEW_HEIGHT,
) -> str | None:
    """Reject floor targets too low for Fetch's stable fixed primary view."""
    if float(relative_height) + 1e-9 < float(min_height):
        return (
            f"floor operation height {float(relative_height):.3f}m is below fixed-primary-view "
            f"minimum {float(min_height):.3f}m"
        )
    return None


def evaluate_manipulation_height(
    primitive: str,
    aabb_min_z: float,
    aabb_max_z: float,
    floor_height: float,
    min_height: float = DEFAULT_MIN_MANIPULATION_HEIGHT,
    max_height: float = DEFAULT_MAX_MANIPULATION_HEIGHT,
) -> dict[str, Any]:
    """Evaluate whether a manipulation point is inside the robot work band.

    GRASP uses the object AABB center, PLACE_ON_TOP uses the support surface,
    and PLACE_INSIDE uses the receptacle center. This deliberately permits a
    sufficiently tall object standing on the floor while rejecting tiny
    floor-level targets that a navigation-only test would incorrectly accept.
    """
    required = primitive in MANIPULATION_PRIMITIVES
    if primitive == "PLACE_ON_TOP":
        point_kind = "support_surface"
        operation_z = float(aabb_max_z)
    else:
        point_kind = (
            "object_center"
            if primitive in {"GRASP", "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}
            else "receptacle_center"
        )
        operation_z = (float(aabb_min_z) + float(aabb_max_z)) / 2.0
    relative_height = operation_z - float(floor_height)
    eligible = not required or float(min_height) <= relative_height <= float(max_height)
    reason = None
    if required and relative_height < float(min_height):
        reason = "operation point is below the robot manipulation height"
    elif required and relative_height > float(max_height):
        reason = "operation point is above the robot manipulation height"
    return {
        "required": required,
        "eligible": eligible,
        "primitive": primitive,
        "point_kind": point_kind,
        "aabb_min_z": float(aabb_min_z),
        "aabb_max_z": float(aabb_max_z),
        "floor_height": float(floor_height),
        "operation_z": operation_z,
        "relative_height": relative_height,
        "min_height": float(min_height),
        "max_height": float(max_height),
        "reason": reason,
    }


@dataclass(frozen=True)
class ExpertStep:
    step_id: int
    primitive: str
    target_object: str | None = None
    target_room: str | None = None
    carried_object: str | None = None
    inventory_before: tuple[str, ...] = ()
    inventory_after: tuple[str, ...] = ()
    useful_objects: tuple[str, ...] = ()
    source_step_ids: tuple[int, ...] = ()
    nl: str = ""
    expected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("inventory_before", "inventory_after", "useful_objects", "source_step_ids"):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class CompiledExpertPlan:
    task_name: str
    task_family: str
    steps: tuple[ExpertStep, ...]
    object_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = "deltasg_expert_plan.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "object_ids": list(self.object_ids),
            "warnings": list(self.warnings),
            "steps": [step.to_dict() for step in self.steps],
        }


def task_environment(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("task_environment") or run


def plan_object_index(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    te = task_environment(run)
    task = te.get("task") or run.get("task") or {}
    result: dict[str, dict[str, Any]] = {}
    for item in task.get("plan_objects") or []:
        object_id = item.get("object_id") or item.get("object_name")
        if object_id:
            result.setdefault(str(object_id), {}).update(item)
    for key in ("added_objects", "state_changed_objects", "task_objects"):
        for item in te.get(key) or run.get(key) or []:
            object_id = item.get("object_id") or item.get("object_name") or item.get("name")
            if object_id:
                result.setdefault(str(object_id), {}).update(
                    {k: v for k, v in item.items() if v is not None}
                )
    # Enrich only explicitly addressable task objects. Adding every graph node
    # would incorrectly make the saved robot instance and all scene furniture
    # mandatory expert targets.
    for node in (run.get("before_graph") or te.get("before_graph") or {}).get("nodes", []):
        object_id = str(node.get("id") or "")
        if object_id in result and node.get("type") == "object":
            enriched = dict(node)
            enriched.update(result[object_id])
            result[object_id] = enriched
    return result


def infer_task_family(task_name: str) -> str:
    if task_name.startswith(("retrieve_", "deliver_", "put_object_")):
        return "retrieval_delivery"
    if task_name.startswith(("open_", "close_")):
        return "open_close"
    if task_name.startswith(("turn_on_", "turn_off_")):
        return "appliance"
    if task_name in SUPPORTED_FIRE_TASKS or "fire" in task_name:
        return "fire"
    return "other"


def _task_target(objects: dict[str, dict[str, Any]]) -> str | None:
    candidates = [
        object_id
        for object_id, item in objects.items()
        if not item.get("reference_only") and (not item.get("reused") or item.get("semantic_role") == "target")
    ]
    if not candidates:
        candidates = [object_id for object_id, item in objects.items() if not item.get("reference_only")]
    return candidates[0] if candidates else None


def _interaction_primitive(task_name: str, nl: str) -> str | None:
    text = f"{task_name} {nl}".lower().replace("-", "_")
    if task_name in SUPPORTED_FIRE_TASKS or any(
        token in text for token in ("extinguish", "suppress", "smoke warning")
    ):
        return "EXTINGUISH"
    if task_name.startswith("open_") or " open " in f" {text} ":
        return "OPEN"
    if task_name.startswith("close_") or " close " in f" {text} ":
        return "CLOSE"
    if task_name.startswith("turn_on_") or any(token in text for token in ("turn on", "toggle on", "switch on")):
        return "TOGGLE_ON"
    if task_name.startswith("turn_off_") or any(token in text for token in ("turn off", "toggle off", "switch off")):
        return "TOGGLE_OFF"
    return None


def _place_primitive(step: dict[str, Any], target: dict[str, Any]) -> str:
    text = str(step.get("nl") or "").lower()
    mode = str(step.get("placement_mode") or target.get("placement_mode") or "").lower()
    category = str(target.get("category") or "").lower()
    if mode in {"inside", "in", "place_inside"} or " inside " in f" {text} ":
        return "PLACE_INSIDE"
    if mode in {"on_top", "on", "place_on_top"}:
        return "PLACE_ON_TOP"
    if any(token in category for token in ("cabinet", "drawer", "fridge", "refrigerator", "container")):
        return "PLACE_INSIDE"
    return "PLACE_ON_TOP"


def _expected(primitive: str, target: str | None, carried: str | None) -> dict[str, Any]:
    if primitive == "GRASP":
        return {"inventory_contains": target}
    if primitive == "PLACE_ON_TOP":
        return {"relation": "OnTop", "subject": carried, "object": target, "inventory_empty": True}
    if primitive == "PLACE_INSIDE":
        return {"relation": "Inside", "subject": carried, "object": target, "inventory_empty": True}
    if primitive == "OPEN":
        return {"state": "Open", "object": target, "value": True}
    if primitive == "CLOSE":
        return {"state": "Open", "object": target, "value": False}
    if primitive == "TOGGLE_ON":
        return {"state": "ToggledOn", "object": target, "value": True}
    if primitive == "TOGGLE_OFF":
        return {"state": "ToggledOn", "object": target, "value": False}
    if primitive == "EXTINGUISH":
        return {"state": "OnFire", "object": target, "value": False}
    return {}


def _raw_plan(run: dict[str, Any]) -> list[dict[str, Any]]:
    te = task_environment(run)
    raw = te.get("solution_plan") or run.get("solution_plan") or []
    if not isinstance(raw, list):
        raise ExpertPlanError("solution_plan must be a list")
    return raw


def _normalize_delivery_open_order(task_name: str, raw_plan: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Move destination OPEN before GRASP when the official primitive needs an empty hand."""
    if not task_name.startswith("deliver_"):
        return raw_plan, []
    pick_index = next(
        (index for index, step in enumerate(raw_plan) if str(step.get("primitive") or "").upper() == "PICK"),
        None,
    )
    if pick_index is None:
        return raw_plan, []
    open_index = next(
        (
            index
            for index, step in enumerate(raw_plan)
            if index > pick_index
            and str(step.get("primitive") or "").upper() == "INTERACT"
            and "open" in str(step.get("nl") or "").lower()
        ),
        None,
    )
    if open_index is None:
        return raw_plan, []
    block_start = open_index
    target = raw_plan[open_index].get("target_object")
    if open_index > 0:
        previous = raw_plan[open_index - 1]
        if str(previous.get("primitive") or "").upper() == "MOVE" and previous.get("target_object") == target:
            block_start = open_index - 1
    block = raw_plan[block_start : open_index + 1]
    # Keep a destination navigation at the original location as well: after
    # opening the receptacle the robot still has to visit the source and return.
    return_navigation = [dict(raw_plan[block_start])] if block_start < open_index else []
    remaining = raw_plan[:block_start] + return_navigation + raw_plan[open_index + 1 :]
    first_pick = next(
        index for index, step in enumerate(remaining) if str(step.get("primitive") or "").upper() == "PICK"
    )
    source_move = first_pick
    while source_move > 0 and str(remaining[source_move - 1].get("primitive") or "").upper() == "MOVE":
        source_move -= 1
    return (
        remaining[:source_move] + block + remaining[source_move:],
        ["moved destination OPEN before GRASP to satisfy empty-hand precondition"],
    )


def compile_expert_plan(run: dict[str, Any]) -> CompiledExpertPlan:
    """Compile an LLM plan into deterministic OmniGibson expert primitives.

    Task semantics may repair an ambiguous INTERACT, but object identifiers are
    never invented.  A missing or unknown action target is a hard rejection.
    """
    if run.get("ok") is False:
        raise ExpertPlanError("DeltaSG sample is not marked ok")
    te = task_environment(run)
    task = te.get("task") or run.get("task") or {}
    task_name = str(task.get("primary_behavior_task") or "").strip()
    if not task_name:
        raise ExpertPlanError("task.primary_behavior_task is missing")
    family = infer_task_family(task_name)
    if family in {"retrieval_delivery", "open_close", "appliance"} and task_name not in SUPPORTED_ENV_A_TASKS:
        raise ExpertPlanError(
            f"task {task_name!r} has no validated Env-A physical contract"
        )
    objects = plan_object_index(run)
    if not objects:
        raise ExpertPlanError("task has no addressable plan objects")
    raw_plan = _raw_plan(run)
    if not raw_plan:
        raise ExpertPlanError("solution_plan is empty")

    default_target = _task_target(objects)
    raw_plan, ordering_warnings = _normalize_delivery_open_order(task_name, raw_plan)
    warnings: list[str] = list(ordering_warnings)
    provisional: list[dict[str, Any]] = []
    inventory: list[str] = []
    last_navigation_target: str | None = None

    for raw_index, raw in enumerate(raw_plan, 1):
        if not isinstance(raw, dict):
            raise ExpertPlanError(f"step {raw_index}: solution step must be an object")
        source = str(raw.get("primitive") or "").upper()
        if source not in SUPPORTED_SOURCE_PRIMITIVES:
            raise ExpertPlanError(f"step {raw_index}: unsupported source primitive {source!r}")
        try:
            source_step_id = int(raw.get("step_id") or raw_index)
        except (TypeError, ValueError) as exc:
            raise ExpertPlanError(f"step {raw_index}: step_id must be an integer") from exc
        target = raw.get("target_object")
        target = str(target) if target else None
        room = raw.get("target_room")
        room = str(room) if room else None
        nl = str(raw.get("nl") or "")

        if source == "MOVE":
            # Room-only MOVE is a topological hint.  The following object-level
            # NAVIGATE_TO performs the executable traversability query.
            if target is None:
                continue
            primitive = "NAVIGATE_TO"
            if target == last_navigation_target:
                continue
            last_navigation_target = target
        elif source == "PICK":
            primitive = "GRASP"
            target = target or default_target
        elif source == "PLACE":
            if not inventory:
                raise ExpertPlanError(f"step {source_step_id}: PLACE with empty inventory")
            primitive = _place_primitive(raw, objects.get(target or "", {}))
        elif source == "INTERACT":
            primitive = _interaction_primitive(task_name, nl)
            if primitive is None and family == "retrieval_delivery" and not inventory:
                primitive = "GRASP"
                target = target or default_target
                warnings.append(f"step {source_step_id}: repaired ambiguous INTERACT to GRASP")
            if primitive is None:
                raise ExpertPlanError(f"step {source_step_id}: INTERACT verb is ambiguous")
        else:
            primitive = "WAIT"

        if primitive != "WAIT":
            if not target:
                raise ExpertPlanError(f"step {source_step_id}: {primitive} target is missing")
            if target not in objects:
                raise ExpertPlanError(f"step {source_step_id}: unknown target object {target!r}")

        before = tuple(inventory)
        carried = inventory[0] if inventory else None
        if primitive == "GRASP":
            if inventory and target not in inventory:
                raise ExpertPlanError(f"step {source_step_id}: GRASP while carrying {inventory[0]!r}")
            inventory = [target] if target else []
            carried = target
        elif primitive in {"PLACE_ON_TOP", "PLACE_INSIDE"}:
            carried = inventory[0]
            inventory = []
        after = tuple(inventory)
        provisional.append(
            {
                "primitive": primitive,
                "target": target,
                "room": room or (objects.get(target or "", {}).get("room") or objects.get(target or "", {}).get("room_id")),
                "carried": carried,
                "before": before,
                "after": after,
                "source_step_id": source_step_id,
                "nl": nl,
            }
        )

    if family == "retrieval_delivery" and not any(item["primitive"] == "GRASP" for item in provisional):
        raise ExpertPlanError("retrieval/delivery plan has no GRASP")
    if task_name.startswith("deliver_") and not any(
        item["primitive"] in {"PLACE_ON_TOP", "PLACE_INSIDE"} for item in provisional
    ):
        raise ExpertPlanError("delivery plan has no PLACE")

    # An inside receptacle with an Open state must be opened with an empty hand.
    # Some LLM plans omit this entirely; add the uniquely implied state change
    # and restore the closed state after placement.
    if task_name.startswith("deliver_"):
        inside = next((item for item in provisional if item["primitive"] == "PLACE_INSIDE"), None)
        if inside is not None:
            destination = inside["target"]
            available_states = set(objects.get(destination or "", {}).get("available_states") or [])
            has_open = any(item["primitive"] == "OPEN" and item["target"] == destination for item in provisional)
            if "Open" in available_states and not has_open:
                source_id = inside["source_step_id"]
                injected = [
                    {
                        "primitive": "NAVIGATE_TO", "target": destination, "room": inside["room"],
                        "carried": None, "before": (), "after": (), "source_step_id": source_id,
                        "nl": "Open the destination before retrieving the object",
                    },
                    {
                        "primitive": "OPEN", "target": destination, "room": inside["room"],
                        "carried": None, "before": (), "after": (), "source_step_id": source_id,
                        "nl": "Open the destination before retrieving the object",
                    },
                ]
                grasp_index = next(index for index, item in enumerate(provisional) if item["primitive"] == "GRASP")
                source_move = grasp_index
                while source_move > 0 and provisional[source_move - 1]["primitive"] == "NAVIGATE_TO":
                    source_move -= 1
                provisional[source_move:source_move] = injected
                place_index = next(index for index, item in enumerate(provisional) if item is inside)
                provisional.insert(
                    place_index + 1,
                    {
                        "primitive": "CLOSE", "target": destination, "room": inside["room"],
                        "carried": None, "before": (), "after": (), "source_step_id": source_id,
                        "nl": "Close the destination after placement",
                    },
                )
                warnings.append("inserted OPEN/CLOSE around PLACE_INSIDE for an openable destination")

    # A support fixture is useful for language grounding, but the final
    # approach pose for a fine operation must face the actual operation target.
    # Rewrite (or insert) the immediately preceding navigation accordingly.
    index = 0
    while index < len(provisional):
        item = provisional[index]
        if item["primitive"] not in {"GRASP", "PLACE_ON_TOP", "PLACE_INSIDE"}:
            index += 1
            continue
        target = item["target"]
        target_room = objects.get(target or "", {}).get("room") or objects.get(target or "", {}).get("room_id")
        if index > 0 and provisional[index - 1]["primitive"] == "NAVIGATE_TO":
            navigation = provisional[index - 1]
            if navigation["target"] != target:
                old_target = navigation["target"]
                navigation.update(
                    {
                        "target": target,
                        "room": target_room or item["room"],
                        "nl": f"Approach the fine-operation target {target}",
                    }
                )
                warnings.append(f"retargeted pre-{item['primitive']} navigation from {old_target} to {target}")
        else:
            provisional.insert(
                index,
                {
                    "primitive": "NAVIGATE_TO", "target": target,
                    "room": target_room or item["room"], "carried": item["carried"],
                    "before": item["before"], "after": item["before"],
                    "source_step_id": item["source_step_id"],
                    "nl": f"Approach the fine-operation target {target}",
                },
            )
            warnings.append(f"inserted pre-{item['primitive']} navigation to {target}")
            index += 1
        index += 1

    steps: list[ExpertStep] = []
    for index, item in enumerate(provisional, 1):
        future = {
            candidate["target"]
            for candidate in provisional[index - 1 :]
            if candidate["target"]
        }
        useful = tuple(sorted(future - set(item["before"])))
        steps.append(
            ExpertStep(
                step_id=index,
                primitive=item["primitive"],
                target_object=item["target"],
                target_room=item["room"],
                carried_object=item["carried"],
                inventory_before=item["before"],
                inventory_after=item["after"],
                useful_objects=useful,
                source_step_ids=(item["source_step_id"],),
                nl=item["nl"],
                expected=_expected(item["primitive"], item["target"], item["carried"]),
            )
        )
    if not steps:
        raise ExpertPlanError("solution_plan contains no executable steps")
    return CompiledExpertPlan(
        task_name=task_name,
        task_family=family,
        steps=tuple(steps),
        object_ids=tuple(sorted(objects)),
        warnings=tuple(warnings),
    )


def validate_env_a_plan_contract(
    run: dict[str, Any],
    *,
    task_object_id: str | None = None,
    destination_object_id: str | None = None,
    native_target_id: str | None = None,
) -> CompiledExpertPlan:
    """Compile and enforce exact Env-A action-object semantics.

    The generic compiler rejects invented IDs and invalid inventory transitions.
    This stricter gate also prevents a syntactically valid LLM plan from grasping
    a support instead of the task item, or manipulating a different fixture of
    the same category.
    """
    compiled = compile_expert_plan(run)
    manipulations = [
        step for step in compiled.steps if step.primitive in MANIPULATION_PRIMITIVES
    ]
    if compiled.task_family == "retrieval_delivery":
        if not task_object_id:
            raise ExpertPlanError("Env-A retrieval contract is missing the exact task object")
        grasps = [step for step in manipulations if step.primitive == "GRASP"]
        if len(grasps) != 1 or grasps[0].target_object != task_object_id:
            raise ExpertPlanError(
                f"retrieval contract must GRASP exact task object {task_object_id!r} once"
            )
        places = [
            step for step in manipulations
            if step.primitive in {"PLACE_ON_TOP", "PLACE_INSIDE"}
        ]
        if compiled.task_name.startswith("deliver_"):
            if not destination_object_id:
                raise ExpertPlanError("Env-A delivery contract is missing the exact destination")
            if len(places) != 1 or places[0].target_object != destination_object_id:
                raise ExpertPlanError(
                    f"delivery contract must PLACE at exact destination {destination_object_id!r} once"
                )
            destination_mode = str(
                plan_object_index(run).get(destination_object_id, {}).get("placement_mode") or ""
            ).lower()
            expected_place = (
                "PLACE_INSIDE"
                if destination_mode in {"inside", "in", "place_inside"}
                else "PLACE_ON_TOP"
                if destination_mode in {"on_top", "on", "place_on_top"}
                else None
            )
            if expected_place and places[0].primitive != expected_place:
                raise ExpertPlanError(
                    f"delivery contract requires {expected_place} for destination mode {destination_mode!r}"
                )
            if grasps[0].step_id >= places[0].step_id:
                raise ExpertPlanError("delivery contract must GRASP before PLACE")
            if len(manipulations) != 2:
                raise ExpertPlanError("delivery contract must contain only GRASP and PLACE")
        elif places:
            raise ExpertPlanError("retrieve-only contract must not PLACE the task object")
        elif len(manipulations) != 1:
            raise ExpertPlanError("retrieve-only contract must contain only one GRASP")
    else:
        if not native_target_id:
            raise ExpertPlanError("Env-A native-task contract is missing the exact target")
        if compiled.task_name.startswith("open_"):
            expected = "OPEN"
        elif compiled.task_name.startswith("close_"):
            expected = "CLOSE"
        elif compiled.task_name.startswith("turn_on_"):
            expected = "TOGGLE_ON"
        else:
            expected = "TOGGLE_OFF"
        if len(manipulations) != 1:
            raise ExpertPlanError("native-task contract must contain exactly one manipulation")
        action = manipulations[0]
        if action.primitive != expected or action.target_object != native_target_id:
            raise ExpertPlanError(
                f"native-task contract must {expected} exact target {native_target_id!r}"
            )
    return compiled


def validate_visibility_snapshot(
    step: ExpertStep,
    robot_visible: Iterable[str],
    global_visible: Iterable[str],
    robot_bboxes: dict[str, dict[str, Any]],
    min_bbox_pixels: int = 8,
) -> list[str]:
    """Return visibility violations for one expert sampling event."""
    robot_visible = set(robot_visible)
    global_visible = set(global_visible)
    union = robot_visible | global_visible
    errors = [
        f"useful object {object_id!r} is invisible"
        for object_id in step.useful_objects
        if object_id not in union
    ]
    if step.primitive in FINE_MANIPULATION_PRIMITIVES and step.target_object:
        if step.target_object not in robot_visible:
            errors.append(f"manipulation target {step.target_object!r} is absent from robot primary view")
        bbox = robot_bboxes.get(step.target_object)
        if not bbox or int(bbox.get("pixel_count", 0)) < min_bbox_pixels:
            errors.append(f"manipulation target {step.target_object!r} has no valid robot bbox")
        elif len(bbox.get("bbox_xyxy") or []) != 4:
            errors.append(f"manipulation target {step.target_object!r} bbox is malformed")
        else:
            x1, y1, x2, y2 = bbox["bbox_xyxy"]
            if x2 <= x1 or y2 <= y1:
                errors.append(f"manipulation target {step.target_object!r} bbox has zero area")
            image_size = bbox.get("image_size") or []
            if len(image_size) == 2:
                width, height = (int(value) for value in image_size)
                margin = max(2, int(round(min(width, height) * 0.01)))
                clipped = x1 < margin or y1 < margin or x2 >= width - margin
                # PLACE operates on the support's visible top/opening. A
                # floor-standing table or cabinet may legitimately continue
                # below the primary image; requiring all legs / the full body
                # adds no manipulation evidence and rejects well-framed tops.
                if step.primitive not in {"PLACE_ON_TOP", "PLACE_INSIDE"}:
                    clipped = clipped or y2 >= height - margin
                if clipped:
                    errors.append(f"manipulation target {step.target_object!r} bbox is clipped")
    return errors


def summarize_expert_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results = list(results)
    accepted = [item for item in results if item.get("accepted") is True]
    by_task: dict[str, dict[str, int]] = {}
    for item in results:
        task = str(item.get("task_name") or "unknown")
        row = by_task.setdefault(task, {"total": 0, "accepted": 0})
        row["total"] += 1
        row["accepted"] += int(item.get("accepted") is True)
    for row in by_task.values():
        row["accept_rate"] = row["accepted"] / row["total"] if row["total"] else 0.0
    return {
        "schema_version": "deltasg_expert_audit.v1",
        "total": len(results),
        "accepted": len(accepted),
        "accept_rate": len(accepted) / len(results) if results else 0.0,
        "by_task": by_task,
    }


def place_descent_corridor_blockers(
    place_pos,
    held_half_extents,
    obstacle_aabbs,
    hover_clearance=PLACE_ACCESS_HOVER_CLEARANCE,
    margin=PLACE_ACCESS_CORRIDOR_MARGIN,
):
    """Obstacle names blocking an OnTop place candidate's descent corridor.

    The corridor is the world-AABB column swept by the held object between its
    final place position and ``hover_clearance`` above it. Any obstacle whose
    ``margin``-inflated AABB intersects that column would be clipped by the
    descending object (or the arm carrying it), so the candidate is access
    blocked and must be re-sampled. This is a conservative rejection gate: it
    only ever removes candidates and never relaxes a collision/reachability
    standard (Fix N, attempt-7 / diag27, Beechwood_0 deliver_drink).

    Args:
        place_pos: (x, y, z) center of the held object at the final place pose
        held_half_extents: (hx, hy, hz) world half extents of the held object
        obstacle_aabbs: iterable of (name, (low_xyz, high_xyz)); the caller
            excludes the placement target, the held object and the robot
        hover_clearance: vertical hover height above the place pose
        margin: inflation applied to every obstacle AABB

    Returns:
        Sorted list of blocker names (empty when the corridor is clear).
    """
    px, py, pz = (float(v) for v in place_pos)
    hx, hy, hz = (float(v) for v in held_half_extents)
    low = (px - hx - margin, py - hy - margin, pz - hz)
    high = (px + hx + margin, py + hy + margin, pz + hz + float(hover_clearance) + margin)
    blockers = set()
    for name, (o_low, o_high) in obstacle_aabbs:
        if (
            float(o_low[0]) < high[0]
            and float(o_high[0]) > low[0]
            and float(o_low[1]) < high[1]
            and float(o_high[1]) > low[1]
            and float(o_low[2]) < high[2]
            and float(o_high[2]) > low[2]
        ):
            blockers.add(str(name))
    return sorted(blockers)
