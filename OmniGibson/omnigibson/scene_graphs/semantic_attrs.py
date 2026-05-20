"""
Helpers for enriching OmniGibson scene graph nodes with task-editing metadata.

The labels here are intentionally conservative. OmniGibson does not expose a
single "interactable" property, so interaction affordances are inferred from
object abilities, instantiated object states, category names, and simple shape
cues.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch as th

from omnigibson import object_states
from omnigibson.object_states.factory import get_state_name


SUPPORT_SURFACE_TOKENS = frozenset(
    {
        "bar",
        "bed",
        "bench",
        "cabinet",
        "cart",
        "chair",
        "counter",
        "desk",
        "dresser",
        "floor",
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
)

INSIDE_RECEPTACLE_TOKENS = frozenset(
    {
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
)

STRUCTURAL_TOKENS = frozenset(
    {
        "ceiling",
        "doorway",
        "floor",
        "room",
        "stair",
        "wall",
        "window",
    }
)

CONTROLLABLE_ABILITIES = frozenset(
    {
        "coldSource",
        "heatSource",
        "particleApplier",
        "particleRemover",
        "particleSink",
        "particleSource",
        "toggleable",
    }
)
ARTICULABLE_ABILITIES = frozenset({"openable"})
MANIPULABLE_ABILITIES = frozenset(
    {
        "attachable",
        "cloth",
        "cookable",
        "fillable",
        "freezable",
        "heatable",
        "meltable",
        "mixingTool",
        "saturable",
        "sliceable",
        "slicer",
    }
)

ABNORMAL_ABILITY_TO_STATE = {
    "breakable": "broken",
    "flammable": "on_fire",
    "cookable": "burnt",
    "coverable": "covered",
}

ABNORMAL_STATE_TYPES = {
    object_states.OnFire: "on_fire",
    object_states.Burnt: "burnt",
    object_states.Covered: "covered",
}


def _as_token_set(value):
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.replace("-", "_")
        return {token for token in value.lower().split("_") if token}
    return set()


def _get_category_tokens(obj):
    tokens = set()
    for attr in ("category", "name"):
        tokens |= _as_token_set(getattr(obj, attr, None))
    return tokens


def _get_abilities(obj):
    abilities = getattr(obj, "abilities", None)
    if abilities is None:
        abilities = getattr(obj, "_abilities", {})
    if isinstance(abilities, Mapping):
        return dict(abilities)
    return {}


def _get_state_types(obj):
    states = getattr(obj, "states", {})
    return set(states.keys()) if isinstance(states, Mapping) else set()


def _extent_to_list(bbox_extent):
    if bbox_extent is None:
        return None
    if isinstance(bbox_extent, th.Tensor):
        return bbox_extent.detach().cpu().tolist()
    if hasattr(bbox_extent, "tolist"):
        return bbox_extent.tolist()
    return list(bbox_extent)


def _is_large_horizontal_surface(bbox_extent):
    extent = _extent_to_list(bbox_extent)
    if not extent or len(extent) < 3:
        return False
    x, y, z = [float(v) for v in extent[:3]]
    return x >= 0.35 and y >= 0.35 and z <= max(x, y) * 0.6


def _safe_get_state_value(obj, state_type):
    states = getattr(obj, "states", {})
    if not isinstance(states, Mapping) or state_type not in states:
        return None
    try:
        return states[state_type].get_value()
    except Exception:
        return None


def infer_receptacle_affordance(obj, bbox_extent=None):
    """
    Infer whether an object can receive added task assets on top or inside.

    This avoids using the generic OnTop / Inside relative states alone because
    those are default kinematic predicates in OmniGibson, not object-specific
    support annotations.
    """

    abilities = _get_abilities(obj)
    state_types = _get_state_types(obj)
    tokens = _get_category_tokens(obj)
    reasons = []

    supports_on_top = bool(tokens & SUPPORT_SURFACE_TOKENS)
    if supports_on_top:
        reasons.append("category_surface_token")
    elif _is_large_horizontal_surface(bbox_extent):
        supports_on_top = True
        reasons.append("large_horizontal_bbox")

    supports_inside = bool(tokens & INSIDE_RECEPTACLE_TOKENS)
    if supports_inside:
        reasons.append("category_container_token")
    if abilities.keys() & {"fillable", "openable"}:
        supports_inside = True
        reasons.append("container_ability")
    if state_types & {object_states.Filled, object_states.Contains, object_states.Open}:
        supports_inside = True
        reasons.append("container_state")

    return {
        "can_support": supports_on_top or supports_inside,
        "supports_on_top": supports_on_top,
        "supports_inside": supports_inside,
        "confidence": "inferred" if reasons else "unknown",
        "reasons": reasons,
    }


def infer_interaction_affordance(obj, bbox_extent=None):
    """
    Infer interaction affordance without relying on a non-existent OG property.

    Priority is controllable > articulable > manipulable > none.
    """

    abilities = _get_abilities(obj)
    ability_names = set(abilities)
    state_types = _get_state_types(obj)
    tokens = _get_category_tokens(obj)
    reasons = []

    if ability_names & CONTROLLABLE_ABILITIES or state_types & {
        object_states.ToggledOn,
        object_states.HeatSourceOrSink,
        object_states.ParticleApplier,
        object_states.ParticleRemover,
        object_states.ParticleSink,
        object_states.ParticleSource,
    }:
        reasons.append("controllable_ability_or_state")
        return {"kind": "controllable", "confidence": "inferred", "reasons": reasons}

    if ability_names & ARTICULABLE_ABILITIES or state_types & {object_states.Open, object_states.Joint}:
        reasons.append("articulable_ability_or_state")
        return {"kind": "articulable", "confidence": "inferred", "reasons": reasons}

    structural = bool(tokens & STRUCTURAL_TOKENS) or "sceneObject" in ability_names
    extent = _extent_to_list(bbox_extent)
    compact = extent is None or max(float(v) for v in extent[:3]) <= 1.5
    if not structural and (ability_names & MANIPULABLE_ABILITIES or compact):
        reasons.append("manipulation_ability_or_compact_bbox")
        return {"kind": "manipulable", "confidence": "inferred", "reasons": reasons}

    if structural:
        reasons.append("structural_category")
    return {"kind": "none", "confidence": "inferred" if reasons else "unknown", "reasons": reasons}


def infer_abnormal_states(obj):
    abilities = _get_abilities(obj)
    state_types = _get_state_types(obj)

    potential = set()
    for ability, abnormal_state in ABNORMAL_ABILITY_TO_STATE.items():
        if ability in abilities:
            potential.add(abnormal_state)

    for state_type, abnormal_state in ABNORMAL_STATE_TYPES.items():
        if state_type in state_types:
            potential.add(abnormal_state)

    current = []
    for state_type, abnormal_state in ABNORMAL_STATE_TYPES.items():
        value = _safe_get_state_value(obj, state_type)
        if value:
            current.append(abnormal_state)

    return {
        "potential": sorted(potential),
        "current": current,
        "confidence": "state_checked" if state_types else "ability_inferred" if potential else "none",
    }


def infer_object_edit_metadata(obj, bbox_extent=None):
    abilities = sorted(_get_abilities(obj))
    state_types = sorted(get_state_name(state_type) for state_type in _get_state_types(obj))
    return {
        "category": getattr(obj, "category", None),
        "model": getattr(obj, "model", None),
        "rooms": getattr(obj, "in_rooms", None),
        "abilities": abilities,
        "state_types": state_types,
        "receptacle": infer_receptacle_affordance(obj, bbox_extent=bbox_extent),
        "interaction": infer_interaction_affordance(obj, bbox_extent=bbox_extent),
        "abnormal_states": infer_abnormal_states(obj),
    }
