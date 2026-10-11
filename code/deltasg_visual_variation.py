"""Shared visual-domain variation parameters for DeltaSG instances.

Two paths must sample *identical* lighting/FOV for the same ``(identity,
variant)`` so that an offline-recolored frame and a physically re-rendered frame
of the same instance agree:

- ``recolor_deltasg_camera`` — post-processes already-rendered ``rgb.png``
  frames (offline, no simulator).
- ``run_deltasg_expert`` — re-renders with the Dome-light + FOV randomization
  applied inside the simulator.

The constants below mirror ``run_deltasg_expert.py`` (``_LIGHTING_*``,
``_VISUAL_FOV_*``, ``_visual_seed``, ``_apply_expert_lighting``). Keep the two in
sync; this module is the single source of truth for any new code.
"""

from __future__ import annotations

import hashlib
import random

# --- lighting (skybox Dome light) -------------------------------------------
LIGHTING_INTENSITY_MIN = 1800.0
LIGHTING_INTENSITY_MAX = 3800.0
# The fixed intensity that pre-randomization renders used (report section 1.2).
LIGHTING_INTENSITY_NOMINAL = 2500.0
# Warm-white Dome-light colour, per channel (r, g, b).
LIGHTING_COLOR_BASE = (1.07, 0.85, 0.61)
LIGHTING_COLOR_JITTER = 0.08
LIGHTING_COLOR_CLIP = (0.5, 1.5)

# --- camera FOV (degrees) ----------------------------------------------------
FOV_MIN_DEG = 55.0
FOV_MAX_DEG = 75.0
FOV_NOMINAL_DEG = 65.0


def visual_seed(identity, variant=0):
    """Deterministic seed for one visual variant of an instance.

    ``identity`` is normally the run id (e.g. ``online_env_a_0013``). Variant 0
    reproduces ``run_deltasg_expert._visual_seed`` exactly; variants > 0 stay
    deterministic but diverge from each other (and from variant 0).
    """
    key = str(identity) if variant == 0 else f"{identity}::{variant}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def _clip(value, low, high):
    return max(low, min(high, value))


def sample_lighting(rng):
    """Draw ``(intensity, rgb_color)`` in the same order as the expert stage.

    Order matters: intensity first, then the three per-channel colour jitters —
    this matches ``_apply_expert_lighting`` so the same seed yields the same
    values in both paths.
    """
    intensity = rng.uniform(LIGHTING_INTENSITY_MIN, LIGHTING_INTENSITY_MAX)
    color = tuple(
        _clip(
            base + rng.uniform(-LIGHTING_COLOR_JITTER, LIGHTING_COLOR_JITTER),
            LIGHTING_COLOR_CLIP[0],
            LIGHTING_COLOR_CLIP[1],
        )
        for base in LIGHTING_COLOR_BASE
    )
    return intensity, color


def sample_fov(rng):
    return rng.uniform(FOV_MIN_DEG, FOV_MAX_DEG)


def lighting_gain(intensity, color):
    """Per-channel multiplier mapping the nominal render to the sampled lighting.

    The Dome light illuminates each channel proportionally to
    ``intensity * color[i]``; dividing by the nominal ``intensity * base[i]``
    yields the exposure/warmth gain to apply to a stored ``rgb`` frame. This is a
    first-order approximation of a true re-render (shadows/highlights unchanged).
    """
    return tuple(
        (intensity / LIGHTING_INTENSITY_NOMINAL)
        * (color[i] / LIGHTING_COLOR_BASE[i])
        for i in range(3)
    )


def draw_variant(identity, variant=0, with_fov=False):
    """One-shot deterministic draw for an instance variant.

    Returns ``(record, gain)`` where ``record`` carries ``seed``, ``intensity``
    and ``color`` (plus ``fov_deg`` when ``with_fov`` is true, drawn after the
    lighting in the same RNG stream) and ``gain`` is the per-channel RGB gain.
    """
    seed = visual_seed(identity, variant)
    rng = random.Random(seed)
    intensity, color = sample_lighting(rng)
    record = {
        "seed": seed,
        "intensity": round(float(intensity), 2),
        "color": [round(float(c), 4) for c in color],
    }
    if with_fov:
        record["fov_deg"] = round(float(sample_fov(rng)), 4)
    return record, lighting_gain(intensity, color)
