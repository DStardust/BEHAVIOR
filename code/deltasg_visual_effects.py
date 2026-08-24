"""DeltaSG-specific configuration for OmniGibson's built-in visual effects."""

from __future__ import annotations


SMOKE_ONLY_ON_FIRE_MODE = "omnigibson_on_fire_smoke_only"
SMOKE_FLOW_WARMUP_STEPS = 120
SMOKE_FLOW_RENDER_WARMUP_FRAMES = 30
SMOKE_FLOW_MAX_EMITTER_RADIUS = 0.12


def smoke_only_on_fire_record():
    return {
        "mode": SMOKE_ONLY_ON_FIRE_MODE,
        "source": "omnigibson_flow_emitter",
        "smoke_visible": True,
        "flame_visible": False,
    }


def configure_on_fire_smoke_only(obj):
    """Keep the official OnFire Flow effect while removing flame colors."""
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.constants import EmitterType

    result = {"ok": False, **smoke_only_on_fire_record(), "errors": []}
    emitter_info = (getattr(obj, "_emitters", None) or {}).get(EmitterType.FIRE)
    mesh = (emitter_info or {}).get("mesh")
    if mesh is None:
        result["errors"].append({"error": "official_fire_emitter_missing"})
        return result

    colormap_path = f"{mesh.prim_path}/flowOffscreen/colormap"
    colormap = og.sim.stage.GetPrimAtPath(colormap_path)
    if not colormap.IsValid():
        result["errors"].append({"error": "official_fire_colormap_missing", "path": colormap_path})
        return result

    # Preserve the official Flow simulation and smoke density. Replacing only
    # its color ramp removes orange/yellow flame pixels while retaining a
    # visible dark-to-light smoke plume.
    rgba_points = [
        lazy.pxr.Gf.Vec4f(0.015, 0.015, 0.015, 0.02),
        lazy.pxr.Gf.Vec4f(0.040, 0.040, 0.040, 0.55),
        lazy.pxr.Gf.Vec4f(0.080, 0.080, 0.080, 0.70),
        lazy.pxr.Gf.Vec4f(0.150, 0.150, 0.150, 0.75),
        lazy.pxr.Gf.Vec4f(0.300, 0.300, 0.300, 0.70),
        lazy.pxr.Gf.Vec4f(0.600, 0.600, 0.600, 0.55),
    ]
    try:
        settings = lazy.carb.settings.get_settings()
        settings.set_bool("/rtx/flow/enabled", True)
        settings.set_bool("/rtx/flow/compositeEnabled", True)
        settings.set_bool("/rtx/flow/pathTracingEnabled", True)
        with og.sim.editing_usd():
            colormap.GetAttribute("rgbaPoints").Set(rgba_points)
            emitter = emitter_info["emitter"]
            radius_attr = emitter.GetAttribute("radius")
            radius = min(float(radius_attr.Get()), SMOKE_FLOW_MAX_EMITTER_RADIUS)
            radius_attr.Set(radius)
            simulate = og.sim.stage.GetPrimAtPath(f"{mesh.prim_path}/flowSimulate")
            simulate.GetAttribute("densityCellSize").Set(radius * 0.2)
            smoke = og.sim.stage.GetPrimAtPath(
                f"{mesh.prim_path}/flowSimulate/advection/smoke"
            )
            ray_march = og.sim.stage.GetPrimAtPath(f"{mesh.prim_path}/flowRender/rayMarch")
            smoke.GetAttribute("fade").Set(0.5)
            ray_march.GetAttribute("attenuation").Set(5.0)
        # OnFire is the source of truth. Force the normal OmniGibson visual
        # update now so a disabled Flow emitter cannot be reported as smoke.
        obj.update_visuals()
    except Exception as exc:
        result["errors"].append({"error": repr(exc)})
        return result

    emitter = emitter_info.get("emitter")
    enabled = bool(emitter and emitter.GetAttribute("enabled").Get())
    if not enabled:
        result["errors"].append({"error": "official_fire_emitter_not_enabled"})
        return result

    result.update(
        {
            "ok": True,
            "colormap_path": colormap_path,
            "smoke_fade": 0.5,
            "ray_march_attenuation": 5.0,
            "emitter_enabled": True,
            "emitter_radius": radius,
            "rtx_flow_enabled": True,
            "rtx_flow_composite_enabled": True,
        }
    )
    return result
