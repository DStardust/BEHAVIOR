"""DeltaSG visual effects that can be reproduced during generation and replay."""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np


SMOKE_ONLY_ON_FIRE_MODE = "omnigibson_on_fire_smoke_only"
SMOKE_FLOW_WARMUP_STEPS = 120
SMOKE_FLOW_RENDER_WARMUP_FRAMES = 30
SMOKE_FLOW_MAX_EMITTER_RADIUS = 0.12
SMOKE_COLUMN_EMITTER_RADIUS = 0.18
SMOKE_COLUMN_UPWARD_VELOCITY = 1.5
SMOKE_COLUMN_FADE = 0.12

USDZ_FLAME_MODE = "deltasg_usdz_flame_v1"
USDZ_FLAME_SMOKE_MODE = "deltasg_usdz_flame_smoke_column_v1"
USDZ_FLAME_ASSET = "code/assets/Flame_Animation.usdz"
USDZ_FLAME_ROOT = "/World/deltasg_visual_effects"
USDZ_FLAME_RENDER_WARMUP_FRAMES = 3
MACRO_PARTICLE_COVERED_MODE = "omnigibson_macro_particle_covered_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def flame_asset_path() -> Path:
    return _repo_root() / USDZ_FLAME_ASSET


def smoke_only_on_fire_record():
    """Legacy record retained so older generated samples remain replayable."""
    return {
        "mode": SMOKE_ONLY_ON_FIRE_MODE,
        "source": "omnigibson_flow_emitter",
        "smoke_visible": True,
        "flame_visible": False,
    }


def usdz_flame_record(*, effect_id=None, target_height=None):
    record = {
        "mode": USDZ_FLAME_MODE,
        "source": "Flame_Animation.usdz",
        "asset": USDZ_FLAME_ASSET,
        "animated": True,
        "smoke_visible": False,
        "flame_visible": True,
        "replaces_official_flow_visual": True,
    }
    if effect_id:
        record["effect_id"] = str(effect_id)
    if target_height is not None:
        record["target_height"] = float(target_height)
    return record


def usdz_flame_smoke_record(*, effect_id=None, target_height=None):
    record = usdz_flame_record(effect_id=effect_id, target_height=target_height)
    record.update(
        mode=USDZ_FLAME_SMOKE_MODE,
        smoke_visible=True,
        replaces_official_flow_visual=False,
        smoke_profile="vertical_column_v1",
        smoke_emitter_radius=SMOKE_COLUMN_EMITTER_RADIUS,
        smoke_upward_velocity=SMOKE_COLUMN_UPWARD_VELOCITY,
        smoke_fade=SMOKE_COLUMN_FADE,
    )
    return record


def covered_particle_record(system_name: str):
    return {
        "mode": MACRO_PARTICLE_COVERED_MODE,
        "source": "omnigibson.Covered",
        "system": str(system_name),
        "particles_visible": True,
    }


def set_covered_particles(obj, system_name: str, value: bool, particle_snapshot=None):
    """Apply and verify OmniGibson's official relative Covered state."""
    import omnigibson as og
    from omnigibson import object_states

    result = {
        "ok": False,
        **covered_particle_record(system_name),
        "value": bool(value),
        "errors": [],
    }
    if obj is None:
        result["errors"].append({"error": "covered_target_missing"})
        return result
    if object_states.Covered not in obj.states:
        result["errors"].append({"error": "covered_state_not_available"})
        return result
    try:
        system = obj.scene.get_system(str(system_name))
        if value and particle_snapshot is not None:
            restore_covered_particle_snapshot(obj, system, particle_snapshot)
            changed = True
        else:
            # Ray-based surface sampling needs the scene query geometry to
            # reflect replay teleports and collision reactivation first.
            if value and og.sim.is_playing():
                og.sim.step()
            changed = bool(obj.states[object_states.Covered].set_value(system, bool(value)))
        result["setter_success"] = changed
        if changed:
            if og.sim.is_playing():
                og.sim.step()
            else:
                og.sim.render()
        actual = bool(obj.states[object_states.Covered].get_value(system))
        result.update({"ok": changed and actual == bool(value), "actual": actual})
        if result["ok"] and value:
            result["particle_snapshot"] = covered_particle_snapshot(obj, system)
        if not result["ok"]:
            result["target_aabb_extent"] = obj.aabb_extent.tolist()
            result["collision_enabled"] = {
                name: [bool(api.GetCollisionEnabledAttr().Get()) for api in link._collision_apis]
                for name, link in obj.links.items()
            }
            result["system_particle_count"] = int(system.n_particles)
    except Exception as exc:
        result["errors"].append({"error": repr(exc)})
    return result


def covered_particle_snapshot(obj, system):
    """Save this rigid object's official particles, not unrelated scene groups."""
    group = system.get_group_name(obj)
    state = system.dump_state(serialized=False)
    info = state["groups"][group]
    links = info["particle_attached_references"]
    if not all(isinstance(name, str) and name in obj.links for name in links):
        raise ValueError("Covered snapshot requires rigid object attachment links")
    positions, orientations = system.get_group_particles_local_pose(group)
    return {
        "version": 1,
        "system": system.name,
        "positions": positions.tolist(),
        "orientations": orientations.tolist(),
        "scales": state["scales"][info["particle_indices"]].tolist(),
        "links": links,
    }


def restore_covered_particle_snapshot(obj, system, snapshot):
    import torch as th

    if snapshot.get("version") != 1 or snapshot.get("system") != system.name:
        raise ValueError("Covered snapshot version or system mismatch")
    links = snapshot["links"]
    count = len(links)
    positions = th.tensor(snapshot["positions"], dtype=th.float32)
    orientations = th.tensor(snapshot["orientations"], dtype=th.float32)
    scales = th.tensor(snapshot["scales"], dtype=th.float32)
    if (not count or positions.shape != (count, 3) or orientations.shape != (count, 4)
            or scales.shape != (count, 3) or any(name not in obj.links for name in links)
            or not all(th.isfinite(x).all() for x in (positions, orientations, scales))
            or not (scales > 0).all()):
        raise ValueError("Invalid Covered particle snapshot")
    group = system.get_group_name(obj)
    if group not in system.groups:
        system.create_attachment_group(obj)
    system.remove_all_group_particles(group)
    system.generate_group_particles(
        group=group, positions=positions.clone(), orientations=orientations,
        scales=scales, link_prim_paths=[obj.links[name].prim_path for name in links],
    )
    # Generation expects world surface coordinates; restore saved local poses
    # afterwards to preserve exact attachment and avoid applying clipping twice.
    system.set_group_particles_local_pose(group, positions=positions, orientations=orientations)


def _safe_effect_id(value) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "fire"))
    return value.strip("_") or "fire"


def _extract_flame_textures(asset_path: Path) -> dict[str, str]:
    output_dir = Path(tempfile.gettempdir()) / "deltasg_flame_textures"
    output_dir.mkdir(parents=True, exist_ok=True)
    textures = {}
    with zipfile.ZipFile(asset_path) as archive:
        for member in archive.namelist():
            basename = os.path.basename(member)
            if not basename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            destination = output_dir / basename
            if not destination.exists() or destination.stat().st_size != archive.getinfo(member).file_size:
                with archive.open(member) as source, destination.open("wb") as target:
                    target.write(source.read())
            textures[basename] = str(destination)
    return textures


def _world_bounds(prim):
    import omnigibson.lazy as lazy

    pxr = lazy.pxr
    cache = pxr.UsdGeom.BBoxCache(
        pxr.Usd.TimeCode.Default(), [pxr.UsdGeom.Tokens.default_]
    )
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lower = np.asarray(aligned.GetMin(), dtype=np.float64)
    upper = np.asarray(aligned.GetMax(), dtype=np.float64)
    return (lower + upper) * 0.5, upper - lower


def _fix_flame_material(prim_path: str, textures: dict[str, str]):
    import omnigibson as og
    import omnigibson.lazy as lazy

    pxr = lazy.pxr
    material_path = f"{prim_path}/Materials/MaterialFire"
    shader_prim = og.sim.stage.GetPrimAtPath(f"{material_path}/pbr_shader")
    if not shader_prim.IsValid():
        raise RuntimeError(f"Flame material shader missing under {prim_path}")
    shader = pxr.UsdShade.Shader(shader_prim)
    for shader_name, texture_name in (
        ("tex_base", "MaterialFire_baseColor.png"),
        ("tex_emissive", "MaterialFire_emissive.jpg"),
        ("tex_metallic", "MaterialFire_metallicRoughness_metal_scale0.jpg"),
        ("tex_roughness", "MaterialFire_metallicRoughness_rough_scale1.jpg"),
    ):
        texture_prim = og.sim.stage.GetPrimAtPath(f"{material_path}/{shader_name}")
        if texture_prim.IsValid() and texture_name in textures:
            pxr.UsdShade.Shader(texture_prim).GetInput("file").Set(textures[texture_name])

    emissive = shader.GetInput("emissiveColor")
    if not emissive.GetConnectedSources():
        texture_prim = og.sim.stage.GetPrimAtPath(f"{material_path}/tex_emissive")
        if texture_prim.IsValid():
            emissive.ConnectToSource(pxr.UsdShade.Shader(texture_prim).GetOutput("rgb"))
    emissive_attr = shader_prim.GetAttribute("inputs:emissiveColor")
    if not emissive_attr.HasAuthoredConnections():
        emissive_attr.Set(pxr.Gf.Vec3f(1.0, 1.0, 1.0))
    shader.CreateInput("opacity", pxr.Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("diffuseColor", pxr.Sdf.ValueTypeNames.Color3f)
    diffuse = shader_prim.GetAttribute("inputs:diffuseColor")
    diffuse.ClearConnections()
    diffuse.Set(pxr.Gf.Vec3f(0.1, 0.05, 0.0))
    shader.CreateInput("roughness", pxr.Sdf.ValueTypeNames.Float)
    roughness = shader_prim.GetAttribute("inputs:roughness")
    roughness.ClearConnections()
    roughness.Set(1.0)


def _disable_official_fire_visual(obj) -> bool:
    import omnigibson as og
    from omnigibson.utils.constants import EmitterType

    emitter_info = (getattr(obj, "_emitters", None) or {}).get(EmitterType.FIRE) or {}
    emitter = emitter_info.get("emitter")
    if emitter is None:
        return False
    enabled = emitter.GetAttribute("enabled")
    with og.sim.editing_usd():
        enabled.Set(False)
    return not bool(enabled.Get())


def _adaptive_flame_height(obj) -> float:
    lower, upper = obj.aabb
    extent = np.asarray((upper - lower).detach().cpu(), dtype=float)
    footprint = max(0.0, min(float(extent[0]), float(extent[1])))
    return float(np.clip(footprint * 0.65, 0.24, 0.42))


def create_usdz_flame(obj, *, effect_id=None, target_height=None, keep_official_flow=False):
    """Place the packaged animated flame on ``obj``."""
    import omnigibson as og
    import omnigibson.lazy as lazy

    asset_path = flame_asset_path()
    result = {
        "ok": False,
        **usdz_flame_record(effect_id=effect_id, target_height=target_height),
        "errors": [],
    }
    result["replaces_official_flow_visual"] = not keep_official_flow
    if obj is None:
        result["errors"].append({"error": "fire_target_missing"})
        return result
    if not asset_path.is_file():
        result["errors"].append({"error": "flame_asset_missing", "path": str(asset_path)})
        return result

    effect_id = _safe_effect_id(effect_id or getattr(obj, "name", "fire"))
    prim_path = f"{USDZ_FLAME_ROOT}/flame_{effect_id}"
    light_path = f"{USDZ_FLAME_ROOT}/light_{effect_id}"
    try:
        stage = og.sim.stage
        pxr = lazy.pxr
        with og.sim.editing_usd():
            stage.DefinePrim(USDZ_FLAME_ROOT, "Scope")
            for path in (prim_path, light_path):
                if stage.GetPrimAtPath(path).IsValid():
                    stage.RemovePrim(path)
            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(str(asset_path))
        og.sim.render()

        textures = _extract_flame_textures(asset_path)
        with og.sim.editing_usd():
            _fix_flame_material(prim_path, textures)
            for child in stage.Traverse():
                path = child.GetPath().pathString
                if child.GetTypeName() == "Mesh" and path.startswith(f"{prim_path}/"):
                    pxr.UsdGeom.PrimvarsAPI(child).CreatePrimvar(
                        "doNotCastShadows", pxr.Sdf.ValueTypeNames.Bool
                    ).Set(True)

        center, size = _world_bounds(prim)
        rotated_size = np.asarray([size[0], size[2], size[1]], dtype=float)
        rotated_center = np.asarray([center[0], -center[2], center[1]], dtype=float)
        height = float(target_height) if target_height is not None else _adaptive_flame_height(obj)
        if not np.isfinite(rotated_size[2]) or rotated_size[2] <= 1e-6:
            raise RuntimeError(f"Invalid flame bounds: center={center.tolist()} size={size.tolist()}")
        scale = float(np.clip(height / rotated_size[2], 1e-5, 1e5))
        scaled_center = rotated_center * scale
        object_lower, object_upper = obj.aabb
        object_lower = np.asarray(object_lower.detach().cpu(), dtype=float)
        object_upper = np.asarray(object_upper.detach().cpu(), dtype=float)
        base = np.asarray(
            [
                (object_lower[0] + object_upper[0]) * 0.5,
                (object_lower[1] + object_upper[1]) * 0.5,
                object_upper[2] + 0.005,
            ],
            dtype=float,
        )
        final_center = base + np.asarray([0.0, 0.0, height * 0.5])
        translation = final_center - scaled_center

        with og.sim.editing_usd():
            xform = pxr.UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(pxr.Gf.Vec3d(*translation))
            xform.AddScaleOp().Set(pxr.Gf.Vec3d(scale, scale, scale))
            xform.AddRotateXOp().Set(90.0)

            light = pxr.UsdLux.SphereLight.Define(stage, light_path)
            light.CreateRadiusAttr(0.01)
            light.CreateIntensityAttr(100000.0)
            light.CreateColorAttr(pxr.Gf.Vec3f(1.0, 0.45, 0.15))
            pxr.UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
                pxr.Gf.Vec3d(float(base[0]), float(base[1]), float(base[2] + 0.03))
            )
        official_disabled = False if keep_official_flow else _disable_official_fire_visual(obj)
        refresh_handles = getattr(og.sim, "update_handles", None)
        if og.sim.is_playing() and callable(refresh_handles):
            refresh_handles()
        for _ in range(USDZ_FLAME_RENDER_WARMUP_FRAMES):
            og.sim.render()
        result.update(
            {
                "ok": True,
                "effect_id": effect_id,
                "target_height": height,
                "prim_path": prim_path,
                "light_path": light_path,
                "base_position": base.tolist(),
                "official_flow_disabled": official_disabled,
            }
        )
    except Exception as exc:
        result["errors"].append({"error": repr(exc)})
        remove_usdz_flame(effect_id)
    return result


def create_usdz_flame_smoke(obj, *, effect_id=None, target_height=None):
    """Combine the packaged flame with a persistent upward official Flow smoke column."""
    effect_id = effect_id or getattr(obj, "name", "fire")
    result = {
        "ok": False,
        **usdz_flame_smoke_record(effect_id=effect_id, target_height=target_height),
        "errors": [],
    }
    flame = create_usdz_flame(
        obj,
        effect_id=effect_id,
        target_height=target_height,
        keep_official_flow=True,
    )
    if not flame.get("ok"):
        result["errors"].append({"error": "usdz_flame_failed", "detail": flame})
        return result
    smoke = configure_on_fire_smoke_column(obj)
    if not smoke.get("ok"):
        remove_usdz_flame(effect_id)
        result["errors"].append({"error": "smoke_column_failed", "detail": smoke})
        return result
    result.update(
        ok=True,
        effect_id=flame["effect_id"],
        target_height=flame["target_height"],
        prim_path=flame["prim_path"],
        light_path=flame["light_path"],
        flame=flame,
        smoke=smoke,
    )
    return result


def remove_usdz_flame(effect_id=None):
    """Remove one flame visual, or every DeltaSG flame when no id is supplied."""
    import omnigibson as og

    stage = og.sim.stage
    if effect_id is None:
        paths = [USDZ_FLAME_ROOT]
    else:
        safe_id = _safe_effect_id(effect_id)
        paths = [
            f"{USDZ_FLAME_ROOT}/flame_{safe_id}",
            f"{USDZ_FLAME_ROOT}/light_{safe_id}",
        ]
    removed = []
    with og.sim.editing_usd():
        for path in paths:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
                removed.append(path)
    refresh_handles = getattr(og.sim, "update_handles", None)
    if removed and og.sim.is_playing() and callable(refresh_handles):
        refresh_handles()
    return removed


def configure_on_fire_smoke_only(
    obj, *, emitter_radius=None, upward_velocity=0.0, smoke_fade=0.5
):
    """Keep the legacy official OnFire Flow effect while removing flame colors."""
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
            radius = (
                min(float(radius_attr.Get()), SMOKE_FLOW_MAX_EMITTER_RADIUS)
                if emitter_radius is None
                else float(emitter_radius)
            )
            radius_attr.Set(radius)
            emitter.GetAttribute("velocity").Set((0.0, 0.0, float(upward_velocity)))
            simulate = og.sim.stage.GetPrimAtPath(f"{mesh.prim_path}/flowSimulate")
            simulate.GetAttribute("densityCellSize").Set(radius * 0.2)
            smoke = og.sim.stage.GetPrimAtPath(f"{mesh.prim_path}/flowSimulate/advection/smoke")
            ray_march = og.sim.stage.GetPrimAtPath(f"{mesh.prim_path}/flowRender/rayMarch")
            smoke.GetAttribute("fade").Set(float(smoke_fade))
            ray_march.GetAttribute("attenuation").Set(5.0)
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
            "smoke_fade": float(smoke_fade),
            "upward_velocity": float(upward_velocity),
            "ray_march_attenuation": 5.0,
            "emitter_enabled": True,
            "emitter_radius": radius,
            "rtx_flow_enabled": True,
            "rtx_flow_composite_enabled": True,
        }
    )
    return result


def configure_on_fire_smoke_column(obj):
    result = configure_on_fire_smoke_only(
        obj,
        emitter_radius=SMOKE_COLUMN_EMITTER_RADIUS,
        upward_velocity=SMOKE_COLUMN_UPWARD_VELOCITY,
        smoke_fade=SMOKE_COLUMN_FADE,
    )
    result["smoke_profile"] = "vertical_column_v1"
    return result
