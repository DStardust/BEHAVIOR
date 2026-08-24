"""Render a minimal official OmniGibson OnFire / smoke-only Flow probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm

gm.ENABLE_OBJECT_STATES = True
gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = True

import numpy as np
import omnigibson as og
import torch as th
from PIL import Image
from omnigibson import object_states
from omnigibson.objects import DatasetObject

from deltasg_visual_effects import (
    SMOKE_FLOW_RENDER_WARMUP_FRAMES,
    SMOKE_FLOW_WARMUP_STEPS,
    configure_on_fire_smoke_only,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    env = og.Environment(configs={"scene": {"type": "Scene"}})
    og.sim.stop()
    obj = DatasetObject(
        name="deltasg_smoke_probe",
        category="plywood",
        model="fkmkqa",
        abilities={"flammable": {}},
    )
    env.scene.add_object(obj)
    obj.set_position_orientation(position=th.tensor([0.0, 0.0, 0.7]), frame="scene")
    og.sim.play()
    if not obj.states[object_states.OnFire].set_value(True):
        raise RuntimeError("failed to set official OnFire state")
    visual = None
    if args.smoke_only:
        visual = configure_on_fire_smoke_only(obj)
        if not visual.get("ok"):
            raise RuntimeError(f"failed to configure smoke-only Flow: {visual}")
    else:
        obj.update_visuals()

    viewer = og.sim.viewer_camera
    viewer.add_modality("rgb")
    viewer.set_position_orientation(
        position=np.asarray([2.5, 0.0, 1.1], dtype=np.float32),
        orientation=np.asarray([0.0, 0.7071068, 0.0, 0.7071068], dtype=np.float32),
    )
    for _ in range(SMOKE_FLOW_WARMUP_STEPS):
        og.sim.step()
    for _ in range(SMOKE_FLOW_RENDER_WARMUP_FRAMES):
        og.sim.render()
    obs, _ = viewer.get_obs()
    rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    print(
        {
            "output": str(output),
            "on_fire": bool(obj.states[object_states.OnFire].get_value()),
            "smoke_only": bool(args.smoke_only),
            "visual": visual,
        },
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
