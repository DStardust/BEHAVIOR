#!/usr/bin/env python3
"""Offline re-color (relight) of already-rendered DeltaSG camera frames.

Re-applies the expert-stage lighting randomization (Dome-light intensity +
color temperature) to the ``rgb.png`` frames that were already rendered under
the old fixed lighting, without re-running the simulator. See
``deltasg_visual_variation`` for the shared parameters; this is the offline
half of the two-path design (the physical re-render half lives in
``rerender_deltasg_camera.sh`` + ``run_deltasg_expert.py --visual-seed``).

Effects:
  * ``lighting``  — intensity in [1800, 3800] and warm-tint color jitter on the
    base (1.07, 0.85, 0.61). Applied as a per-channel exposure/warmth gain to
    each stored rgb frame. First-order approximation of a re-render: shadows,
    highlights and camera geometry are unchanged.
  * ``fov``       — NOT available offline (changing field-of-view requires
    re-rendering); requesting it only warns and skips.

Segmentation masks (``seg_instance.png`` / ``seg_semantic.png``) and action
``.npy`` files are copied unchanged (hard-linked where possible) so each output
instance stays self-contained.

Usage:
    python code/recolor_deltasg_camera.py \
        --input-root  code/outputs/backup_1004/1004_part0 \
        --output-root code/outputs/1004_randomized/1004_part0 \
        --num-variants 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from deltasg_visual_variation import draw_variant, lighting_gain

SCHEMA_VERSION = "deltasg_recolor_manifest.v1"


def _instance_dir_of(rgb_path: Path) -> Path | None:
    """Return the run-id directory that owns a ``.../frames/.../rgb.png``."""
    for parent in rgb_path.parents:
        if parent.name == "frames":
            return parent.parent
    return None


def discover_instances(input_root: Path):
    """Yield ``(instance_dir, [rgb_paths])`` grouped by run-id directory."""
    groups: dict[Path, list[Path]] = {}
    for rgb in sorted(input_root.rglob("rgb.png")):
        instance = _instance_dir_of(rgb)
        if instance is not None:
            groups.setdefault(instance, []).append(rgb)
    return [(instance, groups[instance]) for instance in sorted(groups)]


def variant_target_dir(instance_dir: Path, input_root: Path, output_root: Path,
                       variant: int, num_variants: int) -> Path:
    rel = instance_dir.relative_to(input_root)
    if num_variants == 1 or variant == 0:
        return output_root / rel
    return output_root / rel.parent / f"{rel.name}__v{variant}"


def recolor_rgb(src: Path, dst: Path, gain):
    import numpy as np
    from PIL import Image

    im = Image.open(src)
    mode = im.mode
    arr = np.asarray(im).astype(np.float32)
    im.close()
    if mode in ("RGB", "RGBA"):
        arr[..., :3] = np.clip(arr[..., :3] * np.asarray(gain, dtype=np.float32), 0.0, 255.0)
    elif mode == "L":
        arr = np.clip(arr * float(np.mean(gain)), 0.0, 255.0)
    out = np.round(arr).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode).save(dst)


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def update_expert_result(instance_dir: Path, target_dir: Path, run_id: str,
                         variant: int, num_variants: int, params: dict) -> None:
    """Copy the expert result and stamp the applied visual variation."""
    src = instance_dir / "expert_result.json"
    dst = target_dir / "expert_result.json"
    if not src.is_file():
        # Instance has frames but no expert result; emit a sidecar instead.
        sidecar = target_dir / "visual_variation.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(params, indent=2), encoding="utf-8")
        return
    record = json.loads(src.read_text(encoding="utf-8"))
    if num_variants > 1 and variant > 0:
        record["run_id"] = f"{run_id}__v{variant}"
    record["visual_variation"] = params
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def process_instance(args):
    """Worker: recolor one instance for every variant. Returns manifest entries."""
    (instance_dir_str, input_root_str, output_root_str, num_variants, effects) = args
    instance_dir = Path(instance_dir_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)
    run_id = instance_dir.name
    rgb_paths = sorted(
        p for p in instance_dir.rglob("rgb.png") if _instance_dir_of(p) == instance_dir
    )

    entries = []
    for variant in range(num_variants):
        record, gain = draw_variant(run_id, variant, with_fov=False)
        target_dir = variant_target_dir(instance_dir, input_root, output_root,
                                        variant, num_variants)

        for rgb in rgb_paths:
            rel = rgb.relative_to(instance_dir)
            recolor_rgb(rgb, target_dir / rel, gain)

        # Copy segmentation masks and any other non-rgb files under frames/.
        frames_src = instance_dir / "frames"
        for src in sorted(frames_src.rglob("*")):
            if src.is_file() and src.name != "rgb.png":
                link_or_copy(src, target_dir / src.relative_to(instance_dir))

        # Copy actions/ unchanged.
        actions_src = instance_dir / "actions"
        if actions_src.is_dir():
            for src in sorted(actions_src.rglob("*")):
                if src.is_file():
                    link_or_copy(src, target_dir / src.relative_to(instance_dir))

        params = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "variant": variant,
            "method": "postprocess_recolor",
            "source_run_id": run_id,
            "effects": effects,
            "seed": record["seed"],
            "lighting": {
                "applied": True,
                "source": "postprocess_recolor",
                "intensity": record["intensity"],
                "color": record["color"],
            },
            "camera_horizontal_fov_deg": {},
            "gain": [round(float(g), 6) for g in gain],
        }
        update_expert_result(instance_dir, target_dir, run_id, variant,
                             num_variants, params)
        entries.append({
            "run_id": run_id,
            "variant": variant,
            "output_instance": str(target_dir),
            "seed": record["seed"],
            "intensity": record["intensity"],
            "color": record["color"],
            "gain": params["gain"],
            "num_rgb_frames": len(rgb_paths),
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-variants", type=int, default=1,
                        help="Number of deterministically re-colored variants per instance.")
    parser.add_argument("--effects", default="lighting",
                        help="Comma list. 'lighting' is implemented; 'fov' warns and is skipped offline.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N instances (0=all).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_variants < 1:
        parser.error("--num-variants must be >= 1")
    effects = [e.strip() for e in args.effects.split(",") if e.strip()]
    if "fov" in effects:
        print("[recolor] fov is a render-time effect and cannot be applied offline; skipping it.",
              file=sys.stderr)

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    instances = discover_instances(input_root)
    if args.limit:
        instances = instances[: args.limit]
    print(f"[recolor] discovered {len(instances)} instances under {input_root}", flush=True)

    tasks = [
        (str(inst), str(input_root), str(output_root), args.num_variants, effects)
        for inst, _ in instances
    ]

    if args.dry_run:
        for task in tasks:
            inst = Path(task[0])
            total_rgb = sum(
                1 for p in inst.rglob("rgb.png") if _instance_dir_of(p) == inst
            )
            print(f"[dry-run] {inst.relative_to(input_root)}  variants={args.num_variants}  rgb_frames={total_rgb}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    workers = args.workers or min(8, os.cpu_count() or 1)
    entries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(process_instance, tasks, chunksize=1):
            entries.extend(batch)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_variants": args.num_variants,
        "effects": effects,
        "num_instances": len(instances),
        "num_outputs": len(entries),
        "entries": entries,
    }
    manifest_path = output_root / "recolor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"[recolor] wrote {len(entries)} outputs; manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
