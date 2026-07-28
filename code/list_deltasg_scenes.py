#!/usr/bin/env python3
"""List locally installed OmniGibson scenes suitable for DeltaSG batches."""

from __future__ import annotations

import argparse

from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("interior", "all"),
        default="interior",
        help="interior selects the BEHAVIOR home interior scenes (*_int).",
    )
    args = parser.parse_args()
    scenes = get_available_behavior_1k_scenes()
    if args.scope == "interior":
        scenes = [scene for scene in scenes if scene.endswith("_int")]
    print("\n".join(sorted(scenes)))


if __name__ == "__main__":
    main()
