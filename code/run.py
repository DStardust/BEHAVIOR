import os
import sys
import json
import argparse
import traceback

from api import run_level1


def hard_exit(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main():
    parser = argparse.ArgumentParser(description="Run OmniGibson Level-1 actual outputs for one scene (step2 + step3).")
    parser.add_argument("--scene", dest="scene_model", default="Rs_int", help="Scene model name, e.g. Rs_int")
    parser.add_argument("--robot", dest="robot_model", default="fetch", help="Robot model name, e.g. fetch")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory. Final outputs will be saved to <output-root>/<scene_model>/",
    )
    args = parser.parse_args()

    scene_output_dir = os.path.join(args.output_root, args.scene_model)
    os.makedirs(scene_output_dir, exist_ok=True)

    try:
        result = run_level1(
            scene_model=args.scene_model,
            robot_model=args.robot_model,
            output_dir=scene_output_dir,
        )
        print("\n=== Level-1 actual outputs finished successfully ===")
        print(json.dumps({
            "ok": result.get("ok", False),
            "scene_model": args.scene_model,
            "robot_model": args.robot_model,
            "output_dir": scene_output_dir,
        }, indent=2, ensure_ascii=False))
        hard_exit(0)
    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        hard_exit(1)


if __name__ == "__main__":
    main()
