import os
import sys
import json
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Batch run OmniGibson Level-1 actual outputs across multiple scenes (step2 + step3).")
    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="Scene model names, e.g. Rs_int Benevolence_0 Pomaria_1",
    )
    parser.add_argument("--robot", dest="robot_model", default="fetch", help="Robot model name, e.g. fetch")
    parser.add_argument("--output-root", default="outputs", help="Root directory for all scene outputs")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_py = os.path.join(script_dir, "run.py")

    summary = {
        "ok": True,
        "robot_model": args.robot_model,
        "output_root": args.output_root,
        "results": [],
    }

    for scene_model in args.scenes:
        print(f"\n================ RUNNING SCENE: {scene_model} ================")
        cmd = [
            sys.executable,
            run_py,
            "--scene",
            scene_model,
            "--robot",
            args.robot_model,
            "--output-root",
            args.output_root,
        ]
        ret = subprocess.run(cmd)

        item = {
            "scene_model": scene_model,
            "returncode": int(ret.returncode),
            "output_dir": os.path.join(args.output_root, scene_model),
            "ok": ret.returncode == 0,
        }
        summary["results"].append(item)
        if ret.returncode != 0:
            summary["ok"] = False

    summary_path = os.path.join(args.output_root, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Batch finished ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved to {summary_path}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
