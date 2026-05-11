# step1_headless_scene_smoke.py
import sys,os
import time
import json
import traceback
import subprocess

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from omnigibson.macros import gm
gm.ENABLE_OBJECT_STATES = True

import omnigibson as og


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return out
    except Exception:
        return ["NA"]


def safe_count(scene, category):
    try:
        objs = scene.object_registry("category", category)
        return 0 if objs is None else len(objs)
    except Exception:
        return 0


def summarize_scene(scene):
    return {
        "floors": safe_count(scene, "floors") + safe_count(scene, "floor"),
        "walls": safe_count(scene, "walls") + safe_count(scene, "wall"),
        "ceilings": safe_count(scene, "ceilings") + safe_count(scene, "ceiling"),
        "chairs": safe_count(scene, "chair"),
        "tables": safe_count(scene, "table"),
        "cabinets": safe_count(scene, "cabinet"),
        "sofas": safe_count(scene, "sofa"),
    }


def main():
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 60,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_room_types": None,
            "load_room_instances": None,
        },
        "objects": [],
        "robots": [],
    }

    env = None
    t0 = time.time()

    try:
        print("[1/5] creating env...")
        env = og.Environment(configs=cfg)

        print("[2/5] reset...")
        env.reset()

        print("[3/5] warmup steps...")
        for _ in range(120):
            og.sim.step()

        print("[4/5] scene summary...")
        summary = summarize_scene(env.scene)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("GPU mem:", gpu_mem())

        # 稳定性小测试：3轮 step + reset
        cycle_stats = []
        for cycle in range(3):
            t1 = time.time()
            for _ in range(600):
                og.sim.step()
            env.reset()
            cycle_stats.append({
                "cycle": cycle,
                "seconds": round(time.time() - t1, 2),
                "summary": summarize_scene(env.scene),
                "gpu_mem": gpu_mem(),
            })
            print(f"[cycle {cycle}] done:", cycle_stats[-1])

        # 状态导出测试
        state = og.sim.dump_state(serialized=False)
        print("[5/5] sim state dumped, type =", type(state).__name__)

        result = {
            "ok": True,
            "total_seconds": round(time.time() - t0, 2),
            "scene_summary": summary,
            "cycles": cycle_stats,
        }
        with open("step1_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print("saved to step1_result.json")

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    except Exception as e:
        print("FAILED:", repr(e))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()