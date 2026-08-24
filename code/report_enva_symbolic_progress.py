"""Print compact generation and symbolic-expert progress for an Env-A scene sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _strict_symbolic(result, input_path):
    backend = result.get("backend") or {}
    steps = result.get("steps") or []
    try:
        recorded_input = Path(str(result.get("input") or "")).resolve()
    except (OSError, RuntimeError):
        return False
    return (
        recorded_input == input_path.resolve()
        and result.get("accepted") is True
        and result.get("qa_eligible") is True
        and backend.get("name") == "oracle_symbolic"
        and backend.get("generation_profile_verified") is True
        and bool(steps)
        and all(step.get("pre_observation") and step.get("post_observation") for step in steps)
    )


def _scene_progress(root, scene):
    scene_root = root / scene
    generation_root = scene_root / "generation"
    generated = {}
    for path in generation_root.glob("online_env*.json"):
        item = _read_json(path) or {}
        task = (
            ((item.get("task_environment") or {}).get("task") or item.get("task") or {}).get(
                "primary_behavior_task"
            )
            or ""
        )
        if task and item.get("ok") is True:
            generated[task] = path.resolve()

    attempted = 0
    try:
        log = (scene_root / "generation.log").read_text(encoding="utf-8", errors="replace")
        attempted = len(set(re.findall(
            r"\[online-deltasg\] run \d+/\d+ task=([^\s]+)", log,
        )))
    except OSError:
        pass

    expert_done = 0
    strict_ok = 0
    for path in (scene_root / "expert").rglob("expert_result.json"):
        result = _read_json(path) or {}
        task = result.get("task_name") or ""
        input_path = generated.get(task)
        if input_path is None:
            continue
        expert_done += 1
        strict_ok += int(_strict_symbolic(result, input_path))

    running_marker = scene_root / ".e2e_running"
    report_path = scene_root / "e2e_report.json"
    summary_path = generation_root / "summary.json"
    generation_log = scene_root / "generation.log"
    if running_marker.is_file():
        try:
            stage = running_marker.read_text(encoding="utf-8").strip() or "RUN"
        except OSError:
            stage = "RUN"
    else:
        report = _read_json(report_path)
        report_mtime = report_path.stat().st_mtime if report_path.is_file() else -1.0
        if report is not None and summary_path.is_file() and summary_path.stat().st_mtime > report_mtime:
            stage = "EXP"
        elif report is not None and generation_log.is_file() and generation_log.stat().st_mtime > report_mtime:
            stage = "GEN"
        elif report is not None:
            stage = "DONE" if report.get("passed") is True else "FAIL"
        elif summary_path.is_file():
            stage = "EXP"
        elif generation_log.is_file():
            stage = "PART"
        else:
            stage = "PEND"
    return {
        "scene": scene,
        "stage": stage,
        "attempted": attempted,
        "generated": len(generated),
        "expert_done": expert_done,
        "strict_ok": strict_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--scenes-file", default="code/configs/env_a_scenes.txt")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    scenes = [
        line.strip()
        for line in Path(args.scenes_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = [_scene_progress(root, scene) for scene in scenes]
    print(f"{'SCENE':22} {'STATE':5} {'GEN':>7} {'EXPERT':>9} {'OK':>4} {'RATE':>7}")
    for row in rows:
        rate = (
            f"{100.0 * row['strict_ok'] / row['expert_done']:.1f}%"
            if row["expert_done"]
            else "-"
        )
        print(
            f"{row['scene']:22} {row['stage']:5} "
            f"{row['generated']:2}/{row['attempted']:<2} "
            f"{row['expert_done']:2}/{row['generated']:<2} "
            f"{row['strict_ok']:4} {rate:>7}"
        )
    generated = sum(row["generated"] for row in rows)
    attempted = sum(row["attempted"] for row in rows)
    expert_done = sum(row["expert_done"] for row in rows)
    strict_ok = sum(row["strict_ok"] for row in rows)
    expert_rate = 100.0 * strict_ok / expert_done if expert_done else 0.0
    e2e_rate = 100.0 * strict_ok / generated if generated else 0.0
    completed_scenes = sum(row["stage"] in {"DONE", "FAIL"} for row in rows)
    print(
        f"TOTAL scenes={completed_scenes}/{len(rows)} gen={generated}/{attempted} "
        f"expert={expert_done}/{generated} strict_ok={strict_ok} "
        f"expert_rate={expert_rate:.1f}% e2e_rate={e2e_rate:.1f}%"
    )


if __name__ == "__main__":
    main()
