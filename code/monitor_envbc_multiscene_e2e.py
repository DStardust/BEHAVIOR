"""Print live generation and oracle-expert rates for an Env-B/C scene sweep."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def checkpoint_counts(path: Path):
    data = load_json(path)
    return int(data.get("run_counter") or 0), len(data.get("successful_samples") or [])


def main():
    root = Path(sys.argv[1])
    config = load_json(root / "config.json")
    requested = int(config.get("envb_requested_per_scene", 3)) + int(
        config.get("envc_requested_per_scene", 8)
    )
    try:
        scenes = [line.strip() for line in (root / "scenes.txt").read_text().splitlines() if line.strip()]
    except OSError:
        scenes = []

    totals = Counter()
    rows = []
    failure_stages = Counter()
    for scene in scenes:
        scene_root = root / scene
        b_attempts, b_ok = checkpoint_counts(
            scene_root / "generation" / "envB_fire" / "checkpoint.json"
        )
        c_attempts, c_ok = checkpoint_counts(
            scene_root / "generation" / "envC_all" / "checkpoint.json"
        )
        results = [load_json(path) for path in (scene_root / "expert").rglob("expert_result.json")]
        accepted = sum(item.get("accepted") is True for item in results)
        for item in results:
            if item.get("accepted") is not True:
                stage = ((item.get("rejection") or {}).get("stage") or "unknown")
                failure_stages[stage] += 1
        try:
            state = (scene_root / "state").read_text().strip()
        except OSError:
            state = "PENDING"
        gen_ok = b_ok + c_ok
        attempts = b_attempts + c_attempts
        e2e_rate = accepted / requested if requested else 0.0
        rows.append(
            (scene, state, gen_ok, attempts, len(results), accepted, e2e_rate)
        )
        totals.update(
            requested=requested,
            gen_ok=gen_ok,
            attempts=attempts,
            expert_done=len(results),
            accepted=accepted,
        )

    print(f"root: {root.resolve()}")
    print("scene                    state      gen_ok/raw_try  expert  accepted  e2e")
    print("-" * 79)
    for scene, state, gen_ok, attempts, expert_done, accepted, rate in rows:
        print(
            f"{scene:<24} {state:<10} {gen_ok:>2}/{requested:<2} ({attempts:>2})  "
            f"{expert_done:>2}/{gen_ok:<2}  {accepted:>2}       {rate:>6.1%}"
        )
    total_requested = totals["requested"]
    total_gen_rate = totals["gen_ok"] / total_requested if total_requested else 0.0
    total_e2e_rate = totals["accepted"] / total_requested if total_requested else 0.0
    expert_rate = (
        totals["accepted"] / totals["expert_done"] if totals["expert_done"] else 0.0
    )
    print("-" * 79)
    print(
        f"TOTAL requested={total_requested} generated={totals['gen_ok']} ({total_gen_rate:.1%}) "
        f"raw_attempts={totals['attempts']} expert={totals['expert_done']} "
        f"accepted={totals['accepted']} expert_rate={expert_rate:.1%} "
        f"e2e={total_e2e_rate:.1%}"
    )
    if failure_stages:
        print("expert_failure_stages:", dict(failure_stages))


if __name__ == "__main__":
    main()

