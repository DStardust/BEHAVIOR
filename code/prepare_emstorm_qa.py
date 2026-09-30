"""Convert annotated EM-STORM QRA exports to the local answer-only SFT contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def convert_qa(source: Path, output: Path, *, allow_duplicate_options: bool = False) -> list[dict]:
    records = []
    ids = set()
    root = source.parent.resolve()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            identifier = raw["qra_id"]
            task = raw["task_key"]
            scene, separator, instance = task.partition("__")
            if not separator or not scene.endswith("_int") or not instance:
                raise ValueError("task_key must identify an initial scene and a task instance")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("qra_id must be nonempty and unique")
            ids.add(identifier)
            options = raw["options"]
            answer = raw["answer_index"]
            if not isinstance(options, list) or not 2 <= len(options) <= 8 or not all(
                isinstance(option, str) and option.strip() for option in options
            ) or (not allow_duplicate_options and len(set(options)) != len(options)):
                raise ValueError("options must contain 2-8 distinct nonempty strings")
            if type(answer) is not int or not 0 <= answer < len(options):
                raise ValueError("answer_index must identify a listed option")
            if not isinstance(raw["question"], str) or not raw["question"].strip():
                raise ValueError("question must be nonempty text")
            if not isinstance(raw.get("reasoning", ""), str):
                raise ValueError("reasoning must be annotated text, not generated during conversion")
            images = {}
            if not isinstance(raw["images"], list) or not raw["images"]:
                raise ValueError("each question needs at least one view")
            for image in raw["images"]:
                view = image["view"]
                if not isinstance(view, str) or not view or view in images:
                    raise ValueError("view IDs must be nonempty and unique per question")
                relative = Path(image["rgb"])
                file = (root / relative).resolve()
                if relative.is_absolute() or not file.is_relative_to(root) or not file.is_file():
                    raise ValueError(f"missing or out-of-root RGB: {relative}")
                if relative.parts[:3] != ("images", task, raw["event_id"]):
                    raise ValueError("all RGB views must belong to the question's task and event")
                images[view] = os.path.relpath(file, output.parent.resolve())
            records.append({
                "id": identifier, "scene_id": scene, "task_key": task,
                "event_id": raw["event_id"], "task_type": raw["task_type"],
                "question_type": raw["question_type"], "category": raw.get("category", ""),
                "question": raw["question"], "options": options,
                "answer": chr(65 + answer), "answer_index": answer,
                "answer_text": options[answer], "reasoning": raw.get("reasoning", ""),
                "images": images,
            })
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{source}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"no questions in {source}")
    return records


def check_benchmark_isolation(train: list[dict], benchmark: list[dict]) -> None:
    for key in ("id", "task_key"):
        overlap = {row[key] for row in train} & {row[key] for row in benchmark}
        if overlap:
            raise ValueError(f"train/benchmark {key} overlap: {len(overlap)} ({sorted(overlap)[:3]})")


def prepare(train_source: Path, benchmark_source: Path, output: Path) -> dict:
    sources = {"train": train_source, "benchmark": benchmark_source}
    records = {name: convert_qa(source, output / f"{name}.jsonl", allow_duplicate_options=name == "benchmark")
               for name, source in sources.items()}
    check_benchmark_isolation(records["train"], records["benchmark"])
    from PIL import Image

    image_sizes = Counter()
    image_files = {
        (output / image).resolve() for rows in records.values() for row in rows for image in row["images"].values()
    }
    for file in sorted(image_files):
        with Image.open(file) as image:
            image_sizes[str(image.size)] += 1
            image.verify()
    manifest = {
        "schema": "emstorm.annotated_qa.v1", "stage": 0,
        "has_scene_graph_supervision": False,
        "benchmark_isolation": "disjoint question IDs and task instances; not necessarily OOD scenes",
        "unique_rgb_images": len(image_files), "image_sizes": dict(image_sizes),
        "datasets": {name: {
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "questions": len(records[name]),
            "tasks": len({row["task_key"] for row in records[name]}),
            "scenes": sorted({row["scene_id"] for row in records[name]}),
            "question_types": dict(Counter(row["question_type"] for row in records[name])),
            "task_types": dict(Counter(row["task_type"] for row in records[name])),
            "without_reasoning": sum(not row["reasoning"] for row in records[name]),
            "duplicate_option_question_ids": [row["id"] for row in records[name]
                                              if len(set(row["options"])) != len(row["options"])],
        } for name, source in sources.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in records.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-qa", type=Path, required=True)
    parser.add_argument("--benchmark-qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_qa, args.benchmark_qa, args.output), indent=2))


if __name__ == "__main__":
    main()
