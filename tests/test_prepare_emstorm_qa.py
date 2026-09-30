import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from prepare_emstorm_qa import check_benchmark_isolation, convert_qa, prepare


def export(tmp_path, name, task):
    root = tmp_path / name
    path = root / "images" / task / "step_001_pre" / "robot_primary" / "rgb.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(path)
    row = {"qra_id": f"{task}-q", "task_key": task, "event_id": "step_001_pre",
           "task_type": "B", "question_type": "planning", "category": "bottle",
           "question": "Which action?", "options": ["Pick it up.", "Open it."],
           "answer_index": 0, "answer": "Pick it up.", "reasoning": "The bottle is needed.",
           "images": [{"view": "robot_primary", "rgb": str(path.relative_to(root))}]}
    source = root / "qra.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return source, row


def test_prepares_real_qa_and_independent_benchmark(tmp_path):
    train, _ = export(tmp_path, "raw_train", "Beechwood_0_int__a_1")
    benchmark, _ = export(tmp_path, "raw_bench", "Beechwood_0_int__a_2")
    output = tmp_path / "prepared"
    manifest = prepare(train, benchmark, output)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert rows[0]["scene_id"] == "Beechwood_0_int"
    assert rows[0]["answer"] == "A"
    assert rows[0]["answer_text"] == "Pick it up."
    assert (output / rows[0]["images"]["robot_primary"]).is_file()
    assert "sg_t" not in rows[0]
    assert manifest["unique_rgb_images"] == 2
    assert not manifest["has_scene_graph_supervision"]


@pytest.mark.parametrize("key,value", [("answer_index", -1), ("answer_index", True),
                                       ("options", ["same", "same"])])
def test_rejects_invalid_annotations(tmp_path, key, value):
    source, row = export(tmp_path, "raw", "Beechwood_0_int__a_1")
    row[key] = value
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer_index|options"):
        convert_qa(source, tmp_path / "prepared" / "train.jsonl")


def test_rejects_mismatched_frame_and_path_escape(tmp_path):
    source, row = export(tmp_path, "raw", "Beechwood_0_int__a_1")
    row["event_id"] = "step_002_post"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task and event"):
        convert_qa(source, tmp_path / "train.jsonl")
    row["images"][0]["rgb"] = "../../outside.png"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out-of-root"):
        convert_qa(source, tmp_path / "train.jsonl")


def test_benchmark_cannot_share_training_task_even_with_different_questions():
    with pytest.raises(ValueError, match="task_key overlap"):
        check_benchmark_isolation([{"id": "train-q", "task_key": "task"}],
                                 [{"id": "test-q", "task_key": "task"}])


def test_published_benchmark_duplicates_are_preserved_not_relabeled(tmp_path):
    source, row = export(tmp_path, "bench", "Beechwood_0_int__a_1")
    row["options"] = ["same", "same", "other"]
    row["answer_index"] = 1
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        convert_qa(source, tmp_path / "train.jsonl")
    benchmark = convert_qa(source, tmp_path / "test.jsonl", allow_duplicate_options=True)[0]
    assert benchmark["options"] == row["options"]
    assert benchmark["answer"] == "B"
