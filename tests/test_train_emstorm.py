import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from train_emstorm import (  # noqa: E402
    build_training_rows,
    evaluation_subset,
    load_records,
    make_rows,
    prepare_prediction_output,
    prepare_training_output,
    predict,
    score_outputs,
    split_scenes,
    visible_sg,
    weighted_lm_loss,
    view_label,
)


class Processor:
    tokenizer = type("Tokenizer", (), {"eos_token": "<eos>"})()

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT:" + messages[0]["content"][-1]["text"] + "\nASSISTANT:"


def example(scene, *, action=False):
    row = {
        "id": f"{scene}-0",
        "scene_id": scene,
        "question": "Where is the bottle?",
        "images": {"robot": "robot.png", "global_0": "global.png"},
        "rooms": ["kitchen_0", "living_room_0"],
        "entity_ids": ["robot", "bottle_1", "stove_1"],
        "visible_entity_ids": ["bottle_1"],
        "anchor_ids": ["table_1"],
        "sg_t": (
            "robot | room=living_room_0 | hold=none\n"
            "bottle_1 | room=kitchen_0 | on=table_1\n"
            "stove_1 | room=kitchen_0 | state=on_fire"
        ),
        "answer": "Kitchen",
        "reasoning": "The bottle is on the table.",
        "action": action,
    }
    if action:
        row["sg_next"] = (
            "robot | room=kitchen_0 | hold=bottle_1\n"
            "bottle_1 | room=robot | held_by=robot\n"
            "stove_1 | room=kitchen_0 | state=on_fire"
        )
    return row


def dataset(tmp_path, *, action=False):
    for image in ("robot.png", "global.png"):
        (tmp_path / image).write_bytes(b"image")
    path = tmp_path / "train.jsonl"
    rows = [example(f"scene_{index}", action=action) for index in range(3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_split_and_visible_warmup(tmp_path):
    path = dataset(tmp_path)
    rows = load_records(path)
    splits = split_scenes(rows)
    assert {name: len(items) for name, items in splits.items()} == {"test": 1, "val": 1, "train": 1}
    assert len({items[0]["scene_id"] for items in splits.values()}) == 3
    assert "bottle_1" in visible_sg(rows[0])
    assert "stove_1" not in visible_sg(rows[0])


def test_static_frame_needs_no_question(tmp_path):
    path = dataset(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    static = json.loads(lines[0])
    for key in ("question", "answer", "reasoning"):
        static.pop(key)
    static["visible_entity_ids"] = []
    lines[0] = json.dumps(static)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = load_records(path)
    assert visible_sg(rows[0]) == "robot | room=living_room_0 | hold=none"


def test_joint_components_and_replay(tmp_path):
    path = dataset(tmp_path, action=True)
    rows = load_records(path)
    for row in rows:
        row["_source"] = str(path)
    components = make_rows(rows[0], stage=2, processor=Processor(), lambda_sg=0.3, lambda_next=0.2)
    assert [item["kind"] for item in components] == ["answer", "sg_t", "sg_next"]
    assert [item["weight"] for item in components] == [1.0, 0.3, 0.2]
    assert "</REASONING>" in components[1]["prompt"]
    assert "</SG>" in components[2]["prompt"]
    assert components[-1]["completion"].endswith("<eos>")
    training = build_training_rows(rows, stage=2, processor=Processor(), lambda_sg=0.3,
                                   lambda_next=0.2, replay_ratio=0.25, seed=7)
    assert len(training) == 12
    assert sum(item["kind"] == "sg_only" for item in training) == 3
    baseline = make_rows(rows[0], stage=0, processor=Processor(), lambda_sg=0.3, lambda_next=0.2)
    assert len(baseline) == 1
    assert "[Scene Graph Format]" not in baseline[0]["prompt"]
    assert "<SG>" not in baseline[0]["completion"]


def test_rejects_inconsistent_held_relation(tmp_path):
    path = dataset(tmp_path, action=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["sg_next"] = bad["sg_next"].replace("hold=bottle_1", "hold=none")
    lines[0] = json.dumps(bad)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hold and held_by"):
        load_records(path)


def test_rejects_nested_sg_tags_in_source(tmp_path):
    path = dataset(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["sg_t"] = f"<SG>\n{bad['sg_t']}\n</SG>"
    lines[0] = json.dumps(bad)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without wrapper tags"):
        load_records(path)


def test_score_requires_complete_heldout_outputs(tmp_path):
    path = dataset(tmp_path)
    row = split_scenes(load_records(path))["val"][0]
    output = (
        "<ANSWER>Kitchen</ANSWER><REASONING>The bottle is on the table.</REASONING>"
        f"<SG>{row['sg_t']}</SG>"
    )
    metrics = score_outputs([row], [{"id": row["id"], "output": output}], stage=2)
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["SG_f1"] == 1.0
    assert metrics["SG_object_f1"] == 1.0
    with pytest.raises(ValueError, match="exactly one"):
        score_outputs([row], [], stage=2)
    stage1 = score_outputs([row], [{"id": row["id"], "output": f"<SG>{visible_sg(row)}</SG>"}], stage=1)
    assert stage1["SG_f1"] == 1.0
    assert stage1["SG_object_f1"] == 1.0
    baseline = score_outputs([row], [{"id": row["id"], "output": output}], stage=0)
    assert baseline == {"answer_accuracy": 1.0, "count": 1}


def test_action_next_graph_is_scored(tmp_path):
    row = split_scenes(load_records(dataset(tmp_path, action=True)))["val"][0]
    output = (
        f"<ANSWER>{row['answer']}</ANSWER><REASONING>{row['reasoning']}</REASONING>"
        f"<SG>{row['sg_t']}</SG><SG_next>{row['sg_next']}</SG_next>"
    )
    metrics = score_outputs([row], [{"id": row["id"], "output": output}], stage=2)
    assert metrics["SG_next_f1"] == 1.0
    assert metrics["SG_next_object_f1"] == 1.0


def test_sg_free_anchor_uses_answer_only_prompt_and_metrics(tmp_path):
    row = split_scenes(load_records(dataset(tmp_path)))["val"][0]
    row["sg_free_anchor"] = True
    row["_source"] = str(tmp_path / "train.jsonl")
    training = make_rows(row, stage=2, processor=Processor(), lambda_sg=0.3, lambda_next=0.3)
    assert len(training) == 1
    assert "Output <ANSWER> and <REASONING> only." in training[0]["prompt"]
    assert "[Scene Graph Format]" not in training[0]["prompt"]
    assert "<SG>" not in training[0]["completion"]
    metrics = score_outputs(
        [row], [{"id": row["id"], "output": f"<ANSWER>{row['answer']}</ANSWER>"}], stage=2
    )
    assert metrics == {"answer_accuracy": 1.0, "count": 1}


def test_weighted_loss_ignores_prompt_tokens():
    import torch
    import torch.nn.functional as F

    logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                           [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    labels = torch.tensor([[-100, -100, 0], [-100, -100, 0]])
    weights = torch.tensor([1.0, 0.25])
    expected = (F.cross_entropy(logits[0, 1:2], labels[0, 2:3]) +
                0.25 * F.cross_entropy(logits[1, 1:2], labels[1, 2:3])) / 2
    assert torch.allclose(weighted_lm_loss(logits, labels, weights), expected)


def test_row_mean_gradient_accumulation_matches_full_batch():
    import torch

    logits = torch.tensor([[[0.2, -0.1], [0.6, 0.3], [-0.2, 0.4]],
                           [[-0.1, 0.2], [0.3, 0.6], [0.4, -0.2]]], requires_grad=True)
    labels = torch.tensor([[-100, 0, 1], [-100, -100, 0]])
    weights = torch.tensor([1.0, 0.3])
    full = weighted_lm_loss(logits, labels, weights)
    expected = torch.autograd.grad(full, logits)[0]
    microbatch_mean = sum(weighted_lm_loss(logits[index:index + 1], labels[index:index + 1],
                                         weights[index:index + 1]) / 2 for index in range(2))
    actual = torch.autograd.grad(microbatch_mean, logits)[0]
    assert torch.allclose(actual, expected)


def test_real_qa_contract_does_not_require_or_invent_sg(tmp_path):
    (tmp_path / "robot.png").write_bytes(b"image")
    row = {
        "id": "qa-0", "scene_id": "Beechwood_0_int", "task_key": "Beechwood_0_int__a_123",
        "question": "What should the robot do next?", "answer": "B", "reasoning": "",
        "options": ["Open the window.", "Pick up the bottle."],
        "images": {"robot_primary": "robot.png"},
    }
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_records(path, stage=0)[0]
    assert "sg_t" not in loaded
    loaded["_source"] = str(path)
    training = make_rows(loaded, stage=0, processor=Processor(), lambda_sg=0.3, lambda_next=0.3)
    assert "A. Open the window." in training[0]["prompt"]
    assert "B. Pick up the bottle." in training[0]["prompt"]
    assert "Beechwood" not in training[0]["prompt"]
    assert "<SG>" not in training[0]["completion"]
    assert "<REASONING>" not in training[0]["completion"]
    metrics = score_outputs([loaded], [{"id": "qa-0", "output": "<answer>B</answer>"}], stage=0)
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["answer_format_rate"] == 1.0
    assert metrics["count"] == 1
    with pytest.raises(ValueError, match="missing rooms"):
        load_records(path, stage=1)


@pytest.mark.parametrize("answer,options", [
    ("C", ["one", "two"]), ("A", ["one", "one"]), ("A", ["one"]),
])
def test_real_qa_rejects_invalid_choices(tmp_path, answer, options):
    (tmp_path / "robot.png").write_bytes(b"image")
    row = {"id": "qa", "scene_id": "scene", "images": {"robot": "robot.png"},
           "question": "Q", "answer": answer, "options": options}
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="options|MCQ answer"):
        load_records(path, stage=0)


def test_view_labels_reveal_camera_room_but_not_robot_room():
    assert view_label("robot_primary") == "Robot's first-person (egocentric) view"
    assert view_label("global_living_room_1_2") == "Surveillance camera view of living room 1"


def test_stratified_pilot_covers_question_groups_reproducibly():
    rows = [{"id": f"{task}-{question}-{index}", "task_type": task, "question_type": question}
            for task in ("B", "D", "E") for question in ("bbox", "planning") for index in range(10)]
    selected = evaluation_subset(rows, 6, stratified=True, seed=42)
    assert len({(row["task_type"], row["question_type"]) for row in selected}) == 6
    assert selected == evaluation_subset(rows, 6, stratified=True, seed=42)
    assert len(rows) == 60


def test_mcq_metrics_count_invalid_outputs_as_wrong_not_dropped():
    row = {"id": "qa", "answer": "A", "options": ["one", "two"],
           "task_type": "B", "question_type": "planning"}
    metrics = score_outputs([row], [{"id": "qa", "output": "Maybe one or two"}], stage=0)
    assert metrics["count"] == 1
    assert metrics["answer_accuracy"] == 0.0
    assert metrics["answer_format_rate"] == 0.0
    assert metrics["unparseable"] == 1
    assert metrics["by_question_type"]["planning"]["count"] == 1


def test_answer_baseline_preserves_scene_split_with_static_sg_rows(tmp_path):
    path = dataset(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    static = json.loads(lines[0])
    for key in ("question", "answer", "reasoning"):
        static.pop(key)
    lines[0] = json.dumps(static)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    baseline = load_records(path, stage=0)
    assert len(baseline) == 3
    baseline_splits = split_scenes(baseline)
    sg_splits = split_scenes(load_records(path))
    assert {name: [row["id"] for row in rows] for name, rows in baseline_splits.items()} == {
        name: [row["id"] for row in rows] for name, rows in sg_splits.items()
    }
    baseline[0]["_source"] = str(path)
    assert make_rows(baseline[0], stage=0, processor=Processor(), lambda_sg=0.3, lambda_next=0.3) == []


def prediction_args(tmp_path):
    data = dataset(tmp_path)
    return SimpleNamespace(
        data=data, benchmark=None, model="model", adapter=None, stage=0, split="val", seed=3407,
        max_new_tokens=768, max_seq_length=8192, load_in_4bit=True,
        predictions=tmp_path / "predictions.jsonl", resume=False,
    )


def test_prediction_resume_preserves_complete_outputs_without_loading_model(tmp_path):
    args = prediction_args(tmp_path)
    rows = split_scenes(load_records(args.data, stage=0))["val"]
    assert prepare_prediction_output(args, rows) == []
    completed = [{"id": rows[0]["id"], "output": "<ANSWER>Kitchen</ANSWER>"}]
    args.predictions.write_text(json.dumps(completed[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prepare_prediction_output(args, rows)
    args.resume = True
    assert predict(args, rows) == completed


@pytest.mark.parametrize("field,value", [
    ("model", "another-model"), ("max_new_tokens", 256), ("seed", 1), ("load_in_4bit", False),
])
def test_prediction_resume_rejects_mixed_run_settings(tmp_path, field, value):
    args = prediction_args(tmp_path)
    rows = split_scenes(load_records(args.data, stage=0))["val"]
    prepare_prediction_output(args, rows)
    args.resume = True
    setattr(args, field, value)
    with pytest.raises(ValueError, match="settings differ"):
        prepare_prediction_output(args, rows)


def test_prediction_resume_rejects_changed_adapter_weights(tmp_path):
    args = prediction_args(tmp_path)
    args.adapter = tmp_path / "adapter"
    args.adapter.mkdir()
    weights = args.adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first adapter")
    rows = split_scenes(load_records(args.data, stage=0))["val"]
    prepare_prediction_output(args, rows)
    weights.write_bytes(b"second adapter")
    args.resume = True
    with pytest.raises(ValueError, match="settings differ"):
        prepare_prediction_output(args, rows)


@pytest.mark.parametrize("outputs", [
    [{"id": "unknown", "output": "A"}],
    [{"id": "scene_0-0", "output": None}],
    [{"id": "scene_0-0", "output": "A"}, {"id": "scene_0-0", "output": "A"}],
    ["not an output record"],
])
def test_prediction_resume_rejects_invalid_ids_and_records(tmp_path, outputs):
    args = prediction_args(tmp_path)
    rows = load_records(args.data, stage=0)
    prepare_prediction_output(args, rows)
    args.predictions.write_text("".join(json.dumps(row) + "\n" for row in outputs), encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="ordered, unique prefix"):
        prepare_prediction_output(args, rows)


def test_prediction_resume_rejects_incomplete_jsonl_without_destroying_it(tmp_path):
    args = prediction_args(tmp_path)
    rows = load_records(args.data, stage=0)
    prepare_prediction_output(args, rows)
    partial = '{"id":"scene_0-0","output":'
    args.predictions.write_text(partial, encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="incomplete final line"):
        prepare_prediction_output(args, rows)
    assert args.predictions.read_text(encoding="utf-8") == partial


def training_args(tmp_path):
    args = prediction_args(tmp_path)
    args.output = tmp_path / "training"
    args.resume_from_checkpoint = None
    args.batch_size, args.gradient_accumulation, args.epochs, args.max_steps = 1, 8, 1.0, -1
    args.learning_rate = None
    args.lambda_sg, args.lambda_next, args.replay_ratio = 0.3, 0.3, 0.25
    return args


def test_training_resume_checks_data_scene_split_and_config(tmp_path):
    args = training_args(tmp_path)
    splits = split_scenes(load_records(args.data, stage=0))
    manifest = prepare_training_output(args, splits)
    assert manifest["loss_contract"] == "row_weighted_mean_with_trainer_ga_scaling_v1"
    assert (args.output / "run_config.json").is_file()
    with pytest.raises(ValueError, match="not empty"):
        prepare_training_output(args, splits)
    checkpoint = args.output / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":50}', encoding="utf-8")
    args.resume_from_checkpoint = checkpoint
    assert prepare_training_output(args, splits) == manifest
    args.gradient_accumulation = 2
    with pytest.raises(ValueError, match="training settings differ"):
        prepare_training_output(args, splits)
    args.gradient_accumulation = 8
    args.data.write_text(args.data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training settings differ"):
        prepare_training_output(args, splits)


def test_training_resume_rejects_unrelated_checkpoint_directory(tmp_path):
    args = training_args(tmp_path)
    splits = split_scenes(load_records(args.data, stage=0))
    prepare_training_output(args, splits)
    args.resume_from_checkpoint = tmp_path / "another-run"
    args.resume_from_checkpoint.mkdir()
    (args.resume_from_checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="under --output"):
        prepare_training_output(args, splits)
