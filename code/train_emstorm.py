"""Vision-language QA and scene-graph SFT for EM-STORM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(Path.home() / ".cache" / "unsloth" / "emstorm_compiled"))


STATES = {"on_fire", "dirty", "broken", "wet"}
RELATIONS = {"inside", "on", "held_by"}
SG_RULES = """[Scene Graph Format]
Use one line per entity. The robot is first; other entity IDs are sorted. Slot order is
room -> relation -> state. Robot line: robot | room=<room> | hold=<id|none>.
Relations: inside, on, held_by=robot. A held object
has room=robot and must match robot hold=. States: on_fire, dirty, broken, wet;
omit normal states. Reference anchors by ID without adding an entity line unless
abnormal. Use only listed rooms, entities and anchors. Do not infer unseen facts.
"""


def parse_sg(value: str, *, rooms: set[str], entities: set[str], anchors: set[str]) -> dict[str, dict[str, str]]:
    text = value.strip()
    text = re.sub(r"^<SG(?:_next)?>\s*", "", text)
    text = re.sub(r"\s*</SG(?:_next)?>$", "", text)
    rows: dict[str, dict[str, str]] = {}
    order = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2 or not parts[0]:
            raise ValueError(f"invalid SG line: {line!r}")
        entity = parts[0]
        if entity in rows or entity not in entities:
            raise ValueError(f"duplicate or unlisted SG entity: {entity}")
        slots: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                raise ValueError(f"invalid SG slot: {part!r}")
            key, item = part.split("=", 1)
            if key in slots or not item:
                raise ValueError(f"duplicate or empty SG slot: {part!r}")
            slots[key] = item
        keys = list(slots)
        allowed = ["room", "hold"] if entity == "robot" else ["room", *sorted(RELATIONS), "state"]
        if keys != [key for key in allowed if key in slots] or "room" not in slots:
            raise ValueError(f"invalid SG slot order: {line!r}")
        if entity == "robot":
            if "hold" not in slots or slots["room"] not in rooms:
                raise ValueError("robot requires a listed room and hold slot")
        else:
            if slots["room"] not in rooms | {"robot"}:
                raise ValueError(f"unknown room: {slots['room']}")
            relations = RELATIONS & slots.keys()
            if len(relations) > 1:
                raise ValueError(f"multiple relations for {entity}")
            for relation in relations:
                target = slots[relation]
                if relation == "held_by":
                    if target != "robot" or slots["room"] != "robot":
                        raise ValueError(f"invalid held object: {entity}")
                elif target not in entities | anchors:
                    raise ValueError(f"unknown relation target: {target}")
            if slots["room"] == "robot" and slots.get("held_by") != "robot":
                raise ValueError(f"room=robot requires held_by=robot: {entity}")
            if "state" in slots and (set(slots["state"].split(",")) - STATES):
                raise ValueError(f"unknown state: {slots['state']}")
        rows[entity] = slots
        order.append(entity)
    if order != ["robot", *sorted(entities - {"robot"})]:
        raise ValueError("SG must contain robot first and all listed entities sorted")
    held = [entity for entity, slots in rows.items() if slots.get("held_by") == "robot"]
    robot_hold = rows["robot"]["hold"]
    if held != ([] if robot_hold == "none" else [robot_hold]):
        raise ValueError("robot hold and held_by must agree")
    return rows


def triples(graph: dict[str, dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (entity, key, state if key == "state" else value)
        for entity, slots in graph.items()
        for key, value in slots.items()
        for state in (value.split(",") if key == "state" else [value])
    }


def load_records(path: Path, *, stage: int = 1, allow_duplicate_options: bool = False) -> list[dict]:
    records = []
    ids = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            required = ("id", "scene_id", "images") if stage == 0 else (
                "id", "scene_id", "images", "rooms", "entity_ids", "sg_t", "visible_entity_ids"
            )
            for key in required:
                if key not in row or (key != "visible_entity_ids" and not row[key]):
                    raise ValueError(f"missing {key}")
            if not all(isinstance(row[key], str) and row[key].strip() for key in ("id", "scene_id")):
                raise ValueError("id and scene_id must be nonempty text")
            if row["id"] in ids:
                raise ValueError(f"duplicate id: {row['id']}")
            ids.add(row["id"])
            if not isinstance(row["images"], dict):
                raise ValueError("images must be a view-to-path map")
            for image in row["images"].values():
                file = (path.parent / image).resolve()
                if not file.is_file():
                    raise ValueError(f"missing image: {file}")
            text_fields = ("question", "answer", "reasoning", "sg_next")
            if any(key in row and not isinstance(row[key], str) for key in text_fields):
                raise ValueError("question, answer, reasoning and sg_next must be text when present")
            if "options" in row:
                options = row["options"]
                if not isinstance(options, list) or not 2 <= len(options) <= 8 or not all(
                    isinstance(option, str) and option.strip() for option in options
                ) or (not allow_duplicate_options and len(set(options)) != len(options)):
                    raise ValueError("options must contain 2-8 distinct nonempty strings")
                if row.get("answer") not in [chr(65 + index) for index in range(len(options))]:
                    raise ValueError("MCQ answer must be a listed option letter")
            if stage == 0:
                if row.get("answer") and not row.get("question"):
                    raise ValueError("answered example requires question")
                if not row.get("answer") and not row.get("sg_t"):
                    raise ValueError("missing answer (only SG-only static frames may omit it)")
                records.append(row)
                continue
            if not isinstance(row["rooms"], list):
                raise ValueError("rooms must be a list")
            if not isinstance(row["sg_t"], str) or not isinstance(row["entity_ids"], list):
                raise ValueError("sg_t must be text and entity_ids must be a list")
            if "<SG" in row["sg_t"] or "<SG" in row.get("sg_next", ""):
                raise ValueError("sg_t and sg_next must contain SG rows without wrapper tags")
            if not isinstance(row.get("action", False), bool):
                raise ValueError("action must be boolean")
            entities = set(row["entity_ids"])
            if len(entities) != len(row["entity_ids"]) or "robot" not in entities:
                raise ValueError("entity_ids must be unique and contain robot")
            if not isinstance(row["visible_entity_ids"], list):
                raise ValueError("visible_entity_ids must be a list")
            visible = set(row["visible_entity_ids"])
            if visible - entities:
                raise ValueError("visible_entity_ids must be a subset of entity_ids")
            rooms = set(row["rooms"])
            anchors = set(row.get("anchor_ids", []))
            row["_sg_t"] = parse_sg(row["sg_t"], rooms=rooms, entities=entities, anchors=anchors)
            if row.get("action", False):
                if not row.get("sg_next"):
                    raise ValueError("action example requires sg_next")
                row["_sg_next"] = parse_sg(row["sg_next"], rooms=rooms, entities=entities, anchors=anchors)
            elif row.get("sg_next"):
                raise ValueError("non-action example must not have sg_next")
            if bool(row.get("answer")) != bool(row.get("reasoning")):
                raise ValueError("answer and reasoning must both be present or absent")
            if row.get("answer") and not row.get("question"):
                raise ValueError("joint example requires question")
            records.append(row)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def split_scenes(records: list[dict]) -> dict[str, list[dict]]:
    scenes = sorted({row["scene_id"] for row in records}, key=lambda scene: hashlib.sha256(scene.encode()).digest())
    if len(scenes) < 3:
        raise ValueError("at least three scenes are required for scene-held-out train/val/test")
    n_holdout = max(1, round(len(scenes) * 0.1))
    groups = {
        "test": set(scenes[:n_holdout]),
        "val": set(scenes[n_holdout : 2 * n_holdout]),
    }
    groups["train"] = set(scenes) - groups["test"] - groups["val"]
    return {name: [row for row in records if row["scene_id"] in scene_set] for name, scene_set in groups.items()}


def warmup_entities(row: dict) -> set[str]:
    visible = set(row["visible_entity_ids"]) | {"robot"}
    hold = row["_sg_t"]["robot"]["hold"]
    if hold != "none":
        visible.add(hold)
    return visible


def visible_sg(row: dict) -> str:
    visible = warmup_entities(row)
    return "\n".join(
        line for line in row["sg_t"].strip().splitlines()
        if line.split("|", 1)[0].strip() in visible
    )


def user_text(row: dict, *, stage: int) -> str:
    views = "\n".join(f"View {name}:" for name in row["images"])
    if stage == 0:
        options = row.get("options", [])
        choices = "\nOptions:\n" + "\n".join(
            f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
        ) if options else ""
        output = "Output your final answer in <ANSWER>...</ANSWER>."
        if options:
            output += " The answer must be one option letter, not the option text."
            output += " Bounding boxes use [x1, y1, x2, y2] normalized to 0-1000."
        if row.get("reasoning"):
            output += " Then output the explanation in <REASONING>...</REASONING>."
        return f"Question: {row['question']}{choices}\n{output}"
    elif stage == 1:
        output = "Output <SG> only for the robot, its held object if any, and visible listed entities."
    elif row.get("sg_free_anchor"):
        output = "Output <ANSWER> and <REASONING> only."
    else:
        tags = "<ANSWER>, <REASONING>, <SG> and <SG_next>" if row.get("action") else "<ANSWER>, <REASONING> and <SG>"
        output = f"Output {tags}; include exactly the robot, task-relevant and abnormal listed entities in each SG."
    rules = SG_RULES + "\n" if stage and not (stage == 2 and row.get("sg_free_anchor")) else ""
    return (
        f"{rules}Rooms: {', '.join(row['rooms'])}\n"
        f"Entities: {', '.join(row['entity_ids'])}\n"
        f"Anchors: {', '.join(row.get('anchor_ids', []))}\n"
        f"Question: {row.get('question') or 'Describe the current scene.'}\n{views}\n{output}"
    )


def answer_completion(row: dict) -> str:
    answer = f"<ANSWER>\n{row['answer']}\n</ANSWER>\n"
    if row.get("reasoning"):
        answer += f"<REASONING>\n{row['reasoning']}\n</REASONING>\n"
    return answer


def view_label(name: str) -> str:
    if name == "robot_primary":
        return "Robot's first-person (egocentric) view"
    match = re.fullmatch(r"global_(.+)_\d+", name)
    if match:
        return f"Surveillance camera view of {match.group(1).replace('_', ' ')}"
    return name


def completion_parts(row: dict) -> list[tuple[str, str]]:
    answer = answer_completion(row)
    sg = f"<SG>\n{row['sg_t'].strip()}\n</SG>\n"
    parts = [("answer", answer), ("sg_t", sg)]
    if row.get("action"):
        parts.append(("sg_next", f"<SG_next>\n{row['sg_next'].strip()}\n</SG_next>\n"))
    return parts


def make_rows(row: dict, *, stage: int, processor, lambda_sg: float, lambda_next: float) -> list[dict]:
    if stage != 1 and not row.get("answer"):
        return []
    content = []
    for name in row["images"]:
        content.extend([{"type": "text", "text": f"{view_label(name)}: "}, {"type": "image"}])
    content.append({"type": "text", "text": user_text(row, stage=stage)})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    images = [str((Path(row["_source"]).parent / image).resolve()) for image in row["images"].values()]
    eos = processor.tokenizer.eos_token
    if stage == 1:
        return [{"prompt": prompt, "completion": f"<SG>\n{visible_sg(row)}\n</SG>{eos}",
                 "images": images, "weight": 1.0, "kind": "sg_only"}]
    if stage == 0 or row.get("sg_free_anchor"):
        part = answer_completion(row)
        return [{"prompt": prompt, "completion": part + eos, "images": images,
                 "weight": 1.0, "kind": "answer_anchor"}]
    parts = completion_parts(row)
    weights = {"answer": 1.0, "sg_t": lambda_sg, "sg_next": lambda_next}
    result = []
    prefix = ""
    for index, (kind, text) in enumerate(parts):
        result.append({"prompt": prompt + prefix, "completion": text + (eos if index == len(parts) - 1 else ""),
                       "images": images, "weight": weights[kind], "kind": kind})
        prefix += text
    return result


def build_training_rows(records: list[dict], *, stage: int, processor, lambda_sg: float,
                        lambda_next: float, replay_ratio: float, seed: int) -> list[dict]:
    rows = [item for row in records for item in make_rows(
        row, stage=stage, processor=processor, lambda_sg=lambda_sg, lambda_next=lambda_next
    )]
    if stage == 2 and replay_ratio:
        warmup = [item for row in records for item in make_rows(
            row, stage=1, processor=processor, lambda_sg=lambda_sg, lambda_next=lambda_next
        )]
        count = round(len(rows) * replay_ratio / (1 - replay_ratio))
        rng = random.Random(seed)
        rows += rng.choices(warmup, k=count)
    random.Random(seed).shuffle(rows)
    return rows


def score_outputs(records: list[dict], predictions: list[dict], *, stage: int) -> dict:
    references = {row["id"]: row for row in records}
    predicted_ids = [prediction["id"] for prediction in predictions]
    if len(set(predicted_ids)) != len(predicted_ids) or set(predicted_ids) != set(references):
        raise ValueError("predictions must contain exactly one output per held-out record")
    counts = Counter()
    answer_groups = {"question_type": {}, "task_type": {}}
    for prediction in predictions:
        row = references[prediction["id"]]
        output = prediction["output"]
        if stage == 1:
            visible = warmup_entities(row)
            reference_sg = {entity: slots for entity, slots in row["_sg_t"].items() if entity in visible}
        elif stage == 2:
            visible = set(row["entity_ids"])
            reference_sg = row["_sg_t"]
        else:
            reference_sg = None
        if stage in (0, 2) and row.get("answer"):
            match = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", output, re.S | re.I)
            counts["answer_total"] += 1
            correct = bool(
                match and match.group(1).strip().casefold() == row["answer"].strip().casefold()
            )
            counts["answer_correct"] += correct
            if row.get("options"):
                counts["mcq_total"] += 1
                valid = bool(match and match.group(1).strip().upper() in [
                    chr(65 + index) for index in range(len(row["options"]))
                ])
                counts["mcq_valid"] += valid
                counts["ambiguous_options"] += len(set(row["options"])) != len(row["options"])
                for field, groups in answer_groups.items():
                    if field in row:
                        group = groups.setdefault(row[field], Counter())
                        group.update({"count": 1, "correct": correct, "unparseable": not valid})
        score_sg = stage and not (stage == 2 and row.get("sg_free_anchor"))
        for tag, reference in (("SG", reference_sg if score_sg else None),
                               ("SG_next", row.get("_sg_next") if score_sg and stage == 2 else None)):
            if reference is None:
                continue
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", output, re.S)
            try:
                graph = parse_sg(
                    match.group(1), rooms=set(row["rooms"]), entities=visible,
                    anchors=set(row.get("anchor_ids", [])) | (set(row["entity_ids"]) - visible),
                ) if match else {}
            except ValueError:
                graph = {}
            gold, predicted = triples(reference), triples(graph)
            counts[f"{tag}_tp"] += len(gold & predicted)
            counts[f"{tag}_pred"] += len(predicted)
            counts[f"{tag}_gold"] += len(gold)
            object_gold = {triple for triple in gold if triple[0] != "robot"}
            object_predicted = {triple for triple in predicted if triple[0] != "robot"}
            counts[f"{tag}_object_tp"] += len(object_gold & object_predicted)
            counts[f"{tag}_object_pred"] += len(object_predicted)
            counts[f"{tag}_object_gold"] += len(object_gold)
    result = {}
    if counts["answer_total"]:
        result["answer_accuracy"] = counts["answer_correct"] / counts["answer_total"]
    if counts["mcq_total"]:
        result["answer_format_rate"] = counts["mcq_valid"] / counts["mcq_total"]
        result["unparseable"] = counts["mcq_total"] - counts["mcq_valid"]
        result["ambiguous_option_questions"] = counts["ambiguous_options"]
        for field, groups in answer_groups.items():
            if groups:
                result[f"by_{field}"] = {
                    key: {**group, "accuracy": group["correct"] / group["count"]}
                    for key, group in sorted(groups.items())
                }
    for tag in ("SG", "SG_next"):
        if counts[f"{tag}_gold"]:
            denominator = counts[f"{tag}_pred"] + counts[f"{tag}_gold"]
            result[f"{tag}_f1"] = 2 * counts[f"{tag}_tp"] / denominator
            object_denominator = counts[f"{tag}_object_pred"] + counts[f"{tag}_object_gold"]
            result[f"{tag}_object_f1"] = (
                2 * counts[f"{tag}_object_tp"] / object_denominator if object_denominator else 0.0
            )
    result["count"] = len(predictions)
    return result


def weighted_lm_loss(logits, labels, weights):
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    losses = F.cross_entropy(shifted_logits.view(-1, shifted_logits.shape[-1]), targets.view(-1),
                             ignore_index=-100, reduction="none").view_as(targets)
    per_row = losses.float().sum(dim=1) / (targets != -100).sum(dim=1).clamp_min(1)
    return (per_row * weights).mean()


def make_vision_collator(model, processor):
    from PIL import Image
    from unsloth.trainer import UnslothVisionDataCollator

    # Unsloth 2026.9.6 drops top-level images for prompt-completion records.
    class ImagePathCollator(UnslothVisionDataCollator):
        def _extract_images_for_pc(self, example, p_msgs, c_msgs):
            images = []
            for path in example["images"]:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            return images, [], {"fps": []}

    return ImagePathCollator(model, processor, completion_only_loss=True, resize="max")


def evaluation_subset(records: list[dict], limit: int | None, *, stratified: bool, seed: int) -> list[dict]:
    if not limit or limit >= len(records):
        return records
    if not stratified:
        return records[:limit]
    groups = {}
    for row in records:
        groups.setdefault((row.get("task_type", ""), row.get("question_type", "")), []).append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < limit:
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == limit:
                    break
    return selected


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def adapter_digest(adapter: Path | None) -> str | None:
    if adapter is None:
        return None
    return hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()


def prepare_prediction_output(args, records: list[dict]) -> list[dict]:
    manifest_path = args.predictions.with_suffix(".run.json")
    manifest = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest() if args.benchmark else None,
        "model": str(args.adapter.resolve()) if args.adapter else args.model,
        "adapter_sha256": adapter_digest(args.adapter),
        "stage": args.stage, "split": args.split, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "records_sha256": hashlib.sha256(json.dumps([
            [row["id"], user_text(row, stage=args.stage),
             [(name, view_label(name), image) for name, image in row["images"].items()]]
            for row in records
        ], ensure_ascii=False).encode()).hexdigest(),
        "count": len(records),
    }
    if args.predictions.exists() or manifest_path.exists():
        if not args.resume:
            raise ValueError("prediction output already exists; use --resume or a new path")
        if not args.predictions.is_file() or not manifest_path.is_file():
            raise ValueError("prediction resume requires both JSONL and its .run.json manifest")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("prediction resume data, model, prompt or generation settings differ")
        saved = args.predictions.read_text(encoding="utf-8")
        if saved and not saved.endswith("\n"):
            raise ValueError("saved prediction JSONL has an incomplete final line")
        predictions = [json.loads(line) for line in saved.splitlines() if line.strip()]
        if len(predictions) > len(records) or any(
            not isinstance(prediction, dict) or not isinstance(prediction.get("output"), str)
            or prediction.get("id") != row["id"]
            for row, prediction in zip(records, predictions)
        ):
            raise ValueError("saved predictions must be an ordered, unique prefix of evaluation IDs")
        return predictions
    write_json(manifest_path, manifest)
    args.predictions.touch()
    return []


def predict(args, records: list[dict]) -> list[dict]:
    if not args.adapter and args.stage:
        raise ValueError("SG prediction requires --adapter")
    predictions = prepare_prediction_output(args, records)
    if len(predictions) == len(records):
        print(json.dumps({"predicted": len(predictions), "total": len(records), "resumed": True}), flush=True)
        return predictions
    from unsloth import FastVisionModel
    import torch
    from PIL import Image

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(args.adapter or args.model), load_in_4bit=args.load_in_4bit,
        max_seq_length=args.max_seq_length,
    )
    FastVisionModel.for_inference(model)
    completed = len(predictions)
    counts = Counter()
    if completed and args.stage == 0:
        metrics = score_outputs(records[:completed], predictions, stage=0)
        counts["correct"] = round(metrics.get("answer_accuracy", 0) * completed)
        counts["unparseable"] = metrics.get("unparseable", 0)
    for index in range(completed, len(records)):
        row = records[index]
        started = time.monotonic()
        content = []
        images = []
        for name, image in row["images"].items():
            content.extend([{"type": "text", "text": f"{view_label(name)}: "}, {"type": "image"}])
            with Image.open(Path(row["_source"]).parent / image) as source:
                images.append(source.convert("RGB"))
        content.append({"type": "text", "text": user_text(row, stage=args.stage)})
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            images=[images], text=[prompt], add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        output = processor.batch_decode(tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        prediction = {"id": row["id"], "output": output}
        with args.predictions.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        predictions.append(prediction)
        progress = {"predicted": index + 1, "total": len(records), "id": row["id"],
                    "seconds": round(time.monotonic() - started, 2)}
        if args.stage == 0:
            metrics = score_outputs([row], [prediction], stage=0)
            counts["correct"] += int(metrics["answer_accuracy"] == 1)
            counts["unparseable"] += metrics.get("unparseable", 0)
            progress.update({"correct": counts["correct"], "answer_accuracy": counts["correct"] / (index + 1),
                             "unparseable": counts["unparseable"]})
        print(json.dumps(progress), flush=True)
        write_json(args.predictions.with_suffix(".progress.json"), progress)
        for image in images:
            image.close()
    return predictions


def prepare_training_output(args, splits: dict[str, list[dict]]) -> dict:
    manifest = {
        key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
        for key in ("model", "adapter", "stage", "seed", "load_in_4bit", "max_seq_length", "batch_size",
                    "gradient_accumulation", "epochs", "max_steps", "learning_rate", "lambda_sg",
                    "lambda_next", "replay_ratio")
    }
    manifest.update({
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "adapter_sha256": adapter_digest(args.adapter),
        "scenes": {name: sorted({row["scene_id"] for row in items}) for name, items in splits.items()},
        "loss_contract": "row_weighted_mean_with_trainer_ga_scaling_v1",
    })
    path = args.output / "training_manifest.json"
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        if checkpoint.parent != args.output.resolve() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("resume checkpoint must be a Trainer checkpoint under --output")
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("training resume data, scene split or training settings differ")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("training output is not empty; use --resume-from-checkpoint or a new directory")
        write_json(path, manifest)
    write_json(args.output / "run_config.json", vars(args) | {
        key: str(value) for key, value in vars(args).items() if isinstance(value, Path)
    })
    return manifest


def train(args, records: list[dict], splits: dict[str, list[dict]]) -> None:
    if args.stage == 2:
        if not args.adapter:
            raise ValueError("stage 2 requires --adapter from stage 1")
        metrics = json.loads((Path(args.adapter) / "validation_metrics.json").read_text(encoding="utf-8"))
        if metrics.get("data_sha256") != hashlib.sha256(args.data.read_bytes()).hexdigest():
            raise ValueError("stage 1 validation metrics were produced from different data")
        if metrics["SG_f1"] < args.sg_gate:
            raise ValueError(f"stage 1 SG F1 {metrics['SG_f1']:.3f} is below gate {args.sg_gate:.3f}")
        if metrics["SG_object_f1"] < args.sg_gate:
            raise ValueError(f"stage 1 object SG F1 {metrics['SG_object_f1']:.3f} is below gate {args.sg_gate:.3f}")
    manifest = prepare_training_output(args, splits)
    from unsloth import FastVisionModel
    import torch
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(args.adapter or args.model), load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth", max_seq_length=args.max_seq_length,
    )
    if args.stage in (0, 1):
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=args.seed,
        )
    elif not hasattr(model, "peft_config"):
        raise ValueError("--adapter must contain the stage 1 LoRA adapter, not a base model")
    FastVisionModel.for_training(model)
    # The weighted loss needs logits; Unsloth otherwise returns an empty placeholder.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    for row in records:
        row["_source"] = str(args.data)
    train_rows = build_training_rows(splits["train"], stage=args.stage, processor=processor,
                                     lambda_sg=args.lambda_sg, lambda_next=args.lambda_next,
                                     replay_ratio=args.replay_ratio if args.stage == 2 else 0.0, seed=args.seed)
    if not train_rows:
        raise ValueError("no trainable rows; stages 0 and 2 need answer and reasoning fields")
    print(json.dumps({"stage": args.stage, "train_rows_by_kind": dict(Counter(
        item["kind"] for item in train_rows
    ))}))

    base_collator = make_vision_collator(model, processor)

    def collate(examples):
        weights = torch.tensor([item["weight"] for item in examples], dtype=torch.float32)
        batch = base_collator([{key: value for key, value in item.items() if key not in ("weight", "kind")}
                               for item in examples])
        if not (batch["labels"] != -100).any(dim=1).all():
            raise ValueError("a completion was fully masked or truncated")
        batch["loss_weights"] = weights
        return batch

    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("loss_weights").to(model.device)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_lm_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    class ProgressCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            write_json(args.output / "progress.json", {
                "global_step": state.global_step, "max_steps": state.max_steps,
                "epoch": state.epoch, **(logs or {}),
            })

    trainer = WeightedSFTTrainer(
        model=model, processing_class=processor, data_collator=collate, train_dataset=train_rows,
        callbacks=[ProgressCallback()],
        args=SFTConfig(
            output_dir=str(args.output), per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation, num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate or (5e-5 if args.stage == 2 else 2e-4),
            logging_steps=args.logging_steps, save_strategy="steps" if args.save_steps else "epoch",
            save_steps=args.save_steps or 500, save_total_limit=2 if args.save_steps else None,
            disable_tqdm=True,
            optim="adamw_8bit", seed=args.seed, report_to="none", remove_unused_columns=False,
            dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True}, max_length=None,
        ),
    )
    # This custom loss ignores token counts; Unsloth can overwrite the model's flag.
    trainer.model_accepts_loss_kwargs = False
    print(json.dumps({"model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
                      "accelerator_gradient_accumulation": trainer.accelerator.gradient_accumulation_steps}),
          flush=True)
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint)
                           if args.resume_from_checkpoint else None)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    split_manifest = {
        "data_sha256": manifest["data_sha256"], "scenes": manifest["scenes"],
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "train_metrics.json").write_text(
        json.dumps({**result.metrics, "global_step": trainer.state.global_step,
                    "model": args.model, "stage": args.stage}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    if args.stage == 2:
        (args.output / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"adapter saved to {args.output}; run eval before advancing to the next stage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "train", "predict", "score"))
    parser.add_argument("--data", type=Path, required=True, help="JSONL training contract")
    parser.add_argument("--predictions", type=Path, help="JSONL with id and output for score")
    parser.add_argument("--benchmark", type=Path, help="independent QA JSONL used only with --split test")
    parser.add_argument("--limit", type=int, help="evaluate the first N held-out records (smoke test only)")
    parser.add_argument("--stratified", action="store_true", help="sample --limit across task/question types")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=8192, help="context bound including visual tokens")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=1,
                        help="0=answer-only baseline, 1=SG warmup, 2=joint")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="held-out split for predict/score")
    parser.add_argument("--model", default="unsloth/Qwen3.5-4B")
    parser.add_argument("--adapter", type=Path, help="stage 1 adapter for stage 2")
    parser.add_argument("--output", type=Path, default=Path("outputs/emstorm_train"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="positive optimizer-step limit for a pilot")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, help="save a resumable checkpoint every N optimizer steps; keep two")
    parser.add_argument("--resume-from-checkpoint", type=Path, help="resume training from a checkpoint under --output")
    parser.add_argument("--resume", action="store_true", help="resume matching prediction JSONL without repeating saved IDs")
    parser.add_argument("--learning-rate", type=float, help="default: stage 1 2e-4, stage 2 5e-5")
    parser.add_argument("--lambda-sg", type=float, default=0.3)
    parser.add_argument("--lambda-next", type=float, default=0.3)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--sg-gate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.limit is not None and (args.limit <= 0 or args.command not in ("predict", "score") or args.stage):
        parser.error("--limit must be positive and is only for stage 0 predict/score, not SG stage gates")
    if args.stratified and not args.limit:
        parser.error("--stratified requires --limit")
    if args.max_steps == 0 or args.max_steps < -1 or args.logging_steps <= 0:
        parser.error("--max-steps must be -1 or positive; --logging-steps must be positive")
    if args.save_steps is not None and (args.save_steps <= 0 or args.command != "train"):
        parser.error("--save-steps must be positive and is only for train")
    if args.resume_from_checkpoint and args.command != "train":
        parser.error("--resume-from-checkpoint is only for train")
    if args.resume and args.command != "predict":
        parser.error("--resume is only for predict")
    if min(args.max_new_tokens, args.max_seq_length, args.batch_size, args.gradient_accumulation) <= 0:
        parser.error("token bounds, batch size and gradient accumulation must be positive")
    records = load_records(args.data, stage=args.stage)
    for row in records:
        row["_source"] = str(args.data)
    splits = split_scenes(records)
    if args.benchmark:
        if args.stage or args.command not in ("predict", "score") or args.split != "test":
            parser.error("--benchmark is only for stage 0 test prediction/scoring, never training")
        # Preserve the published benchmark, including its one ambiguous-option question.
        benchmark = load_records(args.benchmark, stage=0, allow_duplicate_options=True)
        train_ids = {row["id"] for row in records}
        train_tasks = {row["task_key"] for row in records if row.get("task_key")}
        if train_ids & {row["id"] for row in benchmark} or train_tasks & {
            row["task_key"] for row in benchmark if row.get("task_key")
        }:
            raise ValueError("benchmark questions/tasks overlap the supplied training data")
        for row in benchmark:
            row["_source"] = str(args.benchmark)
        splits["test"] = benchmark
    print(json.dumps({name: {"scenes": len({row['scene_id'] for row in rows}), "records": len(rows)}
                      for name, rows in splits.items()}, ensure_ascii=False))
    if args.command == "validate":
        return
    if args.command == "train" and args.split != "val":
        parser.error("train always uses the validation split for its stage gate")
    validation = [row for row in splits[args.split] if args.stage == 1 or row.get("answer")]
    validation = evaluation_subset(validation, args.limit, stratified=args.stratified, seed=args.seed)
    if not validation:
        raise ValueError(f"held-out {args.split} scene has no answer-bearing examples")
    if args.command == "predict":
        if not args.predictions:
            parser.error("predict requires --predictions")
        predictions = predict(args, validation)
        metrics = score_outputs(validation, predictions, stage=args.stage)
        metrics["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
        metrics.update({"split": args.split, "subset": bool(args.limit), "seed": args.seed,
                        "model": str(args.adapter or args.model)})
        if args.benchmark:
            metrics["benchmark_sha256"] = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        args.predictions.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
        if args.stage == 1 and args.split == "val":
            (args.adapter / "validation_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
        elif args.stage == 2:
            baseline = args.adapter / "stage1_metrics.json"
            if baseline.is_file():
                stage1 = json.loads(baseline.read_text(encoding="utf-8"))
                for key in ("SG_f1", "SG_object_f1"):
                    if metrics[key] < stage1[key] - 0.05:
                        print(
                            f"WARNING: {key} dropped from {stage1[key]:.3f} to {metrics[key]:.3f}; "
                            "increase replay ratio"
                        )
        return
    if args.command == "score":
        if not args.predictions:
            parser.error("score requires --predictions")
        predictions = [
            json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        metrics = score_outputs(validation, predictions, stage=args.stage)
        print(json.dumps(metrics, indent=2))
        return
    if not (0 <= args.replay_ratio < 1) or min(args.lambda_sg, args.lambda_next) < 0:
        parser.error("loss weights must be nonnegative and replay ratio must be in [0, 1)")
    train(args, records, splits)


if __name__ == "__main__":
    main()
