# EM-STORM EQA / STSG training

This implements an answer-only baseline and two-stage scene-graph-assisted
vision-language SFT. See [the Chinese quick start](../README_EMSTORM_TRAINING.md)
for download, training, monitoring and resume commands.
It trains answers and scene graphs from current robot/global views; it is **not**
low-level VLA control or proof that an expert trajectory is physically valid.
Annotated QRA data is now connected to the answer-only baseline (stage 0).
The supplied archive does not contain STSG or verified visibility labels;
stages 1/2 still require those real labels and do not invent them from answers.

## Environment

Use the separate `emstorm-train` conda environment, not OmniGibson's `behavior`.

```bash
conda create -n emstorm-train python=3.11 -y
PIP_INDEX_URL=https://pypi.org/simple conda run -n emstorm-train python -m pip install \
  -r requirements-emstorm-training.txt
```

Qwen3.5-4B multimodal + Unsloth is the initial baseline. `--model` accepts
another Unsloth-compatible Qwen3.5 vision checkpoint. Do not use API keys for
local SFT. Assign a free GPU with `CUDA_VISIBLE_DEVICES`; no OmniGibson process
is launched by this script. The requirements include SOCKS proxy support for
Hugging Face downloads, and the training script puts Unsloth's generated cache
under `~/.cache/unsloth/` instead of the repository.

## Annotated QRA Data

The supplied ModelScope repository is
`Chonma216/EM-STORM_trainning_set_v0_part1` (keep this spelling).
Download its two archives directly: the generic `MsDataset` image split does
not express the train/benchmark boundary for this repository.

Authenticate interactively, never with a token in source, a command argument,
or a committed `.env` file:

```bash
conda run --no-capture-output -n emstorm-train python -c \
  'import getpass; from modelscope.hub.api import HubApi; HubApi().login(getpass.getpass("ModelScope token: "))'
conda run --no-capture-output -n emstorm-train python -c \
  'from modelscope_hub import HubApi; HubApi().download_repo("Chonma216/EM-STORM_trainning_set_v0_part1", "dataset", local_dir="/path/to/emstorm_qa", max_workers=4)'
```

Extract both archives in that directory with Python's `tarfile` data filter
(`extractall(root, filter="data")`). The resulting input paths are
`data_0918/qra.jsonl` and `bench_v1/qra.jsonl`. The downloaded
`run_vlm_bench.py` is an API-based runner; the local SFT pipeline does not run
it or read its API-key locations.

```bash
conda run --no-capture-output -n emstorm-train python code/prepare_emstorm_qa.py \
  --train-qa /path/to/emstorm_qa/data_0918/qra.jsonl \
  --benchmark-qa /path/to/emstorm_qa/bench_v1/qra.jsonl \
  --output /path/to/emstorm_qa/prepared
conda run --no-capture-output -n emstorm-train python code/train_emstorm.py validate \
  --data /path/to/emstorm_qa/prepared/train.jsonl --stage 0
```

The converter preserves human questions, option order, reasoning and sampled
RGB views. `answer_index` becomes one option letter in `<ANSWER>`; annotated
reasoning follows in `<REASONING>` when present. It validates paths, matching
task/event IDs, image decoding and train/benchmark identity isolation. It does
not resample tasks, regenerate reasoning, or synthesize SG fields. Camera
labels describe the source room; the robot's room is not added to the prompt.
Original source files and all images remain unchanged.

The downloaded data checked on 2026-09-30 contains:

- Training archive: 10,342 questions, 648 tasks, 15 initial scenes.
- Benchmark: 2,777 questions, 350 tasks; no shared question IDs or tasks.
- 13,919 unique RGB files; all decode at 640 x 480.
- Question types: bbox, perception, planning and prediction.
- Training-data scene split: 7,929 train / 1,108 val / 1,305 internal test.
  There are 11 / 2 / 2 scenes respectively. The external benchmark includes
  the same initial scene names but different tasks: it is **not** an OOD-scene
  benchmark. Keep the internal held-out-scene test for OOD experiments.
- One published benchmark question has duplicate option text:
  `b_b57e0a064ce9__step_007_post__planning`. It is preserved, including its
  original correct index; `data_manifest.json` records the ambiguity.

Use a local Qwen3.5-4B checkpoint or the default Hugging Face model. For a
ModelScope copy of the official checkpoint:

```bash
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=4 conda run --no-capture-output -n emstorm-train python -c \
  'from modelscope_hub import HubApi; HubApi().download_repo("Qwen/Qwen3.5-4B", "model", local_dir="/path/to/Qwen3.5-4B", max_workers=2)'
```

Run a bounded pilot on a free GPU, preferably inside tmux:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=16 EVAL_LIMIT=22 bash code/run_emstorm_qa_pilot.sh \
  /path/to/emstorm_qa/prepared/train.jsonl \
  /path/to/emstorm_qa/prepared/benchmark.jsonl \
  /path/to/Qwen3.5-4B outputs/emstorm_qa_pilot
```

The pilot runs base-model validation, 16 LoRA optimizer steps, adapter
validation on the **same** deterministic stratified question subset, and an
independent benchmark subset. It preserves every camera view and uses the
same processor without the collator's automatic downsize. Context is bounded
at 8,192 tokens. `PILOT_PHASE`, per-question progress and per-step losses are
printed; predictions, metrics, adapter, split manifest and run config are
saved. A pilot is a pipeline check, **not** a converged training experiment or
a full benchmark score.

For a full stage-0 run, use `train --epochs 1` without `--max-steps`. Evaluate
the published benchmark without `--limit`:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n emstorm-train python code/train_emstorm.py predict \
  --data /path/to/emstorm_qa/prepared/train.jsonl --stage 0 --adapter /path/to/adapter \
  --benchmark /path/to/emstorm_qa/prepared/benchmark.jsonl --split test \
  --predictions outputs/emstorm_benchmark.jsonl --load-in-4bit
```

Benchmark data is rejected as a training input via `--benchmark`, and tasks
overlapping the supplied train data are rejected. Invalid/unparseable model
outputs count as wrong, not as excluded questions. Metrics include strict
option-letter accuracy, answer-format rate and question/task-type breakdowns.
The local prompt uses structured answer tags rather than the API runner's
`Answer: X` line; compare base and adapted models with the same local prompt,
not directly with a differently prompted API result.

## STSG Input JSONL

One line is one question at one sampled state. Image paths are relative to the
JSONL file, and images must be the *pre-action* robot/global observations. All
views in an example must refer to the same simulation time. Example:

```json
{"id":"scene0_task0_step0_q0","scene_id":"Beechwood_0_int","question":"Where is the bottle?","images":{"robot":"frames/step_000_pre/robot_primary/rgb.png","global_living":"frames/step_000_pre/global/global_living_room_0/rgb.png"},"rooms":["kitchen_0","living_room_0"],"entity_ids":["robot","bottle_1"],"visible_entity_ids":["bottle_1"],"anchor_ids":["table_1"],"sg_t":"robot | room=living_room_0 | hold=none\nbottle_1 | room=kitchen_0 | on=table_1","answer":"Kitchen","reasoning":"The bottle is on the kitchen table.","action":false}
```

- Required: `id`, `scene_id`, `images`, `rooms`, `entity_ids`,
  `visible_entity_ids`, `sg_t`. Joint examples also require `question`.
  `visible_entity_ids` comes from verified
  segmentation/visibility, not an LLM guess. It may be empty.
- For joint SFT: both `answer` and `reasoning`. For SG-only static frames, omit
  both. For an action question, set `action=true` and provide `sg_next` from the
  actual post-action state; otherwise omit `sg_next`.
- `entity_ids` contains robot, task-relevant objects and abnormal objects.
  `anchor_ids` names referenced support/container objects that have no SG row.
  Both graphs must contain exactly the same entities, robot first, other IDs
  sorted. Store SG rows without `<SG>` wrapper tags. Held-object relations must
  agree in both directions.
- `sg_free_anchor=true` selects a small answer-only anchor example in Stage 2.
  This is not inferred automatically.
- A sample must not leak post-action images, answers or GT graphs into the
  model's input. The script uses visibility only for Stage 1 supervision;
  visibility IDs are not placed in the prompt.
- The script validates image existence and SG consistency, then partitions by
  **scene ID**, not by question, into train/val/test (approximately 80/10/10).
  At least three scenes are needed. For serious experiments, keep the resulting
  scene lists frozen across baselines and seeds.

## Run

```bash
conda run -n emstorm-train python code/train_emstorm.py validate --data /path/to/eqa_stsg.jsonl

# Independent answer-only baseline, starting from the base checkpoint.
CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py train \
  --data /path/to/eqa_stsg.jsonl --stage 0 --output outputs/emstorm_baseline --load-in-4bit
CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py predict \
  --data /path/to/eqa_stsg.jsonl --stage 0 --adapter outputs/emstorm_baseline \
  --predictions outputs/emstorm_baseline/val_predictions.jsonl --load-in-4bit

CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py train \
  --data /path/to/eqa_stsg.jsonl --stage 1 --output outputs/emstorm_stage1 --load-in-4bit
CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py predict \
  --data /path/to/eqa_stsg.jsonl --stage 1 --adapter outputs/emstorm_stage1 \
  --predictions outputs/emstorm_stage1/val_predictions.jsonl --load-in-4bit

CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py train \
  --data /path/to/eqa_stsg.jsonl --stage 2 --adapter outputs/emstorm_stage1 \
  --output outputs/emstorm_stage2 --load-in-4bit
CUDA_VISIBLE_DEVICES=1 conda run -n emstorm-train python code/train_emstorm.py predict \
  --data /path/to/eqa_stsg.jsonl --stage 2 --adapter outputs/emstorm_stage2 \
  --predictions outputs/emstorm_stage2/val_predictions.jsonl --load-in-4bit
```

Stage 1 supervises only the visible subgraph, retaining the robot and any held
object for relational consistency. `predict` writes `validation_metrics.json`
beside its Stage 1 adapter. Stage 2 refuses to start when held-out-scene SG
triple F1 **or non-robot-object triple F1** is below `--sg-gate` (default `0.7`),
or when the JSONL differs from
the data used to score Stage 1. Stage 2 reuses the same adapter,
defaults to one-quarter of Stage 1's learning rate, and mixes 25% pure SG replay.
Joint examples use teacher-forced answer, SG_t, and SG_next completions with
weights `1`, `--lambda-sg` (default `0.3`), and `--lambda-next` (default `0.3`).
Because each segment is a separate completion row, the loss is the expected
weighted mean of segment token losses; this re-encodes images for each segment.

`predict` reports answer exact match and micro SG triple F1 on the held-out
scene set. Stage 2 warns if SG F1 falls more than five points below the Stage 1
baseline. Run it on saved checkpoints during longer training; there is no
in-training generation callback yet. A baseline run on real data must verify
the chosen threshold and capture answer-format-specific metrics before treating
this as an experimental result.

After model selection on `val`, repeat `predict` with `--split test` and a
different prediction file. Never use test metrics to choose the SG gate or
hyperparameters. Stage 0 is the answer-only baseline from the same base model;
it uses the same scene split but no SG-format prompt or SG supervision.

For externally produced predictions, use `score --data ... --stage 1|2
--predictions ...`; predictions JSONL contains `{"id":"...","output":"..."}` for
exactly every held-out validation record.

The implementation follows the [Unsloth Qwen3.5 vision notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_%284B%29_Vision.ipynb)
and its [vision collator](https://unsloth.ai/docs/basics/vision-fine-tuning).
The initial environment check on 2026-09-22 passed package/CUDA imports,
dependency checks, and a synthetic two-view action example through the actual
Qwen3.5 processor and vision collator (answer, SG_t, SG_next each had supervised
tokens and image features). That initial check did not load model weights.
The subsequent real-data GPU training and evaluation are recorded below.

## Verified Pilot: 2026-09-30

The first real-data pilot completed on GPU 0 in `emstorm-train`:

- Downloaded and safely extracted both complete annotated archives. Data
  preparation, matching image paths, image decoding and identity isolation
  passed. Data is at `/home2/daiyang/datasets/emstorm_qa_v0_part1/prepared`;
  the official local model is at `/home2/daiyang/models/Qwen3.5-4B`.
- Trained stage 0 for 16 optimizer steps (batch 1, gradient accumulation 2)
  with all robot/global views. This exposed 32 training questions, not a full
  epoch. There were 38,756,352 trainable LoRA parameters; all 344 LoRA B tensors
  changed from their zero initialization. The original Trainer reported finite
  loss: first step 2.557, last step 1.162, mean 1.504 (see the gradient-scaling
  correction below before interpreting these values). Train-loop time including first-step compilation
  was 125.6 seconds; later optimizer steps were approximately 2.4 seconds.
- Reloaded the saved adapter in fresh processes for validation and benchmark
  prediction. The custom row-weighted loss receives real logits and masks
  prompt/image tokens. The pilot did not explicitly override Unsloth's runtime
  `accepts_loss_kwargs` shadow; the full-run launch exposed that scaling issue.
- Full repository unit suite: **472 passed**. Python compilation, shell
  syntax checking and `git diff --check` passed.

| Evaluation | Questions | Correct | Strict Accuracy | Valid Answer Format |
| --- | ---: | ---: | ---: | ---: |
| Base, held-out-scene validation subset | 22 | 6 | 27.3% | 9/22 (40.9%) |
| Adapter, identical validation subset | 22 | 11 | 50.0% | 22/22 (100%) |
| Adapter, independent benchmark subset | 22 | 15 | 68.2% | 22/22 (100%) |

All subsets are deterministic, stratified across task/question types, with
seed 3407 and a 256-new-token budget. Thirteen base responses did not emit a
complete answer tag within that budget; they were counted as wrong. Much of
the early gain is format learning. These small-subset results do **not** prove
convergence or improvement on the full benchmark. The benchmark was not used
to select a checkpoint or tune the pilot. Bbox remains the weakest validation
category (1/6 after the pilot); future full training must evaluate it separately.

Artifacts are under
`code/outputs/emstorm_training_20260930/pilot16/`:

- `adapter/`: adapter, tokenizer/processor, checkpoint-16, `train_metrics.json`,
  `run_config.json` and `split_manifest.json`.
- `base_val.jsonl`, `adapter_val.jsonl`, `adapter_benchmark.jsonl`: raw model
  outputs; each has a matching `.metrics.json` report.
- `../pilot16.log`: phase transitions, training losses and per-question timing.

Next is a full stage-0 training/held-out validation experiment followed by the
full published benchmark. Stage-1/2 STSG training remains gated on receiving
ground-truth SG_t/SG_next and segmentation-derived visible-entity fields;
neither the answers nor model-generated graphs substitute for those labels.

## Full QA Run: 2026-09-30

The corrected full stage-0 run was launched at approximately 15:10 CST on GPU 0,
in tmux session `emstorm_qa_full_20260930_150944`. Its output directory is
`code/outputs/emstorm_training_20260930/full1_150944/`.
Training and full validation have completed; the independent benchmark is
still running as of the 2026-10-01 publication. The pilot is untouched.

```bash
CUDA_VISIBLE_DEVICES=0 bash code/run_emstorm_qa_full.sh \
  /home2/daiyang/datasets/emstorm_qa_v0_part1/prepared/train.jsonl \
  /home2/daiyang/datasets/emstorm_qa_v0_part1/prepared/benchmark.jsonl \
  /home2/daiyang/models/Qwen3.5-4B \
  code/outputs/emstorm_training_20260930/full1_150944
```

This command is recorded for reproduction in a **new** output directory, not
for launching a second process against the currently running directory.

- Fresh base-model LoRA, stage 0, one full epoch on 7,929 training questions;
  batch 1, gradient accumulation 8, seed 3407, LR 2e-4, 4-bit base weights.
- Frozen held-out-scene validation: all 1,108 questions.
- Independent benchmark: all 2,777 questions, evaluated after training.
- No `--limit` or `--max-steps`; all original RGB views are retained.
  Generation budget is 768 new tokens, not the pilot's 256. Do not compare
  full-run accuracy directly with the 22-question pilot as a controlled gain.
- Two rolling Trainer checkpoints, saved every 50 optimizer steps with
  optimizer, scheduler and RNG state. Training progress is logged every 10
  steps and saved in `adapter/progress.json`.
- Every generated answer is appended immediately to its prediction JSONL.
  `*.progress.json` reports completed/total, strict running accuracy and
  invalid-answer count. Final `*.metrics.json` files are written only after
  every question in the corresponding evaluation is predicted.
- `status.json` distinguishes train, validation, benchmark and complete;
  a nonzero subprocess exit sets state to `failed`. A `FULL_COMPLETE` log
  line means all three phases returned successfully, not merely generation.

Monitor from the repository root:

```bash
tail -f code/outputs/emstorm_training_20260930/full1_150944/full.log
watch -n 15 cat code/outputs/emstorm_training_20260930/full1_150944/status.json
watch -n 15 cat code/outputs/emstorm_training_20260930/full1_150944/adapter/progress.json
```

After training, the evaluation progress files are `adapter_val.progress.json`
and `adapter_benchmark.progress.json` in that same directory. Those running
accuracy numbers are **partial results**, not full validation/benchmark scores.

To resume interrupted training, run `train` with the original arguments and
add `--resume-from-checkpoint <output>/adapter/checkpoint-N`. Keep `--output`
pointing at that adapter directory. Resume checks the data hash, scene split
and training settings before loading a checkpoint. It does not restart the
optimizer schedule from zero. To resume either evaluation, repeat its `predict`
command with `--resume` and the same prediction path, model, data and token
budget. The saved run manifest checks these identities and the adapter weights;
only a matching ordered prefix of evaluation IDs may be continued. An
incomplete final JSONL line is reported as an error, never silently discarded.
The older pilot outputs predate these resume manifests; preserve them as-is.

The first launch (`full1_145951`) was deliberately stopped after 40 steps. Its
raw logged loss exposed a custom-loss integration bug: Unsloth 2026.9.7
shadows `accepts_loss_kwargs=True` on compiled model wrappers even though
Qwen3.5 declares it False. Our custom loss uses row means and ignores the
supplied token count, so Trainer must still divide by gradient accumulation.
The corrected code explicitly sets `trainer.model_accepts_loss_kwargs=False`
**after** Trainer construction and records this loss contract in its training
manifest. Old checkpoints with no matching contract cannot be resumed into
the corrected run. The pilot's predictions/scores remain real, but its logged
losses/gradient norms were accumulation-scaled and must not be compared with
the corrected per-row-mean loss. Neither interrupted full-run launch nor
the small pilot is a final full benchmark result.

Checks for this full-run launch: **486 repository tests passed**; Python
compilation, shell syntax and `git diff --check` passed. The new tests cover
resume identity, duplicate/unknown IDs, incomplete JSONL, changed adapter
weights, unrelated checkpoints, training-data/config changes and equivalence
of full-batch versus accumulated row-mean gradients. Full-run
scores and runtime will be recorded only after the job completes.

Launch verification at 15:21 CST: the corrected run reached 60/992 steps
without NaNs or runtime errors. Mean logged loss fell from 1.124 over the first
10 steps to 0.386 over steps 51-60. `checkpoint-50` was inspected: optimizer,
scheduler, RNG state and `trainer_state.json` (50/992, epoch 0.05045) are
present; all adapter weights are finite and all 344 LoRA B tensors are
nonzero. This verifies actual updates and checkpoint persistence, **not**
generalization or completion of the full epoch. A separate two-question GPU-2
checkpoint-reload/streaming smoke test completed successfully and is kept as
`smoke_ckpt50_val.*`. Both responses used valid answer tags, but both answers
were wrong (0/2); this is not a model-quality acceptance result. Repeating the
same prediction command with `--resume` returned the two saved outputs without
loading the model or adding records. This smoke test is not part of the
1,108-question full validation result and does not select a checkpoint for
benchmark evaluation. The main run subsequently reached 110/992 steps with
finite loss (0.310 over steps 101-110); final full-run metrics are still pending.

### Completed Training and Validation

Training finished at 17:35 CST on 2026-09-30: 992 optimizer steps, epoch 1.0,
7,929 training questions, mean row-weighted loss 0.23620. The training loop
took 8,540.9 seconds (2 h 22 min); model setup is additional. The final adapter
and processor are saved in `full1_150944/adapter/`.

Full held-out-scene validation finished at 20:39 CST. Its saved report is
`full1_150944/adapter_val.metrics.json` (`subset=false`, count 1,108).

| Validation Question Type | Correct / Total | Strict Accuracy |
| --- | ---: | ---: |
| bbox | 760 / 768 | 98.96% |
| perception | 97 / 127 | 76.38% |
| planning | 180 / 192 | 93.75% |
| prediction | 21 / 21 | 100.00% |
| Overall | 1,058 / 1,108 | 95.49% |

Answer-format rate is 100%; invalid answers are not excluded from scoring.
Validation and independent benchmark manifests identify the same final adapter
weights, seed 3407, 768-new-token budget, 8,192-token context and 4-bit loading.
These are QA scores, **not** expert-trajectory or robot-control success rates.
There is no full base-model control run yet; the pilot's 22-question/256-token
score must not be used to claim a controlled full-run improvement.

The independent benchmark started automatically after validation and remains
running. A diagnostic snapshot of its first 467/2,777 questions scored 283/467
(60.60%), with valid answer tags for every response. Its per-type snapshot is
bbox 117/261 (44.83%), perception 35/69 (50.72%), planning 106/112 (94.64%) and
prediction 25/25 (100%). This is an **ordered, incomplete prefix**, not a final
or representative benchmark result. The gap to validation is most apparent
in bbox/perception; its cause is not established by these numbers. Do not
change the model, prompts, original labels or generation parameters midway
through this evaluation, or use benchmark failures to tune this checkpoint.
Wait for all 2,777 predictions before reporting a full benchmark score.
