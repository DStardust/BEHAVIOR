# EM-STORM 训练中文使用说明

训练代码位于 `training/emstorm-qa-sft` 分支，与 DeltaSG 生成代码的日常维护分开。
这里只做多视角视觉问答和场景图监督训练，不是机器人动作控制训练。
不需要启动 OmniGibson，也不需要在线 LLM API key。

## 环境和代码

在单独目录获取训练分支，不要在正在生成或评测的目录切分支：

```bash
git clone --branch training/emstorm-qa-sft https://github.com/DStardust/BEHAVIOR.git BEHAVIOR_training
cd BEHAVIOR_training
conda create -n emstorm-train python=3.11 -y
conda run -n emstorm-train python -m pip install -r requirements-emstorm-training.txt
```

需要支持 CUDA 的 NVIDIA GPU。依赖版本来自已完成真实训练的环境，
不要安装进 OmniGibson 的 `behavior` 环境。训练输出建议放在被 Git 忽略的
`code/outputs/`；数据、基础模型放在仓库外。

## 下载和准备数据

已标注数据集是 `Chonma216/EM-STORM_trainning_set_v0_part1`，注意仓库名称拼写。
需要有权访问该数据集的 ModelScope 账号。以下登录会隐藏输入，不把 token 写进
代码、Git 或 shell 命令历史：

```bash
conda run --no-capture-output -n emstorm-train python -c \
  'import getpass; from modelscope.hub.api import HubApi; HubApi().login(getpass.getpass("ModelScope token: "))'

export DATA="$HOME/datasets/emstorm_qa_v0_part1"
export MODEL="$HOME/models/Qwen3.5-4B"
conda run --no-capture-output -n emstorm-train python -c \
  'import os; from modelscope_hub import HubApi; HubApi().download_repo("Chonma216/EM-STORM_trainning_set_v0_part1", "dataset", local_dir=os.environ["DATA"], max_workers=4)'
```

下载整个数据集仓库，使用 `data_0918.tar.gz` 和 `bench_v1.tar.gz`，不要将通用
`MsDataset` 图像 split 当作训练/benchmark 划分。用安全的 tar 数据过滤器解压：

```bash
conda run --no-capture-output -n emstorm-train python -c '
import os
import tarfile
from pathlib import Path
root = Path(os.environ["DATA"])
for name in ("data_0918.tar.gz", "bench_v1.tar.gz"):
    with tarfile.open(root / name, "r:gz") as archive:
        archive.extractall(root, filter="data")
'
conda run --no-capture-output -n emstorm-train python code/prepare_emstorm_qa.py \
  --train-qa "$DATA/data_0918/qra.jsonl" \
  --benchmark-qa "$DATA/bench_v1/qra.jsonl" --output "$DATA/prepared"
conda run --no-capture-output -n emstorm-train python code/train_emstorm.py validate \
  --data "$DATA/prepared/train.jsonl" --stage 0

conda run --no-capture-output -n emstorm-train python -c \
  'import os; from modelscope_hub import HubApi; HubApi().download_repo("Qwen/Qwen3.5-4B", "model", local_dir=os.environ["MODEL"], max_workers=2)'
```

转换脚本保留人工标注、选项顺序、推理文本和所有 RGB 视角，检查图像解码、路径和
训练/benchmark 问题与任务实例隔离。当前数据有 10,342 个训练来源问题，
按场景固定划分为 train 7,929 / val 1,108 / internal test 1,305。
独立 benchmark 有 2,777 个问题，不用于训练或选择 checkpoint。
它与训练来源共享部分初始场景名，所以不是完全未见场景测试。

## 先小试，再完整运行

先用空闲 GPU 检查小规模流程。`TRAIN_STEPS` 是优化步数，`EVAL_LIMIT` 是每个
评测子集的问题数，不代表完整成功率：

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_STEPS=16 EVAL_LIMIT=22 bash code/run_emstorm_qa_pilot.sh \
  "$DATA/prepared/train.jsonl" "$DATA/prepared/benchmark.jsonl" \
  "$MODEL" code/outputs/emstorm_qa_pilot
```

完整流程串行执行一轮 QA LoRA 训练、完整 val 和完整 benchmark。运行在 tmux 内，
为每次新实验使用新输出目录；一次仅指定一张卡，可以把 GPU 0 改为空闲卡号：

```bash
RUN="$PWD/code/outputs/emstorm_qa_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
tmux new-session -d -s emstorm_qa_full \
  "CUDA_VISIBLE_DEVICES=0 bash code/run_emstorm_qa_full.sh '$DATA/prepared/train.jsonl' '$DATA/prepared/benchmark.jsonl' '$MODEL' '$RUN' > '$RUN/full.log' 2>&1"
```

默认 stage 0、seed 3407、4-bit 基础模型、LoRA、batch 1、梯度累积 8、完整一轮。
上下文 8,192 token，评测最多生成 768 token，保留所有输入摄像头视角。
不改标注，也不把答案、GT 场景图或未来帧放进当前问题输入。

## 看进度和结果

在同一 shell 中保留 `RUN`，或重新设置为实际实验的绝对路径：

```bash
tail -f "$RUN/full.log"
watch -n 15 "cat '$RUN/status.json'"
watch -n 15 "cat '$RUN/adapter/progress.json'"
watch -n 15 "cat '$RUN/adapter_benchmark.progress.json'"
```

`status.json` 区分 train / validation / benchmark / complete；子进程非零退出记为
failed。训练进度保存在 `adapter/progress.json`；评测进度是
`adapter_val.progress.json` 和 `adapter_benchmark.progress.json`，显示
`predicted`、`total`、`correct`、`answer_accuracy` 和 `unparseable`。
相应文件只在该阶段开始后出现。

最终结果是 `adapter/train_metrics.json`、`adapter_val.metrics.json` 和
`adapter_benchmark.metrics.json`。模型输出逐条保存到同名 JSONL；完整评测结束才写
最终 metrics。进行中的 accuracy 只是前缀统计，不能作为最终报告。
非法答案记为错误，不从分母剔除；这是 QA 正确率，不是专家动作执行通过率。

## 断点续跑

不要重新运行完整包装脚本到旧目录，它会从训练阶段重新执行。训练每 50 步保存
checkpoint，只保留最近两个，包含优化器、学习率调度和随机状态。中断后按原配置
恢复，例如最新 checkpoint 是 950：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n emstorm-train python code/train_emstorm.py train \
  --data "$DATA/prepared/train.jsonl" --stage 0 --model "$MODEL" --load-in-4bit --seed 3407 \
  --output "$RUN/adapter" --epochs 1 --batch-size 1 --gradient-accumulation 8 \
  --logging-steps 10 --save-steps 50 --resume-from-checkpoint "$RUN/adapter/checkpoint-950"
```

训练恢复完成后单独评测；val 中断也使用第一条命令，benchmark 中断使用第二条：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n emstorm-train python code/train_emstorm.py predict \
  --data "$DATA/prepared/train.jsonl" --stage 0 --model "$MODEL" --load-in-4bit --seed 3407 \
  --adapter "$RUN/adapter" --max-new-tokens 768 --predictions "$RUN/adapter_val.jsonl" --resume
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n emstorm-train python code/train_emstorm.py predict \
  --data "$DATA/prepared/train.jsonl" --stage 0 --model "$MODEL" --load-in-4bit --seed 3407 \
  --adapter "$RUN/adapter" --max-new-tokens 768 --benchmark "$DATA/prepared/benchmark.jsonl" \
  --split test --predictions "$RUN/adapter_benchmark.jsonl" --resume
```

恢复会校验数据、场景划分、训练设置或评测模型及配置的一致性，不跳过重复/未知 ID
或损坏的末尾 JSONL。手动续跑不会更新完整包装脚本的 `status.json`，以当前预测
进度和最终 metrics 为准。

## 场景图阶段和验收范围

当前已标注包不含真实 `SG_t`、`SG_next` 和分割可见性监督，只能直接运行 stage 0。
stage 1/2 的实现已提供，但必须接入真实监督，不能从答案虚构场景图。
stage 2 要求 stage 1 的验证场景图 F1 和非机器人实体 F1 均至少 0.7。
字段格式、损失权重、验收规则和实验记录见
[详细训练说明](docs/emstorm_training.md)。

2026-09-30 完整训练已完成 992 优化步，val 为 1,058/1,108 (95.49%)，答案格式
有效率 100%。2026-10-01 发布时完整 benchmark 仍运行中，尚无最终结果；也没有
完整基础模型对照，不应把小试结果与完整训练直接比较成提升幅度。
本分支不上传数据、模型、预测产物或密钥，需各自有访问权限后下载。
