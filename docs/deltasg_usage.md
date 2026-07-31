# DeltaSG 数据生成与维护指南

## 程序入口

- `code/online_deltasg.py`：DeltaSG 生成引擎，由命令行入口调用，不建议直接执行。
- `code/run_online_deltasg.py`：单场景生成入口。
- `code/run_enva_multiscene.sh`：Env-A 多场景便捷脚本，默认把总样本数分配到 5 个已验证场景。
- `code/run_batch100_all.sh`：正式批处理入口。它覆盖 Env-A/B/C 的全部 8 个任务标签，对每个标签和场景补齐合格样本、生成可视化，并审计最终接受的样本。
- `code/visualize_deltasg_batch.py`：按官方房间摄像头策略生成 RGB、实例分割和 bbox 结果。
- `code/audit_deltasg_outputs.py`：检查生成结果、场景完整性、重复样本和可视化完整性。

支持的任务标签如下：

| 环境 | 任务类型 |
| --- | --- |
| Env-A | `retrieval_delivery`、`open_close`、`appliance` |
| Env-B | `fire` |
| Env-C | `retrieval_delivery`、`open_close`、`appliance`、`fire` |

## 运行约束

OmniGibson 必须使用单张 GPU。仅对 OmniGibson 子进程取消代理，不要在当前 shell 中全局取消代理：

```bash
env -u ALL_PROXY -u all_proxy \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/run_online_deltasg.py ...
```

LLM 是生成和验证流程的一部分。运行前设置 DashScope 密钥，并显式选择模型：

```bash
export DASHSCOPE_API_KEY='<your-key>'
```

所有正式命令必须包含：

```text
--llm-model qwen3.7-max
```

仓库不保存 API 密钥。可选的兼容接口地址通过 `LLM_BASE_URL` 设置。

## 单场景生成

下面的命令在 `Beechwood_0_int` 生成 10 个 Env-A retrieval/delivery 样本：

```bash
env -u ALL_PROXY -u all_proxy \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/run_online_deltasg.py \
    --scene Beechwood_0_int \
    --robot fetch \
    --env-type A \
    --task-categories retrieval_delivery \
    --allow-repeat-tasks \
    --num-envs 10 \
    --llm-model qwen3.7-max \
    --output-dir code/outputs/enva_beechwood \
    --seed 1000
```

`--allow-repeat-tasks` 允许重复抽取任务类别，不允许生成完全相同的样本。样本指纹包含初始场景、任务、房间、物品、承接面和量化后的位置；相同物品放在不同合理位置会被视为不同样本。

## Env-A 多场景

`run_enva_multiscene.sh` 的 `NUM` 是所有场景合计的目标数量，不是每个场景的数量。默认场景列表有 5 个，这只是便捷脚本的默认验证集合，不代表 OmniGibson 只有 5 个初始场景。

```bash
NUM=100 \
TASK_CATEGORIES='retrieval_delivery,open_close,appliance' \
VISUALIZE=1 \
bash code/run_enva_multiscene.sh code/outputs/enva_multiscene
```

可通过 `SCENES` 显式覆盖场景：

```bash
SCENES='Beechwood_0_int Ihlen_0_int Merom_0_int' \
NUM=30 \
bash code/run_enva_multiscene.sh code/outputs/enva_selected
```

## 全任务批处理

正式批量任务应在 `tmux` 中启动。默认情况下，脚本通过 `code/list_deltasg_scenes.py` 发现本机安装的全部室内场景，并将 `NUM` 个目标样本分配给各场景。`MIN_OK_PER_SCENE` 保证每个场景和任务标签至少具有指定数量的合格样本。每个标签都会补齐严格合格的样本，而不是简单执行固定次数。

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="code/outputs/batch100_all_${RUN_ID}"
tmux new-session -d -s "deltasg_${RUN_ID}" \
  "cd '$PWD' && NUM=100 MIN_OK_PER_SCENE=1 \
   STRICT_COVERAGE=1 REQUIRE_ALL_ASSET_MODELS=1 REQUIRE_ALL_NATIVE_TARGETS=1 \
   bash code/run_batch100_all.sh '$OUT' \
   >> '$OUT.master.log' 2>&1"
```

常用覆盖参数：

```bash
# 只跑指定场景和标签
SCENES='Beechwood_0_int Merom_0_int' \
LABELS='envA_retrieval_delivery,envB_fire' \
NUM=20 \
bash code/run_batch100_all.sh code/outputs/smoke

# 枚举将要使用的初始场景
env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/list_deltasg_scenes.py --scope interior
```

查看运行进度：

```bash
tmux ls
tmux attach -t <session-name>
tail -f <output-root>.master.log
find <output-root> -name 'online_*.json' | wc -l
```

批处理结束时会生成：

- `audit_accepted.json`：样本内容、物理稳定性、重复和 bbox 审计。
- `coverage_inventory.json`：本机已安装的任务资产类别和模型清单。
- `coverage_audit.json`：全场景 × 全任务标签矩阵、任务变体、目标类别、模型和原生目标实例覆盖报告。

`STRICT_COVERAGE=1` 要求每个启用任务族的已知任务变体至少出现一次。
`REQUIRE_ALL_ASSET_MODELS=1` 要求 retrieval/fire 使用的已安装任务资产模型全部出现。生成器会优先调度尚未覆盖的任务、模型和场景原生目标，直到数量上限；任何剩余缺口都会使脚本非零退出并写入 `failed_jobs.tsv`。小规模冒烟测试可显式设置这两个变量为 `0`，但这种结果不能标记为全覆盖数据集。
`REQUIRE_ALL_NATIVE_TARGETS=1` 还要求所有与任务状态兼容的门、窗、柜体、冰箱、开关、电器和可燃原生实例至少成为一次真实任务目标。

具体模型连续两次放置失败后会在当前断点中暂停调度，避免一个坏资产造成无限重试；它不会从覆盖清单中消失，因此最终覆盖审计仍会报告该模型缺失。应修复模型或放置策略后续跑，而不是降低审计标准。

## 可视化与审计

可视化使用正常物体放置结果和官方摄像头策略，不依赖 marker：

```bash
env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/visualize_deltasg_batch.py \
    --scene Beechwood_0_int \
    --robot none \
    --input-dir code/outputs/enva_beechwood \
    --output-dir code/outputs/enva_beechwood_vis
```

审计已接受样本：

```bash
python code/audit_deltasg_outputs.py \
  --root <output-root> \
  --vis-root <output-root>/visualizations \
  --ok-only \
  --json-out <output-root>/audit_accepted.json \
  --fail-on-issues
```

只有同时满足生成成功、初始场景完整、物理稳定、非重复且可视化审计通过的样本，才应进入训练数据集。生成输出和大批量图像不提交到 Git；应作为带 manifest 和审计报告的 release artifact 或外部数据集发布。

## 常见错误

### Orientation mismatch between entity prim and root link

该断言来自 OmniGibson 的停止态 `EntityPrim.set_position_orientation()`。部分官方场景中的关节物体，其 entity prim 与 root link 初始朝向并不完全相同；在机器人出生点重试时，如果停止仿真后逐个恢复这些原生物体，就会触发断言。

当前实现会在 PhysX 运行态恢复原生场景物体，仅在停止态设置 Fetch 位姿，然后重新启动仿真。出现此错误时先拉取最新代码，不要通过移动或删除原场景物体规避。

### 没有生成结果

场景首次加载可能需要数分钟。先检查 tmux、日志和 GPU 进程，再确认 `DASHSCOPE_API_KEY` 已传入子进程。不要因为初始化阶段没有 JSON 就并行启动第二个 OmniGibson 进程。
