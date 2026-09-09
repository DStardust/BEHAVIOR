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
| Env-B | `fire`、`dirty_dishes`、`dirty_clothes`、`broken_object` |
| Env-C | `retrieval_delivery`、`open_close`、`appliance`、`fire` |

## 运行约束

OmniGibson 必须使用单张 GPU。仅对 OmniGibson 子进程取消代理，不要在当前 shell 中全局取消代理：

```bash
env -u ALL_PROXY -u all_proxy \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/run_online_deltasg.py ...
```

LLM 是生成和验证流程的一部分。运行前设置 DashScope 密钥：

```bash
export DASHSCOPE_API_KEY='<your-key>'
```

默认模型是 `qwen3.8-max`。单次 Python 入口可以显式指定：

```text
--llm-model qwen3.8-max
```

批处理脚本可通过环境变量统一替换为其他 DashScope 兼容模型：

```bash
export DELTASG_LLM_MODEL='<model-name>'
```

较便宜的模型可用于开发和小规模回测，但不同模型的计划合法率可能不同；正式数据集应在 manifest 中保留实际的 `llm_model`，并分模型审计成功率。

仓库不保存 API 密钥。可选的兼容接口地址通过 `LLM_BASE_URL` 设置。
`code/run_omnigibson_single_gpu.sh` 会加载当前工作树根目录的 `.env`，再仅对
OmniGibson 子进程取消代理；可通过 `DELTASG_ENV_FILE` 显式指定另一个项目内
环境文件。不要从其他项目的 `.env` 读取密钥。

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
    --llm-model qwen3.8-max \
    --output-dir code/outputs/enva_beechwood \
    --seed 1000
```

`--allow-repeat-tasks` 允许重复抽取任务类别，不允许生成完全相同的样本。样本指纹包含初始场景、任务、房间、物品、承接面和量化后的位置；相同物品放在不同合理位置会被视为不同样本。

初始摄像头布置使用真实 `seg_instance` 观测进行验证。系统先检查机器人头部主相机，再针对仍不可见的任务物品，在其所在房间按官方房间相机策略逐台布置全局相机；每个房间最多一台。使用命令行参数 `--max-global-cameras` 或批处理环境变量 `MAX_GLOBAL_CAMERAS` 设置上限（默认 `3`）。达到上限、无法解析物品房间，或房间相机仍看不到对应物品时，该样本会被拒绝并重试。

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

- `TASK_OBJECTS` / `CONTEXT_OBJECTS`：控制单个任务实例中任务物品和上下文物品的目标数量；正式多物品回测可使用 `TASK_OBJECTS=2 CONTEXT_OBJECTS=1`。
- `VISUALIZE_INCREMENTAL=1`（默认）：每个场景和任务标签补齐目标样本后立即生成图片与 bbox，而不是等待全部生成任务结束。

```bash
# 只跑指定场景和标签
SCENES='Beechwood_0_int Merom_0_int' \
LABELS='envA_retrieval_delivery,envB_all' \
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

Env-A 当前可执行闭环为 23 个任务名：9 个 retrieval/delivery、8 个 open/close、6 个 appliance。研究 taxonomy 中的 `retrieve_remote`、`put_object_on_table`、`put_object_in_container` 尚无完整物理资产/目标契约，不计入当前全覆盖结果。

真实 floor 高度正反例回归使用 `code/run_enva_floor_height_regression.sh <output-root>`：竖立水瓶必须通过生成和 oracle expert，低矮手机必须留下 `task_object_floor_height_out_of_range` 证据，不能生成成功样本。

场景或资产版本升级后，用已有干净样本中的完整 `before_graph` 刷新原生目标配置：

```bash
python code/build_enva_native_eligibility.py \
  --input-root code/outputs/<multiscene-run> \
  --output code/configs/env_a_native_eligibility.json
```

工具要求 15 个版本化场景均有图，并且每个场景至少存在一个高度合格的 open/close 和 appliance 目标族，否则非零退出。

具体模型连续两次放置失败后会在当前断点中暂停调度，避免一个坏资产造成无限重试；它不会从覆盖清单中消失，因此最终覆盖审计仍会报告该模型缺失。应修复模型或放置策略后续跑，而不是降低审计标准。

## Env-B 异常任务

Env-B 的正式标签是 `envB_all`，包含四类异常：

| `--env-b-types` | 异常证据 | 解决路径 |
| --- | --- | --- |
| `fire` | 官方 `OnFire=True` + `code/assets/Flame_Animation.usdz` 动画火焰 | `fire_extinguisher` |
| `dirty_dishes` | 官方 `Covered(stain)=True` | 原生 dishwasher，或完整的 sponge + dish soap |
| `dirty_clothes` | 官方 `Covered(dirt)=True` | 原生 washer，或 hamper / basket |
| `broken_object` | `broken_glass` / `broken_light_bulb` 真实破损资产 | 完整的 broom + dustpan + trash can |

火焰的任务状态仍由 OmniGibson 官方 `OnFire` 管理。生成器只关闭其原有 Flow
显示器，并按 `demo_flame_rs_int_backup.py` 的材质、朝向、缩放和光源逻辑加载
仓库内 USDZ；可视化和专家回放会从样本记录重建同一个效果，灭火后再移除。
这不是 marker，也不以图片效果代替官方状态验证。

单场景四类各生成一个样本：

```bash
env -u ALL_PROXY -u all_proxy CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n behavior \
  python code/run_online_deltasg.py \
    --scene Ihlen_1_int --robot Tiago --env-type B \
    --env-b-types fire,dirty_dishes,dirty_clothes,broken_object \
    --allow-repeat-tasks --num-envs 4 \
    --min-global-cameras 2 --max-global-cameras 3 \
    --llm-model qwen3.8-max \
    --output-dir code/outputs/envb_all_rs_int
```

`dirty_dishes` 会先从完整场景图选择原生 dishwasher，或绑定同一房间内的
sink + faucet，再把机器人出生点限制到该基础设施 1.15 m 操作范围内；不会因为
一次随机出生点落在其他连通分量就错误判定 recipe 不可用。其余 recipe 无基础
设施要求并生成整套解决物品。hamper、basket、trash can 等任务终点会先按官方
资产元数据过滤不可操作尺寸，落地后仍必须通过实时 AABB 高度和导航检查。
数据中的 `cloth_basket` 请求会显式记录为 `wicker_basket` 资产替换，因为当前
BEHAVIOR 资产库没有 `cloth_basket` 类别。

对已生成样本执行同一进程的符号专家回放（每一步均保存官方状态变更前后视觉）：

```bash
DELTASG_GPU=0 EXPERT_BACKEND=oracle_symbolic EXPERT_LABELS=all \
  EXPERT_TASKS=all EXPERT_ROBOT=Tiago DELTASG_LLM_MODEL=qwen3.8-max \
  code/run_deltasg_expert_batch.sh \
    code/outputs/envb_all_rs_int \
    code/outputs/envb_all_rs_int_expert 0
```

火灾必须以 `OnFire=False` 结束并移除 USDZ 火焰，手洗必须以官方
`Covered(stain)=False` 结束，衣物和破损物清理必须得到官方 `Inside=True`；
dishwasher / washer 路径还要验证官方 `Open`、`ToggledOn` 状态序列。只有
`expert_result.json` 同时满足 `accepted=true` 与 `qa_eligible=true` 才是专家
端到端合格样本。

### Env-A/B/C 多场景端到端脚本

`code/run_envbc_multiscene_e2e.sh` 按场景串行完成生成、审计和专家回放，避免每个
样本重复冷启动。单个任务失败会记录到日志和 checkpoint，不会阻止后续场景继续
运行；重新执行同一命令时，退出码为 0 的阶段会跳过，失败阶段会从已有 checkpoint
续跑。

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="code/outputs/envbc_e2e_${RUN_ID}"
tmux new-session -d -s "envbc_${RUN_ID}" \
  "cd '$PWD' && DELTASG_GPU=0 DELTASG_LLM_MODEL=qwen3.8-max \
   SCENES='Ihlen_1_int Beechwood_0_int' \
   ENVA_NUM=0 ENVB_NUM=8 \
   ENVB_TYPES='fire,dirty_dishes,dirty_clothes,broken_object' \
   ENVC_NUM=0 RUN_EXPERT=1 \
   bash code/run_envbc_multiscene_e2e.sh '$OUT' \
   > '$OUT.console.log' 2>&1"
```

其中 `ENVA_NUM`、`ENVB_NUM`、`ENVC_NUM` 是每个场景的请求数量。只验证 fire 时
可设 `ENVB_TYPES=fire`；不设置 `SCENES` 时使用
`code/configs/env_a_scenes.txt` 中的场景。脚本通过
`code/run_omnigibson_single_gpu.sh` 保证每个 OmniGibson 子进程只看到一张 GPU，
并且只在子进程内取消代理。

实时查看生成数和专家端到端接受率：

```bash
python code/monitor_envbc_multiscene_e2e.py "$OUT"
tail -f "$OUT"/*/logs/envb.log
tail -f "$OUT"/*/expert/logs/persistent_worker.log
```

结束后每个场景包含 `generation_audit.json`、专家目录和独立退出码；总览写入
`final_report.txt`。生成成功不能代替专家成功，只有监控表中的 `accepted` 才表示
专家结果同时通过 `accepted=true` 和 `qa_eligible=true`。

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
