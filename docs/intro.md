### 动机：

1. 当前的具身智能领域的缺乏多视角场景下的细粒度/时空推理数据集；缺少可扩展的数据生成范式。
2. 纯人工标注数据成本高昂 / 目前的训练范式对于探究具身场景下的推理问题动机不足

### 先前工作(技术基础)：V-STORM Framework

### 目标

1. VSTORM框架在具身场景下/多视角场景下的迁移：基于仿真场景直接构建STSG/基于三维重建构建STSG.
2. 大规模优质数据的生成(低成本/高质量/细粒度)
3. `GPT建议:Graph → Task → Physical Validation → Reasoning Data`

### 任务：

1. 写一个Introduction的Issue,
2. 论证我们Pipeline的合理性
3. 配合AI协助精读RynnBrain
4. 调研：RoboBrain 2.0 /

### 引言：

随着视觉-语言-动作（Vision-Language-Action, VLA）模型在具身智能领域的快速演进，赋予机器人在复杂的家庭照护（Home-Care）场景中执行长程、多步任务的能力，已成为走向通用具身智能的必经之路。然而，面对真实的非结构化环境，现有的主流端到端 VLA 模型极易陷入“记忆遗忘”（缺乏物体恒存性）与“动作幻觉”（违背物理常识）的双重困境。导致这一认知瓶颈的根源在于：**高质量、包含显式物理因果逻辑的长程演示数据极度匮乏**。

尽管近期如 RynnBrain 等先驱工作开始探索基于多视角的 VLA 数据范式，但现有的数据构建往往依赖高昂的人类在环（Human-in-the-loop）标注与昂贵的实体视频采集。更为关键的是，在面临极重遮挡与跨房间规划的 Home-Care 场景中，现有工作缺乏对“宏观空间拓扑”与“微观物理操作”深层耦合推理的系统性探索。要打破 VLA 模型的认知上限，具身智能赛道亟需一种**极低成本、高度可复现，且能够成体系地激发模型多视角时空推理能力**的全新数据范式。

为解决上述挑战，本文提出了一种基于图谱驱动的具身数据程序化合成范式——**[系统名称]**。我们跳出了“在物理仿真中盲目随机化状态”的传统思路，将基础的 3D 场景图（3D Scene Graph, 3DSG）作为物理环境的符号化“数字底座”。通过向基础图谱中注入任务导向的语义拓展规则，我们将静态的场景系统性地增殖为包含复杂时空动态的“任务场景图”。更关键的是，我们利用 V-STORM 框架作为自动化数据的“导师（Oracle Synthesizer）”，将扩展后的符号图谱逆向实例化回高保真物理仿真器（OmniGibson）中进行严格的物理学验证，从根本上杜绝了数据生成的幻觉。

在现有多视角范式的基础上，[系统名称] 在宏微观结合的推理深度上迈出了关键一步。我们独创的“分布式全局监控（IoT Global Cam）+ 局部本体视觉（Local Wrist Cam）”双视角观测体系，结合标准图谱求解器与事件驱动的稀疏采样机制，不仅提取了跨视角的视觉信息，更强制对齐了微观的像素级操作坐标与宏观的物理因果思维链（CoT）。通过这种低成本的合成数据飞轮，本文产出并开源了一套经过严格闭环验证的高质量 Home-Care 数据集，为下一代 VLA 模型提供了高可复现、强推理深度的数据基建。

- **提出了一种低成本、高可复现的图谱驱动数据合成引擎：** 我们构建了一套将基础 3DSG 逻辑增殖为复杂“任务场景图”，并逆向实例化到高保真物理仿真器中的全自动化流水线。该引擎摆脱了对昂贵人工标注的依赖，实现了具身物理轨迹数据的低成本、程序化无限生成。

(原始场景 ⇒ 3D SG ⇒ LLM+规则自动编辑 ⇒ 潜在任务+场景)

原始空间 + “打扫卫生”任务 ⇒ 原始空间 + 污渍 + 拖把 （多视角：机器人视野 + 房间摄像头）

- **深化了面向 Home-Care 场景的宏微观解耦推理机制：** 在双视角协同的基础上，我们设计了结合宏观图谱遍历与微观底层寻路的专家求解器，并引入事件驱动的稀疏关键帧采样技术。这不仅提纯了高密度的离散动作切片，更在 VLA 数据中深度植入了跨房间、抗遮挡的长程时空因果逻辑（CoT）。

原始空间 + 污渍 + 拖把 ⇒ 仿真环境下的验证(1. 场景合理 / 2. 任务可解) ⇒ 采样成真正可用的Data.

- **开源了极具挑战性且已证明行之有效的多视角长程数据集：** 区别于现有的指令数据集，我们开源的数据集高度聚焦于家庭照护中的复杂组合任务。数据在强制注入物理状态跃迁与精准操作坐标 `<affordance>` 的同时，保证了极高的逻辑严密性与 0 幻觉率。

Data工作(生成 + 标注 + baseline)

- **验证了合成数据对 VLA 模型执行能力的规模化提升：** 我们在复杂的 Home-Care 仿真评测中对训练后的 VLA 基座模型进行了闭环验证。实验有力地证明了，基于 [系统名称] 低成本合成的数据集，不仅能使模型在长程任务成功率上实现显著跃升，且其提供的数据结构可以直接作为通用 VLA 模型的能力拓展插件（Plug-and-play）。

训练工作(…… / SFT)

### 技术路线图：

A. 数据准备：

1. 将OmniGibson的初始场景(usd-style)转换为STSG-style(SG-style){**FSG**}.
2. 设计一系列任务 ⇒ 针对任务编辑SG ⇒ 将SG转换为合理的任务环境(即环境要满足稳定/合理) （Contribution 1）
3. 将SG/任务环境概括总结成合适的任务，并通过求解形成任务-解决方案链条。 Contribution 2
4. 根据任务-解决方案链条，在OmniGibson中采样整条路径，从而得到可用于测试/训练的数据。Contribution 2

B. 模型侧工作：…

### 任务类型:

参考(RynnBrain):

!image.png

“主动感知/指令遵循”：**主动感知**指模型主动根据当前的场景，推理出自身需要做的事情 / **指令遵循**指模型依据清晰准确的指令，完成人类的要求。

“第一视角/多视角”：**第一视角**指机器人本身的视角，也就是答案操作的主视角，解决问题所需的所有信息均可在此视角内得到，其他视角存在但对解决问题没有意义 / **多视角**指解决问题所需的线索存在在多个视角中，因此模型必须得分析和理解全部知识

“明确干扰”：即任务空间内存在多个高度相似的目标物体，需要依据具体的指令，判断从中剥离干扰项，选择正确的答案

“输出响应/输出感知”：**输出响应**指的是输出的答案应该是一个具体的动作 / **输出感知**指的是模型输出的是纯粹的场景信息

**A. (主动感知 + 第一视角 + 输出响应)  单视角主动响应：**

eg.  [机器人视角存在异常][第三视角不存在异常]  + <视角、任务空间介绍> 当前的环境中可能存在需要处理的异常问题，给出解决方案。

**B. (主动感知 + 多视角 + 输出响应)  协同主动响应**

eg.  [机器人视角][第三视角存在异常]  + <视角、任务空间介绍> 当前的环境中可能存在需要处理的异常问题，给出解决方案。

**C. (指令遵循 + 第一视角 + 输出响应)  单视角指令遵循**

eg.  [机器人视角包含任务空间][第三视角不包含任务空间]  + <视角、任务空间介绍> 完成XXX的任务，给出操作流程

**D. (指令遵循 + 多视角 + 输出响应)  协同指令遵循**

eg.  [机器人视角包含任务空间][第三视角不包含任务空间]  + <视角、任务空间介绍>完成XXX的任务，给出操作流程

**E.  (指令遵循 + 多视角 + 明确干扰 + 输出响应) 多源信息消歧**

eg.  [机器人视角包含任务空间/干扰项][第三视角包含干扰项/任务空间]  + <视角、任务空间介绍>完成XXX的任务，给出操作流程

**F.  (指令遵循 + 输出感知)  纯粹场景感知**

**G.**  **(主动感知 + 输出感知)  场景异常检测**

## OmniGibson技术金字塔：

```jsx
															[ 层级 4：数据管线层 ]
                      ====================================
                        稀疏关键帧提取 | BBox坐标映射 | QRA格式化输出
                      ====================================

                              [ 层级 3：专家求解层 ]
                 ==============================================
                   双相机同步渲染 | 专家求解器(寻路+抓取) | 物理松弛与碰撞检测
                 ==============================================

                              [ 层级 2：场景语义层 ]
            ========================================================
              ~~3DSG拓扑读取~~ | ~~任务资产元数据库(兜底)~~ | Delta SG 逆向物理实例化
            ========================================================

                              [ 层级 1：基础运行层 ]
       ==================================================================
         ~~OmniGibson Headless 引擎池 | 显存/进程生命周期管理 | 核心 API 桥接层~~
       ==================================================================
```

### **[层级 1：基础运行层]**

该阶段的目标是把OmniGibson彻底地在**我们的服务器(如3卡A6000 / 7卡4090-48G)**上跑起来，具体地，需要完成如下目标：

1. 确保OmniGibson能够在服务器的命令行环境下稳定的进行仿真和并行(对应`OmniGibson Headless 引擎池 | 显存/进程生命周期管理`)
2. 需要完成核心的API：
    1. 控制读取空间中各物体坐标/语义特征/物理状态的API
    2. 生成和控制机器人/摄像头的API
    3. 读取机器人/摄像头的画面的API

***层级 1 标志成果：***

1. OmniGibson系统下初始场景在OmniGibson下的稳定运行.
2. 在初始场景的基础上添加机器人 / 摄像头，并从机器人本身视角和摄像头视角读取画面，画面包括RGB 图、深度图以及**语义分割掩码.**
3. 读取地板/墙壁/天花板/家具的具体信息，包括坐标、物品类型、物体尺寸、房间信息以及物理信息.

demo by jd

---

### **[层级 2：场景语义层 ]**

该阶段主要目标是将OmniGibson中的初始场景转换为我们熟悉的**时空图结构**，并为依据预设任务编辑场景提供**物质条件**和**方法**。

1. **3DSG 拓扑读取：**将场景转换成类似FSG的场景图，但在转换为3D场景图的同时，需要为各个物品增加方便指导编辑的属性(最好以字典形式)：
    1. 可承接 / 不可承接：对于诸如地板、桌子、柜子、椅子之类的物品，可以在其上放置新的物品，那么该物品即具有可承接性。与之相对应，水果、杯子之类的物品，不能再在其上放置新的物品，那么该物品即具有不可承接性。`可能需要区分放置在顶端(On Top)/内部(Inside)`   ←→ `对应操控OmniGibson的Object States API`
    2. ~~可移动 / 不可移动：对于书本、椅子等依靠机器人自身力量就能进行移动的物品，其具备可移动性。与之相对，对于地板、墙面等固定元素，或是冰箱、柜子等依靠机器人自身力量无法移动的物品，其具备不可移动性。 ←→ `对应OmniGibson的pickupable属性`~~
    3. 可交互性：对于电视、冰箱、各类小物件，其虽然不一定可移动，但是机器人可操作，其具备可操作性。反之，对于天花板、烟雾等不具备操作意义或不可达的物品，其不具备可操作性。此外，应当对于可交互性进行细分，如细分为

        - none /  manipulable(可搬运操作) / articulable(可改变结构 eg. 有铰链) / controllable (可控制 eg. 开关、切换模式)

        Pseudo Code

    4. 异常状态：对于杯子、毛巾、衣服，其可能具备污损、破碎的状态，且在仿真环境内有相对应的资源，即本身是有异常状态的。与之相对，对于具备异常状态，但仿真环境内没有相应资源，或是本身没有可转化的异常状态，则具备无异常状态性。如果技术上可行，则应当具体的分为其可能的异常形态(eg. 破损/燃烧/污损)， 同时对于有异常状态的物品，则需要标注其现在是否处于异常状态，以一个杯子为例:
    `{object: cup, abnormal_states: [broken, dirty, …], current_abnormal_states: None}`

2. **任务资产元数据库：**

    为实现初始场景向任务场景的转换，需要构建任务资产元数据库。Behavior-1K 提供了丰富的任务级物体与场景数据，但其语义标注与可编辑属性较为分散，难以直接作为结构化 3DSG 构建基础，因此需要结合 USD scene graph 与仿真状态采样进行语义归纳与统一建模。

    **特别地，对于异常状态建模（如火灾），OmniGibson 中的 `OnFire` object state 已提供标准化实现(见`BEHAVIOR-1K/OmniGibson/omnigibson/object_states/on_fire.py`)，可用于触发视觉与语义层面的状态变化，但如何应用需要进一步调研**

    WebRTC配置

3. **DeltaSG 逆向实例化**

!image.png

参考Embodied-Reasoner，其训练集中包含9390条路径(from 120个初始环境) ↔ 64k图片，相当于每个环境需要生成70个任务场景

由于OmniGibson本身推理场景较少，且一个任务场景可用于生成多种问题，因此上述指标仅供参考，但每个初始环境仍需要生成足够的任务场景，目标统计指标如下:

| Task | Trajectory | QA Pairs | Data Source |  |
| --- | --- | --- | --- | --- |
| 64k+ | 10k+ | 128k+ |  |  |

初始环境 ⇒ 从任务资产元数据库中采样用于编辑环境的物品 ⇒ LLM驱动的基于常识的环境布置 → 基于物理引擎的稳定性验证 → 基础任务环境(Env-A)

基础任务环境(Env-A) ⇒ 固定机器人、摄像头与目标物品 → **纯粹场景感知任务 / 单视角指令遵循 / 协同指令遵循**

Env-A: 具身任务环境，考察”场景中有什么”

基础任务环境(Env-A) ⇒ 添加异常 → 基于物理引擎的稳定性验证 & LLM驱动的场景可解性验证 → ***异常任务环境(Env-B)*** ⇒ 固定机器人、摄像头 → **协同主动响应 / 单视角主动响应 / 场景异常检测**

Env-B: 事件语义环境，考察”场景中发生了什么”

基础任务环境(Env-A) / 异常任务环境(Env-B) ⇒ 增加有歧义物品/挑战性场景 → ***进阶任务环境(Env-C)*** → **多源信息消歧**

Env-C: 认知语义环境，考察”应当如何决策”

为了方便大规模存储与问题实例化，我们需要保存任务环境与任务信息。任务环境可以保存为(初始环境名称) + (添加的任务资产)

任务信息需要保存: 任务类型 / 所有待操作的物品(Unique-id) / 自然语言格式的操作命令 / 目标物品是否直接可见 / 求解所需的

专题：不同类型问题实现路径

!image.png

#### Env A: 基础任务环境

~~为了确保问题的多样与可靠性，我设计了一个初步的Delta SG 生成引擎(生成任务环境A)，伪代码如下:~~

**[2026/6/16更新] 先前的伪代码生成逻辑没有必要完全摒弃，可以在限制生成任务复杂度的基础上，用于快速构造大量简单问题。**

旧Env-A伪代码:

**[2026/6/16更新]** 先前的环境生成范式面临一个很大的问题：**物体选择的随机性和任务规划中出现的幻觉干扰了实例本身的质量**。因此，我们需要从`随机物品` ⇒ `设计任务` ⇒ `构造场景` ⇒ `抽取实例` 变成反向操作：`随机任务门类` ⇒ `确定任务类型` ⇒ `选择相应物品` ⇒ `构造多元场景` ⇒ `抽取实例` 。通过首先限制任务，来确保实例不会过于离谱。

看了一下代码，按照新的env-A范式的话其实不需要改太多地方。env-A的整体逻辑也不太用变，就是首先得对**任务的抽取进行更加严格的约束**，中间得对records抽取加约束。先配置相关的物品，随后再看情况生成干扰项/无关项。

例如，可以初步将任务实例定为下面几种:

```python
VALID_TASKS = {

    "fire_response",
    "spill_cleanup",
    "find_medicine",
    "find_keys",
    "organize_table",
    "close_faucet",
    "retrieve_object",
    ...
}
```

系统首先随机抽取类别门类，以此为目标生成一个最小的可解空间；

```python
target_task = radom.sample(VALID_TASKS)

# 根据target_task, 寻找机器人任务所需的最小物品集合
required_objects = find_objects_via_LLM(target_task, ALL_OBJECT_LIST)

# 根据required_objects和task, 进行第一层级的随机(目标物品的模型 / 位置)
env_a = Add_objects_to_env(LLM, env, required_objects, target_task)

# 根据难度(Env-A/ Env-B / Env-C)，确定机器人的位置，并且细化实例
# (如对于找药，此时就可以将任务实例细化为:
# 位于[...]的机器人移动到[...]位置拿起药物，并将其放到[...]位置，任务中的摄像头布置为[...])
...生成基本实例...

# 为了确保具体实例的可解，再对环境进行微调(如确保标签正对着机器人摄像头/全局摄像头)

```

在确定最小可解空间后，根据难度，可以接着生成干扰项。

环境的保存: 直接保存预编译好的.usd开销太高，因此我们采用“基础usd环境 + json的格式”

```json
{
  "task_id": "clean_table_001",
  "base_env_usd": "Rs_int.usd",
  "added_objects": [
    {
      "name": "apple_1",
      "path": "/assets/models/apple/apple.usd",
      "position": [1.2, 0.5, 0.8],
      "orientation": [0, 0, 0, 1]
    },
    {
      "name": "sponge_1",
      "path": "/assets/models/sponge/sponge.usd",
      "position": [1.4, 0.6, 0.8],
      "orientation": [0, 0.707, 0, 0.707]
    }
  ]
}
```

构造的问题应当有意义且准确无误：例如，针对问题”用水壶给杯子倒水”，在生成相应的问题时，要确保机器人视野/监控中能够看到水壶/杯子，且目标是唯一的，否则则需要给出辅助的自然语言描述，用于引导机器人，如”用水壶给餐厅的杯子倒水”(杯子不可见) / ”用红色的水壶给餐厅的杯子倒水”(有多个水壶)

特别地，元数据的核心是能够方便地得到答案，在专家求解器获取答案过程中有重要意义. 对于上面的例子，如果机器人在卧室，杯子在客厅，壶在厨房，则在生成问题的同时最好生成拓扑路径(当然，留到专家求解器部分也行)

生成的实例的保存参考**实例化的保存:**

#### Env-B：事件任务环境

针对环境B，首先要选择要添加的异常，对于异常的布置，有以下的伪代码:

```python
===============================================================
### Step0 函数/数据定义
===============================================================
# 环境中目前的所有物品
all_objects = env.object() + sampled_objects + task_objects

# 易作为火源的物品
all_fire_source_objects = [
    "stove", "oven", "microwave", "toaster", "toaster_oven",
    "deep_fryer", "electric_kettle", "rice_cooker", "instant_pot",
    "pressure_cooker", "crock_pot", "coffee_maker", "espresso_machine",
    "waffle_maker", "flat_top_grill", "gas_fireplace", "wood_fireplace",
    "space_heater", "sauna_heater", "radiator", "charcoal_grill",
    "smoker", "lighter", "match", "match_box", "beeswax_candle",
    "dip_candle", "pillar_candle", "spirit_lamp", "sparkler",
    "cigar", "cigarette", "tobacco_pipe", "power_strip", "wall_socket",
    "clothes_dryer", "iron", "hair_dryer", "desktop_computer",
    "laptop", "bottle_of_lighter_fluid", "fuel_can", "spray_paint_can",
    "spray_can", "bottle_of_solvent", "bottle_of_paint_remover"
]

# 适合作为异常解决方案的物品
anomaly_resolution_map = # ==========================================
# 具身智能 Home-Care 异常解决图谱 (Recipe Paths 版)
# ==========================================
anomaly_resolution_map = {

    "Fire_Emergency": [
        {
            # 配方 A：使用灭火器
            "path_name": "use_extinguisher",
            "required_infrastructure": [],
            "spawnable_tools": ["fire_extinguisher"]
        }
    ],

    "Dirty_Dishes": [
        {
            # 配方 A：有洗碗机则不生成任何工具
            "path_name": "machine_wash",
            "required_infrastructure": ["dishwasher"],
            "spawnable_tools": [] # 保持空列表，底层循环自动跳过生成
        },
        {
            # 配方 B：手洗流派 (必须同时拥有水槽和水龙头，且打包生成海绵和洗洁精)
            "path_name": "hand_wash",
            "required_infrastructure": ["sink", "faucet"],
            "spawnable_tools": ["sponge", "bottle_of_dish_soap"]
        }
    ],

    "Dirty_Clothes": [
        {
            # 配方 A：直接用洗衣机洗
            "path_name": "machine_wash_clothes",
            "required_infrastructure": ["washer"],
            "spawnable_tools": []
        },
        {
            # 配方 B：放入脏衣篓 (无需基建，动态生成 hamper)
            "path_name": "put_in_hamper",
            "required_infrastructure": [],
            "spawnable_tools": ["hamper"]
        },
        {
            # 配方 C：放在布毯上收集 (无需基建，动态生成 cloth_blanket)
            "path_name": "put_on_cloth_basket",
            "required_infrastructure": [],
            "spawnable_tools": ["cloth_basket"]
        }
    ],

    "Broken_Object": [
        {
            # 配方 A：全套工具清扫 (三个工具必须打包一起生成，缺一不可)
            "path_name": "sweep_up",
            "required_infrastructure": [],
            "spawnable_tools": ["broom", "dustpan", "trash_can"]
        }
    ]
}

===============================================================
### Step1 寻找可异常化的物品
===============================================================
abnormal_obj_list = []

for obj in all_objects:

	if obj.abnormal_state = True or obj in all_fire_source_objects:

		abnormal_obj_list.append(obj)
# 设置异常
for ab_obj in abnormal_obj_list:

	... 采样策略(防止数量爆炸) ...

	# 确保任务可解
	solution_obj = find_solution(anomaly_resolution_map, ab_obj)

	env_b = add_abnormal_to_env(env_a, ab_obj)

	if solution_obj:

		env_b = add_resolution_to_env(env_b, solution_obj)



	... 根据异常布置机器人/摄像头，设计问题 ...

	... 储存问题 ...
```

#### **实例化的保存:**

实例化的保存核心目标是保存一下内容：任务环境、任务内容、任务解决方案

由此最为直观的方法是设计一个字典， 只保留五类ActionType：
MOVE / PICK / PLACE / INTERACT / WAIT

```json
{
"task_environment": "..." ,
"task_NL": "自然语言格式的任务"，
"robot": {
	...
}, # 记录任务场景中使用的robot的初始信息
"camera": [
	...
], # 记录环境中的摄像头信息
"task_objects": [
	{"object_name": "...", "object_position": "...", "object_id": "..."}
],
"task_solution": [
	{"position": "...", "action_type": "...", "action_information": ...},
	{"position": "...", "action_type": "...", "action_information": ...},
] # 逐个step排布的列表(机器人每进入一个新的空间/进行一个新的操作算一个step)
}
```

直观的图片形式如下：

!ChatGPT Image 2026年5月26日 18_09_15.png

具体地，移动到房间的话需要保存房间id，移动到目标点的话需要保存目标点id：

```json
{
  "env_id": "Env-B_Fire",

  // =====================================================
  // Task Definition
  // =====================================================

  "task": {

    "task_id": "fire_task_001",

    "task_type": "...",

    "instruction":
      "Resolve the fire emergency using the extinguisher.",
  },

  // =====================================================
  // Robot Initialization
  // =====================================================

  "robot": {

    "robot_id": "robot_0",

    "initial_room": "bedroom_0",

    "pose": {
      "position": [1.54, -2.10, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0]
    }
  },

  // =====================================================
  // Camera Initialization
  // =====================================================

  "camera": [

    {
      "camera_id": "cam_kitchen",

      "camera_type": "global_camera",

      "room_id": "kitchen_0",

      "pose": {
        "position": [5.5, 2.5, 2.8],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      }
    },

    {
      "camera_id": "cam_bathroom",

      "camera_type": "global_camera",

      "room_id": "bathroom_0",

      "pose": {
        "position": [7.2, -1.0, 2.8],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ],

  // =====================================================
  // Scene Objects
  // =====================================================

  "objects": [

    {
      "object_id": "fire_1",

      "object_name": "kitchen_fire",

      "category": "anomaly",

      "semantic_roles": [
        "goal_target"
      ],

      "states": {
        "on_fire": true
      },

      "room_id": "kitchen_0",

      "pose": {
        "position": [5.10, 2.22, 0.0]
      }
    },

    {
      "object_id": "ext_1",

      "object_name": "fire_extinguisher",

      "category": "tool",

      "semantic_roles": [
        "interaction_tool"
      ],

      "states": {},

      "room_id": "kitchen_0",

      "pose": {
        "position": [4.25, 3.12, 0.85]
      }
    }
  ],
  // =====================================================
  // Executable Plan
  // =====================================================
  "solution_plan": [
    {
      "step_id": 1,
      "primitive": "MOVE",
      "nl": "MOVE to Extinguisher"
      "inventory": [],
      "target_object": "ext_1",
    },
    {
      "step_id": 2,
      "primitive": "PICK",
      "nl": "Pickup Extinguisher",
      "inventory": [],
      "target_object": "ext_1",
    },
    {
      "step_id": 3,
      "primitive": "MOVE",
      "nl": "Move to Fireplace",
      "inventory": ["ext_1"],
      "target_object": "fire_1",
    },

    {
      "step_id": 4,
      "primitive": "INTERACT",
      "nl": "Extinguish Fire",
      "inventory": ["ext_1"],
      "tool_object": "ext_1",
      "target_object": "fire_1",
    }
  ]
}
```

生成”solution plan”部分的prompt可参考一下内容:

```json
你是一个任务规划机器人，你的目标是:

Resolve the fire emergency using the extinguisher.

你需要使用的工具包括:

[{"object_id": "ext_1", "object_name": "fire_extinguisher"}]

与目标相关的核心物品是:

{"object_id": "fire_1", "object_name": "kitchen_fire", "category": "anomaly"}

给出具体的任务规划，其中每一步都基于如下的模板

    {

"step_id": 1,

"primitive": [MOVE / PICK / PLACE / INTERACT / WAIT],

"nl": "MOVE to Extinguisher"

"inventory": [],

"target_object": "ext_1",

},
```

#### Env-C：Constraint-Semantic Environment（约束语义环境）

Env-C 旨在考察模型在多视角、多候选项与隐式任务约束条件下的语义推理能力。区别于 Env-A 的基础任务执行与 Env-B 的事件异常理解，Env-C 更强调模型是否能够结合视觉信息、空间关系、物体 affordance 与世界常识，理解“为什么应该选择这个物体/方案”。

Env-C 基于 Env-A / Env-B 的任务环境进行扩展，通过向场景中注入多个“合理但非最优”的候选项，构建富语义约束的任务空间。例如：在灭火场景中同时放置灭火器、水桶与玩具桶，要求模型选择“最快灭火”的工具；或在多个异常同时存在时（火灾、漏水、柜门未关），要求模型优先处理“最紧急”的异常。此类任务不再依赖显式视觉属性（颜色、位置），而常识性语义约束完成推理。

```json
{
  "env_id": "Env-C_Fire_Disambiguation",

  // =====================================================
  // Task Definition
  // =====================================================

  "task": {

    "task_id": "fire_semantic_task_001",

    "task_type": "semantic_object_grounding",

    "instruction":
      "Quickly extinguish the kitchen fire using the most suitable tool.",

    "semantic_constraints": [
      "fastest_solution",
      "fire_suppression_affordance"
    ]
  },

  // =====================================================
  // Robot Initialization
  // =====================================================

  "robot": {

    "robot_id": "robot_0",

    "initial_room": "living_room_0",

    "pose": {
      "position": [1.20, -1.85, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0]
    }
  },

  // =====================================================
  // Camera Initialization
  // =====================================================

  "camera": [

    {
      "camera_id": "cam_kitchen",

      "camera_type": "global_camera",

      "room_id": "kitchen_0",

      "pose": {
        "position": [5.5, 2.5, 2.8],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      }
    },

    {
      "camera_id": "cam_bathroom",

      "camera_type": "global_camera",

      "room_id": "bathroom_0",

      "pose": {
        "position": [7.2, -1.0, 2.8],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ],

  // =====================================================
  // Scene Objects
  // =====================================================

"objects": [

  {
    "object_id": "fire_1",

    "object_name": "kitchen_fire",

    "category": "anomaly",

    "semantic_roles": [
      "goal_target"
    ],

    "states": {
      "on_fire": true
    },

    "visibility": {
      "robot_initial_view": false,
      "visible_from_cameras": ["cam_kitchen"],
      "occluded": false
    },

    "room_id": "kitchen_0",

    "pose": {
      "position": [5.10, 2.22, 0.0]
    }
  },

  {
    "object_id": "ext_1",

    "object_name": "fire_extinguisher",

    "category": "tool",

    "semantic_roles": [
      "candidate_solution",
      "optimal_solution"
    ],

    "semantic_affordance": [
      "extinguish_fire"
    ],

    "utility_score": 1.0,

    "states": {},

    "visibility": {
      "robot_initial_view": false,
      "visible_from_cameras": ["cam_kitchen"],
      "occluded": false
    },

    "room_id": "kitchen_0",

    "pose": {
      "position": [4.25, 3.12, 0.85]
    }
  },

  {
    "object_id": "bucket_1",

    "object_name": "water_bucket",

    "category": "tool",

    "semantic_roles": [
      "candidate_solution"
    ],

    "semantic_affordance": [
      "carry_water",
      "extinguish_fire"
    ],

    "utility_score": 0.45,

    "states": {},

    "visibility": {
      "robot_initial_view": false,
      "visible_from_cameras": ["cam_bathroom"],
      "occluded": false
    },

    "room_id": "bathroom_0",

    "pose": {
      "position": [7.33, -1.02, 0.0]
    }
  },

  {
    "object_id": "toy_bucket_1",

    "object_name": "toy_bucket",

    "category": "distractor",

    "semantic_roles": [
      "semantic_distractor"
    ],

    "semantic_affordance": [],

    "utility_score": 0.0,

    "states": {},

    "visibility": {
      "robot_initial_view": true,
      "visible_from_cameras": [],
      "occluded": false
    },

    "room_id": "living_room_0",

    "pose": {
      "position": [2.02, 0.55, 0.0]
    }
  }
]

  // =====================================================
  // Executable Plan
  // =====================================================

  "solution_plan": [

    {
      "step_id": 1,

      "primitive": "MOVE",

      "nl": "MOVE to Fire Extinguisher",

      "inventory": [],

      "target_object": "ext_1"
    },

    {
      "step_id": 2,

      "primitive": "PICK",

      "nl": "Pickup Fire Extinguisher",

      "inventory": [],

      "target_object": "ext_1"
    },

    {
      "step_id": 3,

      "primitive": "MOVE",

      "nl": "Move to Kitchen Fire",

      "inventory": ["ext_1"],

      "target_object": "fire_1"
    },

    {
      "step_id": 4,

      "primitive": "INTERACT",

      "nl": "Extinguish Fire",

      "inventory": ["ext_1"],

      "tool_object": "ext_1",

      "target_object": "fire_1"
    }
  ],

  // =====================================================
  // Semantic Reasoning Metadata
  // =====================================================

  "semantic_reasoning": {

    "reasoning_type": [
      "semantic_disambiguation",
      "affordance_grounding",
      "utility_reasoning"
    ],

    "ground_truth": {

      "optimal_object": "ext_1",

      "rejected_candidates": [

        {
          "object_id": "bucket_1",

          "reason":
            "valid_solution_but_lower_efficiency"
        },

        {
          "object_id": "toy_bucket_1",

          "reason":
            "invalid_affordance"
        }
      ]
    }
  }
}
```

---

### **[层级 3：专家求解层 ]**

该层级的核心目标是专家求解器的设计，其目标是根据第二层级DeltaSG生成的solution_plan，实例化为机器人的操纵路径。其直接关联第四层级的采样、QRA化，是第四层级运行的基础。

第三层与第四层在运行时形成执行-采样一体化流水线（execution-sampling integrated pipeline）。由于视觉观测、物理状态与 affordance 信息均依赖仿真器逐帧更新，因此关键帧采样与 QRA 化过程需要伴随专家执行器在线运行。

#### 专家求解器实现与验收协议

专家求解器分为两个明确阶段，二者输出不能混用：

1. `oracle_symbolic` 可解性验证：使用 OmniGibson 官方 `SymbolicSemanticActionPrimitives` 执行导航、抓取、放置和物体状态改变，并检查真实 object state 后置条件。该后端可能对机器人或物体进行传送，因此只能证明任务在当前物理状态下可完成，**不能作为低层 VLA action 轨迹**。
2. `physical_control` 专家轨迹：使用 OmniGibson 官方 `StarterSemanticActionPrimitives` 及其运动规划器产生逐步控制 action。官方实现当前仅支持其声明兼容的 Tiago/R1 控制配置；Fetch 生成实例必须先通过 oracle 验证，但不得把 oracle action 标记成 VLA 训练数据。

执行入口首先把 LLM 生成的 `MOVE/PICK/PLACE/INTERACT/WAIT` 编译为确定性专家原语：

| DeltaSG primitive | 专家原语 |
| --- | --- |
| `MOVE(target_object)` | `NAVIGATE_TO` |
| `PICK` | `GRASP` |
| `PLACE` | `PLACE_ON_TOP` / `PLACE_INSIDE`，由目标类别、placement mode 和指令共同确定 |
| `INTERACT` | `OPEN/CLOSE/TOGGLE_ON/TOGGLE_OFF`，由任务最终状态和步骤语义确定 |
| `WAIT` | 固定数量仿真 step |

只允许修复可以由任务最终状态唯一确定的模糊动作，例如 `retrieve_book` 最后错误生成 `INTERACT(book)` 时规范化为 `GRASP(book)`。求解器不得猜测缺失的 object id；未知目标、空 inventory 下执行 PLACE、delivery 缺少 PLACE 等情况直接拒绝。

Env-A 生成入口在样本标记为成功之前执行相同的专家计划 preflight，并把结果保存为
`task_environment.compiled_expert_plan` 和 `validation.expert_plan_preflight`。无法编译的
LLM 计划进入现有重试流程，不计入合格样本。

每个步骤根据当前及未来步骤重新计算 `useful_objects`：

- 当前或后续步骤仍引用的非 inventory 物品必须在机器人主视角或至少一个全局摄像头中可见。
- inventory 中的物品不检查可见性；后续不再引用的物品不再检查。
- `GRASP/PLACE_ON_TOP/PLACE_INSIDE` 的目标必须额外出现在机器人主视角，且 instance segmentation 像素数及 bbox 面积合法。
- MOVE 后精细操作目标未进入主视角时，只调整机器人头部相机的 pan/tilt 对准目标中心，不移动或后退机器人本体；重新采样后仍不可见则拒绝。
- look-at 不允许设置、传送或额外移动机器人底盘；它只用于视角对准，不得改变 `NAVIGATE_TO` 已到达的可操作位姿。

每一个被保存的采样事件必须同时保存 RGB、`seg_semantic` 和 `seg_instance`。默认保存每个 primitive 的 pre/post 关键帧及 look-at 补救帧；需要更密集的控制过程时通过 `--sample-every N` 保存每 N 个底层 action。语义分割既保存无损 `.npy`，也保存 16-bit PNG 预览。全局摄像头只能使用任务生成阶段已经通过可见性验证的官方墙角/墙面 camera pose，不使用 marker。

一个轨迹只有同时满足以下条件才允许 `accepted=true` 和 `qa_eligible=true`：

1. 计划编译合法，所有 object id 能在重放场景中解析。
2. 每一步 useful object 可见性检查通过。
3. 所有精细操作的机器人主视角 bbox 检查通过。
4. 每一个官方动作原语成功，且 `IsGrasping/OnTop/Inside/Open/ToggledOn` 后置条件成立。
5. 所有原始场景物品的根位姿漂移不超过 5 cm，不存在丢失物体。
6. 最终任务状态成立。抓取、放置或最终状态失败的轨迹直接淘汰，不进行 QA 采样。

第 5 项同时覆盖新增但不应移动的 `task_support/context_object`；被抓取或放置的 task object
由对应动作后置条件单独验收。机器人导航撞动自动承接家具超过 5 cm 时同样直接淘汰。

当前实现文件：

- `code/deltasg_expert.py`：计划编译、inventory/useful-object 推导和离线可见性验收。
- `code/validate_deltasg_expert_plans.py`：不启动仿真的批量计划 preflight。
- `code/run_deltasg_expert.py`：单样本 OmniGibson oracle 执行与在线采样。
- `code/run_deltasg_expert_batch.sh`：逐样本隔离的批量执行。
- `code/audit_deltasg_expert.py`：通过率、失败阶段、QA gate 和后端标注审计。

单样本必须在 tmux 中按如下方式执行：

```bash
ROOT="$(pwd)"
tmux new-session -d -s deltasg_expert_smoke \
  "cd '$ROOT' && \
   DELTASG_GPU=0 code/run_omnigibson_single_gpu.sh \
   conda run --no-capture-output -n behavior \
   python code/run_deltasg_expert.py \
     --input-json code/outputs/<batch>/<sample>.json \
     --output-dir code/outputs/expert_smoke/<sample> \
     --llm-model qwen3.8-max"
```

批量回归同样使用 tmux。`oracle_symbolic` 在同一场景内复用一个 OmniGibson 进程以降低冷启动开销；`physical_control` 仍逐样本隔离：

```bash
ROOT="$(pwd)"
tmux new-session -d -s deltasg_expert_batch \
  "cd '$ROOT' && \
   EXPERT_MAX_PER_CELL=1 bash code/run_deltasg_expert_batch.sh \
     code/outputs/<deltasg_batch> code/outputs/<expert_batch> 30"
```

真实低层 action 验证使用官方支持的机器人配置，并与 oracle 输出分目录保存：

```bash
ROOT="$(pwd)"
tmux new-session -d -s deltasg_expert_physical \
  "cd '$ROOT' && \
   code/run_omnigibson_single_gpu.sh \
   conda run --no-capture-output -n behavior \
   python code/run_deltasg_expert.py \
     --input-json code/outputs/<batch>/<sample>.json \
     --output-dir code/outputs/expert_physical/<sample> \
     --backend physical_control --robot R1 \
     --sample-every 10 \
     --llm-model qwen3.8-max"
```

只有 `physical_control` 且最终 `accepted=true` 的结果，其 `actions/step_*.npy`
才允许标记为低层 VLA action。physical 失败时仍会保存已经执行的部分 action 用于定位，
但 `low_level_vla_actions_eligible=false`；oracle 目录中的同名文件同样只用于调试，
不得进入控制训练集。

审计命令：

```bash
python code/audit_deltasg_expert.py \
  --root code/outputs/<expert_batch> \
  --min-accept-rate 0.8
```

在占用 GPU 前先执行计划 preflight：

```bash
python code/validate_deltasg_expert_plans.py \
  --input code/outputs/<deltasg_batch> \
  --output code/outputs/<expert_batch>/plan_preflight.json \
  --fail-on-invalid
```

所有本项目的 OmniGibson 入口统一经过 `code/run_omnigibson_single_gpu.sh`。默认仍使用
GPU 0；显式设置 `DELTASG_GPU=auto` 时，在候选物理 GPU 中选择显存充足且没有现存
OmniGibson/Isaac Kit 的一张卡。包装器持有 `/tmp/deltasg_omnigibson_gpu<index>.lock`，
保证每个子进程只看到一张卡，并防止生成、可视化和专家执行重叠后触发 Replicator 段错误。
它只对子进程取消 `ALL_PROXY/all_proxy`，不会修改 Codex 或父 shell 的代理环境。

截至 2026-08-04，Env-A 的 retrieval、open/close、appliance 各有一条
`oracle_symbolic` 真实 OmniGibson 轨迹通过全部步骤、后置条件和场景完整性审计，输出分别位于：

- `code/outputs/expert_oracle_preserve_20260804_170255`
- `code/outputs/expert_open_close_20260804_170907`
- `code/outputs/expert_appliance_connected_20260804_171614`

`physical_control` 使用 OmniGibson `setup.py` 锁定的 CuRobo commit
`78612f45cef52c3fa0298de243a54cd7ca614414`。不能把“CuRobo 能初始化”视为验收通过；
必须以完整任务的 `expert_result.json` 中 `accepted=true`、所有动作后置条件通过且
`scene_integrity.ok=true` 为准。

physical eligibility 还要求执行机器人的初始位姿与每个生成任务物体之间存在同一
traversability connected component，且生成物体到最近连通底盘接近点的水平距离不超过
1 m。该诊断保存在 `physical_diagnostics.approach_candidates`。稳定落地或全局相机可见
并不等价于机器人可接近；不连通、狭小浴室内被遮挡的 floor placement 必须淘汰或在
生成阶段重新放置，不能通过关闭碰撞或传送机器人补救。

Env-A retrieval/delivery 生成阶段也执行同一类前置门禁。每个 task object 的候选位置在
通过 relation、AABB 和 contact 检查后，必须在初始机器人 traversability connected
component 内找到不超过 1 m 的接近点；失败时继续尝试下一个 support/floor 候选。成功
诊断写入 `task_environment.added_objects[].placement.robot_approach`，审计缺失或失败的
字段会报告 `envA_retrieval_missing_robot_approach` / `envA_retrieval_robot_approach_failed`。

截至 2026-08-05，门禁进一步要求接近点与任务物品属于同一 official room instance，
防止隔墙或隔门的欧氏近邻被误判为可操作。Fetch 生成使用的连通图还会额外保留
`0.20 m` 专家底盘 clearance，以保守覆盖 Tiago 更大的 footprint。retrieval 仍优先使用
现有可承接家具；稀疏房间没有可靠表面时，可先在可达空地上正常放置 nightstand 作为
`task_support`。delivery 若仍没有与源支撑物不同且满足高度、接近距离约束的承接面，会优先
在另一可达房间正常放置第二个 nightstand，重新经过物理稳定、场景完整性和精确目标绑定检查，
而不是用 marker 或直接改 USD 位姿伪造承接面。floor 仅作为最后候选，不再按“大/小物品”名称硬编码：生成器读取物体落地
后的真实 AABB，以操作点离当前楼层地面的高度执行默认 `[0.10 m, 1.55 m]` 门禁。竖立且
中心足够高的瓶类可通过，钥匙、手机等低矮物体会被拒绝并改试表面或承接家具。诊断写入
`added_objects[].placement.manipulation_height`，审计会以
`envA_retrieval_unmanipulable_floor_target` 淘汰缺少证据或高度不合格的落地样本。
回归时可显式传 `--target-asset-category/--target-asset-model --target-placement-mode floor`
构造正常落地对照；该选项仍执行碰撞、连通接近、高度、可见性和专家门禁，不能绕过验收。

原生 open/close/appliance 任务也执行同一高度门禁，并显式把初始状态设置为任务最终状态的
反态。例如 `close_cabinet` 必须以 `Open=true` 开始，`turn_on_light` 必须以
`ToggledOn=false` 开始。该状态保存在 `state_changed_objects`，专家重放后才执行动作；无法
建立反态的目标不得用“后置条件原本已成立”冒充成功。

物理专家在 NAVIGATE 后还会验证任务物品位移不超过 `0.05 m`。Tiago look-at 使用
CuRobo position-mode 底盘轨迹，并将 head pan/tilt 裁剪到合法关节范围；恢复后仍无主视角
bbox，或 GRASP/PLACE 的 CuRobo 轨迹无解时，样本保持 `accepted=false`，不得进入 QA/VLA。

Beechwood 等大场景的 physical expert 还必须限制 CuRobo collision world。DeltaSG 专家按
机器人到当前目标路径经过的 official rooms 过滤 collision objects，并在动作阶段关闭
环境 observation 渲染。机器人主相机和全局 viewer 的语义/实例分割 annotator 在环境
初始化后只挂载一次，并在整条轨迹内复用；反复 add/remove modality 会重建 Replicator
OmniGraph，在第二个 primitive 中可能触发 native exit 139。

长距离 NAVIGATE 不直接使用单段的直达 CuRobo 轨迹。执行器先在机器人 footprint 侵蚀并
额外保留 `0.20 m` clearance 的官方 traversability map 上求连通路径，再以约 `0.40 m`
间隔交给官方 CuRobo 分段执行。安全路径的最后允许追加一小段到机器人 footprint 合法的
操作站位，以兼顾桌边主相机可见性。每个 primitive 结束后立即检查原生场景物体位移；
若超过 `0.05 m`，立即拒绝并停止后续动作，而不是在整条计划结束后才发现场景破坏。

R1 的 BASE-only CuRobo 轨迹仍会返回完整关节向量。分段执行时必须把 trunk 和双臂保持在
每段开始时的真实关节值，否则细小误差会逐段累积并抬高 torso-mounted camera，造成桌面
目标离开主视角。primitive cleanup 是 best-effort 诊断，最终资格由显式动作后置条件、
可见性和场景完整性共同决定，cleanup 警告不能覆盖已经验证成功的抓取状态。

截至 2026-08-05，`code/outputs/expert_physical_r1_acceptcleanup_20260805_172805`
已完成 Beechwood 桌面水瓶 retrieval 的完整 physical 回归：NAVIGATE 1023 actions、GRASP
599 actions，`accepted=true`、`qa_eligible=true`、`low_level_vla_actions_eligible=true`，
两步 `postcondition_ok=true`，`scene_integrity.moved=[]`。独立 expert audit 为 1/1 accepted，
无 QA、后端标注或采样产物违规；该结果是 physical pipeline 的单样本里程碑，不等价于
全场景/全任务统计通过率，扩大批量前仍需按场景和任务类型分层回归。

Env-A 专家覆盖使用版本化的 15 场景列表 `code/configs/env_a_scenes.txt` 和确定性清单工具：

```bash
python code/build_enva_expert_coverage_manifest.py \
  --scenes-file code/configs/env_a_scenes.txt \
  --scope full --retrieval-variants 7 \
  --output code/outputs/<coverage>/manifest.json
```

当前资产清单会产生 11097 个不重复 job：9975 个 retrieval/delivery 场景-任务-模型-放置
变体、1054 个 open/close/appliance 场景-任务-原生目标组合，以及 68 个在版本化场景图中
没有类别、状态能力和操作高度均合格目标的结构性不适用组合。基础全物品清单为
2547 个 job；任务级清单为 345 个 job，严格覆盖 15 场景乘以 23 个任务。放置变体跨独立进程比较
`sample_fingerprint`，相同支撑和位置的完全重复样本必须重试；原生目标不通过换 seed 重复
计数。运行入口 `code/run_enva_expert_coverage.py` 会隔离生成和专家子进程，只对 native
crash/timeout 重试，并把具有明确高度、状态能力或可达性证据的非法原生目标单列为
`ineligible_target`。覆盖通过率只在实际 eligible 目标上计算，但结构候选必须全部得到
accepted 或有证据的 ineligible 结论。清单预先声明的结构性不适用项与运行时新发现的
不适用项分开统计；后者默认不得超过非结构性清单的 `5%`，防止用动态降级掩盖生成或求解失败。

`--native-eligibility` 默认读取版本化的
`code/configs/env_a_native_eligibility.json`，它记录当前 BEHAVIOR 资产版本下 15 个场景的
14 个原生任务名和 124 个高度合格目标。高柜、高窗和壁挂电器等操作点超出
`[0.10 m, 1.55 m]` 的实例不会进入正式 manifest。只有场景/资产版本发生变化时才应显式
传入重新生成的 inventory；
场景集合或任务契约不一致会直接拒绝构建 manifest。
retrieval/delivery 的 8 个类别、61 个模型则版本化在
`code/configs/env_a_asset_inventory.json`，同门拉取仓库后无需依赖被忽略的 outputs 目录；
资产安装版本变化时才用 `--asset-inventory` 显式覆盖并重新审计。

当前可执行闭环严格包含 23 个任务名：9 个 retrieval/delivery、8 个 open/close 和 6 个
appliance。`retrieve_remote`、`put_object_on_table`、`put_object_in_container` 仍属于研究
taxonomy 的 future scope；本地资产库没有语义精确的 remote-control 资产，两个通用 put
模板也尚未绑定唯一物品与目标容器契约，因此不得混入当前成功率分母或伪装成已覆盖任务。
native smoke/task job 若包含多个候选目标，runner 先在一次场景加载中自动选择可达目标；
若自动选择失败，再逐个精确候选尝试。只有每个候选都有明确不合格证据时，才可将该 job
记为 `ineligible_target`。

Env-A 的 LLM 仍生成任务指令和候选 `solution_plan`，但保存前必须通过精确物体契约：取物只
抓取实际生成的任务物，送物必须先抓取该物体再放到此前选中的唯一目的实体，open/close 和
appliance 只能操作绑定的原生目标。LLM 若产生不存在的 ID、同类别替代实体或错误的动作对象，
生成器会改用同一任务的最小确定性计划。送物目的家具保存精确 ID、`PLACE_ON_TOP` 模式、
操作高度和机器人连通域可达性证据；其高度或接近距离不合格时不会进入成功样本。
每个新生成的 task environment 都持久化 `generation.llm_enabled=true`、
`generation.llm_model=qwen3.8-max` 和精确计划契约策略，不能只依赖运行日志证明模型。

扩大到全场景 oracle 覆盖前，`run_enva_physical_representatives.sh` 必须从 45-job smoke 的
accepted 结果中各取一个 retrieve、deliver、open/close 和 appliance 样本，以 R1 和官方
`StarterSemanticActionPrimitives` 执行。四个结果都必须保存真实低层动作，并由 expert audit
确认 `accepted=true`、`low_level_vla_actions_eligible=true`，否则后续大清单不应启动。
expert audit 对每个采样事件的机器人主视角和全局相机 RGB 执行实际解码、至少 `64x64`
分辨率和非空白检查，同时要求 semantic/instance 文件存在；因此“有路径但图像损坏或全黑”
不能通过可视化门禁。

## 代码实现相关问题

### 任务实例题材列表

```python
# 在生成任务实例的时候，暂时不必考虑特地构造用于输出感知任务的实例。相关实例可直接在响应实例中采样静态场景得到。

1.取物/送物任务。
	Example: pick up medicine / pick up medicine and put it to a place
	这类任务要求机器人根据一个直接的指令，获取相应物品(并将其移动到指定位置)。
	这类任务格式简单，具备现实意义，且可扩展性强。基础的取物/送物任务适配Env-A，而通过添加干扰项/进阶约束(如选择距离机器人最近的某一类物品)，可以将其扩展成适配Env-C的内容。
	任务设计重点: 确保指令精准无歧义

2.异常检测与处理任务
	Example: 灭火/处理打碎餐具/明显的清洁任务
	这类任务不给出机器人明确的操作指令，但指出环境中存在一个可被观测到的异常。机器人应当着手解决异常。
	这类任务在设计上相对来讲比较有挑战性。需要保证异常可见、可解、存在最优解决方案，可以适配Env-B/Env-C.
	设计重点: 确保异常可见、可解、存在最优解决方案

3.操作类任务
	Example: 打开开关/开关电视/倒水/不怎么明显的清洁任务
	这类任务给出一个直接的任务，但实现该任务所需的操作步骤数量不定。机器人应当根据该任务，达成一个最终状态。
	这类任务在设计上挑战性不高，但需要格外关注VLM/VLA的适配性。对于当前我们的VLM阶段，生成的任务应保证具备最优的求解路径。
	设计重点: 确保任务元素可见、可解、存在最优解决方案

4.约束类任务
	Example: 把桌子上最大的物件搬到... /
	这类任务给出一个约束控制下的指令，其需要机器人基于视觉证据，寻找目标元素。
	这类任务格式最为混乱，且与第1、3类任务存在部分重叠，但十分有助于考察机器人对环境的理解。其在设计时需要保证可解性和存在最优解决方案。

```

对应的`VALID_TASK`字段可以为：

```python
VALID_TASKS = {

    # Retrieval & Delivery
    "retrieve_medicine",
    "retrieve_key",
    "retrieve_remote",
    "retrieve_phone",
    "retrieve_book",
    "retrieve_drink",
    "retrieve_food",
    "deliver_medicine",
    "deliver_food",
    "deliver_drink",
    "put_object_on_table",
    "put_object_in_container",

    # Anomaly Response
    "fire_response",
    "spill_cleanup",
    "broken_dish_cleanup",
    "broken_glass_cleanup",
    "knife_recovery",
    "fallen_object_recovery",
    "trash_cleanup",

    # Open / Close
    "open_door",
    "close_door",
    "open_window",
    "close_window",
    "open_fridge",
    "close_fridge",
    "open_cabinet",
    "close_cabinet",

    # Appliance
    "turn_on_light",
    "turn_off_light",
    "turn_on_tv",
    "turn_off_tv",
    "turn_on_stove",
    "turn_off_stove",

    # Liquid
    "fill_cup_with_water",
    "pour_water_into_cup",
    "empty_container",

    # Cleaning
    "clean_table",
    "clean_floor",
    "clean_sink",

    # Organization
    "organize_books",
    "organize_medicine",
    "organize_food",

    # Constraint
    "retrieve_nearest_object",
    "retrieve_largest_object",
    "retrieve_smallest_object",

    # Semantic
    "retrieve_correct_medicine",
    "retrieve_correct_cleaner"
}
```

## 模型与训练相关:

我们的框架优势在于任何任务的任何步骤都能绑定到真正的GroundingTruth，且可以方便地将场景的信息以时空图的形式输出。因此需要将具身任务的时空图数据加入到训练流程中。
