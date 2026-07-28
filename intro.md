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

![image.png](attachment:18931745-4ce3-46fa-8ba6-45e48eef3748:image.png)

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

### OmniGibson技术金字塔：

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

[demo by jd](https://app.notion.com/p/demo-by-jd-34ba0c89137b803b9e1ec8684eab9e4b?pvs=21)

---

### **[层级 2：场景语义层 ]**

该阶段主要目标是将OmniGibson中的初始场景转换为我们熟悉的**时空图结构**，并为依据预设任务编辑场景提供**物质条件**和**方法**。

1. **3DSG 拓扑读取：**将场景转换成类似FSG的场景图，但在转换为3D场景图的同时，需要为各个物品增加方便指导编辑的属性(最好以字典形式)：
    1. 可承接 / 不可承接：对于诸如地板、桌子、柜子、椅子之类的物品，可以在其上放置新的物品，那么该物品即具有可承接性。与之相对应，水果、杯子之类的物品，不能再在其上放置新的物品，那么该物品即具有不可承接性。`可能需要区分放置在顶端(On Top)/内部(Inside)`   ←→ `对应操控OmniGibson的Object States API`
    2. ~~可移动 / 不可移动：对于书本、椅子等依靠机器人自身力量就能进行移动的物品，其具备可移动性。与之相对，对于地板、墙面等固定元素，或是冰箱、柜子等依靠机器人自身力量无法移动的物品，其具备不可移动性。 ←→ `对应OmniGibson的pickupable属性`~~
    3. 可交互性：对于电视、冰箱、各类小物件，其虽然不一定可移动，但是机器人可操作，其具备可操作性。反之，对于天花板、烟雾等不具备操作意义或不可达的物品，其不具备可操作性。此外，应当对于可交互性进行细分，如细分为
        
        - none /  manipulable(可搬运操作) / articulable(可改变结构 eg. 有铰链) / controllable (可控制 eg. 开关、切换模式) 
        
        [Pseudo Code](https://app.notion.com/p/Pseudo-Code-34ba0c89137b8082b28aeacd1bf28ea7?pvs=21)
        
    4. 异常状态：对于杯子、毛巾、衣服，其可能具备污损、破碎的状态，且在仿真环境内有相对应的资源，即本身是有异常状态的。与之相对，对于具备异常状态，但仿真环境内没有相应资源，或是本身没有可转化的异常状态，则具备无异常状态性。如果技术上可行，则应当具体的分为其可能的异常形态(eg. 破损/燃烧/污损)， 同时对于有异常状态的物品，则需要标注其现在是否处于异常状态，以一个杯子为例:
    `{object: cup, abnormal_states: [broken, dirty, …], current_abnormal_states: None}` 
    
2. **任务资产元数据库：**
    
    为实现初始场景向任务场景的转换，需要构建任务资产元数据库。Behavior-1K 提供了丰富的任务级物体与场景数据，但其语义标注与可编辑属性较为分散，难以直接作为结构化 3DSG 构建基础，因此需要结合 USD scene graph 与仿真状态采样进行语义归纳与统一建模。
    
    **特别地，对于异常状态建模（如火灾），OmniGibson 中的 `OnFire` object state 已提供标准化实现(见`BEHAVIOR-1K/OmniGibson/omnigibson/object_states/on_fire.py`)，可用于触发视觉与语义层面的状态变化，但如何应用需要进一步调研**
    
    [WebRTC配置](https://app.notion.com/p/WebRTC-34fa0c89137b800196bfea575275cd4f?pvs=21)
    
3. **DeltaSG 逆向实例化**

![image.png](attachment:4ac8675e-b6ca-48f2-95f1-d064f5d6b01c:image.png)

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

[专题：不同类型问题实现路径](https://app.notion.com/p/364a0c89137b805890b5e2043051a7ad?pvs=21)

![image.png](attachment:f19de78a-6cbe-4588-892b-b9b0f3f9b725:image.png)

#### Env A: 基础任务环境

为了确保问题的多样与可靠性，我设计了一个初步的Delta SG 生成引擎(生成任务环境A)，伪代码如下:

```python
===============================================================
### Step0 函数定义
===============================================================
	# 让LLM回答问题
	response = get_LLM_response(question)  
	# 编辑环境(重点)：将object放置在env的room房间中，修改后的3DSG和增加的物品信息
	new_sg, object_message = add_object_to_env(env, sg, Object_Meta_Database, object, room) 

===============================================================
### Step1 启发式创造初始任务环境
===============================================================
	target_objects = random.sample(Object_Meta_Database, 5) # 选定初始的五个用于编辑环境的物品
	
	...合理性保证一下...
	
	additional_objects = []
	
	for object in task_objects:
		
		question = f"""
你是一个环境设计师，下面给出一个物品{object}, 从下面的物品中选择\
能和它发生动作联系的5个物品，并给出相应的动作:
{Object_Meta_Database.all_object_name()}
以"cup"为例，输出的格式如下:
related_objects = {{
"kettle": "Pour the water from the kettle into the cup.",
"sugar": "Add some sugar to the cup",
"fridge": "Put the cup into the fridge"
...
}}
"""
		response = get_LLM_response(question) # 由LLM确定有哪些可供拓展环境的物品.
		
		...合法性验证(数量合法、输出格式合法、输出的物品在元数据库中存在)...
		
		related_objects[object] = json.load(response)
		
		additional_objects.extend(related_objects[object].keys())
	
	additional_objects = unique(additional_objects)
	# 基于采样措施来决定哪些物品保留在环境中，避免环境过于杂乱
	sampled_objects = random.sample(additional_objects, 12)
	
	object_relations = {}
	# 确定哪些物品-关系对在采样得到的环境中
	for object in task_objects:
		
		object_relations[object] = {}
		
		for related_object in related_objects[object]:
		
			if related_object in sampled_objects:
				# 将采样后仍然保留下来，可用于设计问题的物品和响应的关系进行注册
				object_relations[object][related_object] = related_objects[object][related_object]
		
===============================================================
### Step2 合理的构造相应的任务环境
===============================================================
		
	question = f"""
你是一个环境设计师，目标环境包含如下的几个空间：{env.rooms()}
为每个物品选择合适的房间，物品列表如下: {unique(sampled_objects + task_objects)}
你的回答格式应当如下:
{{
"object_name": room_id,
...
}}
"""
	response = get_LLM_response(question)
	
	...合法性验证(数量合法、输出格式合法、输出的房间存在)...
	
	object_with_room = json.load(response)
	
	additional_objects = {}
	# 启动仿真环境
	env = start_simulation()
	
	origial_env = env.save_state()
	
	
	sg_a = sg
	
	for object in object_with_room.keys()
		# <重点> 结合3DSG
		# 注意，仿真环境env需要始终上卡保存
		sg_a, additional_objects["object"] = add_object_to_env(env, sg_a, Object_Meta_Database, object, room)
	
	env_a = env.save_state()

===============================================================
### Step3 根据构造好的环境设计问题
===============================================================
	
	略
```

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

生成的实例的保存参考[**实例化的保存:**](https://app.notion.com/p/360a0c89137b8023b338df0b161e582c?pvs=21) 

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

![ChatGPT Image 2026年5月26日 18_09_15.png](attachment:5e011d08-0ea0-4fa3-8561-7f52143c8dcb:ChatGPT_Image_2026年5月26日_18_09_15.png)

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