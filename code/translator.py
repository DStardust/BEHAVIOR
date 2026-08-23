"""
translator.py — 层级4 数据管线层: 采样点 → QRA 结构化输出 (EM-STORM)

定位
----
本模块是 EM-STORM 技术金字塔的 **[层级 4：数据管线层]** (intro.md L96-99), 职责是:

    稀疏关键帧提取 | BBox 坐标映射 | QRA 格式化输出

输入 (合作方交付, 见 task_instance/CLAUDE.md 数据使用指南)
------------------------------------------------------------
1. **任务信息**    ``task_instance.json`` —— instruction / solution_plan(符号层
   MOVE/PICK/PLACE/INTERACT/WAIT) / plan_objects / robot / camera
2. **现成采样点**  ``expert_result.json`` 的 ``observation_events`` —— 每个采样点
   (event_id 形如 step_001_pre / step_001_post / ...) 携带机器人主视角与全局视角的
   rgb/分割图路径、bbox、物体与机器人位姿、可见性列表
3. **画面/分割图** ``frames/<event_id>/`` —— 由采样点里的路径指向

输出
----
按 intro.md L69-91 的任务类型, 生成三类任务的 QRA 数据:

    感知类      (F 纯粹场景感知 / G 场景异常检测)   → 问"场景有什么", 答可见物体
    指令遵循类  (C 单视角 / D 协同 / E 多源消歧)   → 问"完成指令的下一步", 答动作
    主动响应类  (A 单视角 / B 协同)                → 问"处理异常的下一步", 答动作

每条样本结构为 ``{ Q, A, Reasoning, context, spatial_context, images }``:
- ``Q``         题干 (自然语言)
- ``A``         答案 (动作 / 可见物体)
- ``Reasoning`` 推理链 (当前**未接入 LLM**, 预留空列表, 见 ``build_reasoning``)
- ``context``   先前主干(规划)问题的 (Q,A) 上下文 (仅规划类非空; 非任务求解类为空)
- ``spatial_context`` 空间上下文: 房间拓扑 (见 ``build_room_topology``, 对同一任务实例所有采样点相同)
- ``images``    该问所需视角图片 (相对输出子文件夹的路径, 由 ``export_task_dataset`` 填充)

输出写入 ``generate_dataset/<task_id>/`` 子文件夹 (``qra.json`` + 图片副本), 与输入数据
(``task_instance/`` 下的 JSON 与 ``frames/``) 分离。

扩展点 (日后接 LLM / 改模板 / 加任务类型只需改这里, 不动主流程)
----------------------------------------------------------------
1. ``resolve_task_type()``            —— task_instance → A~G 问题形式
2. ``QuestionTemplateRegistry``       —— (task_type, question_type) → 题干模板
3. ``build_answer()``                 —— 由任务类型 + 采样点生成答案 A
4. ``build_reasoning()``              —— 推理链 R 的预留入口 (LLM 接入点)
5. ``load_task_context()`` / ``load_sampling_scenes()`` —— 合作方 JSON 读取适配
6. ``QUESTION_GENERATORS``            —— 问题类型生成器注册表 (新增问题类型 = 追加一个生成器)

依赖
----
标准库 + 可选 numpy (读取分割 npy 计算 bbox; 缺失时回退预计算 bbox)。

使用
----
    from translator import generate_task_dataset_files

    generate_task_dataset_files(
        "task_instance/.../task_instance.json",
        "task_instance/.../expert_result.json",
        output_root="generate_dataset",
    )
    # => 在 generate_dataset/<task_id>/ 下生成 qra.json + images/<event_id>/... 图片副本
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # 仅分割 npy 读取需要; 缺失时回退到预计算 bbox
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# 默认 LLM api_key (阿里云 DashScope, OpenAI 兼容); 仅当环境中没有显式
# DASHSCOPE_API_KEY / 未传入 llm 客户端时使用, 用于把结构化 QRA 翻译成自然语言。
DEFAULT_LLM_API_KEY = "sk-e655418c5a8847cca289462d2ca137d0"


# ======================================================================
# 常量: 采样策略 / 问题类型 / 动作原语 / 任务问题形式
# ======================================================================
# 采样策略 (intro.md L929-933)
SAMPLING_INITIAL = "initial"   # 初始采样 (before simulation)
SAMPLING_DURING = "during"     # 步骤间采样 (during simulation)
SAMPLING_AFTER = "after"       # 终止采样 (after simulation)

# 问题类型 (目前区分 感知 / 规划 / bbox grounding; 总结类依赖 deltaSG, 暂不生成)
QTYPE_PERCEPTION = "perception"  # 感知类: 问"场景中有什么 / 发生了什么"
QTYPE_PLANNING = "planning"      # 规划类: 问"下一步该做什么"
QTYPE_BBOX = "bbox"              # 感知类(bbox grounding): 问机器人主视角中目标物品的 bounding box

# 动作原语 (intro.md L400, 符号层, 与 solution_plan 一致)
PRIMITIVE_MOVE = "MOVE"
PRIMITIVE_PICK = "PICK"
PRIMITIVE_PLACE = "PLACE"
PRIMITIVE_INTERACT = "INTERACT"
PRIMITIVE_WAIT = "WAIT"

# 任务问题形式 A~G (intro.md L69-91)
TASK_SINGLE_VIEW_ACTIVE = "A"     # 单视角主动响应
TASK_MULTI_VIEW_ACTIVE = "B"      # 协同主动响应
TASK_SINGLE_VIEW_INSTRUCT = "C"   # 单视角指令遵循
TASK_MULTI_VIEW_INSTRUCT = "D"    # 协同指令遵循
TASK_DISAMBIGUATION = "E"         # 多源信息消歧
TASK_SCENE_PERCEPTION = "F"       # 纯粹场景感知
TASK_ANOMALY_DETECTION = "G"      # 场景异常检测

# 感知类问题形式 (答案输出"场景信息", 无动作)
PERCEPTION_TASK_TYPES = {TASK_SCENE_PERCEPTION, TASK_ANOMALY_DETECTION}


# ======================================================================
# 数据模型
# ======================================================================
@dataclass
class StaticScene:
    """一次采样的"静态问题场景" (intro.md L938-967), 对应一个采样点
    (observation_events 的 event)。"""

    environment_name: str = ""
    observation_id: str = ""          # e.g. "step_001_pre"
    phase: str = ""                   # "pre" | "post"
    global_task: str = ""
    global_target: list[str] = field(default_factory=list)
    temporal_task: str | None = None  # 当前目标 (无下一步则 None)
    simulation_step: int = 0          # 下一步应执行的 solution_plan 步下标
    temporal_target: list[str] = field(default_factory=list)
    scene: dict[str, Any] = field(default_factory=dict)     # {视角: {paths, labels}}
    scene_with_grounding: dict[str, Any] = field(default_factory=dict)  # {视角: {obj: bbox}}
    object_poses: dict[str, Any] = field(default_factory=dict)
    robot_pose: dict[str, Any] = field(default_factory=dict)
    robot_visible: list[str] = field(default_factory=list)
    global_visible: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticScene":
        return cls(
            environment_name=data.get("environment_name", ""),
            observation_id=data.get("observation_id", data.get("event_id", "")),
            phase=data.get("phase", ""),
            global_task=data.get("global_task", ""),
            global_target=list(data.get("global_target", [])),
            temporal_task=data.get("temporal_task"),
            simulation_step=int(data.get("simulation_step", 0)),
            temporal_target=list(data.get("temporal_target", [])),
            scene=data.get("scene", {}),
            scene_with_grounding=data.get("scene_with_grounding", {}),
            object_poses=data.get("object_poses", {}),
            robot_pose=data.get("robot_pose", {}),
            robot_visible=list(data.get("robot_visible", [])),
            global_visible=list(data.get("global_visible", [])),
        )


@dataclass
class QRAPair:
    """一条 QRA 数据: { Q, A, Reasoning }。"""

    qra_id: str = ""
    task_instance_id: str = ""
    task_type: str = ""          # A~G
    question_type: str = ""      # perception / planning
    sampling_strategy: str = ""  # initial / during / after
    simulation_step: int = 0
    event_id: str = ""           # 采样点 id (e.g. "step_001_pre"), 用于定位该问所需视角图片

    Q: str = ""                                   # 题干
    A: dict[str, Any] = field(default_factory=dict)  # 答案 (结构化)
    A_nl: str = ""                                # 答案的自然语言翻译 (LLM 翻译模块填充)
    Reasoning: list[str] = field(default_factory=list)  # 推理链 (预留 LLM 接入)
    context: list[dict[str, Any]] = field(default_factory=list)  # 先前主干(规划)问题 (Q,A) 上下文
    spatial_context: dict[str, Any] = field(default_factory=dict)  # 空间上下文: 房间拓扑 (见 build_room_topology)
    images: list[dict[str, Any]] = field(default_factory=list)  # 所需视角图片 (相对输出子文件夹的路径)
    options: list[dict[str, Any]] = field(default_factory=list)  # 选择题选项 (distractor.attach_mcq 填充)
    answer_index: int = -1                        # 正确项在 options 中的下标 (-1 表示非选择题)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# 采样时机 / 问题类型 分类
# ======================================================================
def classify_strategy(scene: StaticScene) -> str:
    """把一次采样归类到三种采样时机 (intro.md L929-933)。

    - step_001_pre (simulation_step==0) → initial
    - 末步 post (temporal_task 为 None, 无下一步) → after
    - 其余 → during
    """
    if scene.simulation_step == 0:
        return SAMPLING_INITIAL
    if scene.temporal_task is None:
        return SAMPLING_AFTER
    return SAMPLING_DURING


def classify_question(task_type: str, strategy: str) -> str:
    """由 (问题形式, 采样时机) 决定"主问题"类型。

    - 终止采样 → 感知类 ("描述最终场景", intro.md L932-933)
    - 感知类任务 (F/G) → 恒为感知
    - 响应类任务 (A~E) 初始/步骤间 → 规划 ("下一步做什么")
    """
    if strategy == SAMPLING_AFTER:
        return QTYPE_PERCEPTION
    if task_type in PERCEPTION_TASK_TYPES:
        return QTYPE_PERCEPTION
    return QTYPE_PLANNING


# ======================================================================
# 问题形式映射: task_instance → A~G  (扩展点 1)
# ======================================================================
def _has_global_camera(task_instance: dict[str, Any]) -> bool:
    """实例是否带全局摄像头 (用于区分单视角/协同多视角)。"""
    for cam in task_instance.get("camera") or []:
        if cam.get("camera_type") == "global_camera":
            return True
    return False


def resolve_task_type(task_instance: dict[str, Any]) -> str:
    """把 task_instance 映射到 A~G 问题形式。

    ``env_type`` 为 "Env-A"/"Env-B"/"Env-C"; 据此 + 摄像头配置推断问题形式。

    [LLM] 目前为规则映射; 若未来 Env-A 内再细分子任务, 可接入 LLM 依据
    instruction + plan_objects 细分子类, 但不建议改变 A~G 大类。
    """
    env_type = str(task_instance.get("task_type") or task_instance.get("env_type") or "")
    multi_view = _has_global_camera(task_instance)

    # Env-C 带 semantic_constraints → 多源信息消歧
    if env_type.startswith("Env-C") or task_instance.get("semantic_constraints"):
        return TASK_DISAMBIGUATION

    # Env-B 异常处理 → 主动响应 (机器人自己发现异常并处置)
    if env_type.startswith("Env-B"):
        return TASK_MULTI_VIEW_ACTIVE if multi_view else TASK_SINGLE_VIEW_ACTIVE

    # Env-A 基础任务 → 指令遵循 (带全局摄像头即协同多视角, 否则单视角)
    if env_type.startswith("Env-A"):
        return TASK_MULTI_VIEW_INSTRUCT if multi_view else TASK_SINGLE_VIEW_INSTRUCT

    # 兜底: 有明确 instruction 即按指令遵循, 否则按场景感知
    if task_instance.get("instruction"):
        return TASK_MULTI_VIEW_INSTRUCT if multi_view else TASK_SINGLE_VIEW_INSTRUCT
    return TASK_SCENE_PERCEPTION


# ======================================================================
# 题干模板注册表 (扩展点 2)
# ======================================================================
class QuestionTemplateRegistry:
    """(task_type, question_type) → 题干模板。

    扩展新问题形式时, 用 ``register`` 登记即可, 主流程无需改动。
    模板占位符: {global_task} {temporal_task} {global_target} {instruction}
    """

    _templates: dict[tuple[str, str], str] = {}
    _fallback: dict[str, str] = {}

    @classmethod
    def register(cls, task_type: str, question_type: str, template: str) -> None:
        cls._templates[(task_type, question_type)] = template

    @classmethod
    def get(cls, task_type: str, question_type: str) -> str:
        key = (task_type, question_type)
        if key in cls._templates:
            return cls._templates[key]
        if question_type in cls._fallback:
            return cls._fallback[question_type]
        return "{global_task}"

    @classmethod
    def _seed(cls) -> None:
        """预置 A~G 各问题形式 × 各问题类型的模板 (intro.md L69-91)。"""
        # 主动响应类: 规划 → 下一步动作
        cls.register(TASK_SINGLE_VIEW_ACTIVE, QTYPE_PLANNING,
                     "观察当前环境, 判断是否存在需要处理的异常, 并给出下一步操作。")
        cls.register(TASK_MULTI_VIEW_ACTIVE, QTYPE_PLANNING,
                     "结合机器人视角与全局监控视角, 判断环境中需要处理的异常, 并给出下一步操作。")

        # 指令遵循类: 规划 → 下一步动作
        cls.register(TASK_SINGLE_VIEW_INSTRUCT, QTYPE_PLANNING,
                     "任务: {global_task}。当前需要处理: {temporal_task}。请给出下一步操作。")
        cls.register(TASK_MULTI_VIEW_INSTRUCT, QTYPE_PLANNING,
                     "任务: {global_task}。结合多视角观察, 当前需要处理: {temporal_task}。"
                     "请给出下一步操作。")
        cls.register(TASK_DISAMBIGUATION, QTYPE_PLANNING,
                     "为完成 {global_task}, 在候选工具 {global_target} 中选择最合适的一个, "
                     "并给出操作步骤与理由。")

        # 感知类: 感知 → 场景描述 / 异常指出
        cls.register(TASK_SCENE_PERCEPTION, QTYPE_PERCEPTION,
                     "描述当前场景中存在的物体及其状态。")
        cls.register(TASK_ANOMALY_DETECTION, QTYPE_PERCEPTION,
                     "观察环境, 指出其中存在的异常。")

        # 兜底模板
        cls._fallback = {
            QTYPE_PERCEPTION: "描述当前场景中存在哪些物体及其状态。",
            QTYPE_PLANNING: "任务: {global_task}。请给出下一步操作。",
        }


# ======================================================================
# 构建器: Q / A / Reasoning
# ======================================================================
def build_question(task_type: str, question_type: str,
                   scene: StaticScene, task_instance: dict[str, Any]) -> str:
    """填充题干模板, 生成自然语言问题 Q。

    [MLLM] 当前为模板占位填充; 日后可用 MLLM 结合当前帧图像把 {temporal_task} 等占位
    改写为更具体、更口语化的问题 (同时保留物体 id 便于 grounding 对齐)。
    """
    template = QuestionTemplateRegistry.get(task_type, question_type)
    values = {
        "global_task": scene.global_task or task_instance.get("instruction", ""),
        "temporal_task": scene.temporal_task or "",
        "global_target": ", ".join(scene.global_target) or "无",
        "instruction": task_instance.get("instruction", ""),
    }
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template


def _visible_objects(scene: StaticScene) -> list[str]:
    """汇总该采样点可观测到的物体 (bbox 键 + 各视角可见性列表)。"""
    objects: set[str] = set()
    for cam_boxes in scene.scene_with_grounding.values():
        if isinstance(cam_boxes, dict):
            objects.update(cam_boxes.keys())
    objects.update(scene.robot_visible)
    objects.update(scene.global_visible)
    return sorted(objects)


def _object_category(object_id: str | None, task_instance: dict[str, Any] | None) -> str:
    """从任务实例反查物品的人类可读类别 (object_id → category, 下划线转空格)。

    用于 LLM 翻译时不暴露物品 ID, 而以类别名 (如 "bottle of water") 指代物品。
    """
    if not object_id or not task_instance:
        return ""
    for key in ("plan_objects", "task_objects", "added_objects"):
        for obj in task_instance.get(key) or []:
            if isinstance(obj, dict) and obj.get("object_id") == object_id:
                return str(obj.get("category") or "").replace("_", " ").strip()
    return ""


def _target_bbox(scene: StaticScene, object_id: str | None) -> list[int] | None:
    """取目标物体在机器人主视角的 bbox (xyxy 像素坐标); 不可见时返回 None。

    供 LLM 翻译用 bounding box 定位物品 (替代物品 ID)。
    """
    if not object_id:
        return None
    bbox = (scene.scene_with_grounding.get("robot_camera") or {}).get(object_id)
    if isinstance(bbox, dict):
        xyxy = bbox.get("bbox_xyxy")
        if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
            return [int(v) for v in xyxy]
    return None


def _humanize(name: str | None) -> str:
    """把 id 风格名 (下划线分隔) 转为人类可读 (空格分隔), 如 television_room_0 → television room 0。"""
    return (name or "").replace("_", " ").strip()


def _global_camera_rooms(task_instance: dict[str, Any] | None) -> list[str]:
    """返回安装了全局(监控)摄像头的房间名列表 (去重、保持原始顺序, 已 humanize)。

    仅统计 camera_type == "global_camera" 且带 room_id 的摄像头; 机器人自带相机所在房间
    = 机器人当前房间 (由 ``_resolve_rooms`` 给出), 不在此列。
    """
    rooms: list[str] = []
    seen: set[str] = set()
    for cam in (task_instance or {}).get("camera") or []:
        if cam.get("camera_type") != "global_camera":
            continue
        room = _humanize(cam.get("room_id"))
        if room and room not in seen:
            seen.add(room)
            rooms.append(room)
    return rooms


def _resolve_rooms(task_instance: dict[str, Any] | None, scene: StaticScene,
                   next_step: dict[str, Any] | None) -> tuple[str, str, bool]:
    """求 (当前房间, 目标房间, 是否跨房间)。

    当前房间 = 上一步的 target_room (首步用 robot_initial_room);
    目标房间 = 本步的 target_room。二者都已知且不同 → 跨房间。
    """
    if not next_step:
        return "", "", False
    target_room = next_step.get("target_room") or ""
    plan = (task_instance or {}).get("solution_plan") or []
    t = scene.simulation_step
    if t <= 0:
        current_room = (task_instance or {}).get("robot_initial_room") or ""
    else:
        prev = plan[t - 1] if 0 <= t - 1 < len(plan) else {}
        current_room = (prev or {}).get("target_room") or ""
    cross_room = bool(target_room and current_room and target_room != current_room)
    return current_room, target_room, cross_room


def build_answer(question_type: str, scene: StaticScene,
                 next_step: dict[str, Any] | None,
                 task_instance: dict[str, Any] | None = None) -> dict[str, Any]:
    """组装答案 A。

    - 感知类 → 场景中可见物体列表
    - 响应类 → 下一步动作 (无下一步则任务完成); 附 target 的类别与主视角 bbox,
      供 LLM 翻译用 bounding box 定位物品 (不暴露物品 ID)

    [MLLM] 感知类答案目前只列物体 id; 日后可接入 MLLM 从 rgb/seg 图生成物体状态的
    自然语言描述 (如"瓶子放在下层橱柜上")。异常检测类 (G) 的异常标注待 Env-B 数据就绪。
    """
    if question_type == QTYPE_PERCEPTION:
        return {
            "kind": "perception",
            "objects": _visible_objects(scene),
        }

    if next_step is None:
        return {"kind": "done", "message": "任务已完成"}

    target = next_step.get("target_object")
    _, target_room, cross_room = _resolve_rooms(task_instance, scene, next_step)
    return {
        "kind": "response",
        "action": next_step.get("primitive", PRIMITIVE_WAIT),
        "target_object": target,
        "tool_object": next_step.get("tool_object"),
        "nl": next_step.get("nl", ""),
        "category": _object_category(target, task_instance),
        "bbox": _target_bbox(scene, target),
        "room": _humanize(target_room),
        "cross_room": cross_room,
    }


# bbox 过滤阈值 (intro.md: bbox 大小需"适中")
BBOX_MIN_AREA_RATIO = 0.005   # 面积占比下限: 过小 = 太远/遮挡, 不采
BBOX_MAX_AREA_RATIO = 0.75    # 面积占比上限: 过大 = 几乎占满画面, 不采

# 结构/背景类别 (非"物品", 不用于 bbox 提问; 可按需扩展)
NON_OBJECT_CATEGORIES = {"background", "walls", "floors", "ceilings", "robot"}

# bbox 坐标归一化到 [0, BBOX_NORM_SCALE]
BBOX_NORM_SCALE = 1000


def _normalize_bbox(bbox_xyxy: list[Any] | None, image_size: list[Any] | None) -> list[int] | None:
    """把像素坐标 bbox 归一化到 [0, BBOX_NORM_SCALE] (x 按宽、y 按高分别缩放)。"""
    if bbox_xyxy is None or image_size is None:
        return None
    W, H = image_size
    if not W or not H:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    return [
        int(round(x1 / W * BBOX_NORM_SCALE)),
        int(round(y1 / H * BBOX_NORM_SCALE)),
        int(round(x2 / W * BBOX_NORM_SCALE)),
        int(round(y2 / H * BBOX_NORM_SCALE)),
    ]


def _resolve_robot_npy(scene: StaticScene, frames_root: str | None, kind: str) -> str | None:
    """解析机器人主视角分割 npy 的本地路径 (kind: seg_semantic | seg_instance)。

    优先 frames_root/frames/<event_id>/robot_primary/<kind>.npy (合作方交付目录),
    否则回退采样点自带的 paths[<kind>] (可能指向合作方原始机器路径)。
    """
    event_id = scene.observation_id
    if frames_root and event_id:
        local = os.path.join(frames_root, "frames", event_id, "robot_primary", f"{kind}.npy")
        if os.path.exists(local):
            return local
    paths = (scene.scene.get("robot_camera") or {}).get("paths") or {}
    path = paths.get(kind)
    return path if path and os.path.exists(path) else None


def _seg_category_bboxes(scene: StaticScene, frames_root: str | None) -> list[dict[str, Any]] | None:
    """从语义/实例分割 npy 计算机器人主视角中各"唯一且适中"物品类别的 bbox。

    仅保留: 类别非结构/背景 且 该类别唯一 (该类别仅一个实例) 且 bbox 大小适中。
    npy 不可用时返回 None (交由调用方回退)。
    """
    if np is None:
        return None
    sem_path = _resolve_robot_npy(scene, frames_root, "seg_semantic")
    inst_path = _resolve_robot_npy(scene, frames_root, "seg_instance")
    if not sem_path or not inst_path:
        return None

    sem = np.load(sem_path)
    inst = np.load(inst_path)
    if sem.ndim != 2 or sem.shape != inst.shape:
        return None

    H, W = sem.shape
    image_area = H * W
    labels = (scene.scene.get("robot_camera") or {}).get("labels") or {}
    sem2name = {int(k): v for k, v in (labels.get("seg_semantic") or {}).items()}
    inst2name = {int(k): v for k, v in (labels.get("seg_instance") or {}).items()}

    candidates: list[dict[str, Any]] = []
    for sid in np.unique(sem):
        name = sem2name.get(int(sid))
        if not name or name in NON_OBJECT_CATEGORIES:
            continue
        mask = sem == sid
        ratio = int(mask.sum()) / image_area
        if not (BBOX_MIN_AREA_RATIO <= ratio <= BBOX_MAX_AREA_RATIO):
            continue
        # 类别唯一性: 该类别像素覆盖的实例 id 须恰好一个
        inst_ids = [i for i in np.unique(inst[mask]) if int(i) in inst2name]
        if len(inst_ids) != 1:
            continue
        ys, xs = np.where(mask)
        candidates.append({
            "category": name,
            "object_id": inst2name.get(int(inst_ids[0])),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "image_size": [W, H],
        })
    return candidates


def _fallback_bbox_candidates(scene: StaticScene) -> list[dict[str, Any]]:
    """npy 不可用时回退: 用预计算的 robot_primary.bboxes (仅计划对象, 已可见)。"""
    candidates: list[dict[str, Any]] = []
    for obj_id, bbox in (scene.scene_with_grounding.get("robot_camera") or {}).items():
        if not isinstance(bbox, dict):
            continue
        candidates.append({
            "category": obj_id,
            "object_id": obj_id,
            "bbox_xyxy": bbox.get("bbox_xyxy"),
            "image_size": bbox.get("image_size"),
        })
    return candidates


def build_bbox_question(candidate: dict[str, Any]) -> str:
    """感知类(bbox grounding)题干: 问机器人主视角中物品类别/物体的 bounding box。"""
    name = _humanize(candidate.get("category") or candidate.get("object_id") or "the object")
    return f"What is the bounding box of the {name} in the robot's primary view?"


def build_bbox_answer(candidate: dict[str, Any]) -> dict[str, Any]:
    """感知类(bbox grounding)答案: 物品在主视角的归一化 bbox (不可见则 visible=false)。"""
    bbox = _normalize_bbox(candidate.get("bbox_xyxy"), candidate.get("image_size"))
    if bbox is None:
        return {"kind": "bbox", "category": candidate.get("category"), "visible": False}
    return {
        "kind": "bbox",
        "category": candidate.get("category"),
        "object_id": candidate.get("object_id"),
        "visible": True,
        "bbox_xyxy": bbox,
        "image_size": candidate.get("image_size"),
    }


def build_reasoning(task_type: str, question_type: str, scene: StaticScene,
                    next_step: dict[str, Any] | None, llm: Any = None) -> list[str]:
    """生成推理链 Reasoning (预留入口)。

    当前**未接入 LLM**, 返回空列表。接入后在此依据 (Q, 场景, 下一步) 生成逐步推理。

    [LLM] 接入点: 用 llm_client 生成自然语言推理链, 并以 solution_plan 为事实锚点
    (防止幻觉); E 类多源消歧额外需要候选物体的 affordance 语义推理。
    """
    # TODO(LLM): 接入 llm_client 后在此生成推理链, 例如:
    #   if llm is not None:
    #       return _llm_reasoning(scene, next_step, llm)
    return []


# ======================================================================
# 问题类型生成器 (扩展点 6): 一个采样点 → 尝试生成每一类问题
# ======================================================================
@dataclass
class GenContext:
    """问题类型生成器的上下文: 一个采样点 + 任务信息。

    每个生成器据此"尝试"为该采样点生成 0..N 条 QRA 数据, 不适用则返回空列表。
    """
    task_instance: dict[str, Any]
    task_type: str
    scene: StaticScene
    strategy: str
    next_step: dict[str, Any] | None
    base_id: str
    frames_root: str | None = None  # 任务实例目录 (定位 frames/<event_id>/.../seg_*.npy)
    llm: Any = None
    backbone_context: list[dict[str, Any]] = field(default_factory=list)  # 本轮之前的主干(规划)问题 (Q,A)
    spatial_context: dict[str, Any] = field(default_factory=dict)  # 空间上下文: 房间拓扑 (对同一任务实例所有采样点相同)


def _make_pair(ctx: GenContext, question_type: str, suffix: str,
               question: str, answer: dict[str, Any],
               reasoning: list[str] | None = None,
               context: list[dict[str, Any]] | None = None) -> QRAPair:
    """用生成器上下文 + 问题类型组装一条 QRA 数据 (统一公共字段)。"""
    return QRAPair(
        qra_id=f"{ctx.base_id}__{suffix}",
        task_instance_id=str(ctx.task_instance.get("task_id", "")),
        task_type=ctx.task_type,
        question_type=question_type,
        sampling_strategy=ctx.strategy,
        simulation_step=ctx.scene.simulation_step,
        event_id=ctx.scene.observation_id,
        Q=question,
        A=answer,
        Reasoning=reasoning or [],
        context=context or [],
        spatial_context=ctx.spatial_context,
    )


def _context_entry(pair: QRAPair) -> dict[str, Any]:
    """把一条主干(规划)问题压缩为 (Q, A) 上下文条目, 供后续轮次作为上下文。"""
    return {"Q": pair.Q, "A": pair.A}


def _gen_bbox(ctx: GenContext) -> list[QRAPair]:
    """感知类(bbox grounding): 基于语义分割 npy 找"唯一且适中"物品, 每件各一条。

    非任务求解类问题: 不携带主干问题上下文 (context 为空)。
    """
    candidates = _seg_category_bboxes(ctx.scene, ctx.frames_root)
    if candidates is None:
        candidates = _fallback_bbox_candidates(ctx.scene)
    return [
        _make_pair(
            ctx, QTYPE_BBOX, f"{QTYPE_BBOX}__{c.get('object_id') or c.get('category')}",
            build_bbox_question(c),
            build_bbox_answer(c),
            build_reasoning(ctx.task_type, QTYPE_BBOX, ctx.scene, None, ctx.llm),
        )
        for c in candidates
    ]


def _room_structure_nl(spatial_context: dict[str, Any]) -> str:
    """把房间拓扑 (spatial_context) 渲染成结构串, 供题干"room with structure {}"使用。

    仅保留房间名列表 (不塞房间间距离边, 避免题干被全量连通图撑爆)。
    """
    rooms = spatial_context.get("rooms") or []
    return json.dumps({"rooms": rooms}, ensure_ascii=False)


def _render_step_nl(step: dict[str, Any], task_instance: dict[str, Any],
                    carried: str | None) -> str:
    """把单个 solution_plan 步骤渲染成英文动作短句 (供题干"上一步动作"使用)。

    PLACE 步骤的"被放置物"取传入的 carried (此前最近一次 PICK 的对象)。
    """
    prim = step.get("primitive") or ""
    target = step.get("target_object")
    cat = _object_category(target, task_instance) or _humanize(target)
    if prim == PRIMITIVE_MOVE:
        return f"Move to the {cat}."
    if prim == PRIMITIVE_PICK:
        return f"Pick up the {cat}."
    if prim == PRIMITIVE_PLACE:
        return f"Place the {carried or 'object'} on the {cat}."
    if prim == PRIMITIVE_INTERACT:
        tool = _object_category(step.get("tool_object"), task_instance)
        return f"Operate the {cat} with the {tool}." if tool else f"Operate the {cat}."
    if prim == PRIMITIVE_WAIT:
        return "Wait."
    return f"{prim} the {cat}."


def _previous_action_nl(task_instance: dict[str, Any], t: int) -> str | None:
    """渲染"上一步动作" (仅 plan[t-1]), 供题干展示; 首步无上一步时返回 None。

    只给最近一步、不再罗列完整历史 (Markov 形式), 避免文本泄露答案。PLACE 的"被放置物"
    从 plan[:t-1] 中最近一次 PICK 推断, 与专家方案语义一致。
    """
    plan = task_instance.get("solution_plan") or []
    if t <= 0 or t > len(plan):
        return None
    carried: str | None = None
    for s in plan[: t - 1]:
        if (s.get("primitive") or "") == PRIMITIVE_PICK:
            carried = _object_category(s.get("target_object"), task_instance) or _humanize(s.get("target_object"))
    return _render_step_nl(plan[t - 1], task_instance, carried)


def _build_english_question(ctx: GenContext, question_type: str) -> str:
    """生成英文题干 (home-care robot 场景), 避免中英混杂。

    规划类: 场景结构 + 任务指令(目标) + 上一步动作 + "下一步做什么"。
    只给最近一步动作 (Markov 形式), 不罗列完整历史, 避免文本泄露答案。
    感知类: 让机器人指出当前可见物体 (discriminator 侧再改写为判别式选择题)。
    """
    if question_type == QTYPE_PLANNING:
        structure = _room_structure_nl(ctx.spatial_context)
        instruction = (ctx.scene.global_task or "").strip().rstrip(".")
        current_room, _, _ = _resolve_rooms(ctx.task_instance, ctx.scene, ctx.next_step)
        previous = _previous_action_nl(ctx.task_instance, ctx.scene.simulation_step)
        previous_text = previous.rstrip(".") if previous else "none — the task has just started"
        current_line = f"You are currently in the {_humanize(current_room)}.\n" if current_room else ""

        # 全局(监控)摄像头所在房间: 多视角任务 (B/D/E) 才有; 单视角任务为空 → 不输出该行
        cam_rooms = _global_camera_rooms(ctx.task_instance)
        if not cam_rooms:
            cam_line = ""
        elif len(cam_rooms) == 1:
            cam_line = f"There is a surveillance camera in the {cam_rooms[0]}.\n"
        else:
            rooms_str = ", ".join(f"the {r}" for r in cam_rooms[:-1]) + f", and the {cam_rooms[-1]}"
            cam_line = f"There are surveillance cameras in {rooms_str}.\n"

        return (
            "You are a home-care robot.\n"
            f"You are in a room with structure {structure}.\n"
            f"{current_line}"
            f"{cam_line}"
            f"Now, your task is: {instruction}.\n"
            f"You have just completed: {previous_text}.\n"
            "Now what should you do next?"
        )
    return "Describe the objects present in the current scene and their states."


def _primary_pair(ctx: GenContext, question_type: str,
                  context: list[dict[str, Any]] | None = None) -> QRAPair:
    """主问题 (规划 / 通用感知) 共用的组装逻辑: 出一类一条。"""
    return _make_pair(
        ctx, question_type, question_type,
        _build_english_question(ctx, question_type),
        build_answer(question_type, ctx.scene, ctx.next_step, ctx.task_instance),
        build_reasoning(ctx.task_type, question_type, ctx.scene, ctx.next_step, ctx.llm),
        context=context,
    )


def _gen_planning(ctx: GenContext) -> list[QRAPair]:
    """规划类"主干"问题: 仅当主问题类型为规划时生成 (A~E 的 initial/during)。

    规划类涉及任务求解过程, 生成时把本轮之前的主干(规划)问题作为上下文携带。
    """
    if classify_question(ctx.task_type, ctx.strategy) != QTYPE_PLANNING:
        return []
    return [_primary_pair(ctx, QTYPE_PLANNING, context=ctx.backbone_context)]


def _gen_perception(ctx: GenContext) -> list[QRAPair]:
    """通用感知类主问题: 仅当主问题类型为感知时生成 (F/G 或 after)。

    非任务求解类问题: 不携带主干问题上下文 (context 为空)。
    """
    if classify_question(ctx.task_type, ctx.strategy) != QTYPE_PERCEPTION:
        return []
    return [_primary_pair(ctx, QTYPE_PERCEPTION)]


# 问题类型生成器注册表: 采样点内按此顺序"尝试"生成每一类问题。
# 规划类须在首位 (作为"主干"), 感知 / bbox 等非任务求解类紧随其后。
# (新增问题类型 = 在此追加一个生成器函数, 无需改动主流程)
QUESTION_GENERATORS: list[Callable[[GenContext], list[QRAPair]]] = [
    _gen_planning,
    _gen_perception,
    _gen_bbox,
]


# ======================================================================
# 读取适配: 合作方 JSON → 任务上下文 + 采样点 (扩展点 5)
# ======================================================================
def _parse_event_id(event_id: str) -> tuple[int, str]:
    """把 "step_001_pre" / "step_003_post" 解析为 (step 序号, phase)。"""
    m = re.match(r"step_(\d+)_(pre|post)", event_id or "")
    if not m:
        return 0, "post"
    return int(m.group(1)), m.group(2)


def build_room_topology(task_instance: dict[str, Any]) -> dict[str, Any]:
    """从任务实例的场景图提取房间拓扑 (连通图 + 距离), 作为空间上下文。

    数据源: task_instance 的 ``before_graph.navigation`` (回退 after_graph / debug.before_graph)。
    返回::

        {"rooms": ["bathroom_0", ...],                      # 所有房间名 (排序)
         "edges": [["bathroom_0", "bathroom_1", 9.8], ...]}  # [源房间, 目标房间, 距离(米, 1位小数)]

    无可用拓扑信息时返回空结构 (rooms / edges 为空)。
    """
    graph = (task_instance.get("before_graph")
             or task_instance.get("after_graph")
             or (task_instance.get("debug") or {}).get("before_graph")
             or {})
    nav = graph.get("navigation") or {}
    raw_edges = nav.get("room_edges") or []

    rooms: set[str] = set()
    edges: list[list[Any]] = []
    for e in raw_edges:
        src, tgt = e.get("source"), e.get("target")
        if not src or not tgt:
            continue
        rooms.add(src)
        rooms.add(tgt)
        dist = e.get("distance")
        edges.append([
            src,
            tgt,
            round(dist, 1) if isinstance(dist, (int, float)) else None,
        ])

    return {"rooms": sorted(rooms), "edges": edges}


def load_task_context(task_instance: dict[str, Any]) -> dict[str, Any]:
    """把 task_instance.json 拍平成翻译器需要的扁平任务上下文。

    任务信息集中在 ``task`` 子字典 (task_id/instruction/solution_plan/plan_objects)。
    """
    task = task_instance.get("task") or task_instance.get("task_environment", {}).get("task") or {}
    robot = task_instance.get("robot") or task_instance.get("task_environment", {}).get("robot") or {}
    nested = task_instance.get("task_instance") or {}

    env_type = (task.get("task_type") or nested.get("task_type")
                or task_instance.get("task_environment", {}).get("env_type") or "")
    instruction = task.get("instruction") or nested.get("instruction") or ""
    task_id = task.get("task_id") or nested.get("task_id") or task_instance.get("run_id") or ""
    run_id = task_instance.get("run_id") or task_instance.get("task_environment", {}).get("env_id") or ""
    robot_id = robot.get("robot_id") or "robot_0"

    # 符号层方案 (MOVE/PICK/PLACE/INTERACT/WAIT), 与 intro.md 实例格式一致
    plan = task.get("solution_plan") or task_instance.get("solution_plan") or []

    return {
        "task_id": task_id,
        "run_id": run_id,
        "task_type": env_type,
        "env_type": env_type,
        "primary_behavior_task": task.get("primary_behavior_task", ""),
        "instruction": instruction,
        "target_room": task.get("target_room", ""),
        "semantic_constraints": list(task.get("semantic_constraints") or []),
        "robot_id": robot_id,
        "robot_pose": robot.get("pose") or {},
        "robot_initial_room": robot.get("initial_room") or "",
        "plan_objects": list(task.get("plan_objects") or nested.get("plan_objects") or []),
        "task_objects": list(task.get("task_objects") or task_instance.get("task_objects") or []),
        "camera": list(task_instance.get("camera")
                       or task_instance.get("task_environment", {}).get("camera") or []),
        "solution_plan": list(plan),
        "room_topology": build_room_topology(task_instance),
    }


def _extract_scene(event: dict[str, Any]) -> dict[str, Any]:
    """从采样点提取图像路径 + 语义标签, 组织为 {视角: {paths, labels}}。

    视角命名对齐 intro.md: robot_primary → "robot_camera", 全局相机用其 camera_id。
    """
    scene: dict[str, Any] = {}
    rp = event.get("robot_primary") or {}
    if rp.get("paths"):
        scene["robot_camera"] = {"paths": rp["paths"], "labels": rp.get("labels", {})}
    for gc in event.get("global_cameras") or []:
        cid = gc.get("camera_id") or gc.get("room_id") or "global_camera"
        if gc.get("paths"):
            scene[cid] = {"paths": gc["paths"], "labels": gc.get("labels", {})}
    return scene


def _extract_grounding(event: dict[str, Any]) -> dict[str, Any]:
    """从采样点提取各视角 bbox, 组织为 {视角: {object_id: bbox_dict}}。

    bbox_dict 为 {"bbox_xyxy": [...], "image_size": [...], ...} (见 expert_result.json)。
    """
    grounding: dict[str, Any] = {}
    rp = event.get("robot_primary") or {}
    if rp.get("bboxes"):
        grounding["robot_camera"] = rp["bboxes"]
    for gc in event.get("global_cameras") or []:
        cid = gc.get("camera_id") or gc.get("room_id") or "global_camera"
        if gc.get("bboxes"):
            grounding[cid] = gc["bboxes"]
    return grounding


def load_sampling_scenes(expert_result: dict[str, Any], task_ctx: dict[str, Any]) -> list[StaticScene]:
    """把 expert_result.json 的 observation_events 转成 list[StaticScene] (采样点)。

    simulation_step 取"下一步应执行的 plan 步"下标:
      step_N 的 pre 帧 → 下标 N-1; step_N 的 post 帧 → 下标 N (下一步即 step_{N+1})。
    """
    plan = task_ctx.get("solution_plan") or []
    global_task = task_ctx.get("instruction") or ""
    global_target = [o.get("object_id") for o in task_ctx.get("plan_objects") or []]

    scenes: list[StaticScene] = []
    for ev in expert_result.get("observation_events") or []:
        event_id = ev.get("event_id", "")
        step_no, phase = _parse_event_id(event_id)
        next_idx = (step_no - 1) if phase == "pre" else step_no
        temporal = plan[next_idx] if 0 <= next_idx < len(plan) else None

        scenes.append(StaticScene(
            environment_name=task_ctx.get("run_id") or task_ctx.get("task_id") or "",
            observation_id=event_id,
            phase=phase,
            global_task=global_task,
            global_target=global_target,
            temporal_task=(temporal or {}).get("nl") if temporal else None,
            simulation_step=next_idx,
            temporal_target=[(temporal or {}).get("target_object")] if temporal else [],
            scene=_extract_scene(ev),
            scene_with_grounding=_extract_grounding(ev),
            object_poses=ev.get("object_poses", {}),
            robot_pose=ev.get("robot_pose", {}),
            robot_visible=list(ev.get("robot_visible") or []),
            global_visible=list(ev.get("global_visible") or []),
        ))
    return scenes


def load_json(path: str) -> dict[str, Any]:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ======================================================================
# 主流程: 任务实例迭代 → 采样点迭代 → 问题类型生成迭代
# ======================================================================
def translate_task_instance(task_instance: dict[str, Any],
                            scenes: list[StaticScene | dict[str, Any]],
                            llm: Any = None,
                            frames_root: str | None = None) -> list[QRAPair]:
    """把一个任务实例的所有采样帧翻译成 QRA-Pair 列表 ({Q, A, Reasoning})。

    三级嵌套结构:
      采样点迭代 (按时间顺序) → 问题类型生成迭代 (尝试生成每一类问题)
    """
    QuestionTemplateRegistry._seed()

    task_type = resolve_task_type(task_instance)
    plan = task_instance.get("solution_plan") or []

    # 空间上下文 (房间拓扑): 优先取 load_task_context 预计算的结果, 否则就地提取 (内联调用时)
    room_topology = task_instance.get("room_topology")
    if room_topology is None:
        room_topology = build_room_topology(task_instance)

    # 采样点迭代: 按时间顺序处理每一个静态采样点
    parsed = (raw if isinstance(raw, StaticScene) else StaticScene.from_dict(raw) for raw in scenes)
    ordered_scenes = sorted(parsed, key=lambda s: (s.simulation_step, s.phase != "pre"))

    pairs: list[QRAPair] = []
    backbone_history: list[dict[str, Any]] = []  # 已生成的主干(规划)问题, 按时间顺序累积
    for scene in ordered_scenes:
        strategy = classify_strategy(scene)
        t = scene.simulation_step
        next_step = plan[t] if 0 <= t < len(plan) else None

        base_id = f"{task_instance.get('run_id') or task_instance.get('task_id', 'task')}" \
                  f"__{scene.observation_id or scene.simulation_step}"
        ctx = GenContext(
            task_instance=task_instance,
            task_type=task_type,
            scene=scene,
            strategy=strategy,
            next_step=next_step,
            base_id=base_id,
            frames_root=frames_root,
            llm=llm,
            backbone_context=list(backbone_history),  # 本轮之前的主干问题快照
            spatial_context=room_topology,
        )

        # 问题类型生成迭代: 规划类在前(主干), 其余类型随后; 不适用则生成器返回空
        for generator in QUESTION_GENERATORS:
            generated = generator(ctx)
            pairs.extend(generated)
            # 主干(规划)问题生成后记入历史, 供后续轮次作为上下文
            for pair in generated:
                if pair.question_type == QTYPE_PLANNING:
                    backbone_history.append(_context_entry(pair))

    return pairs


def translate_expert_result(task_instance: dict[str, Any],
                            expert_result: dict[str, Any],
                            llm: Any = None,
                            frames_root: str | None = None) -> list[QRAPair]:
    """从合作方交付的两份 JSON dict 直接产出 QRA-Pair 列表。"""
    task_ctx = load_task_context(task_instance)
    scenes = load_sampling_scenes(expert_result, task_ctx)
    return translate_task_instance(task_ctx, scenes, llm, frames_root)


def translate_expert_results(task_instances: list[dict[str, Any]],
                             expert_results: list[dict[str, Any]],
                             llm: Any = None,
                             frames_root: str | None = None) -> list[QRAPair]:
    """任务实例迭代 (最外层): 逐个 (task_instance, expert_result) 翻译并汇总。

    三级嵌套结构的顶层: 任务实例迭代 → (translate_expert_result → 采样点迭代 → 问题类型生成迭代)
    """
    pairs: list[QRAPair] = []
    for task_instance, expert_result in zip(task_instances, expert_results):
        pairs.extend(translate_expert_result(task_instance, expert_result, llm, frames_root))
    return pairs


def translate_expert_result_files(task_instance_path: str,
                                  expert_result_path: str,
                                  llm: Any = None,
                                  frames_root: str | None = None) -> list[QRAPair]:
    """从合作方交付的两份 JSON 文件路径直接产出 QRA-Pair 列表。

    frames_root 缺省取 expert_result_path 所在目录 (内含 frames/<event_id>/...)。
    """
    if frames_root is None:
        frames_root = os.path.dirname(expert_result_path)
    return translate_expert_result(load_json(task_instance_path),
                                   load_json(expert_result_path),
                                   llm,
                                   frames_root)


# ======================================================================
# LLM 翻译模块: 结构化 QRA → 自然语言 (可读 + 可用于 zero-shot 推理)
# ======================================================================
def _ensure_llm(llm: Any = None) -> Any:
    """返回可用的 LLM 客户端。

    优先级: 显式传入的 ``llm`` > 环境变量 ``DASHSCOPE_API_KEY`` > 工程默认
    ``DEFAULT_LLM_API_KEY``。LLM 不可用 (缺 openai / 缺 key) 时返回 None,
    由调用方回退到规则拼装。
    """
    if llm is not None:
        return llm
    try:
        from llm_client import create_llm_client
    except ImportError:
        logger.warning("llm_client unavailable; NL translation disabled")
        return None
    api_key = os.environ.get("DASHSCOPE_API_KEY") or DEFAULT_LLM_API_KEY
    return create_llm_client(api_key=api_key)


def _rule_answer_nl(pair: QRAPair) -> str:
    """规则兜底: 把结构化答案 A 拼成一句英文 (LLM 不可用 / 失败时)。"""
    a = pair.A or {}
    kind = a.get("kind")
    if kind == "perception":
        objs = a.get("objects") or []
        return "The visible objects in the scene are: " + (", ".join(objs) if objs else "none") + "."
    if kind == "done":
        return a.get("message") or "The task is complete."
    if kind == "bbox":
        name = _humanize(a.get("category") or a.get("object_id") or "the object")
        if not a.get("visible", True):
            return f"The {name} is not visible in the current view."
        bbox = a.get("bbox_xyxy")
        return f"The bounding box of the {name} is {bbox}."
    # response / 兜底: 优先用类别 + bbox 定位 (不暴露物品 ID); MOVE 额外说明房间
    action = a.get("action", PRIMITIVE_WAIT)
    target = a.get("target_object")
    category = a.get("category") or ""
    bbox = a.get("bbox")
    room = a.get("room") or ""
    cross_room = bool(a.get("cross_room", False))
    name = category or (target or "").replace("_", " ") or "the object"
    if action == PRIMITIVE_MOVE:
        if cross_room and room:
            return f"Move to the {name} in {room}."
        if room:
            return f"Move to the {name} in the current room ({room})."
        if bbox:
            return f"Move to the {name} (bbox {bbox})."
        return f"Move to the {name}."
    if bbox:
        return f"{action} the {name} (bbox {bbox})."
    if category:
        return f"{action} the {name}."
    if target:
        return f"The robot should {action} the {target}."
    return f"The robot should {action}."


def translate_pair_to_nl(pair: QRAPair, llm: Any = None) -> QRAPair:
    """把一条结构化 QRA 翻译成自然语言, 原地填充 ``A_nl`` 与 ``Reasoning``。

    用 LLM 把结构化答案 ``A`` 改写为自然语言句子 ``A_nl``, 并生成推理链
    ``Reasoning`` (使样本可读、可直接用于 zero-shot 推理)。LLM 不可用或调用失败时
    回退到规则拼装, 保证不抛异常、不丢样本。
    """
    client = _ensure_llm(llm)
    if client is not None:
        try:
            from llm_client import translate_qra_to_nl
            result = translate_qra_to_nl(client, pair.to_dict())
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM QRA translation failed, fallback to rule-based: %s", exc)
            result = None
    else:
        result = None

    if pair.options:
        # 选择题模式: A_nl 已在 attach_mcq 中设为正确选项文本, 不再用 LLM 重写 (只补推理链)
        pass
    else:
        pair.A_nl = (result or {}).get("answer") if result else ""
        if not pair.A_nl:
            pair.A_nl = _rule_answer_nl(pair)

    reasoning = (result or {}).get("reasoning") if result else ""
    if reasoning:
        pair.Reasoning = [line.strip() for line in str(reasoning).splitlines() if line.strip()]
    return pair


def translate_pairs_to_nl(pairs: list[QRAPair], llm: Any = None) -> list[QRAPair]:
    """批量把结构化 QRA 翻译为自然语言 (可读 + zero-shot 推理), 原地修改并返回。

    主流程在导出数据集前调用一次, 复用同一个 LLM 客户端, 避免逐条重复建连。
    """
    client = _ensure_llm(llm)
    if client is None:
        for pair in pairs:
            if not pair.options:  # 选择题的 A_nl 已在 attach_mcq 设好, 不覆盖
                pair.A_nl = _rule_answer_nl(pair)
        return pairs
    for pair in pairs:
        translate_pair_to_nl(pair, client)
    return pairs


# ======================================================================
# 导出
# ======================================================================
def export(pairs: list[QRAPair], out_path: str) -> None:
    """把 QRA-Pair 列表写成 JSONL (一行一条, 便于流式读取)。"""
    with open(out_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Exported %d QRA pairs to %s", len(pairs), out_path)


# ======================================================================
# 导出到 generate_dataset: 输入输出分离 (Step1)
# ======================================================================
# 输出根目录: 一个任务实例 → 一个子文件夹 (内含 qra.json + images/), 与输入数据
# (task_instance/ 下的 task_instance.json / expert_result.json / frames) 分离。
DEFAULT_OUTPUT_ROOT = "/home2/jiaodian/BEHAVIOR-main/generate_dataset"

# 多视角任务形式 (B/D/E): 除机器人主视角外, 还需全局相机视角
MULTI_VIEW_TASK_TYPES = {TASK_MULTI_VIEW_ACTIVE, TASK_MULTI_VIEW_INSTRUCT, TASK_DISAMBIGUATION}

# 各视角下要转存的图片: 仅 rgb (分割图 sem/inst 不保留)
_VIEW_FILES = {"rgb": "rgb.png"}


def _sanitize_dirname(name: str) -> str:
    """把任务实例 id 转成安全的目录名 (仅保留字母/数字/_/-)。"""
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", name or "").strip("_")
    return cleaned or "task"


def _discover_views(event_id: str, frames_root: str) -> tuple[bool, list[str]]:
    """扫描 frames/<event_id>/ 目录, 返回 (是否含机器人主视角, 全局相机 id 列表)。"""
    event_dir = os.path.join(frames_root, "frames", event_id)
    robot_exists = os.path.isdir(os.path.join(event_dir, "robot_primary"))
    global_ids: list[str] = []
    global_dir = os.path.join(event_dir, "global")
    if os.path.isdir(global_dir):
        global_ids = sorted(
            name for name in os.listdir(global_dir)
            if os.path.isdir(os.path.join(global_dir, name))
        )
    return robot_exists, global_ids


def _required_views(task_type: str, robot_exists: bool,
                    global_ids: list[str]) -> list[tuple[str, str]]:
    """由任务形式决定该采样点所需视角: 返回 (视角名, 相对 event 目录的磁盘子路径) 列表。"""
    views: list[tuple[str, str]] = []
    if robot_exists:
        views.append(("robot_primary", "robot_primary"))
    if task_type in MULTI_VIEW_TASK_TYPES:
        for gid in global_ids:
            views.append((gid, os.path.join("global", gid)))
    return views


def _copy_view_images(event_id: str, view_name: str, rel_dir: str,
                      frames_root: str, subfolder: str) -> dict[str, Any]:
    """把某视角的图片从 frames_root 转存到输出子文件夹, 返回相对路径字典。

    返回形如 ``{"view": "...", "rgb": "images/...", ...}``; 仅当源文件存在时才写入键。
    """
    file_map = _VIEW_FILES
    src_base = os.path.join(frames_root, "frames", event_id, rel_dir)
    entry: dict[str, Any] = {"view": view_name}
    for key, fn in file_map.items():
        src = os.path.join(src_base, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(subfolder, "images", event_id, rel_dir, fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        entry[key] = os.path.join("images", event_id, rel_dir, fn)
    return entry


def export_task_dataset(pairs: list[QRAPair],
                        output_root: str = DEFAULT_OUTPUT_ROOT,
                        frames_root: str | None = None,
                        meta: dict[str, Any] | None = None) -> str | None:
    """把一个任务实例的所有 QRA 对 + 所需视角图片写出到 ``output_root/<task_id>/``。

    输入数据 (task_instance.json / expert_result.json / frames) 保持不变; 输出数据集中到
    generate_dataset 下的一个子文件夹, 实现输入输出分离::

        <task_id>/
        ├── qra.json                            # 全部采样点问题 (含相对图片路径)
        └── images/<event_id>/<view>/<file>     # 所需视角图片副本

    返回子文件夹绝对路径; 无样本时返回 None。
    """
    if not pairs:
        return None
    if frames_root is None:
        frames_root = os.getcwd()

    task_id = pairs[0].task_instance_id or "task"
    task_type = pairs[0].task_type or ""
    subfolder = os.path.join(output_root, _sanitize_dirname(task_id))
    os.makedirs(subfolder, exist_ok=True)

    # 逐采样点转存所需视角图片, 建立 event_id → 图片相对路径映射
    event_ids = sorted({p.event_id for p in pairs if p.event_id})
    image_map: dict[str, list[dict[str, Any]]] = {}
    for event_id in event_ids:
        robot_exists, global_ids = _discover_views(event_id, frames_root)
        image_map[event_id] = [
            _copy_view_images(event_id, view_name, rel_dir, frames_root, subfolder)
            for view_name, rel_dir in _required_views(task_type, robot_exists, global_ids)
        ]

    # 把图片相对路径写回每条样本
    for p in pairs:
        p.images = image_map.get(p.event_id, [])

    doc: dict[str, Any] = {
        "task_instance_id": task_id,
        "task_type": task_type,
        "num_samples": len(pairs),
        "samples": [p.to_dict() for p in pairs],
    }
    if meta:
        for k, v in meta.items():
            doc.setdefault(k, v)

    out_path = os.path.join(subfolder, "qra.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    logger.info("Exported %d QRA pairs + images to %s", len(pairs), out_path)
    return subfolder


def generate_task_dataset_files(task_instance_path: str,
                                expert_result_path: str,
                                output_root: str = DEFAULT_OUTPUT_ROOT,
                                llm: Any = None) -> str | None:
    """端到端: 两份 JSON 文件 → QRA 问题 → LLM 翻译为自然语言 → 导出到 generate_dataset 子文件夹。

    frames_root 缺省取 expert_result_path 所在目录 (内含 frames/<event_id>/...)。
    结构化答案 ``A`` 会经 LLM 翻译为自然语言 ``A_nl`` (并生成推理链 ``Reasoning``),
    使样本可读、可直接用于 zero-shot 推理; LLM 不可用时回退到规则拼装。
    """
    frames_root = os.path.dirname(expert_result_path)
    task_instance = load_json(task_instance_path)
    expert_result = load_json(expert_result_path)
    pairs = translate_expert_result(task_instance, expert_result, llm, frames_root)

    # 选择题改造: 基于场景真实数据生成干扰项 (非 LLM 启发式), 原地填充 pair.options/answer_index。
    # 先解析一次 LLM 客户端, 供选项措辞 (attach_mcq) 与答案翻译 (translate_pairs_to_nl) 复用;
    # 不可用时传 None, attach_mcq 内部回退规则渲染。
    client = _ensure_llm(llm)
    from distractor import attach_mcq
    task_ctx = load_task_context(task_instance)
    scenes_by_event = {s.observation_id: s for s in load_sampling_scenes(expert_result, task_ctx)}
    attach_mcq(pairs, task_instance, scenes_by_event, client)

    pairs = translate_pairs_to_nl(pairs, client)
    meta = {
        "run_id": expert_result.get("run_id") or task_instance.get("run_id") or "",
        "instruction": (task_instance.get("task") or {}).get("instruction") or "",
    }
    return export_task_dataset(pairs, output_root, frames_root, meta)


def generate_datasets(task_instance_paths: list[str],
                      expert_result_paths: list[str],
                      output_root: str = DEFAULT_OUTPUT_ROOT,
                      llm: Any = None) -> list[str]:
    """任务实例迭代 (最外层): 逐个 (task_instance, expert_result) 翻译并导出。

    返回所有已生成的子文件夹绝对路径列表。
    """
    subfolders: list[str] = []
    for ti_path, er_path in zip(task_instance_paths, expert_result_paths):
        sf = generate_task_dataset_files(ti_path, er_path, output_root, llm)
        if sf:
            subfolders.append(sf)
    return subfolders


# ======================================================================
# 示例用法
# ======================================================================
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    demo_dir = "../task_instance/demo_expert_beechwood1_deliver_drink_headonly"
    ti_path = os.path.join(demo_dir, "task_instance.json")
    er_path = os.path.join(demo_dir, "expert_result.json")

    if os.path.exists(ti_path) and os.path.exists(er_path):
        subfolder = generate_task_dataset_files(ti_path, er_path)
        print(f"数据集已生成: {subfolder}")
    else:
        # 内联示意 (真实数据来自合作方交付的 JSON)
        demo_task = {
            "task_id": "fire_task_001",
            "task_type": "Env-B",
            "instruction": "Resolve the fire emergency using the extinguisher.",
            "solution_plan": [
                {"step_id": 1, "primitive": "MOVE", "nl": "Move to extinguisher",
                 "target_object": "ext_1"},
                {"step_id": 2, "primitive": "PICK", "nl": "Pick up extinguisher",
                 "target_object": "ext_1"},
                {"step_id": 3, "primitive": "INTERACT", "nl": "Extinguish fire",
                 "target_object": "fire_1", "tool_object": "ext_1"},
            ],
        }
        demo_scenes = [
            StaticScene(global_task="灭火", temporal_task="捡起灭火器", simulation_step=0,
                        global_target=["ext_1", "fire_1"]),
            StaticScene(global_task="灭火", temporal_task="捡起灭火器", simulation_step=1,
                        global_target=["ext_1", "fire_1"]),
        ]
        pairs = translate_task_instance(demo_task, demo_scenes)

        for p in pairs:
            print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2))
