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

内部数据模型 ``QRAPair`` 含 ``{ Q, A, Reasoning, context, spatial_context, images, options, ... }``:
- ``Q``         题干 (自然语言)
- ``A``         答案 (动作 / 可见物体 / bbox, 结构化)
- ``Reasoning`` 推理链 (当前**未接入 LLM**, 预留空列表, 见 ``build_reasoning``)
- ``context``   先前主干(规划)问题的 (Q,A) 上下文 (仅规划类非空; 非任务求解类为空)
- ``spatial_context`` 空间上下文: 房间拓扑 (见 ``build_room_topology``, 对同一任务实例所有采样点相同)
- ``images``    该问所需视角图片 (相对输出子文件夹的路径, 由 ``export_task_dataset`` 填充)

最终导出到 ``qra.json`` 时经 ``QRAPair.to_simple_dict`` 精简, 仅保留自然语言
题干/选项/答案、图片路径与必要标识字段 (qra_id / task_type / question_type / category)。

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

from extract_feature import (
    STRUCTURAL_CATEGORIES,
    build_room_topology,
    derive_anomaly,
    disambiguation_candidates,
    global_camera_rooms,
    humanize,
    object_category,
    resolve_rooms,
)


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

# 问题类型 (目前区分 感知 / 规划 / bbox grounding / 预测; 总结类依赖 deltaSG, 暂不生成)
QTYPE_PERCEPTION = "perception"  # 感知类: 问"场景中有什么 / 发生了什么"
QTYPE_PLANNING = "planning"      # 规划类: 问"下一步该做什么"
QTYPE_BBOX = "bbox"              # 感知类(bbox grounding): 问机器人主视角中目标物品的 bounding box
QTYPE_PREDICTION = "prediction"  # 预测类: 给上一步动作+目标, 推断下一步并预测执行后的场景状态 (ΔSG 效应)

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
    is_retry: bool = False             # nav_retry 事件 (step_NNN_post_nav_retry_M), 不重复生成主干规划/预测

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
            is_retry=bool(data.get("is_retry", False)),
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

    def to_simple_dict(self) -> dict[str, Any]:
        """导出精简格式: 仅保留自然语言问题/选项/答案 + 图片 + 必要标识字段。

        相比 ``to_dict`` 去掉了结构化答案 A、推理链 Reasoning、上下文 context、
        空间拓扑 spatial_context 与采样细节 (sampling_strategy/simulation_step/event_id),
        选项也压平为纯文本数组 (正确项由 answer_index 指向)。
        """
        return {
            "qra_id": self.qra_id,
            "task_type": self.task_type,          # 任务类型 A~G
            "question_type": self.question_type,  # 问题类型 planning/perception/bbox
            "category": (self.A or {}).get("category") or "",  # 物品类别 (如 "bottle of water")
            "question": self.Q,
            "options": [opt.get("text", "") for opt in self.options],
            "answer": self.A_nl,
            "answer_index": self.answer_index,
            "reasoning": "\n".join(self.Reasoning) if self.Reasoning else "",
            "images": self.images,
        }


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


def build_answer(question_type: str, scene: StaticScene,
                 next_step: dict[str, Any] | None,
                 task_instance: dict[str, Any] | None = None,
                 task_type: str | None = None) -> dict[str, Any]:
    """组装答案 A。

    - 感知类 → 场景中可见物体列表
    - 消歧类 (E) → optimal_object + 候选/干扰项 + 下一步动作
    - 响应类 → 下一步动作 (无下一步则任务完成); 附 target 的类别与主视角 bbox,
      供 LLM 翻译用 bounding box 定位物品 (不暴露物品 ID)。主动响应 (A/B) 额外附异常上下文。

    [MLLM] 感知类答案目前只列物体 id; 日后可接入 MLLM 从 rgb/seg 图生成物体状态的
    自然语言描述 (如"瓶子放在下层橱柜上")。异常检测类 (G) 的异常标注见 ``_gen_anomaly``。
    """
    if question_type == QTYPE_PERCEPTION:
        return {
            "kind": "perception",
            "objects": _visible_objects(scene),
        }

    # 多源信息消歧 (E): 正确对象 + 候选/干扰项 (含 reason) + 下一步动作
    if task_type == TASK_DISAMBIGUATION:
        sr = (task_instance or {}).get("semantic_reasoning") or {}
        gt = sr.get("ground_truth") or {}
        optimal = gt.get("optimal_object")
        rejected = [
            {"object_id": rc.get("object_id"),
             "category": object_category(rc.get("object_id"), task_instance),
             "reason": rc.get("reason")}
            for rc in gt.get("rejected_candidates") or []
            if isinstance(rc, dict) and rc.get("object_id")
        ]
        if optimal:
            # 消歧主干(规划)题应跟随 solution_plan 每一步的实际操作对象:
            # ``optimal_object`` 是"该选哪个工具"的消歧答案 (如灭火器), 仅当该步 target
            # 恰好就是工具时两者重合; 后续步骤 (导航到着火点 / 用工具灭火) 的 target 会
            # 推进到着火物体 (如 picture), 故此处 target 取 next_step.target_object 优先。
            target = (next_step or {}).get("target_object") or optimal
            return {
                "kind": "disambiguation",
                "optimal_object": optimal,
                "target_object": target,
                "category": object_category(target, task_instance),
                "action": (next_step or {}).get("primitive", PRIMITIVE_WAIT),
                "tool_object": (next_step or {}).get("tool_object"),
                "bbox": _target_bbox(scene, target),
                "rejected_candidates": rejected,
            }

    if next_step is None:
        return {"kind": "done", "message": "任务已完成"}

    target = next_step.get("target_object")
    _, target_room, cross_room = resolve_rooms(task_instance, scene, next_step)
    answer = {
        "kind": "response",
        "action": next_step.get("primitive", PRIMITIVE_WAIT),
        "target_object": target,
        "tool_object": next_step.get("tool_object"),
        "nl": next_step.get("nl", ""),
        "category": object_category(target, task_instance),
        "bbox": _target_bbox(scene, target),
        "room": humanize(target_room),
        "cross_room": cross_room,
    }
    # 主动响应 (A/B): 附异常上下文 (异常由模型从视觉中自行发现, 题干不点名)
    if task_type in (TASK_SINGLE_VIEW_ACTIVE, TASK_MULTI_VIEW_ACTIVE):
        anomaly = (task_instance or {}).get("anomaly")
        if anomaly:
            answer["anomaly_object"] = anomaly.get("object_id")
            answer["anomaly_state"] = anomaly.get("state") or {}
            answer["anomaly_phase"] = anomaly.get("phase")
    return answer


# bbox 过滤阈值 (intro.md: bbox 大小需"适中")
BBOX_MIN_AREA_RATIO = 0.005   # 面积占比下限: 过小 = 太远/遮挡, 不采
BBOX_MAX_AREA_RATIO = 0.75    # 面积占比上限: 过大 = 几乎占满画面, 不采


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
        if not name or name in STRUCTURAL_CATEGORIES:
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
    name = humanize(candidate.get("category") or candidate.get("object_id") or "the object")
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
    cat = object_category(target, task_instance) or humanize(target)
    if prim == PRIMITIVE_MOVE:
        return f"Move to the {cat}."
    if prim == PRIMITIVE_PICK:
        return f"Pick up the {cat}."
    if prim == PRIMITIVE_PLACE:
        return f"Place the {carried or 'object'} on the {cat}."
    if prim == PRIMITIVE_INTERACT:
        tool = object_category(step.get("tool_object"), task_instance)
        if tool and tool != cat:
            return f"Operate the {cat} with the {tool}."
        return f"Operate the {cat}."
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
            carried = object_category(s.get("target_object"), task_instance) or humanize(s.get("target_object"))
    return _render_step_nl(plan[t - 1], task_instance, carried)


def _carried_object_id(task_instance: dict[str, Any], t: int) -> str | None:
    """返回 ``plan[:t]`` 中最近一次 PICK 的 target_object (object_id, 非类别); 无则 None。

    供预测题求 PLACE 的"被放置物": PLACE 步骤本身不含被放置物, 需回看此前最近一次 PICK
    (与 ``_previous_action_nl`` 的 carried 推断同源, 但返回 object_id 而非人类可读类别)。
    """
    plan = task_instance.get("solution_plan") or []
    for s in reversed(plan[:t]):
        if (s.get("primitive") or "") == PRIMITIVE_PICK:
            return s.get("target_object")
    return None


def _surveillance_cam_line(task_instance: dict[str, Any] | None) -> str:
    """渲染全局(监控)摄像头所在房间行 (多视角任务 B/D/E 才有; 单视角任务返回空串)。"""
    cam_rooms = global_camera_rooms(task_instance)
    if not cam_rooms:
        return ""
    if len(cam_rooms) == 1:
        return f"There is a surveillance camera in the {cam_rooms[0]}.\n"
    rooms_str = ", ".join(f"the {r}" for r in cam_rooms[:-1]) + f", and the {cam_rooms[-1]}"
    return f"There are surveillance cameras in {rooms_str}.\n"


def _build_english_question(ctx: GenContext, question_type: str) -> str:
    """生成英文题干 (home-care robot 场景), 避免中英混杂。

    规划类: 场景结构 + 任务指令(目标) + 上一步动作 + "下一步做什么"。
    预测类: 与规划类同结构, 但末句改为"执行下一步后场景会如何" (不给当前一步动作)。
    主动响应类 (A/B): 点明"可能存在需要处理的异常", 但不点名异常是什么/在哪里 (异常由模型
        从视觉观测自行发现); 不渲染 instruction 与上一步动作 (避免泄露灭火器/房间)。
    消歧类 (E): 任务指令 (保持原句) + 候选对象类别列表 + "选择正确对象并给出动作步骤"。
    只给最近一步动作 (Markov 形式), 不罗列完整历史, 避免文本泄露答案。
    感知类: 让机器人指出当前可见物体 (discriminator 侧再改写为判别式选择题)。
    """
    if question_type in (QTYPE_PLANNING, QTYPE_PREDICTION):
        structure = _room_structure_nl(ctx.spatial_context)
        current_room, _, _ = resolve_rooms(ctx.task_instance, ctx.scene, ctx.next_step)
        current_line = f"You are currently in the {humanize(current_room)}.\n" if current_room else ""
        cam_line = _surveillance_cam_line(ctx.task_instance)

        # 多源信息消歧 (E): instruction + 候选对象 + "选择正确对象"
        if ctx.task_type == TASK_DISAMBIGUATION:
            instruction = (ctx.scene.global_task or "").strip().rstrip(".")
            candidates = disambiguation_candidates(ctx.task_instance)
            cand_line = f"The candidate objects are: {', '.join(candidates)}.\n" if candidates else ""
            return (
                "You are a home-care robot.\n"
                f"You are in a room with structure {structure}.\n"
                f"{current_line}"
                f"{cam_line}"
                f"Your task is: {instruction}.\n"
                f"{cand_line}"
                "Select the correct object, give the action steps and your reasoning."
            )

        # 主动响应 (A/B): 泛化异常提示, 不点名异常的具体内容/位置
        if ctx.task_type in (TASK_SINGLE_VIEW_ACTIVE, TASK_MULTI_VIEW_ACTIVE):
            tail = (
                "Determine what needs to be handled and give the next step."
                if question_type == QTYPE_PLANNING
                else "After your next action, which of the following will describe the scene?"
            )
            return (
                "You are a home-care robot.\n"
                f"You are in a room with structure {structure}.\n"
                f"{current_line}"
                f"{cam_line}"
                "There may be an anomaly in the environment that needs to be handled.\n"
                f"{tail}"
            )

        # 指令遵循 (C/D): 明确指令 + 上一步动作
        instruction = (ctx.scene.global_task or "").strip().rstrip(".")
        previous = _previous_action_nl(ctx.task_instance, ctx.scene.simulation_step)
        previous_text = previous.rstrip(".") if previous else "none — the task has just started"
        tail = (
            "Now what should you do next?"
            if question_type == QTYPE_PLANNING
            else "After your next action, which of the following will describe the scene?"
        )
        return (
            "You are a home-care robot.\n"
            f"You are in a room with structure {structure}.\n"
            f"{current_line}"
            f"{cam_line}"
            f"Now, your task is: {instruction}.\n"
            f"You have just completed: {previous_text}.\n"
            f"{tail}"
        )
    return "Describe the objects present in the current scene and their states."


def _primary_pair(ctx: GenContext, question_type: str,
                  context: list[dict[str, Any]] | None = None) -> QRAPair:
    """主问题 (规划 / 通用感知) 共用的组装逻辑: 出一类一条。"""
    return _make_pair(
        ctx, question_type, question_type,
        _build_english_question(ctx, question_type),
        build_answer(question_type, ctx.scene, ctx.next_step, ctx.task_instance, ctx.task_type),
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


def _gen_anomaly(ctx: GenContext) -> list[QRAPair]:
    """异常检测题 (G 形式): 仅对主动响应任务 (A/B) 在初始采样点生成一条感知类问题。

    题干让模型从视觉观测中指出场景异常 (不点名异常是什么/在哪里); 答案 ``kind="anomaly"``
    携带异常物体 id / 状态 / 阶段, 供 distractor 生成判别式选择题。
    """
    if ctx.task_type not in (TASK_SINGLE_VIEW_ACTIVE, TASK_MULTI_VIEW_ACTIVE):
        return []
    if ctx.strategy != SAMPLING_INITIAL:
        return []
    anomaly = (ctx.task_instance or {}).get("anomaly")
    if not anomaly:
        return []
    question = "What anomaly or hazard do you detect in the scene?"
    answer = {
        "kind": "anomaly",
        "object_id": anomaly.get("object_id"),
        "objects": [anomaly.get("object_id")] if anomaly.get("object_id") else [],
        "category": anomaly.get("category"),
        "state": anomaly.get("state") or {},
        "phase": anomaly.get("phase"),
        "smoke_visible": anomaly.get("smoke_visible"),
        "flame_visible": anomaly.get("flame_visible"),
    }
    return [
        _make_pair(ctx, QTYPE_PERCEPTION, "anomaly", question, answer,
                   build_reasoning(ctx.task_type, QTYPE_PERCEPTION, ctx.scene, None, ctx.llm))
    ]


def _predict_effect(next_step: dict[str, Any], task_instance: dict[str, Any],
                    t: int) -> dict[str, Any] | None:
    """由下一步动作预测其执行后的结果状态 (ΔSG 效应, 纯符号规则)。

    仅覆盖 PICK / PLACE (参考 translator_deltasg.py ``_step_delta`` 的符号规则):
      - PICK  → 目标物体被机器人持有 (held)
      - PLACE → 被放置物(此前最近一次 PICK 的对象)放到支撑面上 (on support)

    注意: 预测只落在符号状态层 (held / on support), 不预测 bbox —— 机器人下一步的精确
    位姿与视角不确定, 视觉 bounding box 会随之大幅变化, 要求模型预测一个不确定视觉场景中
    的 bbox 不合理。故 effect 一律不带 bbox 锚点。不可预测时返回 None。
    """
    prim = next_step.get("primitive")
    target = next_step.get("target_object")
    if prim == PRIMITIVE_PICK:
        return {
            "object_id": target,
            "relation": "held",
            "support_id": None,
            "category": object_category(target, task_instance) or humanize(target),
            "support_category": None,
        }
    if prim == PRIMITIVE_PLACE:
        moved = _carried_object_id(task_instance, t)
        if not moved:
            return None
        return {
            "object_id": moved,
            "relation": "on",
            "support_id": target,
            "category": object_category(moved, task_instance) or humanize(moved),
            "support_category": object_category(target, task_instance) or humanize(target),
        }
    return None


def _gen_prediction(ctx: GenContext) -> list[QRAPair]:
    """预测类"主干"问题: 给上一步动作+目标, 让模型推断下一步并预测执行后的场景状态。

    仅覆盖 PICK / PLACE (MOVE/INTERACT/WAIT 暂不生成, 见计划); 预测落在符号状态层
    (held / on support), 不预测 bbox。非任务求解类问题: 不携带主干问题上下文 (context 为空),
    也不喂入 backbone (避免与本步 planning 互相泄露动作)。
    """
    # 主动响应 (A/B) 不生成预测题: 预测题"给上一步动作+目标"与主动响应"自行发现异常"相矛盾
    if ctx.task_type in (TASK_SINGLE_VIEW_ACTIVE, TASK_MULTI_VIEW_ACTIVE):
        return []
    if ctx.strategy != SAMPLING_DURING:
        return []
    next_step = ctx.next_step
    if not next_step or next_step.get("primitive") not in {PRIMITIVE_PICK, PRIMITIVE_PLACE}:
        return []
    t = ctx.scene.simulation_step
    effect = _predict_effect(next_step, ctx.task_instance, t)
    if effect is None:
        return []
    answer = {
        "kind": "prediction",
        "action": next_step.get("primitive"),
        "target_object": next_step.get("target_object"),
        "step_id": next_step.get("step_id"),
        "category": effect.get("category"),
        "effect": effect,
    }
    question = _build_english_question(ctx, QTYPE_PREDICTION)
    return [
        _make_pair(
            ctx, QTYPE_PREDICTION, QTYPE_PREDICTION, question, answer,
            build_reasoning(ctx.task_type, QTYPE_PREDICTION, ctx.scene, next_step, ctx.llm),
            context=[],
        )
    ]


# 问题类型生成器注册表: 采样点内按此顺序"尝试"生成每一类问题。
# 规划类须在首位 (作为"主干"), 感知 / bbox / 预测 等非任务求解类紧随其后。
# (新增问题类型 = 在此追加一个生成器函数, 无需改动主流程)
QUESTION_GENERATORS: list[Callable[[GenContext], list[QRAPair]]] = [
    _gen_planning,
    _gen_anomaly,
    _gen_perception,
    _gen_bbox,
    _gen_prediction,
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


def _is_retry_event(event_id: str) -> bool:
    """是否 nav_retry 事件 (step_NNN_post_nav_retry_M)。

    这类事件与同名 step 的 post 帧共享同一 simulation_step, 仅用于补充 bbox/perception 采样,
    不重复生成主干规划/预测问题 (见 translate_task_instance 的去重)。
    """
    return bool(re.search(r"_nav_retry_\d+", event_id or ""))


def load_task_context(task_instance: dict[str, Any]) -> dict[str, Any]:
    """把 task_instance.json 拍平成翻译器需要的扁平任务上下文。

    任务信息集中在 ``task`` 子字典 (task_id/instruction/solution_plan/plan_objects);
    异常源 (Env-B) 在 ``task_environment.state_changed_objects``, 消歧信息 (Env-C) 在
    ``task.semantic_reasoning``。
    """
    task = task_instance.get("task") or task_instance.get("task_environment", {}).get("task") or {}
    task_env = task_instance.get("task_environment") or {}
    robot = task_instance.get("robot") or task_env.get("robot") or {}
    nested = task_instance.get("task_instance") or {}

    env_type = (task.get("task_type") or nested.get("task_type")
                or task_env.get("env_type") or "")
    instruction = task.get("instruction") or nested.get("instruction") or ""
    task_id = task.get("task_id") or nested.get("task_id") or task_instance.get("run_id") or ""
    run_id = task_instance.get("run_id") or task_env.get("env_id") or ""
    robot_id = robot.get("robot_id") or "robot_0"

    # 符号层方案 (MOVE/PICK/PLACE/INTERACT/WAIT), 与 intro.md 实例格式一致
    plan = task.get("solution_plan") or task_instance.get("solution_plan") or []

    state_changed_objects = list(task_env.get("state_changed_objects") or [])
    semantic_reasoning = task.get("semantic_reasoning") or task_env.get("semantic_reasoning")

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
        "added_objects": list(task_instance.get("added_objects") or task_env.get("added_objects") or []),
        "camera": list(task_instance.get("camera") or task_env.get("camera") or []),
        "solution_plan": list(plan),
        "state_changed_objects": state_changed_objects,
        "semantic_reasoning": semantic_reasoning,
        "anomaly": derive_anomaly(state_changed_objects),
        "room_topology": build_room_topology(task_instance),
        # 场景图 (含 navigation.room_centers + nodes.pose/rooms), 供消歧题干生成空间特征。
        "before_graph": (task_instance.get("before_graph")
                         or task_instance.get("after_graph")
                         or (task_instance.get("debug") or {}).get("before_graph")
                         or {}),
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
            is_retry=_is_retry_event(event_id),
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
            # nav_retry 事件仅保留 bbox/perception 采样, 不重复生成主干规划/预测
            if scene.is_retry and generator in (_gen_planning, _gen_prediction):
                continue
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
        name = humanize(a.get("category") or a.get("object_id") or "the object")
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
            return f"Find the {name} in {room}."
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
                      frames_root: str, subfolder: str,
                      task_key: str | None = None) -> dict[str, Any]:
    """把某视角的图片从 frames_root 转存到输出子文件夹, 返回相对路径字典。

    返回形如 ``{"view": "...", "rgb": "images/...", ...}``; 仅当源文件存在时才写入键。
    ``task_key`` 非空时, 图片落盘于 ``images/<task_key>/<event_id>/...`` 并在相对路径中
    保留该前缀 (批次模式下按任务环境分目录); 为空时退化为单任务布局 ``images/<event_id>/...``。
    """
    file_map = _VIEW_FILES
    src_base = os.path.join(frames_root, "frames", event_id, rel_dir)
    entry: dict[str, Any] = {"view": view_name}
    img_prefix = os.path.join("images", task_key) if task_key else "images"
    for key, fn in file_map.items():
        src = os.path.join(src_base, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(subfolder, img_prefix, event_id, rel_dir, fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        entry[key] = os.path.join(img_prefix, event_id, rel_dir, fn)
    return entry


def export_task_dataset(pairs: list[QRAPair],
                        output_root: str = DEFAULT_OUTPUT_ROOT,
                        frames_root: str | None = None) -> str | None:
    """把一个任务实例的所有 QRA 对 + 所需视角图片写出到 ``output_root/<task_id>/``。

    输入数据 (task_instance.json / expert_result.json / frames) 保持不变; 输出数据集中到
    generate_dataset 下的一个子文件夹, 实现输入输出分离::

        <task_id>/
        ├── qra.json                            # 精简后的全部采样点问题 (见 QRAPair.to_simple_dict)
        └── images/<event_id>/<view>/<file>     # 所需视角图片副本

    每条样本仅保留自然语言题干/选项/答案、图片路径与必要标识字段 (qra_id + 类别)。
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
        "num_samples": len(pairs),
        "samples": [p.to_simple_dict() for p in pairs],
    }

    out_path = os.path.join(subfolder, "qra.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    logger.info("Exported %d QRA pairs + images to %s", len(pairs), out_path)
    return subfolder


def _jsonl_line(pair: QRAPair, task_key: str | None = None) -> dict[str, Any]:
    """把一条 QRA 精简为批次 jsonl 的一行: 精简字段 + 唯一任务键 / task_instance_id / event_id。

    ``task_key`` (形如 ``<scene>__<sample_id>``) 是唯一标识, 与 images/ 目录键一致;
    ``task_instance_id`` 为源 task_id (跨场景可能重名), ``event_id`` 为采样点。三者组合
    便于在合并成单一 jsonl 后回溯每个问题所属的任务环境与采样点。
    """
    line = pair.to_simple_dict()
    return {
        "task_key": task_key or "",
        "task_instance_id": pair.task_instance_id,
        "event_id": pair.event_id,
        **line,
    }


def export_batch_dataset(batch_items: list[dict[str, Any]],
                         output_dir: str) -> str | None:
    """把一批任务实例的 QRA 对汇总写到一个 jsonl + images/<task_key>/ 目录树。

    ``batch_items`` 每项为 ``{"task_key", "pairs", "frames_root", "task_type"}``;
    ``task_key`` 用于图片按任务环境分目录 (形如 ``<scene>__<sample_id>``)。输出结构::

        <output_dir>/
        ├── qra.jsonl                     # 全部问题, 一行一条
        └── images/<task_key>/<event_id>/<view>/<file>   # 图片按任务环境分目录

    返回 output_dir; 无样本时返回 None。
    """
    usable = [it for it in batch_items if it.get("pairs")]
    if not usable:
        return None
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "qra.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for item in usable:
            task_key = item["task_key"]
            pairs: list[QRAPair] = item["pairs"]
            frames_root = item.get("frames_root") or os.getcwd()
            task_type = item.get("task_type") or (pairs[0].task_type if pairs else "")

            event_ids = sorted({p.event_id for p in pairs if p.event_id})
            image_map: dict[str, list[dict[str, Any]]] = {}
            for event_id in event_ids:
                robot_exists, global_ids = _discover_views(event_id, frames_root)
                image_map[event_id] = [
                    _copy_view_images(event_id, vn, rd, frames_root, output_dir, task_key=task_key)
                    for vn, rd in _required_views(task_type, robot_exists, global_ids)
                ]
            for p in pairs:
                p.images = image_map.get(p.event_id, [])
                fh.write(json.dumps(_jsonl_line(p, task_key=task_key), ensure_ascii=False) + "\n")

    logger.info("Exported batch (%d tasks) to %s", len(usable), jsonl_path)
    return output_dir


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
    return export_task_dataset(pairs, output_root, frames_root)


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


def discover_task_instances(data_root: str,
                            env_types: tuple[str, ...] = ("Env-A", "Env-B", "Env-C")) -> list[dict[str, Any]]:
    """遍历 ``data_root/Env-X/<scene>/<sample_id>/`` 目录树, 定位 generation.json + expert/expert_result.json。

    返回 ``[{env_type, scene, sample_id, generation_path, expert_result_path}, ...]``,
    跳过缺失任一文件的目录。``frames_root`` 可由 ``dirname(expert_result_path)=<sample>/expert``
    正确推导, 无需在此处理。
    """
    discovered: list[dict[str, Any]] = []
    for env_type in env_types:
        env_dir = os.path.join(data_root, env_type)
        if not os.path.isdir(env_dir):
            continue
        for scene in sorted(os.listdir(env_dir)):
            scene_dir = os.path.join(env_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for sample_id in sorted(os.listdir(scene_dir)):
                sample_dir = os.path.join(scene_dir, sample_id)
                gen_path = os.path.join(sample_dir, "generation.json")
                er_path = os.path.join(sample_dir, "expert", "expert_result.json")
                if os.path.isfile(gen_path) and os.path.isfile(er_path):
                    discovered.append({
                        "env_type": env_type,
                        "scene": scene,
                        "sample_id": sample_id,
                        "generation_path": gen_path,
                        "expert_result_path": er_path,
                    })
    return discovered


def generate_datasets_from_data_root(data_root: str,
                                     output_root: str = DEFAULT_OUTPUT_ROOT,
                                     llm: Any = None) -> list[str]:
    """从底层数据文件夹批量生成数据集 (``data_root`` 指向 Env-A/B/C 的父目录)。

    逐条 (generation.json, expert/expert_result.json) 调 ``generate_task_dataset_files`` 导出;
    返回所有已生成子文件夹的绝对路径列表。
    """
    subfolders: list[str] = []
    for inst in discover_task_instances(data_root):
        sf = generate_task_dataset_files(
            inst["generation_path"], inst["expert_result_path"], output_root, llm)
        if sf:
            subfolders.append(sf)
    return subfolders


def generate_batch_dataset(instances: list[dict[str, Any]],
                           output_dir: str,
                           llm: Any = None) -> str | None:
    """从一批 ``discover_task_instances`` 返回的实例生成批次数据集 (单一 qra.jsonl + 按任务环境分目录的图片)。

    复用单实例生成链路 (translate → attach_mcq → NL 翻译), 但不逐实例导出, 而是汇总后经
    ``export_batch_dataset`` 一次性写出; 图片目录 key 用 ``<scene>__<sample_id>`` 避免
    run_id 跨场景重名冲突。
    """
    from distractor import attach_mcq

    client = _ensure_llm(llm)
    batch_items: list[dict[str, Any]] = []
    for inst in instances:
        gen_path = inst["generation_path"]
        er_path = inst["expert_result_path"]
        frames_root = os.path.dirname(er_path)
        task_instance = load_json(gen_path)
        expert_result = load_json(er_path)
        pairs = translate_expert_result(task_instance, expert_result, llm, frames_root)

        task_ctx = load_task_context(task_instance)
        scenes_by_event = {s.observation_id: s for s in load_sampling_scenes(expert_result, task_ctx)}
        attach_mcq(pairs, task_instance, scenes_by_event, client)
        translate_pairs_to_nl(pairs, client)

        task_key = _sanitize_dirname(f"{inst.get('scene', '')}__{inst.get('sample_id', '')}")
        batch_items.append({
            "task_key": task_key,
            "pairs": pairs,
            "frames_root": frames_root,
            "task_type": pairs[0].task_type if pairs else "",
        })

    return export_batch_dataset(batch_items, output_dir)


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
