"""
distractor.py — 选择题干扰项生成 (基于场景真实数据, 非 LLM 启发式)

定位
----
配合 translator.py 把"开放式问题"改造成 N 选 1 选择题。核心约束 (两条铁律):

1. **内容不编造**  —— 每个干扰项原子 (房间 / 物体 / bbox / 动作原语 / 支撑面 / 工具)
   都必须来自场景真实数据 (``before_graph`` 场景图 / ``solution_plan`` / 采样点可见性
   / 分割 bbox)。LLM 只负责"措辞", 永远不负责"想出"一个物体或房间。
2. **可行但错误**  —— 干扰项必须是在仿真里物理/语义可行、但相对正确答案是错的。可行性
   由场景图里真实存在的 affordance 判定 (见 ``SceneIndex.affordance``)。

干扰项来源 (按 question_type 分发, 见 ``_generate_pair_options``)
------------------------------------------------------------------
- **planning** (动作): 对正确动作三元组 (primitive, target, [tool], room) 做"单点替换":
    时序混淆 (solution_plan 的其它步骤) / 原语替换 / 目标物体替换 / 房间替换(MOVE)
    / 支撑面替换(PLACE) / 工具替换(INTERACT)。每个候选须通过可行性闸门。
- **perception** (可见物体): 正确项 = 当前可见物体; 干扰项 = 真实存在但当前不可见的物体
    (优先同房间 / 同类别, 更挑战)。
- **bbox** (bounding box): 正确项 = 目标物体归一化 bbox; 干扰项 = 同视角其它真实物体 bbox
    / 正确 bbox 的确定性扰动。

依赖
----
纯标准库, 无 numpy / LLM 硬依赖。不 import translator (避免循环导入); 需要 translator 侧
把 ``scenes_by_event`` (event_id → StaticScene) 传入 ``attach_mcq`` 以支撑 bbox 干扰项。

使用
----
    from distractor import attach_mcq
    attach_mcq(pairs, task_instance_raw, scenes_by_event={...}, llm=None)
    # => 原地给每个 pair 填上 pair.options / pair.answer_index / pair.A_nl
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# 动作原语 (与 translator.py 一致)
PRIMITIVE_MOVE = "MOVE"
PRIMITIVE_PICK = "PICK"
PRIMITIVE_PLACE = "PLACE"
PRIMITIVE_INTERACT = "INTERACT"
PRIMITIVE_WAIT = "WAIT"

# 问题类型 (与 translator.py 一致)
QTYPE_PLANNING = "planning"
QTYPE_PERCEPTION = "perception"
QTYPE_BBOX = "bbox"
QTYPE_PREDICTION = "prediction"

# 选择题选项数: 1 正确 + (NUM_OPTIONS-1) 干扰
NUM_OPTIONS = 4

# bbox 归一化到 [0, BBOX_NORM_SCALE] (与 translator.BBOX_NORM_SCALE 一致)
BBOX_NORM_SCALE = 1000

# bbox 扰动平移幅度 (相对宽/高的比例): 在 [MIN, MAX] 内采样。
# MAX 越小越近误、越难与真值区分; MIN 保证干扰框与真值有可辨差异 (避免与真值几乎重合)。
# 用于 bbox 题的近误干扰框, 以及 planning 题"同动作、bbox 近误"的混淆干扰项。
BBOX_PERTURB_MIN_SHIFT = 0.05
BBOX_PERTURB_MAX_SHIFT = 0.15

# interaction kind → 该物体可行的动作原语集合 (可行性闸门的核心)
_INTERACTION_PRIMITIVES: dict[str, set[str]] = {
    "manipulable": {PRIMITIVE_PICK, PRIMITIVE_INTERACT, PRIMITIVE_MOVE},
    "articulable": {PRIMITIVE_INTERACT, PRIMITIVE_MOVE},
    "controllable": {PRIMITIVE_INTERACT, PRIMITIVE_MOVE},
    "none": {PRIMITIVE_MOVE},
    "agent": set(),
}

# 抓取尺寸上限 (米): 原生场景图里 "manipulable" 是粗粒度标签 (bookcase/bed/sofa 也算),
# 必须叠加尺寸闸门, 否则会生成 "拿起书柜" 这类物理不可行的干扰项。
GRASP_MAX_EXTENT = 0.4


def _graspable(aff: dict[str, Any]) -> bool:
    """判定物体是否可被机器人抓取 (PICK 可行性)。

    闸门: manipulable 且 非支撑面 且 尺寸足够小。任务物体 (added via delta_sg) 无尺寸信息,
    默认可抓; 原生物体按 3D bbox extent 判定, 超过 GRASP_MAX_EXTENT 视为不可抓。
    """
    if aff.get("interaction") != "manipulable":
        return False
    if aff.get("supports_on_top"):
        return False
    ext = aff.get("extent")
    if ext is not None and ext > GRASP_MAX_EXTENT:
        return False
    return True


# 各原语对"目标物体" affordance 的要求 (目标替换时的候选过滤器)
_PRIMITIVE_TARGET_FILTER: dict[str, Any] = {
    PRIMITIVE_PICK: _graspable,
    PRIMITIVE_INTERACT: lambda aff: aff.get("interaction") in {
        "manipulable", "articulable", "controllable"},
    PRIMITIVE_MOVE: lambda aff: True,  # 任何物体都可导航接近
}

# 结构/背景类别 (非"物品", 不用于干扰项; 与 translator.NON_OBJECT_CATEGORIES 对齐)
_STRUCTURAL_CATEGORIES = {"walls", "floors", "ceilings", "background", "robot"}

# 语义角色中表示"支撑面"的标签 (delta_sg.nodes / plan_objects 的 semantic_role(s))
_SUPPORT_ROLES = {"task_support", "delivery_destination", "support"}


def _is_structural(aff: dict[str, Any]) -> bool:
    """是否为结构/背景类别 (不作为干扰项候选)。"""
    return aff.get("category") in _STRUCTURAL_CATEGORIES


# ======================================================================
# 场景索引: task_instance → 房间 / 物体 / affordance / 空间关系
# ======================================================================
@dataclass
class SceneIndex:
    """从 task_instance 一次性抽取的"真实场景素材"索引, 干扰项生成器的数据底座。"""

    rooms: list[str] = field(default_factory=list)                       # 房间名 (排序)
    room_edges: list[tuple[str, str, float | None]] = field(default_factory=list)  # (src, tgt, dist)
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)     # object_id -> 原始 node
    affordance: dict[str, dict[str, Any]] = field(default_factory=dict)  # object_id -> {category, rooms, interaction, supports_on_top, supports_inside, reachable}
    objects_by_category: dict[str, list[str]] = field(default_factory=dict)  # category -> [object_id]
    objects_by_room: dict[str, list[str]] = field(default_factory=dict)  # room -> [object_id]
    near: dict[str, list[str]] = field(default_factory=dict)             # object_id -> 邻近物体
    support_surfaces: list[str] = field(default_factory=list)            # 可作为支撑面的物体
    task_objects: set[str] = field(default_factory=set)                  # 任务相关物体 (plan/task/added)


def _strip_room_prefix(name: str) -> str:
    """去掉节点 id 的 ``room::`` 前缀 (edges 的 source/target 有时带前缀)。"""
    return name[6:] if name.startswith("room::") else name


def build_scene_index(task_instance: dict[str, Any]) -> SceneIndex:
    """构建场景索引, 合并 (before_graph 原生物体) + (任务相关物体)。

    原生物体来自 ``before_graph`` (回退 after_graph / debug.before_graph); 任务相关物体来自
    ``task.plan_objects`` / ``task_objects`` / ``added_objects`` (它们常不在 before_graph 中,
    但却是正确答案的目标物, 必须纳入以便查类别/房间/affordance)。
    """
    idx = SceneIndex()
    graph = (task_instance.get("before_graph")
             or task_instance.get("after_graph")
             or (task_instance.get("debug") or {}).get("before_graph")
             or {})

    # 房间 + 房间间距离
    nav = graph.get("navigation") or {}
    room_centers = nav.get("room_centers") or {}
    idx.rooms = sorted(room_centers.keys())
    for e in nav.get("room_edges") or []:
        src, tgt = e.get("source"), e.get("target")
        if src and tgt:
            d = e.get("distance")
            idx.room_edges.append((
                src, tgt, round(d, 1) if isinstance(d, (int, float)) else None))

    # 原生物体节点
    for n in graph.get("nodes") or []:
        if n.get("type") != "object":
            continue
        oid = n.get("id") or n.get("name")
        if not oid:
            continue
        sem = n.get("semantic") or {}
        rec = sem.get("receptacle") or {}
        rooms = list(n.get("rooms") or [])
        ext = (n.get("bbox") or {}).get("extent")
        aff = {
            "category": n.get("category") or oid,
            "rooms": rooms,
            "interaction": (sem.get("interaction") or {}).get("kind") or "none",
            "supports_on_top": bool(rec.get("supports_on_top")),
            "supports_inside": bool(rec.get("supports_inside")),
            "reachable": True,
            "extent": max(ext) if ext else None,
        }
        _index_object(idx, oid, n, aff)

    # 空间关系边: near (物体邻近, 供"邻近但错误"的物体干扰项)
    for e in graph.get("edges") or []:
        if e.get("relation") == "near":
            src = _strip_room_prefix(e.get("source") or "")
            tgt = _strip_room_prefix(e.get("target") or "")
            idx.near.setdefault(src, []).append(tgt)
            idx.near.setdefault(tgt, []).append(src)

    # 任务相关物体 (plan_objects / task_objects / added_objects / delta_sg.nodes)
    _ingest_task_objects(idx, task_instance)

    # 支撑面: supports_on_top 的家具 (排除结构/背景类别)
    for oid, aff in idx.affordance.items():
        if aff.get("supports_on_top") and not _is_structural(aff):
            idx.support_surfaces.append(oid)

    return idx


def _index_object(idx: SceneIndex, oid: str, node: dict[str, Any], aff: dict[str, Any]) -> None:
    """把一个物体登记进索引的各映射。"""
    idx.objects[oid] = node
    idx.affordance[oid] = aff
    cat = aff["category"]
    idx.objects_by_category.setdefault(cat, []).append(oid)
    for r in aff["rooms"]:
        idx.objects_by_room.setdefault(r, []).append(oid)


def _ingest_task_objects(idx: SceneIndex, task_instance: dict[str, Any]) -> None:
    """把任务相关物体并入索引 (类别/房间/affordance 供正确答案与干扰项共用)。

    任务资产的 affordance (interaction.kind / receptacle) 来自 ``delta_sg.nodes``
    (added_objects 本身没有 ``semantic`` 字段); 类别/房间由 plan_objects / task_objects /
    added_objects 补充。
    """
    task = task_instance.get("task") or {}
    # 每条: (object_id, category, room, semantic, semantic_roles)
    entries: list[tuple[str | None, str | None, str, dict[str, Any], list[str]]] = []

    # 1) delta_sg.nodes —— 任务资产权威 affordance
    for n in (task_instance.get("delta_sg") or {}).get("nodes") or []:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        entries.append((
            n.get("id"), n.get("category"), n.get("room_id") or "",
            n.get("semantic") or {}, list(n.get("semantic_roles") or [])))

    # 2) plan_objects / task_objects / added_objects —— 补充类别/房间/角色
    for obj in task.get("plan_objects") or []:
        if isinstance(obj, dict):
            roles = list(obj.get("semantic_roles") or [])
            if obj.get("semantic_role"):
                roles.append(obj["semantic_role"])
            entries.append((obj.get("object_id"), obj.get("category"), obj.get("room") or "", {}, roles))
    for obj in task_instance.get("task_objects") or []:
        if isinstance(obj, dict):
            entries.append((obj.get("object_id"), obj.get("category"), "", {}, []))
    for obj in task_instance.get("added_objects") or []:
        if isinstance(obj, dict):
            entries.append((obj.get("object_id"), obj.get("category"),
                            obj.get("room_id") or obj.get("room") or "",
                            {}, list(obj.get("semantic_roles") or [])))

    for oid, cat, room, sem, roles in entries:
        if not oid:
            continue
        rec = sem.get("receptacle") or {}
        inter = (sem.get("interaction") or {}).get("kind") or "none"
        # 已存在的物体只补房间, 不覆盖原生 affordance
        if oid in idx.affordance:
            if room and room not in idx.affordance[oid]["rooms"]:
                idx.affordance[oid]["rooms"].append(room)
            continue
        supports_on_top = bool(rec.get("supports_on_top")) or bool(set(roles) & _SUPPORT_ROLES)
        idx.task_objects.add(oid)
        _index_object(idx, oid, {"id": oid, "category": cat, "rooms": [room] if room else []}, {
            "category": cat or oid,
            "rooms": [room] if room else [],
            "interaction": inter,
            "supports_on_top": supports_on_top,
            "supports_inside": bool(rec.get("supports_inside")),
            "reachable": True,
            "extent": None,  # 任务物体无原生 bbox, 尺寸未知 (默认可抓, 见 _graspable)
        })


# ======================================================================
# 小工具: 类别/房间可读化 / 确定性 RNG / 去重
# ======================================================================
def _humanize(name: str | None) -> str:
    """把 id 风格名 (下划线分隔) 转为人类可读 (空格分隔)。"""
    return (name or "").replace("_", " ").strip()


def _cat(index: SceneIndex, object_id: str | None) -> str:
    """object_id → 人类可读类别名 (回退 object_id 本身)。"""
    if not object_id:
        return ""
    aff = index.affordance.get(object_id)
    if aff and aff.get("category"):
        return _humanize(aff["category"])
    return _humanize(object_id)


def _room_of(index: SceneIndex, object_id: str | None) -> str:
    """object_id → 所在房间 (取第一个; 未知返回空串)。"""
    if not object_id:
        return ""
    aff = index.affordance.get(object_id)
    return (aff.get("rooms") or [""])[0] if aff else ""


def _rng(seed: str, salt: str = "") -> "random.Random":
    """由 (qra_id, salt) 构造确定性 RNG, 保证同一实例多次生成结果一致。"""
    h = hashlib.md5(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return random.Random(int(h, 16))


def _action_key(a: dict[str, Any]) -> tuple:
    """动作干扰项去重键。bbox 也纳入键, 使"同动作、不同 bbox"的混淆项可被区分。"""
    bbox = tuple(a.get("object_bbox") or [])
    return (a.get("kind"), a.get("primitive"), a.get("target_object"),
            a.get("tool_object"), a.get("room"), bbox)


def _option_key(s: dict[str, Any]) -> tuple:
    """选项去重键。"""
    kind = s.get("kind")
    if kind == "action":
        return _action_key(s)
    if kind == "object":
        return (kind, s.get("object_id"))
    if kind == "bbox":
        return (kind, tuple(s.get("bbox_xyxy") or []))
    if kind == "state":
        return (kind, s.get("object_id"), s.get("relation"), s.get("support_id"))
    return (kind,)


# ======================================================================
# 可行性闸门 + 候选池
# ======================================================================
def _feasible_primitives(object_id: str | None, index: SceneIndex) -> set[str]:
    """某物体可行的动作原语集合 (来自 interaction kind)。未知物体仅可 MOVE。"""
    aff = index.affordance.get(object_id)
    inter = (aff or {}).get("interaction") or "none"
    return _INTERACTION_PRIMITIVES.get(inter, {PRIMITIVE_MOVE})


def _target_swap_pool(prim: str, target: str | None, index: SceneIndex) -> list[str]:
    """目标替换候选: 同原语、换一个真实可行的其它物体 (优先同房间)。"""
    if prim == PRIMITIVE_PLACE:
        pool = [o for o in index.support_surfaces if o != target]
    else:
        filt = _PRIMITIVE_TARGET_FILTER.get(prim, lambda aff: True)
        pool = [o for o in index.affordance
                if o != target and filt(index.affordance[o]) and not _is_structural(index.affordance[o])]
    room = _room_of(index, target)
    pool.sort(key=lambda o: (0 if (room and room in index.affordance[o].get("rooms", [])) else 1, o))
    return pool


def _room_swap_pool(room: str, index: SceneIndex) -> list[str]:
    """房间替换候选: 真实存在的其它房间 (优先"同名不同序号", 如 bathroom_0 vs bathroom_1)。"""
    base = re.sub(r"_\d+$", "", room)
    rooms = [r for r in index.rooms if r != room]
    rooms.sort(key=lambda r: (0 if re.sub(r"_\d+$", "", r) == base else 1, r))
    return rooms


def _support_swap_pool(target: str | None, index: SceneIndex) -> list[str]:
    """支撑面替换候选 (PLACE): 其它真实支撑面。"""
    return [o for o in index.support_surfaces if o != target]


def _tool_swap_pool(tool: str | None, target: str | None, index: SceneIndex) -> list[str]:
    """工具替换候选 (INTERACT): 其它真实可交互物体。"""
    pool = [o for o in index.affordance
            if o != tool and o != target
            and index.affordance[o].get("interaction") in {"manipulable", "articulable", "controllable"}]
    return pool


def _sample_distinct(cands: list[tuple[dict[str, Any], dict[str, Any]]],
                     k: int,
                     rng: "random.Random",
                     key_fn: Any = None,
                     bucket_fn: Any = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """去重后按"来源类型"取样, 保证干扰项跨维度多样 (时序/原语/目标/房间…)。

    类型(桶)顺序与桶内顺序均用 rng 打乱后再轮询, 使干扰项类型的组合跨样本随机
    (同一样本内仍跨类型多样、且结果由 qra_id 种子确定可复现)。
    ``key_fn`` / ``bucket_fn`` 可覆盖默认的去重键 (动作键) 与多样性桶 (source.type),
    供其它结构化选项 (如状态命题, 桶改用 source.subtype) 复用同一取样逻辑。
    """
    key_fn = key_fn or (lambda a, src: _action_key(a))
    bucket_fn = bucket_fn or (lambda a, src: src.get("type", "other"))

    seen: set = set()
    uniq: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for a, src in cands:
        key = key_fn(a, src)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a, src))
    if len(uniq) <= k:
        return uniq

    buckets: "OrderedDict[str, list]" = OrderedDict()
    for a, src in uniq:
        buckets.setdefault(bucket_fn(a, src), []).append((a, src))

    lists: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    for _, items in buckets.items():
        rng.shuffle(items)
        lists.append(items)
    rng.shuffle(lists)

    picked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    i = 0
    while len(picked) < k and any(lists):
        lst = lists[i % len(lists)]
        if lst:
            picked.append(lst.pop(0))
        i += 1
    return picked


# ======================================================================
# 干扰项生成: 规划类 (动作)
# ======================================================================
def _action_distractors(step: dict[str, Any], index: SceneIndex,
                        plan: list[dict[str, Any]], t: int,
                        qra_id: str, rng: "random.Random") -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """为规划类下一步动作生成干扰项 (动作三元组单点替换)。返回 [(action, source)]。"""
    prim = step.get("primitive") or PRIMITIVE_WAIT
    target = step.get("target_object")
    tool = step.get("tool_object")
    room = step.get("target_room")
    cands: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def _action(**kw: Any) -> dict[str, Any]:
        a = {"kind": "action", "primitive": prim, "target_object": target,
             "tool_object": tool, "room": room}
        a.update(kw)
        return a

    # 1) 时序混淆: solution_plan 的其它步骤 (真实、可行、但当前时刻错误)
    for i, s in enumerate(plan):
        if i == t:
            continue
        a = {"kind": "action", "primitive": s.get("primitive"), "target_object": s.get("target_object"),
             "tool_object": s.get("tool_object"), "room": s.get("target_room")}
        if _action_key(a) == _action_key(_action()):
            continue
        cands.append((a, {"type": "temporal_confusion", "step_id": s.get("step_id")}))

    # 2) 原语替换: 对正确目标换一个可行原语
    if target:
        for alt in sorted(_feasible_primitives(target, index)):
            if alt == prim:
                continue
            a = _action(primitive=alt,
                        tool_object=tool if alt == PRIMITIVE_INTERACT else None)
            cands.append((a, {"type": "primitive_swap", "feasible_for": target}))

    # 3) 目标替换: 同原语换真实物体
    for oid in _target_swap_pool(prim, target, index):
        cands.append((_action(target_object=oid, tool_object=None, room=_room_of(index, oid)),
                      {"type": "object_swap", "object_id": oid}))

    # 4) 房间替换 (MOVE): 换真实房间
    if prim == PRIMITIVE_MOVE and room:
        for r in _room_swap_pool(room, index):
            cands.append((_action(target_object=None, tool_object=None, room=r),
                          {"type": "room_swap", "room": r}))

    # 5) 支撑面替换 (PLACE)
    if prim == PRIMITIVE_PLACE:
        for oid in _support_swap_pool(target, index):
            cands.append((_action(target_object=oid, tool_object=None, room=_room_of(index, oid)),
                          {"type": "support_swap", "support": oid}))

    # 6) 工具替换 (INTERACT)
    if prim == PRIMITIVE_INTERACT:
        for oid in _tool_swap_pool(tool, target, index):
            cands.append((_action(tool_object=oid), {"type": "tool_swap", "tool": oid}))

    return _sample_distinct(cands, NUM_OPTIONS - 1, rng)


# ======================================================================
# 干扰项生成: 感知类 (可见物体)
# ======================================================================
def _perception_options(pair: Any, index: SceneIndex,
                        llm: Any = None) -> tuple[list[dict[str, Any]], int]:
    """感知题 → 判别式选择题: 正确项 = 当前可见物体, 干扰项 = 真实存在但不可见物体。"""
    visible = list(pair.A.get("objects") or [])
    if not visible:
        return [], -1

    # 正确项: 优先任务相关物体, 否则取字典序第一个可见物体
    correct = next((o for o in sorted(visible) if o in index.task_objects), sorted(visible)[0])
    correct_aff = index.affordance.get(correct, {})

    # 干扰项: 真实存在但不可见的物体, 按 (同房间, 同类别) 优先
    dist_pool = [o for o in index.affordance
                 if o not in set(visible) and not _is_structural(index.affordance[o])]
    croom = _room_of(index, correct)
    ccat = correct_aff.get("category")

    def rank(o: str) -> tuple:
        aff = index.affordance[o]
        return (0 if (croom and croom in aff.get("rooms", [])) else 1,
                0 if aff.get("category") == ccat else 1,
                o)

    dist_pool.sort(key=rank)
    distractors = [
        {"structured": {"kind": "object", "object_id": o,
                        "category": index.affordance[o].get("category"), "room": _room_of(index, o)},
         "source": {"type": "not_visible", "object_id": o}}
        for o in dist_pool  # 全量交给 _finalize 去重+截断 (避免过早截断导致选项不足)
    ]
    correct_opt = {"structured": {"kind": "object", "object_id": correct,
                                  "category": correct_aff.get("category"), "room": croom},
                   "source": {"type": "visible_object", "object_id": correct}}

    # 判别式题干
    pair.Q = "Which of the following objects is visible in the current view?"
    return _finalize(pair.qra_id, correct_opt, distractors, index, llm=llm, question=pair.Q)


# ======================================================================
# 干扰项生成: bbox 类
# ======================================================================
def _normalize_bbox(bbox_xyxy: list[Any] | None, image_size: list[Any] | None) -> list[int] | None:
    """像素 bbox → [0, BBOX_NORM_SCALE] 归一化 (与 translator._normalize_bbox 一致)。"""
    if not bbox_xyxy or not image_size:
        return None
    W, H = image_size
    if not W or not H:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    return [int(round(x1 / W * BBOX_NORM_SCALE)), int(round(y1 / H * BBOX_NORM_SCALE)),
            int(round(x2 / W * BBOX_NORM_SCALE)), int(round(y2 / H * BBOX_NORM_SCALE))]


def _robot_camera_grounding(scene: Any) -> dict[str, Any]:
    """取该采样点机器人主视角的 grounding (object_id → bbox dict)。"""
    return (getattr(scene, "scene_with_grounding", None) or {}).get("robot_camera") or {}


def _annotate_action_bbox(action: dict[str, Any], cam: dict[str, Any]) -> None:
    """给动作选项的目标物体附上主视角归一化 bbox (可见时), 存入 ``object_bbox``。

    仅在 robot_camera grounding 里能找到该目标物体且有合法 bbox 时写入; 否则不加 (不可见)。
    """
    oid = action.get("target_object")
    if not oid:
        return
    b = cam.get(oid)
    if not isinstance(b, dict):
        return
    nb = _normalize_bbox(b.get("bbox_xyxy"), b.get("image_size"))
    if nb is not None:
        action["object_bbox"] = nb


def _perturb_bbox(bbox: list[int], rng: "random.Random") -> list[int] | None:
    """对正确 bbox 做确定性近误扰动 (平移幅度在 [MIN, MAX] 内, 保证在画面内且不等于真值)。

    近误扰动 (与真值大面积重叠但仍有可辨差异) 比粗扰动更难区分, 用于 bbox 题的干扰框
    与 planning 题的 bbox 混淆项。MIN 下限避免生成与真值几乎重合、难以判定的干扰框。
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    for _ in range(20):
        fx = rng.uniform(BBOX_PERTURB_MIN_SHIFT, BBOX_PERTURB_MAX_SHIFT) * rng.choice((-1, 1))
        fy = rng.uniform(BBOX_PERTURB_MIN_SHIFT, BBOX_PERTURB_MAX_SHIFT) * rng.choice((-1, 1))
        dx = int(w * fx)
        dy = int(h * fy)
        nx1 = max(0, min(BBOX_NORM_SCALE - 1, x1 + dx))
        nx2 = max(0, min(BBOX_NORM_SCALE - 1, x2 + dx))
        ny1 = max(0, min(BBOX_NORM_SCALE - 1, y1 + dy))
        ny2 = max(0, min(BBOX_NORM_SCALE - 1, y2 + dy))
        if nx2 <= nx1:
            nx2 = nx1 + 1
        if ny2 <= ny1:
            ny2 = ny1 + 1
        cand = [nx1, ny1, nx2, ny2]
        if cand != list(bbox):
            return cand
    return None


def _bbox_options(pair: Any, index: SceneIndex, scene: Any,
                  llm: Any = None) -> tuple[list[dict[str, Any]], int]:
    """bbox 题 → 选择题: 正确项 = 目标归一化 bbox; 干扰项 = 其它真实 bbox / 扰动 bbox。"""
    a = pair.A or {}
    if a.get("kind") != "bbox" or not a.get("visible", True):
        return [], -1
    bbox = a.get("bbox_xyxy")
    category = a.get("category") or a.get("object_id")
    correct_oid = a.get("object_id")
    if not bbox or len(bbox) != 4:
        return [], -1

    rng = _rng(pair.qra_id, "bbox")
    distractors: list[dict[str, Any]] = []

    # 1) 同视角其它真实物体的 bbox (真实且易混淆)
    if scene is not None:
        cam = (getattr(scene, "scene_with_grounding", None) or {}).get("robot_camera") or {}
        for oid, b in cam.items():
            if oid == correct_oid or not isinstance(b, dict):
                continue
            nb = _normalize_bbox(b.get("bbox_xyxy"), b.get("image_size"))
            if nb is None or nb == list(bbox):
                continue
            distractors.append({"structured": {"kind": "bbox", "bbox_xyxy": nb},
                                "source": {"type": "other_object_bbox", "object_id": oid}})

    # 2) 正确 bbox 的确定性扰动
    for _ in range(10):
        nb = _perturb_bbox(list(bbox), rng)
        if nb is not None:
            distractors.append({"structured": {"kind": "bbox", "bbox_xyxy": nb},
                                "source": {"type": "perturbed_bbox"}})

    correct_opt = {"structured": {"kind": "bbox", "bbox_xyxy": list(bbox)},
                   "source": {"type": "ground_truth_bbox"}}
    pair.Q = f"What is the bounding box of the {_humanize(category)} in the robot's primary view?"
    return _finalize(pair.qra_id, correct_opt, distractors, index, llm=llm, question=pair.Q)


# ======================================================================
# 规划类选项组装 (正确项 + 干扰项)
# ======================================================================
def _planning_options(pair: Any, index: SceneIndex,
                      plan: list[dict[str, Any]], scene: Any = None,
                      llm: Any = None) -> tuple[list[dict[str, Any]], int]:
    """规划题 → 选择题: 正确项 = 下一步动作, 干扰项 = 单点替换的可行动作 + bbox 混淆。

    目标物体在该采样点机器人主视角可见时, 附上其归一化 bbox (object_bbox), 供选项文本
    内联在物品名后做图像定位消歧 (见 ``_target_nl``)。干扰项类型用 qra_id 种子随机化。
    """
    t = pair.simulation_step
    step = plan[t] if 0 <= t < len(plan) else None
    if step is None or pair.A.get("kind") == "done":
        return [], -1

    rng = _rng(pair.qra_id, "distractors")
    cam = _robot_camera_grounding(scene)

    correct = {"kind": "action", "primitive": step.get("primitive") or PRIMITIVE_WAIT,
               "target_object": step.get("target_object"), "tool_object": step.get("tool_object"),
               "room": step.get("target_room")}
    _annotate_action_bbox(correct, cam)
    correct_opt = {"kind": "action", "structured": correct,
                   "source": {"type": "solution_plan_step", "step_id": step.get("step_id")}}

    dist_opts = [{"kind": "action", "structured": a, "source": src}
                 for a, src in _action_distractors(step, index, plan, t, pair.qra_id, rng)]
    for d in dist_opts:
        _annotate_action_bbox(d["structured"], cam)

    # bbox 混淆: 若正确目标在主视角可见, 追加"同动作、bbox 近误"的干扰项 (与正确项仅
    # bbox 数值不同, 是 planning 里最易混淆的一类, 考察细粒度定位)。
    correct_bbox = correct.get("object_bbox")
    if isinstance(correct_bbox, (list, tuple)) and len(correct_bbox) == 4:
        for _ in range(2):
            nb = _perturb_bbox(list(correct_bbox), rng)
            if nb is None:
                continue
            confused = dict(correct)
            confused["object_bbox"] = nb
            dist_opts.append({"kind": "action", "structured": confused,
                              "source": {"type": "bbox_confusion"}})

    return _finalize(pair.qra_id, correct_opt, dist_opts, index,
                     llm=llm, question=getattr(pair, "Q", "") or "")


# ======================================================================
# 预测类选项组装 (正确项 + 干扰项)
# ======================================================================
def _prediction_options(pair: Any, index: SceneIndex,
                        plan: list[dict[str, Any]], llm: Any = None) -> tuple[list[dict[str, Any]], int]:
    """预测题 → 选择题: 正确项 = 执行下一步动作后的结果状态命题; 干扰项 = 状态混淆。

    状态命题 (state proposition) 不含动作动词, 迫使模型先推断动作、再判断后果 (否则预测题
    会退化成 planning)。正确项来自 translator 预计算的 effect (符号状态, 不带 bbox); 干扰项
    source.type 均为 "state_confusion", 靠 subtype 区分动作层错误 (support / object) 与
    效应层错误 (possession / relation), 便于 post-hoc 拆分两类错误指标。
    """
    a = pair.A or {}
    if a.get("kind") != "prediction":
        return [], -1
    effect = a.get("effect") or {}
    moved = effect.get("object_id")
    relation = effect.get("relation")
    if not moved or relation not in ("held", "on"):
        return [], -1
    support = effect.get("support_id")
    rng = _rng(pair.qra_id, "prediction")

    def _state(object_id: str | None, relation: str, support_id: str | None = None) -> dict[str, Any]:
        return {
            "kind": "state",
            "object_id": object_id,
            "category": _cat(index, object_id) or None,
            "relation": relation,
            "support_id": support_id,
            "support_category": _cat(index, support_id) if support_id else None,
        }

    t = pair.simulation_step
    step_id = (plan[t] or {}).get("step_id") if 0 <= t < len(plan) else None
    correct_state = _state(moved, relation, support)
    correct_opt = {"kind": "state", "structured": correct_state,
                   "source": {"type": "solution_plan_step", "step_id": step_id}}

    # 干扰项候选 (structured, source), 全部来自真实场景素材 (支撑面 / 可抓物体 / bbox)。
    # 后续用 _sample_distinct 按 subtype 分桶轮询取样, 保证干扰项跨维度多样。
    cands: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if relation == "on":
        # 支撑面错误 (动作层): 被放置物放到错的支撑面
        for oid in _support_swap_pool(support, index):
            cands.append((_state(moved, "on", oid), {"type": "state_confusion", "subtype": "support"}))
        # 物体错误 (动作层): 错的物体放到正确支撑面
        for oid in _target_swap_pool(PRIMITIVE_PICK, moved, index):
            cands.append((_state(oid, "on", support), {"type": "state_confusion", "subtype": "object"}))
        # 归属错误 (效应层): 物体仍被机器人持有
        cands.append((_state(moved, "held"), {"type": "state_confusion", "subtype": "possession"}))
    else:  # held (PICK)
        # 归属错误 (效应层): 物体仍在原地 (未被持有)
        cands.append((_state(moved, "unchanged"), {"type": "state_confusion", "subtype": "possession"}))
        # 关系错误 (效应层): 物体落在真实支撑面上而非被持有 (动作对、后果错)
        for oid in index.support_surfaces:
            cands.append((_state(moved, "on", oid), {"type": "state_confusion", "subtype": "relation"}))
        # 物体错误 (动作层): 别的物体被持有
        for oid in _target_swap_pool(PRIMITIVE_PICK, moved, index):
            cands.append((_state(oid, "held"), {"type": "state_confusion", "subtype": "object"}))

    picked = _sample_distinct(cands, NUM_OPTIONS - 1, rng,
                              key_fn=lambda a, src: _option_key(a),
                              bucket_fn=lambda a, src: src.get("subtype", "other"))
    dist_opts = [{"kind": "state", "structured": a, "source": src} for a, src in picked]

    return _finalize(pair.qra_id, correct_opt, dist_opts, index,
                     llm=llm, question=getattr(pair, "Q", "") or "")


# ======================================================================
# 选项渲染 (结构化 → 自然语言)
# ======================================================================
def _target_nl(a: dict[str, Any], index: SceneIndex) -> str:
    """目标物体短语: 类别 + (主视角可见时的) bbox, 直接跟在物品名后面用于定位消歧。"""
    tgt = _cat(index, a.get("target_object"))
    b = a.get("object_bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        tgt = f"{tgt} (bbox {list(b)})"
    return tgt


def _action_nl(a: dict[str, Any], index: SceneIndex) -> str:
    """动作 → 自然语言一句话 (英文, 规则兜底)。目标物体带 bbox (见 ``_target_nl``)。"""
    prim = a.get("primitive")
    target = a.get("target_object")
    tool = a.get("tool_object")
    room = a.get("room")
    tgt = _target_nl(a, index)
    if prim == PRIMITIVE_MOVE:
        if room and target:
            return f"Move to the {tgt} in {_humanize(room)}"
        if room:
            return f"Move to {_humanize(room)}"
        if target:
            return f"Move to the {tgt}"
        return "Move"
    if prim == PRIMITIVE_PICK:
        return f"Pick up the {tgt}"
    if prim == PRIMITIVE_PLACE:
        return f"Place the object on the {tgt}"
    if prim == PRIMITIVE_INTERACT:
        return f"Use the {_cat(index, tool)} to operate the {tgt}" if tool else f"Operate the {tgt}"
    if prim == PRIMITIVE_WAIT:
        return "Wait"
    return f"{prim} the {tgt}"


def _object_nl(s: dict[str, Any], index: SceneIndex) -> str:
    """物体 → 自然语言 (类别 + 房间定位, 英文, 规则兜底)。"""
    cat = _humanize(s.get("category")) if s.get("category") else _humanize(s.get("object_id"))
    room = _humanize(s.get("room"))
    return f"the {cat} in {room}" if room else f"the {cat}"


def _bbox_nl(s: dict[str, Any]) -> str:
    """bbox → 自然语言 (左上/右下坐标)。"""
    b = s.get("bbox_xyxy")
    return f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]" if b and len(b) == 4 else "?"


def _state_nl(s: dict[str, Any], index: SceneIndex) -> str:
    """状态命题 → 自然语言 (英文, 规则兜底, 未来时)。

    不含动作动词 (只有结果状态), 是预测题"先推断动作、再判断后果"的关键约束。
    """
    obj = _humanize(s.get("category")) if s.get("category") else _cat(index, s.get("object_id"))
    relation = s.get("relation")
    if relation == "held":
        return f"The {obj} will be held by the robot."
    if relation == "unchanged":
        return f"The {obj} will remain where it is."
    # "on"
    support = (_humanize(s.get("support_category")) if s.get("support_category")
               else _cat(index, s.get("support_id")))
    return f"The {obj} will be on the {support}"


def _render_option(structured: dict[str, Any], index: SceneIndex) -> str:
    """按结构化选项的 kind 渲染成自然语言选项文本 (规则兜底)。"""
    kind = structured.get("kind")
    if kind == "action":
        return _action_nl(structured, index)
    if kind == "object":
        return _object_nl(structured, index)
    if kind == "bbox":
        return _bbox_nl(structured)
    if kind == "state":
        return _state_nl(structured, index)
    return str(structured)


# ======================================================================
# 选项措辞 (LLM): 把结构化选项 (整合信息) 措辞成自然语言
# ======================================================================
# LLM 的唯一职责是"措辞"——把结构化选项写成人话。铁律: 实体名称逐字保留, 语义零改动,
# 否则退回规则渲染。这样既能满足"LLM 根据整合信息转自然语言"的要求, 又守住内容不编造。
_OPTION_NL_SYSTEM = """\
You are a phrasing assistant for embodied-AI multiple-choice questions. Given the structured \
information of each option (action / object / bounding box / state), rewrite it as a concise, \
natural, accurate English option phrase or short sentence.

Hard constraints (follow all of them):
1. Entity names (object categories, room names) must be preserved verbatim: do not translate, \
rename, add, or drop them. For example, "bottle of water" must appear exactly as-is.
2. Only do phrasing — never change the meaning: do not change the action, object, or room, and \
do not add or omit information.
3. Action primitive semantics: PICK = "Pick up", PLACE = "Place ... on", MOVE = "Move to", \
INTERACT = "Operate / Use the <tool> to operate the <target>", WAIT = "Wait". \
Important: MOVE means navigating to the target's location — always write "Move to the <target>"; \
never write "Move the <target>", which would wrongly mean carrying it.
4. State proposition semantics (future tense, no action verb): relation "held" = \
"The <object> will be held by the robot"; "on" = "The <object> will be on the <support>"; \
"unchanged" = "The <object> will remain where it is". Keep the future-tense state wording, \
never rewrite it into an action instruction.
5. For bbox options, output the coordinates verbatim as [x1, y1, x2, y2]; do not alter any number.
6. One sentence per option, consistent style and similar length; do not hint which option is correct.
7. Output only JSON in the form: {"texts": ["option 1", "option 2", ...]}, with the same length \
and order as the input options."""


def _option_payload(structured: dict[str, Any], index: SceneIndex) -> dict[str, Any]:
    """把结构化选项压缩成"整合信息"字典 (人类可读), 作为 LLM 措辞的输入。

    注意: 这里只做 id → 类别/房间的人类可读映射 (``_cat`` / ``_humanize``), 不改动语义;
    因此 LLM 拿到的就是场景里真实存在的实体名, 无法"想出"不存在的物体。
    """
    kind = structured.get("kind")
    if kind == "action":
        prim = structured.get("primitive")
        target = _cat(index, structured.get("target_object")) or None
        b = structured.get("object_bbox")
        if target and isinstance(b, (list, tuple)) and len(b) == 4:
            target = f"{target} (bbox {list(b)})"  # bbox 跟在物品名后面, 供 LLM 措辞保留
        return {
            "type": "action",
            "action": prim,
            "target": target,
            "tool": _cat(index, structured.get("tool_object")) or None,
            "room": _humanize(structured.get("room")) or None,
        }
    if kind == "object":
        cat = structured.get("category")
        return {
            "type": "object",
            "name": _humanize(cat) if cat else _humanize(structured.get("object_id")),
            "room": _humanize(structured.get("room")) or None,
        }
    if kind == "bbox":
        return {"type": "bbox", "bbox": list(structured.get("bbox_xyxy") or [])}
    if kind == "state":
        relation = structured.get("relation")
        obj = structured.get("category") or _cat(index, structured.get("object_id"))
        payload: dict[str, Any] = {"type": "state", "object": obj, "relation": relation}
        if relation == "on":
            support = structured.get("support_category") or _cat(index, structured.get("support_id"))
            if support:
                payload["support"] = support
        return payload
    return {"type": "other", "raw": str(structured)}


def _bbox_nl_exact(bbox: list[Any]) -> str:
    """bbox 的标准文本形态 (用于校验 LLM 是否逐字保留了坐标)。"""
    return f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"


def _llm_render_options(options: list[dict[str, Any]], question: str,
                        index: SceneIndex, llm: Any) -> list[str] | None:
    """让 LLM 把一整道题的所有选项一次性措辞成自然语言。失败返回 None (调用方回退规则渲染)。

    返回与 ``options`` 等长的文本列表; 任一选项失败 / 文本重复 / bbox 坐标被改动 → 整题回退,
    保证选择题可辨识、答案不受措辞影响。
    """
    if llm is None or not options:
        return None
    call = getattr(llm, "call", None)
    if not callable(call):
        return None

    payloads = [_option_payload(o["structured"], index) for o in options]
    user_prompt = (
        f"Question: {question or '(no question stem)'}\n\n"
        f"Rewrite each of the following {len(payloads)} options as a natural-language phrase "
        f"(do not number them A/B/C/D):\n"
        f"{json.dumps(payloads, ensure_ascii=False)}"
    )
    try:
        result = call(_OPTION_NL_SYSTEM, user_prompt, json_mode=True)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    texts = result.get("texts")
    if not isinstance(texts, list) or len(texts) != len(options):
        return None

    cleaned = []
    for i, t in enumerate(texts):
        s = str(t).strip() if t is not None else ""
        if not s:
            return None
        # 坐标必须逐字保留, 否则整题回退 (数值是正确答案的关键, 不能由 LLM 重写)
        structured = options[i]["structured"]
        if structured.get("kind") == "bbox":
            exact = _bbox_nl_exact(list(structured.get("bbox_xyxy") or []))
            if exact not in s:
                return None
        elif structured.get("kind") == "action":
            b = structured.get("object_bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                exact = _bbox_nl_exact(list(b))
                if exact not in s:
                    return None
        cleaned.append(s)
    if len(set(cleaned)) != len(cleaned):
        return None
    return cleaned


# ======================================================================
# 选项合成: 去重 + 打乱 + 编号 + 渲染
# ======================================================================
def _finalize(qra_id: str, correct_opt: dict[str, Any],
              dist_opts: list[dict[str, Any]], index: SceneIndex,
              llm: Any = None, question: str = "") -> tuple[list[dict[str, Any]], int]:
    """把正确项 + 干扰项合成一份选择题选项列表, 返回 (options, answer_index)。

    选项文本优先由 LLM 依据结构化整合信息措辞 (``_llm_render_options``), LLM 不可用/失败/
    坐标被改动时回退到规则渲染 (``_render_option``), 保证不丢样本、答案不受措辞影响。
    """
    # 去重: 干扰项 vs 正确项 & 彼此 (结构化键 + 渲染文本, 避免"同类别多实例"渲染成同文案;
    # 动作文本已在物品名后内联 bbox, 使"同类别不同实例"若 bbox 不同仍可区分)
    seen = {_option_key(correct_opt["structured"])}
    seen_text = {_render_option(correct_opt["structured"], index)}
    uniq: list[dict[str, Any]] = []
    for d in dist_opts:
        k = _option_key(d["structured"])
        if k in seen:
            continue
        text = _render_option(d["structured"], index)
        if text in seen_text:
            continue
        seen.add(k)
        seen_text.add(text)
        uniq.append(d)

    rng = _rng(qra_id, "shuffle")
    rng.shuffle(uniq)                 # 随机化"哪些干扰项入选" (类型组合跨样本随机)
    uniq = uniq[:NUM_OPTIONS - 1]

    opts = [correct_opt] + uniq
    rng.shuffle(opts)
    ans_idx = next(i for i, o in enumerate(opts) if o is correct_opt)

    # LLM 措辞 (整题一次调用); 失败则整题回退规则渲染 (bbox 已内联在物品名后, 不经二次拼接)。
    llm_texts = _llm_render_options(opts, question, index, llm)

    labels = "ABCDEFGH"
    for i, o in enumerate(opts):
        o["id"] = labels[i] if i < len(labels) else str(i)
        o["is_correct"] = (o is correct_opt)
        o["text"] = llm_texts[i] if llm_texts else _render_option(o["structured"], index)
        o.pop("structured", None)  # 结构化体仅在内部使用, 输出保留 source/text/is_correct
    return opts, ans_idx


# ======================================================================
# 分发: 一条 pair → 选择题选项
# ======================================================================
def _generate_pair_options(pair: Any, index: SceneIndex,
                           plan: list[dict[str, Any]], scene: Any,
                           llm: Any = None) -> tuple[list[dict[str, Any]], int]:
    """按 question_type 分发到对应生成器。返回 (options, answer_index) 或 ([], -1)。"""
    qt = pair.question_type
    if qt == QTYPE_PLANNING:
        return _planning_options(pair, index, plan, scene, llm)
    if qt == QTYPE_PERCEPTION:
        return _perception_options(pair, index, llm)
    if qt == QTYPE_BBOX:
        return _bbox_options(pair, index, scene, llm)
    if qt == QTYPE_PREDICTION:
        return _prediction_options(pair, index, plan, llm)
    return [], -1


# ======================================================================
# 主入口: 给整批 pair 附加选择题选项 (原地修改)
# ======================================================================
def attach_mcq(pairs: list[Any], task_instance: dict[str, Any],
               scenes_by_event: dict[str, Any] | None = None, llm: Any = None) -> None:
    """给每条 QRA pair 原地填充 ``options`` / ``answer_index`` / ``A_nl``。

    Args:
        pairs: translator 产出的 QRAPair 列表 (原地修改)。
        task_instance: 原始 task_instance.json dict (含 before_graph / solution_plan)。
        scenes_by_event: {event_id: StaticScene} 映射 (bbox 干扰项需要, 可缺省)。
        llm: LLM 客户端 (LLMClient). 仅用于把结构化选项措辞成自然语言 (内容仍由场景生成);
            传 None 或调用失败时回退规则渲染。

    不适用选择题的 pair (如任务完成 / 无可见物体) 会保留 options=[] / answer_index=-1,
    交由审计脚本标记, 不静默降级。
    """
    index = build_scene_index(task_instance)
    plan = task_instance.get("solution_plan") or []
    for pair in pairs:
        scene = (scenes_by_event or {}).get(getattr(pair, "event_id", ""))
        options, ans_idx = _generate_pair_options(pair, index, plan, scene, llm)
        pair.options = options
        pair.answer_index = ans_idx
        if options and ans_idx >= 0:
            pair.A_nl = options[ans_idx].get("text", "")
