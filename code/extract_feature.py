"""
extract_feature.py — 基于时空图抽取物体特征 (EM-STORM)

定位
----
把「task_instance 的场景图 (before_graph / after_graph / delta_sg / solution_plan) →
物体/空间/房间特征」的代码统一到本叶子模块, 供 ``translator.py`` 与 ``distractor.py``
共同 import。本模块**只依赖标准库**, 不 import translator / distractor (避免循环导入)。

内容
----
- 场景索引 ``SceneIndex`` / ``build_scene_index``         —— 房间/物体/affordance/空间关系
- 空间特征描述器 ``build_candidate_descriptors``          —— 消歧候选「独一无二空间特征」短语
- 房间拓扑 ``build_room_topology``                         —— 连通图 + 距离
- 房间解析 ``current_room`` / ``resolve_rooms``            —— 当前/目标/跨房间
- 类别/房间/异常/摄像头/消歧候选等轻量抽取 ``object_category`` / ``derive_anomaly`` / …
- 通用工具 ``humanize`` / ``category_of`` / ``room_of`` / ``is_structural`` 等
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 结构/背景类别 (非"物品", 不用于 bbox 提问 / 干扰项候选; 可按需扩展)
# 原 translator.NON_OBJECT_CATEGORIES 与 distractor._STRUCTURAL_CATEGORIES 统一于此。
STRUCTURAL_CATEGORIES = {"background", "walls", "floors", "ceilings", "robot"}

# 语义角色中表示"支撑面"的标签 (delta_sg.nodes / plan_objects 的 semantic_role(s))
SUPPORT_ROLES = {"task_support", "delivery_destination", "support"}

# 方位判定阈值 (米): 物体相对房间中心偏移超过该值才判定为某方位, 避免微小抖动误判。
DIRECTION_EPS = 0.5

# 地标锚定: 候选距最近地标超过该距离视为"无可锚地标", 退到门相对/罗盘兜底。
NEAR_MAX = 2.5


def humanize(name: str | None) -> str:
    """把 id 风格名 (下划线分隔) 转为人类可读 (空格分隔), 如 television_room_0 → television room 0。"""
    return (name or "").replace("_", " ").strip()


def is_structural(aff: dict[str, Any]) -> bool:
    """是否为结构/背景类别 (不作为干扰项候选)。"""
    return aff.get("category") in STRUCTURAL_CATEGORIES


def category_of(index: "SceneIndex", object_id: str | None) -> str:
    """object_id → 人类可读类别名 (回退 object_id 本身)。"""
    if not object_id:
        return ""
    aff = index.affordance.get(object_id)
    if aff and aff.get("category"):
        return humanize(aff["category"])
    return humanize(object_id)


def room_of(index: "SceneIndex", object_id: str | None) -> str:
    """object_id → 所在房间 (取第一个; 未知返回空串)。"""
    if not object_id:
        return ""
    aff = index.affordance.get(object_id)
    return (aff.get("rooms") or [""])[0] if aff else ""


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


def strip_room_prefix(name: str) -> str:
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
        index_object(idx, oid, n, aff)

    # 空间关系边: near (物体邻近, 供"邻近但错误"的物体干扰项)
    for e in graph.get("edges") or []:
        if e.get("relation") == "near":
            src = strip_room_prefix(e.get("source") or "")
            tgt = strip_room_prefix(e.get("target") or "")
            idx.near.setdefault(src, []).append(tgt)
            idx.near.setdefault(tgt, []).append(src)

    # 任务相关物体 (plan_objects / task_objects / added_objects / delta_sg.nodes)
    ingest_task_objects(idx, task_instance)

    # 支撑面: supports_on_top 的家具 (排除结构/背景类别)
    for oid, aff in idx.affordance.items():
        if aff.get("supports_on_top") and not is_structural(aff):
            idx.support_surfaces.append(oid)

    return idx


def index_object(idx: SceneIndex, oid: str, node: dict[str, Any], aff: dict[str, Any]) -> None:
    """把一个物体登记进索引的各映射。"""
    idx.objects[oid] = node
    idx.affordance[oid] = aff
    cat = aff["category"]
    idx.objects_by_category.setdefault(cat, []).append(oid)
    for r in aff["rooms"]:
        idx.objects_by_room.setdefault(r, []).append(oid)


def ingest_task_objects(idx: SceneIndex, task_instance: dict[str, Any]) -> None:
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
        supports_on_top = bool(rec.get("supports_on_top")) or bool(set(roles) & SUPPORT_ROLES)
        idx.task_objects.add(oid)
        index_object(idx, oid, {"id": oid, "category": cat, "rooms": [room] if room else []}, {
            "category": cat or oid,
            "rooms": [room] if room else [],
            "interaction": inter,
            "supports_on_top": supports_on_top,
            "supports_inside": bool(rec.get("supports_inside")),
            "reachable": True,
            "extent": None,  # 任务物体无原生 bbox, 尺寸未知 (默认可抓, 见 _graspable)
        })


def build_candidate_descriptors(task_instance: dict[str, Any],
                                candidate_ids: list[str],
                                index: "SceneIndex | None" = None) -> dict[str, str]:
    """为消歧候选对象生成"独一无二的空间特征"短语 (题干与选项共用, 保证标签一致)。

    返回 ``{object_id: 描述短语}``; 短语不含冠词, 形如::

        "electric switch near the door"      # 图 near 边 / 几何最近显著地标
        "electric switch opposite the door"  # 无可锚地标 → 门相对兜底
        "standing tv in living room 0"       # 类别唯一 → 只加房间

    同类别多实例按序用更细的特征区分 (直到唯一): 房间 → 显著地标 (门/窗 > 家具 > 其它, 类别
    在房间内唯一) → 同一地标远近 → 门相对 (near/opposite the door) → 罗盘方位 → 数字后缀。
    数据源为 ``before_graph`` (edges.near + nodes.pose/rooms), 回退 after_graph / debug.before_graph;
    图缺失时直接退到数字后缀 (保证不坍缩)。
    """
    if index is None:
        index = build_scene_index(task_instance)
    graph = (task_instance.get("before_graph")
             or task_instance.get("after_graph")
             or (task_instance.get("debug") or {}).get("before_graph")
             or {})
    nav = graph.get("navigation") or {}
    room_centers = nav.get("room_centers") or {}
    nodes = {n.get("id") or n.get("name"): n for n in (graph.get("nodes") or [])
             if n.get("id") or n.get("name")}

    ids = [oid for oid in candidate_ids if oid]

    def _room(oid: str) -> str:
        aff = index.affordance.get(oid)
        if aff:
            rooms = aff.get("rooms") or []
            if rooms:
                return rooms[0]
        n = nodes.get(oid)
        if isinstance(n, dict):
            rooms = n.get("rooms") or []
            if rooms:
                return rooms[0]
        return ""

    def _cat(oid: str) -> str:
        cat = ""
        aff = index.affordance.get(oid)
        if aff:
            cat = aff.get("category") or ""
        if not cat:
            n = nodes.get(oid)
            cat = (n.get("category") or "") if isinstance(n, dict) else ""
        return humanize(cat) if cat else humanize(oid)

    def _pos(oid: str) -> tuple[float, float] | None:
        n = nodes.get(oid)
        if not isinstance(n, dict):
            return None
        p = (n.get("pose") or {}).get("position")
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                return (float(p[0]), float(p[1]))
            except (TypeError, ValueError):
                return None
        return None

    def _center(room: str) -> tuple[float, float] | None:
        c = room_centers.get(room)
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                return (float(c[0]), float(c[1]))
            except (TypeError, ValueError):
                return None
        return None

    def _direction(oid: str, room: str) -> str | None:
        pos, c = _pos(oid), _center(room)
        if not pos or not c:
            return None
        dx, dy = pos[0] - c[0], pos[1] - c[1]
        v = "north" if dy > DIRECTION_EPS else ("south" if dy < -DIRECTION_EPS else "")
        h = "east" if dx > DIRECTION_EPS else ("west" if dx < -DIRECTION_EPS else "")
        return (v + h) if (v and h) else (v or h or None)

    def _dist(oid: str, room: str) -> float | None:
        pos, c = _pos(oid), _center(room)
        if not pos or not c:
            return None
        return ((pos[0] - c[0]) ** 2 + (pos[1] - c[1]) ** 2) ** 0.5

    # ---- 地标锚定辅助: 房间内显著地标 / 候选→地标锚定 / 门相对兜底 ----
    def _geom_dist(a: str, b: str) -> float | None:
        pa, pb = _pos(a), _pos(b)
        if not pa or not pb:
            return None
        return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5

    def _landmark_label(oid: str) -> str:
        """地标人类可读名: door/window 归一为 door/window, 其余取类别。"""
        cat = (_cat(oid) or "").lower()
        if "door" in cat:
            return "door"
        if "window" in cat:
            return "window"
        return _cat(oid)

    def _landmark_pool(room: str, exclude_cats: set[str]) -> list[str]:
        """房间内可当地标的物体: 同房间 + 非结构 + 非候选 + 类别唯一 + 有位置, 按显著度排序。"""
        pool: list[str] = []
        for oid in nodes:
            if oid in ids or not _pos(oid):
                continue
            c = _cat(oid)
            if not c or c in exclude_cats or c in STRUCTURAL_CATEGORIES:
                continue
            if room and _room(oid) != room:
                continue
            pool.append(oid)
        # 类别唯一性: 避免 "near the bookcase" 指向多个 bookcase 之一
        counts: dict[str, int] = {}
        for oid in pool:
            counts[_cat(oid)] = counts.get(_cat(oid), 0) + 1
        pool = [oid for oid in pool if counts[_cat(oid)] == 1]

        def _salience(oid: str) -> tuple:
            aff = index.affordance.get(oid) or {}
            cat = (_cat(oid) or "").lower()
            if "door" in cat or "window" in cat:
                return (0, oid)
            if aff.get("supports_on_top") or aff.get("supports_inside"):
                return (1, oid)
            ext = aff.get("extent")
            if ext is not None and ext >= 1.0:
                return (2, oid)
            return (3, oid)
        pool.sort(key=_salience)
        return pool

    def _landmark_anchor(oid: str, pool: list[str]) -> str | None:
        """候选 → 锚定地标: 图 near 边优先, 否则几何最近 (≤ NEAR_MAX)。"""
        pool_set = set(pool)
        for lm in index.near.get(oid) or []:
            if lm in pool_set:
                return lm
        best, bestd = None, None
        for lm in pool:
            d = _geom_dist(oid, lm)
            if d is not None and d <= NEAR_MAX and (bestd is None or d < bestd):
                best, bestd = lm, d
        return best

    def _room_door(room: str) -> str | None:
        """房间内唯一门 (无门则唯一窗), 作为门相对兜底的锚点。"""
        for key in ("door", "window"):
            found = [oid for oid in nodes
                     if key in (_cat(oid) or "").lower() and _pos(oid)
                     and (not room or _room(oid) == room)]
            if len(found) == 1:
                return found[0]
        return None

    groups: dict[str, list[str]] = {}
    for oid in ids:
        groups.setdefault(_cat(oid), []).append(oid)

    descriptors: dict[str, str] = {}
    for cat, members in groups.items():
        if len(members) == 1:
            oid = members[0]
            room = _room(oid)
            descriptors[oid] = f"{cat} in {humanize(room)}" if room else cat
            continue

        # 多实例: 先房间区分
        rooms = [_room(oid) for oid in members]
        if all(rooms) and len(set(rooms)) == len(members):
            for oid in members:
                room = _room(oid)
                descriptors[oid] = f"{cat} in {humanize(room)}" if room else cat
            continue

        # 同房间: 显著地标锚定
        room = _room(members[0]) or _room(members[-1])
        pool = _landmark_pool(room, set(_cat(m) for m in members))
        anchors = {oid: _landmark_anchor(oid, pool) for oid in members}

        # 地标互异 → "near the {landmark}"
        if all(anchors.values()) and len(set(anchors.values())) == len(members):
            for oid in members:
                lm = anchors[oid]
                d = _geom_dist(oid, lm)
                rel = "next to" if d is not None and d < 1.0 else "near"
                descriptors[oid] = f"{cat} {rel} the {_landmark_label(lm)}"
            continue

        # 地标撞车 (同一非空地标) → 远近区分
        if len(members) == 2 and anchors[members[0]] and anchors[members[0]] == anchors[members[1]]:
            lm = anchors[members[0]]
            d0, d1 = _geom_dist(members[0], lm), _geom_dist(members[1], lm)
            if d0 is not None and d1 is not None and abs(d0 - d1) > DIRECTION_EPS:
                nearer = members[0] if d0 < d1 else members[1]
                farther = members[1] if d0 < d1 else members[0]
                descriptors[nearer] = f"{cat} nearer the {_landmark_label(lm)}"
                descriptors[farther] = f"{cat} farther from the {_landmark_label(lm)}"
                continue

        # 门相对兜底 (唯一门/窗) → near / opposite the door
        door = _room_door(room)
        if door and len(members) == 2:
            d0, d1 = _geom_dist(members[0], door), _geom_dist(members[1], door)
            if d0 is not None and d1 is not None and abs(d0 - d1) > DIRECTION_EPS:
                near_oid = members[0] if d0 < d1 else members[1]
                far_oid = members[1] if d0 < d1 else members[0]
                lbl = _landmark_label(door)
                descriptors[near_oid] = f"{cat} near the {lbl}"
                descriptors[far_oid] = f"{cat} opposite the {lbl}"
                continue

        # 罗盘方位 (相对房间中心) —— 地标/门兜底都失败时
        dirs = [_direction(oid, _room(oid)) for oid in members]
        if all(d for d in dirs) and len(set(dirs)) == len(members):
            for oid, d in zip(members, dirs):
                room = _room(oid)
                descriptors[oid] = (f"{cat} in the {d} part of {humanize(room)}" if room
                                    else f"{cat} ({d})")
            continue

        # 方位冲突: 两两按到房间中心的远近区分
        if len(members) == 2:
            d0, d1 = _dist(members[0], _room(members[0])), _dist(members[1], _room(members[1]))
            if d0 is not None and d1 is not None and abs(d0 - d1) > DIRECTION_EPS:
                room = _room(members[0]) or _room(members[1])
                nearer, farther = (members[0], members[1]) if d0 < d1 else (members[1], members[0])
                if room:
                    descriptors[nearer] = f"{cat} nearer to the center of {humanize(room)}"
                    descriptors[farther] = f"{cat} farther from the center of {humanize(room)}"
                else:
                    descriptors[nearer] = f"{cat} nearer to the room center"
                    descriptors[farther] = f"{cat} farther from the room center"
                continue

        # 兜底: 数字后缀 (必唯一)
        for i, oid in enumerate(members, 1):
            m = re.search(r"_(\d+)$", oid)
            suffix = m.group(1) if m else str(i)
            room = _room(oid)
            descriptors[oid] = f"{cat} {suffix} in {humanize(room)}" if room else f"{cat} {suffix}"

    return descriptors


# ======================================================================
# 类别 / 房间拓扑 / 房间解析 / 异常 / 摄像头 / 消歧候选
# ======================================================================
def object_category(object_id: str | None, task_instance: dict[str, Any] | None) -> str:
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


def current_room(task_instance: dict[str, Any] | None, plan: list[dict[str, Any]], t: int,
                 index: "SceneIndex | None" = None) -> str:
    """当前房间: t<=0 → robot 初始房间; 否则上一动作目标物体所在房间 (从场景图解析)。

    注意: solution_plan 步骤只带 ``target_object`` (不带 ``target_room``), 机器人当前房间
    需从场景图 ``room_of`` 反查; 初始房间在 raw 里位于 ``robot.initial_room`` (顶层
    ``robot_initial_room`` 常为 None, 扁平化后才是该值), 故两处都尝试。
    """
    ti = task_instance or {}
    if index is None:
        index = build_scene_index(ti)
    if t <= 0:
        room = ti.get("robot_initial_room") or ""
        if not room:
            room = (ti.get("robot") or {}).get("initial_room") or ""
        return room
    prev = plan[t - 1] if 0 <= t - 1 < len(plan) else {}
    return room_of(index, (prev or {}).get("target_object")) or ""


def resolve_rooms(task_instance: dict[str, Any] | None, scene: Any,
                  next_step: dict[str, Any] | None) -> tuple[str, str, bool]:
    """求 (当前房间, 目标房间, 是否跨房间)。

    当前房间 = 上一动作目标物体所在房间 (首步用 robot 初始房间);
    目标房间 = 本步 target_object 所在房间 (``target_room`` 字段优先)。二者都已知且不同 → 跨房间。
    """
    if not next_step:
        return "", "", False
    ti = task_instance or {}
    index = build_scene_index(ti)
    target = next_step.get("target_object")
    target_room = next_step.get("target_room") or room_of(index, target) or ""
    plan = ti.get("solution_plan") or []
    current = current_room(ti, plan, scene.simulation_step, index)
    cross_room = bool(target_room and current and target_room != current)
    return current, target_room, cross_room


def global_camera_rooms(task_instance: dict[str, Any] | None) -> list[str]:
    """返回安装了全局(监控)摄像头的房间名列表 (去重、保持原始顺序, 已 humanize)。

    仅统计 camera_type == "global_camera" 且带 room_id 的摄像头; 机器人自带相机所在房间
    = 机器人当前房间 (由 ``resolve_rooms`` 给出), 不在此列。
    """
    rooms: list[str] = []
    seen: set[str] = set()
    for cam in (task_instance or {}).get("camera") or []:
        if cam.get("camera_type") != "global_camera":
            continue
        room = humanize(cam.get("room_id"))
        if room and room not in seen:
            seen.add(room)
            rooms.append(room)
    return rooms


def derive_anomaly(state_changed_objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 ``state_changed_objects`` 提取异常条目, 规整为异常上下文 (None 表示无异常)。

    命中条件: 条目 ``semantic_roles`` 含 ``anomaly`` / ``goal_target`` (Env-B 的异常源)。
    返回 ``{"object_id", "category", "room_id", "state", "phase", "smoke_visible", "flame_visible"}``,
    供 Env-B 主动响应答案与异常检测题 (G 形式) 使用。
    """
    for entry in state_changed_objects:
        if not isinstance(entry, dict):
            continue
        roles = set(entry.get("semantic_roles") or [])
        if not (roles & {"anomaly", "goal_target"}):
            continue
        states = entry.get("states") or {}
        visual = entry.get("visual_effect") or {}
        return {
            "object_id": entry.get("object_id"),
            "category": entry.get("category") or "",
            "room_id": entry.get("room_id") or "",
            "state": dict(states) if isinstance(states, dict) else {},
            "phase": entry.get("anomaly_phase") or "",
            "smoke_visible": bool(visual.get("smoke_visible")),
            "flame_visible": bool(visual.get("flame_visible")),
        }
    return None


def disambiguation_candidates(task_instance: dict[str, Any] | None) -> list[str]:
    """Env-C (E) 候选对象的人类可读描述列表 (optimal + rejected, 不暴露 instance id)。

    同类别多实例用"独一无二的空间特征"区分 (见 ``build_candidate_descriptors``),
    使题干可解; 角色/理由不写入题干 (避免泄露哪一个是 optimal), 它们只出现在答案
    ``rejected_candidates[].reason`` 里。
    """
    sr = (task_instance or {}).get("semantic_reasoning") or {}
    gt = sr.get("ground_truth") or {}
    ids: list[str] = []
    optimal = gt.get("optimal_object")
    if optimal:
        ids.append(optimal)
    for rc in gt.get("rejected_candidates") or []:
        if not isinstance(rc, dict):
            continue
        oid = rc.get("object_id")
        if oid:
            ids.append(oid)
    if not ids:
        return []

    desc = build_candidate_descriptors(task_instance, ids)
    return [desc.get(oid) or object_category(oid, task_instance) or humanize(oid)
            for oid in ids]
