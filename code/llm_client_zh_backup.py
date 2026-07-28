"""
LLM API client for online DeltaSG pipeline.

Uses OpenAI-compatible API to call Qwen models via Alibaba Cloud DashScope.

Environment variables:
    DASHSCOPE_API_KEY   – Alibaba Cloud DashScope API key (required)
    LLM_MODEL           – model name, default qwen-plus
    LLM_BASE_URL        – API base URL, default https://dashscope.aliyuncs.com/compatible-mode/v1

Usage:
    from llm_client import LLMClient, create_llm_client

    client = create_llm_client()          # reads env vars
    result = client.select_task(tasks, scene_context)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 1
DEFAULT_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from an LLM response string."""
    if not text:
        return None

    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in markdown code block
    for pattern in (r"```json\s*\n?(.*?)\n?\s*```", r"```\s*\n?(.*?)\n?\s*```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try to find JSON object by brace matching
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------
class LLMClient:
    """OpenAI-compatible LLM client configured for Qwen / DashScope."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        resolved_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)

        self._client = None
        if not resolved_key:
            logger.warning(
                "No DASHSCOPE_API_KEY found; LLM features disabled. "
                "Set the environment variable or pass api_key=..."
            )
            return

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning(
                "openai package not installed. Run: pip install openai"
            )
            return

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=self.timeout,
        )
        logger.info(
            "LLMClient ready: model=%s base_url=%s", self.model, resolved_url
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Core call with retry
    # ------------------------------------------------------------------
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float | None = None,
    ) -> dict | None:
        """Send a chat completion request. Returns parsed JSON dict or None."""
        if not self._client:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                # Disable thinking mode for Qwen3 reasoning models to avoid timeout
                if "qwen3" in self.model.lower():
                    kwargs["extra_body"] = {"enable_thinking": False}

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                result = _extract_json(content)
                if result is not None:
                    return result

                logger.warning(
                    "LLM attempt %d: could not parse JSON from response", attempt
                )
            except Exception as exc:
                logger.warning("LLM attempt %d (model=%s) failed: %s", attempt, self.model, exc)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        return None


# ======================================================================
# Prompt: Task Selection  (物品关联性 — 选任务)
# ======================================================================
_SYSTEM_TASK_SELECT = """\
你是一个具身智能家庭场景的任务规划专家。
给定场景信息和可用的 BEHAVIOR 任务列表，你需要选择一个最适合当前场景的任务。

核心要求：任务必须具有明确的线性执行顺序（每一步依赖上一步的结果）。

选择依据（按优先级排序）：
1. **线性依赖**：任务的步骤之间有明确的先后关系，如"拿起A→放到B上→操作B→得到结果"。拒绝并行型任务（如"收拾桌子"中移动各物品的顺序无关紧要）。
2. **可验证性**：任务的完成状态可以通过物理状态变化来验证（如：物体位置改变、开关状态改变、容器被填满等）。
3. **路径简短**：优先选择 3-5 步即可完成的短链任务，避免需要跨多个房间或涉及 10+ 物品的复杂任务。
4. 任务所需的物品类别在场景中有良好的放置位置。
5. 任务与场景中的房间类型语义匹配。

好的任务示例：
- make_coffee: 拿杯子→放到咖啡机下→按按钮（线性、可验证）
- put_away_dishes: 拿起盘子→打开柜子→放入柜子→关上柜子（线性链）
- set_table: 拿桌布→铺在桌上→摆放餐具（有顺序依赖）

差的任务示例：
- clearing_table: 移动多个物品，顺序无关（并行型，不可验证）
- organize_room: 整理房间，无明确目标状态（模糊）
- buy_groceries: 需要商店环境（场景不匹配）

你必须以合法 JSON 格式回复。"""


def select_task(
    client: LLMClient,
    candidate_tasks: list[dict],
    scene_context: dict,
) -> dict | None:
    """Use LLM to choose a BEHAVIOR task suited to the current scene.

    Args:
        client: LLMClient instance.
        candidate_tasks: list of {"task": str, "num_objects": int, "sample_categories": list[str]}
        scene_context: {"rooms": list[str], "category_counts": dict[str,int]}

    Returns:
        {"selected_task": str, "reason": str} or None
    """
    rooms = scene_context.get("rooms", [])
    category_counts = scene_context.get("category_counts", {})

    # Summarise furniture for context (top-25 by count)
    furniture_lines = sorted(
        category_counts.items(), key=lambda x: -x[1]
    )[:25]
    furniture_str = ", ".join(f"{k}({v})" for k, v in furniture_lines)

    # Format task list (cap at 30 to keep prompt short and avoid timeout)
    task_lines = []
    for t in candidate_tasks[:30]:
        cats = ", ".join(t.get("sample_categories", [])[:4])
        task_lines.append(
            f"- {t['task']}: {t['num_objects']} objects [{cats}]"
        )
    tasks_str = "\n".join(task_lines)

    user_prompt = f"""\
场景信息：
- 房间: {', '.join(rooms) if rooms else 'unknown'}
- 主要家具/物品: {furniture_str}

可选任务列表（共 {len(candidate_tasks)} 个，展示前 {min(len(candidate_tasks), 30)} 个）:
{tasks_str}

选择一个最适合当前场景的任务。输出 JSON:
{{
  "selected_task": "任务名称(必须与列表中完全一致)",
  "reason": "选择理由，说明场景与任务的匹配关系"
}}"""

    return client.call(_SYSTEM_TASK_SELECT, user_prompt)


# ======================================================================
# Prompt: Task Object Selection  (从候选池中选关键任务物品)
# ======================================================================
_SYSTEM_TASK_OBJ_SELECT = """\
你是一个具身智能任务规划专家。
给定一个 BEHAVIOR 任务名称和该任务可用的物品池，你需要选出能构成线性执行链的关键物品。

核心要求：
1. 选出的物品必须能构成明确的线性执行顺序（每步依赖上一步的结果）
2. 必须包含完成该任务的核心物品（如：做咖啡必须有咖啡机和杯子）
3. 物品数量控制在 {num} 个以内，优先选最重要的
4. 不要选大型固定家具（桌子、柜子等），只选可操作的小物件
5. **少而精**：3 个有强依赖关系的物品比 5 个无关物品更好
   - 好：冰箱 + 牛奶 + 台面（取出→放到，线性链）
   - 差：盘子 + 叉子 + 餐巾 + 鸡翅 + 保鲜盒（并行搬运，无依赖）

你必须以合法 JSON 格式回复。"""


def select_task_objects(
    client: LLMClient,
    task_name: str,
    candidate_objects: list[dict],
    num_to_select: int,
) -> dict | None:
    """Use LLM to select key task objects that form a linear chain.

    Args:
        task_name: BEHAVIOR task name
        candidate_objects: list of {"synset": str, "definition": str, "categories": list[str]}
        num_to_select: how many to select

    Returns:
        {"selected_synsets": list[str], "reasoning": str} or None
    """
    candidate_lines = []
    for obj in candidate_objects[:40]:
        cats = ", ".join(obj.get("categories", [])[:3])
        candidate_lines.append(
            f"- {obj['synset']} ({cats}): {obj.get('definition', 'N/A')}"
        )

    user_prompt = f"""\
任务: {task_name}

可用物品池（共 {len(candidate_objects)} 个，展示前 {min(len(candidate_objects), 40)} 个）:
{chr(10).join(candidate_lines)}

从中选出 {num_to_select} 个能构成线性执行链的关键物品。输出 JSON:
{{
  "selected_synsets": ["synset1", "synset2", ...],
  "reasoning": "为什么选这些物品，它们的线性执行顺序是什么"
}}"""

    return client.call(
        _SYSTEM_TASK_OBJ_SELECT.format(num=num_to_select),
        user_prompt,
    )


# ======================================================================
# Prompt: Context Object Selection  (物品关联性 — 选关联物品)
# ======================================================================
_SYSTEM_CONTEXT_SELECT = """\
你是一个具身智能家庭场景的物品关联专家。
给定一组已选定的任务物品和候选的上下文中物品，你需要选出与任务物品有功能性或空间关联的物品。
选择依据：
1. 与任务物品有功能互补（如：水壶→杯子、刀→砧板）
2. 在真实家庭环境中自然共现（如：牙刷→牙膏→漱口杯）
3. 能丰富场景的语义信息，使环境更加真实自然
你必须以合法 JSON 格式回复。"""


def select_context_objects(
    client: LLMClient,
    task_name: str,
    task_objects: list[dict],
    candidate_objects: list[dict],
    num_to_select: int,
    scene_context: dict | None = None,
) -> dict | None:
    """Use LLM to pick context objects with meaningful associations.

    Args:
        task_objects: list of {"synset": str, "definition": str, "categories": list[str]}
        candidate_objects: list of {"synset": str, "definition": str, "categories": list[str]}
        num_to_select: how many to pick

    Returns:
        {"selected_synsets": list[str], "reasoning": list[{"synset": str, "reason": str}]} or None
    """
    task_lines = []
    for obj in task_objects:
        cats = ", ".join(obj.get("categories", [])[:3])
        task_lines.append(f"- {obj['synset']} ({cats}): {obj.get('definition', 'N/A')}")

    candidate_lines = []
    for obj in candidate_objects[:30]:
        cats = ", ".join(obj.get("categories", [])[:3])
        candidate_lines.append(f"- {obj['synset']} ({cats}): {obj.get('definition', 'N/A')}")

    scene_hint = ""
    if scene_context:
        rooms = scene_context.get("rooms", [])
        if rooms:
            scene_hint = f"\n场景房间: {', '.join(rooms)}"

    user_prompt = f"""\
任务: {task_name}{scene_hint}

已选任务物品:
{chr(10).join(task_lines)}

候选上下文中物品（共 {len(candidate_objects)} 个，展示前 {min(len(candidate_objects), 50)} 个）:
{chr(10).join(candidate_lines)}

选出 {num_to_select} 个与任务物品最有关联的上下文中物品。输出 JSON:
{{
  "selected_synsets": ["synset1", "synset2", ...],
  "reasoning": [
    {{"synset": "synset1", "reason": "与XX物品的功能关联：..."}},
    ...
  ]
}}"""

    return client.call(_SYSTEM_CONTEXT_SELECT, user_prompt)


# ======================================================================
# Prompt: Room Assignment  (空间语义分配)
# ======================================================================
_SYSTEM_ROOM_ASSIGN = """\
你是一个具身智能家庭场景的空间规划专家。
给定场景中的房间列表和需要放置的物品，你需要为每个物品分配最合适的房间。
分配依据：
1. 物品的功能属性与房间用途匹配（厨具→厨房，洗漱用品→浴室）
2. 同一任务的物品应尽量在同一房间或相邻房间，方便机器人完成任务
3. 考虑房间的已有家具是否适合放置该物品
你必须以合法 JSON 格式回复。"""


def assign_rooms(
    client: LLMClient,
    rooms: list[str],
    objects_to_place: list[dict],
    task_name: str,
    scene_context: dict | None = None,
) -> dict | None:
    """Use LLM to assign each object to a room.

    Args:
        rooms: available room IDs
        objects_to_place: list of {"synset": str, "category": str, "definition": str, "role": str}
        task_name: BEHAVIOR task name
        scene_context: optional {"room_furniture": {room_id: [category, ...]}}

    Returns:
        {"assignments": [{"synset": str, "room": str, "reason": str}]} or None
    """
    rooms_str = ", ".join(rooms)

    object_lines = []
    for obj in objects_to_place:
        object_lines.append(
            f"- {obj['synset']} (类别: {obj.get('category', 'N/A')}, "
            f"角色: {obj.get('role', 'N/A')}): {obj.get('definition', 'N/A')}"
        )

    # Optional: show furniture per room
    furniture_section = ""
    if scene_context and scene_context.get("room_furniture"):
        furniture_lines = []
        for room, furniture in scene_context["room_furniture"].items():
            top = ", ".join(furniture[:8])
            furniture_lines.append(f"  {room}: {top}")
        furniture_section = "\n各房间主要家具:\n" + "\n".join(furniture_lines)

    user_prompt = f"""\
任务: {task_name}
场景房间: {rooms_str}{furniture_section}

需要放置的物品:
{chr(10).join(object_lines)}

为每个物品分配最合适的房间。输出 JSON:
{{
  "assignments": [
    {{"synset": "物品synset", "room": "房间ID", "reason": "分配理由"}},
    ...
  ]
}}"""

    return client.call(_SYSTEM_ROOM_ASSIGN, user_prompt)


# ======================================================================
# Prompt: Task Feasibility Validation  (任务可解性验证)
# ======================================================================
_SYSTEM_VALIDATE = """\
你是一个具身智能仿真环境的任务可行性评估专家。

重要背景：这是一个 BEHAVIOR benchmark 仿真环境中的任务生成系统。
- 任务是在 OmniGibson 物理仿真器中执行的抽象家庭活动
- 物品通过语义角色（task_object / context_object）参与任务，不要求物理世界中的所有工具都出现
- 机器人通过 MOVE / PICK / PLACE / INTERACT 等原语完成任务

**核心要求：任务必须具有线性执行顺序**
- 每一步的结果必须是下一步的前置条件
- 如果交换任意两个步骤的顺序后任务仍然可行，说明缺乏线性依赖 → 应拒绝
- 好的线性任务：拿起杯子→放到咖啡机下→按按钮（顺序不可交换）
- 差的并行任务：把A移到厨房、把B移到厨房、把C移到厨房（顺序可交换）

评估维度：
1. **线性依赖（最重要）**：步骤间是否有明确的先后关系？交换步骤顺序后任务是否仍然可行？
2. **操作可行性（重要）**：指令中的每个操作是否在机器人能力范围内？（不能切割、倒液体、搅拌、量取）
3. 任务与房间语义是否匹配（如：烹饪任务在厨房）
4. 任务物品是否放在了合理的房间中
5. 是否存在明显的逻辑矛盾（如：要求灭火但场景中没有火源或灭火工具）

**机器人能力范围**：
- 拿起物体（PICK）
- 将物体放到表面上（PLACE on）或放入容器中（PLACE inside）— OmniGibson 支持 OnTop 和 Inside 两种状态
- 移动到位置/房间（MOVE）
- 打开/关闭门、按按钮、切换开关（INTERACT）

**超出能力**：切割、倒液体、搅拌、量取、舀、混合等精细操作。

**验证关键原则**：
- 只评估指令文字中明确描述的操作，不要因为任务名称或物品组合"暗示"了超出能力的操作就拒绝
- 例：指令说"把桃子放进锅里"是合法的（PLACE inside），即使任务名叫"can_fruit"暗示了罐装
- 例：指令说"把刀放在砧板旁"是合法的（PLACE on），即使任务暗示了切割
- 只有当指令文字本身要求切割/倒液体/搅拌等操作时才拒绝

不要拒绝的情况：
- 缺少某些辅助工具（漏斗、开罐器等）— 仿真环境中任务可以简化
- 指令用词不够自然 — 只要有明确的动作和目标即可
- 液体/粉末等物质没有作为独立对象 — 仿真中可以隐含处理

你必须以合法 JSON 格式回复。"""


# ======================================================================
# Prompt: Generate Natural Language Instruction  (生成自然语言指令)
# ======================================================================
_SYSTEM_INSTRUCTION = """\
你是一个具身智能任务描述专家。
给定一个 BEHAVIOR benchmark 任务名称和场景中的物品列表，你需要生成一条清晰、自然的英文任务指令。

**机器人能力限制（重要）**：
仿真环境中的机器人只能执行以下操作：
- 拿起物体（pick up）
- 将物体放到某个表面上（place on）或放入容器中（place inside）— 支持 OnTop 和 Inside 两种状态
- 移动到另一个位置或房间（move to）
- 与物体简单交互：打开/关闭门、按下按钮、切换开关（open/close/press/toggle）

机器人**不能**执行：切割、倒液体、搅拌、量取、舀、削皮等精细操作。
所有食材视为已预处理好的（如：草莓已切好、水已量好）。
注意：将物体放入锅、碗、柜子、冰箱等容器中是合法的（place inside）。

指令应该：
1. 用自然语言描述任务目标（不要用内部任务ID如 xxx-0）
2. 明确指出需要操作的物品和目标位置
3. **强调步骤的先后顺序**（使用 "first... then... finally..." 等连接词）
4. 简洁明了，2-3句话即可
5. 每个步骤都应涉及一个可观察的物理状态变化
6. **只描述机器人能力范围内的操作**

好的指令示例：
"First, pick up the coffee mug from the countertop. Then, place it under the coffee maker spout. Finally, press the brew button on the coffee maker."
"First, open the cabinet door. Then, take out the clean plate and place it on the dining table. Finally, close the cabinet."

差的指令示例：
"Cut the strawberries on the cutting board and blend them."（切割和搅拌超出能力范围）
"Measure 200ml of water and pour it into the blender."（量取和倒液体超出能力范围）

你必须以合法 JSON 格式回复。"""


def generate_instruction(
    client: LLMClient,
    task_name: str,
    task_objects: list[dict],
    target_room: str,
) -> dict | None:
    """Generate a natural language instruction from a BEHAVIOR task name.

    Returns:
        {"instruction": str, "task_description": str} or None
    """
    objects_str = json.dumps(task_objects, ensure_ascii=False, indent=2)

    user_prompt = f"""\
BEHAVIOR 任务名称: {task_name}
目标房间: {target_room}

场景中的任务物品:
{objects_str}

请生成一条自然的英文任务指令。输出 JSON:
{{
  "instruction": "自然语言格式的任务指令(英文)",
  "task_description": "任务的中文简要说明"
}}"""

    return client.call(_SYSTEM_INSTRUCTION, user_prompt)


# ======================================================================
# Prompt: Solution Plan Generation  (解决方案规划)
# ======================================================================
_SYSTEM_SOLUTION_PLAN = """\
你是一个具身智能机器人的任务规划专家。
给定任务指令、场景中的物品列表（含位置和是否可移动），你需要生成一个精确的步骤序列。

可用的动作原语（primitive）：
- MOVE: 移动到目标物品或房间附近。需指定 target_object 或 target_room
- PICK: 拿起一个可移动的物品。需指定 target_object
- PLACE: 将手中物品放到目标位置/物体上。需指定 target_object（放置目标）
- INTERACT: 与物品交互（打开、按下、倒、搅拌等）。可指定 tool_object 和 target_object

关键规则：
1. **inventory 追踪**：每步必须列出当前手中持有的物品列表 inventory: [...]
2. **PICK 前必须 MOVE 到物品位置**：不能直接 PICK 远处的物品
3. **PLACE/INTERACT 前必须 MOVE 到目标位置**
4. **reused 物品（场景自带的大型家具）不能 PICK**：它们是固定的，只能 MOVE 到它们旁边然后 INTERACT
5. **跨房间需要 MOVE 到目标房间**：如机器人在 kitchen_0 但需要去 living_room_0 的物品
6. **步骤间有线性依赖**：每步的结果是下一步的前提
7. **工具使用链**：如果需要用工具A操作目标B，先 PICK A → MOVE 到 B → INTERACT(tool=A, target=B)
8. **使用实际放置位置**：每个物品信息中包含 placed_on（实际放在哪个支撑面上）和 room（实际在哪个房间）。MOVE 和 PLACE 步骤必须引用实际位置，不要假设物品在桌子或其他家具上。例如物品放在 bottom_cabinet_xxx 上，MOVE 目标就应该是该 cabinet。
9. **严禁编造 object_id**：所有 target_object 必须使用提供的物品列表中的 object_id。如果指令提到某个物体（如 fridge、countertop）但它不在列表中，用 target_room 代替或跳过该步骤。绝对不要编造如 "fridge"、"coffee_maker_001"、"dispenser_area" 等虚构 ID。标记为 reference_only 的场景物体也可以作为 INTERACT 的目标。

**机器人能力限制（严格遵守）**：
- 只能 PICK（拿起）、PLACE（放下——支持放到表面上 OnTop 或放入容器中 Inside）、MOVE（移动）、INTERACT（开关/按压/触发）
- **不能**：切割、倒液体、搅拌、量取、舀、混合
- 食材视为已预处理，不需要切割或准备
- INTERACT 只用于：打开/关闭门、按按钮、切换开关等简单操作
- PLACE inside 容器（锅、碗、柜子、冰箱）是合法的

你必须以合法 JSON 格式回复。"""


def generate_solution_plan(
    client: LLMClient,
    task_instruction: str,
    task_objects: list[dict],
    target_room: str,
    rooms: list[str],
    robot_room: str | None = None,
) -> dict | None:
    """Use LLM to generate a step-by-step solution plan.

    Args:
        task_instruction: natural language instruction
        task_objects: list of {
            "object_id": str, "category": str, "room": str,
            "reused": bool  # True = scene-native, cannot PICK
        }
        target_room: primary task room
        rooms: all rooms in the scene
        robot_room: where the robot starts (default: target_room)

    Returns:
        {
            "solution_plan": [
                {"step_id": 1, "primitive": "MOVE", "nl": "...",
                 "target_object": "...", "inventory": []},
                ...
            ],
            "reasoning": "规划的简要说明"
        } or None
    """
    objects_str = json.dumps(task_objects, ensure_ascii=False, indent=2)

    user_prompt = f"""\
任务指令: {task_instruction}
机器人初始位置: {robot_room or target_room}
场景房间: {', '.join(rooms)}
目标房间: {target_room}

场景中的任务物品:
{objects_str}

生成精确的步骤序列。输出 JSON:
{{
  "solution_plan": [
    {{
      "step_id": 1,
      "primitive": "MOVE",
      "nl": "Move to the saucepan on the countertop",
      "target_object": "saucepan_001",
      "target_room": "kitchen_0",
      "inventory": []
    }},
    {{
      "step_id": 2,
      "primitive": "PICK",
      "nl": "Pick up the saucepan",
      "target_object": "saucepan_001",
      "inventory": []
    }},
    {{
      "step_id": 3,
      "primitive": "MOVE",
      "nl": "Move to the stove",
      "target_object": "stove_001",
      "target_room": "kitchen_0",
      "inventory": ["saucepan_001"]
    }},
    {{
      "step_id": 4,
      "primitive": "PLACE",
      "nl": "Place saucepan on the stove",
      "target_object": "stove_001",
      "inventory": []
    }}
  ],
  "reasoning": "简要说明规划的线性逻辑"
}}"""

    return client.call(_SYSTEM_SOLUTION_PLAN, user_prompt)


def validate_task(
    client: LLMClient,
    task_instruction: str,
    task_objects: list[dict],
    target_room: str,
    rooms: list[str],
    solution_plan: list[dict] | None = None,
) -> dict | None:
    """Use LLM to validate task feasibility.

    Returns:
        {
            "feasible": bool,
            "confidence": float,
            "instruction_clear": bool,
            "objects_sufficient": bool,
            "room_assignment_reasonable": bool,
            "issues": list[str],
            "improved_instruction": str | None
        } or None
    """
    objects_str = json.dumps(task_objects, ensure_ascii=False, indent=2)

    plan_section = ""
    if solution_plan:
        plan_str = json.dumps(solution_plan, ensure_ascii=False, indent=2)
        plan_section = f"\n\n预设解决方案:\n{plan_str}"

    user_prompt = f"""\
评估以下任务环境的可行性：

任务指令: {task_instruction}
目标房间: {target_room}
场景所有房间: {', '.join(rooms)}

任务相关物品:
{objects_str}{plan_section}

请评估并输出 JSON:
{{
  "feasible": true/false,
  "is_linear": true/false,
  "confidence": 0.0-1.0,
  "instruction_clear": true/false,
  "objects_sufficient": true/false,
  "room_assignment_reasonable": true/false,
  "linearity_analysis": "简要说明步骤间的依赖关系，为什么是/不是线性任务",
  "issues": ["问题1(如有)", "问题2(如有)"],
  "improved_instruction": "如果指令需要改进，给出改进后的指令；否则为null"
}}"""

    return client.call(_SYSTEM_VALIDATE, user_prompt)


# ======================================================================
# Factory
# ======================================================================
def create_llm_client(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMClient | None:
    """Create an LLMClient from explicit args or environment variables.

    Returns None if the client cannot be initialised.
    """
    client = LLMClient(api_key=api_key, model=model, base_url=base_url)
    return client if client.available else None
