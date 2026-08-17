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

import copy
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
DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 1
DEFAULT_TIMEOUT = 300.0


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
        self._cache: dict[tuple[str, str, bool, float, str], dict] = {}

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
        cache_key = (system_prompt, user_prompt, json_mode, float(temp), self.model)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

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

                # Disable thinking mode for Qwen3 reasoning models to avoid timeout.
                if "qwen3" in self.model.lower():
                    kwargs["extra_body"] = {"enable_thinking": False}

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                result = _extract_json(content)
                if result is not None:
                    self._cache[cache_key] = copy.deepcopy(result)
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
# Prompt: Task Selection
# ======================================================================
_SYSTEM_TASK_SELECT = """\
You are a task planning expert for embodied AI in domestic environments.
Given scene information and a list of available BEHAVIOR tasks, select the task that best fits the current scene.

Core requirement: The task must have a clear linear execution order (each step depends on the result of the previous step).

Selection criteria (in priority order):
1. **Linear dependency**: Steps have a clear sequential relationship, e.g. "pick up A → place on B → operate B → get result". Reject parallel tasks (e.g. "clear table" where the order of moving items is irrelevant).
2. **Verifiability**: Task completion can be verified through physical state changes (e.g. object position changed, door opened/closed, container filled).
3. **Short path**: Prefer short-chain tasks completable in 3-5 steps. Avoid complex tasks requiring multiple rooms or 10+ objects.
4. Required object categories have good placement locations in the scene.
5. Task semantically matches the room types in the scene.

**Hard constraints (MUST follow)**:
- Only use existing scene objects and explicitly available support/destination objects. If a task requires sink/stove/microwave/dishwasher/refrigerator/trash, verify that object exists in the scene's category_counts.
- Do NOT pour, cut, cook liquid, batch pick/place, or "place both" items.
- Every step must be strictly dependent on the previous step's result (PICK / PLACE / MOVE / INTERACT).
- Do NOT select tasks requiring outdoor/lawn/mailbox placement unless the object already exists in the scene.
- Avoid tasks that need a "designated cleaning area" or "specific disposal location" that is not in the scene.

Good task examples:
- make_coffee: pick up mug → place under coffee maker → press button (linear, verifiable)
- put_away_dishes: pick up plate → open cabinet → place inside → close cabinet (linear chain)
- set_table: pick up placemat → place on table → arrange cutlery (sequential dependency)

Bad task examples:
- clearing_table: moving multiple objects, order irrelevant (parallel, unverifiable)
- organize_room: tidying up, no clear goal state (vague)
- buy_groceries: requires a store environment (scene mismatch)
- water_plants: requires outdoor/lawn (unstable placement)
- clean_floor: requires designated cleaning area (missing destination)

You must respond in valid JSON format."""


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

    # Format task list (cap to keep prompt short and avoid timeout)
    max_candidates = min(15, len(candidate_tasks))
    task_lines = []
    for t in candidate_tasks[:max_candidates]:
        cats = ", ".join(t.get("sample_categories", [])[:3])
        task_lines.append(
            f"- {t['task']}: [{cats}]"
        )
    tasks_str = "\n".join(task_lines)

    user_prompt = f"""\
Scene information:
- Rooms: {', '.join(rooms) if rooms else 'unknown'}
- Main furniture/objects: {furniture_str}

Available tasks ({len(candidate_tasks)} total, showing {max_candidates} scene-compatible):
{tasks_str}

Select the best task. Output JSON:
{{
  "selected_task": "task name (must exactly match one from the list)",
  "reason": "one sentence why this task fits the scene"
}}"""

    return client.call(_SYSTEM_TASK_SELECT, user_prompt)


# ======================================================================
# Prompt: Task Object Selection
# ======================================================================
_SYSTEM_TASK_OBJ_SELECT = """\
You are an embodied AI task planning expert.
Given a BEHAVIOR task name and a pool of available objects, select the key objects that form a linear execution chain.

Core requirements:
1. Selected objects must form a clear linear execution order (each step depends on the previous)
2. Must include core objects needed to complete the task (e.g. coffee maker and mug for making coffee)
3. Limit to {num} objects, prioritising the most important ones
4. Do not select large fixed furniture (tables, cabinets, etc.), only select manipulable small objects
5. **Quality over quantity**: 3 objects with strong dependencies are better than 5 unrelated objects
   - Good: fridge + milk + countertop (take out → place down, linear chain)
   - Bad: plate + fork + napkin + chicken wing + tupperware (parallel carrying, no dependency)

You must respond in valid JSON format."""


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
Task: {task_name}

Available object pool ({len(candidate_objects)} total, showing first {min(len(candidate_objects), 40)}):
{chr(10).join(candidate_lines)}

Select {num_to_select} key objects that form a linear execution chain. Output JSON:
{{
  "selected_synsets": ["synset1", "synset2", ...],
  "reasoning": "why these objects were selected and what their linear execution order is"
}}"""

    return client.call(
        _SYSTEM_TASK_OBJ_SELECT.format(num=num_to_select),
        user_prompt,
    )


# ======================================================================
# Prompt: Context Object Selection
# ======================================================================
_SYSTEM_CONTEXT_SELECT = """\
You are an object association expert for embodied AI in domestic environments.
Given a set of selected task objects and candidate context objects, select objects that have functional or spatial associations with the task objects.
Selection criteria:
1. Functionally complementary to task objects (e.g. kettle → cup, knife → cutting board)
2. Naturally co-occur in real domestic environments (e.g. toothbrush → toothpaste → cup)
3. Enrich the semantic information of the scene, making the environment more realistic
You must respond in valid JSON format."""


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
            scene_hint = f"\nScene rooms: {', '.join(rooms)}"

    user_prompt = f"""\
Task: {task_name}{scene_hint}

Selected task objects:
{chr(10).join(task_lines)}

Candidate context objects ({len(candidate_objects)} total, showing first {min(len(candidate_objects), 30)}):
{chr(10).join(candidate_lines)}

Select {num_to_select} context objects with the strongest associations to the task objects. Output JSON:
{{
  "selected_synsets": ["synset1", "synset2", ...],
  "reasoning": [
    {{"synset": "synset1", "reason": "functional association with XX: ..."}},
    ...
  ]
}}"""

    return client.call(_SYSTEM_CONTEXT_SELECT, user_prompt)


# ======================================================================
# Prompt: Room Assignment
# ======================================================================
_SYSTEM_ROOM_ASSIGN = """\
You are a spatial planning expert for embodied AI in domestic environments.
Given a list of rooms in the scene and objects to place, assign each object to the most appropriate room.
Assignment criteria:
1. Object function matches room purpose (kitchenware → kitchen, toiletries → bathroom)
2. Objects from the same task should be in the same room or adjacent rooms for efficient robot execution
3. Consider whether existing furniture in the room is suitable for placing the object
You must respond in valid JSON format."""


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
            f"- {obj['synset']} (category: {obj.get('category', 'N/A')}, "
            f"role: {obj.get('role', 'N/A')}): {obj.get('definition', 'N/A')}"
        )

    # Optional: show furniture per room
    furniture_section = ""
    if scene_context and scene_context.get("room_furniture"):
        furniture_lines = []
        for room, furniture in scene_context["room_furniture"].items():
            top = ", ".join(furniture[:8])
            furniture_lines.append(f"  {room}: {top}")
        furniture_section = "\nFurniture per room:\n" + "\n".join(furniture_lines)

    user_prompt = f"""\
Task: {task_name}
Scene rooms: {rooms_str}{furniture_section}

Objects to place:
{chr(10).join(object_lines)}

Assign each object to the most appropriate room. Output JSON:
{{
  "assignments": [
    {{"synset": "object synset", "room": "room ID", "reason": "assignment rationale"}},
    ...
  ]
}}"""

    return client.call(_SYSTEM_ROOM_ASSIGN, user_prompt)


# ======================================================================
# Prompt: Task Feasibility Validation
# ======================================================================
_SYSTEM_VALIDATE = """\
You are a task feasibility checker for a household robot simulation. Your job is to catch ONLY clearly impossible tasks. Default to feasible=True unless there is a clear, unambiguous problem.

**CRITICAL RULES — read carefully:**

1. **Task objects ARE spawned by the system.** The "Task-related objects" list shows what WILL be placed in the scene. Do NOT flag them as missing. They DO NOT need to pre-exist.
2. **Default to feasible.** Only reject if you see an OBVIOUS problem. If unsure, say feasible.
3. **Linear means sequential, not parallel.** "Pick A, place A, pick B, place B" IS linear enough — the robot does one thing at a time. Only reject if steps are explicitly parallel (e.g. "simultaneously", "both at once", "move everything").
4. **Only flag missing_destination if the instruction requires placing ON/IN a furniture object not in the scene inventory.** Example: "put in fridge" when no fridge exists. But if a cabinet or table is mentioned and exists, that's fine.

**Robot can:** PICK (pick up), PLACE (on surfaces or inside containers), MOVE (navigate), INTERACT (open/close/press/toggle).
**Robot cannot:** cut, pour, stir, measure, scoop, mix.

**Only reject for these clear reasons:**
- Instruction explicitly requires cutting, pouring, stirring, etc.
- Task is explicitly parallel ("move everything", "both at once")
- Instruction requires a destination (fridge/stove/sink) that clearly doesn't exist in scene inventory
- Task makes no logical sense in the given room

**Do NOT reject for:**
- Task objects "missing" from scene inventory (they will be spawned!)
- Vague or awkward wording
- Simple 2-step tasks ("too simple" is not a reason to reject)
- Steps that could theoretically be reordered (if robot does them one at a time, it's linear)

You must respond in valid JSON format. Default to feasible=True, confidence=0.9 unless there is a clear problem."""


# ======================================================================
# Prompt: Generate Natural Language Instruction
# ======================================================================
_SYSTEM_INSTRUCTION = """\
You are a robot task instruction writer. Write a short, mechanical instruction with exactly TWO actions: pick up and place.

Rules:
- Always include both a source and a destination: "Pick up the X from the Y. Place it on the Z."
- Use the actual placed_on info from the object list as the source location.
- Choose a reasonable destination from the scene furniture.
- Never add reasons or explanations.
- Never use raw object IDs.

Examples:
"Pick up the medicine bottle from the coffee table. Place it on the bookcase."
"Pick up the apple from the bottom cabinet. Place it on the countertop."

For appliance/open_close tasks (no objects to pick up):
"Open the bottom cabinet in the kitchen."
"Turn off the standing TV in the living room."

You must respond in valid JSON format."""


def generate_instruction(
    client: LLMClient,
    task_name: str,
    task_objects: list[dict],
    target_room: str,
    scene_furniture: dict[str, list[str]] | None = None,
    task_category: str | None = None,
) -> dict | None:
    """Generate a natural language task instruction.

    Args:
        task_name: e.g. "retrieve_medicine", "open_cabinet"
        task_objects: objects placed in the scene
        target_room: primary room for the task
        scene_furniture: {room: [category, ...]} for available destinations
        task_category: e.g. "retrieval_delivery", "open_close" for context

    Returns:
        {"instruction": str, "task_description": str} or None
    """
    objects_str = json.dumps(task_objects, ensure_ascii=False, indent=2)

    scene_section = ""
    if scene_furniture:
        furniture_lines = []
        for room, cats in sorted(scene_furniture.items()):
            furniture_lines.append(f"  {room}: {', '.join(sorted(set(cats)))}")
        scene_section = "\n\nFurniture available in scene:\n" + "\n".join(furniture_lines)

    category_note = ""
    if task_category:
        category_descriptions = {
            "retrieval_delivery": "a retrieval/delivery task — bring an object to a person or place",
            "open_close": "an open/close task — access or secure something",
            "appliance": "an appliance task — operate a device",
            "cleaning": "a cleaning task — tidy up a surface",
            "organization": "an organization task — sort and arrange items",
            "constraint": "a constraint-based task — find the right object by property",
            "semantic": "a semantic reasoning task — pick the correct item for a situation",
            "anomaly_response": "an emergency response task — handle an anomaly or hazard",
        }
        desc = category_descriptions.get(task_category, "")
        if desc:
            category_note = f"\nThis is {desc}. Make the instruction convey the task PURPOSE, not just mechanical steps."

    user_prompt = f"""\
Task: {task_name} ({task_category or 'unknown category'})
Target room: {target_room}{category_note}{scene_section}

Objects placed in the scene:
{objects_str}

Write a short mechanical instruction. Use actual object names from the list. For scene-native objects, don't claim a source location.
Output JSON:
{{"instruction": "the instruction", "task_description": "one-line summary"}}"""

    return client.call(_SYSTEM_INSTRUCTION, user_prompt)


# ======================================================================
# Prompt: Solution Plan Generation
# ======================================================================
_SYSTEM_SOLUTION_PLAN = """\
You are a task planning expert for embodied AI robots.
Given a task instruction, a list of objects in the scene (with positions and movability), generate a precise step sequence.

Available action primitives:
- MOVE: Move near a target object or room. **MUST include target_object** (the specific object to navigate to). target_room is required when moving to a different room (e.g. from kitchen_0 to living_room_0). Never leave target_object empty — if the target is a general area, pick the nearest object in that area from the task_objects list.
- PICK: Pick up a movable object. Requires target_object
- PLACE: Place held object on a target surface/object. Requires target_object (placement target)
- INTERACT: Interact with an object (open, press, toggle). Can specify tool_object and target_object

Key rules:
1. **Inventory tracking**: Every step must list the objects in hand AFTER completing that step. After PICK, inventory gains the object. After PLACE, inventory loses the object. MOVE and INTERACT do not change inventory — carry forward the inventory from the previous step.
2. **MOVE before PICK**: Cannot PICK a distant object without moving to it first
3. **MOVE before PLACE/INTERACT**: Must move to the target location first
4. **MOVE must always have a target_object**: Every MOVE step must specify a concrete target_object from the object list. target_room alone is NOT sufficient (except for the very first step if robot is in a different room). Use the nearest scene object as the navigation target.
5. **Reused objects (scene-native large furniture) cannot be PICK-ed**: They are fixed; only MOVE to them and INTERACT
6. **Cross-room requires MOVE to target room**: e.g. if robot is in kitchen_0 but needs an object in living_room_0
7. **Linear dependency between steps**: Each step's result is a prerequisite for the next
8. **Tool use chain**: If tool A is needed to operate target B: PICK A → MOVE to B → INTERACT(tool=A, target=B)
9. **Use actual placement positions**: Each object's info includes placed_on (actual support surface) and room (actual room). MOVE and PLACE steps must reference actual positions; do not assume objects are on tables or other furniture. For example, if an object is on bottom_cabinet_xxx, the MOVE target should be that cabinet.
10. **NEVER fabricate object_ids**: All target_object values must use object_ids from the provided object list. If the instruction mentions an object (e.g. fridge, countertop) but it is not in the list, use target_room instead or skip that step. Never invent IDs like "fridge", "coffee_maker_001", "dispenser_area". Objects marked as reference_only may also serve as INTERACT targets.
11. **Plan must end with PLACE or INTERACT**: The last step should be a PLACE (final placement) or INTERACT (e.g. closing a cabinet after placing something inside).

**Robot capability constraints (strictly enforced)**:
- Can only PICK (pick up), PLACE (put down — supports OnTop on surfaces or Inside in containers), MOVE (navigate), INTERACT (open/close/press/toggle)
- **Cannot**: cut, pour liquids, stir, measure, scoop, mix
- Food ingredients are treated as pre-prepared; no cutting or preparation needed
- INTERACT only for: opening/closing doors, pressing buttons, toggling switches
- PLACE inside containers (pots, bowls, cabinets, fridges) is valid

You must respond in valid JSON format."""


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
            "reasoning": "brief explanation of the plan"
        } or None
    """
    objects_str = json.dumps(task_objects, ensure_ascii=False, indent=2)

    user_prompt = f"""\
Task instruction: {task_instruction}
Robot initial position: {robot_room or target_room}
Scene rooms: {', '.join(rooms)}
Target room: {target_room}

Task objects in scene:
{objects_str}

Generate a precise step sequence. Output JSON:
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
      "inventory": ["saucepan_001"]
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
  "reasoning": "brief explanation of the linear logic"
}}"""

    return client.call(_SYSTEM_SOLUTION_PLAN, user_prompt)


def validate_task(
    client: LLMClient,
    task_instruction: str,
    task_objects: list[dict],
    target_room: str,
    rooms: list[str],
    solution_plan: list[dict] | None = None,
    scene_furniture: dict[str, list[str]] | None = None,
) -> dict | None:
    """Use LLM to validate task feasibility.

    Args:
        scene_furniture: optional {room_name: [category1, ...]} showing what
            furniture/appliances exist in each room. Used to check if the
            instruction references non-existent destination objects.

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

    scene_section = ""
    if scene_furniture:
        furniture_lines = []
        for room, cats in sorted(scene_furniture.items()):
            furniture_lines.append(f"  {room}: {', '.join(sorted(set(cats)))}")
        scene_section = "\n\nScene furniture (already exists):\n" + \
                        "\n".join(furniture_lines)

    plan_section = ""
    if solution_plan:
        plan_str = json.dumps(solution_plan, ensure_ascii=False, indent=2)
        plan_section = f"\n\nProposed solution plan:\n{plan_str}"

    user_prompt = f"""\
Task: {task_instruction}
Target room: {target_room}
All rooms: {', '.join(rooms)}{scene_section}

Objects to be spawned:
{objects_str}{plan_section}

Check if this task is clearly impossible. Default to feasible.
Output JSON:
{{
  "feasible": true/false,
  "is_linear": true/false,
  "is_linear": true/false,
  "confidence": 0.0-1.0,
  "instruction_clear": true/false,
  "objects_sufficient": true/false,
  "room_assignment_reasonable": true/false,
  "linearity_analysis": "briefly explain the dependency between steps, why it is/is not a linear task",
  "issues": ["issue 1 (if any)", "issue 2 (if any)"],
  "improved_instruction": "improved instruction if needed; otherwise null"
}}"""

    return client.call(_SYSTEM_VALIDATE, user_prompt)


# ======================================================================
# Prompt: Select Task Category (Env-A new paradigm)
# ======================================================================
_SYSTEM_TASK_CATEGORY = """\
You are a task designer for a household robot simulation. Pick a task category for the current scene.

Categories:
- **retrieval_delivery**: pick up an object and bring it somewhere
- **open_close**: open or close a door, window, cabinet, fridge
- **appliance**: turn on/off a light, TV, stove
- **cleaning**: clean a table, floor, or sink
- **organization**: organize books, medicine, or food items
- **constraint**: retrieve based on a property (nearest, largest, smallest)
- **semantic**: retrieve the correct object based on context

Pick a category that fits the scene furniture. Slightly prefer variety — the given recently used categories can still be chosen but are less preferred.

Respond in JSON: {"selected_category": "category_name", "reason": "one sentence"}"""


def select_task_category(
    client: LLMClient,
    scene_furniture: dict[str, list[str]],
    used_categories: set[str] | None = None,
) -> dict | None:
    """Use LLM to pick a task category based on scene contents.

    Args:
        used_categories: categories already used in previous runs, to avoid repetition.

    Returns:
        {"selected_category": str, "reason": str} or None
    """
    furniture_lines = []
    for room, cats in sorted(scene_furniture.items()):
        furniture_lines.append(f"  {room}: {', '.join(sorted(set(cats)))}")

    avoid_note = ""
    if used_categories:
        avoid_note = f"\nRecently used categories (slightly less preferred): {', '.join(sorted(used_categories))}"

    user_prompt = f"""\
Scene furniture:
{chr(10).join(furniture_lines)}{avoid_note}

Pick a category. Slightly prefer variety but don't force it. Output JSON:
{{"selected_category": "category_name", "reason": "one sentence"}}"""

    return client.call(_SYSTEM_TASK_CATEGORY, user_prompt)


# ======================================================================
# Prompt: Find Required Objects (Env-A new paradigm)
# ======================================================================
_SYSTEM_FIND_OBJECTS = """\
You are a task designer for a household robot simulation. Given a task name and the scene contents, determine the minimum set of objects needed.

The robot can: PICK, PLACE (on surfaces or inside containers), MOVE, INTERACT (open/close, press/toggle).

Rules:
- 1-2 target objects are ideal (the objects the robot manipulates)
- Choose objects that can be placed on available furniture in the scene
- Do NOT choose large fixed furniture (table, bed, sofa, countertop, cabinet) as target objects — they are support surfaces
- If the task needs a tool (e.g., fire extinguisher for fire), include it
- If the task is open/close or appliance, the target is the fixture itself (e.g., cabinet, light switch)

Respond in JSON: {"objects": [{"name": "descriptive_name", "category_hint": "what_kind_of_object", "role": "target/tool/support"}], "reasoning": "brief explanation"}"""


def find_required_objects(
    client: LLMClient,
    task_name: str,
    task_category: str,
    scene_furniture: dict[str, list[str]],
) -> dict | None:
    """Use LLM to determine the minimum required objects for a task.

    Returns:
        {"objects": [{"name": str, "category_hint": str, "role": str}], "reasoning": str} or None
    """
    furniture_lines = []
    for room, cats in sorted(scene_furniture.items()):
        furniture_lines.append(f"  {room}: {', '.join(sorted(set(cats)))}")

    user_prompt = f"""\
Task: {task_name}
Category: {task_category}

Available furniture in scene:
{chr(10).join(furniture_lines)}

Determine the minimum objects needed. Output JSON:
{{"objects": [{{"name": "object_name", "category_hint": "what_kind_of_object", "role": "target/tool/support"}}], "reasoning": "brief explanation"}}"""

    return client.call(_SYSTEM_FIND_OBJECTS, user_prompt)


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
    if model and str(model).strip().lower() in {"none", "off", "disabled", "false", "0"}:
        return None
    client = LLMClient(api_key=api_key, model=model, base_url=base_url)
    return client if client.available else None
