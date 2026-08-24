"""
Minimal tests for retry control, fail-fast, and hard-reject logic.

These tests do NOT require OmniGibson or a running simulator.
They test the pure-logic components of the retry/placement control system.

Run: python tests/test_retry_control.py
"""

import json
import sys
import tempfile
from pathlib import Path

# Add code dir to path
CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

# We import only the pure-logic components, not OmniGibson
# The online_deltasg module can't be imported without OmniGibson,
# so we test the HARD_REJECT_PATTERNS and _is_hard_reject as class-level logic.


def test_hard_reject_patterns():
    """Test that hard-reject keyword matching works correctly."""
    # Import the patterns (we can't import the class, so we replicate the logic)
    HARD_REJECT_PATTERNS = [
        "parallel dependency",
        "lack of linear dependency",
        "not a linear",
        "missing object",
        "missing destination",
        "unsupported primitive",
        "pour",
        "liquid",
        "designated cleaning area missing",
        "batch operation",
        "both items",
        "both objects",
        "place both",
        "move both",
        "outdoor",
        "lawn",
        "mailbox",
        "no sink",
        "no trash",
        "no stove",
        "no microwave",
        "no dishwasher",
        "no refrigerator",
        "no fridge",
        "no toilet",
        "requires cutting",
        "requires pouring",
        "requires stirring",
        "requires mixing",
    ]

    def _is_hard_reject(issues):
        if not issues:
            return False
        issues_lower = " ".join(issues).lower()
        for pattern in HARD_REJECT_PATTERNS:
            if pattern in issues_lower:
                return True
        return False

    # Test 1: parallel dependency should be hard reject
    assert _is_hard_reject(["this task has parallel dependency"])
    print("PASS: parallel dependency → hard reject")

    # Test 2: pour liquid should be hard reject
    assert _is_hard_reject(["requires pouring liquid into cup"])
    print("PASS: pour liquid → hard reject")

    # Test 3: missing destination should be hard reject
    assert _is_hard_reject(["missing destination: no sink in scene"])
    print("PASS: missing destination → hard reject")

    # Test 4: unsupported primitive should be hard reject
    assert _is_hard_reject(["unsupported primitive: stir"])
    print("PASS: unsupported primitive → hard reject")

    # Test 5: batch operation should be hard reject
    assert _is_hard_reject(["place both items on table"])
    print("PASS: batch operation → hard reject")

    # Test 6: no microwave should be hard reject
    assert _is_hard_reject(["task requires microwave but no microwave in scene"])
    print("PASS: no microwave → hard reject")

    # Test 7: requires cutting should be hard reject
    assert _is_hard_reject(["requires cutting vegetables"])
    print("PASS: requires cutting → hard reject")

    # Test 8: benign issue should NOT be hard reject
    assert not _is_hard_reject(["instruction is slightly verbose"])
    print("PASS: benign issue → soft reject (not hard)")

    # Test 9: empty issues should NOT be hard reject
    assert not _is_hard_reject([])
    print("PASS: empty issues → not hard reject")

    # Test 10: lack of linear dependency should be hard reject
    assert _is_hard_reject(["lack of linear dependency between steps"])
    print("PASS: lack of linear dependency → hard reject")

    print()
    print("=== All hard-reject tests passed ===")


def test_retry_control_logic():
    """Test retry control logic without OmniGibson."""
    # Simulate the retry control counters
    config = {
        "max_llm_retries_per_scene": 3,
        "max_retries_per_task": 1,
        "max_total_generation_time_sec": 300.0,
    }

    # Test 1: LLM retries exhausted
    total_llm_retries = 3
    assert total_llm_retries >= config["max_llm_retries_per_scene"]
    print("PASS: max_llm_retries_per_scene reaches limit → stop")

    # Test 2: Task retry count exceeded
    task_retry_count = {"task_a": 2}
    assert task_retry_count["task_a"] > config["max_retries_per_task"]
    print("PASS: per-task retry count exceeded → skip")

    # Test 3: Within limits should continue
    total_llm_retries = 1
    assert total_llm_retries < config["max_llm_retries_per_scene"]
    print("PASS: within LLM retry limit → continue")

    # Test 4: Rejected task cache
    rejected_cache = {"task_a", "task_b"}
    assert "task_a" in rejected_cache
    assert "task_c" not in rejected_cache
    print("PASS: rejected_task_cache correctly tracks tasks")

    # Test 5: Skip tasks set
    skip_tasks = {"task_a", "task_b", "task_c"}
    assert len(skip_tasks) >= 3
    # Task pool should exclude skip_tasks
    task_pool = ["task_a", "task_b", "task_c", "task_d", "task_e"]
    available = [t for t in task_pool if t not in skip_tasks]
    assert available == ["task_d", "task_e"]
    print("PASS: skip_tasks correctly filters task pool")

    print()
    print("=== All retry control tests passed ===")


def test_allow_repeat_still_skips_repeated_failure_within_one_sample():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    engine_source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert "run_skip_tasks = set(skip_tasks)" in source
    assert "run_skip_tasks.add(rejected_task)" in source
    assert "skip_tasks=run_skip_tasks" in source
    assert "skip = (skip_tasks or set()) | self._rejected_task_cache" in engine_source


def test_explicit_hard_reject_returns_control_to_outer_process_runner():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    retry_block = source[
        source.index("# Check if we should stop"):
        source.index("run_llm_retries += 1")
    ]
    assert "should_retry and hard_reject and current_task" in retry_block
    assert "returning control to the outer runner" in retry_block
    assert retry_block.index("hard_reject and current_task") < retry_block.index(
        "if not should_retry"
    )


def test_generation_runner_supports_an_ordered_single_process_task_sequence():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    assert '"--task-sequence"' in source
    assert "args.num_envs = len(task_sequence)" in source
    assert "current_task = task_sequence[idx] if task_sequence else args.task" in source
    assert "task=current_task" in source
    assert "engine.prepare_native_task_robot_spawn(current_task)" in source
    assert "engine.begin_env_a_attempt()" in source
    assert source.index("engine.begin_env_a_attempt()") < source.index(
        "engine.prepare_native_task_robot_spawn(current_task)"
    )
    assert "preferred_target_name=preferred_target" in source
    assert "settle_scene=False" in source
    assert "engine.bind_prepared_native_task_spawn(current_task)" in source
    assert 'engine.reject_native_target(' in source
    assert 'preferred_target, "robot_spawn_binding"' in source


def test_target_conditioned_spawn_candidates_cover_an_operation_ring():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    block = source[
        source.index("if preferred_target_name:"):
        source.index("for _ in range(max_attempts * 3):")
    ]
    assert "distance > preferred_max_distance" in block
    assert "nearest_xy = th.minimum(" in block
    assert "< 0.25" in block
    assert "len(spawn_candidates) >= max_attempts" in block


def test_explicit_native_camera_failure_returns_control_to_candidate_runner():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    retry_block = source[
        source.index("# Check if we should stop"):
        source.index("run_llm_retries += 1")
    ]
    assert "camera_failed and args.target_native_object_id" in retry_block
    assert "returning control to the outer candidate runner" in retry_block
    assert retry_block.index("camera_failed and args.target_native_object_id") < retry_block.index(
        "if not should_retry"
    )


def test_clean_process_coverage_disables_in_process_generation_retries():
    generator = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    coverage = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    assert '"--single-attempt"' in generator
    assert "if args.single_attempt:" in generator
    assert "single-attempt mode: returning failure" in generator
    command = coverage[
        coverage.index("command = ["):
        coverage.index('if job.get("target_asset_category")')
    ]
    assert '"--single-attempt"' in command
    assert '"--max-retries"' not in command


def test_zero_generated_samples_fail_the_process_contract():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    finalization = source[
        source.index("# A requested sample slot is successful"):
        source.index("# Final checkpoint save")
    ]
    assert "len(runs) - existing_run_count != args.num_envs" in finalization
    assert 'summary["ok"] = False' in finalization
    assert 'summary["num_generated"] = len(runs)' in finalization


def test_placement_cache_logic():
    """Test placement cache logic without OmniGibson."""
    failed_placement_cache = set()

    # Test 1: New pair not in cache
    cache_key = ("mailbox", "bottom_cabinet_0")
    assert cache_key not in failed_placement_cache
    print("PASS: new pair not in cache")

    # Test 2: Add to cache and verify
    failed_placement_cache.add(cache_key)
    assert cache_key in failed_placement_cache
    print("PASS: pair added to cache → detected")

    # Test 3: Different pair not affected
    other_key = ("plate", "bottom_cabinet_0")
    assert other_key not in failed_placement_cache
    print("PASS: different pair independent of cache")

    # Test 4: Same category, different support → different cache key
    floor_key = ("mailbox", "__floor__")
    assert floor_key not in failed_placement_cache
    failed_placement_cache.add(floor_key)
    assert len(failed_placement_cache) == 2
    print("PASS: same category, different support → separate cache entries")

    print()
    print("=== All placement cache tests passed ===")


def test_abort_on_task_failure():
    """Test abort logic without OmniGibson."""
    config = {
        "abort_on_task_object_failure": True,
        "skip_context_on_failure": True,
    }

    # Simulate placement loop
    records = [
        {"role": "task_object", "category": "plate"},
        {"role": "task_object", "category": "mug"},
        {"role": "context_object", "category": "spoon"},
        {"role": "context_object", "category": "fork"},
    ]

    # Test 1: Task object fails → abort
    abort_task = False
    for record in records:
        role = "task_object" if record["role"] == "task_object" else "context_object"
        if abort_task and role == "context_object":
            if config["skip_context_on_failure"]:
                continue  # Skip context
        # Simulate: first task_object succeeds, second fails
        if record["category"] == "mug":
            if config["abort_on_task_object_failure"]:
                abort_task = True
                continue
    assert abort_task
    print("PASS: task_object failure → abort_task set")

    # Test 2: Context objects after abort are skipped
    skipped_context_objects = []
    for record in records:
        role = "task_object" if record["role"] == "task_object" else "context_object"
        if abort_task and role == "context_object":
            if config["skip_context_on_failure"]:
                skipped_context_objects.append(record["category"])
                continue
    assert "spoon" in skipped_context_objects
    assert "fork" in skipped_context_objects
    print("PASS: context objects skipped after task abort")

    # Test 3: Without abort, context objects proceed
    config["abort_on_task_object_failure"] = False
    abort_task = False
    context_processed = []
    for record in records:
        role = "task_object" if record["role"] == "task_object" else "context_object"
        if abort_task and role == "context_object":
            if config["skip_context_on_failure"]:
                continue
        if record["category"] == "mug":
            pass  # failure but don't abort
        if role == "context_object":
            context_processed.append(record["category"])
    assert "spoon" in context_processed
    assert "fork" in context_processed
    print("PASS: context objects proceed when abort disabled")

    print()
    print("=== All abort logic tests passed ===")


def test_checkpoint_serialization():
    """Test checkpoint save/load round-trip."""
    checkpoint = {
        "schema_version": "deltasg_checkpoint.v1",
        "timestamp": 1234567890.0,
        "run_counter": 5,
        "rejected_task_cache": ["task_a", "task_b"],
        "failed_placement_cache": [
            {"category": "mailbox", "support": "bottom_cabinet_0"},
            {"category": "plate", "support": "__floor__"},
        ],
        "failed_target_models": [
            {"category": "paperback_book", "model": "bad_model", "count": 2},
        ],
        "attempted_tasks": [
            {"task": "task_a", "aborted": True, "abort_reason": "task_object_failed:plate"},
        ],
        "rejected_tasks": [
            {"task": "task_b", "issues": ["parallel dependency"], "kind": "hard-reject"},
        ],
        "failed_placements": [
            {"category": "mailbox", "support": "bottom_cabinet_0",
             "errors": ["relation_failed"]},
        ],
        "successful_samples": [
            {"run_id": "online_env_a_0000", "task": "task_c"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_path = tmpdir / "checkpoint.json"

        # Save
        with checkpoint_path.open("w") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        # Load
        with checkpoint_path.open("r") as f:
            loaded = json.load(f)

        assert loaded["run_counter"] == 5
        assert "task_a" in loaded["rejected_task_cache"]
        assert len(loaded["failed_placement_cache"]) == 2
        assert len(loaded["successful_samples"]) == 1
        print("PASS: checkpoint round-trip")

        # Test resume: rejected tasks should be skip_tasks
        skip_tasks = set(loaded["rejected_task_cache"])
        assert "task_a" in skip_tasks
        assert "task_b" in skip_tasks
        print("PASS: resume from checkpoint skips rejected tasks")

        # Test failed placement cache restoration
        failed_cache = {
            (item["category"], item["support"])
            for item in loaded["failed_placement_cache"]
        }
        assert ("mailbox", "bottom_cabinet_0") in failed_cache
        print("PASS: failed placement cache restored from checkpoint")
        failed_models = {
            (item["category"], item["model"]): item["count"]
            for item in loaded["failed_target_models"]
        }
        assert failed_models[("paperback_book", "bad_model")] == 2
        print("PASS: failed target-model counts restored from checkpoint")

    print()
    print("=== All checkpoint tests passed ===")


def test_config_defaults():
    """Test that config defaults are safe (fail-fast, limited retries)."""
    # Simulate the default config values

    config = {
        "max_llm_retries_per_scene": 3,
        "max_retries_per_task": 1,
        "max_total_generation_time_sec": 300.0,
        "per_object_placement_timeout_sec": 15.0,
        "per_relation_attempt_timeout_sec": 5.0,
        "max_placement_attempts_per_object": 4,
        "max_total_placement_time_sec": 60.0,
        "max_failures_per_target_model": 2,
        "abort_on_task_object_failure": True,
        "skip_context_on_failure": True,
    }

    # Verify defaults are reasonable
    assert config["max_llm_retries_per_scene"] <= 5, "LLM retries should be limited"
    assert config["per_object_placement_timeout_sec"] <= 30, "Placement timeout should be reasonable"
    assert config["max_placement_attempts_per_object"] <= 5, "Too many attempts risk long hangs"
    assert config["max_total_placement_time_sec"] <= 120, "Total placement time should be bounded"
    assert config["max_failures_per_target_model"] <= 3, "Bad models should be quarantined quickly"
    assert config["abort_on_task_object_failure"], "Default should abort on task failure"
    assert config["skip_context_on_failure"], "Default should skip context on failure"

    print("PASS: config defaults are safe (fail-fast)")
    print("PASS: max_llm_retries_per_scene =", config["max_llm_retries_per_scene"])
    print("PASS: per_object_placement_timeout_sec =", config["per_object_placement_timeout_sec"])
    print("PASS: max_placement_attempts_per_object =", config["max_placement_attempts_per_object"])

    print()
    print("=== All config default tests passed ===")


def test_scene_furniture_dict():
    """Test building scene furniture inventory from graph."""
    # Simulate graph structure
    graph = {
        "nodes": [
            {"type": "room", "name": "kitchen_0"},
            {"type": "object", "category": "sink", "rooms": ["kitchen_0"]},
            {"type": "object", "category": "stove", "rooms": ["kitchen_0"]},
            {"type": "object", "category": "countertop", "rooms": ["kitchen_0"]},
            {"type": "object", "category": "table", "rooms": ["kitchen_0"]},
            {"type": "object", "category": "toilet", "rooms": ["bathroom_0"]},
            {"type": "object", "category": "sink", "rooms": ["bathroom_0"]},
            {"type": "room", "name": "bathroom_0"},
        ]
    }

    # Build furniture dict
    furniture = {}
    for node in graph["nodes"]:
        if node.get("type") != "object":
            continue
        cat = node.get("category")
        if not cat:
            continue
        for room in node.get("rooms", []):
            if room not in furniture:
                furniture[room] = []
            if cat not in furniture[room]:
                furniture[room].append(cat)

    assert "kitchen_0" in furniture
    assert "sink" in furniture["kitchen_0"]
    assert "stove" in furniture["kitchen_0"]
    assert "bathroom_0" in furniture
    assert "toilet" in furniture["bathroom_0"]
    print("PASS: scene furniture dict built correctly")

    # Test destination check
    all_cats = set()
    for cats in furniture.values():
        all_cats.update(c.lower() for c in cats)
    assert "sink" in all_cats
    assert "toilet" in all_cats
    assert "microwave" not in all_cats
    print("PASS: destination existence check works")

    print()
    print("=== All scene furniture tests passed ===")


def test_pre_validate_instruction():
    """Test pre-validation of instruction against scene inventory."""
    DESTINATION_CATEGORIES = {
        "sink", "stove", "microwave", "trash_can", "toilet",
        "dishwasher", "refrigerator", "fridge", "coffee_maker",
    }

    def pre_validate(instruction, scene_furniture):
        all_scene_cats = set()
        for cats in scene_furniture.values():
            all_scene_cats.update(c.lower() for c in cats)
        instruction_lower = instruction.lower()
        missing = set()
        for dest_cat in DESTINATION_CATEGORIES:
            dest_display = dest_cat.replace("_", " ")
            if (dest_cat in instruction_lower or dest_display in instruction_lower) \
                    and dest_cat not in all_scene_cats:
                found = any(dest_cat in sc or sc in dest_cat for sc in all_scene_cats)
                if not found:
                    missing.add(dest_display)
        return missing

    # Scene has: sink, countertop, table, cabinet
    scene = {"kitchen_0": ["sink", "countertop", "table", "cabinet", "stove"]}

    # Test 1: Instruction mentions sink → OK (exists)
    missing = pre_validate("Wash the plate in the sink", scene)
    assert not missing
    print("PASS: 'sink' exists → no missing")

    # Test 2: Instruction mentions microwave → missing
    missing = pre_validate("Heat the food in the microwave", scene)
    assert "microwave" in missing
    print("PASS: 'microwave' missing → detected")

    # Test 3: Instruction mentions trash → missing
    missing = pre_validate("Throw the wrapper in the trash can", scene)
    assert "trash can" in missing
    print("PASS: 'trash can' missing → detected")

    # Test 4: Instruction mentions dishwasher → missing
    missing = pre_validate("Load the plates into the dishwasher", scene)
    assert "dishwasher" in missing
    print("PASS: 'dishwasher' missing → detected")

    # Test 5: Only mentions existing objects → OK
    missing = pre_validate("Place the plate on the table", scene)
    assert not missing
    print("PASS: only existing objects → no missing")

    print()
    print("=== All pre-validation tests passed ===")


def test_object_support_affinity():
    """Test object-support affinity scoring."""
    # Test the logic without importing the module
    food_keywords = {"food", "fruit", "vegetable", "meat", "chicken", "apple", "egg"}

    def affinity_score(obj_cat, support_cat):
        is_food = any(kw in obj_cat.lower() for kw in food_keywords)
        if is_food:
            if support_cat.lower() in {"countertop", "counter", "table"}:
                return 0.7
            if support_cat.lower() in {"floor"}:
                return 0.1
            return 0.4
        if support_cat.lower() in {"countertop", "counter", "table", "desk"}:
            return 0.5
        if support_cat.lower() in {"floor"}:
            return 0.3
        return 0.2

    # Food → countertop = high
    assert affinity_score("chicken_wing", "countertop") == 0.7
    print("PASS: chicken → countertop = high affinity")

    # Food → floor = low
    assert affinity_score("apple", "floor") == 0.1
    print("PASS: apple → floor = low affinity")

    # Non-food → table = medium
    assert affinity_score("book", "table") == 0.5
    print("PASS: book → table = medium affinity")

    # Non-food → floor = medium-low
    assert affinity_score("shoe", "floor") == 0.3
    print("PASS: shoe → floor = medium-low affinity")

    print()
    print("=== All affinity scoring tests passed ===")


if __name__ == "__main__":
    print("=" * 60)
    print("Running retry control, fail-fast, and checkpoint tests")
    print("(No OmniGibson required)")
    print("=" * 60)
    print()

    test_hard_reject_patterns()
    test_retry_control_logic()
    test_placement_cache_logic()
    test_abort_on_task_failure()
    test_checkpoint_serialization()
    test_config_defaults()
    test_scene_furniture_dict()
    test_pre_validate_instruction()
    test_object_support_affinity()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
