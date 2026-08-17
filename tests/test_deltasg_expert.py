from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from deltasg_expert import (  # noqa: E402
    ExpertPlanError,
    ExpertStep,
    PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION,
    PLACE_SAMPLING_SURFACE_XY_MARGIN,
    PLACE_SAMPLING_SURFACE_Z_EPSILON,
    PLACE_SUPPORT_EMBED_TOLERANCE,
    PLACE_SUPPORT_FLOAT_TOLERANCE,
    PLACE_RELEASE_CONTACT_CLEARANCE,
    PLACE_SUPPORT_PROBE_MARGIN,
    PLACE_SUPPORT_PROBE_RADIUS_FRACTION,
    PLACE_SUPPORT_REFERENCE_SURFACE_TOLERANCE,
    PLACE_SUPPORT_REFERENCE_TOP_EPSILON,
    PLACE_SUPPORT_REFERENCE_XY_RADIUS,
    SUPPORTED_APPLIANCE_TASKS,
    SUPPORTED_ENV_A_TASKS,
    SUPPORTED_OPEN_CLOSE_TASKS,
    SUPPORTED_RETRIEVAL_DELIVERY_TASKS,
    compile_expert_plan,
    direct_floor_primary_view_error,
    evaluate_manipulation_height,
    validate_env_a_plan_contract,
    validate_visibility_snapshot,
)
from audit_deltasg_expert import (  # noqa: E402
    action_artifact_error,
    is_strict_vla_eligible,
    rgb_artifact_error,
    segmentation_artifact_error,
)


def sample(task_name, plan, plan_objects):
    return {
        "ok": True,
        "task_environment": {
            "task": {
                "primary_behavior_task": task_name,
                "plan_objects": plan_objects,
            },
            "solution_plan": plan,
        },
    }


@pytest.mark.parametrize("task_name", sorted(SUPPORTED_ENV_A_TASKS))
def test_every_supported_enva_task_compiles_to_its_physical_primitive(task_name):
    if task_name in SUPPORTED_RETRIEVAL_DELIVERY_TASKS:
        objects = [{"object_id": "item_0", "category": "bottle_of_water"}]
        plan = [{"step_id": 1, "primitive": "PICK", "target_object": "item_0"}]
        expected = {"GRASP"}
        if task_name.startswith("deliver_"):
            objects.append({"object_id": "table_0", "category": "table", "reference_only": True})
            plan.append({"step_id": 2, "primitive": "PLACE", "target_object": "table_0"})
            expected.add("PLACE_ON_TOP")
    else:
        objects = [{"object_id": "fixture_0", "category": "fixture", "reference_only": True}]
        plan = [{
            "step_id": 1,
            "primitive": "INTERACT",
            "target_object": "fixture_0",
            "nl": task_name.replace("_", " "),
        }]
        if task_name in SUPPORTED_OPEN_CLOSE_TASKS:
            expected = {"OPEN" if task_name.startswith("open_") else "CLOSE"}
        else:
            assert task_name in SUPPORTED_APPLIANCE_TASKS
            expected = {"TOGGLE_ON" if task_name.startswith("turn_on_") else "TOGGLE_OFF"}
    compiled = compile_expert_plan(sample(task_name, plan, objects))
    assert expected <= {step.primitive for step in compiled.steps}


def test_repairs_retrieval_interact_to_grasp():
    run = sample(
        "retrieve_book",
        [
            {"step_id": 1, "primitive": "MOVE", "target_room": "living_room_0"},
            {"step_id": 2, "primitive": "MOVE", "target_object": "book_0"},
            {"step_id": 3, "primitive": "INTERACT", "target_object": "book_0"},
        ],
        [{"object_id": "book_0", "category": "paperback_book", "room": "living_room_0"}],
    )
    compiled = compile_expert_plan(run)
    assert [step.primitive for step in compiled.steps] == ["NAVIGATE_TO", "GRASP"]
    assert compiled.steps[0].target_object == "book_0"
    assert compiled.steps[-1].inventory_after == ("book_0",)
    assert "repaired ambiguous INTERACT to GRASP" in compiled.warnings[0]


def test_compiles_delivery_inventory_and_inside_relation():
    run = sample(
        "deliver_food",
        [
            {"primitive": "MOVE", "target_object": "can_0"},
            {"primitive": "PICK", "target_object": "can_0"},
            {"primitive": "MOVE", "target_object": "cabinet_0"},
            {"primitive": "INTERACT", "target_object": "cabinet_0", "nl": "Open the cabinet"},
            {"primitive": "PLACE", "target_object": "cabinet_0", "nl": "Place it inside"},
            {"primitive": "INTERACT", "target_object": "cabinet_0", "nl": "Close the cabinet"},
        ],
        [
            {"object_id": "can_0", "category": "canned_food", "room": "kitchen_0"},
            {"object_id": "cabinet_0", "category": "cabinet", "room": "kitchen_0", "reference_only": True},
        ],
    )
    compiled = compile_expert_plan(run)
    assert [step.primitive for step in compiled.steps] == [
        "NAVIGATE_TO", "OPEN", "NAVIGATE_TO", "GRASP", "NAVIGATE_TO", "PLACE_INSIDE", "CLOSE"
    ]
    assert "moved destination OPEN before GRASP" in compiled.warnings[0]
    place = compiled.steps[5]
    assert place.carried_object == "can_0"
    assert place.inventory_before == ("can_0",)
    assert place.inventory_after == ()
    assert place.expected["relation"] == "Inside"
    assert "can_0" not in place.useful_objects
    assert compiled.steps[2].target_object == "can_0"


def test_explicit_on_top_destination_overrides_container_category_heuristic():
    run = sample(
        "deliver_food",
        [
            {"primitive": "PICK", "target_object": "can_0"},
            {"primitive": "PLACE", "target_object": "cabinet_0"},
        ],
        [
            {"object_id": "can_0", "category": "canned_food"},
            {
                "object_id": "cabinet_0",
                "category": "bottom_cabinet",
                "reference_only": True,
                "placement_mode": "on_top",
            },
        ],
    )
    compiled = validate_env_a_plan_contract(
        run, task_object_id="can_0", destination_object_id="cabinet_0"
    )
    assert compiled.steps[-1].primitive == "PLACE_ON_TOP"


def test_exact_plan_contract_rejects_semantically_wrong_but_known_target():
    run = sample(
        "retrieve_book",
        [{"primitive": "PICK", "target_object": "table_0"}],
        [
            {"object_id": "book_0", "category": "paperback_book"},
            {"object_id": "table_0", "category": "table", "reference_only": True},
        ],
    )
    with pytest.raises(ExpertPlanError, match="exact task object"):
        validate_env_a_plan_contract(run, task_object_id="book_0")


def test_exact_native_plan_contract_rejects_same_category_substitution():
    run = sample(
        "open_cabinet",
        [{"primitive": "INTERACT", "target_object": "cabinet_1"}],
        [
            {"object_id": "cabinet_0", "category": "cabinet", "reference_only": True},
            {"object_id": "cabinet_1", "category": "cabinet", "reference_only": True},
        ],
    )
    with pytest.raises(ExpertPlanError, match="exact target"):
        validate_env_a_plan_contract(run, native_target_id="cabinet_0")


@pytest.mark.parametrize(
    ("bad_plan", "message"),
    [
        ({"primitive": "PICK"}, "must be a list"),
        (["PICK"], "solution step must be an object"),
        ([{"primitive": "PICK", "target_object": "book_0", "step_id": "bad"}], "step_id"),
    ],
)
def test_malformed_llm_plan_is_a_structured_contract_error(bad_plan, message):
    run = sample(
        "retrieve_book",
        [],
        [{"object_id": "book_0", "category": "paperback_book"}],
    )
    run["task_environment"]["solution_plan"] = bad_plan
    with pytest.raises(ExpertPlanError, match=message):
        compile_expert_plan(run)


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [("open_fridge", "OPEN"), ("close_cabinet", "CLOSE"), ("turn_on_tv", "TOGGLE_ON"), ("turn_off_light", "TOGGLE_OFF")],
)
def test_native_task_interactions(task_name, expected):
    run = sample(
        task_name,
        [{"primitive": "MOVE", "target_object": "fixture_0"}, {"primitive": "INTERACT", "target_object": "fixture_0"}],
        [{"object_id": "fixture_0", "category": "fixture", "room": "room_0", "reused": True}],
    )
    assert compile_expert_plan(run).steps[-1].primitive == expected


def test_rejects_delivery_without_place():
    run = sample(
        "deliver_food",
        [{"primitive": "PICK", "target_object": "can_0"}],
        [{"object_id": "can_0", "category": "canned_food"}],
    )
    with pytest.raises(ExpertPlanError, match="no PLACE"):
        compile_expert_plan(run)


def test_inserts_open_close_for_openable_inside_destination():
    run = sample(
        "deliver_food",
        [
            {"primitive": "MOVE", "target_object": "can_0"},
            {"primitive": "PICK", "target_object": "can_0"},
            {"primitive": "MOVE", "target_object": "cabinet_0"},
            {"primitive": "PLACE", "target_object": "cabinet_0", "nl": "Place it inside"},
        ],
        [
            {"object_id": "can_0", "category": "canned_food"},
            {"object_id": "cabinet_0", "category": "cabinet", "reference_only": True},
        ],
    )
    run["before_graph"] = {
        "nodes": [{"id": "cabinet_0", "type": "object", "available_states": ["Open", "Inside"]}]
    }
    compiled = compile_expert_plan(run)
    assert [step.primitive for step in compiled.steps] == [
        "NAVIGATE_TO", "OPEN", "NAVIGATE_TO", "GRASP", "NAVIGATE_TO", "PLACE_INSIDE", "CLOSE"
    ]
    assert any("inserted OPEN/CLOSE" in warning for warning in compiled.warnings)


def test_useful_visibility_union_and_fine_operation_robot_bbox():
    step = ExpertStep(
        step_id=1,
        primitive="GRASP",
        target_object="book_0",
        useful_objects=("book_0", "table_0"),
    )
    bbox = {"book_0": {"pixel_count": 20, "bbox_xyxy": [1, 2, 8, 10]}}
    assert validate_visibility_snapshot(step, ["book_0"], ["table_0"], bbox) == []
    errors = validate_visibility_snapshot(step, [], ["book_0", "table_0"], {})
    assert any("absent from robot primary view" in error for error in errors)
    assert any("no valid robot bbox" in error for error in errors)


def test_fine_operation_rejects_bbox_clipped_by_image_edge():
    step = ExpertStep(
        step_id=1,
        primitive="GRASP",
        target_object="book_0",
        useful_objects=("book_0",),
    )
    bbox = {
        "book_0": {
            "pixel_count": 503,
            "bbox_xyxy": [295, 12, 319, 36],
            "image_size": [320, 240],
        }
    }
    errors = validate_visibility_snapshot(step, ["book_0"], [], bbox)
    assert any("bbox is clipped" in error for error in errors)


def test_place_support_may_continue_below_image_when_top_is_framed():
    step = ExpertStep(
        step_id=1,
        primitive="PLACE_ON_TOP",
        target_object="table_0",
        useful_objects=("table_0",),
    )
    bbox = {
        "table_0": {
            "pixel_count": 6000,
            "bbox_xyxy": [60, 70, 260, 239],
            "image_size": [320, 240],
        }
    }
    assert validate_visibility_snapshot(step, ["table_0"], [], bbox) == []

    bbox["table_0"]["bbox_xyxy"] = [0, 70, 260, 239]
    errors = validate_visibility_snapshot(step, ["table_0"], [], bbox)
    assert any("bbox is clipped" in error for error in errors)


def test_inventory_objects_are_not_required_visible():
    step = ExpertStep(
        step_id=3,
        primitive="NAVIGATE_TO",
        target_object="table_0",
        inventory_before=("book_0",),
        useful_objects=("table_0",),
    )
    assert validate_visibility_snapshot(step, [], ["table_0"], {}) == []


def test_before_graph_does_not_add_unaddressed_robot_or_furniture():
    run = sample(
        "retrieve_book",
        [{"primitive": "PICK", "target_object": "book_0"}],
        [{"object_id": "book_0", "category": "book"}],
    )
    run["before_graph"] = {
        "nodes": [
            {"id": "book_0", "type": "object", "available_states": ["OnTop"]},
            {"id": "robot_random_name", "type": "object", "category": "agent"},
            {"id": "table_0", "type": "object", "category": "table"},
        ]
    }
    compiled = compile_expert_plan(run)
    assert compiled.object_ids == ("book_0",)


def test_manipulation_height_allows_tall_floor_object_but_rejects_tiny_one():
    bottle = evaluate_manipulation_height("GRASP", 0.0, 0.28, 0.0)
    key = evaluate_manipulation_height("GRASP", 0.0, 0.04, 0.0)
    assert bottle["eligible"] is True
    assert bottle["relative_height"] == pytest.approx(0.14)
    assert key["eligible"] is False
    assert "below" in key["reason"]


def test_manipulation_height_uses_support_surface_for_place():
    table = evaluate_manipulation_height("PLACE_ON_TOP", 0.0, 0.75, 0.0)
    floor = evaluate_manipulation_height("PLACE_ON_TOP", -0.02, 0.02, 0.0)
    assert table["eligible"] is True
    assert table["point_kind"] == "support_surface"
    assert floor["eligible"] is False


def test_manipulation_height_applies_to_native_interactions():
    switch = evaluate_manipulation_height("TOGGLE_ON", 0.90, 1.10, 0.0)
    high_cabinet = evaluate_manipulation_height("OPEN", 1.70, 2.10, 0.0)
    assert switch["eligible"] is True
    assert switch["point_kind"] == "object_center"
    assert high_cabinet["eligible"] is False


def test_direct_floor_primary_view_gate_bootstraps_support_for_low_targets():
    assert direct_floor_primary_view_error(0.108) is not None
    assert direct_floor_primary_view_error(0.18) is None


def test_uncontracted_taxonomy_task_is_rejected_by_expert_layer():
    run = sample(
        "retrieve_remote",
        [{"step_id": 1, "primitive": "PICK", "target_object": "ice_0"}],
        [{"object_id": "ice_0", "category": "ice"}],
    )
    with pytest.raises(ExpertPlanError, match="no validated Env-A physical contract"):
        compile_expert_plan(run)


@pytest.mark.parametrize(
    "task_name",
    [
        "retrieve_medicine", "retrieve_key", "retrieve_phone", "retrieve_book",
        "retrieve_drink", "retrieve_food",
    ],
)
def test_compiles_every_supported_retrieval_task(task_name):
    run = sample(
        task_name,
        [
            {"primitive": "MOVE", "target_object": "item_0"},
            {"primitive": "PICK", "target_object": "item_0"},
        ],
        [{"object_id": "item_0", "category": "task_item", "room": "room_0"}],
    )
    compiled = compile_expert_plan(run)
    assert compiled.task_family == "retrieval_delivery"
    assert compiled.steps[-1].primitive == "GRASP"


@pytest.mark.parametrize("task_name", ["deliver_medicine", "deliver_food", "deliver_drink"])
def test_compiles_every_supported_delivery_task(task_name):
    run = sample(
        task_name,
        [
            {"primitive": "MOVE", "target_object": "item_0"},
            {"primitive": "PICK", "target_object": "item_0"},
            {"primitive": "MOVE", "target_object": "table_0"},
            {"primitive": "PLACE", "target_object": "table_0"},
        ],
        [
            {"object_id": "item_0", "category": "task_item", "room": "room_0"},
            {"object_id": "table_0", "category": "table", "room": "room_1", "reference_only": True},
        ],
    )
    compiled = compile_expert_plan(run)
    assert compiled.task_family == "retrieval_delivery"
    assert compiled.steps[-1].primitive == "PLACE_ON_TOP"


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("open_door", "OPEN"), ("close_door", "CLOSE"),
        ("open_window", "OPEN"), ("close_window", "CLOSE"),
        ("open_fridge", "OPEN"), ("close_fridge", "CLOSE"),
        ("open_cabinet", "OPEN"), ("close_cabinet", "CLOSE"),
        ("turn_on_light", "TOGGLE_ON"), ("turn_off_light", "TOGGLE_OFF"),
        ("turn_on_tv", "TOGGLE_ON"), ("turn_off_tv", "TOGGLE_OFF"),
        ("turn_on_stove", "TOGGLE_ON"), ("turn_off_stove", "TOGGLE_OFF"),
    ],
)
def test_compiles_every_supported_native_state_task(task_name, expected):
    run = sample(
        task_name,
        [
            {"primitive": "MOVE", "target_object": "fixture_0"},
            {"primitive": "INTERACT", "target_object": "fixture_0"},
        ],
        [{"object_id": "fixture_0", "category": "fixture", "room": "room_0", "reused": True}],
    )
    compiled = compile_expert_plan(run)
    assert compiled.steps[-1].primitive == expected


def test_rgb_artifact_gate_rejects_missing_small_and_blank_images(tmp_path):
    assert rgb_artifact_error(None) == "missing"
    assert rgb_artifact_error(tmp_path / "missing.png") == "missing"

    small = tmp_path / "small.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(small)
    assert rgb_artifact_error(small) == "resolution_too_small:32x32"

    blank = tmp_path / "blank.png"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(blank)
    assert rgb_artifact_error(blank) == "blank_or_uniform"


def test_rgb_artifact_gate_accepts_a_decodable_nonblank_frame(tmp_path):
    path = tmp_path / "render.png"
    image = Image.new("RGB", (320, 240), (10, 20, 30))
    for x in range(160, 320):
        for y in range(240):
            image.putpixel((x, y), (180, 120, 70))
    image.save(path)
    assert rgb_artifact_error(path) is None


def test_segmentation_artifact_gate_rejects_missing_invalid_and_background(tmp_path):
    assert segmentation_artifact_error(None) == "missing"
    assert segmentation_artifact_error(tmp_path / "missing.npy") == "missing"

    small = tmp_path / "small.npy"
    np.save(small, np.ones((32, 32), dtype=np.uint16))
    assert segmentation_artifact_error(small) == "resolution_too_small:32x32"

    floating = tmp_path / "floating.npy"
    np.save(floating, np.ones((240, 320), dtype=np.float32))
    assert segmentation_artifact_error(floating) == "non_integer_dtype:float32"

    background = tmp_path / "background.npy"
    np.save(background, np.zeros((240, 320), dtype=np.uint16))
    assert segmentation_artifact_error(background) == "all_background"


def test_segmentation_artifact_gate_accepts_nonzero_integer_mask(tmp_path):
    path = tmp_path / "segmentation.npy"
    mask = np.zeros((240, 320), dtype=np.uint16)
    mask[20:80, 40:100] = 3
    np.save(path, mask)
    assert segmentation_artifact_error(path) is None


def test_vla_action_artifact_gate_rejects_empty_mismatch_and_nonfinite(tmp_path):
    empty = tmp_path / "empty.npy"
    np.save(empty, np.empty((0, 4), dtype=np.float32))
    assert action_artifact_error(empty, 0) == "empty"

    valid = tmp_path / "valid.npy"
    np.save(valid, np.ones((3, 4), dtype=np.float32))
    assert action_artifact_error(valid, 2) == "count_mismatch:3!=2"
    assert action_artifact_error(valid, 3) is None

    invalid = tmp_path / "invalid.npy"
    values = np.ones((2, 4), dtype=np.float32)
    values[0, 0] = np.nan
    np.save(invalid, values)
    assert action_artifact_error(invalid, 2) == "non_finite"


def test_strict_vla_gate_rejects_symbolic_assisted_and_profile_mismatch():
    result = {
        "accepted": True,
        "backend": {
            "name": "physical_control",
            "generation_solvability_profile": "physical_control",
            "generation_profile_verified": True,
            "physical_solubility_validation": True,
            "low_level_vla_actions_eligible": True,
            "assisted_interaction": False,
            "complete_action_trace": True,
        },
    }
    assert is_strict_vla_eligible(result, "physical_control")
    assert not is_strict_vla_eligible(result, "oracle_symbolic")
    result["backend"]["name"] = "oracle_symbolic"
    assert not is_strict_vla_eligible(result, "physical_control")
    result["backend"]["name"] = "physical_control"
    result["backend"]["assisted_interaction"] = True
    assert not is_strict_vla_eligible(result, "physical_control")
    result["backend"]["assisted_interaction"] = False
    result["backend"]["complete_action_trace"] = False
    assert not is_strict_vla_eligible(result, "physical_control")


def test_oracle_place_uses_official_relation_before_releasing_inventory():
    source = (
        Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py"
    ).read_text(encoding="utf-8")
    oracle = source[
        source.index("class DeltaSGOraclePrimitives"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    place = oracle[
        oracle.index("def _place_with_predicate"):
        oracle.index("def _navigate_to_obj")
    ]
    assert "state.set_value(obj, True)" in place
    assert "state.get_value(obj)" in place
    assert place.index("state.set_value(obj, True)") < place.index("yield from self._release()")
    assert "not changed" in place
    assert "or not reached" in place


def test_oracle_navigation_preserves_unheld_online_target_pose():
    source = (
        Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py"
    ).read_text(encoding="utf-8")
    oracle = source[
        source.index("class DeltaSGOraclePrimitives"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    navigate = oracle[
        oracle.index("def _navigate_to_obj"):
        oracle.index("def _navigate_to_pose")
    ]
    assert 'obj.name.startswith("online_env_")' in navigate
    assert "self._get_obj_in_hand() is not obj" in navigate
    assert navigate.index("yield from self._navigate_to_pose(candidate_pose)") < navigate.index(
        "obj.set_position_orientation("
    )


def test_delta_replay_sink_requires_root_pose_and_aabb_to_move_down():
    source = (
        Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py"
    ).read_text(encoding="utf-8")
    gate = source[
        source.index("def _delta_replay_integrity"):
        source.index("def _scene_objects")
    ]
    assert "vertical_displacement = float(" in gate
    assert "vertical_displacement < -0.01" in gate
    assert "replayed_aabb_top < comparison_aabb_top - 0.015" in gate
    assert '"vertical_displacement": vertical_displacement' in gate


def test_place_access_rejects_far_edge_candidate_with_blocked_corridor():
    from deltasg_expert import place_descent_corridor_blockers

    # attempt-7 / diag27 geometry: bottle candidate on the far west edge of
    # breakfast_table_skczfi_2 with armchair_vzhxuf_0 against that edge. The
    # armchair AABB intersects the swept descent column, so the candidate is
    # access-blocked and must be re-sampled.
    blockers = place_descent_corridor_blockers(
        place_pos=(-1.73, 1.11, 0.62),
        held_half_extents=(0.04, 0.04, 0.11),
        obstacle_aabbs=[
            ("armchair_vzhxuf_0", ([-2.35, 1.05, 0.0], [-1.76, 1.75, 1.05])),
            ("breakfast_table_skczfi_2", ([-1.793, 0.89, 0.0], [-1.329, 1.328, 0.506])),
        ],
    )
    assert blockers == ["armchair_vzhxuf_0"]


def test_place_access_accepts_robot_facing_candidate_with_clear_corridor():
    from deltasg_expert import place_descent_corridor_blockers

    # South-facing candidate on the same table: the support surface itself is
    # excluded by the caller and nothing else intersects the descent column.
    blockers = place_descent_corridor_blockers(
        place_pos=(-1.56, 0.98, 0.62),
        held_half_extents=(0.04, 0.04, 0.11),
        obstacle_aabbs=[
            ("armchair_vzhxuf_0", ([-2.35, 1.05, 0.0], [-1.76, 1.75, 1.05])),
        ],
    )
    assert blockers == []


def test_place_access_corridor_margin_and_hover_are_conservative():
    from deltasg_expert import place_descent_corridor_blockers

    # An obstacle just outside the bare footprint must still block once the
    # corridor margin is applied, and the swept column must extend through the
    # full hover clearance above the place pose.
    blockers = place_descent_corridor_blockers(
        place_pos=(-1.56, 0.98, 0.62),
        held_half_extents=(0.04, 0.04, 0.11),
        obstacle_aabbs=[
            ("tall_jar", ([-1.615, 0.94, 0.55], [-1.55, 1.02, 0.9])),
        ],
    )
    assert blockers == ["tall_jar"]


def test_physical_place_override_arms_access_and_hover_gates():
    source = (
        Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py"
    ).read_text(encoding="utf-8")
    assert "def _place_with_predicate(self, obj, predicate)" in source
    assert "place_descent_corridor_blockers(" in source
    assert "PLACE_ACCESS_REACH_MARGIN" in source
    assert "PLACE_ACCESS_HOVER_CLEARANCE" in source
    assert "_deltasg_place_hover" in source
    # Gates may reject candidates but must keep the stock flow as fallback.
    assert "fallback_used" in source
    assert '"reason": "descent_corridor_blocked"' in source
    assert '"reason": "reach_margin"' in source
    # Fix O: the reach margin only rejects far candidates that are IK-reachable
    # from the current base (stock would skip re-navigation and sweep low);
    # far non-reachable candidates are accepted (stock re-navigates), and the
    # exhaustion fallback prefers the closest corridor-clear candidate.
    assert '"in_reach": True' in source
    assert "fallback_best" in source
    assert "_target_in_reach_of_robot(" in source
    # Fix P: draws are sanity-checked against the pre-grasp reference surface
    # (get_base_aligned_bbox snapshot); corrupted draws never seed a fallback —
    # the primitive fails so apply_ref force-rebuilds and resamples.
    assert "PLACE_SAMPLING_SURFACE_XY_MARGIN" in source
    assert "PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION" in source
    assert "_place_sampling_surface_corruption(" in source
    assert '"reason": "sampling_surface_corrupted"' in source
    assert "_deltasg_reference_bboxes" in source
    assert "_deltasg_place_sampling_corrupted" in source
    assert "_capture_reference_scene_bboxes(" in source
    # Fix P2 (attempt-10): a corrupted batch voids gate-passing survivors
    # (clean_candidate_voided_corrupted_batch) so apply_ref rebuilds and
    # resamples instead of trusting a shifted candidate.
    assert '"reason": "clean_candidate_voided_corrupted_batch"' in source
    assert "held_base_aligned_extent" in source
    assert "target_ray_box_extent" in source
    # Fix P3 (adversarial review): the z gate compares reference-snapshot
    # quantities only (never live corruptible geometry) and derives its
    # tolerances from the sampler ray geometry — f*h + Z_OFFSET upward,
    # (1+f)*h + Z_OFFSET downward — so tall targets and tiered supports are
    # not falsely rejected while the corruption signatures (+0.09/+0.108 m on
    # the 0.506 m table) stay far above the derived upward bound.
    assert PLACE_SAMPLING_SURFACE_XY_MARGIN <= 0.05
    assert PLACE_SAMPLING_SURFACE_RAY_OFFSET_FRACTION == 0.02
    assert PLACE_SAMPLING_SURFACE_Z_EPSILON <= 0.02
    assert "held_reference" in source
    assert "z_up_tolerance" in source
    assert "z_down_tolerance" in source
    assert "PLACE_SAMPLING_SURFACE_Z_TOLERANCE" not in source
    assert "PLACE_SAMPLING_SURFACE_MAX_Z_DEFICIT" not in source
    # Fix P4 (attempt-11): accepted candidates are verified against the live
    # surface with a downward ray fan — the through-table draw class (bottom
    # embedded ~4 cm in solid tabletop, z_dev -0.0395 inside the reference
    # gate's tiered-support tolerance) must be rejected, the attempt-10
    # floating class stays rejected without needing a reference bbox, and a
    # whole batch of support failures forces the same stop/play rebuild.
    assert "_place_support_contact_check(" in source
    assert '"reason": "support_contact_inconsistent"' in source
    assert '"support_rejected"' in source or "'support_rejected'" in source
    assert "embedded_below_local_surface" in source
    assert "floating_above_support" in source
    assert "no_support_within_probe" in source
    assert "support_not_target" in source
    assert "accepted_support_check" in source
    assert "Fix P4: OnTop place candidates failed support-contact" in source
    assert PLACE_SUPPORT_EMBED_TOLERANCE <= 0.02
    assert PLACE_SUPPORT_FLOAT_TOLERANCE <= 0.02
    assert PLACE_SUPPORT_PROBE_MARGIN >= 0.05
    assert 0.5 <= PLACE_SUPPORT_PROBE_RADIUS_FRACTION <= 1.0
    # Fix P5 (attempt-11): the live probe shares the sampler's query layer,
    # so a uniform top-mesh loss needs the pre-grasp raycast map as the
    # independent source; static supports make the 0.015 tolerance noise-only.
    assert "support_surface_hits" in source
    assert "support_surface_lost_vs_reference" in source
    assert "reference_surface_min_z" in source
    assert PLACE_SUPPORT_REFERENCE_SURFACE_TOLERANCE <= 0.02
    assert 0.05 <= PLACE_SUPPORT_REFERENCE_XY_RADIUS <= 0.2
    # Review R1-R4 (adversarial review of P4/P5 before attempt 12):
    # R1 — embed/float verdicts compare against the LOCAL surface (the
    # center ray hit), not the max over the whole fan (which false-rejected
    # clean draws on uneven tabletops), and the P5 live side counts only
    # target hits as the live surface.
    assert 'embed_deficit = center["hit_z"] - bottom_z' in source
    assert "support_live_top_z" in source
    # Attempt 12: the stock OnTop sampler uses the base-aligned bbox and its
    # offset from the object base link. The returned candidate position is an
    # object-base pose, so subtracting half of the grasped world AABB from it
    # falsely marked valid bottle poses as 12 mm embedded in the tabletop.
    assert "held_obj.get_base_aligned_bbox()" in source
    assert "bbox_pos_in_base" in source
    assert "T.pose_transform(" in source
    assert 'held_extent_source = "world_aligned_bbox"' in source
    assert 'held_extent_source = "base_aligned_bbox"' in source
    assert source.index('held_extent_source = "world_aligned_bbox"') < source.index(
        'held_extent_source = "reference"'
    )
    # The custom hover waypoint is stricter than the stock final-pose IK gate.
    # Validate it both at the current base and at a sampled navigation stance;
    # otherwise a final-pose-valid candidate can fail only when Fix N inserts
    # the extra 0.25 m vertical waypoint.
    assert "def _place_hover_pose(" in source
    assert '"place_hover_not_reachable_from_current_base"' in source
    assert '"hover_reachable_and_collision_free"' in source
    assert '"place_hover_not_reachable"' in source
    assert 'hover_kwargs["low_precision"] = True' in source
    assert '"intermediate_low_precision": True' in source
    # The stock release returns as soon as the fingers open. A physically
    # released object gets a bounded sequence of recorded hold actions so the
    # official OnTop state can become stable before the stock postcondition.
    assert "def _execute_release(self):" in source
    assert '"place_release_settle"' in source
    assert "stable_steps >= 5" in source
    assert "released_obj.states[object_states.OnTop].get_value(place_target)" in source
    assert 0.0 < PLACE_RELEASE_CONTACT_CLEARANCE < 0.02
    assert "position[2] -= contact_lowering" in source
    assert "released_obj.states[object_states.Touching].get_value(place_target)" in source
    assert '"Held object did not physically converge above the placement target"' in source
    assert '"ready": ready_to_release' in source
    assert "xy_on_target and -0.005 <= vertical_gap <= 0.03" in source
    assert '"place_final_hand"' in source
    assert '"position_error_m"' in source
    # R3 — reference hits above the captured AABB top belong to objects
    # resting on the support and are discarded before the P5 comparison.
    assert "PLACE_SUPPORT_REFERENCE_TOP_EPSILON" in source
    assert PLACE_SUPPORT_REFERENCE_TOP_EPSILON <= 0.02
    # R4 — corner rays at +-0.45 of the extents supplement the interior
    # 4x4 grid so P3-corridor corner candidates stay covered by the P5
    # cross-check (0.139 m gap on the breakfast table > old 0.12 radius).
    assert "(-0.45, 0.45)" in source
    assert PLACE_SUPPORT_REFERENCE_XY_RADIUS >= 0.12


def test_visual_post_frame_revalidates_the_official_postcondition():
    source = (Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py").read_text(
        encoding="utf-8"
    )
    execute = source[source.index("def execute("):source.index("def main()")]
    capture = execute.index('f"step_{step.step_id:03d}_post"')
    stable_check = execute.index(
        "stable_post_ok, stable_postcondition = _check_postcondition"
    )
    assert capture < stable_check
    assert 'record["postcondition_after_capture"] = stable_postcondition' in execute
    assert "post_ok = post_ok and stable_post_ok" in execute


def test_symbolic_inventory_follows_navigation_without_claiming_physical_grasp():
    source = (Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py").read_text(
        encoding="utf-8"
    )
    oracle = source[
        source.index("class DeltaSGOraclePrimitives"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    assert "def _sync_inventory_to_eef" in oracle
    assert "self._sync_inventory_to_eef()" in oracle
    assert "link.disable_gravity()" in oracle
    assert "link.enable_gravity()" in oracle
    assert oracle.index("self._set_inventory_gravity(True)") < oracle.index(
        "changed = bool(state.set_value(obj, True))"
    )
    assert "DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 2" in oracle
    assert "DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 2" in oracle
    execute = source[source.index("def execute("):source.index("def main()")]
    assert '"type": "oracle_head_only_look_at"' in execute
    assert '"base_motion_commanded": False' in execute
    assert "controller._sync_inventory_to_eef()" not in execute


def test_symbolic_navigation_and_placement_stay_in_manipulation_range():
    source = (Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py").read_text(
        encoding="utf-8"
    )
    execute = source[source.index("def execute("):source.index("def main()")]

    assert "def _horizontal_target_aabb_distance(" in source
    assert "max_target_aabb_distance=DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE" in source
    assert '"stage": "manipulation_approach"' in execute
    assert 'postcondition["horizontal_target_distance"]' in execute
    assert 'postcondition["placed_object_horizontal_distance"]' in execute
    assert "approach_distances = {}" in source
    assert "distance < approach_distances[object_id]" in source
    oracle = source[
        source.index("class DeltaSGOraclePrimitives"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    assert "saved_xy = None" in oracle
    assert "for _ in range(4):" in oracle
    assert "placed_object_distance = _horizontal_target_aabb_distance(" in oracle
    assert "placed_object_distance > DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE" in oracle


def test_symbolic_replay_isolates_preloaded_objects_without_kinematic_task_objects():
    source = (Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py").read_text(
        encoding="utf-8"
    )

    assert "anchored_for_replay = task_support" in source
    assert '"fixed_base": anchored_for_replay' in source
    assert '"kinematic_only": anchored_for_replay' in source
    assert "def configure_preloaded_delta_objects(" in source
    assert "obj.visible = active" in source
    assert "link.disable_collisions()" in source
    assert "obj.disable_gravity()" in source


def test_expert_frames_are_high_resolution_and_disable_temporal_ghosting():
    source = (Path(__file__).resolve().parents[1] / "code" / "run_deltasg_expert.py").read_text(
        encoding="utf-8"
    )

    assert 'parser.add_argument("--view-width", type=int, default=640)' in source
    assert 'parser.add_argument("--view-height", type=int, default=480)' in source
    assert 'settings.set_int("/rtx/post/aa/op", 0)' in source
    assert 'settings.set_int("/rtx-defaults/post/aa/op", 0)' in source
    assert 'settings.set_bool("/omni/replicator/captureMotionBlur", False)' in source
    assert 'settings.set_bool("/rtx/post/motionblur/enabled", False)' in source
    assert 'settings.set_bool("/rtx/raytracing/enableAccumulation", False)' in source
    capture_globals = source[
        source.index("def _capture_globals("):source.index("def _capture_event_unprotected(")
    ]
    assert "cut_position[2] += 0.01" in capture_globals
    assert capture_globals.count("sensor.set_position_orientation(") >= 2
