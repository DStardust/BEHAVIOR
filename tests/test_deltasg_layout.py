from deltasg_layout import (
    FIRE_EXTINGUISHER_LAYOUT_POLICY,
    FIRE_EXTINGUISHER_MIN_TARGET_GAP,
    evaluate_tool_layout,
    horizontal_bbox_gap,
    rank_fitting_models,
    rigid_bbox_fits,
)


def test_laundry_floor_grasp_policy_is_scoped_and_has_a_lower_bound():
    from deltasg_expert import task_grasp_minimum, evaluate_manipulation_height
    minimum = task_grasp_minimum("collect_dirty_clothes", "sock", 0.10)
    assert minimum == 0.04
    assert evaluate_manipulation_height("GRASP", 0, 0.116, 0, minimum, 1.55)["eligible"]
    assert not evaluate_manipulation_height("GRASP", 0, 0.078, 0, minimum, 1.55)["eligible"]
    assert task_grasp_minimum("retrieve_object", "sock", 0.10) == 0.10
    assert task_grasp_minimum("collect_dirty_clothes", "hamper", 0.10) == 0.10
    assert task_grasp_minimum("wash_dirty_dishes", "bowl", 0.10) == 0.10


def test_rigid_clothing_requires_real_container_capacity():
    basket = [0.3888, 0.4122, 0.4312]
    assert not rigid_bbox_fits([0.7845, 0.8465, 0.3205], basket)
    assert not rigid_bbox_fits([1.0075, 0.6656, 0.2181], basket)
    assert rigid_bbox_fits([0.2828, 0.1890, 0.1165], basket)
    assert not rigid_bbox_fits([0.2828, 0.1890, 0.1165], [0.10, 0.10, 0.15])


def test_container_model_ranking_keeps_only_compact_fitting_models():
    models = {
        "wide": [0.28, 0.28, 0.04],
        "compact": [0.10, 0.12, 0.02],
        "small": [0.12, 0.14, 0.03],
        "too_tall": [0.08, 0.09, 0.14],
    }
    assert rank_fitting_models(models, [[0.32, 0.31, 0.13]], limit=2) == [
        "compact", "small"
    ]


def test_fire_storage_filter_precedes_floor_shortlisting():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "code/online_deltasg.py").read_text()
    block = source[source.index("def _build_floor_placement"):source.index("def _hypothetical_support_has_operation_approach")]
    assert block.index('separated_candidates=') < block.index('ranked_pixels[:96]')


def test_tool_distance_uses_edges_not_centres_or_height():
    target = ([0, 0, 0], [1, 1, 1])
    tool = ([1.2, 0, 5], [1.4, 0.2, 6])
    assert round(horizontal_bbox_gap(tool, target), 3) == 0.2
    assert not evaluate_tool_layout(tool, target, {"min_target_gap": 0.4})["ok"]


def test_cleaning_station_requires_both_target_separation_and_sink_proximity():
    target = ([0, 0, 0], [0.2, 0.2, 1])
    tool = ([0.7, 0, 0], [0.9, 0.2, 1])
    sink = ([1, 0, 0], [1.5, 1, 1])
    policy = {"min_target_gap": 0.4, "max_anchor_gap": 1.0}
    assert evaluate_tool_layout(tool, target, policy, sink)["ok"]
    assert not evaluate_tool_layout(tool, target, policy)["ok"]
    assert not evaluate_tool_layout(tool, target, policy, ([3, 0, 0], [4, 1, 1]))["ok"]


def test_extinguisher_is_not_accepted_next_to_fire():
    fire = ([0, 0, 0], [0.2, 0.2, 1])
    assert FIRE_EXTINGUISHER_LAYOUT_POLICY == "separate_fire_safety_tool_v2"
    assert FIRE_EXTINGUISHER_MIN_TARGET_GAP == 2.5
    policy = {"min_target_gap": FIRE_EXTINGUISHER_MIN_TARGET_GAP}
    assert not evaluate_tool_layout(([0.4, 0, 0], [0.6, 0.2, 1]), fire, policy)["ok"]
    assert evaluate_tool_layout(([3, 0, 0], [3.2, 0.2, 1]), fire, policy)["ok"]
