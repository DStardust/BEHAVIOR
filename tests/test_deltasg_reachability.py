"""Regression tests for generated task-object reachability evidence."""

import copy
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from audit_deltasg_outputs import check_run


def _run():
    return {
        "ok": True,
        "task_environment": {
            "env_type": "Env-A",
            "task": {
                "primary_behavior_task": "retrieve_book",
                "instruction": "Retrieve the book from the coffee table.",
            },
            "robot": {"pose": {"position": [0, 0, 0]}},
            "camera": [{"camera_id": "robot_primary"}],
            "solution_plan": [{"primitive": "MOVE"}],
            "added_objects": [
                {
                    "object_id": "online_env_a_0000_book",
                    "semantic_roles": ["task_object"],
                    "placement": {
                        "support_category": "coffee_table",
                        "robot_approach": {
                            "ok": True,
                            "horizontal_distance": 0.7,
                            "max_horizontal_distance": 1.0,
                        },
                    },
                }
            ],
            "validation": {
                "ok": True,
                "scene_integrity": {"ok": True},
                "settling": {"all_within_threshold": True},
                "camera_coverage": {"ok": True},
            },
        },
    }


def test_reachable_task_object_passes_audit():
    issues = check_run(Path("sample.json"), _run())
    assert not any("robot_approach" in issue for issue in issues)


def test_missing_task_object_approach_fails_audit():
    run = _run()
    del run["task_environment"]["added_objects"][0]["placement"]["robot_approach"]
    assert "envA_retrieval_missing_robot_approach" in check_run(Path("sample.json"), run)


def test_failed_or_distant_task_object_approach_fails_audit():
    failed = _run()
    failed["task_environment"]["added_objects"][0]["placement"]["robot_approach"]["ok"] = False
    assert "envA_retrieval_robot_approach_failed" in check_run(Path("sample.json"), failed)

    distant = copy.deepcopy(_run())
    approach = distant["task_environment"]["added_objects"][0]["placement"]["robot_approach"]
    approach["horizontal_distance"] = 1.1
    assert "envA_retrieval_robot_approach_too_far" in check_run(Path("sample.json"), distant)


def test_floor_retrieval_target_requires_height_gate_evidence():
    run = _run()
    obj = run["task_environment"]["added_objects"][0]
    obj["category"] = "paperback_book"
    obj["placement"]["support_category"] = "floor"
    assert "envA_retrieval_missing_floor_height_gate" in check_run(
        Path("sample.json"), run
    )

    obj["placement"]["manipulation_height"] = {
        "eligible": False,
        "relative_height": 0.03,
        "min_height": 0.10,
    }
    assert "envA_retrieval_unmanipulable_floor_target" in check_run(Path("sample.json"), run)

    obj["placement"]["manipulation_height"]["eligible"] = True
    assert not any("floor_target" in issue or "floor_height" in issue for issue in check_run(
        Path("sample.json"), run
    ))


def test_standardized_added_object_persists_floor_height_gate_evidence():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    block = source[
        source.index("def _standard_added_objects"):
        source.index("def _standard_validation")
    ]
    assert block.count('"manipulation_height": (item.get("placement") or {}).get(') >= 2


def test_generated_floor_support_does_not_confuse_source_instruction_audit():
    run = _run()
    run["task_environment"]["task"]["instruction"] = "Retrieve the book from the coffee table."
    run["task_environment"]["added_objects"].append({
        "object_id": "online_env_a_0000_support",
        "category": "nightstand",
        "semantic_roles": ["task_support"],
        "placement": {"support_category": "floor", "support_object_id": None},
    })
    assert "envA_retrieval_instruction_support_mismatch" not in check_run(
        Path("sample.json"), run
    )


def test_native_state_tasks_require_official_reversible_transition_preflight():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    block = source[
        source.index("def _prepare_native_task_initial_state"):
        source.index("def _native_target_manipulation_height")
    ]
    assert 'apply_and_observe("task_initial", required_initial)' in block
    assert 'apply_and_observe("task_final_preflight", desired_final)' in block
    assert 'apply_and_observe("restore_task_initial", required_initial)' in block
    assert 'setter_returned and immediate == requested and settled == requested' in block
    assert "official_state_transition_preflight" in block


def test_native_state_task_reselects_after_official_transition_failure():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    task_block = source[
        source.index('if task_category in {"open_close", "appliance"}'):
        source.index("# If no objects matched and no scene-native targets")
    ]
    assert "while True:" in task_block
    assert "self.reject_native_target(" in task_block
    assert "no_officially_transitionable_native_target" in task_block


def test_serial_native_tasks_choose_target_before_robot_spawn():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    block = source[
        source.index("def prepare_native_task_robot_spawn"):
        source.index("def reject_native_target")
    ]
    assert "require_robot_approach=False" in block
    assert "self._prepared_native_target_id" in block
    assert "def invalidate_robot_reachability" in block
    assert '"reason": "target_conditioned_spawn"' in block
    assert "candidate_position_xy" in block
    assert "def bind_prepared_native_task_spawn" in block
    assert "self._prepared_native_target = None" in block


def test_task_object_ontop_grid_prefers_real_robot_operation_side():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    placement = source[
        source.index('if semantic_role == "task_object":'):
        source.index("# Relation placement is comparatively expensive")
    ]
    relation = source[
        source.index("def _apply_relation"):
        source.index("def _validate_on_top_pose")
    ]
    assert 'candidate["prefer_robot_access"] = True' in placement
    assert 'if placement.get("prefer_robot_access")' in relation
    assert "target_aabb_xy=((sup_x_min, sup_y_min), (sup_x_max, sup_y_max))" in relation
    assert "grid_points.sort" in relation


def test_placement_loop_has_independent_task_floor_height_guard():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert '"error": "task_object_floor_height_out_of_range"' in source
    assert 'semantic_role == "task_object"' in source
    assert "_validate_floor_manipulation_height" in source
    assert 'target_placement_mode == "floor"' in source


def test_task_support_candidates_are_preflight_ranked_before_physics():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    rank_at = source.index("ranked_candidates = []")
    relation_at = source.index("relation_result = self._apply_relation", rank_at)
    assert rank_at < relation_at
    assert "_validate_task_approach_position" in source[rank_at:relation_at]
    assert "preflight_robot_approach" in source[rank_at:relation_at]


def test_advertised_retrieval_tasks_all_have_safe_asset_contracts():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert 'assert VALID_TASKS["retrieval_delivery"] == set(SAFE_RETRIEVAL_ASSETS)' in source
    assert 'PLANNED_RETRIEVAL_TASKS: set[str]' in source


def test_expert_batch_timeout_starts_after_single_gpu_acquisition():
    wrapper = (CODE_DIR / "run_omnigibson_single_gpu.sh").read_text(encoding="utf-8")
    batch = (CODE_DIR / "run_deltasg_expert_batch.sh").read_text(encoding="utf-8")
    assert 'CHILD_TIMEOUT="${DELTASG_CHILD_TIMEOUT:-0}"' in wrapper
    assert "[single-gpu] acquired" in wrapper
    assert wrapper.index("[single-gpu] acquired") < wrapper.index("timeout --signal=TERM")
    assert 'DELTASG_CHILD_TIMEOUT="$SAMPLE_TIMEOUT"' in batch
    assert 'if [[ "$failed" -gt 0 || "$audit_status" -ne 0 ]]' in batch


def test_single_gpu_external_process_polling_is_configurable():
    wrapper = (CODE_DIR / "run_omnigibson_single_gpu.sh").read_text(encoding="utf-8")
    assert 'POLL_INTERVAL="${DELTASG_GPU_POLL_INTERVAL:-5}"' in wrapper
    assert wrapper.count('sleep "$POLL_INTERVAL"') == 2
    assert "DELTASG_GPU_POLL_INTERVAL must be a positive integer" in wrapper
    assert 'ALLOW_EXTERNAL_GPU_PROCESSES="${DELTASG_ALLOW_EXTERNAL_GPU_PROCESSES:-0}"' in wrapper
    assert "shared mode gpu=" in wrapper


def test_floor_height_regression_has_real_positive_and_negative_jobs():
    source = (CODE_DIR / "run_enva_floor_height_regression.sh").read_text(encoding="utf-8")
    assert "retrieve_drink bottle_of_water cytqio" in source
    assert "retrieve_phone cell_phone dbhfuh" in source
    assert "--target-placement-mode floor" in source
    assert "task_object_floor_height_out_of_range" in source
    assert "python code/run_deltasg_expert.py" in source
    assert 'ROBOT="${ROBOT:-Tiago}"' in source
    assert source.count('--robot "$ROBOT"') == 2
    assert 'rm -f "$ROOT/report.json"' in source


def test_expert_batch_resumes_only_audited_artifacts():
    source = (CODE_DIR / "run_deltasg_expert_batch.sh").read_text(encoding="utf-8")
    assert '--root "$output" --min-accept-rate 1.0' in source


def test_retrieval_room_selection_requires_a_legal_task_support():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    start = source.index("def _choose_safe_target_room")
    end = source.index("def _match_required_objects", start)
    room_selector = source[start:end]
    assert "self._category_prefers_floor(category)" in room_selector
    assert "_validate_task_approach_position" in room_selector
    assert '{"floor", "floors"}' in room_selector
    assert "support_capacity = -min(len(support_nodes), 6)" in room_selector
    assert "best_capacity" in room_selector
    assert "return None" in room_selector
    assert '"no_reachable_compatible_support_room"' in source


def test_sparse_scene_bootstraps_support_before_task_object():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    bootstrap_at = source.index("generated_support_record = copy.deepcopy(")
    refresh_at = source.index("placement_graph = self.snapshot()", bootstrap_at)
    add_at = source.index("self.add_task_asset(", bootstrap_at)
    assert bootstrap_at < add_at < refresh_at
    assert 'semantic_role == "task_support"' in source
    assert "_choose_support_bootstrap_room" in source
    assert "preferred_support_id=(generated_support_id" in source
    assert "preferred_placement" in source
    assert "expected_children" in source
    assert "support_name == name" in source
    expert_source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    assert "stationary_delta_baseline" in expert_source
    assert "_combined_scene_integrity" in expert_source


def test_delivery_prefers_a_validated_native_destination_then_bootstraps_if_needed():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    selection_at = source.index("delivery_destination = None")
    spawn_at = source.index("self._spawn_delivery_destination_support(", selection_at)
    task_at = source.index("task_instance = self._build_task_instance", spawn_at)
    block = source[selection_at:task_at]
    assert selection_at < spawn_at < task_at
    assert 'semantic_role="task_support"' in source[spawn_at:]
    assert "_delivery_destination_from_node" in source[spawn_at:]
    assert "excluded_rooms = {source_room}" in source[spawn_at:]
    assert 'destination["generated_support"] = True' in source[spawn_at:]
    choose_at = block.index("self._choose_delivery_destination(")
    fallback_at = block.index("if delivery_destination is None:")
    spawn_in_block_at = block.index("self._spawn_delivery_destination_support(")
    assert choose_at < fallback_at < spawn_in_block_at
    assert 'validation["settling"] = self._collect_settling_report(created_names)' in block
    assert "after_graph = self.snapshot()" in block


def test_floor_overlap_gate_includes_the_robot_footprint():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    start = source.index("def _check_aabb_overlap")
    end = source.index("def _has_unexpected_contacts", start)
    overlap_gate = source[start:end]
    assert 'robots = list(getattr(self.env, "robots", None) or [])' in overlap_gate
    assert "collision_objects.extend(robots)" in overlap_gate
    assert "for other in collision_objects" in overlap_gate
    assert "if target_room and other not in robots" in overlap_gate


def test_generated_support_floor_candidates_cover_the_reachable_room():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert "spread_across_room=generated_support_fixture" in source
    assert "pool = ordered if spread_across_room else ordered[: min(25, len(ordered))]" in source


def test_generated_support_records_do_not_mutate_shared_asset_database_records():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert source.count('copy.deepcopy(self._record_for_category("breakfast_table"))') >= 1
    assert 'copy.deepcopy(self._record_for_category("coffee_table"))' in source
    assert "if record.get(\"_generated_support_fixture\")" in source
    assert "self._forget_generated_placement(result)" in source


def test_support_occupancy_failure_records_obstacle_geometry():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    relation = source[source.index("def _apply_relation"):source.index("def _state_by_name")]
    assert '"name": other_name' in relation
    assert '"obstacles": obstacles' in relation
    assert '"support_bounds": {' in relation


def test_expert_batch_and_physical_representatives_reuse_generation_robot():
    batch = (CODE_DIR / "run_deltasg_expert_batch.sh").read_text(encoding="utf-8")
    physical = (CODE_DIR / "run_enva_physical_representatives.sh").read_text(encoding="utf-8")
    assert 'sample_robot="$(jq -r' in batch
    assert '--robot "$sample_robot"' in batch
    assert "does not match generation robot" in batch
    assert 'tiago) sample_robot="Tiago"' in batch
    assert 'sample_robot="$(jq -r' in physical
    assert '--robot "$sample_robot"' in physical
    assert 'r1) sample_robot="R1"' in physical


def test_generated_support_models_are_not_reported_as_task_target_models():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    start = source.index("def sample_diversity_record")
    end = source.index("def load_existing_fingerprints", start)
    block = source[start:end]
    assert '"task_object" in set(item.get("semantic_roles") or [])' in block
    assert "for item in task_objects:" in block
    assert "set(item.get(\"room_id\") for item in task_objects" in block


def test_native_target_requires_matching_reachable_plan_object():
    run = _run()
    run["task_environment"]["task"] = {
        "primary_behavior_task": "turn_on_light",
        "instruction": "Turn on the light.",
        "plan_objects": [{
            "object_id": "switch_0",
            "category": "electric_switch",
            "reference_only": True,
            "semantic_role": "target",
            "robot_approach": {
                "ok": True,
                "horizontal_distance": 0.6,
                "max_horizontal_distance": 1.0,
            },
            "manipulation_height": {
                "eligible": True,
                "relative_height": 1.0,
            },
        }],
    }
    run["task_environment"]["solution_plan"] = [
        {"primitive": "MOVE", "target_object": "switch_0"},
        {"primitive": "INTERACT", "target_object": "switch_0"},
    ]
    run["task_environment"]["state_changed_objects"] = [{
        "ok": True,
        "object_id": "switch_0",
        "states": {"toggled_on": False},
        "expected_task_final_states": {"toggled_on": True},
        "semantic_roles": ["task_target", "task_initial_state"],
        "official_state_transition_preflight": [
            {
                "phase": "task_initial", "requested": False,
                "setter_returned": True, "immediate": False,
                "settled": False, "ok": True,
            },
            {
                "phase": "task_final_preflight", "requested": True,
                "setter_returned": True, "immediate": True,
                "settled": True, "ok": True,
            },
            {
                "phase": "restore_task_initial", "requested": False,
                "setter_returned": True, "immediate": False,
                "settled": False, "ok": True,
            },
        ],
    }]
    run["task_environment"]["added_objects"] = []
    assert not any("native_target" in issue for issue in check_run(Path("sample.json"), run))

    run["task_environment"]["task"]["plan_objects"][0]["object_id"] = "switch_1"
    assert "envA_native_target_identity_mismatch" in check_run(Path("sample.json"), run)


def test_native_target_rejects_missing_or_nontransitioning_initial_state():
    run = _run()
    run["task_environment"]["task"] = {
        "primary_behavior_task": "close_door",
        "instruction": "Close the door.",
        "plan_objects": [{
            "object_id": "door_0",
            "category": "door",
            "robot_approach": {"ok": True, "horizontal_distance": 0.5},
            "manipulation_height": {"eligible": True},
        }],
    }
    run["task_environment"]["solution_plan"] = [
        {"primitive": "INTERACT", "target_object": "door_0"},
    ]
    run["task_environment"]["added_objects"] = []
    assert "envA_native_target_initial_state_invalid" in check_run(Path("sample.json"), run)


def test_native_target_rejects_missing_official_transition_preflight():
    run = _run()
    run["task_environment"]["task"] = {
        "primary_behavior_task": "close_door",
        "instruction": "Close the door.",
        "plan_objects": [{
            "object_id": "door_0",
            "category": "door",
            "robot_approach": {"ok": True, "horizontal_distance": 0.5},
            "manipulation_height": {"eligible": True},
        }],
    }
    run["task_environment"]["solution_plan"] = [
        {"primitive": "INTERACT", "target_object": "door_0"},
    ]
    run["task_environment"]["state_changed_objects"] = [{
        "ok": True,
        "object_id": "door_0",
        "states": {"open": True},
        "expected_task_final_states": {"open": False},
        "semantic_roles": ["task_target", "task_initial_state"],
    }]
    run["task_environment"]["added_objects"] = []
    assert "envA_native_target_official_state_preflight_invalid" in check_run(
        Path("sample.json"), run
    )
    run["task_environment"]["state_changed_objects"] = [{
        "ok": True,
        "object_id": "door_0",
        "states": {"open": False},
        "expected_task_final_states": {"open": False},
        "semantic_roles": ["task_target", "task_initial_state"],
    }]
    assert "envA_native_target_initial_state_invalid" in check_run(Path("sample.json"), run)


def test_native_target_selection_preflights_reachability_and_preserves_identity():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    selector_start = source.index("def _select_native_task_target")
    selector_end = source.index("def _native_task_instruction", selector_start)
    selector = source[selector_start:selector_end]
    assert "_validate_task_approach_position" in selector
    assert "target_object_id=None" in selector
    assert '"robot_approach": robot_approach' in selector
    assert "native_target=native_target" in source
    assert 'native_target["object_id"]' in source
    assert "_native_target_manipulation_height" in selector


def test_formal_enva_plans_do_not_add_fuzzy_scene_reference_objects():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    builder = source[
        source.index("def _build_task_instance"):
        source.index("def _scene_objects_prefer_room")
    ]
    exclusion = '{\n            "retrieval_delivery", "open_close", "appliance",\n        }'
    assert builder.count(exclusion) >= 2
    assert "self._enforce_enva_solution_plan(" in builder


def test_camera_visibility_search_is_bounded_before_rendering():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    camera_records = source[source.index("def _camera_records"):source.index("def _global_camera_candidates")]
    camera_visibility = source[source.index("def _global_camera_visibility"):source.index("def _iter_instance_observations")]
    assert "max_camera_pose_attempts_per_room" in camera_records
    assert "raycast_closest" in camera_visibility
    assert 'sensor.add_modality("seg_instance")' not in camera_visibility


def test_generation_global_visibility_allows_large_clipped_fixtures():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    global_visibility = source[
        source.index("def _global_camera_visibility"):
        source.index("def _iter_instance_observations")
    ]
    assert '"bbox_clipped": bool(' in global_visibility
    assert 'visibility_source": "official_camera_frustum_physx_raycast"' in global_visibility
    robot_visibility = source[
        source.index("def _robot_camera_visibility"):
        source.index("def _global_camera_visibility")
    ]
    assert "self._geometric_camera_visibility" in robot_visibility
    assert "sensor.get_position_orientation()" in robot_visibility
    assert 'robot.set_position_orientation(' not in global_visibility
    assert 'sensor.set_position_orientation(' not in global_visibility
    assert "viewer.add_modality" not in global_visibility


def test_camera_candidate_budget_covers_room_geometry_before_angle_variants():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    camera_records = source[
        source.index("def _camera_records"):source.index("def _global_camera_candidates")
    ]
    candidates = source[
        source.index("def _global_camera_candidates"):
        source.index("def _compute_inward_wall_camera")
    ]
    first_wall = candidates.index('official_wall_{wall_name}_{angle}')
    first_corner = candidates.index('official_corner_{corner_name}_45')
    low_full_height = candidates.index('official_corner_{corner_name}_20')
    angle_variants = candidates.index('for angle in (30, 60)')
    assert low_full_height < first_corner
    assert first_corner < angle_variants
    assert first_wall > angle_variants
    runner = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    assert '"--max-global-cameras", "3", "--max-camera-pose-attempts", "6"' in runner
    assert 'if not item[2].endswith("_20")' in camera_records
    assert "-distance" in camera_records
    assert "camera_candidates = camera_candidates[:2]" not in camera_records
    assert "max_camera_pose_attempts_per_room" in camera_records


def test_degenerate_room_geometry_cannot_create_nan_corner_camera_poses():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    candidates = source[
        source.index("def _global_camera_candidates"):
        source.index("def _compute_inward_wall_camera")
    ]
    corner_camera = source[
        source.index("def _compute_corner_camera"):
        source.index("def _compute_wall_center_camera")
    ]
    assert "np.isfinite(room_diagonal) and room_diagonal >= 1e-6" in candidates
    assert 'raise ValueError("invalid room diagonal")' in corner_camera


def test_automatic_native_camera_failures_do_not_repeat_in_one_scene_process():
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    runner = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    assert "self._rejected_native_target_cache: set[str] = set()" in engine
    assert "node.get(\"id\") in self._rejected_native_target_cache" in engine
    assert "def reject_native_target" in engine
    assert 'interaction_target(run), "initial_camera_coverage"' in runner


def test_llm_fixture_names_normalize_hyphens_before_native_scene_matching():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    matcher = source[
        source.index("def _match_required_objects"):
        source.index("def _select_context_records_for_objects")
    ]
    assert 're.sub(r"[^a-z0-9]+", "_"' in matcher
    assert "scene_cats = {normalize(category)" in matcher
    assert 'name = normalize(obj.get("name", ""))' in matcher
    assert "exact_word_match = bool(name_words) and name_words <= sc_words" in matcher


def test_articulated_instance_links_are_merged_into_root_object_bboxes():
    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    assert 'target_id in path_parts' in expert
    assert "np.isin" in expert


def test_observation_pose_tries_all_framed_yaws_before_rejecting_position():
    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    observation_pose = expert[
        expert.index("def _connected_observation_pose"):
        expert.index("def _target_framing_distance")
    ]
    assert "for _, yaw in sorted(projected_yaws" in observation_pose
    assert "blockers = _navigation_blocker_aabbs(env, robot, obj)" in observation_pose
    assert "collision_geometry = _robot_collision_geometry(robot)" in observation_pose
    assert "blockers=blockers, collision_geometry=collision_geometry" in observation_pose
    assert "if candidate_yaw is None:" in observation_pose


def test_generation_reachability_matches_physical_corner_and_clearance_rules():
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    component = engine[
        engine.index("def _robot_component_pixels"):
        engine.index("def _robot_reachable_room_pixels")
    ]
    occupancy = engine[
        engine.index("def _collision_free_approach_candidate"):
        engine.index("def _validate_task_object_approach")
    ]
    assert "cv2.connectedComponents" not in component
    assert "not free[row + drow, col] or not free[row, col + dcol]" in component
    assert "lower - 0.10" in occupancy
    assert "upper + 0.10" in occupancy


def test_generated_support_surface_scan_excludes_the_robot():
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    placement = engine[
        engine.index("def _apply_relation"):
        engine.index("def _validate_on_top_pose")
    ]
    assert 'other in (getattr(self.env, "robots", None) or [])' in placement
    assert '"agent",' in placement


def test_native_support_displacement_is_rejected_and_rolled_back_during_placement():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    placement = source[
        source.index("def add_task_asset"):
        source.index("def _remove_object_safe_by_name")
    ]
    assert "support_pose_before = (" in placement
    assert 'not support_id.startswith("online_env_")' in placement
    assert "self._step(20)" in placement
    assert "support_displacement > 0.01" in placement
    assert '"native_support_displaced_by_placement"' in placement
    assert "object_position[2] += 2.0" in placement
    assert "position=support_pose_before[0]" in placement
    assert "orientation=support_pose_before[1]" in placement


def test_physics_rebuild_anchors_only_observed_unstable_native_fixtures():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert "self._stabilize_rebuilt_native_fixtures(rebuild_pose_restore)" in source
    helper = source[
        source.index("def _stabilize_rebuilt_native_fixtures"):
        source.index("def _tokens", source.index("def _stabilize_rebuilt_native_fixtures"))
    ]
    assert 'name.startswith("online_env_")' in helper
    assert "has_joints" in helper
    assert "tokens & STRUCTURAL_CATEGORIES" in helper
    assert "self._step(20)" in helper
    assert "displacement <= 0.01" in helper
    assert 'obj.root_link.set_attribute("physics:kinematicEnabled", True)' in helper
    assert "self._anchored_native_fixtures.add(obj.name)" in helper
    assert 'name.startswith("online_env_")' in helper
    assert "obj.set_position_orientation(position=position, orientation=orientation)" in helper

    restore = source[
        source.index("def _restore_physics_rebuild_poses"):
        source.index("def _stabilize_rebuilt_native_fixtures")
    ]
    assert 'not in self._anchored_native_fixtures' in restore


def test_retrieval_generation_uses_physical_grasp_height_and_sized_supports():
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert 'self._record_for_category("nightstand")' not in engine
    assert engine.count('self._record_for_category("breakfast_table")') >= 2
    assert 'self._record_for_category("coffee_table")' in engine
    assert 'self._compact_support_models("coffee_table")' in engine
    assert '"task_object_physical_grasp_height_out_of_range"' in engine
    assert "DEFAULT_MIN_PORTABLE_OBJECT_HEIGHT" in engine
    assert 'self.config.solvability_profile == "physical_control"' in engine
    assert 'else self.config.min_manipulation_height' in engine


def test_delivery_uses_generated_destination_only_as_validated_fallback():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    builder = source[
        source.index("delivery_destination = None"):
        source.index("task_instance = self._build_task_instance", source.index("delivery_destination = None"))
    ]
    assert "self._choose_delivery_destination(" in builder
    assert "if delivery_destination is None:" in builder
    assert "self._spawn_delivery_destination_support(" in builder
    assert builder.index("self._choose_delivery_destination(") < builder.index(
        "self._spawn_delivery_destination_support("
    )
    assert 'validation["settling"]["all_within_threshold"]' in builder
    support = source[
        source.index("def _spawn_delivery_destination_support"):
        source.index("def _choose_safe_target_room")
    ]
    assert 'self._compact_support_models("coffee_table")' in support
    assert "max_local_attempts = 6" in support
    assert "self._support_model_has_floor_pose(" in support
    assert '"no_footprint_clear_delivery_support_pose"' in support
    assert "self._approach_pose_clears_object(" in support
    assert 'avoid_position=source_pose.get("position")' in support
    floor_start = source.index("def _build_floor_placement")
    floor = source[floor_start:source.index("\n    def ", floor_start + 10)]
    assert "if avoid_position is not None:" in floor
    assert "min_separation_pixels" in floor
    assert "ranked_pixels[:96]" in floor
    assert "self._hypothetical_support_has_operation_approach(" in floor


def test_symbolic_coverage_propagates_generation_solvability_profile():
    runner = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    cli = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert '"--solvability-profile", args.expert_backend' in runner
    assert 'choices=("oracle_symbolic", "physical_control")' in cli
    assert "DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE" in cli
    assert '"solvability_profile": self.config.solvability_profile' in engine

    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = expert[expert.index("def execute("):expert.index("def main()")]
    assert "generation_profile != args.backend" in execute
    assert '"generation_profile_verified": generation_profile_verified' in execute
    assert "and generation_profile_verified" in execute
    assert '"complete_action_trace": complete_action_trace' in execute


def test_generated_support_requires_an_external_operation_approach():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    floor_loop = source[
        source.index('if semantic_role == "task_support":'):
        source.index('if semantic_role == "task_object" and not reachability["ok"]:')
    ]
    assert "target_aabb_xy=target_aabb_xy" in floor_loop
    assert "target_object_id=None" in floor_loop
    assert '"task_support_no_collision_free_operation_approach"' in floor_loop

    destination = source[
        source.index("def _delivery_destination_from_node"):
        source.index("def _delivery_top_surface_feasible")
    ]
    assert "fallback_room=None" in destination
    assert "target_object_id=None" in destination
    assert "[delivery-destination] reject" in destination

    spawn = source[
        source.index("def _spawn_delivery_destination_support"):
        source.index("def _forget_generated_placement")
    ]
    assert 'self.env.scene.object_registry(' in spawn
    assert "fallback_room=target_room" in spawn


def test_physical_manipulation_uses_official_eef_reachable_base_pose():
    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = expert[
        expert.index("class DeltaSGPhysicalPrimitives"):
        expert.index("def _jsonable")
    ]
    assert "self._sample_pose_near_object(" in physical
    assert 'pose_source = "official_eef_reachable_pose"' in physical
    assert "MAX_PHYSICAL_MANIPULATION_STANDOFF" not in physical


def test_oracle_visibility_recovery_moves_only_the_head_after_navigation():
    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = expert[expert.index("def execute("):expert.index("def main()")]
    assert '"type": "oracle_head_only_look_at"' in execute
    assert '"base_motion_commanded": False' in execute
    assert 'f"step_{step.step_id:03d}_pre_head_aim"' in execute
    assert "oracle_aim = _rotate_toward(" not in execute


def test_coverage_controls_bind_exact_spawned_and_native_targets():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    safe_records = source[source.index("def _safe_retrieval_records"):source.index("def _build_retrieval_instruction")]
    native_selector = source[source.index("def _select_native_task_target"):source.index("def _native_task_instruction")]
    model_selector = source[source.index("def _choose_target_model"):source.index("def _is_valid_support_node")]
    assert "target_asset_category" in safe_records
    assert "target_native_object_id" in native_selector
    assert "target_asset_model" in model_selector
    assert "category == self.config.target_asset_category" in model_selector


def test_exact_native_coverage_uses_target_conditioned_stable_robot_spawn():
    runner = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    engine = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    api = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    assert "preferred_target_name=args.target_native_object_id" in runner
    assert "preferred_max_distance=1.0" in api
    assert "effective_max_distance = preferred_max_distance + target_xy_radius" in api
    assert "expert_base_clearance_margin: float = 0.20" in engine
    assert 'scene_model == "Benevolence_2_int"' in api
    assert "max(90, warmup_steps * 3)" in api
    assert "max(60, warmup_steps * 2)" in api
    assert "get_room_instance_by_point" in api
    assert "native_displacement_limit=0.05" in api
    target_spawn = api[
        api.index("if preferred_target_name:"):api.index("for _ in range(max_attempts * 3):")
    ]
    assert "_erode_trav_map" in target_spawn
    assert "cv2.erode" not in target_spawn
    assert "range(max_attempts * 3)" in api
    assert "official-map candidates=" in api
    assert "semantic-room candidates=" in api
    assert "semantic_room_rank(candidate)" in api
    assert "room={spawn_room}" in api
    assert "native_to_restore = {name for name, _ in moved_native}" in api
    assert "class RobotSpawnError(RuntimeError)" in api
    assert "zero physical ground hits; installing official invisible" in api
    assert '"physical_ground_query_unavailable"' in api
    assert '"physics_rebuild_attempted": True' in api
    assert "saved_sim_state = og.sim.dump_state(serialized=False)" in api
    assert "og.sim.add_ground_plane(floor_plane_visible=False)" in api
    assert '"official_ground_plane_added": official_ground_plane_added' in api
    assert "og.sim.load_state(saved_sim_state, serialized=False)" in api
    assert "og.sim.stop()" in api
    assert "og.sim.play()" in api
    assert "robot_ground_offset" in api
    assert "hit = raytest(" in api
    assert "ignore_bodies=robot_body_paths" in api
    assert 'category not in {"floor", "floors", "carpet", "rug"}' in api
    assert "normal[2] >= 0.7" in api
    assert "ground_gap" in api
    assert "-0.05 <= ground_gap <= 0.15" in api
    assert '"no_physically_stable_robot_spawn"' in api
    assert "for name, (position, orientation, _) in native_baseline.items()" not in api
    preferred_block = api[
        api.index("if preferred_target_name:"):api.index("for _ in range(max_attempts * 3):")
    ]
    assert "falling back to ordinary stable spawn" in preferred_block
    assert "raise ValueError" not in preferred_block
    assert "raise RuntimeError" not in preferred_block


def test_coverage_stops_native_candidate_rotation_on_scene_spawn_failure():
    runner = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    online_runner = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    assert 'error["reason"] = exc.reason' in online_runner
    assert 'error["detail"] = exc.detail' in online_runner
    assert 'error.get("error_type") != "RobotSpawnError"' in runner
    assert 'status = "scene_initialization_failed"' in runner
    failure_check = runner.index(
        "scene_initialization_failure = _scene_initialization_failure(generation_dir)"
    )
    attempt_loop = runner.index(
        "for attempt, scheduled_native_object_id in enumerate(attempt_schedule, 1):"
    )
    assert failure_check > attempt_loop


def test_delivery_binds_exact_destination_and_preflights_operation_point():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    selector = source[
        source.index("def _choose_delivery_destination"):
        source.index("def _select_native_task_target")
    ]
    builder = source[
        source.index("def _fallback_enva_solution_plan"):
        source.index("def _scene_objects_prefer_room")
    ]
    assert '"PLACE_ON_TOP"' in selector
    assert "target_aabb_xy=(aabb_min[:2], aabb_max[:2])" in selector
    assert '"placement_mode": "on_top"' in selector
    assert 'delivery_destination["object_id"]' in builder
    assert '"semantic_role": "delivery_destination"' in builder
    assert "validate_env_a_plan_contract" in builder


def test_delivery_approach_distance_can_use_furniture_aabb_edge():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    validator = source[
        source.index("def _validate_task_approach_position"):
        source.index("def _validate_task_object_approach")
    ]
    assert "target_aabb_xy=None" in validator
    assert 'distance_reference = "aabb_edge"' in validator


def test_native_task_approach_uses_object_aabb_edge():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    validator = source[
        source.index("def _validate_task_object_approach"):
        source.index("def _floor_height_for_position")
    ]
    assert "self._safe_aabb(obj)" in validator
    assert "target_aabb_xy=target_aabb_xy" in validator
    selector = source[
        source.index("def _select_native_task_target"):
        source.index("def _fallback_enva_solution_plan")
    ]
    assert 'bbox = node.get("bbox") or {}' in selector
    assert "target_aabb_xy=target_aabb_xy" in selector
    assert "max_horizontal_distance=1.35" in selector


def test_generation_approach_uses_explicit_selected_room():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert "_validate_task_object_approach(obj, target_room=target_room)" in source
    assert "def _validate_task_object_approach(self, obj, target_room=None)" in source


def test_physical_expert_reaims_after_navigation_visibility_failure():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    look_at = source[
        source.index("def _physical_look_at"):
        source.index("def _check_postcondition")
    ]
    assert "target_position[1] - robot_position[1]" in look_at
    assert "controller._rotate_base_to_yaw(yaw)" in look_at
    assert "def _rotate_base_to_yaw(self, target_yaw)" in source
    assert "self.robot.q_to_action" not in look_at
    rotate = source[
        source.index("def _rotate_base_to_yaw"):
        source.index("@staticmethod\n    def _quat_to_yaw")
    ]
    assert "self._execute_motion_plan(joint_goal.unsqueeze(0), ignore_failure=True)" in rotate
    assert "position_drift > 0.08 or yaw_error > 0.35" in rotate
    assert '"post_navigation_target_not_visible"' in source
    assert 'plan.steps[step_index - 1].primitive == "NAVIGATE_TO"' in source
    assert "recovered[\"robot_visible\"]" in source
    assert "recovered[\"robot_primary\"][\"bboxes\"]" in source
    assert "nearest_target_xy" in source
    assert '"distance_reference": "aabb_edge"' in source
    assert "DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE" in source
    assert 'pose_source = "official_eef_reachable_pose"' in source
    assert "MAX_PHYSICAL_MANIPULATION_STANDOFF" not in source


def test_expert_requires_same_robot_as_generation():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    assert 'help="Expert robot. Defaults to the robot recorded by generation."' in source
    assert "if args.robot is None:" in source
    assert '"tiago": "Tiago"' in source
    assert 'generation_robot = str((run.get("robot") or {}).get("model") or "")' in source
    assert "generation_robot.casefold() != str(args.robot).casefold()" in source
    assert "does not match generation robot" in source
    oracle_env = source[
        source.index("def _create_expert_env"):
        source.index("def _initialize_segmentation_streams")
    ]
    assert "robot_model=str(robot_model).lower()" in oracle_env


def test_visualizer_normalizes_same_robot_name_for_omnigibson_registry():
    source = (CODE_DIR / "visualize_deltasg_batch.py").read_text(encoding="utf-8")
    assert 'robot_name = str(args.robot).lower()' in source
    assert 'else robot_name' in source


def test_post_navigation_oracle_visibility_uses_head_only_recovery():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    assert (
        "if step.primitive in MANIPULATION_PRIMITIVES and visibility_errors "
        "and target is not None:"
    ) in source
    assert '"post_navigation_target_not_visible"' in source
    execute = source[source.index("def execute("):source.index("def main()")]
    assert '"type": "oracle_head_only_look_at"' in execute
    assert '"base_motion_commanded": False' in execute
    assert 'f"step_{step.step_id:03d}_pre_head_aim"' in execute
    assert 'record["post_visibility_errors"]' in execute
    assert '"stage": "post_visibility"' in execute


def test_exact_native_target_spawn_precedes_largest_component_fallback():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    spawn = source[
        source.index("def stabilize_robot_spawn"):
        source.index("def run_camera_capture_on_env")
    ]
    assert "spawn_candidates.append((position, yaw, 0))" in spawn
    assert "spawn_candidates.append((position.clone(), yaw, 1))" in spawn
    sort_block = spawn[spawn.index("spawn_candidates.sort("):]
    assert sort_block.index("candidate[2]") < sort_block.index("semantic_room_rank(candidate)")
    assert sort_block.index("semantic_room_rank(candidate)") < sort_block.index("largest_component_rank(candidate)")


def test_exact_native_spawn_keeps_farther_same_room_collision_fallbacks():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    spawn = source[
        source.index("if preferred_target_name:"):
        source.index("for _ in range(max_attempts * 3):")
    ]
    assert "len(spawn_candidates) >= max_attempts" in spawn
    assert "if distance > effective_max_distance" in spawn
    assert "< 0.25" in spawn


def test_native_pose_restore_does_not_overwrite_spawn_candidate_position():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    spawn = source[
        source.index("def stabilize_robot_spawn"):
        source.index("def run_camera_capture_on_env")
    ]
    restore = spawn[
        spawn.index("if native_to_restore:"):
        spawn.index("grounded, raw_ground_hits = grounded_candidate(position)")
    ]
    assert "native_position, native_orientation, _ = native_baseline[name]" in restore
    assert "position=native_position" in restore
    assert "orientation=native_orientation" in restore


def test_robot_spawn_ground_raycast_ignores_robot_links():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    spawn = source[
        source.index("def stabilize_robot_spawn"):
        source.index("def run_camera_capture_on_env")
    ]
    assert "robot_body_paths =" in spawn
    assert "hit = raytest(" in spawn
    assert "ignore_bodies=robot_body_paths" in spawn

def test_retrieval_bootstraps_support_after_native_surface_failure():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    generation = source[
        source.index("def generate_env_a"):
        source.index("def _build_llm_rejected_result")
    ]
    assert 'not add_result["ok"]' in generation
    assert 'task_category == "retrieval_delivery"' in generation
    assert 'self._record_for_category("breakfast_table")' in generation
    assert 'preferred_support_id=generated_support_id' in generation
    assert "ignore_floor_coverings=generated_support_fixture" in source
    assert 'other_tokens & {"carpet", "rug"}' in source
    relation = source[source.index("def _apply_relation"):source.index("def _state_by_name")]
    chooser = source[
        source.index("def _choose_support_node"):
        source.index("def _build_placement_for_support")
    ]
    assert 'support_tokens & {"carpet", "rug", "mat", "doormat"}' in chooser
    assert '"reason": "structural_surface_requires_floor_mode"' in relation
    assert '"floor", "floors", "carpet", "rug",' in relation
    assert '"wall", "walls", "ceiling", "ceilings",' in relation
    assert "center_offset = obj_center - obj_position" in relation
    assert "sup_z_top + obj_h / 2.0 + 0.01" in relation
    assert "target_position = target_center - center_offset" in relation
    assert "official_on_top_resample" in relation
    assert "reset_before_sampling=True" in relation


def test_floor_grounding_accounts_for_asset_origin_offset():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    grounding = source[
        source.index("def _ground_object_on_floor"):
        source.index("def _validate_floor_manipulation_height")
    ]
    assert "live_lo, live_hi = obj.aabb" in grounding
    assert "aabb_center" in grounding
    assert "center_offset = aabb_center - current_position" in grounding
    assert "position = self._to_list(target_center - center_offset)" in grounding


def test_floor_pose_sampling_filters_the_live_object_footprint_before_spawn():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    builder = source[
        source.index("def _build_floor_placement"):
        source.index("def _apply_relation")
    ]
    assert "placement_obj=None" in builder
    assert "placement_obj.aabb" in builder
    assert "center_offset = (live_lo + live_hi) * 0.5 - live_position" in builder
    assert "half_extent = (live_hi - live_lo)[:2] * 0.5 + 0.02" in builder
    assert "footprint_clear_pixels" in builder
    assert "pixels = ranked_pixels" in builder
    assert "conservative footprint preflight" in builder
    assert "using {len(pixels)} reachable candidates" in builder
    assert "elif require_footprint_clear:" in builder
    assert "return None" in builder
    floor_call = source[
        source.index("floor_candidates = []"):
        source.index("\n            candidates = []", source.index("floor_candidates = []"))
    ]
    assert "placement_obj=obj" in floor_call
    assert "require_footprint_clear=generated_support_fixture" in floor_call


def test_bootstrap_room_allows_sparse_candidates_before_live_fixture_gates():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    selector = source[
        source.index("def _choose_support_bootstrap_room"):
        source.index("def _match_required_objects")
    ]
    assert "if pixel_count < 8:" in selector
    assert "if pixel_count < 25:" not in selector


def test_expert_uses_one_instance_stream_and_fixed_official_global_rgb_views():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    initializer = source[
        source.index("def _initialize_segmentation_streams"):
        source.index("def _physical_added_object_configs")
    ]
    assert "VisionSensor(" in initializer
    assert 'modalities=["rgb"]' in initializer
    assert 'relative_prim_path=f"/deltasg_global_cameras/camera_{index}_{safe_id}"' in initializer
    assert initializer.index("global_sensor.set_position_orientation") < initializer.index(
        "global_sensor.initialize()"
    )
    assert initializer.count('add_modality("seg_instance")') == 1
    assert 'global_sensor.add_modality("seg_instance")' not in initializer
    assert 'add_modality("seg_semantic")' not in initializer
    assert "og.sim.render()" in initializer
    capture_globals = source[
        source.index("def _capture_globals"):
        source.index("def _capture_event")
    ]
    assert "position, orientation = sensor.get_position_orientation()" in capture_globals
    assert "sensor.set_position_orientation(position=position, orientation=orientation)" in capture_globals
    assert "fixed_official_global_rgb_sensor" in capture_globals
    assert "generation_frustum_physx_raycast" in capture_globals
    assert 'camera.get("visibility")' in capture_globals
    assert 'frame="parent"' in source
    assert "derived_from_official_instance_segmentation" in source


def test_expert_replays_generated_support_furniture_as_kinematic():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    configs = source[
        source.index("def _physical_added_object_configs"):
        source.index("def _spawn_added_objects")
    ]
    spawn = source[
        source.index("def _spawn_added_objects"):
        source.index("def _apply_saved_initial_states")
    ]
    expected = '"task_support" in set(record.get("semantic_roles") or [])'
    assert expected in configs
    assert expected in spawn
    assert '"kinematic_only":' in configs
    assert "kinematic_only=" in spawn
    assert '"fixed_base": anchored_for_replay' in configs
    assert "fixed_base=task_support" in spawn


def test_replay_sink_gate_uses_generation_final_geometry_not_unsynced_reset_aabb():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    gate = source[source.index("def _delta_replay_integrity"):source.index("def _scene_objects")]
    assert 'generation_aabb_top = manipulation_height.get("aabb_max_z")' in gate
    assert "if generation_aabb_top is not None" in gate
    assert "replayed_aabb_top < comparison_aabb_top - 0.015" in gate
    assert '"generation_aabb_top": generation_aabb_top' in gate


def test_delivery_support_preserves_source_manipulation_pose():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    helper = source[
        source.index("def _approach_pose_clears_object"):
        source.index("def _spawn_delivery_destination_support")
    ]
    assert "robot.collision_points_world" in helper
    assert "local_points @ rotation.T" in helper
    assert "blocker.aabb" in helper
    spawn = source[
        source.index("def _spawn_delivery_destination_support"):
        source.index("def _forget_generated_placement")
    ]
    assert "source_item" in spawn
    assert "self._approach_pose_clears_object(" in spawn
    assert "occupies source manipulation pose" in spawn
    assert "max_local_attempts = 6" in spawn
    assert 'compact_models = self._compact_support_models("coffee_table")' in spawn
    assert "attempt_candidates = [" in spawn
    assert "for model in compact_models" in spawn
    assert "self._support_model_has_floor_pose(" in spawn
    assert 'self._record_for_category("coffee_table")' in spawn
    assert 'record["_preferred_models"]' in spawn


def test_generated_delivery_support_models_are_compact_and_height_eligible():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    helper = source[
        source.index("def _compact_support_models"):
        source.index("def _forget_generated_placement")
    ]
    assert 'get_dataset_path("behavior-1k-assets")' in helper
    assert 'json.loads(metadata_path.read_text(encoding="utf-8"))["bbox_size"]' in helper
    assert "min(width, depth) < 0.30 or max(width, depth) > 0.75" in helper
    assert "self.config.min_manipulation_height" in helper
    assert "self.config.max_manipulation_height" in helper
    assert "sorted(ranked)[:12]" in helper

    generator = source[
        source.index("def add_task_asset"):
        source.index("def _remove_object_safe_by_name")
    ]
    assert 'preferred_models=record.get("_preferred_models")' in generator


def test_delivery_destination_requires_primary_camera_operation_visibility():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    destination = source[
        source.index("def _delivery_destination_from_node"):
        source.index("def _delivery_top_surface_feasible")
    ]
    assert "self._primary_camera_operation_visible(node, robot_approach)" in destination
    helper = source[
        source.index("def _primary_camera_operation_visible"):
        source.index("def _delivery_top_surface_feasible")
    ]
    assert "relative_camera = th.linalg.inv(robot_matrix) @ camera_matrix" in helper
    assert "candidate_camera_orientation = self._look_at_quat(" in helper
    assert "operation_point" in helper
    assert "0.5 * (float(lower[0]) + float(upper[0]))" in helper
    assert "float(np.linalg.norm(size[:2])) >= 1.0" in helper
    assert "float(lower[2]) + 0.05 * float(size[2])" in helper
    assert "self._geometric_camera_visibility(" in helper
    assert "return x1 >= margin and y1 >= margin and x2 < width - margin" in helper


def test_delivery_destination_reserves_the_expert_framing_distance():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    destination = source[
        source.index("def _delivery_destination_from_node"):
        source.index("def _primary_camera_operation_visible")
    ]
    assert "xy_diagonal = math.hypot(" in destination
    assert "1.75," in destination
    assert "math.tan(math.radians(25.0))" in destination
    assert "preferred_center_distance=preferred_center_distance" in destination

    approach = source[
        source.index("def _validate_task_approach_position"):
        source.index("def _collision_free_approach_candidate")
    ]
    assert "preferred_center_distance=None" in approach
    assert "th.abs(distances - preferred_pixels)" in approach
    assert "eligible_distance_mask = edge_distances <= threshold" in approach
    assert "candidate_pixels = candidate_pixels[eligible_distance_mask]" in approach
    assert approach.index("eligible_distance_mask =") < approach.index(
        "candidate_xy, occupancy_rejections ="
    )
    assert "candidate_order" in approach


def test_camera_look_at_basis_is_a_proper_rotation_not_a_reflection():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    helper = source[
        source.index("def _look_at_quat"):
        source.index("def _nearest_room")
    ]
    assert "right = np.cross(d, up)" in helper
    assert "up = np.cross(-d, right)" in helper
    assert "right = np.cross(up, d)" not in helper
    assert "R = np.column_stack([right, up, -d])" in helper


def test_on_top_pose_requires_a_stable_inset_from_support_edges():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    validator = source[
        source.index("def _validate_on_top_pose"):
        source.index("def _collect_settling_report")
    ]
    assert "max(0.03, min(0.08, 0.5 * min(obj_w, obj_d)))" in validator
    assert "ox_min < sx_min + clearance" in validator
    assert "ox_max > sx_max - clearance" in validator


def test_on_top_sampler_uses_same_clearance_and_access_aware_order_as_validator():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    relation = source[source.index("def _apply_relation"):source.index("def _validate_on_top_pose")]
    clearance = "max(0.03, min(0.08, 0.5 * min(obj_w, obj_d)))"
    assert clearance in relation
    assert "grid_indices = sorted(" in relation
    assert "for ix in grid_indices" in relation
    assert "for iy in grid_indices" in relation
    assert 'if placement.get("prefer_robot_access")' in relation
    assert "grid_points.sort" in relation


def test_every_delivery_keeps_all_future_objects_on_fixed_global_views():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    cameras = source[source.index("def _camera_records"):source.index("def _global_camera_candidates")]
    assert 'if primary_task.startswith("deliver_")' in cameras
    assert 'len(target_room_set) > 1' not in cameras
    assert "uncovered = (target_names - set(robot_visible)) | persistent_global_targets" in cameras
    assert "set(global_visible) >= persistent_global_targets" in cameras
    assert '"persistent_global_targets": sorted(persistent_global_targets)' in cameras


def test_delivery_global_camera_keeps_official_positions_and_aims_at_task_objects():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    cameras = source[source.index("def _camera_records"):source.index("def _global_camera_candidates")]
    assert 'if primary_task.startswith("deliver_") and room_targets:' in cameras
    assert "task_focus = np.mean(focus_points, axis=0)" in cameras
    assert "self._look_at_quat(position, task_focus)" in cameras
    assert 'f"{method}_task_aim"' in cameras


def test_expert_robot_teleports_do_not_restart_scene_physics():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    helper = source[
        source.index("def _teleport_robot_preserving_delta_objects"):
        source.index("def _native_pose_snapshot")
    ]
    assert "og.sim.stop()" not in helper
    assert "robot.set_position_orientation" in helper


def test_expert_frames_large_manipulation_supports_from_aabb_size():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    framing = source[
        source.index("def _target_framing_distance"):
        source.index("def _floor_height_below")
    ]
    assert "xy_diagonal" in framing
    assert "math.tan(math.radians(25.0))" in framing
    assert "preferred_distance=_target_framing_distance(obj)" in source
    assert "th.linalg.norm((upper - lower)[:2])" in source
    assert "math.radians(15.0)" in source
    assert "_quat_multiply_xyzw(camera_orientation, local_pitch)" in source


def test_oracle_visibility_recovery_does_not_move_the_base():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = source[source.index("def execute("):source.index("def main()")]
    assert "_rotate_toward(" not in execute
    assert "_restore_visible_observation_pose(" not in execute
    assert '"type": "oracle_head_only_look_at"' in execute
    assert '"base_motion_commanded": False' in execute
    capture = source[source.index("def _capture_event"):source.index("def _set_robot_pose")]
    assert '"camera_joint_positions": _jsonable(camera_joint_positions)' in capture


def test_expert_observation_pose_rejects_occupied_centres_without_blanket_aabb_clearance():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    navigation = source[
        source.index("def _connected_observation_pose"):
        source.index("def _target_framing_distance")
    ]
    assert "_erode_trav_map" in navigation
    assert "line_of_sight" in navigation
    assert "_native_occupant_at_pose" in navigation
    assert 'rejected["native_occupancy"] += 1' in navigation
    assert "obstacle_bounds" not in navigation
    assert "robot_clearance = 0.45" not in navigation


def test_expert_native_occupancy_filter_ignores_structure_and_target():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    occupancy = source[
        source.index("def _navigation_blocker_aabbs"):
        source.index("def _connected_observation_pose")
    ]
    assert "scene_object is target" in occupancy
    assert "NON_BLOCKING_NAVIGATION_CATEGORIES" in occupancy
    assert "candidate_points >= lower - margin" in occupancy
    assert "candidate_points <= upper + margin" in occupancy


def test_expert_projects_live_robot_collision_boundary_to_candidate_pose():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    projection = source[
        source.index("def _robot_collision_geometry"):
        source.index("def _native_occupant_at_pose")
    ]
    assert "robot.collision_points_world" in projection
    assert "T.quat2mat(robot_orientation)" in projection
    assert "local_points @ candidate_rotation.T" in projection


def test_generation_approach_uses_robot_eroded_map_without_duplicate_aabb_filter():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    validator = source[
        source.index("def _validate_task_approach_position"):
        source.index("def _validate_task_object_approach")
    ]
    assert "_robot_component_pixels" in validator
    assert "room_pixels = self._robot_reachable_room_pixels(floor)" in validator
    assert "component_rooms" in validator
    assert "_collision_free_approach_candidate" in validator
    collision_filter = source[
        source.index("def _collision_free_approach_candidate"):
        source.index("def _validate_task_object_approach")
    ]
    assert "robot.collision_points_world" in collision_filter
    assert "NON_BLOCKING_NAVIGATION_CATEGORIES" in collision_filter
    assert '"no_collision_free_approach"' in validator
    assert "obstacle_bounds" not in validator
    assert "robot_clearance = 0.45" not in validator
    assert '"no_same_room_approach"' in validator
    room_selector = source[
        source.index("def _choose_safe_target_room"):
        source.index("def _choose_support_bootstrap_room")
    ]
    assert "if not reachable_rooms:" in room_selector
    assert "return None" in room_selector


def test_expert_integrity_prefers_rendered_geometry_center_over_entity_root():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    integrity = source[
        source.index("def _integrity_pose_record"):
        source.index("def _combined_scene_integrity")
    ]
    assert 'state.__class__.__name__ == "AABB"' in integrity
    assert 'link.get_position_orientation()[0]' in integrity
    assert 'obj.root_link.get_position_orientation()[0]' in integrity
    assert '"fixed_base": bool(getattr(obj, "fixed_base", False))' in integrity
    assert 'start.get("fixed_base")' in integrity
    assert 'end.get("fixed_base")' in integrity
    assert 'not name.startswith("online_env_")' in integrity
    assert 'end["root_link_position"] - start_root' in integrity
    assert 'common_links = set(start_links or {})' in integrity
    assert 'end["link_positions"][name] - start_links[name]' in integrity
    assert 'end["geometry_center"] - start_center' in integrity


def test_oracle_expert_uses_short_settle_without_restarting_physics():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    settle = source[
        source.index("def _settle_robot(self):"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    assert "for _ in range(5):" in settle
    assert "og.sim.stop()" not in settle
    pose_helper = source[source.index("def _set_robot_pose"):source.index("def _create_expert_env")]
    assert "for _ in range(30):" in pose_helper


def test_physical_representative_gate_covers_every_enva_primitive_family():
    source = (CODE_DIR / "run_enva_physical_representatives.sh").read_text(
        encoding="utf-8"
    )
    assert "for selector in retrieve deliver open_close appliance" in source
    assert "--backend physical_control" in source
    assert 'sample_robot="$(jq -r' in source
    assert '--robot "$sample_robot"' in source
    assert 'generation robot must be R1 or Tiago' in source
    assert "--llm-model \"$MODEL\"" in source
    assert ".total == 4" in source
    assert ".accepted == 4" in source
    assert ".completed == .total" in source
    assert ".runtime_ineligible_rate <= .max_runtime_ineligible_rate" in source
    assert ".expert_audit_process_code == 0" in source
    assert "declare -A REPRESENTATIVE_INPUTS" in source
    assert source.index("declare -A REPRESENTATIVE_INPUTS") < source.index("run_one()")


def test_generation_record_persists_llm_model_and_exact_plan_policy():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert '"llm_enabled": self._llm_client is not None' in source
    assert '"llm_model": self.config.llm_model' in source
    assert '"solution_plan_policy": "llm_with_exact_env_a_contract_fallback"' in source


def test_oracle_expert_preloads_added_objects_before_physics_views():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute_block = source[source.index("def execute("):source.index("def main()")]
    # The one-shot path still preloads objects. Only an explicitly persistent
    # clean base scene may replay them dynamically between hard resets.
    assert "_physical_added_object_configs(run, backend=args.backend)" in execute_block
    assert '_physical_added_object_configs(run) if args.backend' not in execute_block
    persistent_branch = execute_block[
        execute_block.index("if env is None:"):
        execute_block.index('_sink_diag_trace(env, "post_load")')
    ]
    assert "else:" in persistent_branch
    assert "if not persistent:" in persistent_branch
    assert "prepare_persistent_scene_reset(env)" in persistent_branch
    assert "env.scene.reset(hard=True)" in persistent_branch
    assert "_spawn_added_objects(env, run)" in persistent_branch
    # Plan-object existence is confirmed before replaying saved state changes.
    assert execute_block.index("plan objects missing after replay") < execute_block.index(
        "_apply_saved_initial_states(env, run)"
    )
    # The stability gate runs before camera/segmentation setup.
    assert "replay_integrity = _delta_replay_integrity(" in execute_block
    assert execute_block.index("_delta_replay_integrity") < execute_block.index(
        "_initialize_segmentation_streams("
    )


def test_oracle_replay_anchors_portables_without_weakening_physical_replay():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute_block = source[source.index("def execute("):source.index("def main()")]
    configs = source[
        source.index("def _physical_added_object_configs"):
        source.index("def _saved_robot_approaches")
    ]
    assert 'backend="physical_control"' in configs
    assert "anchored_for_replay = task_support" in configs
    assert '"fixed_base": anchored_for_replay' in configs
    assert '"kinematic_only": anchored_for_replay' in configs
    execute = source[source.index("def execute("):source.index("def main()")]
    assert 'settle_steps=0 if args.backend == "oracle_symbolic" else 20' in execute
    assert '"stage": "post_camera_delta_replay_integrity"' in execute
    assert '"delta_replay_integrity": replay_integrity' in execute_block
    gate = source[source.index("def _delta_replay_integrity"):source.index("def _scene_objects")]
    assert "for _ in range(settle_steps):" in gate
    assert '"task_support" in set(record.get("semantic_roles") or [])' in gate
    assert '"task_object" not in set(record.get("semantic_roles") or [])' in gate
    for key in ('"object_id"', '"saved_pose"', '"replayed_pose"', '"displacement"', '"kinematic_only"'):
        assert key in gate
    create_env_block = source[
        source.index("def _create_expert_env"):source.index("def _initialize_segmentation_streams")
    ]
    assert "added_objects=added_objects" in create_env_block
    api_source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    api_block = api_source[
        api_source.index("def create_env"):api_source.index("def stabilize_robot_spawn")
    ]
    assert "added_objects" in api_block
    assert 'cfg["objects"] = list(added_objects or [])' in api_block


def test_official_on_top_fallback_is_budgeted_once_per_object():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    relation = source[source.index("def _apply_relation"):source.index("def _state_by_name")]
    assert "def _apply_relation(self, obj, placement, official_fallback=None):" in relation
    # Existing geometric + official resample contract stays intact.
    assert "official_on_top_resample" in relation
    assert "reset_before_sampling=True" in relation
    # The blocking official sampler is budgeted, measured, and marks fatal
    # physics-view errors instead of being retried behind a logging timeout.
    assert "official_sampler_budget_exhausted" in relation
    assert "exceeded_relation_budget" in relation
    assert 'result["fatal_physics_error"] = True' in relation
    budget_init = source.index('official_fallback = {"calls": 0, "seconds": 0.0}')
    call_site = source.index("relation_result = self._apply_relation")
    assert budget_init < call_site
    assert "official_fallback=official_fallback" in source[budget_init:]
    assert '"error": "official_sampler_physics_error"' in source
    abort_at = source.index('"error": "official_sampler_physics_error"')
    assert "break" in source[abort_at:abort_at + 400]


def test_retrieval_prefers_vetted_native_support_before_bootstrap():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    start = source.index('if task_category == "retrieval_delivery":')
    end = source.index("native_target = None", start)
    setup = source[start:end]
    # A geometrically vetted native surface avoids an unnecessary furniture
    # spawn. Bootstrap support remains the fallback for sparse scenes.
    bootstrap_call = setup.index("self._choose_support_bootstrap_room(")
    safe_call = setup.index("self._choose_safe_target_room(")
    assert safe_call < bootstrap_call
    assert "excluded_rooms=self._rejected_rooms" in setup
    assert "generated_support_record = copy.deepcopy(" in setup
    assert "no native surface; placing breakfast_table in" in setup
    assert "using vetted native support in" in setup
    assert '"no_reachable_compatible_support_room"' in setup
    assert "preferred_target_room = target_room" in setup
    assert "alternate_exclusions.add(preferred_target_room)" in setup

    selector = source[
        source.index("def _choose_safe_target_room"):
        source.index("def _choose_support_bootstrap_room")
    ]
    assert "if approach_quality:" in selector
    assert "continue" in selector[selector.index("if approach_quality:"):]


def test_physics_rebuild_restores_displaced_native_furniture():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    capture = source[
        source.index("def _capture_physics_rebuild_poses"):
        source.index("def _restore_physics_rebuild_poses")
    ]
    restore = source[
        source.index("def _restore_physics_rebuild_poses"):
        source.index("def _tokens", source.index("def _restore_physics_rebuild_poses"))
    ]
    assert "targets.extend(self._scene_objects())" in capture
    assert "position_changed" in restore
    assert "orientation_delta" in restore
    assert "obj.set_position_orientation(position=position, orientation=orientation)" in restore


def test_failed_rooms_are_excluded_from_later_selection():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    assert "self._rejected_rooms: set[str] = set()" in source
    # A dead-end room is rejected when a task attempt aborts on placement.
    assert "self._rejected_rooms.add(target_room)" in source
    assert '"all_task_objects_failed", "fatal_physics_error",' in source
    # The native-surface room selector skips rejected rooms.
    selector = source[
        source.index("def _choose_safe_target_room"):source.index("def _match_required_objects")
    ]
    assert "if room in self._rejected_rooms:" in selector
    # Both bootstrap room selections honor the rejected-room memory.
    assert "excluded_rooms=self._rejected_rooms" in source
    assert "excluded_rooms={target_room} | self._rejected_rooms" in source


def test_retrieval_camera_failure_excludes_room_from_same_process_retry():
    source = (CODE_DIR / "online_deltasg.py").read_text(encoding="utf-8")
    builder = source[
        source.index("def _build_task_environment_record"):
        source.index("def _standard_added_objects")
    ]
    assert 'self._category_for_task(task_name) == "retrieval_delivery"' in builder
    assert "self._rejected_rooms.add(target_room)" in builder
    assert "(initial_camera_coverage)" in builder


def test_robot_spawn_rejects_full_link_collisions_before_warmup():
    source = (CODE_DIR / "api.py").read_text(encoding="utf-8")
    assert "def robot_native_link_overlaps" in source
    assert "penetration > min_penetration" in source
    assert '"wall", "walls", "ceiling", "ceilings"' in source
    collision_check = source.index("link_overlaps = robot_native_link_overlaps()")
    warmup = source.index("for _ in range(warmup_steps):", collision_check)
    assert collision_check < warmup


def test_physical_navigation_executes_clearance_route_with_base_controller():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    navigate = physical[
        physical.index("def _navigate_to_pose("):
        physical.index("def _rotate_base_to_yaw")
    ]
    assert 'self.robot.controller_action_idx["base"]' in navigate
    assert "step_distance = min(position_error, 0.02)" in navigate
    assert 'base_action[2] = max(-0.04, min(0.04, relative_yaw))' in navigate
    assert "_plan_joint_motion" not in navigate
    assert "CuRoboEmbodimentSelection.BASE" not in navigate
    assert "position_error <= 0.1 and yaw_error <= 0.2" in navigate

    planner = source[
        source.index("def _connected_navigation_waypoints("):
        source.index("def _closed_doors_on_route")
    ]
    assert "spacing=0.10" in planner
    assert "_navigation_blocker_aabbs(env, robot, target)" in planner
    # Blocker inflation must be z-aware: only boundary points whose height
    # overlaps the blocker can touch it. The constant arm-inclusive radius
    # (robot_radius + 0.05) severed real corridors (Beechwood_0 deliver_drink
    # attempt 5: 0.40 m ottoman blocked the carried-bottle route) and is gone.
    assert "robot_radius + 0.05" not in planner
    assert "boundary_horizontal_reach = np.linalg.norm(local_points[:, :2], axis=1)" in planner
    assert "band_z_min = float(lower[2]) - robot_z - 0.05" in planner
    assert "band_z_max = float(upper[2]) - robot_z + 0.05" in planner
    assert "if not len(band_reach):" in planner
    assert "inflation = float(band_reach.max()) + 0.05" in planner
    # Nothing else relaxed: clearance erosion, exact-footprint carve, BFS.
    assert "clearance_margin=0.20" in planner
    assert "carve_poses" in planner
    assert "free[pixels[:, 0], pixels[:, 1]] = True" in planner
    assert "No clearance-safe map path to manipulation stand-off" in planner


def test_physical_navigation_refuses_noop_standoff_for_every_target():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    navigate = physical[
        physical.index("def _navigate_to_obj("):
        physical.index("def _navigate_to_pose(")
    ]
    gate = "if eef_pose is None and target_distance > DEFAULT_MAX_PHYSICAL_APPROACH_DISTANCE:"
    assert gate in navigate
    # The honesty gate must fire before route planning / waypoint driving and
    # must not be restricted to spawned (online_env_) targets anymore.
    gate_block = navigate[
        navigate.index(gate):navigate.index("waypoints = _connected_navigation_waypoints")
    ]
    gate_code = "\n".join(
        line for line in gate_block.splitlines() if not line.strip().startswith("#")
    )
    assert "PLANNING_ERROR" in gate_block
    assert "refusing a no-op NAVIGATE_TO" in gate_block
    assert "online_env_" not in gate_code
    assert "horizontal_target_distance" in gate_block


def test_physical_navigation_keeps_a_collision_free_pose_already_near_target():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    navigate = physical[
        physical.index("def _navigate_to_obj("):
        physical.index("def _navigate_to_pose(")
    ]
    assert "current_position, current_orientation = self.robot.get_position_orientation()" in navigate
    assert "eef_pose is None and current_target_distance <= 0.75" in navigate
    assert "yaw_error <= 0.2" in navigate
    assert 'pose_source = "current_satisfied_pose"' in navigate
    assert "yield self._postprocess_action(self._empty_action())" in navigate
    assert "_native_occupant_at_pose(" in navigate
    assert 'pose_source = "current_reachable_pose"' in navigate
    assert "yield from self._navigate_to_pose(pose" in navigate


def test_expert_integrity_baseline_is_recorded_after_sensor_initialization_settles():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = source[source.index("def execute("):]
    camera_init = execute.index("camera_streams = _initialize_segmentation_streams")
    final_warmup = execute.index("_warm_natives_to_rest(env)", camera_init)
    baseline = execute.index("baseline = _native_pose_snapshot(env)")
    first_step = execute.index("for step_index, step in enumerate")
    assert camera_init < final_warmup < baseline < first_step


def test_coverage_reports_strict_and_assisted_physical_acceptance_separately():
    source = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    assert '"assisted_physical"' in source
    assert '"strict_physical"' in source
    assert '"expert_acceptance_kind_counts"' in source
    assert '"strict_physical_rate"' in source
    assert '"assisted_physical_rate"' in source


def test_adjacent_physical_steps_reuse_controller_for_the_same_target_and_rooms():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = source[source.index("def execute("):]
    assert "physical_controller_key = None" in execute
    assert "controller_key = (step.target_object, tuple(route_rooms))" in execute
    assert 'assisted_primitive = step.primitive in {' in execute
    assert "reuse_controller = (" in execute
    assert "if not reuse_controller:" in execute
    assert "physical_controller_key = controller_key" in execute


def test_physical_coverage_wrapper_uses_repo_env_and_physical_backend():
    source = (CODE_DIR / "run_enva_physical_coverage.sh").read_text(encoding="utf-8")
    assert 'source "$repo_root/.env"' in source
    assert "VLMEvalKit-main" not in source
    assert "--expert-backend physical_control" in source
    assert "--robot Tiago" in source
    assert "--process-retries 1" in source
    assert 'DELTASG_MIN_ELIGIBLE_ACCEPT_RATE:-0.80' in source


def test_spawned_portables_try_side_then_official_top_grasp():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    grasp = physical[physical.index("def _sample_grasp_pose("):physical.index("def _get_head_goal_q")]
    assert 'attempts[obj.name] += 1' in grasp
    assert 'not obj.name.startswith("online_env_") or attempt % 2 == 1' in grasp
    assert "return super()._sample_grasp_pose(obj)" in grasp
    assert "grasp_position[2] = upper[2]" in grasp
    assert "starter_primitives.m.GRASP_APPROACH_DISTANCE" in grasp


def test_physical_grasp_validates_generation_approach_before_random_sampling():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    sampler = physical[
        physical.index("def _sample_pose_near_object("):
        physical.index("def _move_hand(")
    ]
    assert 'getattr(self, "_deltasg_saved_robot_approaches", {})' in sampler
    assert "self._validate_poses(" in sampler
    assert '"arm_reachable_and_collision_free": valid' in sampler
    assert "pose = super()._sample_pose_near_object(" in sampler
    assert "return pose" in sampler
    assert "controller._deltasg_saved_robot_approaches = saved_robot_approaches" in source


def test_physical_place_primes_scene_query_visibility_before_each_attempt():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    probe = physical[
        physical.index("def _scene_query_visible("):
        physical.index("def _prime_scene_query_visibility(")
    ]
    assert "sampling_utils.raytest(" in probe
    assert "link_paths" in probe
    primer = physical[
        physical.index("def _prime_scene_query_visibility("):
        physical.index("def apply_ref(")
    ]
    assert "self._scene_query_visible(obj)" in primer
    assert "yield self._postprocess_action(" in primer
    apply_ref = physical[physical.index("def apply_ref("):physical.index("\ndef _jsonable(")]
    assert "StarterSemanticActionPrimitiveSet.PLACE_ON_TOP" in apply_ref
    assert "StarterSemanticActionPrimitiveSet.PLACE_INSIDE" in apply_ref
    assert "yield from self._prime_scene_query_visibility(place_target)" in apply_ref


def test_physical_place_rebuilds_physics_when_scene_queries_stay_blind():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    rebuild = physical[
        physical.index("def _rebuild_physics_scene_queries("):
        physical.index("def apply_ref(")
    ]
    # The heal is a full PhysX scene rebuild (stop/play), the mechanism proven
    # in diag23/diag24 and in the generation-path broadphase fix.
    assert "og.sim.stop()" in rebuild
    assert "og.sim.play()" in rebuild
    # Grasp must be protected across the rebuild (robot.py post_step gate),
    # the exact joint state restored (play() runs robot.reset()), and reverted
    # poses re-applied.
    assert "robot._disable_grasp_handling = True" in rebuild
    assert "robot.set_joint_positions(saved_joints, drive=False)" in rebuild
    assert "set_position_orientation(" in rebuild
    assert "keep_still()" in rebuild
    # AG constraint prim references are refreshed so the later release works.
    assert "_ag_obj_constraints" in rebuild
    # Settle steps stay in the recorded trace as no-op actions.
    assert "yield self._postprocess_action(" in rebuild
    apply_ref = physical[physical.index("def apply_ref("):physical.index("\ndef _jsonable(")]
    assert "if not rebuilt and not self._scene_query_visible(place_target):" in apply_ref
    assert "yield from self._rebuild_physics_scene_queries(place_target)" in apply_ref
    # Fix P: a corrupted sampling surface (reference-gate rejections) forces a
    # rebuild even though the visibility probe still sees the target's links.
    assert "_deltasg_place_sampling_corrupted" in apply_ref
    # Fix P2 (attempt-10): the sampler override sets that flag from an
    # exhaustion raise — a corrupted batch voids gate-passing survivors and
    # never returns a fallback from the corrupted surface, so the next
    # apply_ref attempt always runs the stop/play rebuild before resampling.
    override = physical[
        physical.index("def _sample_pose_with_object_and_predicate("):
        physical.index("def _move_fingers_to_limit(")
    ]
    assert '"reason": "clean_candidate_voided_corrupted_batch"' in override
    assert "self._deltasg_place_sampling_corrupted = True" in override
    assert "forcing physics rebuild and resample" in override


def test_symbolic_navigation_reuses_generation_validated_approach():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    oracle = source[
        source.index("class DeltaSGOraclePrimitives"):
        source.index("class DeltaSGPhysicalPrimitives")
    ]
    navigate = oracle[
        oracle.index("def _navigate_to_obj("):
        oracle.index("def _navigate_to_pose(")
    ]
    assert 'getattr(self, "_deltasg_saved_robot_approaches", {})' in navigate
    assert "route = [candidate_pose]" in navigate
    assert "_connected_observation_pose(" in navigate
    assert "controller._deltasg_saved_robot_approaches = saved_robot_approaches" in source


def test_sticky_approach_avoids_removed_attached_mesh_conflict():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    move = physical[physical.index("def _move_hand("):physical.index("def _get_head_goal_q")]
    assert "held is not None and held in (ignore_objects or [])" in move
    assert "update_obstacles(ignore_objects=ignore_objects)" in move
    assert "_convert_cartesian_to_joint_space(target_pose)" in move
    assert "_move_hand_direct_joint(joint_position)" in move
    assert "yield from super()._move_hand(" in move


def test_gripper_completion_ignores_unrelated_head_joint_motion():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    fingers = physical[
        physical.index("def _move_fingers_to_limit("):
        physical.index("def _get_head_goal_q")
    ]
    assert "gripper_indices = self.robot.gripper_control_idx[self.arm]" in fingers
    assert "current_gripper = self.robot.get_joint_positions()[gripper_indices]" in fingers
    assert "target_gripper = target_joint_positions[gripper_indices]" in fingers
    assert "current_joint_positions, target_joint_positions" not in fingers


def test_physical_open_close_revives_trajectory_with_real_state_gate():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    implementation = physical[
        physical.index("def _open_or_close("):
        physical.index("def _toggle(")
    ]
    assert "get_grasp_position_for_open" in implementation
    assert "_move_hand_linearly_cartesian" in implementation
    assert "bool(state.get_value()) == should_open" in implementation
    assert "NotImplementedError" not in implementation

    helper = (
        CODE_DIR.parent / "OmniGibson/omnigibson/utils/grasping_planning_utils.py"
    ).read_text(encoding="utf-8")
    assert "th.randperm(len(relevant_joints)).tolist()" in helper
    assert "relevant_joints.size(0)" not in helper
    assert "m.OPENNESS_THRESHOLD_TO_OPEN" in helper
    assert "m.OPENNESS_FRACTION_TO_OPEN" not in helper


def test_physical_toggle_uses_official_toggle_link_and_contact_steps():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    physical = source[source.index("class DeltaSGPhysicalPrimitives"):]
    implementation = physical[
        physical.index("def _toggle("):
        physical.index("def _get_head_goal_q")
    ]
    assert "toggle_position = state.link.get_position_orientation()[0]" in implementation
    assert "approach_position = toggle_position + approach_direction * 0.10" in implementation
    assert "_navigate_if_needed(obj, eef_pose=approach_hand_pose)" in implementation
    assert "_move_hand_linearly_cartesian(" in implementation
    assert "toggle_state.m.CAN_TOGGLE_STEPS" in implementation
    assert "bool(state.get_value()) == value" in implementation


def test_open_and_toggle_fast_path_is_audited_and_not_vla_eligible():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = source[source.index("def execute("):source.index("def main()")]
    assert 'step.primitive in {"OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}' in execute
    assert '"mode": "omnigibson_assisted_state_transition"' in execute
    assert 'record["assisted_interaction"] = interaction' in execute
    assert '"assisted_interaction": used_assisted_interaction' in execute
    assert "and not used_assisted_interaction" in execute

    audit = (CODE_DIR / "audit_deltasg_expert.py").read_text(encoding="utf-8")
    assert 'backend.get("assisted_interaction") is True' in audit
    assert 'backend.get("assisted_interaction") is not True' in audit
    assert 'backend.get("physical_solubility_validation") is True' in audit


def test_physical_visibility_recovery_is_part_of_the_exported_action_trace():
    source = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = source[source.index("def execute("):source.index("def main()")]
    assert "recovery_action_rows = []" in execute
    assert "recovery_action_rows = list(recovery_actions)" in execute
    assert '"actions_executed": len(recovery_action_rows)' in execute
    assert "action_rows = list(recovery_action_rows)" in execute
    assert "complete_action_trace = all(" in execute


def test_physical_dataset_consumers_preserve_strict_and_hybrid_contracts():
    audit = (CODE_DIR / "audit_deltasg_expert.py").read_text(encoding="utf-8")
    replay = (CODE_DIR / "replay_deltasg_expert_video.py").read_text(encoding="utf-8")
    manifest = (CODE_DIR / "run_deltasg_physical_manifest.sh").read_text(encoding="utf-8")
    assert 'parser.add_argument("--require-low-level-vla-actions"' in audit
    assert "expert result does not belong to the requested input JSON" in replay
    assert 'backend.get("complete_action_trace") is True' in replay
    assert 'object_states.Open' in replay
    assert 'object_states.ToggledOn' in replay
    assert 'replayed_assisted_interactions' in replay
    assert "--require-low-level-vla-actions" in manifest
    assert '"vla_eligible_rate"' in manifest


def test_single_scene_e2e_runner_uses_configurable_model_and_backend_specific_80_percent_gate():
    source = (CODE_DIR / "run_enva_single_scene_e2e.sh").read_text(encoding="utf-8")
    assert 'MODEL="${DELTASG_LLM_MODEL:-qwen3.8-max}"' in source
    assert '--task-sequence "$TASKS"' in source
    assert 'bash code/run_deltasg_expert_batch.sh' in source
    assert 'backend.get("physical_trajectory_available") is True' in source
    assert 'backend.get("name") == "oracle_symbolic"' in source
    assert "and bool(steps)" in source
    assert 'step.get("pre_observation") and step.get("post_observation")' in source
    assert 'qualified = symbolic if required_backend == "oracle_symbolic"' in source
    assert 'recorded_input.resolve() == Path(generated[task]).resolve()' in source
    assert '"generation_coverage_rate": generation_ok / len(tasks)' in source
    assert '"expert_success_rate": expert_success_rate' in source
    assert "and expert_completed == generation_ok" in source
    assert '"required_rate": 0.80' in source
    assert "and end_to_end_rate >= 0.80" in source


def test_serial_native_spawn_failure_uses_the_generation_retry_budget():
    source = (CODE_DIR / "run_online_deltasg.py").read_text(encoding="utf-8")
    block = source[
        source.index("if spawn_failed:"):
        source.index('if args.env_type == "A":')
    ]
    assert "attempt += 1" in block
    assert "limit = args.max_retries" in block
    assert "if limit > 0 and attempt > limit:" in block
    assert "continue" in block
    assert 'f"[robot-spawn] retry {attempt}"' in block
    runner = (CODE_DIR / "run_enva_single_scene_e2e.sh").read_text(encoding="utf-8")
    assert '--max-llm-retries 4 --max-retries 3' in runner


def test_persistent_symbolic_expert_reuses_preloaded_same_scene_environments():
    expert = (CODE_DIR / "run_deltasg_expert.py").read_text(encoding="utf-8")
    execute = expert[expert.index("def execute("):expert.index("def main()")]
    assert "env=None, persistent=False" in execute
    assert "env.scene.reset(hard=True)" in execute
    assert "_spawn_added_objects(env, run)" in execute
    assert "cleanup_persistent_camera_streams(camera_streams)" in execute

    worker = (CODE_DIR / "run_deltasg_expert_persistent.py").read_text(encoding="utf-8")
    assert "rows.sort(key=lambda row: (row[0], row[1], str(row[2])))" in worker
    assert "def _preloaded_object_configs(" in worker
    assert 'parked["position"]' in worker
    assert "expert._create_expert_env(" in worker
    assert "execute_args.preloaded_delta_names = preloaded_names" in worker
    assert "fresh_environment" in worker
    assert "fresh_environment = key != environment_key or env is None" in worker
    assert "environment_loads += 1" in worker
    assert "persistent=True" in worker
    assert '"environment_loads": environment_loads' in worker
    batch = (CODE_DIR / "run_deltasg_expert_batch.sh").read_text(encoding="utf-8")
    symbolic_branch = batch[
        batch.index('if [[ "$BACKEND" == "oracle_symbolic" ]]'):
        batch.index("mapfile -d '' FILES")
    ]
    assert "python code/run_deltasg_expert_persistent.py" in symbolic_branch
    assert '--input-root "$INPUT_ROOT"' in symbolic_branch
    assert 'DELTASG_CHILD_TIMEOUT=0 code/run_omnigibson_single_gpu.sh' in symbolic_branch


def test_all_scene_symbolic_progress_reports_generation_and_expert_rates_separately():
    progress = (CODE_DIR / "report_enva_symbolic_progress.py").read_text(encoding="utf-8")
    assert "recorded_input == input_path.resolve()" in progress
    assert 'backend.get("name") == "oracle_symbolic"' in progress
    assert 'expert_rate={expert_rate:.1f}%' in progress
    assert 'e2e_rate={e2e_rate:.1f}%' in progress
    aggregate = (CODE_DIR / "run_enva_symbolic_all_scenes.sh").read_text(encoding="utf-8")
    assert '"generation_coverage_rate": generated / total if total else 0.0' in aggregate
    assert '"expert_success_rate": qualified / expert_completed if expert_completed else 0.0' in aggregate
