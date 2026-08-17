import json
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from build_enva_expert_coverage_manifest import RETRIEVAL_TASK_ASSETS, build_jobs
from build_enva_native_eligibility import build_native_eligibility
from run_enva_expert_coverage import (
    _all_native_candidates_resolved,
    _native_attempt_schedule,
    _native_ineligibility,
    _coverage_status_for_expert_state,
    _identity_errors,
    _resumable_ineligible,
    _selected_native_target,
    _resumable_generation,
)


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("accepted", "accepted"),
        ("rejected", "expert_rejected"),
        ("process_error", "expert_process_error"),
        ("missing", "expert_process_error"),
        ("invalid", "expert_process_error"),
    ],
)
def test_expert_process_failure_is_not_classified_as_task_rejection(state, status):
    assert _coverage_status_for_expert_state(state) == status


def test_coverage_runner_has_a_strict_eligible_acceptance_gate():
    source = (CODE_DIR / "run_enva_expert_coverage.py").read_text(encoding="utf-8")
    assert '"--min-eligible-accept-rate", type=float, default=0.90' in source
    assert 'report["eligible_accept_rate"] >= args.min_eligible_accept_rate' in source
    assert 'report["runtime_ineligible_rate"] <= args.max_runtime_ineligible_rate' in source
    assert 'parser.add_argument("--max-runtime-ineligible-rate", type=float, default=0.05)' in source
    assert 'if not os.environ.get("DASHSCOPE_API_KEY")' in source
    assert "forbids heuristic-only generation" in source
    assert "def _terminate_active_process" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source
    assert "signal.SIGHUP" in source
    assert 'parser.add_argument("--robot", default=None)' in source
    assert '"--robot", generation_robot' in source
    assert "generation --robot and --expert-robot must match" in source
    assert '"generation_robot": generation_robot' in source
    assert 'formal_errors.append("generation_robot_mismatch")' in source
    assert "generation_robot_matches" in source
    assert "resumed_robot.casefold() == generation_robot.casefold()" in source
    assert source.count('"robot_approach",') >= 2
    assert 'parser.add_argument("--num-shards", type=int, default=1)' in source
    assert 'parser.add_argument("--shard-index", type=int, default=0)' in source
    assert 'scenes = sorted({job["scene"] for job in jobs})' in source
    assert 'jobs = [job for job in jobs if job["scene"] in shard_scenes]' in source


def _inputs():
    categories = {category for values in RETRIEVAL_TASK_ASSETS.values() for category in values}
    inventory = {
        "groups": {
            "retrieval_delivery": {
                category: [f"{category}_model_0", f"{category}_model_1"]
                for category in categories
            }
        }
    }
    native = {
        "scenes": {
            scene: {
                "eligible_tasks": {
                    "open_door": [f"{scene}_door"],
                    "turn_on_light": [f"{scene}_switch"],
                }
            }
            for scene in ("scene_0", "scene_1")
        }
    }
    return inventory, native


def test_smoke_manifest_has_one_job_per_family_and_scene():
    inventory, native = _inputs()
    jobs = build_jobs(["scene_0", "scene_1"], inventory, native, "smoke")
    assert len(jobs) == 6
    assert {(job["scene"], job["label"]) for job in jobs} == {
        (scene, label)
        for scene in ("scene_0", "scene_1")
        for label in ("envA_retrieval_delivery", "envA_open_close", "envA_appliance")
    }
    assert all(
        job["target_native_candidates"] and job["target_native_object_id"] is None
        for job in jobs
        if job["label"] != "envA_retrieval_delivery"
    )


def test_full_manifest_enumerates_every_compatible_model_and_native_target():
    inventory, native = _inputs()
    jobs = build_jobs(["scene_0"], inventory, native, "full")
    retrieval = [job for job in jobs if job["label"] == "envA_retrieval_delivery"]
    expected_retrieval = sum(
        2 * len(categories) for categories in RETRIEVAL_TASK_ASSETS.values()
    )
    assert len(retrieval) == expected_retrieval
    assert {(job["task"], job["target_native_object_id"]) for job in jobs if job["target_native_object_id"]} == {
        ("open_door", "scene_0_door"),
        ("turn_on_light", "scene_0_switch"),
    }
    native_jobs = [job for job in jobs if job["label"] != "envA_retrieval_delivery"]
    assert {job["task"] for job in native_jobs} == {
        "open_door", "close_door", "open_window", "close_window",
        "open_fridge", "close_fridge", "open_cabinet", "close_cabinet",
        "turn_on_light", "turn_off_light", "turn_on_tv", "turn_off_tv",
        "turn_on_stove", "turn_off_stove",
    }
    assert sum(bool(job["preflight_ineligibility"]) for job in native_jobs) == 12


def test_task_manifest_resolves_every_scene_task_cell():
    inventory, native = _inputs()
    jobs = build_jobs(["scene_0", "scene_1"], inventory, native, "tasks")
    assert len(jobs) == 2 * (len(RETRIEVAL_TASK_ASSETS) + 14)
    assert all(
        job["target_native_candidates"] or job["preflight_ineligibility"]
        for job in jobs
        if job["label"] != "envA_retrieval_delivery"
    )


def test_retrieval_variants_expand_only_placeable_targets():
    inventory, native = _inputs()
    base = build_jobs(["scene_0"], inventory, native, "full")
    expanded = build_jobs(
        ["scene_0"], inventory, native, "full", retrieval_variants=3
    )
    base_retrieval = [job for job in base if job["label"] == "envA_retrieval_delivery"]
    base_native = [job for job in base if job["label"] != "envA_retrieval_delivery"]
    expanded_retrieval = [
        job for job in expanded if job["label"] == "envA_retrieval_delivery"
    ]
    expanded_native = [job for job in expanded if job["label"] != "envA_retrieval_delivery"]
    assert len(expanded_retrieval) == 3 * len(base_retrieval)
    assert len(expanded_native) == len(base_native)
    assert {job["variant_index"] for job in expanded_retrieval} == {0, 1, 2}


def test_exact_native_height_rejection_is_classified_as_ineligible():
    job = {"target_native_object_id": "cabinet_0"}
    rejected = {
        "validation": {
            "llm_validation": {
                "issues": ["no_matching_native_stateful_target"],
                "detail": {
                    "stage": "manipulation_height",
                    "object_id": "cabinet_0",
                },
            }
        }
    }
    assert _native_ineligibility(job, rejected)["stage"] == "manipulation_height"
    assert _native_ineligibility({}, rejected) is None


def test_native_candidate_schedule_attempts_every_fallback():
    job = {"target_native_candidates": ["cabinet_0", "cabinet_1", "cabinet_2"]}
    assert _native_attempt_schedule(job, 2) == ["cabinet_0", "cabinet_1", "cabinet_2"]
    assert _native_attempt_schedule(job, 4) == ["cabinet_0", "cabinet_1", "cabinet_2"]
    assert _native_attempt_schedule({"target_native_object_id": "cabinet_0"}, 3) == [
        "cabinet_0", "cabinet_0", "cabinet_0",
    ]


def test_smoke_manifest_prioritizes_roomy_native_targets_without_dropping_any():
    inventory, native = _inputs()
    native["scenes"]["scene_0"]["eligible_tasks"]["turn_on_light"] = [
        "switch_storage", "switch_kitchen", "switch_utility",
    ]
    native["scenes"]["scene_0"]["eligible_target_objects"] = {
        "switch_storage": {"rooms": ["storage_room_0"]},
        "switch_kitchen": {"rooms": ["kitchen_0"]},
        "switch_utility": {"rooms": ["utility_room_0"]},
    }
    jobs = build_jobs(["scene_0"], inventory, native, "smoke")
    appliance = next(job for job in jobs if job["label"] == "envA_appliance")
    assert appliance["target_native_candidates"] == [
        "switch_kitchen", "switch_utility", "switch_storage",
    ]


def test_smoke_rotation_never_promotes_confined_room_over_roomy_target():
    inventory, native = _inputs()
    for scene in ("scene_0", "scene_1"):
        native["scenes"][scene]["eligible_tasks"]["turn_on_light"] = [
            "switch_storage", "switch_kitchen_0", "switch_kitchen_1",
        ]
        native["scenes"][scene]["eligible_target_objects"] = {
            "switch_storage": {"rooms": ["storage_room_0"]},
            "switch_kitchen_0": {"rooms": ["kitchen_0"]},
            "switch_kitchen_1": {"rooms": ["kitchen_0"]},
        }
    jobs = build_jobs(["scene_0", "scene_1"], inventory, native, "smoke")
    appliance = [job for job in jobs if job["label"] == "envA_appliance"]
    assert all(job["target_native_candidates"][-1] == "switch_storage" for job in appliance)
    assert appliance[0]["target_native_candidates"][:2] != appliance[1]["target_native_candidates"][:2]


def test_automatic_native_selection_records_the_exact_interaction_target():
    run = {
        "task_environment": {
            "solution_plan": [
                {"primitive": "MOVE", "target_object": "switch_0"},
                {"primitive": "INTERACT", "target_object": "switch_0"},
            ]
        }
    }
    assert _selected_native_target(run) == "switch_0"
    run["task_environment"]["solution_plan"].append(
        {"primitive": "INTERACT", "target_object": "switch_1"}
    )
    assert _selected_native_target(run) is None


def test_expert_retry_reuses_only_audited_same_robot_generation(tmp_path, monkeypatch):
    generation = tmp_path / "online_env_a_0000.json"
    generation.write_text(json.dumps({
        "ok": True,
        "robot": {"model": "tiago"},
        "validation": {"sample_fingerprint": "unique-fingerprint"},
        "task_environment": {
            "generation": {"solvability_profile": "oracle_symbolic"},
            "task": {"primary_behavior_task": "retrieve_key"},
            "added_objects": [{
                "category": "key_chain",
                "model": "gqbwfv",
                "semantic_roles": ["task_object"],
            }],
        },
    }))
    result_path = tmp_path / "job_result.json"
    result_path.write_text(json.dumps({
        "generation_json": str(generation),
        "generation_process_codes": [0],
        "generation_seconds": 12.5,
    }))
    monkeypatch.setattr("run_enva_expert_coverage.check_run", lambda _path, _run: [])
    job = {
        "task": "retrieve_key",
        "target_asset_category": "key_chain",
        "target_asset_model": "gqbwfv",
        "target_native_object_id": None,
    }
    resumed = _resumable_generation(
        job, result_path, "Tiago", set(), "oracle_symbolic"
    )
    assert resumed["generation_json"] == generation
    assert resumed["sample_fingerprint"] == "unique-fingerprint"
    assert _resumable_generation(
        job, result_path, "R1", set(), "oracle_symbolic"
    ) is None
    assert _resumable_generation(
        job, result_path, "Tiago", {"unique-fingerprint"}, "oracle_symbolic"
    ) is None
    assert _resumable_generation(
        job, result_path, "Tiago", set(), "physical_control"
    ) is None
    saved = json.loads(generation.read_text())
    del saved["task_environment"]["generation"]
    generation.write_text(json.dumps(saved))
    assert _resumable_generation(
        job, result_path, "Tiago", set(), "oracle_symbolic"
    ) is None


def test_identity_requires_the_requested_generation_solvability_profile():
    job = {
        "task": "retrieve_key",
        "target_asset_category": None,
        "target_native_object_id": None,
    }
    run = {
        "task_environment": {
            "generation": {"solvability_profile": "oracle_symbolic"},
            "task": {"primary_behavior_task": "retrieve_key"},
        }
    }
    assert not _identity_errors(job, run, "oracle_symbolic")
    assert _identity_errors(job, run, "physical_control") == [
        "generation_solvability_profile_mismatch"
    ]
    del run["task_environment"]["generation"]
    assert _identity_errors(job, run, "oracle_symbolic") == [
        "generation_solvability_profile_mismatch"
    ]


def test_native_candidate_job_is_resolved_only_after_every_fixture_has_evidence():
    candidates = ["cabinet_0", "cabinet_1"]
    assert not _all_native_candidates_resolved(
        candidates, [{"object_id": "cabinet_0", "stage": "manipulation_height"}]
    )
    assert _all_native_candidates_resolved(
        candidates,
        [
            {"object_id": "cabinet_0", "stage": "manipulation_height"},
            {"object_id": "cabinet_1", "stage": "robot_approach"},
        ],
    )


def test_resumable_ineligible_requires_matching_robot_and_complete_candidates(tmp_path):
    job = {
        "job_id": "job_1",
        "scene": "scene_0",
        "task": "close_window",
        "target_native_object_id": None,
        "target_native_candidates": ["window_0", "window_1"],
    }
    result_path = tmp_path / "job_result.json"
    result_path.write_text(json.dumps({
        **job,
        "status": "ineligible_target",
        "generation_robot": "Tiago",
        "expert_robot": "Tiago",
        "ineligibility_reasons": [
            {"object_id": "window_0", "stage": "robot_approach"},
            {"object_id": "window_1", "stage": "room"},
        ],
    }))
    resumed = _resumable_ineligible(job, result_path, "Tiago", "Tiago")
    assert resumed is not None and resumed["resumed"] is True
    assert _resumable_ineligible(job, result_path, "R1", "R1") is None
    prior = json.loads(result_path.read_text())
    prior["ineligibility_reasons"].pop()
    result_path.write_text(json.dumps(prior))
    assert _resumable_ineligible(job, result_path, "Tiago", "Tiago") is None


def test_manifest_rejects_incomplete_scene_inventory():
    inventory, native = _inputs()
    with pytest.raises(ValueError, match="missing scenes"):
        build_jobs(["scene_0", "scene_missing"], inventory, native, "smoke")


def test_versioned_scene_list_and_native_inventory_are_complete_and_aligned():
    scenes = [
        line.strip()
        for line in (CODE_DIR / "configs" / "env_a_scenes.txt").read_text().splitlines()
        if line.strip()
    ]
    native = json.loads(
        (CODE_DIR / "configs" / "env_a_native_eligibility.json").read_text()
    )
    assert len(scenes) == len(set(scenes)) == 15
    assert set(native["scenes"]) == set(scenes)
    assert all(scene["eligible_tasks"] for scene in native["scenes"].values())
    assert len({task for scene in native["scenes"].values() for task in scene["eligible_tasks"]}) == 14
    assert len({
        object_id
        for scene in native["scenes"].values()
        for object_ids in scene["eligible_tasks"].values()
        for object_id in object_ids
    }) == 124
    assert sum(
        len(object_ids)
        for scene in native["scenes"].values()
        for object_ids in scene["eligible_tasks"].values()
    ) == 1054


def test_manifest_rejects_unknown_native_task_contract():
    inventory, native = _inputs()
    native["scenes"]["scene_0"]["eligible_tasks"]["future_unvalidated_task"] = ["thing_0"]
    with pytest.raises(ValueError, match="unsupported tasks"):
        build_jobs(["scene_0"], inventory, native, "smoke")


def test_versioned_retrieval_inventory_covers_every_contract_asset_model():
    inventory_path = CODE_DIR / "configs" / "env_a_asset_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    models = inventory["groups"]["retrieval_delivery"]
    expected_categories = {
        category for categories in RETRIEVAL_TASK_ASSETS.values() for category in categories
    }
    assert set(models) == expected_categories
    assert sum(len(category_models) for category_models in models.values()) == 61
    assert all(category_models == sorted(set(category_models)) for category_models in models.values())

    source = (CODE_DIR / "build_enva_expert_coverage_manifest.py").read_text(
        encoding="utf-8"
    )
    assert '"configs" / "env_a_asset_inventory.json"' in source


def test_native_eligibility_rebuilds_both_families_from_saved_graph():
    graph = {
        "nodes": [
            {
                "id": "floor_0", "type": "object", "category": "floor",
                "rooms": ["room_0"], "bbox": {"min": [0, 0, -0.02], "max": [2, 2, 0]},
            },
            {
                "id": "door_0", "type": "object", "category": "door",
                "rooms": ["room_0"], "available_states": ["Open"],
                "bbox": {"min": [0, 0, 0], "max": [1, 0.1, 2]},
            },
            {
                "id": "switch_0", "type": "object", "category": "electric_switch",
                "rooms": ["room_0"], "available_states": ["ToggledOn"],
                "bbox": {"min": [0, 0, 0.9], "max": [0.1, 0.1, 1.1]},
            },
            {
                "id": "window_without_room", "type": "object", "category": "openable_window",
                "rooms": [], "available_states": ["Open"],
                "bbox": {"min": [0, 0, 0.5], "max": [0.1, 0.1, 1.5]},
            },
        ]
    }
    runs = [("sample.json", {"base_scene": {"scene_model": "scene_0"}, "before_graph": graph})]
    result = build_native_eligibility(["scene_0"], runs)
    tasks = result["scenes"]["scene_0"]["eligible_tasks"]
    assert {"open_door", "close_door", "turn_on_light", "turn_off_light"} <= set(tasks)
    assert set(result["scenes"]["scene_0"]["eligible_target_objects"]) == {"door_0", "switch_0"}


def test_native_eligibility_rebuild_rejects_missing_scene_graph():
    with pytest.raises(ValueError, match="no saved before_graph"):
        build_native_eligibility(["scene_missing"], [])
