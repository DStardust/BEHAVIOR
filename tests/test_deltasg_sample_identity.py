import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from deltasg_sample_identity import sample_fingerprint


def sample(prefix):
    return {"task_environment": {
        "env_type": "Env-B", "base_scene": {"scene_model": "Ihlen_1_int"},
        "task": {"primary_behavior_task": "clean_dirty_dishes", "plan_objects": [
            {"object_id": prefix + "plate", "category": "plate"}]},
        "added_objects": [
            {"object_name": prefix + "plate", "category": "plate", "model": "a",
             "pose": {"position": [1, 2, 0.8]},
             "placement": {"support_object_id": prefix + "table"}},
            {"object_name": prefix + "table", "category": "table", "model": "b",
             "pose": {"position": [1, 2, 0.4]}},
        ],
        "state_changed_objects": [{"object_id": prefix + "plate",
                                   "states": {"covered": {"system": "stain", "value": True}}}],
    }}


def test_generated_names_and_record_order_do_not_create_diversity():
    first, second = sample("online_1_"), sample("online_2_")
    second["task_environment"]["added_objects"].reverse()
    assert sample_fingerprint(first)[0] == sample_fingerprint(second)[0]


def test_particle_render_details_do_not_disguise_duplicate_task_placement():
    first, second = sample("online_1_"), sample("online_2_")
    second["task_environment"]["state_changed_objects"][0]["states"]["covered"]["particle_snapshot"] = {
        "positions": [[0.01, 0.02, 0.03]]}
    assert sample_fingerprint(first)[0] == sample_fingerprint(second)[0]


def test_position_model_state_and_scene_are_real_diversity():
    first = sample("online_1_")
    for dimension in ("position", "model", "state", "scene"):
        other = copy.deepcopy(first)
        te = other["task_environment"]
        if dimension == "position":
            te["added_objects"][0]["pose"]["position"][0] += 0.25
        elif dimension == "model":
            te["added_objects"][0]["model"] = "different"
        elif dimension == "state":
            te["state_changed_objects"][0]["states"]["covered"]["value"] = False
        else:
            te["base_scene"]["scene_model"] = "Beechwood_0_int"
        assert sample_fingerprint(first)[0] != sample_fingerprint(other)[0]
