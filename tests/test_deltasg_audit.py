"""Regression checks for scene-aware DeltaSG visualization auditing."""

import sys
import tempfile
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from audit_deltasg_outputs import bbox_file_for_run, robot_pose_upright_error
from audit_deltasg_expert import is_physical_trajectory_eligible


def test_physical_trajectory_gate_separates_hybrid_from_symbolic():
    hybrid = {
        "accepted": True,
        "backend": {
            "name": "physical_control",
            "generation_solvability_profile": "physical_control",
            "generation_profile_verified": True,
            "complete_action_trace": True,
            "physical_trajectory_available": True,
            "physical_action_count": 120,
            "physical_nonzero_action_count": 95,
            "assisted_interaction": True,
            "low_level_vla_actions_eligible": False,
        },
    }
    assert is_physical_trajectory_eligible(hybrid, "physical_control")
    hybrid["backend"]["name"] = "oracle_symbolic"
    assert not is_physical_trajectory_eligible(hybrid, "physical_control")


def test_robot_pose_upright_audit_rejects_fallen_generation_pose():
    stable = {
        "position": [2.5, -3.8, 0.002],
        "orientation_xyzw": [0.0, 0.0, 0.7, 0.714],
    }
    fallen = {
        "position": [2.5, -3.8, 0.327],
        "orientation_xyzw": [-0.404, 0.324, 0.831, 0.202],
    }
    assert robot_pose_upright_error(stable) is None
    assert robot_pose_upright_error(fallen).startswith("robot_pose_not_upright")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_a = root / "outputs" / "envB_fire" / "Scene_A" / "online_env_b_fire_0000.json"
        run_b = root / "outputs" / "envB_fire" / "Scene_B" / "online_env_b_fire_0000.json"
        vis_a = root / "vis" / "envB_fire" / "Scene_A" / "online_env_b_fire_0000_after_bboxes.json"
        vis_b = root / "vis" / "envB_fire" / "Scene_B" / "online_env_b_fire_0000_after_bboxes.json"
        for path in (run_a, run_b, vis_a, vis_b):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        assert bbox_file_for_run(run_a, root / "vis") == vis_a
        assert bbox_file_for_run(run_b, root / "vis") == vis_b

    print("PASS: DeltaSG bbox audit resolves repeated stems by label and scene")


if __name__ == "__main__":
    main()
