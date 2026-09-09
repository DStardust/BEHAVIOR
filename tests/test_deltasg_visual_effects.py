from deltasg_visual_effects import (
    SMOKE_FLOW_RENDER_WARMUP_FRAMES,
    SMOKE_FLOW_MAX_EMITTER_RADIUS,
    SMOKE_FLOW_WARMUP_STEPS,
    SMOKE_COLUMN_EMITTER_RADIUS,
    SMOKE_COLUMN_FADE,
    SMOKE_COLUMN_UPWARD_VELOCITY,
    SMOKE_ONLY_ON_FIRE_MODE,
    USDZ_FLAME_ASSET,
    USDZ_FLAME_MODE,
    USDZ_FLAME_SMOKE_MODE,
    flame_asset_path,
    smoke_only_on_fire_record,
    usdz_flame_record,
    usdz_flame_smoke_record,
)


def test_smoke_only_on_fire_record_is_explicit():
    record = smoke_only_on_fire_record()
    assert record == {
        "mode": SMOKE_ONLY_ON_FIRE_MODE,
        "source": "omnigibson_flow_emitter",
        "smoke_visible": True,
        "flame_visible": False,
    }


def test_smoke_flow_warmup_advances_two_seconds_at_default_rate():
    assert SMOKE_FLOW_WARMUP_STEPS == 120
    assert SMOKE_FLOW_RENDER_WARMUP_FRAMES == 30
    assert SMOKE_FLOW_MAX_EMITTER_RADIUS == 0.12


def test_usdz_flame_record_is_portable_and_asset_is_packaged():
    record = usdz_flame_record(effect_id="stove_0", target_height=0.3)
    assert record["mode"] == USDZ_FLAME_MODE
    assert record["asset"] == USDZ_FLAME_ASSET
    assert record["effect_id"] == "stove_0"
    assert record["target_height"] == 0.3
    assert record["flame_visible"] is True
    assert record["replaces_official_flow_visual"] is True
    assert flame_asset_path().is_file()


def test_combined_fire_record_has_large_vertical_smoke_column():
    record = usdz_flame_smoke_record(effect_id="stove_0", target_height=0.3)
    assert record["mode"] == USDZ_FLAME_SMOKE_MODE
    assert record["flame_visible"] is True
    assert record["smoke_visible"] is True
    assert record["replaces_official_flow_visual"] is False
    assert record["smoke_profile"] == "vertical_column_v1"
    assert record["smoke_emitter_radius"] == SMOKE_COLUMN_EMITTER_RADIUS == 0.18
    assert record["smoke_upward_velocity"] == SMOKE_COLUMN_UPWARD_VELOCITY == 1.5
    assert record["smoke_fade"] == SMOKE_COLUMN_FADE == 0.12


def test_fire_audit_requires_combined_flame_and_smoke_contract():
    from audit_deltasg_outputs import fire_visual_issue

    state = [{
        "states": {"on_fire": True},
        "anomaly_phase": "visible_flame",
        "visual_effect": usdz_flame_smoke_record(effect_id="stove_0", target_height=0.3),
    }]
    assert fire_visual_issue(state) is None
    state[0]["visual_effect"]["smoke_visible"] = False
    assert fire_visual_issue(state) == "on_fire_visual_contract_invalid"


def test_generation_and_persistent_expert_clear_previous_usdz_flames():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "code"
    generator = (root / "online_deltasg.py").read_text(encoding="utf-8")
    expert = (root / "run_deltasg_expert.py").read_text(encoding="utf-8")
    fire = generator[
        generator.index("def generate_env_b_fire") :
        generator.index("def generate_env_c_fire_disambiguation")
    ]
    replay = expert[
        expert.index("def _apply_saved_initial_states") :
        expert.index("def _delta_replay_integrity")
    ]
    persistent_reset = expert[
        expert.index("def prepare_persistent_scene_reset") :
        expert.index("def configure_preloaded_delta_objects")
    ]
    assert "remove_usdz_flame()" in fire
    assert "remove_usdz_flame()" in replay
    assert "remove_usdz_flame()" in persistent_reset


def test_covered_snapshot_restores_local_attachments_without_resampling():
    from types import SimpleNamespace
    import torch
    from deltasg_visual_effects import restore_covered_particle_snapshot

    calls = {}
    system = SimpleNamespace(
        name="stain", groups={"plate"}, get_group_name=lambda obj: "plate",
        remove_all_group_particles=lambda group: calls.update(removed=group),
        generate_group_particles=lambda **kwargs: calls.update(generated=kwargs),
        set_group_particles_local_pose=lambda group, **kwargs: calls.update(local=kwargs),
    )
    obj = SimpleNamespace(links={"base_link": SimpleNamespace(prim_path="/World/plate/base_link")})
    snapshot = {"version": 1, "system": "stain", "links": ["base_link"],
                "positions": [[0.1, 0.2, 0.003]], "orientations": [[0, 0, 0, 1]],
                "scales": [[0.01, 0.02, 0.01]]}
    restore_covered_particle_snapshot(obj, system, snapshot)
    assert calls["removed"] == "plate"
    assert calls["generated"]["link_prim_paths"] == ["/World/plate/base_link"]
    assert torch.allclose(calls["local"]["positions"], torch.tensor(snapshot["positions"]))
    assert torch.allclose(calls["generated"]["scales"], torch.tensor(snapshot["scales"]))


def test_invalid_covered_snapshot_is_rejected_before_particle_mutation():
    from types import SimpleNamespace
    import pytest
    from deltasg_visual_effects import restore_covered_particle_snapshot

    system = SimpleNamespace(name="stain")
    obj = SimpleNamespace(links={"base_link": object()})
    snapshot = {"version": 1, "system": "stain", "links": ["missing"],
                "positions": [[0, 0, 0]], "orientations": [[0, 0, 0, 1]],
                "scales": [[1, 1, 1]]}
    with pytest.raises(ValueError, match="Invalid Covered"):
        restore_covered_particle_snapshot(obj, system, snapshot)


def test_covered_snapshot_only_saves_target_group():
    from types import SimpleNamespace
    import torch
    from deltasg_visual_effects import covered_particle_snapshot

    system = SimpleNamespace(
        name="stain", get_group_name=lambda obj: "plate",
        dump_state=lambda serialized: {
            "groups": {"plate": {"particle_attached_references": ["base_link"],
                                  "particle_indices": [1]}},
            "scales": torch.tensor([[9, 9, 9], [1, 2, 3]]),
        },
        get_group_particles_local_pose=lambda group: (
            torch.tensor([[0, 0, 1]]), torch.tensor([[0, 0, 0, 1]])),
    )
    snapshot = covered_particle_snapshot(SimpleNamespace(links={"base_link": object()}), system)
    assert snapshot["scales"] == [[1, 2, 3]]
    assert snapshot["positions"] == [[0, 0, 1]]
    assert snapshot["links"] == ["base_link"]
