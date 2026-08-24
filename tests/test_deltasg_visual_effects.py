from deltasg_visual_effects import (
    SMOKE_FLOW_RENDER_WARMUP_FRAMES,
    SMOKE_FLOW_MAX_EMITTER_RADIUS,
    SMOKE_FLOW_WARMUP_STEPS,
    SMOKE_ONLY_ON_FIRE_MODE,
    smoke_only_on_fire_record,
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
