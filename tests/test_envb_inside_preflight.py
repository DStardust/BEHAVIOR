"""Verify preflight restores both scene state and sampler settings."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call
from contextlib import nullcontext
import sys
import pytest


@pytest.mark.parametrize("reused", [False, True])
def test_container_retry_preserves_subject_and_never_removes_native_container(reused):
    from collections import Counter
    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    loop = next(n for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == "container_attempt")
    live = SimpleNamespace(model="a")
    engine = Mock()
    engine._env_b_attempt_counts = Counter()
    engine.env.scene.object_registry.return_value = live
    engine.add_task_asset.side_effect = [
        {"ok": True, "object_name": "bin", "reused": reused},
        {"ok": True, "object_name": "bin", "reused": False},
    ]
    engine._preflight_env_b_inside.side_effect = [{"ok": False}, {"ok": True}]
    subject = SimpleNamespace(name="clothing")
    namespace = dict(self=engine, container_trials=2, container_models=["a", "b"],
                     role="task_destination", inside_key="pair:", record={},
                     run_id="test", actual_category="hamper", target_room="room",
                     tool_position=None, preferred_position=None, anomaly_obj=subject,
                     recipe={"path_name": "put_in_hamper"})
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), "exec"), namespace)
    assert engine.add_task_asset.call_count == (1 if reused else 2)
    assert engine._env_b_attempt_counts["pair:a"] == 1
    if reused:
        engine._remove_object_safe_by_name.assert_not_called()
    else:
        engine._remove_object_safe_by_name.assert_called_once_with("bin")
        assert namespace["solution"]["destination_preflight"]["ok"]
        assert namespace["record"]["_preferred_models"] == ["b"]


def test_navigation_preflight_chains_virtual_poses_without_moving_robot(monkeypatch):
    import torch
    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    method = next(n for n in ast.walk(ast.parse(source.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "_preflight_env_b_navigation_sequence")
    poses = [torch.tensor([1., 2., 0.]), torch.tensor([3., 4., 0.])]
    planner = Mock(side_effect=poses)
    monkeypatch.setitem(sys.modules, "run_deltasg_expert", SimpleNamespace(
        _connected_observation_pose=planner,
        _container_operation_point=lambda obj: None,
        _target_framing_distance=lambda *a, **kw: 1.25))
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    objects = {name: SimpleNamespace(name=name) for name in ("tool", "target")}
    robot = Mock()
    engine = SimpleNamespace(env=SimpleNamespace(robots=[robot], scene=SimpleNamespace(
        object_registry=lambda _, name: objects[name])))
    result = namespace[method.name](engine, ["tool", "target"])
    assert result["ok"] and result["robot_moved"] is False
    assert planner.call_args_list[0].kwargs["start_pose"] is None
    assert planner.call_args_list[1].kwargs["start_pose"] is poses[0]
    assert planner.call_args_list[1].kwargs["held_object"] is objects["tool"]
    robot.set_position_orientation.assert_not_called()


def test_navigation_preflight_uses_explicit_inventory_for_hand_wash(monkeypatch):
    import torch
    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    method = next(n for n in ast.walk(ast.parse(source.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "_preflight_env_b_navigation_sequence")
    planner = Mock(side_effect=[torch.tensor([1., 2., 0.]), torch.tensor([3., 4., 0.])])
    operation_point = torch.tensor([4., 5., 1.])
    monkeypatch.setitem(sys.modules, "run_deltasg_expert", SimpleNamespace(
        _connected_observation_pose=planner,
        _container_operation_point=lambda obj: operation_point,
        _target_framing_distance=lambda *a, **kw: 1.15))
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    objects = {name: SimpleNamespace(name=name) for name in ("dish", "sink")}
    engine = SimpleNamespace(env=SimpleNamespace(robots=[Mock()], scene=SimpleNamespace(
        object_registry=lambda _, name: objects[name])))
    result = namespace[method.name](engine, [
        {"target": "dish", "held_object": None},
        {"target": "sink", "held_object": "dish", "operation_relation": "inside"},
    ])
    assert result["approaches"][0]["held_object"] is None
    assert result["approaches"][1]["held_object"] == "dish"
    assert planner.call_args_list[0].kwargs["held_object"] is None
    assert planner.call_args_list[1].kwargs["held_object"] is objects["dish"]
    assert planner.call_args_list[1].kwargs["operation_target_position"] is operation_point
    assert planner.call_args_list[1].kwargs["max_operation_target_distance"] == 1.15


def test_hand_wash_tools_cannot_use_sink_as_their_support():
    source = (Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py").read_text()
    hand_wash = source[source.index('if recipe["path_name"] == "hand_wash":'):
                       source.index('if actual_category in {"fire_extinguisher"')]
    support_selection = source[source.index("def _choose_support_node"):
                               source.index("def _build_placement_for_support")]
    assert 'record["_excluded_support_ids"] = [sink_id]' in hand_wash
    assert 'node.get("id") in set(record.get("_excluded_support_ids") or [])' in support_selection


def test_exhausted_envb_slot_stops_before_robot_respawn():
    source = Path(__file__).resolve().parents[1] / "code" / "run_online_deltasg.py"
    tree = ast.parse(source.read_text())
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.BoolOp)
                 and "all requested Env-B tasks exhausted" in ast.unparse(n))
    function = ast.parse(
        "def exhausted(args, env_b_types, run_skip_tasks):\n"
        "    while True:\n        pass\n    return True\n"
    ).body[0]
    function.body[0].body = [guard, ast.Return(value=ast.Constant(False))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"ENV_B_TYPES": ("fire", "dirty_clothes"),
                 "ENV_B_TASK_NAMES": {"fire": "extinguish", "dirty_clothes": "collect"}}
    exec(compile(module, str(source), "exec"), namespace)
    check = namespace["exhausted"]
    assert check(SimpleNamespace(env_type="B"), ["dirty_clothes"], {"collect"})
    assert not check(SimpleNamespace(env_type="B"), None, {"collect"})
    assert check(SimpleNamespace(env_type="B"), None, {"collect", "extinguish"})
    assert not check(SimpleNamespace(env_type="A"), None, {"collect", "extinguish"})


def test_envb_clothing_cloth_mode_is_explicit_and_replay_preserves_type():
    code_dir = Path(__file__).resolve().parents[1] / "code"
    generator = (code_dir / "online_deltasg.py").read_text(encoding="utf-8")
    runner = (code_dir / "run_online_deltasg.py").read_text(encoding="utf-8")
    expert = (code_dir / "run_deltasg_expert.py").read_text(encoding="utf-8")
    batch = (code_dir / "run_envbc_multiscene_e2e.sh").read_text(encoding="utf-8")
    assert 'anomaly_record["_cloth_configuration"] = "crumpled"' in generator
    assert '"object_type": item.get("object_type") or "rigid"' in generator
    assert "gm.USE_GPU_DYNAMICS = args.allow_cloth" in runner
    assert "PrimType.CLOTH if is_cloth else PrimType.RIGID" in expert
    assert 'args.allow_cloth = True' not in runner
    assert '--env-b-types "$ENVB_TYPES" --allow-repeat-tasks' in batch


def test_broken_object_tools_are_graspable_interaction_tools():
    source = (
        Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    ).read_text(encoding="utf-8")
    assert 'actual_category in {"broom", "dustpan"}' in source
    assert '"fire_extinguisher", "sponge", "broom", "dustpan"' in source
    assert 'record["_prefer_support_first"] = True' in source
    assert 'record["_open_surface_only"] = True' in source


def test_envb_skipped_task_is_not_prepared_or_generated():
    import pytest
    from collections import Counter

    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    tree = ast.parse(source.read_text())
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name in {"generate_env_b", "prepare_env_b_robot_spawn"}]
    namespace = {"Counter": Counter, "ENV_B_TYPES": ("dirty_clothes",),
                 "ENV_B_TASK_NAMES": {"dirty_clothes": "collect_dirty_clothes"}}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    engine = SimpleNamespace(snapshot=lambda: {}, _checkpoint={},
                             _scene_model=lambda: "test_scene")
    skipped = {"collect_dirty_clothes"}
    assert namespace["prepare_env_b_robot_spawn"](engine, skip_tasks=skipped) is None
    assert engine._prepared_env_b_type is None
    with pytest.raises(RuntimeError, match="No requested Env-B"):
        namespace["generate_env_b"](engine, skip_tasks=skipped)


@pytest.mark.parametrize("changed,reached", [(False, False), (False, True), (True, False), (True, True)])
def test_inside_preflight_restores_scene_and_sampling_limits(monkeypatch, changed, reached):
    import torch
    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_preflight_env_b_inside")
    macros = SimpleNamespace(DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS=10,
                             DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS=20,
                             unlocked=nullcontext)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.object_state_utils",
                        SimpleNamespace(m=macros))
    sim = Mock()
    sim.dump_state.return_value = {"initial": "state"}
    states = SimpleNamespace(Open="Open", Inside="Inside")
    fallback = Mock(return_value={"ok": False})
    namespace = {"og": SimpleNamespace(sim=sim), "object_states": states,
                 "INSIDE_LOW_LEVEL_SAMPLING_ATTEMPTS": 10,
                 "place_inside_official_volume": fallback}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    relation = Mock()
    relation.set_value.return_value = changed
    relation.get_value.return_value = reached
    extent = SimpleNamespace(tolist=lambda: [0.1, 0.1, 0.1])
    obj = SimpleNamespace(name="dish", states={"Inside": relation}, aabb_extent=extent)
    container = SimpleNamespace(name="bin", states={}, links={
        "volume": SimpleNamespace(is_meta_link=True, meta_link_type="openfillable", aabb_extent=extent,
                                  visual_boundary_points_world=torch.tensor([[0., 0., 0.], [1., 1., 1.]]))})
    result = namespace[method.name](None, obj, container)
    assert result["ok"] is reached
    assert result["setter_return"] is changed
    assert result["predicate_after_set"] is reached
    assert result["sampling_attempts"] == {"high": 2, "low": 10}
    assert fallback.call_count == (0 if reached else 1)
    relation.clear_cache.assert_called_once()
    sim.load_state.assert_called_once_with({"initial": "state"}, serialized=False)
    assert macros.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS == 10
    assert macros.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS == 20
    sim.reset_mock()
    container.links = {}
    result = namespace[method.name](None, obj, container)
    assert result["reason"] == "missing_official_container_volume"
    sim.dump_state.assert_not_called()


def test_faucet_toggle_preflight_exercises_both_states_and_restores_scene():
    source = Path(__file__).resolve().parents[1] / "code" / "online_deltasg.py"
    method = next(n for n in ast.walk(ast.parse(source.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == "_preflight_env_b_toggle")
    sim = Mock()
    sim.dump_state.return_value = {"initial": "state"}
    toggled_on = object()
    state = Mock()
    state.get_value.side_effect = [False, True, False]
    state.set_value.side_effect = [True, True]
    obj = SimpleNamespace(name="sink", states={toggled_on: state})
    namespace = {"og": SimpleNamespace(sim=sim),
                 "object_states": SimpleNamespace(ToggledOn=toggled_on)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace[method.name](None, obj)
    assert result["ok"] is True
    assert state.set_value.call_args_list == [call(True), call(False)]
    sim.load_state.assert_called_once_with({"initial": "state"}, serialized=False)


def test_official_inside_volume_fallback_requires_real_support_and_stable_settling():
    source = (Path(__file__).resolve().parents[1] / "code" / "api.py").read_text()
    helper = source[source.index("def place_inside_official_volume"):
                    source.index("def validate_robot_stability")]
    assert "support_body" in helper
    assert 'float(hit_normal[2]) < 0.7' in helper
    assert "settled_center_displacement <= 0.03" in helper
    assert "reached and inside_bounds and settled_center_displacement" in helper
