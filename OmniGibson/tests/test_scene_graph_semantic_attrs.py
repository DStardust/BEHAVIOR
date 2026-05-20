import torch as th

from omnigibson import object_states
from omnigibson.scene_graphs.semantic_attrs import (
    infer_abnormal_states,
    infer_interaction_affordance,
    infer_object_edit_metadata,
    infer_receptacle_affordance,
)
from omnigibson.scene_graphs.task_asset_database import _infer_category_from_object_name


class FakeState:
    def __init__(self, value=False):
        self.value = value

    def get_value(self):
        return self.value


class FakeObject:
    def __init__(self, category, abilities=None, states=None, name=None, in_rooms=None):
        self.category = category
        self.name = name or category
        self.abilities = abilities or {}
        self.states = states or {}
        self.in_rooms = in_rooms
        self.model = "fake_model"


def test_receptacle_infers_on_top_and_inside_from_metadata():
    table = FakeObject("breakfast_table")
    table_meta = infer_receptacle_affordance(table, bbox_extent=th.tensor([1.2, 0.8, 0.4]))
    assert table_meta["can_support"]
    assert table_meta["supports_on_top"]
    assert not table_meta["supports_inside"]

    bowl = FakeObject("bowl", abilities={"fillable": {}})
    bowl_meta = infer_receptacle_affordance(bowl, bbox_extent=th.tensor([0.2, 0.2, 0.1]))
    assert bowl_meta["can_support"]
    assert bowl_meta["supports_inside"]


def test_interaction_is_inferred_without_movable_property():
    switch = FakeObject("light_switch", abilities={"toggleable": {}})
    assert infer_interaction_affordance(switch)["kind"] == "controllable"

    cabinet = FakeObject("bottom_cabinet", abilities={"openable": {}})
    assert infer_interaction_affordance(cabinet)["kind"] == "articulable"

    ceiling = FakeObject("ceiling", abilities={"sceneObject": {}})
    assert infer_interaction_affordance(ceiling)["kind"] == "none"


def test_abnormal_state_current_values_are_state_checked():
    candle = FakeObject(
        "beeswax_candle",
        abilities={"flammable": {}},
        states={object_states.OnFire: FakeState(True)},
    )
    meta = infer_abnormal_states(candle)
    assert "on_fire" in meta["potential"]
    assert meta["current"] == ["on_fire"]


def test_object_edit_metadata_keeps_graph_ready_shape():
    obj = FakeObject("cabinet", abilities={"openable": {}}, in_rooms=["kitchen_0"])
    meta = infer_object_edit_metadata(obj, bbox_extent=th.tensor([1.0, 0.5, 1.5]))
    assert meta["category"] == "cabinet"
    assert meta["rooms"] == ["kitchen_0"]
    assert meta["interaction"]["kind"] == "articulable"
    assert meta["receptacle"]["can_support"]
    assert "abnormal_states" in meta


def test_category_inference_prefers_longest_known_category_prefix():
    categories = ["cabinet", "bottom_cabinet", "metal_bottom_cabinet"]
    assert _infer_category_from_object_name("bottom_cabinet_nddvba_0", categories) == "bottom_cabinet"
