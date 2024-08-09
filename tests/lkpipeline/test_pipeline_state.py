from pytest import raises

from poprox_recommender.lkpipeline import PipelineState


def test_empty():
    state = PipelineState()
    assert len(state) == 0
    assert not state
    assert "scroll" not in state

    with raises(KeyError):
        state["scroll"]


def test_single_value():
    state = PipelineState({"scroll": "HACKEM MUCHE"})
    assert len(state) == 1
    assert state
    assert "scroll" in state
    assert state["scroll"] == "HACKEM MUCHE"


def test_alias():
    state = PipelineState({"scroll": "HACKEM MUCHE"}, {"book": "scroll"})
    assert len(state) == 1
    assert state
    assert "scroll" in state
    assert "book" in state
    assert state["book"] == "HACKEM MUCHE"


def test_alias_missing():
    state = PipelineState({"scroll": "HACKEM MUCHE"}, {"book": "manuscript"})
    assert len(state) == 1
    assert state
    assert "scroll" in state
    assert "book" not in state
