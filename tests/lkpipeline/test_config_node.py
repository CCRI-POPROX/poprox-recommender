from poprox_recommender.lkpipeline.config import PipelineInput
from poprox_recommender.lkpipeline.nodes import InputNode


def test_untyped_input():
    node = InputNode("scroll")

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types is None


def test_input_with_type():
    node = InputNode("scroll", types={str})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"str"}


def test_input_with_none():
    node = InputNode("scroll", types={str, type(None)})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"None", "str"}


def test_input_with_generic():
    node = InputNode("scroll", types={list[str]})

    cfg = PipelineInput.from_node(node)
    print(cfg)
    assert cfg.name == "scroll"
    assert cfg.types == {"list"}
