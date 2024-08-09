"""
Pydantic models for pipeline configuration and serialization support.
"""

# pyright: strict
from __future__ import annotations

import base64
import pickle
from collections import OrderedDict
from hashlib import sha256
from types import FunctionType
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, ValidationError
from typing_extensions import Any, Optional, Self

from .components import ConfigurableComponent
from .nodes import ComponentNode, InputNode
from .types import type_string


class PipelineConfig(BaseModel):
    """
    Root type for serialized pipeline configuration.  A pipeline config contains
    the full configuration, components, and wiring for the pipeline, but does
    not contain the
    """

    meta: PipelineMeta
    "Pipeline metadata."
    inputs: list[PipelineInput] = Field(default_factory=list)
    "Pipeline inputs."
    defaults: dict[str, str] = Field(default_factory=dict)
    "Default pipeline wirings."
    components: OrderedDict[str, PipelineComponent] = Field(default_factory=OrderedDict)
    "Pipeline components, with their configurations and wiring."
    aliases: dict[str, str] = Field(default_factory=dict)
    "Pipeline node aliases."
    literals: dict[str, PipelineLiteral] = Field(default_factory=dict)
    "Literals"


class PipelineMeta(BaseModel):
    """
    Pipeline metadata.
    """

    name: str | None = None
    "The pipeline name."
    version: str | None = None
    "The pipeline version."
    hash: str | None = None
    """
    The pipeline configuration hash.  This is optional, particularly when
    hand-crafting pipeline configuration files.
    """


class PipelineInput(BaseModel):
    name: str
    "The name for this input."
    types: Optional[set[str]]
    "The list of types for this input."

    @classmethod
    def from_node(cls, node: InputNode[Any]) -> Self:
        if node.types is not None:
            types = {type_string(t) for t in node.types}
        else:
            types = None

        return cls(name=node.name, types=types)


class PipelineComponent(BaseModel):
    code: str
    """
    The path to the component's implementation, either a class or a function.
    This is a Python qualified path of the form ``module:name``.

    Special nodes, like :class:`lenskit.pipeline.Pipeline.use_first_of`, are
    serialized as components whose code is a magic name beginning with ``@``
    (e.g. ``@use-first-of``).
    """

    config: dict[str, object] | None = Field(default=None)
    """
    The component configuration.  If not provided, the component will be created
    with its default constructor parameters.
    """

    inputs: dict[str, str] | list[str] = Field(default_factory=dict)
    """
    The component's input wirings, mapping input names to node names.  For
    certain meta-nodes, it is specified as a list instead of a dict.
    """

    @classmethod
    def from_node(cls, node: ComponentNode[Any], mapping: dict[str, str] | None = None) -> Self:
        if mapping is None:
            mapping = {}

        comp = node.component
        if isinstance(comp, FunctionType):
            ctype = comp
        else:
            ctype = comp.__class__

        code = f"{ctype.__module__}:{ctype.__qualname__}"

        config = comp.get_config() if isinstance(comp, ConfigurableComponent) else None

        return cls(
            code=code,
            config=config,
            inputs={n: mapping.get(t, t) for (n, t) in node.connections.items()},
        )


class PipelineLiteral(BaseModel):
    """
    Literal nodes represented in the pipeline.
    """

    encoding: Literal["json", "base85"]
    value: JsonValue

    @classmethod
    def represent(cls, data: Any) -> Self:
        try:
            return cls(encoding="json", value=data)
        except ValidationError:
            # data is not basic JSON values, so let's pickle it
            dbytes = pickle.dumps(data)
            return cls(encoding="base85", value=base64.b85encode(dbytes).decode("ascii"))

    def decode(self) -> Any:
        "Decode the represented literal."
        match self.encoding:
            case "json":
                return self.value
            case "base85":
                assert isinstance(self.value, str)
                return pickle.loads(base64.b85decode(self.value))


def hash_config(config: BaseModel) -> str:
    """
    Compute the hash of a configuration model.
    """
    json = config.model_dump_json(exclude_none=True)
    h = sha256()
    h.update(json.encode())
    return h.hexdigest()
