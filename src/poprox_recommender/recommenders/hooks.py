"""
POPROX recommendation hooks.
"""

from typing import Any

from lenskit.pipeline.nodes import ComponentInstanceNode
from pydantic import BaseModel


def shallow_copy_pydantic_model(
    node: ComponentInstanceNode[Any], input_name: str, input_type: Any, value: Any, **context: Any
) -> Any:
    """
    LensKit component-input hook to perform shallow copies of Pydantic models
    before they are passed to components.
    """
    if isinstance(value, BaseModel):
        return value.model_copy()
    else:
        return value
