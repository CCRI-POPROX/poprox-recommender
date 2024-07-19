"""
Decorators to help with Torch usage.
"""

# pyright: strict
from __future__ import annotations

from typing import Callable, ParamSpec, TypeVar

import torch

P = ParamSpec("P")
R = TypeVar("R")


def torch_inference(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator for functions or methods that use PyTorch for inference (an thus
    :func:`torch.inference_mode` should be set for performance). It wraps the
    function in the following logic::

        with torch.inference_mode():
            func()

    Args:
        func: the function to wrap.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper
