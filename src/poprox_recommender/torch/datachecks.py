"""
Data-checking routines.
"""

import torch as th

__all__ = ["assert_tensor_size"]


def assert_tensor_size(tensor: th.Tensor, *size: int, label: str | None = None, prefix: bool = True):
    """
    Check that a tensor is of the expected size, and fail with an assertion
    error if it is the incorrect size.

    Args:
        tensor:
            The tensor to check.
        *size:
            The expected size of the tensor.
        label:
            A label for the tensor.
        prefix:
            If ``True`` and the tensor has more dimensions than the expected
            size, only match the common prefix of dimensions.
    """
    actual = tensor.shape
    expected = th.Size(size)
    if prefix and len(actual) > len(expected):
        actual = actual[: len(expected)]
    if actual != expected:
        msg = f"unexpected tensor size {_size_string(tensor.shape)} (expected {_size_string(expected)})"
        if label:
            msg = f"{label}: {msg}"
        raise AssertionError(msg)


def _size_string(size: th.Size) -> str:
    return " ⨉ ".join(str(d) for d in size)
