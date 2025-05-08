"""
Access to configuration settings for the POPROX recommender pipeline.
"""

import os
from typing import Literal


def default_device() -> str:
    """
    Get the default device for POPROX components.  This is read from the
    ``POPROX_REC_DEVICE`` environment variable, if it exists; otherwise it
    selects ``cuda`` if it is available and ``cpu`` otherwise.
    """
    import torch

    configured = os.environ.get("POPROX_REC_DEVICE", None)

    if configured:
        return configured
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def allow_data_test_failures(what: Literal["models", "mind"] = "models") -> bool:
    "Whether to allow tests to fail because the DVC-managed data is missing."
    if "CI" in os.environ:
        if "POPROX_CI_WITHOUT_MODELS" in os.environ:
            return True
        elif what == "mind" and "POPROX_CI_WITHOUT_MIND" in os.environ:
            return True
        else:
            return False

    return True
