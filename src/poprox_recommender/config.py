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


def available_cpu_parallelism(max: int | None = None) -> int:
    """
    Get the available amount of CPU parallelism.  If the ``POPROX_CPU_COUNT``
    environment variable is set, that is consulted; otherwise the number of CPUs
    detected on the system is returned.

    Args:
        max:
            The maximum # of CPUs (ignored if CPU count comes from the
            environment variable).

    Returns:
        # The number of CPUs to use for parallelism.
    """

    env_cpus = os.environ.get("POPROX_CPU_COUNT", None)
    if env_cpus is not None:
        return int(env_cpus)

    try:
        # linux allows cpu limits
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpus = os.cpu_count()
    if n_cpus is None:
        n_cpus = 1

    if max is not None and n_cpus > max:
        n_cpus = max

    return n_cpus


def allow_data_test_failures(what: Literal["models", "mind"] = "models") -> bool:
    "Whether to allow tests to fail because the DVC-managed data is missing."
    if "CI" in os.environ:
        if "PORPOX_CI_WITHOUT_MODELS" in os.environ:
            return True
        elif what == "mind" and "POPROX_CI_WITHOUT_MIND" in os.environ:
            return True
        else:
            return False

    return True
