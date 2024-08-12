"""
Access to configuration settings for the POPROX recommender pipeline.
"""

import os


def default_device() -> str:
    """
    Get the default device for POPROX components.  This is read from the
    ``POPROX_REC_DEVICE`` environment variable, and defaults to ``cpu`` if that
    variable is not set.
    """
    return os.environ.get("POPROX_REC_DEVICE", "cpu")


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

    if "POPROX_CPU_COUNT" in os.environ:
        return int(os.environ("POPROX_CPU_COUNT"))

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
