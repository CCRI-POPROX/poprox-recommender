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
