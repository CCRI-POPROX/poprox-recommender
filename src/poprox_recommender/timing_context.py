"""
Context manager for tracking request timing and detecting potential timeouts.

This module provides utilities to track how long a request has been running
and detect when we're approaching Lambda timeout limits.
"""

import os
import time
from contextvars import ContextVar
from typing import Optional

# Context variable to store request start time
_request_start_time: ContextVar[Optional[float]] = ContextVar("request_start_time", default=None)

# Lambda timeout from environment or default to 30 seconds
LAMBDA_TIMEOUT_SECONDS = int(os.getenv("AWS_LAMBDA_FUNCTION_TIMEOUT", "30"))

# Buffer before timeout to consider as "at risk" (5 seconds)
TIMEOUT_RISK_BUFFER_SECONDS = 5


def set_request_start_time(start_time: Optional[float] = None) -> None:
    """
    Set the start time for the current request.

    Args:
        start_time: Unix timestamp of when request started. If None, uses current time.
    """
    if start_time is None:
        start_time = time.time()
    _request_start_time.set(start_time)


def get_request_start_time() -> Optional[float]:
    """
    Get the start time for the current request.

    Returns:
        Unix timestamp of when request started, or None if not set.
    """
    return _request_start_time.get()


def get_elapsed_seconds() -> Optional[float]:
    """
    Get the number of seconds elapsed since request started.

    Returns:
        Seconds elapsed, or None if request start time not set.
    """
    start_time = get_request_start_time()
    if start_time is None:
        return None
    return time.time() - start_time


def get_remaining_seconds() -> Optional[float]:
    """
    Get the number of seconds remaining before Lambda timeout.

    Returns:
        Seconds remaining, or None if request start time not set.
    """
    elapsed = get_elapsed_seconds()
    if elapsed is None:
        return None
    return LAMBDA_TIMEOUT_SECONDS - elapsed


def is_approaching_timeout(buffer_seconds: float = TIMEOUT_RISK_BUFFER_SECONDS) -> bool:
    """
    Check if we're approaching the Lambda timeout.

    Args:
        buffer_seconds: Number of seconds before timeout to consider "approaching".

    Returns:
        True if we're within buffer_seconds of timing out, False otherwise.
    """
    remaining = get_remaining_seconds()
    if remaining is None:
        return False
    return remaining <= buffer_seconds


def get_timeout_risk_info() -> dict:
    """
    Get information about timeout risk for the current request.

    Returns:
        Dictionary with timeout risk information including:
        - elapsed_seconds: How long the request has been running
        - remaining_seconds: How much time is left before timeout
        - timeout_limit_seconds: The Lambda timeout limit
        - is_at_risk: Whether we're approaching timeout
        - timeout_risk_percentage: What percentage of timeout has elapsed (0-100)
    """
    elapsed = get_elapsed_seconds()
    remaining = get_remaining_seconds()

    if elapsed is None or remaining is None:
        return {
            "elapsed_seconds": None,
            "remaining_seconds": None,
            "timeout_limit_seconds": LAMBDA_TIMEOUT_SECONDS,
            "is_at_risk": False,
            "timeout_risk_percentage": 0,
        }

    percentage = (elapsed / LAMBDA_TIMEOUT_SECONDS) * 100

    return {
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining,
        "timeout_limit_seconds": LAMBDA_TIMEOUT_SECONDS,
        "is_at_risk": is_approaching_timeout(),
        "timeout_risk_percentage": percentage,
    }


def clear_request_context() -> None:
    """Clear the request timing context."""
    _request_start_time.set(None)
