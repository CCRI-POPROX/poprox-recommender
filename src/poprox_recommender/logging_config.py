"""
Logging configuration logic for CLI tools in the POPROX recommenders.
"""

# pyright: strict
import logging
import os
import sys
from pathlib import Path

from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)


def setup_logging(*, verbose: bool | None = False, log_file: str | Path | None = None):
    """
    Set up the Python logging infrastructure.

    If the configuration options are not passed, it checks the environment
    variables ``POPROX_LOG_VERBOSE`` and ``POPROX_LOG_FILE``.

    Args:
        verbose:
            If ``True``, include DEBUG output on the console.
        log_file:
            A log file to record logging output (in addition to the console).
    """
    if verbose is None and "POPROX_LOG_VERBOSE" in os.environ:
        verbose = True

    if log_file is None:
        log_file = os.environ.get("POPROX_LOG_FILE", None)

    # determine level for terminal
    term_level = logging.DEBUG if verbose else logging.INFO
    # determine level for root logger — need DEBUG if anything wants debug
    root_level = logging.DEBUG if verbose or log_file else logging.INFO

    # root logging level based on most verbose level needed for any output
    root = logging.getLogger()
    root.setLevel(root_level)

    # numba's debug logs are noisy and not very useful
    # we don't use numba yet but in case we add deps that use it
    logging.getLogger("numba").setLevel(logging.INFO)

    # set up the terminal logging
    term_h = logging.StreamHandler(sys.stderr)
    term_h.setLevel(term_level)

    term_fmt = ColoredFormatter(
        "[%(blue)s%(asctime)s%(reset)s] %(log_color)s%(levelname)8s%(reset)s %(cyan)s%(name)s%(reset)s %(message)s",  # noqa: E501
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
        # options to detect whether to colorize
        stream=sys.stderr,
        no_color=os.environ.get("NO_COLOR", "") != "",
        force_color=os.environ.get("FORCE_COLOR", "") != "",
    )
    term_h.setFormatter(term_fmt)

    root.addHandler(term_h)

    # set up the log file (if any)
    if log_file is not None:
        logger.debug("copying logs to %s", log_file)
        file_h = logging.FileHandler(log_file, "w")
        file_h.setLevel(logging.DEBUG)

        file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        file_h.setFormatter(file_fmt)

        root.addHandler(file_h)

    logger.info("logging initialized")
