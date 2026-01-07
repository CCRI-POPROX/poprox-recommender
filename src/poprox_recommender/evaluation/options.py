from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def load_eval_options(path: Path) -> EvalOptionRoot:
    """
    Load evaluation options from a path (or its parent directory).

    Args:
        path:
            The path to the eval options.

    Returns:
        The evaluation options.
    """
    if not path.is_dir():
        raise RuntimeError(f"{path} is not a directory")

    for opt_file in [path / "options.toml", path.parent / "options.toml"]:
        logger.debug("checking for %s", opt_file)
        if opt_file.exists():
            logger.info("loading eval options from %s", opt_file)
            with opt_file.open("rb") as f:
                data = tomllib.load(f)
            return EvalOptionRoot.model_validate(data)

    logger.info("no configuration file found, using defaults")
    return EvalOptionRoot()


class EvalOptions(BaseModel):
    """
    Model class for configuring measurement.
    """

    print_effectiveness: bool = True
    "Whether or not to print effectiveness metrics."


class EvalOptionRoot(BaseModel):
    "Root schema of the evaluation options."

    evaluate: EvalOptions = EvalOptions()
