# pyright: strict
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import overload

logger = logging.getLogger(__name__)
_cached_root: Path | None = None


@overload
def project_root() -> Path: ...
@overload
def project_root(*, require: bool) -> Path | None: ...
def project_root(*, require: bool = True) -> Path | None:
    """
    Find the project root directory (when we are running in the project).

    This searches upwards from the **current working directory** to find the
    root of the project, which it identifies by the ``pyproject.toml`` file.  If
    this function is called from a directory that is not within a checkout of
    the ``poprox-recommender`` repository, it will raise an error.

    Args:
        require:
            Whether to fail when the project root is not found, or return
            ``None``. If ``require=False`` this function will stil fail on a
            *defective* project root (contains an invalid ``pyproject.toml``).

    Returns:
        The full path to the project root directory.  If the project root is
        not found and ``require=False``, returns ``None``.
    """
    global _cached_root
    if _cached_root is None:
        cwd = Path(".").resolve()
        candidate = cwd
        logger.debug("searching for project root upwards from %s", candidate)
        while not _is_project_root(candidate):
            candidate = candidate.parent
            if not candidate or str(candidate) == "/":
                if require:
                    msg = f"cannot find project root for {cwd}"
                    raise RuntimeError(msg)
                else:
                    # don't cache None
                    return None

        logger.debug("found project root at  %s", candidate)
        _cached_root = candidate

    return _cached_root


def model_file_path(name: str) -> Path:
    """
    Get the path of a model file.  It looks in the following locations, in
    order:

    * The path specified by the ``POPROX_MODELS`` environment variable.
    * The ``models`` directory under the :func:`project_root`.
    * ``$CONDA_PREFIX/models`` (if env var ``CONDA_PREFIX`` is defined, which is
      done by ``conda activate``).

    Args:
        name: The path to the model file (or directory), relative to ``models``.

    Returns:
        The full path to the model file, if it exists.
    """
    model_dirs: list[Path] = []
    if "POPROX_MODELS" in os.environ:
        model_dirs.append(Path(os.environ["POPROX_MODELS"]))
    root = project_root(require=False)
    if root is not None:
        model_dirs.append(root / "models")
    if "CONDA_PREFIX" in os.environ:
        model_dirs.append(Path(os.environ["CONDA_PREFIX"]) / "models")

    if not model_dirs:
        msg = "no model directories found"
        raise RuntimeError(msg)

    for md in model_dirs:
        logger.debug("looking for %s in %s", name, md)
        mf = md / name
        if mf.exists():
            logger.info("resolved %s: %s", name, mf)
            return mf

    logger.error("could not find model file %s in any of %d locations", name, len(model_dirs))
    msg = f"model file {name}"
    raise FileNotFoundError(msg)


def _is_project_root(path: Path) -> bool:
    tomlf = path / "pyproject.toml"
    if tomlf.exists():
        return True
    else:
        return False
