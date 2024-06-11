import logging
import os
from pathlib import Path
import tomllib

_logger = logging.getLogger(__name__)


def project_root(require: bool = True) -> Path | None:
    """
    Find the project root directory (when we are running in the project).

    This searches upwards from the **current working directory** to find the
    root of the project, which it identifies by the presence of the
    `pyproject.toml` file.  If this function is called from a directory that is
    not within a checkout of the ``poprox-recommender`` repository , it will
    raise an error.

    Args:
        require:
            Whether to fail when the project root is not found, or return
            ``None``. If ``require=False`` this function will stil fail on a
            *defective* project root (it contains an invalid
            ``pyproject.toml``).
    """
    cwd = Path(".").resolve()
    candidate = cwd
    while not _is_project_root(candidate):
        candidate = candidate.parent
        if not candidate:
            if require:
                msg = f"cannot find project root for {cwd}"
                raise RuntimeError(msg)
            else:
                return None

    return candidate


def model_file_path(name: str) -> Path:
    """
    Get the path of a model file.  It looks in the following locations, in order:

    * The path specified by the ``POPROX_MODELS`` directory.
    * The ``models`` directory under the :func:`project_root`.
    * ``$CONDA_PREFIX/models`` (if ``CONDA_PREFIX`` is defined).
    """
    model_dirs = []
    if "POPROX_MODELS" in os.environ:
        model_dirs.append(Path(os.environ["POPROX_MODELS"]))
    root = project_root(False)
    if root is not None:
        model_dirs.append(root / "models")
    if "CONDA_PREFIX" in os.environ:
        model_dirs.append(Path(os.environ["CONDA_PREFIX"]) / "models")

    if not model_dirs:
        msg = "no model directories found"
        raise RuntimeError(msg)

    for md in model_dirs:
        _logger.debug("looking for %s in %s", name, md)
        mf = md / name
        if mf.exists():
            _logger.info("resolved %s: %s", name, mf)
            return mf

    msg = f"model file {name}"
    raise FileNotFoundError(msg)


def _is_project_root(path: Path) -> bool:
    tomlf = path / "pyproject.toml"
    if tomlf.exists():
        # we found the root, but double-check
        ppt = tomllib.loads(tomlf.read_text())
        # bad pyproject.toml indicates something *deeply* wrong so fail instead of returning false
        try:
            proj = ppt["project"]
        except AttributeError as e:
            msg = f"{ppt} has no project definition"
            raise RuntimeError(msg) from e
        if proj.get("name", None) != "poprox-recommender":
            msg = f"found {ppt}, but it is not for poprox-recommender, wrong working directory?"
            raise RuntimeError(msg)
        # got this far, all good!
        return True
    else:
        return False
