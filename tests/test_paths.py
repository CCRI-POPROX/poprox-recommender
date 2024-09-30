import logging
import os
import platform
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import mark, skip

from poprox_recommender.paths import model_file_path, project_root

logger = logging.getLogger(__name__)


def test_project_root():
    # we only ever run tests from within the git repo
    root = project_root()
    logger.info("project root: %s", root)
    assert root.exists()
    assert (root / "pyproject.toml").exists
    me = root / "tests"
    assert me == Path(__file__).parent


def test_module_file_path_project_root():
    old = os.environ.get("POPROX_MODELS", None)
    try:
        if old:
            del os.environ["POPROX_MODELS"]
        root = project_root()
        if not (root / "models" / "model.safetensors").exists():
            warnings.warn("model source file not available")
            skip("model data not available")

        mfile = model_file_path("model.safetensors")

        assert mfile == root / "models" / "model.safetensors"
    finally:
        if old:
            os.environ["POPROX_MODELS"] = old


@mark.skipif(platform.system() == "Windows", reason="hard-coded paths are unixy")
def test_module_file_path_env_dir():
    old = os.environ.get("POPROX_MODELS", None)
    try:
        with TemporaryDirectory() as tdir:
            td_path = Path(tdir)
            os.environ["POPROX_MODELS"] = tdir
            root = project_root()

            logger.info("copying model file")
            model_src = root / "models" / "model.safetensors"
            if not model_src.exists():
                warnings.warn("model source file not available")
                skip("model data not available")

            shutil.copy(model_src, td_path / "model.safetensors")

            mfile = model_file_path("model.safetensors")

            assert mfile == td_path / "model.safetensors"
    finally:
        if old:
            os.environ["POPROX_MODELS"] = old
        else:
            del os.environ["POPROX_MODELS"]
