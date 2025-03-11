"""
Functions and logic for loading recommender pipeline configurations.
"""

# pyright: basic
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from lenskit.pipeline import Pipeline, PipelineBuilder, PipelineCache
from structlog.stdlib import get_logger

from poprox_recommender.config import default_device

logger = get_logger(__name__)


_cached_pipelines: dict[str, Pipeline] = {}
_component_cache = PipelineCache()


class PipelineLoadError(Exception):
    """
    Exception raised when a pipeline cannot be loaded or instantiated, to
    separate those errors from errors running the pipeline.
    """


def discover_pipelines() -> list[str]:
    """
    Discover the list of available pipeline configuration names.
    """
    cfg_dir = Path(__file__).parent / "configurations"
    names = [p.stem for p in cfg_dir.glob("*.py") if not p.name.startswith("_")]
    logger.debug("scanned pipeline configurations", path=str(cfg_dir), count=len(names))
    return names


def get_pipeline_builder(name: str, device: str | None = None, num_slots: int = 10) -> PipelineBuilder:
    """
    Get a pipeline builder by name.
    """
    if device is None:
        device = default_device()

    # get a Python-compatible name
    norm_name = name.replace("-", "_")
    logger.debug("configuring pipeline", name=norm_name, device=device, num_slots=num_slots)
    # load the module
    mod_name = f"poprox_recommender.recommenders.configurations.{name}"
    pipe_mod = import_module(mod_name)

    pipe_ver = getattr(mod_name, "VERSION", None)
    if pipe_ver is None:
        pipe_ver = version("poprox-recommender")

    builder = PipelineBuilder(norm_name, pipe_ver)
    pipe_mod.configure(builder)
    return builder


def get_pipeline(name: str, device: str | None = None, num_slots: int = 10) -> Pipeline:
    """
    Get a built pipeline by name.
    """
    pipeline = _cached_pipelines.get(name, None)
    if pipeline is None:
        builder = get_pipeline_builder(name, device, num_slots)
        pipeline = builder.build(_component_cache)
        _cached_pipelines[name] = pipeline

    return pipeline


def load_all_pipelines(device: str | None = None, num_slots: int = 10) -> dict[str, Pipeline]:
    logger.debug("loading all pipelines")

    names = discover_pipelines()
    return {n: get_pipeline(n, device, num_slots) for n in names}
