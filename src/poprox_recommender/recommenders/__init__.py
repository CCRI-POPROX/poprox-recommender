# recommenders/__init__.py
import logging
from typing import Any

from lenskit.pipeline import PipelineState

from poprox_concepts.domain import CandidateSet, InterestProfile

from .load import (
    PipelineLoadError,
    default_pipeline,
    discover_pipelines,
    get_pipeline,
    get_pipeline_builder,
    load_all_pipelines,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineLoadError",
    "get_pipeline_builder",
    "get_pipeline",
    "discover_pipelines",
    "load_all_pipelines",
    "select_articles",
]


def select_articles(
    candidate_articles: CandidateSet,
    clicked_articles: CandidateSet,
    interest_profile: InterestProfile,
    pipeline_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration. It returns a
    pipeline state whose "image-selector" key contains the final list of recommendations
    with selected images.
    """
    name = None
    if pipeline_params and "pipeline" in pipeline_params:
        name = pipeline_params["pipeline"]
    if name is None:
        name = default_pipeline()

    pipeline = get_pipeline(name)

    recs = pipeline.node("recommender")
    image_selector = pipeline.node("image-selector")
    topk = pipeline.node("ranker", missing="none")

    if topk is None:
        wanted = (image_selector, recs)
    else:
        wanted = (image_selector, topk, recs)

    pipeline_state = pipeline.run_all(
        *wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile
    )
    logger.info(f"Pipeline state: {pipeline_state}")

    return pipeline_state
