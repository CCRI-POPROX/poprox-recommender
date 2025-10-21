import logging
from typing import Any

from poprox_concepts.domain import (
    CandidateSet,
    ImpressedRecommendations,
    InterestProfile,
)

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
) -> tuple[ImpressedRecommendations, Any]:
    """
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    name = None
    if pipeline_params and "pipeline" in pipeline_params:
        name = pipeline_params["pipeline"]

    if name is None:
        name = default_pipeline()

    pipeline = get_pipeline(name)

    recs = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (recs,)
    else:
        wanted = (topk, recs)

    outputs = pipeline.run_all(
        *wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile
    )

    return outputs.default, outputs.meta
