import logging
from typing import Any

from lenskit.pipeline import PipelineState

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.config import default_device

from .load import PipelineLoadError, load_all_pipelines

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineLoadError",
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
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    available_pipelines = load_all_pipelines(device=default_device())
    pipeline = available_pipelines["nrms"]

    if pipeline_params and pipeline_params.get("pipeline"):
        pipeline_name = pipeline_params["pipeline"]
        pipeline = available_pipelines[pipeline_name]

    recs = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (recs,)
    else:
        wanted = (topk, recs)

    return pipeline.run_all(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)
