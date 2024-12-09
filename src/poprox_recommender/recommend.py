# pyright: basic
import logging
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.config import default_device
from poprox_recommender.lkpipeline import Pipeline, PipelineState
from poprox_recommender.recommenders import (
    locality_calibration_pipeline,
    nrms_mmr_pipeline,
    nrms_pfar_pipeline,
    nrms_pipeline,
    nrms_softmax_pipeline,
    topic_calibration_pipeline,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_cached_pipelines = None


class PipelineLoadError(Exception):
    """
    Exception raised when a pipeline cannot be loaded or instantiated, to
    separate those errors from errors running the pipeline.
    """


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    pipeline_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    available_pipelines = recommendation_pipelines(device=default_device())
    pipeline = available_pipelines["nrms"]

    if pipeline_params and "pipeline" in pipeline_params:
        pipeline_name = pipeline_params["pipeline"]
        pipeline = available_pipelines[pipeline_name]

    recs = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (recs,)
    else:
        wanted = (topk, recs)

    return pipeline.run_all(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)


def recommendation_pipelines(device: str | None = None, num_slots: int = 10) -> dict[str, Pipeline]:
    global _cached_pipelines
    if device is None:
        device = default_device()
    logger.debug("loading pipeline components on device %s", device)

    if _cached_pipelines is None:
        try:
            _cached_pipelines = build_pipelines(num_slots=num_slots, device=device)
        except Exception as e:
            e.add_note("did you remember to `dvc pull`?")
            raise PipelineLoadError("could not instantiate pipelines", e)

    return _cached_pipelines


def build_pipelines(num_slots: int, device: str) -> dict[str, Pipeline]:
    """
    Create the default personalized recommendation pipeline.

    Args:
        num_slots: The number of items to recommend.
    """

    return {
        "nrms": nrms_pipeline(num_slots, device),
        "mmr": nrms_mmr_pipeline(num_slots, device),
        "pfar": nrms_pfar_pipeline(num_slots, device),
        "topic-cali": topic_calibration_pipeline(num_slots, device),
        "locality-cali": locality_calibration_pipeline(num_slots, device),
        "softmax": nrms_softmax_pipeline(num_slots, device),
    }
