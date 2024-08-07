# pyright: basic
import logging
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers import MMRDiversifier, PFARDiversifier, TopicCalibrator
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.config import default_device
from poprox_recommender.paths import model_file_path
from poprox_recommender.pipeline import PipelineState, RecommendationPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_cached_pipelines = None


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    algo_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration.
    """
    available_pipelines = recommendation_pipelines(device=default_device())
    pipeline = available_pipelines["nrms"]

    if algo_params and "diversity_algo" in algo_params:
        diversifier = algo_params["diversity_algo"]
        pipeline = available_pipelines[diversifier]

    inputs = {
        "candidate": candidate_articles,
        "clicked": clicked_articles,
        "profile": interest_profile,
    }

    return pipeline(inputs)


def recommendation_pipelines(device=None, num_slots=10) -> dict[str, RecommendationPipeline]:
    global _cached_pipelines
    if device is None:
        device = default_device()
    logger.debug("loading pipeline components on device %s", device)

    if _cached_pipelines is None:
        _cached_pipelines = build_pipelines(num_slots=num_slots, device=device)

    return _cached_pipelines


def build_pipelines(num_slots: int, device: str) -> dict[str, RecommendationPipeline]:
    """
    Create the default personalized recommendation pipeline.

    Args:
        num_slots: The number of items to recommend.
    """

    article_embedder = NRMSArticleEmbedder(model_file_path("news_encoder.safetensors"), device)
    user_embedder = NRMSUserEmbedder(model_file_path("user_encoder.safetensors"), device)

    topk_ranker = TopkRanker(num_slots=num_slots)
    mmr = MMRDiversifier(num_slots=num_slots)
    pfar = PFARDiversifier(num_slots=num_slots)
    calibrator = TopicCalibrator(num_slots=num_slots)

    nrms_pipe = build_pipeline(
        "plain-NRMS",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=topk_ranker,
        num_slots=num_slots,
    )

    mmr_pipe = build_pipeline(
        "NRMS+MMR",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=mmr,
        num_slots=num_slots,
    )

    pfar_pipe = build_pipeline(
        "NRMS+PFAR",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=pfar,
        num_slots=num_slots,
    )

    cali_pipe = build_pipeline(
        "NRMS+Calibration",
        article_embedder=article_embedder,
        user_embedder=user_embedder,
        ranker=calibrator,
        num_slots=num_slots,
    )

    return {"nrms": nrms_pipe, "mmr": mmr_pipe, "pfar": pfar_pipe, "topic-cali": cali_pipe}


def build_pipeline(name, article_embedder, user_embedder, ranker, num_slots):
    article_scorer = ArticleScorer()
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)
    fill = Fill(num_slots=num_slots)
    topk_ranker = TopkRanker(num_slots=num_slots)

    pipe = RecommendationPipeline(name=name)

    # Compute embeddings
    pipe.add(article_embedder, inputs=["candidate"], output="candidate")
    pipe.add(article_embedder, inputs=["clicked"], output="clicked")
    pipe.add(user_embedder, inputs=["clicked", "profile"], output="profile")

    # Score and rank articles with diversification/calibration reranking
    pipe.add(article_scorer, inputs=["candidate", "profile"], output="candidate")
    pipe.add(ranker, inputs=["candidate", "profile"], output="reranked")

    # Output the plain descending-by-score ranking for comparison
    pipe.add(topk_ranker, inputs=["candidate", "profile"], output="ranked")

    # Fallback in case not enough articles came from the ranker
    pipe.add(topic_filter, inputs=["candidate", "profile"], output="topical")
    pipe.add(sampler, inputs=["topical", "candidate"], output="sampled")
    pipe.add(fill, inputs=["reranked", "sampled"], output="recs")

    return pipe
