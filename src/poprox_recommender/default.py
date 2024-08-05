# pyright: basic
import logging
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers import MMRDiversifier, PFARDiversifier, TopicCalibrator
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.config import default_device
from poprox_recommender.lkpipeline import Pipeline, PipelineState
from poprox_recommender.paths import model_file_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    num_slots: int,
    pipeline_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    pipeline = None

    pipeline = personalized_pipeline(num_slots, pipeline_params)
    assert pipeline is not None

    recs = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (recs,)
    else:
        wanted = (topk, recs)

    return pipeline.run_all(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)


def personalized_pipeline(num_slots: int, pipeline_params: dict[str, Any] | None = None) -> Pipeline | None:
    """
    Create the default personalized recommendation pipeline.

    Args:
        num_slots: The number of items to recommend.
        pipeline_params: Additional parameters to the reocmmender algorithm.
    """
    if not pipeline_params:
        pipeline_params = {}

    if "pipeline" in pipeline_params:
        pipeline_name = pipeline_params["pipeline"]
        del pipeline_params["pipeline"]
    else:
        pipeline_name = None

    article_embedder = NRMSArticleEmbedder(model_file_path("news_encoder.safetensors"), default_device())
    user_embedder = NRMSUserEmbedder(model_file_path("user_encoder.safetensors"), default_device())
    article_scorer = ArticleScorer()
    topk_ranker = TopkRanker(algo_params={}, num_slots=num_slots)
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)
    fill = Fill(num_slots=num_slots)

    if pipeline_name == "mmr":
        logger.info("Recommendations will be re-ranked with mmr.")
        ranker = MMRDiversifier(pipeline_params, num_slots)
    elif pipeline_name == "pfar":
        logger.info("Recommendations will be re-ranked with pfar.")
        ranker = PFARDiversifier(pipeline_params, num_slots)
    elif pipeline_name == "topic-cali":
        logger.info("Recommendations will be re-ranked with topic calibration.")
        ranker = TopicCalibrator(pipeline_params, num_slots)
    else:
        logger.info("Recommendations will be ranked with plain top-k.")
        ranker = topk_ranker

    # TODO put pipeline name back in
    pipeline = Pipeline()

    # Define pipeline inputs
    candidates = pipeline.create_input("candidate", ArticleSet)
    clicked = pipeline.create_input("clicked", ArticleSet)
    profile = pipeline.create_input("profile", InterestProfile)

    # Compute embeddings
    e_cand = pipeline.add_component("candidate-embedder", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("history-emberdder", article_embedder, article_set=clicked)
    e_user = pipeline.add_component("user-embedder", user_embedder, clicked_articles=e_click, interest_profile=profile)

    # Score and rank articles with diversification/calibration reranking
    o_scored = pipeline.add_component("scorer", article_scorer, candidate_articles=e_cand, interest_profile=e_user)
    o_topk = pipeline.add_component("ranker", topk_ranker, candidate_articles=o_scored, interest_profile=e_user)
    if ranker is topk_ranker:
        o_rank = o_topk
    else:
        o_rank = pipeline.add_component("reranker", ranker, candidate_articles=o_scored, interest_profile=e_user)

    # Fallback in case not enough articles came from the ranker
    # TODO: make this lazy so the sampler only runs if the reranker isn't enough
    o_filtered = pipeline.add_component("topic-filter", topic_filter, candidate=candidates, interest_profile=profile)
    o_sampled = pipeline.add_component("sampler", sampler, candidate=o_filtered, backup=candidates)
    pipeline.add_component("recommender", fill, candidates1=o_rank, candidates2=o_sampled)

    return pipeline
