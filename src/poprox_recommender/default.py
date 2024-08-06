# pyright: basic
import logging
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers import MMRDiversifier, PFARDiversifier, TopicCalibrator
from poprox_recommender.components.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.lkpipeline import Pipeline, PipelineState
from poprox_recommender.model import get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> PipelineState:
    """
    Select articles with default recommender configuration.  It returns a
    pipeline state whose ``default`` is the final list of recommendations.
    """
    pipeline = None

    if interest_profile.click_history.article_ids:
        pipeline = personalized_pipeline(num_slots, algo_params)

    if pipeline is None:
        pipeline = fallback_pipeline(num_slots)

    rank = pipeline.node("recommender")
    topk = pipeline.node("ranker", missing="none")
    if topk is None:
        wanted = (rank,)
    else:
        wanted = (topk, rank)

    return pipeline.run_all(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)


def personalized_pipeline(num_slots: int, algo_params: dict[str, Any] | None = None) -> Pipeline | None:
    """
    Create the default personalized recommendation pipeline.

    Args:
        num_slots: The number of items to recommend.
        algo_params: Additional parameters to the reocmmender algorithm.
    """
    model = get_model()
    if model is None:
        return None

    if not algo_params:
        algo_params = {}

    if "diversity_algo" in algo_params:
        diversify = algo_params["diversity_algo"]
        del algo_params["diversity_algo"]
    else:
        diversify = None

    article_embedder = ArticleEmbedder(model.model, model.tokenizer, model.device)
    user_embedder = UserEmbedder(model.model, model.device)
    article_scorer = ArticleScorer(model.model)
    topk_ranker = TopkRanker(algo_params={}, num_slots=num_slots)

    if diversify == "mmr":
        logger.info("Recommendations will be re-ranked with mmr.")
        ranker = MMRDiversifier(algo_params, num_slots)
    elif diversify == "pfar":
        logger.info("Recommendations will be re-ranked with pfar.")
        ranker = PFARDiversifier(algo_params, num_slots)
    elif diversify == "topic-cali":
        logger.info("Recommendations will be re-ranked with topic calibration.")
        ranker = TopicCalibrator(algo_params, num_slots)
    else:
        logger.info("Recommendations will be ranked with plain top-k.")
        ranker = topk_ranker

    pipeline = Pipeline()
    candidates = pipeline.create_input("candidate", ArticleSet)
    clicked = pipeline.create_input("clicked", ArticleSet)
    profile = pipeline.create_input("profile", InterestProfile)
    e_cand = pipeline.add_component("candidate-embedder", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("history-emberdder", article_embedder, article_set=clicked)
    e_user = pipeline.add_component("user-embedder", user_embedder, clicked_articles=e_click, interest_profile=profile)
    scored = pipeline.add_component("scorer", article_scorer, candidate_articles=e_cand, interest_profile=e_user)
    topk = pipeline.add_component("ranker", topk_ranker, candidate_articles=scored, interest_profile=e_user)
    if ranker is topk_ranker:
        pipeline.alias("recommender", topk)
    else:
        rerank = pipeline.add_component("reranker", ranker, candidate_articles=scored, interest_profile=e_user)
        pipeline.alias("recommender", rerank)

    return pipeline


def fallback_pipeline(num_slots: int) -> Pipeline:
    """
    Create the fallback (non-personalized) pipeline.

    Args:
        num_slots: The number of items to recommend.
    """
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)

    pipeline = Pipeline()
    candidates = pipeline.create_input("candidate", ArticleSet)
    _clicked = pipeline.create_input("clicked", ArticleSet)
    profile = pipeline.create_input("profile", InterestProfile)
    filtered = pipeline.add_component("topic-filter", topic_filter, candidate=candidates, interest_profile=profile)
    sampled = pipeline.add_component("sampler", sampler, candidate=filtered, backup=candidates)
    pipeline.alias("recommender", sampled)
    return pipeline
