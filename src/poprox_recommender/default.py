# pyright: basic
import logging
from dataclasses import dataclass
from typing import Any

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers import MMRDiversifier, PFARDiversifier, TopicCalibrator
from poprox_recommender.components.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.lkpipeline import Pipeline
from poprox_recommender.model import get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Recommendations:
    recs: list[Article]
    initial: ArticleSet | None = None
    reranked: ArticleSet | None = None


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> Recommendations:
    """
    Select articles with default recommender configuration.
    """
    pipeline = None

    if interest_profile.click_history.article_ids:
        pipeline = personalized_pipeline(num_slots, algo_params)

    if pipeline is None:
        pipeline = fallback_pipeline(num_slots)

    rank = pipeline.node("rank")
    topk = pipeline.node("rank-topk", missing="none")
    if topk is None:
        wanted = (rank,)
    else:
        wanted = (rank, topk)

    result = pipeline.run(*wanted, candidate=candidate_articles, clicked=clicked_articles, profile=interest_profile)

    if topk is None:
        assert isinstance(result, ArticleSet)
        return Recommendations(result.articles, initial=result)
    else:
        assert isinstance(result, tuple)
        reranked, initial = result
        return Recommendations(reranked.articles, initial=initial, reranked=reranked)


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
    e_cand = pipeline.add_component("embed-candidates", article_embedder, article_set=candidates)
    e_click = pipeline.add_component("embed-clicked-articles", article_embedder, article_set=clicked)
    e_user = pipeline.add_component("embed-user", user_embedder, clicked_articles=e_click, interest_profile=profile)
    scored = pipeline.add_component(
        "score-articles", article_scorer, candidate_articles=e_cand, interest_profile=e_user
    )
    topk = pipeline.add_component("rank-topk", topk_ranker, candidate_articles=scored, interest_profile=e_user)
    if ranker is topk_ranker:
        pipeline.alias("rank", topk)
    else:
        _final = pipeline.add_component("rank", ranker, candidate_articles=scored, interest_profile=e_user)

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
    filtered = pipeline.add_component("filter-articles", topic_filter, candidate=candidates, interest_profile=profile)
    _sampled = pipeline.add_component("rank", sampler, candidate=filtered, backup=candidates)
    return pipeline
