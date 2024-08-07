# pyright: basic
import logging
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.components.diversifiers import MMRDiversifier, PFARDiversifier, TopicCalibrator
from poprox_recommender.components.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Fill
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.components.scorers import ArticleScorer
from poprox_recommender.model import get_model
from poprox_recommender.pipeline import PipelineState, RecommendationPipeline

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
    Select articles with default recommender configuration.
    """
    pipeline = None

    pipeline = personalized_pipeline(num_slots, algo_params)

    inputs = {
        "candidate": candidate_articles,
        "clicked": clicked_articles,
        "profile": interest_profile,
    }

    return pipeline(inputs)


def personalized_pipeline(num_slots: int, algo_params: dict[str, Any] | None = None) -> RecommendationPipeline | None:
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
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)
    fill = Fill(num_slots=num_slots)

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

    pipeline = RecommendationPipeline(name=diversify)

    # Compute embeddings
    pipeline.add(article_embedder, inputs=["candidate"], output="candidate")
    pipeline.add(article_embedder, inputs=["clicked"], output="clicked")
    pipeline.add(user_embedder, inputs=["clicked", "profile"], output="profile")

    # Score and rank articles with diversification/calibration reranking
    pipeline.add(article_scorer, inputs=["candidate", "profile"], output="candidate")
    pipeline.add(ranker, inputs=["candidate", "profile"], output="reranked")

    # Output the plain descending-by-score ranking for comparison
    pipeline.add(topk_ranker, inputs=["candidate", "profile"], output="ranked")

    # Fallback in case not enough articles came from the ranker
    pipeline.add(topic_filter, inputs=["candidate", "profile"], output="topical")
    pipeline.add(sampler, inputs=["topical", "candidate"], output="sampled")
    pipeline.add(fill, inputs=["reranked", "sampled"], output="recs")

    return pipeline
