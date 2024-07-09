# pyright: basic
from typing import Any

from poprox_concepts import ArticleSet, InterestProfile
from poprox_recommender.diversifiers import MMRDiversifier, PFARDiversifier
from poprox_recommender.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.filters import TopicFilter
from poprox_recommender.model import get_model
from poprox_recommender.pipeline import RecommendationPipeline
from poprox_recommender.samplers import UniformSampler
from poprox_recommender.scorers import ArticleScorer


def select_articles(
    candidate_articles: ArticleSet,
    clicked_articles: ArticleSet,
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> ArticleSet:
    """
    Select articles with default recommender configuration.
    """
    pipeline = None

    if interest_profile.click_history.article_ids:
        pipeline = personalized_pipeline(num_slots, algo_params)

    if pipeline is None:
        pipeline = fallback_pipeline(num_slots)

    return pipeline(
        {
            "candidate": candidate_articles,
            "clicked": clicked_articles,
            "profile": interest_profile,
        }
    )


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
    diversify = str(algo_params.get("diversity_algo", "pfar"))

    article_embedder = ArticleEmbedder(model.model, model.tokenizer, model.device)
    user_embedder = UserEmbedder(model.model, model.device)
    article_scorer = ArticleScorer(model.model)

    if diversify == "mmr":
        diversifier = MMRDiversifier(algo_params, num_slots)
    elif diversify == "pfar":
        diversifier = PFARDiversifier(algo_params, num_slots)

    pipeline = RecommendationPipeline(name=diversify)
    pipeline.add(article_embedder, inputs=["candidate"], output="candidate")
    pipeline.add(article_embedder, inputs=["clicked"], output="clicked")
    pipeline.add(user_embedder, inputs=["clicked", "profile"], output="profile")
    pipeline.add(article_scorer, inputs=["candidate", "profile"], output="candidate")
    pipeline.add(diversifier, inputs=["candidate", "profile"], output="recs")
    return pipeline


def fallback_pipeline(num_slots: int) -> RecommendationPipeline:
    """
    Create the fallback (non-personalized) pipeline.

    Args:
        num_slots: The number of items to recommend.
    """
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=num_slots)

    pipeline = RecommendationPipeline(name="random_topical")
    pipeline.add(topic_filter, inputs=["candidate", "profile"], output="topical")
    pipeline.add(sampler, inputs=["topical", "candidate"], output="recs")
    return pipeline
