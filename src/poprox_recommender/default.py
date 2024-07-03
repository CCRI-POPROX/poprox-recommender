from typing import Any
from uuid import UUID

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.diversifiers import MMRDiversifier, PFARDiversifier
from poprox_recommender.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.filters import TopicFilter
from poprox_recommender.model import get_model
from poprox_recommender.pipeline import RecommendationPipeline
from poprox_recommender.samplers import UniformSampler
from poprox_recommender.scorers import ArticleScorer
from poprox_recommender.topics import user_topic_preference


def select_articles(
    candidate_articles: list[Article],
    past_articles: list[Article],
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> dict[UUID, list[Article]]:
    candidate_articles = ArticleSet(articles=candidate_articles)
    past_articles = ArticleSet(articles=past_articles)

    click_history = interest_profile.click_history
    clicked_articles = list(filter(lambda a: a.article_id in set(click_history.article_ids), past_articles.articles))
    clicked_articles = ArticleSet(articles=clicked_articles)

    # This could be a component but should likely be moved upstream to the platform
    interest_profile.click_topic_counts = user_topic_preference(past_articles.articles, interest_profile.click_history)
    account_id = click_history.account_id

    algo_params = algo_params or {}
    diversify = str(algo_params.get("diversity_algo", "pfar"))

    model = get_model()
    recommendations = {}

    # The following code should ONLY access the InterestProfile and ArticleSets defined above
    if model and click_history.article_ids:
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

        recs = pipeline(
            {
                "candidate": candidate_articles,
                "clicked": clicked_articles,
                "profile": interest_profile,
            }
        )

        recommendations[account_id] = recs.articles
    else:
        topic_filter = TopicFilter()
        sampler = UniformSampler(num_slots=10)

        pipeline = RecommendationPipeline(name="random_topical")
        pipeline.add(topic_filter, inputs=["candidate", "profile"], output="topical")
        pipeline.add(sampler, inputs=["topical", "candidate"], output="recs")

        recommendations[account_id] = pipeline(
            {
                "candidate": candidate_articles,
                "clicked": clicked_articles,
                "profile": interest_profile,
            }
        )

    return recommendations
