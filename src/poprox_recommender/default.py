import random
from typing import Any
from uuid import UUID

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.diversifiers import MMRDiversifier, PFARDiversifier
from poprox_recommender.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.model import get_model
from poprox_recommender.pipeline import RecommendationPipeline
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
        pipeline.add(article_embedder)
        pipeline.add(article_scorer)
        pipeline.add(diversifier)

        clicked_articles = article_embedder(clicked_articles)
        interest_profile = user_embedder(clicked_articles, interest_profile)

        recs = pipeline(candidate_articles, interest_profile)

        recommendations[account_id] = recs.articles
    else:
        recommendations[account_id] = select_by_topic(
            candidate_articles.articles,
            interest_profile,
            num_slots,
        )

    return recommendations


def select_by_topic(todays_articles: list[Article], interest_profile: InterestProfile, num_slots: int):
    # Preference values from onboarding are 1-indexed, where 1 means "absolutely no interest."
    # We might want to normalize them to 0-indexed somewhere upstream, but in the mean time
    # this is one of the simpler ways to filter out topics people aren't interested in from
    # their early newsletters
    profile_topics = {
        interest.entity_name for interest in interest_profile.onboarding_topics if interest.preference > 1
    }

    other_articles = []
    topical_articles = []
    for article in todays_articles:
        article_topics = {mention.entity.name for mention in article.mentions}
        if len(profile_topics.intersection(article_topics)) > 0:
            topical_articles.append(article)
        else:
            other_articles.append(article)

    if len(topical_articles) >= num_slots:
        return random.sample(topical_articles, num_slots)
    else:
        return random.sample(topical_articles, len(topical_articles)) + random.sample(
            other_articles, num_slots - len(topical_articles)
        )
