import random
from typing import Any
from uuid import UUID

import torch as th

from poprox_concepts import Article, InterestProfile
from poprox_recommender.diversifiers import mmr_diversification, pfar_diversification
from poprox_recommender.diversifiers.mmr import compute_similarity_matrix
from poprox_recommender.embedders import ArticleEmbedder, UserEmbedder
from poprox_recommender.model import DEVICE, MODEL, TOKENIZER
from poprox_recommender.scorers import ArticleScorer
from poprox_recommender.topics import normalized_topic_count, user_topic_preference


def select_articles(
    candidate_articles: list[Article],
    past_articles: list[Article],
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> dict[UUID, list[Article]]:
    click_history = interest_profile.click_history
    clicked_articles = filter(lambda a: a.article_id in set(click_history.article_ids), past_articles)
    interest_profile.click_topic_counts = user_topic_preference(past_articles, interest_profile.click_history)
    account_id = click_history.account_id

    algo_params = algo_params or {}
    diversify = str(algo_params.get("diversity_algo", "pfar"))

    recommendations = {}
    if MODEL and TOKENIZER and click_history.article_ids:
        article_embedder = ArticleEmbedder(MODEL, TOKENIZER, DEVICE)
        user_embedder = UserEmbedder(MODEL, DEVICE)
        article_scorer = ArticleScorer(MODEL)

        candidate_article_lookup, candidate_article_tensor = article_embedder(candidate_articles)
        clicked_article_lookup, clicked_article_tensor = article_embedder(clicked_articles)

        user_embedding = user_embedder(interest_profile, clicked_article_lookup)
        article_scores = article_scorer(candidate_article_tensor, user_embedding)

        if diversify == "mmr":
            theta = float(algo_params.get("theta", 0.8))

            # Compute today's article similarity matrix
            similarity_matrix = compute_similarity_matrix(candidate_article_tensor)

            recs = mmr_diversification(article_scores, similarity_matrix, theta=theta, topk=num_slots)

        elif diversify == "pfar":
            lamb = float(algo_params.get("pfar_lamb", 1))
            tau = algo_params.get("pfar_tau", None)
            article_scores = th.sigmoid(th.tensor(article_scores)).cpu().detach().numpy()

            topic_preferences: dict[str, int] = {}

            for interest in interest_profile.onboarding_topics:
                topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] = click_count

            normalized_topic_prefs = normalized_topic_count(topic_preferences)

            recs = pfar_diversification(
                article_scores, candidate_articles, normalized_topic_prefs, lamb, tau, topk=num_slots
            )

        recommendations[account_id] = [candidate_articles[int(rec)] for rec in recs]
    else:
        recommendations[account_id] = select_by_topic(
            todays_articles,
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
