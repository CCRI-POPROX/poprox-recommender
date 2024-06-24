import random
from typing import Any
from uuid import UUID

import torch as th

from poprox_concepts import Article, ClickHistory, InterestProfile
from poprox_recommender.diversifiers.mmr import compute_similarity_matrix, mmr_diversification
from poprox_recommender.diversifiers.pfar import pfar_diversification
from poprox_recommender.embedders import ArticleEmbedder
from poprox_recommender.model import DEVICE, MODEL, TOKENIZER
from poprox_recommender.topics import normalized_topic_count, user_topic_preference


# Compute a vector for each user
def build_user_embedding(click_history: ClickHistory, article_embeddings, model, device, max_clicks_per_user):
    article_ids = list(dict.fromkeys(click_history.article_ids))[
        -max_clicks_per_user:
    ]  # deduplicate while maintaining order

    padded_positions = max_clicks_per_user - len(article_ids)
    assert padded_positions >= 0

    article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
    default = article_embeddings["PADDED_NEWS"]
    clicked_article_embeddings = [
        article_embeddings.get(clicked_article, default).to(device) for clicked_article in article_ids
    ]
    clicked_news_vector = (
        th.stack(
            clicked_article_embeddings,
            dim=0,
        )
        .unsqueeze(0)
        .to(device)
    )

    return model.get_user_vector(clicked_news_vector)


def generate_recommendations(
    model,
    articles,
    article_vectors,
    similarity_matrix,
    user_embedding,
    interest_profile: InterestProfile,
    num_slots: int = 10,
    algo_params: dict[str, Any] | None = None,
) -> list[Article]:
    algo_params = algo_params or {}
    diversify = str(algo_params.get("diversity_algo", "pfar"))

    pred = model.get_prediction(article_vectors, user_embedding.squeeze())

    if diversify == "mmr":
        theta = float(algo_params.get("theta", 0.8))
        pred = pred.cpu().detach().numpy()
        recs = mmr_diversification(pred, similarity_matrix, theta=theta, topk=num_slots)

    elif diversify == "pfar":
        lamb = float(algo_params.get("pfar_lamb", 1))
        tau = algo_params.get("pfar_tau", None)
        pred = th.sigmoid(pred).cpu().detach().numpy()

        topic_preferences: dict[str, int] = {}

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        for topic, click_count in interest_profile.click_topic_counts.items():
            topic_preferences[topic] = click_count

        normalized_topic_prefs = normalized_topic_count(topic_preferences)

        recs = pfar_diversification(pred, articles, normalized_topic_prefs, lamb, tau, topk=num_slots)

    return [articles[int(rec)] for rec in recs]


def select_with_model(
    todays_articles: list[Article],
    todays_article_vectors: th.Tensor,
    article_similarity_matrix,
    clicked_article_embeddings: dict[str, th.Tensor],
    interest_profile: InterestProfile,
    model,
    model_device,
    num_slots: int = 10,
    max_clicks_per_user: int = 50,
    algo_params: dict[str, Any] | None = None,
):
    # Build embedding tables
    user_embedding = build_user_embedding(
        interest_profile.click_history,
        clicked_article_embeddings,
        model,
        model_device,
        max_clicks_per_user,
    )

    recommendations = generate_recommendations(
        model,
        todays_articles,
        todays_article_vectors,
        article_similarity_matrix,
        user_embedding,
        interest_profile,
        num_slots=num_slots,
        algo_params=algo_params,
    )

    return recommendations


def select_articles(
    todays_articles: list[Article],
    past_articles: list[Article],
    interest_profile: InterestProfile,
    num_slots: int,
    algo_params: dict[str, Any] | None = None,
) -> dict[UUID, list[Article]]:
    article_embedder = ArticleEmbedder(MODEL, TOKENIZER, DEVICE)

    click_history = interest_profile.click_history
    clicked_articles = filter(lambda a: a.article_id in set(click_history.article_ids), past_articles)

    todays_article_lookup, todays_article_tensor = article_embedder(todays_articles)
    clicked_article_lookup, clicked_article_tensor = article_embedder(clicked_articles)

    # Compute today's article similarity matrix
    similarity_matrix = compute_similarity_matrix(todays_article_tensor)

    interest_profile.click_topic_counts = user_topic_preference(past_articles, interest_profile.click_history)

    recommendations = {}
    account_id = click_history.account_id
    if MODEL and TOKENIZER and click_history.article_ids:
        recommendations[account_id] = select_with_model(
            todays_articles,
            todays_article_tensor,
            similarity_matrix,
            clicked_article_lookup,
            interest_profile,
            MODEL,
            DEVICE,
            num_slots=num_slots,
            algo_params=algo_params,
        )
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
