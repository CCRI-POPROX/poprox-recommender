import logging
from collections import defaultdict

import numpy as np
import torch as th
from poprox_concepts import Article, ArticleSet, InterestProfile

from poprox_recommender.rankers import Ranker
from poprox_recommender.topics import (extract_general_topics,
                                       normalized_topic_count)


# Topic Calibration uses MMR
# to rerank recommendations according to
# topic calibration
class TopicCalibrator(Ranker):
    def __init__(self, algo_params, num_slots=10):
        self.validate_algo_params(algo_params, ["theta"])

        # Theta term controls the score and calibration tradeoff, the higher
        # the theta the higher the resulting recommendation will be calibrated.
        # We choose the binary search way to start in the middle
        self.theta = float(algo_params.get("theta", 0.1))
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        article_scores = th.sigmoid(th.tensor(candidate_articles.scores)).cpu().detach().numpy()

        topic_preferences: dict[str, int] = {}

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] = click_count

        normalized_topic_prefs = normalized_topic_count(topic_preferences)

        article_indices = topic_calibration(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            self.theta,
            topk=self.num_slots,
        )

        return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])


def topic_calibration(relevance_scores, articles, topic_preferences, theta, topk) -> list[Article]:
    # MR_i = \theta * reward_i - (1 - \theta)*C(S + i) # C is calibration
    # R is all candidates (not selected yet)

    recommendations = []  # final recommendation (topk index)
    rec_topics = defaultdict(int)  # frequency distribution of topics of S
    # first recommended item
    recommendations.append(relevance_scores.argmax())
    add_article_to_topics(rec_topics, articles[int(relevance_scores.argmax())])

    for k in range(topk - 1):
        candidate = None  # next item
        best_candidate_score = float("-inf")

        for article_idx, article_score in enumerate(relevance_scores):  # iterate R for next item
            if article_idx in recommendations:
                continue

            normalized_candidate_topics = normalized_topics_with_candidate(rec_topics, articles[article_idx])
            calibration = compute_kl_divergence(topic_preferences, normalized_candidate_topics)

            adjusted_candidate_score = (1 - theta) * article_score - (theta * calibration)
            if adjusted_candidate_score > best_candidate_score:
                best_candidate_score = adjusted_candidate_score
                candidate = article_idx

        if candidate is not None:
            recommendations.append(candidate)
            add_article_to_topics(rec_topics, articles[candidate])

    return recommendations


def add_article_to_topics(rec_topics, article):
    topics = extract_general_topics(article)
    for topic in topics:
        rec_topics[topic] = rec_topics.get(topic, 0) + 1


def normalized_topics_with_candidate(rec_topics, article):
    rec_topics_with_candidate = rec_topics.copy()
    add_article_to_topics(rec_topics_with_candidate, article)
    return normalized_topic_count(rec_topics_with_candidate)


# from https://github.com/CCRI-POPROX/poprox-recommender/blob/feature/experiment0/tests/test_calibration.ipynb
def compute_kl_divergence(interacted_distr, reco_distr):
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    kl_div = 0.0
    alpha = 0.01
    for genre, score in interacted_distr.items():
        reco_score = reco_distr.get(genre, 0.0)
        reco_score = (1 - alpha) * reco_score + alpha * score
        kl_div += score * np.log2(score / reco_score)

    return kl_div
