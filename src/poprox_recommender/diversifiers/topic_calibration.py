from collections import defaultdict

import numpy as np
import torch as th

from poprox_concepts import Article, ArticleSet, InterestProfile
from poprox_recommender.topics import extract_general_topics, normalized_topic_count


# Topic Calibration uses MMR
# to rerank recommendations according to
# topic calibration
class TopicCalibrator:
    def __init__(self, algo_params, num_slots):
        self.theta = float(algo_params.get("theta", 0.8))
        self.num_slots = num_slots

    def __call__(self, candidate_articles: ArticleSet, interest_profile: InterestProfile) -> ArticleSet:
        article_scores = th.sigmoid(th.tensor(candidate_articles.scores)).cpu().detach().numpy()

        topic_preferences: dict[str, int] = {}

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] = click_count

        else:
            # if click history does not exist we cannot compute
            # topic calibration, so return the top k articles based on original relevance score
            article_indices = article_scores.argsort()[-self.num_slots :][::-1]
            return ArticleSet(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])

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

    S = []  # final recommendation (topk index)
    S_distr = defaultdict(int)  # frequency distribution of topics of S
    # first recommended item
    S.append(relevance_scores.argmax())
    update_S_distr(S_distr, articles[int(relevance_scores.argmax())])

    for k in range(topk - 1):
        candidate = None  # next item
        best_MR = float("-inf")

        for i, reward_i in enumerate(relevance_scores):  # iterate R for next item
            if i in S:
                continue

            normalized_S_count = get_normalized_S_count(S_distr, articles[i])
            calibration = compute_kl_divergence(topic_preferences, normalized_S_count)

            mr_i = (1 - theta) * reward_i - (theta * calibration)
            if mr_i > best_MR:
                best_MR = mr_i
                candidate = i

        if candidate is not None:
            S.append(candidate)
            update_S_distr(S_distr, articles[candidate])
    return S

    # take articles and construct frequency dictionary
    # then normalized topic count
    return []


def update_S_distr(S_distr, article):
    topics = extract_general_topics(article)
    for topic in topics:
        S_distr[topic] += S_distr.get(topic, 0) + 1


def get_normalized_S_count(S_distr, article):
    S_distr_with_candidate = S_distr.copy()
    update_S_distr(S_distr_with_candidate, article)
    return normalized_topic_count(S_distr_with_candidate)


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
