import math

import torch as th
from lenskit.pipeline import Component

from poprox_concepts.domain import Article, CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.pytorch.decorators import torch_inference
from poprox_recommender.topics import GENERAL_TOPICS, extract_general_topics, normalized_category_count


class PFARDiversifier(Component):
    def __init__(self, lambda_: float = 1.0, tau: float | None = None, num_slots: int = 10):
        self.lambda_ = lambda_
        self.tau = tau
        self.num_slots = num_slots

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        if candidate_articles.scores is None:
            return candidate_articles

        article_scores = th.sigmoid(th.tensor(candidate_articles.scores)).cpu().detach().numpy()

        topic_preferences: dict[str, int] = {}

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        if interest_profile.click_topic_counts:
            for topic, click_count in interest_profile.click_topic_counts.items():
                topic_preferences[topic] = click_count

        normalized_topic_prefs = normalized_category_count(topic_preferences)

        article_indices = pfar_diversification(
            article_scores,
            candidate_articles.articles,
            normalized_topic_prefs,
            self.lambda_,
            self.tau,
            topk=self.num_slots,
        )

        return RecommendationList(articles=[candidate_articles.articles[int(idx)] for idx in article_indices])


def pfar_diversification(relevance_scores, articles, topic_preferences, lamb, tau, topk) -> list[Article]:
    # p(v|u) + lamb*tau \sum_{d \in D} P(d|u)I{v \in d} \prod_{i \in S} I{i \in d} for each user

    if tau is None:
        tau = 0
        for topic, weight in topic_preferences.items():
            if weight > 0:
                tau -= weight * math.log(weight)
    else:
        tau = float(tau)

    S = []  # final recommendation LIST[candidate index]
    initial_item = relevance_scores.argmax()
    S.append(initial_item)

    S_topic = set()
    article = articles[int(initial_item)]
    S_topic.update(extract_general_topics(article))

    for k in range(topk - 1):
        candidate_idx = None
        best_score = float("-inf")

        for i, relevance_i in enumerate(relevance_scores):  # iterate R for next item
            if i in S:
                continue
            product = 1
            summation = 0

            candidate_topics = set(extract_general_topics(articles[int(i)]))

            for topic in candidate_topics:
                if topic in S_topic:
                    product = 0
                    break

            for topic in candidate_topics:
                if topic in topic_preferences:
                    summation += 1.0 / len(GENERAL_TOPICS)

            pfar_score_i = relevance_i + lamb * tau * summation * product

            if pfar_score_i > best_score:
                best_score = pfar_score_i
                candidate_idx = i

        if candidate_idx is not None:
            candidate_topics = set(extract_general_topics(articles[int(candidate_idx)]))
            S.append(candidate_idx)
            S_topic.update(candidate_topics)

    return S  # LIST(candidate index)
