import math

import torch as th

from poprox_recommender.topics import GENERAL_TOPICS, extract_general_topics, normalized_topic_count


class PFARDiversifier:
    def __init__(self, algo_params):
        self.lamb = float(algo_params.get("pfar_lamb", 1))
        self.tau = algo_params.get("pfar_tau", None)

    def __call__(self, article_scores, interest_profile, candidate_articles, topk):
        article_scores = th.sigmoid(th.tensor(article_scores)).cpu().detach().numpy()

        topic_preferences: dict[str, int] = {}

        for interest in interest_profile.onboarding_topics:
            topic_preferences[interest.entity_name] = max(interest.preference - 1, 0)

        for topic, click_count in interest_profile.click_topic_counts.items():
            topic_preferences[topic] = click_count

        normalized_topic_prefs = normalized_topic_count(topic_preferences)

        recs = pfar_diversification(
            article_scores, candidate_articles, normalized_topic_prefs, self.lamb, self.tau, topk=topk
        )

        return recs


def pfar_diversification(relevance_scores, articles, topic_preferences, lamb, tau, topk):
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
