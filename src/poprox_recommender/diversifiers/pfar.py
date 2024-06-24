import math


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
    S_topic.update([mention.entity.name for mention in article.mentions])

    for k in range(topk - 1):
        candidate_idx = None
        best_score = float("-inf")

        for i, relevance_i in enumerate(relevance_scores):  # iterate R for next item
            if i in S:
                continue
            product = 1
            summation = 0

            candidate_topics = [mention.entity.name for mention in articles[int(i)].mentions]
            for topic in candidate_topics:
                if topic in S_topic:
                    product = 0
                    break

            for topic in candidate_topics:
                if topic in topic_preferences:
                    summation += topic_preferences[topic]

            pfar_score_i = relevance_i + lamb * tau * summation * product

            if pfar_score_i > best_score:
                best_score = pfar_score_i
                candidate_idx = i

        if candidate_idx is not None:
            S.append(candidate_idx)
            S_topic.update([mention.entity.name for mention in articles[candidate_idx].mentions])

    return S  # LIST(candidate index)
