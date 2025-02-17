from collections import defaultdict

import numpy as np  # type: ignore

from poprox_concepts.domain import ArticleSet
from poprox_recommender.topics import find_topic


def gini(final_recs: ArticleSet) -> float:
    # should we count the categories
    topic_count_dict = defaultdict(int)

    for article_id in [article.article_id for article in final_recs.articles]:
        recommended_topics = find_topic(final_recs.articles, article_id) or set()
        for topic in recommended_topics:
            topic_count_dict[topic] += 1

    topic_counts = sorted(topic_count_dict.values())

    if not topic_counts or len(topic_counts) == 1:
        return 0.0
    n = len(topic_counts)
    cumulative_diffs = sum(abs(xi - xj) for i, xi in enumerate(topic_counts) for j, xj in enumerate(topic_counts))
    mean_x = np.mean(topic_counts)
    gini = cumulative_diffs / (2 * n**2 * mean_x)

    return gini
