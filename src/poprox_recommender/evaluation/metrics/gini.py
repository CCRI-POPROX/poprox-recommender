from collections import Counter

import numpy as np  # type: ignore

from poprox_concepts.domain import ArticleSet
from poprox_recommender.topics import extract_general_topics


def gini(final_recs: ArticleSet) -> float:
    topic_counter = Counter()
    for article in final_recs.articles:
        topic_counter.update(extract_general_topics(article))

    topic_counts = sorted(topic_counter.values())

    if not topic_counts or len(topic_counts) == 1:
        return 0.0

    n = len(topic_counts)
    cumulative_diffs = sum(abs(xi - xj) for i, xi in enumerate(topic_counts) for j, xj in enumerate(topic_counts))
    mean_x = np.mean(topic_counts)
    gini = cumulative_diffs / (2 * n**2 * mean_x)

    return gini
