from collections import Counter

import numpy as np  # type: ignore

from poprox_concepts.domain import ArticleSet
from poprox_recommender.topics import extract_general_topics


def rank_bias_entropy(final_recs: ArticleSet, k: int):
    # RBE score (higher = more diverse, lower = less diverse).
    top_k_articles = final_recs.articles[:k]

    topic_counter = Counter()
    for article in top_k_articles:
        topic_counter.update(extract_general_topics(article))

    total_articles = sum(topic_counter.values())
    if total_articles == 0:
        return 0.0

    topic_probs = {topic: count / total_articles for topic, count in topic_counter.items()}
    entropy = -sum(p * np.log2(p) for p in topic_probs.values() if p > 0)

    return entropy
