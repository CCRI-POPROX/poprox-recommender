from collections import defaultdict

import numpy as np

from poprox_concepts.domain import CandidateSet
from poprox_recommender.topics import extract_general_topics


def rank_bias_entropy(final_recs: CandidateSet, k: int, d: float = 0.5):
    top_k_articles = final_recs.articles[:k]
    weighted_counts = defaultdict(float)

    for rank, article in enumerate(top_k_articles):
        weight = d ** (rank + 1)  # d is the rank-based discount
        for topic in extract_general_topics(article):
            weighted_counts[topic] += float(weight)

    total_weight = sum(weighted_counts.values())
    if total_weight == 0:
        return 0.0

    topic_probs = {topic: count / total_weight for topic, count in weighted_counts.items()}
    entropy = -sum(p * np.log2(p) for p in topic_probs.values() if p > 0)

    return entropy
