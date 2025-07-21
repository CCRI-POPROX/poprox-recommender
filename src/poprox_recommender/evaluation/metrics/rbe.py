from collections import defaultdict

import numpy as np
from lenskit.metrics import GeometricRankWeight

from poprox_concepts.domain import CandidateSet
from poprox_recommender.data.eval import EvalData


def rank_bias_entropy(final_recs: CandidateSet, k: int, d: float = 0.85, eval_data: EvalData | None = None):
    top_k_articles = final_recs.articles[:k]
    weighted_counts = defaultdict(float)

    rank_weight = GeometricRankWeight(d)
    weights = rank_weight.weight(np.arange(1, k + 1))

    for rank, (article, weight) in enumerate(zip(top_k_articles, weights), start=1):
        mentions = article.mentions
        if not mentions and eval_data is not None:
            if not mentions:
                return np.nan

        for mention in mentions:
            topic = mention.entity.name
            weighted_counts[topic] += float(weight)

    total_weight = sum(weighted_counts.values())
    if total_weight == 0:
        return 0.0

    topic_probs = {topic: count / total_weight for topic, count in weighted_counts.items()}
    entropy = -sum(p * np.log2(p) for p in topic_probs.values() if p > 0)
    return entropy
