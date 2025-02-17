from collections import defaultdict

import numpy as np  # type: ignore

from poprox_concepts.domain import ArticleSet
from poprox_recommender.topics import find_topic


def rank_bias_entropy(final_recs: ArticleSet, k: int):
    """
    final_recs: A RecommendationList object containing recommended articles.
    k: The number of top-ranked items to consider.
    return: RBE score (higher = more diverse, lower = less diverse).
    """
    top_k_articles = final_recs.articles[:k]

    topic_count_dict = defaultdict(int)
    for article_id in [article.article_id for article in top_k_articles]:
        recommended_topics = find_topic(top_k_articles, article_id) or set()
        for topic in recommended_topics:
            topic_count_dict[topic] += 1

    total_articles = sum(topic_count_dict.values())
    if total_articles == 0:
        return 0.0

    topic_probs = {c: count / total_articles for c, count in topic_count_dict.items()}
    entropy = -sum(p * np.log2(p) for p in topic_probs.values() if p > 0)

    return entropy
