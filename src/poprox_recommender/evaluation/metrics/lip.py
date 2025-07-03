import numpy as np

from poprox_concepts.domain import CandidateSet
from poprox_recommender.data.eval import EvalData


def least_item_promoted(
    reference_article_set: CandidateSet,
    reranked_article_set: CandidateSet,
    k: int = 10,
    eval_data: EvalData | None = None,
) -> float:
    if not reference_article_set.articles:
        return np.nan

    reference_articles_list = reference_article_set.articles
    ranks = []
    for item in reranked_article_set.articles[:k]:
        try:
            rank = reference_articles_list.index(item)
            ranks.append(rank)
        except ValueError:
            continue
    if ranks:
        lip_rank = max(ranks)  # worst (least promoted) rank among matches
        return (lip_rank - (k - 1)) / len(reference_articles_list)
    else:
        return 0.0
