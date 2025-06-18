import numpy as np

from poprox_concepts.domain import CandidateSet


def least_item_promoted(reference_article_set: CandidateSet, reranked_article_set: CandidateSet, k: int = 10) -> float:
    if not reference_article_set.articles:
        return np.nan

    lip_rank = 0
    reranked_articles_list = reranked_article_set.articles

    for item in reference_article_set.articles:
        try:
            rank_item = reranked_articles_list.index(item) + 1
        except ValueError:
            rank_item = k + 1

        if rank_item > lip_rank:
            lip_rank = rank_item

    return (lip_rank - k) / len(reference_article_set.articles)
