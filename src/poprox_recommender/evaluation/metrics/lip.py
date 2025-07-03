import numpy as np

from poprox_concepts.domain import CandidateSet


def least_item_promoted(reference_article_set: CandidateSet, reranked_article_set: CandidateSet, k: int = 10) -> float:
    if not reference_article_set.articles:
        return np.nan
    lip_rank = k
    for item in reranked_article_set.articles[:k]:
        try:
            rank = reference_article_set.articles.index(item)
            if rank > lip_rank:
                lip_rank = rank
        except ValueError:
            continue
    return lip_rank - k
