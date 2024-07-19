import numpy as np

from poprox_concepts.domain import ArticleSet


def rank_biased_overlap(recs_list_a: ArticleSet, recs_list_b: ArticleSet, p: float = 0.9, k: int = 10) -> float:
    """
    Computes the RBO metric defined in:
    Webber, William, Alistair Moffat, and Justin Zobel. "A similarity measure for indefinite rankings."
    ACM Transactions on Information Systems (TOIS) 28.4 (2010): 20.

    https://dl.acm.org/doi/10.1145/1852102.1852106
    """
    summands = []
    for d in range(1, k + 1):
        set_a = set([a.article_id for a in recs_list_a.articles[:d]])
        set_b = set([b.article_id for b in recs_list_b.articles[:d]])
        overlap = len(set_a.intersection(set_b))
        agreement = overlap / d
        summands.append(agreement * np.power(p, d))

    rbo = (1 - p) / p * sum(summands)

    return rbo
