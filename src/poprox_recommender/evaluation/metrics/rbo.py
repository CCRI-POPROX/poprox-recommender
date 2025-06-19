import numpy as np
from lenskit.metrics import GeometricRankWeight

from poprox_concepts.domain import CandidateSet


def rank_biased_overlap(recs_list_a: CandidateSet, recs_list_b: CandidateSet, p: float = 0.85, k: int = 10) -> float:
    """
    Computes the RBO metric defined in:
    Webber, William, Alistair Moffat, and Justin Zobel. "A similarity measure for indefinite rankings."
    ACM Transactions on Information Systems (TOIS) 28.4 (2010): 20.

    https://dl.acm.org/doi/10.1145/1852102.1852106
    """

    rank_weight = GeometricRankWeight(p)
    weights = rank_weight.weight(np.arange(1, k + 1))

    sum = 0
    total_weights = 0

    for d, weight in enumerate(weights, start=1):
        set_a = set([a.article_id for a in recs_list_a.articles[:d]])
        set_b = set([b.article_id for b in recs_list_b.articles[:d]])
        overlap = len(set_a.intersection(set_b))
        agreement = overlap / d
        sum += agreement * weight
        total_weights += weight

    rbo = sum / total_weights

    return rbo
