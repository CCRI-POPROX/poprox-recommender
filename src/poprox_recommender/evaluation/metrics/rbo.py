import numpy as np

from poprox_concepts.domain import CandidateSet


def rank_biased_overlap(recs_list_a: CandidateSet, recs_list_b: CandidateSet, p: float = 0.9, k: int = 10) -> float:
    """
    Computes the RBO metric defined in:
    Webber, William, Alistair Moffat, and Justin Zobel. "A similarity measure for indefinite rankings."
    ACM Transactions on Information Systems (TOIS) 28.4 (2010): 20.

    https://dl.acm.org/doi/10.1145/1852102.1852106
    """

    if p == 0:
        raise Exception("RBO: p=0")
    # if len(recs_list_a.articles) < k or len(recs_list_b.articles) < k:
    #     raise Exception(f"RBO: ranked or reranked list lesser than {k}")

    sum = 0
    weights = 0
    for d in range(1, k + 1):
        set_a = set([a.article_id for a in recs_list_a.articles[:d]])
        set_b = set([b.article_id for b in recs_list_b.articles[:d]])
        overlap = len(set_a.intersection(set_b))
        agreement = overlap / d
        weight = np.power(p, d)
        sum += agreement * weight
        weights += weight

    rbo = sum / weights

    return rbo
