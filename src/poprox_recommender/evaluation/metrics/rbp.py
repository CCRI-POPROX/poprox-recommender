import numpy as np
from lenskit.metrics import GeometricRankWeight

from poprox_concepts.domain import CandidateSet


def rank_biased_precision(
    final_recs: CandidateSet,
    eval_data: CandidateSet,
    k: int = 100,
    patience: float = 0.85,
    normalize: bool = False,
) -> float:
    """
    Compute Rank-Biased Precision (RBP) for a ranked list of recommendations.

    Args:
        final_recs: CandidateSet containing ranked recommendations.
        eval_data: CandidateSet containing relevant (ground truth) articles.
        k: Cutoff for the top-k items in the recommendation list.
        patience: Probability that a user continues browsing (default=0.85).
        normalize: If True, normalizes by the maximum achievable RBP (like nDCG).

    Returns:
        RBP score (float) or np.nan if undefined.
    """
    if not final_recs.articles or not eval_data.articles:
        return np.nan

    top_k_articles = final_recs.articles[:k]
    n_rel = len(eval_data.articles)
    if n_rel == 0:
        return np.nan

    # Boolean mask for whether recommended items are relevant
    good = np.array([article in eval_data.articles for article in top_k_articles])

    # Compute geometric rank weights
    rank_weight = GeometricRankWeight(patience)
    weights = rank_weight.weight(np.arange(1, len(top_k_articles) + 1))

    if normalize:
        # Normalize by the maximum achievable RBP (the top n_rel could be all relevant)
        normalization = np.sum(weights[: min(n_rel, len(weights))])
    else:
        # Use the theoretical normalization (sum of infinite geometric series)
        wmax = rank_weight.series_sum()
        normalization = wmax if wmax is not None else np.sum(weights)

    if normalization == 0:
        return np.nan

    rbp_score = np.sum(weights[good]) / normalization
    return float(rbp_score)
