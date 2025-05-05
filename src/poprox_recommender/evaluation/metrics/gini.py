from collections import Counter

import numpy as np
from lenskit.stats import gini

from poprox_concepts.domain import CandidateSet


def gini_coeff(final_recs: CandidateSet) -> float:
    article_ids = [article.article_id for article in final_recs.articles]
    item_counts = list(Counter(article_ids).values())
    if len(item_counts) == 0:
        return 0.0

    item_counts = np.sort(item_counts)

    return gini(item_counts)
