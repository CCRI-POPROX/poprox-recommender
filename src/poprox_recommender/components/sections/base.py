import logging
from uuid import UUID

import numpy as np

from poprox_concepts.domain import Article, CandidateSet

logger = logging.getLogger(__name__)


def select_from_candidates(candidates: CandidateSet, num_articles: int, excluding: set[UUID] = None) -> list[Article]:
    excluding = excluding or set()

    if hasattr(candidates, "scores") and candidates.scores is not None:
        # rank candidates by score if scores are available
        sorted_indices = np.argsort(np.array(candidates.scores))[::-1]
        ranked_articles = [
            candidates.articles[int(i)]
            for i in sorted_indices
            if candidates.articles[int(i)].article_id not in excluding
        ][:num_articles]
    else:
        # otherwise select from the top of the list of candidates preserving order
        ranked_articles = [a for a in candidates.articles if a.article_id not in excluding][:num_articles]

    return ranked_articles
