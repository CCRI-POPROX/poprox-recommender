import logging
from uuid import UUID

import numpy as np
from lenskit.pipeline import Component

from poprox_concepts.domain import CandidateSet

logger = logging.getLogger(__name__)


class ImpressionFilter(Component):
    config: None

    def __call__(self, candidates: CandidateSet, impressed_article_ids: list[UUID]) -> CandidateSet:
        # Get the set of article IDs the user has already received
        impressed_ids = set(impressed_article_ids)

        if not impressed_ids:
            return candidates

        # Filter out articles that have already been impressed
        kept_articles = []
        kept_scores = []
        for idx, article in enumerate(candidates.articles):
            if article.article_id not in impressed_ids:
                kept_articles.append(article)
                if hasattr(candidates, "scores") and candidates.scores is not None:
                    kept_scores.append(candidates.scores[idx])

        # Build filtered candidate set
        filtered = CandidateSet(articles=kept_articles)
        if kept_scores:
            filtered.scores = np.array(kept_scores)
        else:
            filtered.scores = None

        return filtered
