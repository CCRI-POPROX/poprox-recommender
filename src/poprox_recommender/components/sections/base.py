import logging
from uuid import UUID

import numpy as np

from poprox_concepts.domain import CandidateSet

logger = logging.getLogger(__name__)


def select_mentioning(candidate: CandidateSet, entity_ids: list[UUID]):
    kept_articles = []
    kept_scores = []
    for idx, article in enumerate(candidate.articles):
        mentioned = set(m.entity.entity_id for m in article.mentions if m.relevance and m.relevance >= 76)
        if len(set(entity_ids).intersection(mentioned)) > 0:
            kept_articles.append(article)
            if hasattr(candidate, "scores") and candidate.scores is not None:
                kept_scores.append(candidate.scores[idx])

    filtered = CandidateSet(articles=kept_articles)
    if kept_scores:
        filtered.scores = np.array(kept_scores)
    else:
        filtered.scores = None

    return filtered
