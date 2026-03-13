import logging
from uuid import UUID

import numpy as np

from poprox_concepts.domain import Article, CandidateSet, Entity

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


def select_mentioning(candidate: CandidateSet, entities: list[Entity]):
    entity_ids = set(e.entity_id for e in entities)

    kept_articles = []
    kept_scores = []
    for idx, article in enumerate(candidate.articles):
        mentioned = set(m.entity.entity_id for m in article.mentions if m.relevance and m.relevance >= 76)
        if len(entity_ids.intersection(mentioned)) > 0:
            kept_articles.append(article)
            if hasattr(candidate, "scores") and candidate.scores is not None:
                kept_scores.append(candidate.scores[idx])

    filtered = CandidateSet(articles=kept_articles)
    if kept_scores:
        filtered.scores = np.array(kept_scores)
    else:
        filtered.scores = None

    return filtered
