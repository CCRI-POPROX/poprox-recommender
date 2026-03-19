import numpy as np
from lenskit.pipeline import Component

from poprox_concepts.domain import CandidateSet, ImpressedSection


class DuplicateFilter(Component):
    def __call__(self, candidate: CandidateSet, sections: list[ImpressedSection]) -> CandidateSet:
        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)

        kept_articles = []
        kept_scores = []
        for idx, article in enumerate(candidate.articles):
            if article.article_id not in used_ids:
                kept_articles.append(article)
                if hasattr(candidate, "scores") and candidate.scores is not None:
                    kept_scores.append(candidate.scores[idx])

        filtered = CandidateSet(articles=kept_articles)
        if kept_scores:
            filtered.scores = np.array(kept_scores)
        else:
            filtered.scores = None

        return filtered
