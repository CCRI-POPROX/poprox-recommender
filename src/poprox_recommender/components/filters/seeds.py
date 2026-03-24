import numpy as np
from lenskit.pipeline import Component

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection
from poprox_recommender.components.sections.base import select_mentioning


class PreviousSectionsFilter(Component):
    def __call__(
        self, candidate: CandidateSet, article_packages: list[ArticlePackage], sections: list[ImpressedSection]
    ) -> CandidateSet:
        prev_section_seed_ids = [
            package.seed.entity_id
            for package in article_packages
            for section in sections
            if section.seed_entity_id == package.seed.entity_id
        ]

        used_topic_articles = select_mentioning(candidate, prev_section_seed_ids)
        used_ids = set(article.article_id for article in used_topic_articles.articles)

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
