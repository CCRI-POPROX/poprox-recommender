import logging
from uuid import UUID

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet

logger = logging.getLogger(__name__)


class PackageFilterConfig(BaseModel):
    package_entity_id: UUID


class PackageFilter(Component):
    config: PackageFilterConfig

    def __call__(self, candidate_articles: CandidateSet, article_packages: list[ArticlePackage]) -> CandidateSet:
        article_index_lookup = {article.article_id: i for i, article in enumerate(candidate_articles.articles)}
        selected_articles = []
        selected_indices = []

        for package in article_packages:
            if package.seed and package.seed.entity_id == self.config.package_entity_id:
                for article_id in package.article_ids:
                    if article_id in article_index_lookup:
                        idx = article_index_lookup[article_id]
                        selected_articles.append(candidate_articles.articles[idx])
                        selected_indices.append(idx)

        logger.debug(
            "PackageFilter selected %d of %d candidate articles from'%d' packages",
            len(selected_articles),
            len(candidate_articles.articles),
            len(article_packages),
        )

        filtered = CandidateSet(articles=selected_articles)
        scores = getattr(candidate_articles, "scores", None)
        if scores is not None:
            filtered.scores = [scores[i] for i in selected_indices]
        else:
            filtered.scores = None

        return filtered
