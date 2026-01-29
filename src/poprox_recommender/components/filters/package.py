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
        article_lookup = {article.article_id: article for article in candidate_articles.articles}
        selected_articles = []

        for package in article_packages:
            if package.seed.entity_id == self.config.package_entity_id:
                articles = [
                    article_lookup[article_id] for article_id in package.article_ids if article_id in article_lookup
                ]
                selected_articles.extend(articles)

        logger.debug(
            "PackageFilter selected %d of %d candidate articles from'%d' packages",
            len(selected_articles),
            len(candidate_articles.articles),
            len(article_packages),
        )
        return CandidateSet(articles=selected_articles)
