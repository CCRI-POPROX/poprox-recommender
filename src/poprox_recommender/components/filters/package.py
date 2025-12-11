import logging
from uuid import UUID

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet

logger = logging.getLogger(__name__)


class PackageFilterConfig(BaseModel):
    """Configuration for PackageFilter component."""

    package_entity_id: UUID


class PackageFilter(Component):
    """
    Filter component that extracts articles from packages matching a specific entity ID.

    This component processes article packages and returns a CandidateSet containing
    only the articles from packages matching the configured entity_id.

    Attributes:
        package_entity_id: UUID of the entity to match (e.g., TOP_NEWS_ENTITY_ID)
    """

    config: PackageFilterConfig

    def __call__(self, candidate_articles: CandidateSet, article_packages: list[ArticlePackage]) -> CandidateSet:
        """
        Filter candidate articles based on article packages.

        Args:
            candidate_articles: Set of candidate articles to filter from
            article_packages: List of article packages from the platform

        Returns:
            CandidateSet containing only articles from matching packages
        """
        article_lookup = {article.article_id: article for article in candidate_articles.articles}
        selected_articles = []

        for package in article_packages:
            # Skip packages without seed or with wrong entity_id
            if not package.seed or package.seed.entity_id != self.config.package_entity_id:
                continue

            # Extract articles from this package
            for article_id in package.article_ids:
                # Skip if not in candidates
                if article_id not in article_lookup:
                    continue

                # Add article
                selected_articles.append(article_lookup[article_id])

        logger.debug(
            "PackageFilter selected %d of %d candidate articles from %d packages (entity: %s)",
            len(selected_articles),
            len(candidate_articles.articles),
            len(article_packages),
            self.config.package_entity_id,
        )

        return CandidateSet(articles=selected_articles)
