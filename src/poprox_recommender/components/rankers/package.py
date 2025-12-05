import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection

logger = logging.getLogger(__name__)


class PackageConfig(BaseModel):
    package_name: str


class PackageRanker(Component):
    config: PackageConfig

    def __call__(self, candidate_articles: CandidateSet, article_packags: list[ArticlePackage]) -> ImpressedSection:
        article_lookup = {article.article_id: article for article in candidate_articles.articles}

        selected_articles = []
        for package in article_packags:
            selected_articles.extend(
                article_lookup[article_id] for article_id in package.article_ids if article_id in article_lookup
            )

        logger.debug(
            "PackageRanker selected %d of %d candidate articles from'%d' packages",
            len(selected_articles),
            len(candidate_articles.articles),
            len(article_packags),
        )

        return ImpressedSection.from_articles(selected_articles)
