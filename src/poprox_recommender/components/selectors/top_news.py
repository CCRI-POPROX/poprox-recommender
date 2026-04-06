import logging
from uuid import UUID

from lenskit.pipeline import Component

from poprox_concepts.domain import ArticlePackage, CandidateSet

logger = logging.getLogger(__name__)


class TopStoryCandidates(Component):
    def __call__(self, candidate_articles: CandidateSet, article_packages: list[ArticlePackage]) -> CandidateSet:
        article_index_lookup = {article.article_id: i for i, article in enumerate(candidate_articles.articles)}
        selected_articles = []
        selected_indices = []

        package_article_ids = set(article_id for package in article_packages for article_id in package.article_ids)

        for article_id in package_article_ids:
            if article_id in article_index_lookup:
                idx = article_index_lookup[article_id]
                selected_articles.append(candidate_articles.articles[idx])
                selected_indices.append(idx)

        logger.debug(
            "TopStoryCandidates selected %d of %d candidate articles using %s packages",
            len(selected_articles),
            len(candidate_articles.articles),
            ", ".join([p.title for p in article_packages]),
        )

        filtered = CandidateSet(articles=selected_articles)
        scores = getattr(candidate_articles, "scores", None)
        if scores is not None:
            filtered.scores = [scores[i] for i in selected_indices]
        else:
            filtered.scores = None

        return filtered


class TopGeneralNewsCandidates(Component):
    def __call__(self, candidate_articles: CandidateSet, article_packages: list[ArticlePackage]) -> CandidateSet:
        general_news_uuid = UUID("72bb7674-7bde-4f3e-a351-ccdeae888502")

        # Find article ids from the General News package
        general_news_package = next(
            (package for package in article_packages if package.seed.entity_id == general_news_uuid), None
        )

        article_ids = general_news_package.article_ids if general_news_package else []

        # Iterate through the candidates to find the right articles and build an ordered list
        selected = []
        for article_id in article_ids:
            article = next(
                (article for article in candidate_articles.articles if article.article_id == article_id), None
            )
            if article:
                selected.append(article)

        # This component *does not* preserve relevance scores so that downstream
        # TopkRanker will maintain the order that's constructed here
        return CandidateSet(articles=selected)
