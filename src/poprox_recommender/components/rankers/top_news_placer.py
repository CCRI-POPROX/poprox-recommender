from uuid import UUID

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection


class TopNewsPlacerConfig(BaseModel):
    max_top_news: int = 3  # Maximum top news to place at beginning (positions 1-3)
    total_slots: int = 12


class TopNewsPlacer(Component):
    config: TopNewsPlacerConfig

    def __call__(
        self,
        ranked_articles: ImpressedSection,
        top_news_article_ids: set[UUID] | None = None,
        top_news_candidates: CandidateSet | None = None,
    ) -> ImpressedSection:
        if not ranked_articles.impressions:
            return ImpressedSection.from_articles([])

        if top_news_candidates is not None and top_news_article_ids is None:
            top_news_article_ids = {article.article_id for article in top_news_candidates.articles}
        elif top_news_article_ids is None:
            top_news_article_ids = set()  # No top news

        # Extract articles from impressions
        all_articles = [imp.article for imp in ranked_articles.impressions]

        # Separate top news from regular articles
        top_news_articles = []
        regular_articles = []

        for article in all_articles:
            if article.article_id in top_news_article_ids:
                top_news_articles.append(article)
            else:
                regular_articles.append(article)

        # Take up to max_top_news articles
        selected_top_news = top_news_articles[: self.config.max_top_news]

        # Build final list: top news first, then regular articles
        final_recommendations = selected_top_news + regular_articles

        # Trim to total_slots
        final_recommendations = final_recommendations[: self.config.total_slots]

        return ImpressedSection.from_articles(final_recommendations)
