from uuid import UUID

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection


class TopNewsPlacerConfig(BaseModel):
    max_top_news: int = 3
    total_slots: int = 12
    top_news_entity_id: UUID | None = None
    add_section_metadata: bool = True


class TopNewsPlacer(Component):
    config: TopNewsPlacerConfig

    def __call__(
        self,
        ranked_articles: ImpressedSection,
        top_news_article_ids: set[UUID] | None = None,
        top_news_candidates: CandidateSet | None = None,
    ) -> list[ImpressedSection]:
        if not ranked_articles.impressions:
            return []

        if top_news_candidates is not None and top_news_article_ids is None:
            top_news_article_ids = {article.article_id for article in top_news_candidates.articles}
        elif top_news_article_ids is None:
            top_news_article_ids = set()

        # extract articles from impressions
        all_articles = [imp.article for imp in ranked_articles.impressions]

        top_news_articles = []
        regular_articles = []

        for article in all_articles:
            if article.article_id in top_news_article_ids:
                top_news_articles.append(article)
            else:
                regular_articles.append(article)

        # take up to max_top_news articles
        selected_top_news = top_news_articles[: self.config.max_top_news]

        # calculate remaining slots for personalized content
        remaining_slots = self.config.total_slots - len(selected_top_news)
        selected_personalized = regular_articles[:remaining_slots]

        # create seperate sections
        sections = []

        # top news section
        if selected_top_news:
            top_news_section = ImpressedSection.from_articles(selected_top_news)
            if self.config.add_section_metadata:
                top_news_section.title = "Top News"
                top_news_section.personalized = True
                if self.config.top_news_entity_id:
                    top_news_section.seed_entity_id = self.config.top_news_entity_id
            sections.append(top_news_section)

        # rest of the personalized section
        if selected_personalized:
            personalized_section = ImpressedSection.from_articles(selected_personalized)
            if self.config.add_section_metadata:
                personalized_section.title = "For You"
                personalized_section.personalized = True
            sections.append(personalized_section)

        return sections
