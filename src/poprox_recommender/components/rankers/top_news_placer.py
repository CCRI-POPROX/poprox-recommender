import logging
from uuid import UUID

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection
from poprox_recommender.components.filters import PackageFilter, PackageFilterConfig

logger = logging.getLogger(__name__)


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
        article_packages: list[ArticlePackage],
    ) -> list[ImpressedSection]:
        if not ranked_articles.impressions:
            logger.debug("No ranked articles.")
            return []

        candidate_set = CandidateSet(articles=[imp.article for imp in ranked_articles.impressions])

        # filter top news articles
        package_filter = PackageFilter(config=PackageFilterConfig(package_entity_id=self.config.top_news_entity_id))
        filtered_candidates = package_filter(candidate_set, article_packages)

        top_news_articles = filtered_candidates.articles[: self.config.max_top_news]
        selected_ids = {a.article_id for a in top_news_articles}

        remaining_articles = [a for a in candidate_set.articles if a.article_id not in selected_ids]
        remaining_slots = self.config.total_slots - len(top_news_articles)
        selected_personalized = remaining_articles[:remaining_slots]

        sections = []

        # top news section
        if top_news_articles:
            top_news_section = ImpressedSection.from_articles(top_news_articles)
            if self.config.add_section_metadata:
                top_news_section.title = "Top News"
                top_news_section.personalized = True
                top_news_section.seed_entity_id = self.config.top_news_entity_id
            sections.append(top_news_section)

        # personalized section
        if selected_personalized:
            personalized_section = ImpressedSection.from_articles(selected_personalized)
            if self.config.add_section_metadata:
                personalized_section.title = "For You"
                personalized_section.personalized = True
            sections.append(personalized_section)

        logger.debug(
            "TopNewsPlacer created %d sections with %d top news and %d personalized articles",
            len(sections),
            len(top_news_articles),
            len(selected_personalized),
        )

        return sections
