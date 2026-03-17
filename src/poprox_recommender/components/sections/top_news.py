import logging
from datetime import date

from lenskit.pipeline import Component
from pydantic import BaseModel
from recommender.src.poprox_recommender.components.selectors.top_news import TopStoryCandidates

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.sections.base import select_from_candidates

logger = logging.getLogger(__name__)


class PersonalizedTopNewsConfig(BaseModel):
    max_articles: int = 3


class PersonalizedTopNews(Component):
    config: PersonalizedTopNewsConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)

        selector = TopStoryCandidates()
        top_articles = selector(candidate_set, article_packages)

        topic_filter = TopicFilter()
        filtered_top = topic_filter(top_articles, interest_profile)

        logger.info(f"Creating Top Stories section from {len(filtered_top.articles)} filtered candidates")
        ranked_articles = select_from_candidates(filtered_top, self.config.max_articles, used_ids)

        if len(ranked_articles) < self.config.max_articles:
            logger.info(f"Falling back to full pool of {len(candidate_set.articles)} top candidates")
            ranked_articles = select_from_candidates(top_articles, self.config.max_articles, used_ids)

        top_section = ImpressedSection.from_articles(ranked_articles, title="Your Top Stories", personalized=True)

        if len(top_section.impressions) > 0:
            used_ids.update(a.article_id for a in ranked_articles)
            sections.append(top_section)

        return sections
