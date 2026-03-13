import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.sections.base import select_from_candidates, select_mentioning

logger = logging.getLogger(__name__)


class InOtherNewsConfig(BaseModel):
    max_articles: int = 3


class InOtherNews(Component):
    config: InOtherNewsConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        article_packages: list[ArticlePackage],
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        topic_filter = TopicFilter()

        used_ids = set(impression.article.article_id for section in sections for impression in section.impressions)
        topic_seeds = [
            package.seed
            for package in article_packages
            for section in sections
            if section.seed_entity_id == package.seed.entity_id
        ]

        used_topic_articles = select_mentioning(candidate_set, topic_seeds)
        for article in used_topic_articles.articles:
            used_ids.add(article.article_id)

        topic_filtered = topic_filter(candidate_set, interest_profile)
        logger.info(f"Creating Other News section from {len(topic_filtered.articles)} filtered candidates")
        ranked_articles = select_from_candidates(topic_filtered, self.config.max_articles, used_ids)
        if len(ranked_articles) < self.config.max_articles:
            logger.info(f"Falling back to full pool of {len(candidate_set.articles)} candidates")
            ranked_articles = select_from_candidates(candidate_set, self.config.max_articles, used_ids)

        misc_section = ImpressedSection.from_articles(ranked_articles, title="In Other News", personalized=True)

        if len(misc_section.impressions) > 0:
            sections.append(misc_section)

        return sections
