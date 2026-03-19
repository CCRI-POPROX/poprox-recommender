import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.duplicate import DuplicateFilter
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

        dup_filter = DuplicateFilter()
        deduped_candidates = dup_filter(candidate_set, sections)

        topic_seeds = [
            package.seed
            for package in article_packages
            for section in sections
            if section.seed_entity_id == package.seed.entity_id
        ]

        used_topic_articles = select_mentioning(deduped_candidates, topic_seeds)
        used_ids = set(article.article_id for article in used_topic_articles.articles)

        topic_filter = TopicFilter()
        topic_filtered = topic_filter(deduped_candidates, interest_profile)

        logger.info(f"Creating Other News section from {len(topic_filtered.articles)} filtered candidates")
        ranked_articles = select_from_candidates(topic_filtered, self.config.max_articles, used_ids)
        if len(ranked_articles) < self.config.max_articles:
            logger.info(f"Falling back to full pool of {len(deduped_candidates.articles)} candidates")
            ranked_articles = select_from_candidates(deduped_candidates, self.config.max_articles, used_ids)

        misc_section = ImpressedSection.from_articles(ranked_articles, title="In Other News", personalized=True)

        if len(misc_section.impressions) > 0:
            sections.append(misc_section)

        return sections
