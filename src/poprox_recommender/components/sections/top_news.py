import logging
from datetime import date

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillConfig, FillRecs
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker
from poprox_recommender.components.selectors.top_news import TopStoryCandidates

logger = logging.getLogger(__name__)


class PersonalizedTopNewsConfig(BaseModel):
    max_articles: int = 3


class LazyShim:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


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

        selector = TopStoryCandidates()
        top_articles = selector(candidate_set, article_packages)

        dup_filter = DuplicateFilter()
        deduped_top = dup_filter(top_articles, sections)

        topic_filter = TopicFilter()
        filtered_top = topic_filter(deduped_top, interest_profile)

        logger.info(f"Creating Top Stories section from {len(filtered_top.articles)} filtered candidates")

        filtered_config = TopkConfig(num_slots=self.config.max_articles)
        filtered_topk = TopkRanker(filtered_config)
        filtered_articles = filtered_topk(filtered_top)

        # The maximum overlap with the articles chosen above is self.config.max_articles,
        # so here we pull twice as many to cover the worst case
        unfiltered_config = TopkConfig(num_slots=self.config.max_articles * 2)
        unfiltered_topk = TopkRanker(unfiltered_config)
        unfiltered_articles = LazyShim(unfiltered_topk(deduped_top))

        joiner_config = FillConfig(num_slots=self.config.max_articles)
        joiner = FillRecs(joiner_config)
        top_section = joiner(filtered_articles, unfiltered_articles)

        top_section.title = "Your Top Stories"
        top_section.personalized = True

        if len(top_section.impressions) > 0:
            sections.append(top_section)

        return sections
