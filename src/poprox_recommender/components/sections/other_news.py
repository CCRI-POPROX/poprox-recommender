import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ArticlePackage, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.filters.seeds import PreviousSectionsFilter
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillConfig, FillRecs
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker

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

        section_filter = PreviousSectionsFilter()
        narrow_candidates = section_filter(candidate_set, article_packages, sections)

        dup_filter = DuplicateFilter()
        deduped_candidates = dup_filter(narrow_candidates, sections)

        topic_filter = TopicFilter()
        topic_filtered = topic_filter(deduped_candidates, interest_profile)

        filtered_config = TopkConfig(num_slots=self.config.max_articles)
        filtered_topk = TopkRanker(filtered_config)
        filtered_articles = filtered_topk(topic_filtered)

        # The maximum overlap with the articles chosen above is self.config.max_articles,
        # so here we pull twice as many to cover the worst case
        unfiltered_config = TopkConfig(num_slots=self.config.max_articles * 2)
        unfiltered_topk = TopkRanker(unfiltered_config)
        unfiltered_articles = unfiltered_topk(deduped_candidates)

        joiner_config = FillConfig(num_slots=self.config.max_articles)
        joiner = FillRecs(joiner_config)
        other_news_section = joiner(filtered_articles, unfiltered_articles)

        other_news_section.title = "In Other News"
        other_news_section.personalized = True

        if len(other_news_section.impressions) > 0:
            sections.append(other_news_section)

        return sections
