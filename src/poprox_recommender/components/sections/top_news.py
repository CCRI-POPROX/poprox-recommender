import logging

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection
from poprox_recommender.components.joiners.fill import FillConfig, FillRecs
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker

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
        filtered: CandidateSet,
        unfiltered: CandidateSet,
        sections: list[ImpressedSection] | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        logger.info(f"Creating Top Stories section from {len(filtered.articles)} filtered candidates")

        filtered_config = TopkConfig(num_slots=self.config.max_articles)
        filtered_topk = TopkRanker(filtered_config)
        filtered_articles = filtered_topk(filtered)

        # The maximum overlap with the articles chosen above is self.config.max_articles,
        # so here we pull twice as many to cover the worst case
        unfiltered_config = TopkConfig(num_slots=self.config.max_articles * 2)
        unfiltered_topk = TopkRanker(unfiltered_config)
        unfiltered_articles = LazyShim(unfiltered_topk(unfiltered))

        joiner_config = FillConfig(num_slots=self.config.max_articles)
        joiner = FillRecs(joiner_config)
        top_section = joiner(filtered_articles, unfiltered_articles)

        top_section.title = "Your Top Stories"
        top_section.personalized = True

        if len(top_section.impressions) > 0:
            sections.append(top_section)

        return sections
