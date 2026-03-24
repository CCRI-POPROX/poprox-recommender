import logging
from datetime import date

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker
from poprox_recommender.components.sections.combine import AddSection, AddSectionConfig
from poprox_recommender.components.selectors.topical import TopicalCandidates, TopicalCandidatesConfig

logger = logging.getLogger(__name__)


class TopicalSectionConfig(BaseModel):
    max_articles_per_topic: int = 3
    random_seed: int = 22


class TopicalSection(Component):
    config: TopicalSectionConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
    ) -> list[ImpressedSection]:
        sections = sections or []

        dedup_filter = DuplicateFilter()
        deduped_candidates = dedup_filter(candidate_set, sections)

        config = TopicalCandidatesConfig(
            min_candidates=self.config.max_articles_per_topic, random_seed=self.config.random_seed
        )
        selector = TopicalCandidates(config)
        candidates = selector(deduped_candidates, interest_profile, sections, today)

        ranker = TopkRanker(TopkConfig(num_slots=self.config.max_articles_per_topic))
        topic_section = ranker(candidates)

        config = AddSectionConfig(
            title=candidates.seed_entity_name, seed_entity_id=candidates.seed_entity_id, personalized=True
        )
        sections = AddSection(config).__call__(topic_section, sections)

        return sections
