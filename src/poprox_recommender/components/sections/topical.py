import hashlib
import logging
from datetime import date

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker
from poprox_recommender.components.sections.base import select_mentioning

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

        if today is None:
            today = date.today()

        random_seed = self.random_daily_seed(interest_profile.profile_id, today, self.config.random_seed)

        topical_interests = list(interest_profile.interests_by_type("topic"))
        rng = np.random.default_rng(random_seed)
        rng.shuffle(topical_interests)
        sorted_interests = sorted(
            topical_interests,
            key=lambda i: i.preference,
            reverse=True,
        )

        dedup_filter = DuplicateFilter()
        deduped_candidates = dedup_filter(candidate_set, sections)

        prev_section_seed_ids = [section.seed_entity_id for section in sections]
        sorted_interests = [i for i in sorted_interests if i.entity_id not in prev_section_seed_ids]

        interest = None
        candidates = None

        for potential_interest in sorted_interests:
            relevant_candidates = select_mentioning(deduped_candidates, [potential_interest.entity_id])
            if len(relevant_candidates.articles) >= self.config.max_articles_per_topic:
                interest = potential_interest
                candidates = relevant_candidates
                break

        if interest and candidates:
            logger.info(f"Creating {interest.entity_name} section from {len(candidates.articles)} topical candidates")

            ranker = TopkRanker(TopkConfig(num_slots=self.config.max_articles_per_topic))
            topic_section = ranker(candidates)

            topic_section.title = interest.entity_name
            topic_section.personalized = True
            topic_section.seed_entity_id = interest.entity_id

            sections.append(topic_section)

        return sections

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return int(hash_hex, 16)
