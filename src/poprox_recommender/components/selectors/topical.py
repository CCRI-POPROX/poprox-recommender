import hashlib
from datetime import date

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.sections.base import select_mentioning


class TopicalCandidatesConfig(BaseModel):
    min_candidates: int = 3
    random_seed: int = 22


class TopicalCandidates(Component):
    config: TopicalCandidatesConfig

    def __call__(
        self,
        candidate_set: CandidateSet,
        interest_profile: InterestProfile,
        sections: list[ImpressedSection] | None = None,
        today: date | None = None,
        descending: bool | None = True,
    ) -> CandidateSet:
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
            reverse=descending,
        )

        prev_section_seed_ids = [section.seed_entity_id for section in sections]
        sorted_interests = [i for i in sorted_interests if i.entity_id not in prev_section_seed_ids]

        candidates = CandidateSet(articles=[])

        for interest in sorted_interests:
            relevant_candidates = select_mentioning(candidate_set, [interest.entity_id])
            if len(relevant_candidates.articles) >= self.config.min_candidates:
                candidates = relevant_candidates
                candidates.seed_entity_id = interest.entity_id
                candidates.seed_entity_name = interest.entity_name
                candidates.seed_entity_type = interest.entity_type
                break

        return candidates

    def random_daily_seed(self, profile_id, day, base_seed: int) -> int:
        seed_str = f"{profile_id}_{day.isoformat()}_{base_seed}"
        hash_obj = hashlib.sha256(seed_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        return int(hash_hex, 16)
