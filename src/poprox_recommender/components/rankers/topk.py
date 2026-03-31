import re

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, ImpressedSection


class TopkConfig(BaseModel):
    num_slots: int = 10


class TopkRanker(Component):
    config: TopkConfig

    def __call__(self, candidate_articles: CandidateSet) -> ImpressedSection:
        ranked_articles = []

        if hasattr(candidate_articles, "scores") and candidate_articles.scores is not None:
            # rank candidates by score if scores are available
            sorted_indices = np.argsort(np.array(candidate_articles.scores))[::-1]
            ranked_articles = [candidate_articles.articles[int(i)] for i in sorted_indices][: self.config.num_slots]
        else:
            # otherwise select from the top of the list of candidates preserving order
            ranked_articles = candidate_articles.articles[: self.config.num_slots]

        # Python's native title() method struggles with unpunctuated abbreviations
        # like "US" and lowercases everything but the first letter. This custom
        # version uppercases the first letter of each word without changing others
        titleized_name = re.sub(
            r"(?:(?<=\W)|^)\w", lambda x: x.group(0).upper(), candidate_articles.seed_entity_name or ""
        )

        return ImpressedSection.from_articles(
            ranked_articles,
            title=titleized_name,
            seed_entity_id=candidate_articles.seed_entity_id,
        )
