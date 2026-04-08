import logging

import numpy as np
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet

logger = logging.getLogger(__name__)


class MentionedEntitiesConfig(BaseModel):
    entities: set[str]
    entity_type: str | None = None
    min_relevance: int = 0
    include: bool | None = True


class MentionedEntitiesFilter(Component):
    config: MentionedEntitiesConfig

    def __call__(self, candidates: CandidateSet) -> CandidateSet:
        kept_articles = []
        kept_scores = []
        for idx, article in enumerate(candidates.articles):
            # Check if the article mentions at least one of the entities
            mentioned = set(
                mention.entity.name
                for mention in article.mentions
                if mention.relevance >= self.config.min_relevance
                and (mention.entity.entity_type == self.config.entity_type or not self.config.entity_type)
            )
            overlap = self.config.entities.intersection(mentioned)

            # Keep or exclude articles that mention at least one of the entities
            # depending on the value of `self.config.include`
            if (len(overlap) > 0 and self.config.include) or (len(overlap) == 0 and not self.config.include):
                kept_articles.append(article)
                if hasattr(candidates, "scores") and candidates.scores is not None:
                    kept_scores.append(candidates.scores[idx])

        filtered = CandidateSet(articles=kept_articles)
        if kept_scores:
            filtered.scores = np.array(kept_scores)
        else:
            filtered.scores = None

        return filtered
