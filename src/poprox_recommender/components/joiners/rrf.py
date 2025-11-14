from collections import defaultdict

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import ImpressedSection


class RRFConfig(BaseModel):
    num_slots: int
    k: int = 60


class ReciprocalRankFusion(Component):
    config: RRFConfig

    def __call__(self, recs1: ImpressedSection, recs2: ImpressedSection) -> ImpressedSection:
        article_scores = defaultdict(float)
        impressions_by_article_id = {}

        # Since impressions carry information about how articles are displayed,
        # we have to make a choice about which impression to prioritize. Inserting
        # impressions from `recs1` first means that we're prioritizing the impressions
        # from `recs2`, which will override impressions with the same `article_id`
        for i, impression in enumerate(recs1.impressions, 1):
            score = 1 / (i + self.config.k)
            article_scores[impression.article.article_id] = article_scores[impression.article.article_id] + score
            impressions_by_article_id[impression.article.article_id] = impression

        for i, impression in enumerate(recs2.impressions, 1):
            score = 1 / (i + self.config.k)
            article_scores[impression.article.article_id] = article_scores[impression.article.article_id] + score
            impressions_by_article_id[impression.article.article_id] = impression

        sorted_article_scores = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)

        sorted_article_ids = [article_id for article_id, article_score in sorted_article_scores]
        rrf_impressions = [
            impressions_by_article_id[article_id] for article_id in sorted_article_ids[: self.config.num_slots]
        ]

        return ImpressedSection(impressions=rrf_impressions)
