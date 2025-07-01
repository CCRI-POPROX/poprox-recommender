from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts import CandidateSet
from poprox_concepts.api.recommendations.v3 import RecommendationList_v3, RecommendationResponseSection
from poprox_concepts.domain import RecommendationList
from poprox_concepts.domain.newsletter import Impression


class SectionConfig(BaseModel):
    title: str


class SingleSection(Component):
    config: SectionConfig

    def __call__(self, recs1: RecommendationList) -> list[RecommendationResponseSection]:
        articles = recs1.articles
        extras = recs1.extras

        impressions = [Impression(newsletter_id=None, position=i, article=articles[i]) for i in range(len(articles))]

        recs_list = RecommendationList_v3(impressions=impressions, extras=extras)
        return [RecommendationResponseSection(title=self.config.title, recommendations=recs_list)]
