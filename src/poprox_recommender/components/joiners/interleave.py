from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts.domain import ImpressedRecommendations


class Interleave(Component):
    config: None

    def __call__(self, recs1: ImpressedRecommendations, recs2: ImpressedRecommendations) -> ImpressedRecommendations:
        impressions = []

        position = 1
        for pair in zip_longest(recs1.impressions, recs2.impressions):
            for impression in pair:
                if impression is not None:
                    impression.position = position
                    impressions.append(impression)
                    position += 1

        return ImpressedRecommendations(impressions=impressions)
