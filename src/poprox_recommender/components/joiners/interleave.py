from itertools import zip_longest

from lenskit.pipeline import Component

from poprox_concepts.domain import ImpressedSection


class Interleave(Component):
    config: None

    def __call__(self, recs1: ImpressedSection, recs2: ImpressedSection) -> ImpressedSection:
        impressions = []

        for pair in zip_longest(recs1.impressions, recs2.impressions):
            for impression in pair:
                if impression is not None:
                    impressions.append(impression)

        return ImpressedSection(impressions=impressions)
