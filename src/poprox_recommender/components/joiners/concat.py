from lenskit.pipeline import Component

from poprox_concepts.domain import ImpressedSection


class Concatenate(Component):
    config: None

    def __call__(self, recs1: ImpressedSection, recs2: ImpressedSection) -> ImpressedSection:
        """
        Concatenates two sets of candidates, while deduplicating them, keeping the
        first occurrence of each article (by id), and maintaining their original order.

        This is achieved by inserting articles into a dict in reverse order, so that
        articles from the second candidate set are written first and then overwritten
        by articles from the first candidate set (if there are collisions.) Afterward,
        the dict keys can be ignored and the dict values are the deduplicated candidates
        in reverse order. Reversing them one more time returns them to the original order.
        """
        reverse_impressions = reversed(recs1.impressions + recs2.impressions)
        impressions = {impression.article.article_id: impression for impression in reverse_impressions}.values()
        unreversed_impressions = list(reversed(impressions))

        return ImpressedSection(impressions=unreversed_impressions)
