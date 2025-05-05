import logging
import unittest
from datetime import datetime
from uuid import UUID

from poprox_concepts.domain import Article, CandidateSet, Entity, Mention
from poprox_recommender.evaluation.metrics.rbe import rank_bias_entropy


def make_article(id_str, mentions=None):
    return Article(
        article_id=UUID(id_str),
        headline="",
        subhead=None,
        body=None,
        url=None,
        preview_image_id=None,
        mentions=mentions or [],
        source=None,
        external_id=None,
        raw_data=None,
        images=[],
        published_at=datetime.now(),
        created_at=None,
    )


class TestRankBiasEntropy(unittest.TestCase):
    def setUp(self):
        mention_a = Mention(
            entity=Entity(name="sports", entity_type="category", source="test"), relevance=1, source="test"
        )
        mention_b = Mention(
            entity=Entity(name="health", entity_type="category", source="test"), relevance=1, source="test"
        )
        mention_c = Mention(
            entity=Entity(name="tech", entity_type="category", source="test"), relevance=1, source="test"
        )

        self.m1 = make_article("00000000-0000-0000-0000-000000000001", mentions=[mention_a, mention_b])
        self.m2 = make_article("00000000-0000-0000-0000-000000000002", mentions=[mention_a])
        self.m3 = make_article("00000000-0000-0000-0000-000000000003", mentions=[mention_c, mention_b])
        self.m4 = make_article("00000000-0000-0000-0000-000000000004", mentions=[mention_a])

        self.all_articles = [self.m1, self.m2, self.m3]

    def test_entropy_with_same_mentions_present(self):
        ranked = CandidateSet(articles=[self.m2, self.m4])
        score = rank_bias_entropy(ranked, k=2)
        self.assertEqual(score, 0.0)

    def test_entropy_with_diverse_mentions(self):
        ranked = CandidateSet(articles=self.all_articles)
        score = rank_bias_entropy(ranked, k=3)
        self.assertEqual(score, 1.3250112108241772)

    def test_entropy_with_different_discount(self):
        ranked = CandidateSet(articles=self.all_articles)
        score = rank_bias_entropy(ranked, k=3, d=0.7)
        self.assertEqual(score, 1.4301561108018046)

    def test_entropy_all_empty(self):
        empty_set = CandidateSet(articles=[])
        score = rank_bias_entropy(empty_set, k=3)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
