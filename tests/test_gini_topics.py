import logging
import unittest
from collections import Counter
from datetime import datetime
from unittest.mock import patch
from uuid import UUID

import numpy as np

from poprox_concepts.domain import Article, CandidateSet, Entity, Mention
from poprox_recommender.evaluation.metrics.gini import gini_coeff

logger = logging.getLogger(__name__)


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


class TestGiniCoefficient(unittest.TestCase):
    def setUp(self):
        self.a1 = make_article("00000000-0000-0000-0000-000000000001")
        self.a2 = make_article("00000000-0000-0000-0000-000000000002")
        self.a3 = make_article("00000000-0000-0000-0000-000000000003")
        self.a4 = make_article("00000000-0000-0000-0000-000000000004")
        self.a5 = make_article("00000000-0000-0000-0000-000000000005")
        mention_a = Mention(
            entity=Entity(name="sports", entity_type="category", source="test"), relevance=1, source="test"
        )
        mention_b = Mention(
            entity=Entity(name="news", entity_type="category", source="test"), relevance=1, source="test"
        )
        mention_c = Mention(
            entity=Entity(name="tech", entity_type="category", source="test"), relevance=1, source="test"
        )

        self.m1 = make_article("00000000-0000-0000-0000-000000000006", mentions=[mention_a])
        self.m2 = make_article("00000000-0000-0000-0000-000000000007", mentions=[mention_b])
        self.m3 = make_article("00000000-0000-0000-0000-000000000008", mentions=[mention_c])

    def test_gini_using_existing_mentions(self):
        candidate_set = CandidateSet(articles=[self.m1] * 3 + [self.m2] * 3 + [self.m3])
        score = gini_coeff(candidate_set)
        self.assertAlmostEqual(score, 0.190476, places=5)

    def test_gini_uniform_distribution(self):
        duplicated_articles = [self.m1] * 10
        candidate_set = CandidateSet(articles=duplicated_articles)
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.0)

    def test_gini_empty_set(self):
        candidate_set = CandidateSet(articles=[])
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.0)

    def test_gini_single_article(self):
        candidate_set = CandidateSet(articles=[self.m1])
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
