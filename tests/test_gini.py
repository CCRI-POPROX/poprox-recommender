import logging
import unittest
from datetime import datetime
from uuid import UUID

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics.gini import gini_coeff

logger = logging.getLogger(__name__)


def make_article(id_str):
    return Article(
        article_id=UUID(id_str),
        headline="",
        subhead=None,
        body=None,
        url=None,
        preview_image_id=None,
        mentions=[],
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
        self.all_articles = [self.a1, self.a2, self.a3, self.a4, self.a5]

    def test_gini_uniform_distribution(self):
        duplicated_articles = [self.a1] * 10
        candidate_set = CandidateSet(articles=duplicated_articles)
        score = gini_coeff(candidate_set)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_gini_skewed_distribution(self):
        articles = [self.a1] * 15 + [self.a2, self.a3]
        candidate_set = CandidateSet(articles=articles)
        score = gini_coeff(candidate_set)
        self.assertGreater(score, 0.5)

    def test_gini_single_item(self):
        candidate_set = CandidateSet(articles=[self.a1])
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.0)

    def test_gini_empty_set(self):
        candidate_set = CandidateSet(articles=[])
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.0)

    def test_gini_known_values(self):
        article_ids = (
            [UUID("7cdf15b4-d3b3-4a1d-b1ce-9f41722a5d28")] * 5
            + [UUID("96289d55-7e71-4138-a5ce-8cb1b8c190b4")] * 3
            + [UUID("38e4bc07-0b10-48db-b3f6-fff4863bca7e")]
        )
        articles = [make_article(str(aid)) for aid in article_ids]
        candidate_set = CandidateSet(articles=articles)
        score = gini_coeff(candidate_set)
        self.assertEqual(score, 0.2962962962962963)


if __name__ == "__main__":
    unittest.main()
