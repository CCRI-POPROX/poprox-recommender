import logging
import random
import unittest
from datetime import datetime, timezone
from uuid import UUID

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics import rank_biased_overlap

logger = logging.getLogger(__name__)


# Load articles from file
def make_article(id_str):
    return Article(
        article_id=UUID(id_str),
        headline="Test Headline",
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


class TestRBOMetric(unittest.TestCase):
    def setUp(self):
        self.all_articles = [make_article(f"00000000-0000-0000-0000-0000000000{str(i).zfill(2)}") for i in range(1, 16)]
        (
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.a5,
            self.a6,
            self.a7,
            self.a8,
            self.a9,
            self.a10,
            self.a11,
            self.a12,
            self.a13,
            self.a14,
            self.a15,
        ) = self.all_articles

    def test_rbo_identical_lists(self):
        candidate_set_a = CandidateSet(articles=self.all_articles[:10])
        candidate_set_b = CandidateSet(articles=self.all_articles[:10])
        rbo_score = rank_biased_overlap(candidate_set_a, candidate_set_b)
        self.assertAlmostEqual(rbo_score, 1.0, places=10)

    def test_rbo_partial_overlap_1(self):
        original = self.all_articles[:5]
        partial = self.all_articles[3:8]
        recs_a = CandidateSet(articles=original)
        recs_b = CandidateSet(articles=partial)
        rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.9, k=5)
        self.assertAlmostEqual(rbo_score, 0.1085907548, places=5)

    def test_rbo_partial_overlap_2(self):
        original = self.all_articles[:2] + self.all_articles[10:13]
        partial = self.all_articles[:5]
        recs_a = CandidateSet(articles=original)
        recs_b = CandidateSet(articles=partial)
        rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.9, k=5)
        self.assertAlmostEqual(rbo_score, 0.7489292081, places=5)

    def test_rbo_different_discount(self):
        original = self.all_articles[:2] + self.all_articles[10:13]
        partial = self.all_articles[:5]
        recs_a = CandidateSet(articles=original)
        recs_b = CandidateSet(articles=partial)
        rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.5, k=5)
        self.assertAlmostEqual(rbo_score, 0.9053763441, places=5)

    # def test_rbo_list_len(self):
    #     original = self.all_articles[:5]
    #     partial = self.all_articles[:2]
    #     recs_a = CandidateSet(articles=original)
    #     recs_b = CandidateSet(articles=partial)
    #     with self.assertRaises(Exception):
    #         rank_biased_overlap(recs_a, recs_b, p=0.5, k=5)

    def test_rbo_p_zero(self):
        original = self.all_articles[:5]
        partial = self.all_articles[:2]
        recs_a = CandidateSet(articles=original)
        recs_b = CandidateSet(articles=partial)
        with self.assertRaises(Exception):
            rank_biased_overlap(recs_a, recs_b, p=0, k=5)

    def test_rbo_no_overlap(self):
        candidate_set_a = CandidateSet(articles=self.all_articles[:5])
        candidate_set_b = CandidateSet(articles=self.all_articles[5:10])
        rbo_score = rank_biased_overlap(candidate_set_a, candidate_set_b, p=0.9, k=5)
        self.assertAlmostEqual(rbo_score, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
