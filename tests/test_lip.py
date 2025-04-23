import logging
import unittest
from datetime import datetime
from uuid import UUID

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics import least_item_promoted

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


class TestLeastItemPromoted(unittest.TestCase):
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

    def test_identical_lists(self):
        reference = CandidateSet(articles=self.all_articles[:10])
        reranked = CandidateSet(articles=self.all_articles[:10])
        lip_score = least_item_promoted(reference, reranked, k=10)
        self.assertEqual(lip_score, 0.0)

    def test_mild_promotion_beyond_k(self):
        reference = CandidateSet(articles=self.all_articles[:5])
        reranked = CandidateSet(articles=self.all_articles[2:5] + self.all_articles[0:2])
        lip_score = least_item_promoted(reference, reranked, k=3)
        self.assertAlmostEqual(lip_score, 0.4, places=5)

    def test_no_overlap(self):
        reference = CandidateSet(articles=self.all_articles[:5])
        reranked = CandidateSet(articles=self.all_articles[5:10])
        lip_score = least_item_promoted(reference, reranked, k=3)
        expected = (4 - 3) / 5
        self.assertAlmostEqual(lip_score, expected, places=5)

    def test_partial_overlap(self):
        reference = CandidateSet(articles=self.all_articles[:5])
        reranked = CandidateSet(
            articles=[
                self.all_articles[1],
                self.all_articles[2],
                self.all_articles[6],
                self.all_articles[0],
                self.all_articles[4],
            ]
        )
        lip_score = least_item_promoted(reference, reranked, k=3)
        print("test_partial_overlap", lip_score)

        self.assertEqual(0.4, lip_score)

    def test_empty_reference(self):
        reference = CandidateSet(articles=[])
        reranked = CandidateSet(articles=self.all_articles[:5])
        lip_score = least_item_promoted(reference, reranked, k=3)
        self.assertEqual(lip_score, 0.0)

    def test_empty_reranked(self):
        reference = CandidateSet(articles=self.all_articles[:5])
        reranked = CandidateSet(articles=[])
        lip_score = least_item_promoted(reference, reranked, k=3)
        print("test_empty_reranked", lip_score)
        expected = (4 - 3) / 5
        self.assertAlmostEqual(lip_score, expected, places=5)


if __name__ == "__main__":
    unittest.main()
