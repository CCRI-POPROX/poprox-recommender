import unittest
from datetime import datetime
from uuid import UUID

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics import k_coverage_score


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


class TestKCoverageScore(unittest.TestCase):
    def setUp(self):
        self.a1 = make_article("00000000-0000-0000-0000-000000000001")
        self.a2 = make_article("00000000-0000-0000-0000-000000000002")
        self.a3 = make_article("00000000-0000-0000-0000-000000000003")
        self.a4 = make_article("00000000-0000-0000-0000-000000000004")
        self.a5 = make_article("00000000-0000-0000-0000-000000000005")
        self.a6 = make_article("00000000-0000-0000-0000-000000000006")
        self.all_articles = [self.a1, self.a2, self.a3, self.a4, self.a5, self.a6]

    def test_full_coverage_k1(self):
        ranked = CandidateSet(articles=self.all_articles[:3])
        reranked = CandidateSet(articles=[self.a1] * 2 + [self.a2] * 2 + [self.a3])
        score = k_coverage_score(ranked, reranked, k=1)
        self.assertEqual(score, 1.0)

    def test_no_overlap(self):
        ranked = CandidateSet(articles=self.all_articles[:3])
        reranked = CandidateSet(articles=self.all_articles[3:6])
        score = k_coverage_score(ranked, reranked, k=1)
        self.assertEqual(score, 0.0)

    def test_empty_ranked(self):
        ranked = CandidateSet(articles=[])
        reranked = CandidateSet(articles=self.all_articles[:3])
        score = k_coverage_score(ranked, reranked, k=1)
        self.assertEqual(score, 0.0)

    def test_some_overlap_k1(self):
        ranked = CandidateSet(articles=self.all_articles[:5])
        reranked = CandidateSet(articles=self.all_articles[:3] + self.all_articles[:2])
        score = k_coverage_score(ranked, reranked, k=1)
        print(score)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
