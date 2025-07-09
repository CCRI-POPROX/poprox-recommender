import logging
import random
from datetime import datetime, timezone
from uuid import UUID

import pytest

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics import rank_biased_overlap

logger = logging.getLogger(__name__)


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


@pytest.fixture
def all_articles():
    return [make_article(f"00000000-0000-0000-0000-0000000000{str(i).zfill(2)}") for i in range(1, 16)]


def test_rbo_identical_lists(all_articles):
    candidate_set_a = CandidateSet(articles=all_articles[:10])
    candidate_set_b = CandidateSet(articles=all_articles[:10])
    rbo_score = rank_biased_overlap(candidate_set_a, candidate_set_b)
    assert rbo_score == pytest.approx(1.0, rel=1e-10)


def test_rbo_partial_overlap_1(all_articles):
    original = all_articles[:5]
    partial = all_articles[3:8]
    recs_a = CandidateSet(articles=original)
    recs_b = CandidateSet(articles=partial)
    rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.9, k=5)
    assert rbo_score == pytest.approx(0.1085907548, rel=1e-5)


def test_rbo_partial_overlap_2(all_articles):
    original = all_articles[:2] + all_articles[10:13]
    partial = all_articles[:5]
    recs_a = CandidateSet(articles=original)
    recs_b = CandidateSet(articles=partial)
    rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.9, k=5)
    assert rbo_score == pytest.approx(0.7489292081, rel=1e-5)


def test_rbo_different_discount(all_articles):
    original = all_articles[:2] + all_articles[10:13]
    partial = all_articles[:5]
    recs_a = CandidateSet(articles=original)
    recs_b = CandidateSet(articles=partial)
    rbo_score = rank_biased_overlap(recs_a, recs_b, p=0.5, k=5)
    assert rbo_score == pytest.approx(0.9053763441, rel=1e-5)


def test_rbo_p_zero(all_articles):
    original = all_articles[:5]
    partial = all_articles[:2]
    recs_a = CandidateSet(articles=original)
    recs_b = CandidateSet(articles=partial)
    with pytest.raises(Exception):
        rank_biased_overlap(recs_a, recs_b, p=0, k=5)


def test_rbo_no_overlap(all_articles):
    candidate_set_a = CandidateSet(articles=all_articles[:5])
    candidate_set_b = CandidateSet(articles=all_articles[5:10])
    rbo_score = rank_biased_overlap(candidate_set_a, candidate_set_b, p=0.9, k=5)
    assert rbo_score == pytest.approx(0.0, rel=1e-5)
