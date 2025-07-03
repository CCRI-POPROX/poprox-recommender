import logging
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest

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


@pytest.fixture
def all_articles():
    return [make_article(f"00000000-0000-0000-0000-0000000000{str(i).zfill(2)}") for i in range(1, 16)]


def test_identical_lists(all_articles):
    reference = CandidateSet(articles=all_articles[:10])
    reranked = CandidateSet(articles=all_articles[:10])
    lip_score = least_item_promoted(reference, reranked, k=10)
    assert lip_score == 0.0


def test_mild_promotion_beyond_k(all_articles):
    reference = CandidateSet(articles=all_articles[:5])
    reranked = CandidateSet(articles=all_articles[2:5] + all_articles[0:2])
    lip_score = least_item_promoted(reference, reranked, k=3)
    assert lip_score == 1.0


def test_no_overlap(all_articles):
    reference = CandidateSet(articles=all_articles[:5])
    reranked = CandidateSet(articles=all_articles[5:10])
    lip_score = least_item_promoted(reference, reranked, k=3)
    assert lip_score == 0.0


def test_partial_overlap(all_articles):
    reference = CandidateSet(articles=all_articles[:5])
    reranked = CandidateSet(
        articles=[all_articles[1], all_articles[2], all_articles[6], all_articles[0], all_articles[4]]
    )
    lip_score = least_item_promoted(reference, reranked, k=3)
    assert lip_score == 0.0


def test_empty_reference(all_articles):
    reference = CandidateSet(articles=[])
    reranked = CandidateSet(articles=all_articles[:5])
    lip_score = least_item_promoted(reference, reranked, k=3)
    assert np.isnan(lip_score)


def test_empty_reranked(all_articles):
    reference = CandidateSet(articles=all_articles[:5])
    reranked = CandidateSet(articles=[])
    lip_score = least_item_promoted(reference, reranked, k=3)
    assert lip_score == 0.0
