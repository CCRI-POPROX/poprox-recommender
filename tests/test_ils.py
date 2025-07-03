import logging
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest
import torch as th

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.evaluation.metrics.ils import intralist_similarity

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


def test_identical_embeddings(all_articles):
    embeddings = th.ones((10, 768))
    reference = CandidateSet(articles=all_articles[:10], embeddings=embeddings)
    ils_score = intralist_similarity(reference, k=10)
    assert ils_score == pytest.approx(1.0)


def test_orthogonal_embeddings(all_articles):
    embeddings = th.eye(5)
    reference = CandidateSet(articles=all_articles[:5], embeddings=embeddings)
    ils_score = intralist_similarity(reference, k=5)
    assert ils_score == 0.0


def test_partial_similarity(all_articles):
    embeddings = th.zeros((5, 768))
    embeddings[0] = th.ones(768)
    embeddings[1] = th.ones(768) * 0.5
    embeddings[2:] = th.randn(3, 768)
    reference = CandidateSet(articles=all_articles[:5], embeddings=embeddings)
    ils_score = intralist_similarity(reference, k=5)
    assert 0 < ils_score < 1.0


def test_single_article(all_articles):
    embeddings = th.ones((1, 768))
    reference = CandidateSet(articles=all_articles[:1], embeddings=embeddings)
    ils_score = intralist_similarity(reference, k=1)
    assert ils_score == 1.0


def test_empty_list(all_articles):
    reference = CandidateSet(articles=[], embeddings=th.zeros((0, 768)))
    ils_score = intralist_similarity(reference, k=0)
    assert ils_score == 1.0
