import logging
from datetime import datetime
from uuid import UUID

import pytest

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


@pytest.fixture
def test_data():
    mention_a = Mention(entity=Entity(name="sports", entity_type="category", source="test"), relevance=1, source="test")
    mention_b = Mention(entity=Entity(name="news", entity_type="category", source="test"), relevance=1, source="test")
    mention_c = Mention(entity=Entity(name="tech", entity_type="category", source="test"), relevance=1, source="test")

    m1 = make_article("00000000-0000-0000-0000-000000000006", mentions=[mention_a])
    m2 = make_article("00000000-0000-0000-0000-000000000007", mentions=[mention_b])
    m3 = make_article("00000000-0000-0000-0000-000000000008", mentions=[mention_c])

    return {"m1": m1, "m2": m2, "m3": m3}


def test_gini_using_existing_mentions(test_data):
    candidate_set = CandidateSet(articles=[test_data["m1"]] * 3 + [test_data["m2"]] * 3 + [test_data["m3"]])
    score = gini_coeff(candidate_set)
    assert score == pytest.approx(0.190476, rel=1e-5)


def test_gini_uniform_distribution(test_data):
    duplicated_articles = [test_data["m1"]] * 10
    candidate_set = CandidateSet(articles=duplicated_articles)
    score = gini_coeff(candidate_set)
    assert score == 0.0


def test_gini_empty_set():
    candidate_set = CandidateSet(articles=[])
    score = gini_coeff(candidate_set)
    assert score == 0.0


def test_gini_single_article(test_data):
    candidate_set = CandidateSet(articles=[test_data["m1"]])
    score = gini_coeff(candidate_set)
    assert score == 0.0
