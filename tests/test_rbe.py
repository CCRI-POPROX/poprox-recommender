import logging
from datetime import datetime
from uuid import UUID

import pytest

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


@pytest.fixture
def test_data():
    mention_a = Mention(entity=Entity(name="sports", entity_type="category", source="test"), relevance=1, source="test")
    mention_b = Mention(entity=Entity(name="health", entity_type="category", source="test"), relevance=1, source="test")
    mention_c = Mention(entity=Entity(name="tech", entity_type="category", source="test"), relevance=1, source="test")

    m1 = make_article("00000000-0000-0000-0000-000000000001", mentions=[mention_a, mention_b])
    m2 = make_article("00000000-0000-0000-0000-000000000002", mentions=[mention_a])
    m3 = make_article("00000000-0000-0000-0000-000000000003", mentions=[mention_c, mention_b])
    m4 = make_article("00000000-0000-0000-0000-000000000004", mentions=[mention_a])

    return {
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
        "all_articles": [m1, m2, m3],
    }


def test_entropy_with_same_mentions_present(test_data):
    ranked = CandidateSet(articles=[test_data["m2"], test_data["m4"]])
    score = rank_bias_entropy(ranked, k=2)
    assert score == 0.0


def test_entropy_with_diverse_mentions(test_data):
    ranked = CandidateSet(articles=test_data["all_articles"])
    score = rank_bias_entropy(ranked, k=3)
    assert score == pytest.approx(1.484630757682105)


def test_entropy_with_different_discount(test_data):
    ranked = CandidateSet(articles=test_data["all_articles"])
    score = rank_bias_entropy(ranked, k=3, d=0.7)
    assert score == pytest.approx(1.4301561108018046)


def test_entropy_all_empty():
    empty_set = CandidateSet(articles=[])
    score = rank_bias_entropy(empty_set, k=3)
    assert score == 0.0
