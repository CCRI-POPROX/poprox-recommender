import logging
from datetime import datetime
from uuid import UUID

import pytest

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


@pytest.fixture
def all_articles():
    a1 = make_article("00000000-0000-0000-0000-000000000001")
    a2 = make_article("00000000-0000-0000-0000-000000000002")
    a3 = make_article("00000000-0000-0000-0000-000000000003")
    a4 = make_article("00000000-0000-0000-0000-000000000004")
    a5 = make_article("00000000-0000-0000-0000-000000000005")
    return [a1, a2, a3, a4, a5]


def test_gini_uniform_distribution(all_articles):
    duplicated_articles = [all_articles[0]] * 10
    candidate_set = CandidateSet(articles=duplicated_articles)
    score = gini_coeff(candidate_set)
    assert score == pytest.approx(0.0, rel=1e-5)


def test_gini_skewed_distribution(all_articles):
    articles = [all_articles[0]] * 15 + [all_articles[1], all_articles[2]]
    candidate_set = CandidateSet(articles=articles)
    score = gini_coeff(candidate_set)
    assert score > 0.5


def test_gini_single_item(all_articles):
    candidate_set = CandidateSet(articles=[all_articles[0]])
    score = gini_coeff(candidate_set)
    assert score == 0.0


def test_gini_empty_set():
    candidate_set = CandidateSet(articles=[])
    score = gini_coeff(candidate_set)
    assert score == 0.0


def test_gini_known_values():
    article_ids = (
        [UUID("7cdf15b4-d3b3-4a1d-b1ce-9f41722a5d28")] * 5
        + [UUID("96289d55-7e71-4138-a5ce-8cb1b8c190b4")] * 3
        + [UUID("38e4bc07-0b10-48db-b3f6-fff4863bca7e")]
    )
    articles = [make_article(str(aid)) for aid in article_ids]
    candidate_set = CandidateSet(articles=articles)
    score = gini_coeff(candidate_set)
    assert score == pytest.approx(0.2962962962962963, rel=1e-5)
