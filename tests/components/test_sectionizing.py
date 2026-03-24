from uuid import uuid4

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.components.sections.base import select_from_candidates


def test_select_from_candidates_with_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles, scores=[0.1, 0.2, 0.3, 0.4])
    selected = select_from_candidates(candidates, 3)
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[3].article_id, articles[2].article_id, articles[1].article_id]


def test_select_from_candidates_excluding_with_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles, scores=[0.1, 0.2, 0.3, 0.4])
    selected = select_from_candidates(candidates, 3, excluding=[articles[3].article_id])
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[2].article_id, articles[1].article_id, articles[0].article_id]


def test_select_from_candidates_without_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)
    selected = select_from_candidates(candidates, 3)
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[0].article_id, articles[1].article_id, articles[2].article_id]


def test_select_from_candidates_excluding_without_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)
    selected = select_from_candidates(candidates, 3, excluding=[articles[0].article_id])
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[1].article_id, articles[2].article_id, articles[3].article_id]
