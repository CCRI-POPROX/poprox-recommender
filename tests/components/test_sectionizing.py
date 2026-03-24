from uuid import uuid4

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker


def test_select_from_candidates_with_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles, scores=[0.1, 0.2, 0.3, 0.4])
    section = TopkRanker(TopkConfig(num_slots=3))(candidates)
    selected = [impression.article for impression in section.impressions]
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[3].article_id, articles[2].article_id, articles[1].article_id]


def test_select_from_candidates_without_scores():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)
    section = TopkRanker(TopkConfig(num_slots=3))(candidates)
    selected = [impression.article for impression in section.impressions]
    assert len(selected) == 3
    assert [a.article_id for a in selected] == [articles[0].article_id, articles[1].article_id, articles[2].article_id]
