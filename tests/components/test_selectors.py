from uuid import uuid4

from poprox_concepts.domain import (
    Article,
    ArticlePackage,
    CandidateSet,
)
from poprox_recommender.components.selectors.top_news import TopStoryCandidates


def test_select_using_one_package():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)

    package_article_ids = [articles[1].article_id, articles[3].article_id]
    package = ArticlePackage(title="half the articles", source="test", article_ids=package_article_ids)

    selector = TopStoryCandidates()
    filtered = selector(candidates, [package])
    filtered_ids = [a.article_id for a in filtered.articles]

    for article_id in package_article_ids:
        assert article_id in filtered_ids


def test_select_using_multiple_packages():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)

    package_1 = ArticlePackage(title="half the articles", source="test", article_ids=[articles[1].article_id])
    package_2 = ArticlePackage(title="half the articles", source="test", article_ids=[articles[3].article_id])

    selector = TopStoryCandidates()
    filtered = selector(candidates, [package_1, package_2])
    filtered_ids = [a.article_id for a in filtered.articles]

    for article_id in package_1.article_ids + package_2.article_ids:
        assert article_id in filtered_ids
