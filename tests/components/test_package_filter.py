from uuid import uuid4

from poprox_concepts.domain import Article, ArticlePackage, CandidateSet, Entity
from poprox_recommender.components.filters import PackageFilter


def test_package_filter_selects_articles_from_matching_package():
    target_entity_id = uuid4()
    other_entity_id = uuid4()

    article1 = Article(article_id=uuid4(), headline="Article 1")
    article2 = Article(article_id=uuid4(), headline="Article 2")
    article3 = Article(article_id=uuid4(), headline="Article 3")
    article4 = Article(article_id=uuid4(), headline="Article 4")

    candidates = CandidateSet(articles=[article1, article2, article3, article4])

    # Create packages with seed entities
    target_entity = Entity(entity_id=target_entity_id, name="Technology", entity_type="topic", source="AP")
    other_entity = Entity(entity_id=other_entity_id, name="Sports", entity_type="topic", source="AP")

    packages = [
        ArticlePackage(
            title="Tech Package",
            source="AP",
            seed=target_entity,
            article_ids=[article1.article_id, article2.article_id],
        ),
        ArticlePackage(
            title="Sports Package",
            source="AP",
            seed=other_entity,
            article_ids=[article3.article_id],
        ),
    ]

    package_filter = PackageFilter(package_entity_id=target_entity_id)
    result = package_filter(candidate_articles=candidates, article_packages=packages)

    assert len(result.articles) == 2
    result_ids = {article.article_id for article in result.articles}
    assert article1.article_id in result_ids
    assert article2.article_id in result_ids


def test_package_filter_handles_multiple_matching_packages():
    target_entity_id = uuid4()

    article1 = Article(article_id=uuid4(), headline="Article 1")
    article2 = Article(article_id=uuid4(), headline="Article 2")
    article3 = Article(article_id=uuid4(), headline="Article 3")

    candidates = CandidateSet(articles=[article1, article2, article3])

    target_entity = Entity(entity_id=target_entity_id, name="Technology", entity_type="topic", source="AP")

    packages = [
        ArticlePackage(
            title="Package 1",
            source="AP",
            seed=target_entity,
            article_ids=[article1.article_id],
        ),
        ArticlePackage(
            title="Package 2",
            source="AP",
            seed=target_entity,
            article_ids=[article2.article_id, article3.article_id],
        ),
    ]

    package_filter = PackageFilter(package_entity_id=target_entity_id)
    result = package_filter(candidate_articles=candidates, article_packages=packages)

    assert len(result.articles) == 3


def test_package_filter_ignores_missing_articles():
    target_entity_id = uuid4()

    article1 = Article(article_id=uuid4(), headline="Article 1")
    article2 = Article(article_id=uuid4(), headline="Article 2")
    missing_article_id = uuid4()

    candidates = CandidateSet(articles=[article1, article2])

    target_entity = Entity(entity_id=target_entity_id, name="Technology", entity_type="topic", source="AP")

    packages = [
        ArticlePackage(
            title="Tech Package",
            source="AP",
            seed=target_entity,
            article_ids=[article1.article_id, missing_article_id, article2.article_id],
        ),
    ]

    package_filter = PackageFilter(package_entity_id=target_entity_id)
    result = package_filter(candidate_articles=candidates, article_packages=packages)

    assert len(result.articles) == 2


def test_package_filter_returns_empty_when_no_matches():
    target_entity_id = uuid4()
    other_entity_id = uuid4()

    article1 = Article(article_id=uuid4(), headline="Article 1")
    candidates = CandidateSet(articles=[article1])

    other_entity = Entity(entity_id=other_entity_id, name="Sports", entity_type="topic", source="AP")

    packages = [
        ArticlePackage(title="Sports Package", source="AP", seed=other_entity, article_ids=[article1.article_id])
    ]

    package_filter = PackageFilter(package_entity_id=target_entity_id)
    result = package_filter(candidate_articles=candidates, article_packages=packages)

    assert len(result.articles) == 0


def test_package_filter_handles_packages_without_seed():
    target_entity_id = uuid4()

    article1 = Article(article_id=uuid4(), headline="Article 1")
    article2 = Article(article_id=uuid4(), headline="Article 2")
    candidates = CandidateSet(articles=[article1, article2])

    packages = [
        ArticlePackage(title="Package 1", source="AP", seed=None, article_ids=[article1.article_id]),
        ArticlePackage(title="Package 2", source="AP", seed=None, article_ids=[article2.article_id]),
    ]

    package_filter = PackageFilter(package_entity_id=target_entity_id)
    result = package_filter(candidate_articles=candidates, article_packages=packages)

    assert len(result.articles) == 0
