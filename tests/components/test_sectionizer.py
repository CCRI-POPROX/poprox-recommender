from uuid import UUID, uuid4

from poprox_concepts.domain import AccountInterest, Article, ArticlePackage, CandidateSet, Entity, InterestProfile
from poprox_recommender.components.rankers.sectionizer import (
    Sectionizer,
    SectionizerConfig,
    filter_using_packages,
    select_from_candidates,
)


def make_interest_profile(topic_ids: list[UUID]) -> InterestProfile:
    interests = [
        AccountInterest(
            entity_id=tid,
            entity_name=f"Topic {i}",
            entity_type="topic",
            preference=5 - i,
        )
        for i, tid in enumerate(topic_ids)
    ]
    return InterestProfile(entity_interests=interests, click_history=[])


def make_package(entity_id, title, articles):
    return ArticlePackage(
        title=title,
        source="AP",
        seed=Entity(entity_id=entity_id, name=title, entity_type="topic", source="AP"),
        article_ids=[a.article_id for a in articles],
    )


def test_sectionizer_creates_sections():
    top_news_id = uuid4()
    topic1_id, topic2_id = uuid4(), uuid4()
    articles = [Article(article_id=uuid4(), headline=f"Article {i}") for i in range(1, 7)]
    candidates = CandidateSet(articles=articles)
    packages = [
        make_package(top_news_id, "Top News", articles[:2]),
        make_package(topic1_id, "Sports", articles[2:4]),
        make_package(topic2_id, "Tech", articles[4:6]),
    ]

    # user likes both topics
    profile = make_interest_profile([topic1_id, topic2_id])

    config = SectionizerConfig(
        top_news_entity_id=top_news_id,
        max_top_news=2,
        max_topic_sections=2,
        max_articles_per_topic=2,
        max_misc_articles=2,
    )

    sectionizer = Sectionizer(config=config)
    sections = sectionizer(candidate_set=candidates, article_packages=packages, interest_profile=profile)

    # there should be 3 sections
    assert len(sections) == 3
    titles = [s.title for s in sections]
    assert "Top News" in titles
    assert "Sports" in titles
    assert "Tech" in titles


def test_sectionizer_creates_misc_section():
    top_news_id = uuid4()
    topic_id = uuid4()

    # articles
    a1, a2, a3 = (
        Article(article_id=uuid4(), headline="A1"),
        Article(article_id=uuid4(), headline="A2"),
        Article(article_id=uuid4(), headline="A3"),
    )
    candidates = CandidateSet(articles=[a1, a2, a3])

    packages = [make_package(top_news_id, "Top News", [a1])]
    profile = make_interest_profile([topic_id])

    config = SectionizerConfig(
        top_news_entity_id=top_news_id, max_top_news=1, max_topic_sections=1, max_misc_articles=2
    )

    sectionizer = Sectionizer(config=config)
    sections = sectionizer(candidate_set=candidates, article_packages=packages, interest_profile=profile)

    assert len(sections) == 2
    assert sections[0].title == "Top News"
    assert sections[1].title == "In Other News"
    used_ids = {imp.article.article_id for s in sections for imp in s.impressions}
    assert a1.article_id in used_ids
    assert a2.article_id in used_ids
    assert a3.article_id in used_ids


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


def test_filter_using_one_package():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)

    package_article_ids = [articles[1].article_id, articles[3].article_id]
    package = ArticlePackage(title="half the articles", source="test", article_ids=package_article_ids)

    filtered = filter_using_packages(candidates, [package])
    filtered_ids = [a.article_id for a in filtered.articles]

    for article_id in package_article_ids:
        assert article_id in filtered_ids


def test_filter_using_multiple_packages():
    articles = [
        Article(article_id=uuid4(), headline="Article 1"),
        Article(article_id=uuid4(), headline="Article 2"),
        Article(article_id=uuid4(), headline="Article 3"),
        Article(article_id=uuid4(), headline="Article 4"),
    ]
    candidates = CandidateSet(articles=articles)

    package_1 = ArticlePackage(title="half the articles", source="test", article_ids=[articles[1].article_id])
    package_2 = ArticlePackage(title="half the articles", source="test", article_ids=[articles[3].article_id])

    filtered = filter_using_packages(candidates, [package_1, package_2])
    filtered_ids = [a.article_id for a in filtered.articles]

    for article_id in package_1.article_ids + package_2.article_ids:
        assert article_id in filtered_ids
