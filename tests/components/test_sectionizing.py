from uuid import UUID, uuid4

from poprox_concepts.domain import (
    AccountInterest,
    Article,
    ArticlePackage,
    CandidateSet,
    Entity,
    InterestProfile,
    Mention,
)
from poprox_recommender.components.filters.duplicate import DuplicateFilter
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillConfig, FillRecs
from poprox_recommender.components.rankers.topk import TopkConfig, TopkRanker
from poprox_recommender.components.sections.base import select_from_candidates
from poprox_recommender.components.sections.other_news import InOtherNews, InOtherNewsConfig
from poprox_recommender.components.sections.top_news import (
    AddSection,
    AddSectionConfig,
)
from poprox_recommender.components.sections.topical import TopicalSections, TopicalSectionsConfig
from poprox_recommender.components.selectors.top_news import TopStoryCandidates


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


class LazyShim:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_sectionizer_creates_sections():
    general_news_id = uuid4()
    sports_id, technology_id, business_id = uuid4(), uuid4(), uuid4()
    sports_entity = Entity(entity_id=sports_id, name="Sports", entity_type="topic", source="test")
    tech_entity = Entity(entity_id=technology_id, name="Technology", entity_type="topic", source="test")
    business_entity = Entity(entity_id=business_id, name="Business", entity_type="topic", source="test")

    articles = [
        Article(
            article_id=uuid4(),
            headline=f"Article {i}",
            mentions=[
                Mention(source="test", entity=[sports_entity, tech_entity, business_entity][i % 3], relevance=99.0)
            ],
        )
        for i in range(1, 20)
    ]
    candidates = CandidateSet(articles=articles)
    packages = [
        make_package(general_news_id, "General News", articles[:5]),
        make_package(sports_id, "Top Sports Stories", articles[5:10]),
        make_package(technology_id, "Top Technology Stories", articles[10:15]),
    ]

    # user likes both topics
    profile = make_interest_profile([sports_id, technology_id])

    sections = []

    selector = TopStoryCandidates()
    top_articles = selector(candidates, packages)

    dup_filter = DuplicateFilter()
    deduped_top = dup_filter(top_articles, sections)

    topic_filter = TopicFilter()
    filtered_top = topic_filter(deduped_top, profile)

    filtered_config = TopkConfig(num_slots=2)
    filtered_topk = TopkRanker(filtered_config)
    filtered_articles = filtered_topk(filtered_top)

    # The maximum overlap with the articles chosen above is self.config.max_articles,
    # so here we pull twice as many to cover the worst case
    unfiltered_config = TopkConfig(num_slots=4)
    unfiltered_topk = TopkRanker(unfiltered_config)
    unfiltered_articles = LazyShim(unfiltered_topk(deduped_top))

    joiner_config = FillConfig(num_slots=2)
    joiner = FillRecs(joiner_config)
    top_section = joiner(filtered_articles, unfiltered_articles)

    top_news_config = AddSectionConfig(max_articles=2, title="Your Top Stories", personalized=True)
    sections = AddSection(top_news_config).__call__(top_section)

    topical_config = TopicalSectionsConfig(
        max_topic_sections=2,
        max_articles_per_topic=2,
    )
    sections = TopicalSections(topical_config).__call__(candidates, packages, profile, sections)

    other_news_config = InOtherNewsConfig(max_articles=2)
    sections = InOtherNews(other_news_config).__call__(
        candidates,
        packages,
        profile,
        sections,
    )

    # We should get a top news section, two topical sections, and a misc section
    assert len(sections) == 4
    titles = [s.title for s in sections]
    assert "Your Top Stories" in titles
    assert "Sports" in titles
    assert "Technology" in titles
    assert "In Other News" in titles


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

    sections = []

    selector = TopStoryCandidates()
    top_articles = selector(candidates, packages)

    dup_filter = DuplicateFilter()
    deduped_top = dup_filter(top_articles, sections)

    topic_filter = TopicFilter()
    filtered_top = topic_filter(deduped_top, profile)

    filtered_config = TopkConfig(num_slots=2)
    filtered_topk = TopkRanker(filtered_config)
    filtered_articles = filtered_topk(filtered_top)

    # The maximum overlap with the articles chosen above is self.config.max_articles,
    # so here we pull twice as many to cover the worst case
    unfiltered_config = TopkConfig(num_slots=4)
    unfiltered_topk = TopkRanker(unfiltered_config)
    unfiltered_articles = LazyShim(unfiltered_topk(deduped_top))

    joiner_config = FillConfig(num_slots=2)
    joiner = FillRecs(joiner_config)
    top_section = joiner(filtered_articles, unfiltered_articles)

    top_news_config = AddSectionConfig(max_articles=1, title="Your Top Stories", personalized=True)
    sections = AddSection(top_news_config).__call__(top_section, sections)

    topical_config = TopicalSectionsConfig(
        max_topic_sections=1,
    )
    sections = TopicalSections(topical_config).__call__(candidates, packages, profile, sections)

    other_news_config = InOtherNewsConfig(max_articles=2)
    sections = InOtherNews(other_news_config).__call__(
        candidates,
        packages,
        profile,
        sections,
    )

    assert len(sections) == 2
    assert sections[0].title == "Your Top Stories"
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
