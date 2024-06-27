from uuid import uuid4

from poprox_concepts import Article, ClickHistory, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.default import select_by_topic


def test_select_by_topic_filters_articles():
    profile = InterestProfile(
        click_history=ClickHistory(article_ids=[]),
        onboarding_topics=[
            AccountInterest(entity_id=uuid4(), entity_name="U.S. News", preference=2, frequency=1),
            AccountInterest(entity_id=uuid4(), entity_name="Politics", preference=3, frequency=2),
        ],
    )

    us_news = Entity(name="U.S. News", entity_type="topic", source="AP", raw_data={})
    politics = Entity(name="Politics", entity_type="topic", source="AP", raw_data={})
    entertainment = Entity(name="Entertainment", entity_type="topic", source="AP", raw_data={})

    articles = [
        Article(
            title="Something about TV",
            mentions=[Mention(source="AP", relevance=50.0, entity=entertainment)],
        ),
        Article(
            title="Something about the US",
            mentions=[Mention(source="AP", relevance=50.0, entity=us_news)],
        ),
        Article(
            title="Something about politics",
            mentions=[Mention(source="AP", relevance=50.0, entity=politics)],
        ),
        Article(
            title="Something about books",
            mentions=[Mention(source="AP", relevance=50.0, entity=entertainment)],
        ),
    ]

    # If we can, only select articles matching interests
    recs = select_by_topic(articles, profile, num_slots=2)

    for article in recs:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics

    # If we need to, fill out the end of the list with other random articles
    recs = select_by_topic(articles, profile, num_slots=3)
    assert len(recs) == 3

    for article in recs[:2]:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics
