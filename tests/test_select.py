from uuid import uuid4

from poprox_concepts import Article, ArticleSet, ClickHistory, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.filters import TopicFilter
from poprox_recommender.pipeline import RecommendationPipeline
from poprox_recommender.samplers import UniformSampler


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

    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=2)

    pipeline = RecommendationPipeline(name="random_topical")
    pipeline.add(topic_filter, inputs=["candidate", "profile"], output="topical")
    pipeline.add(sampler, inputs=["topical", "candidate"], output="recs")

    # If we can, only select articles matching interests
    inputs = {
        "candidate": ArticleSet(articles=articles),
        "clicked": ArticleSet(articles=[]),
        "profile": profile,
    }
    outputs = pipeline(inputs)

    for article in outputs.recs:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics

    # If we need to, fill out the end of the list with other random articles
    sampler.num_slots = 3
    outputs = pipeline(inputs)

    assert len(outputs.recs) == 3

    for article in outputs.recs[:2]:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics
