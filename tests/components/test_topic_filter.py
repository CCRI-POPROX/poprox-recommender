from uuid import uuid4

from poprox_concepts import Article, ArticleSet, Click, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.lkpipeline import Pipeline


def test_select_by_topic_filters_articles():
    profile = InterestProfile(
        click_history=[],
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

    pipeline = Pipeline()
    i_profile = pipeline.create_input("profile", InterestProfile)
    i_cand = pipeline.create_input("candidates", ArticleSet)
    c_filter = pipeline.add_component(topic_filter, candidate=i_cand, interest_profile=i_profile)
    c_sampler = pipeline.add_component(sampler, candidate=c_filter, backup=i_cand)

    # If we can, only select articles matching interests
    outputs = pipeline.run(c_sampler, candidates=ArticleSet(articles=articles), profile=profile)

    for article in outputs.recs:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics

    # If we need to, fill out the end of the list with other random articles
    sampler.num_slots = 3
    outputs = pipeline.run(c_sampler, candidates=ArticleSet(articles=articles), profile=profile)

    assert len(outputs.recs) == 3

    for article in outputs.recs[:2]:
        topics = [mention.entity.name for mention in article.mentions]
        assert "U.S. News" in topics or "Politics" in topics
