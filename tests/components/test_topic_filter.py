from uuid import uuid4

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import Article, CandidateSet, Click, Entity, Mention
from poprox_concepts.domain.profile import AccountInterest, InterestProfile
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.samplers import UniformSampler


def test_select_by_topic_filters_articles():
    profile = InterestProfile(
        click_history=[],
        onboarding_topics=[
            AccountInterest(entity_id=uuid4(), entity_name="U.S. News", preference=4, frequency=1),
            AccountInterest(entity_id=uuid4(), entity_name="Politics", preference=5, frequency=2),
            AccountInterest(entity_id=uuid4(), entity_name="Entertainment", preference=1, frequency=1),
        ],
    )

    us_news = Entity(name="U.S. News", entity_type="topic", source="AP", raw_data={})
    politics = Entity(name="Politics", entity_type="topic", source="AP", raw_data={})
    entertainment = Entity(name="Entertainment", entity_type="topic", source="AP", raw_data={})

    articles = [
        Article(
            article_id=uuid4(),
            headline="Something about TV",
            mentions=[Mention(source="AP", relevance=50.0, entity=entertainment)],
        ),
        Article(
            article_id=uuid4(),
            headline="Something about the US",
            mentions=[Mention(source="AP", relevance=50.0, entity=us_news)],
        ),
        Article(
            article_id=uuid4(),
            headline="Something about politics",
            mentions=[Mention(source="AP", relevance=50.0, entity=politics)],
        ),
        Article(
            article_id=uuid4(),
            headline="Something about books",
            mentions=[Mention(source="AP", relevance=50.0, entity=entertainment)],
        ),
    ]

    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=2)

    builder = PipelineBuilder()
    i_profile = builder.create_input("profile", InterestProfile)
    i_cand = builder.create_input("candidates", CandidateSet)
    c_filter = builder.add_component("topic-filter", topic_filter, candidates=i_cand, interest_profile=i_profile)
    c_sampler = builder.add_component("sampler", sampler, candidates1=c_filter, candidates2=i_cand)
    pipeline = builder.build()

    # If we can, only select articles matching interests
    result = pipeline.run(c_sampler, candidates=CandidateSet(articles=articles), profile=profile)

    # there are 2 valid articles that match their preferences (us news & politics)
    assert len(result.impressions) == 2
    for impression in result.impressions:
        topics = [mention.entity.name for mention in impression.article.mentions]
        assert "U.S. News" in topics or "Politics" in topics

    # If we need to, fill out the end of the list with other random articles
    sampler.config.num_slots = 3
    result = pipeline.run(c_sampler, candidates=CandidateSet(articles=articles), profile=profile)

    assert len(result.impressions) == 3

    for impression in result.impressions[:2]:
        topics = [mention.entity.name for mention in impression.article.mentions]
        assert "U.S. News" in topics or "Politics" in topics
