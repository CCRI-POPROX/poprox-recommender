from uuid import uuid4

from lenskit.pipeline import Pipeline, PipelineBuilder

from poprox_concepts.domain import Article, CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Concatenate, FillCandidates, FillRecs, Interleave
from poprox_recommender.components.samplers import UniformSampler


def init_builder(total_slots):
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=total_slots)

    builder = PipelineBuilder(name="random_concat")
    in_cand = builder.create_input("candidate", CandidateSet)
    in_prof = builder.create_input("profile", InterestProfile)
    tf = builder.add_component("topic-filter", topic_filter, candidates=in_cand, interest_profile=in_prof)
    builder.add_component("sampler", sampler, candidates1=tf, candidates2=in_cand)

    return builder


total_slots = 10
inputs = {
    "candidate": CandidateSet(
        articles=[Article(article_id=uuid4(), headline="headline") for _ in range(2 * total_slots)]
    ),
    "profile": InterestProfile(click_history=[], onboarding_topics=[]),
}


def test_concat_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    concat = Concatenate()

    builder = init_builder(int(total_slots / 2))
    s2 = builder.add_component(
        "sampler2", sampler2, candidates1=builder.node("topic-filter"), candidates2=builder.node("candidate")
    )
    builder.add_component("joiner", concat, recs1=builder.node("sampler"), recs2=s2)
    builder.alias("recommender", "joiner")
    pipeline = builder.build()

    outputs = pipeline.run_all(**inputs)

    expected_articles = outputs["sampler"].articles + outputs["sampler2"].articles
    expected_article_ids = [article.article_id for article in expected_articles]
    assert set(article.article_id for article in outputs["recommender"].articles) == set(expected_article_ids)


def test_interleave_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    joiner = Interleave()

    builder = init_builder(int(total_slots / 2))
    s2 = builder.add_component(
        "sampler2", sampler2, candidates1=builder.node("topic-filter"), candidates2=builder.node("candidate")
    )
    builder.add_component("joiner", joiner, recs1=builder.node("sampler"), recs2=s2)
    builder.alias("recommender", "joiner")
    pipeline = builder.build()

    outputs = pipeline.run_all(**inputs)

    recs1_article_ids = [article.article_id for article in outputs["sampler"].articles]
    recs2_article_ids = [article.article_id for article in outputs["sampler2"].articles]

    assert all([article.article_id in recs1_article_ids for article in outputs["recommender"].articles[::2]])
    assert all([article.article_id in recs2_article_ids for article in outputs["recommender"].articles[1::2]])


def test_fill_out_one_recs_list_from_another():
    sampled_slots = 7

    backup = RecommendationList(articles=inputs["candidate"].articles[: total_slots - sampled_slots])

    joiner = FillRecs(num_slots=total_slots, deduplicate=False)

    builder = init_builder(sampled_slots)
    builder.create_input("backup", RecommendationList)
    builder.add_component("joiner", joiner, recs1=builder.node("sampler"), recs2=backup)
    builder.alias("recommender", "joiner")
    pipeline = builder.build()

    outputs = pipeline.run_all(**{**inputs, **{"backup": backup}})

    assert len(outputs["sampler"].articles) == sampled_slots
    assert len(outputs["recommender"].articles) == total_slots

    assert outputs["recommender"].articles[:sampled_slots] == outputs["sampler"].articles
    assert outputs["recommender"].articles[sampled_slots:] == backup.articles


def test_fill_removes_duplicates():
    inputs = {
        "recs": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots / 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    fill = FillRecs(num_slots=total_slots)

    builder = PipelineBuilder(name="duplicate_concat")
    recs_input = builder.create_input("recs", RecommendationList)

    builder.add_component("fill", fill, recs1=recs_input, recs2=recs_input)
    builder.alias("recommender", "fill")
    pipeline = builder.build()

    outputs = pipeline.run_all(**inputs)

    article_ids = [article.article_id for article in outputs["recommender"].articles]

    assert len(outputs["recommender"].articles) == 5
    assert len(article_ids) == len(set(article_ids))
