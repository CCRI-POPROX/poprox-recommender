from uuid import uuid4

from lenskit.pipeline import Pipeline, PipelineBuilder

from poprox_concepts.domain import Article, CandidateSet, ImpressedSection, InterestProfile
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Concatenate, FillRecs, Interleave
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
    "profile": InterestProfile(click_history=[], entity_interests=[]),
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

    expected_impressions = outputs["sampler"].impressions + outputs["sampler2"].impressions
    expected_article_ids = [impression.article.article_id for impression in expected_impressions]
    assert set(impression.article.article_id for impression in outputs["recommender"].impressions) == set(
        expected_article_ids
    )


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

    recs1_article_ids = [impression.article.article_id for impression in outputs["sampler"].impressions]
    recs2_article_ids = [impression.article.article_id for impression in outputs["sampler2"].impressions]

    assert all(
        [impression.article.article_id in recs1_article_ids for impression in outputs["recommender"].impressions[::2]]
    )
    assert all(
        [impression.article.article_id in recs2_article_ids for impression in outputs["recommender"].impressions[1::2]]
    )


def test_fill_out_one_recs_list_from_another():
    sampled_slots = 7

    backup = ImpressedSection.from_articles(articles=inputs["candidate"].articles[: total_slots - sampled_slots])

    joiner = FillRecs(num_slots=total_slots, deduplicate=False)

    builder = init_builder(sampled_slots)
    builder.create_input("backup", ImpressedSection)
    builder.add_component("joiner", joiner, recs1=builder.node("sampler"), recs2=backup)
    builder.alias("recommender", "joiner")
    pipeline = builder.build()

    outputs = pipeline.run_all(**{**inputs, **{"backup": backup}})

    assert len(outputs["sampler"].impressions) == sampled_slots
    assert len(outputs["recommender"].impressions) == total_slots

    assert outputs["recommender"].impressions[:sampled_slots] == outputs["sampler"].impressions
    assert outputs["recommender"].impressions[sampled_slots:] == backup.impressions


def test_fill_removes_duplicates():
    inputs = {
        "recs": ImpressedSection.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots / 2))]
        ),
        "profile": InterestProfile(click_history=[], entity_interests=[]),
    }

    fill = FillRecs(num_slots=total_slots)

    builder = PipelineBuilder(name="duplicate_concat")
    recs_input = builder.create_input("recs", ImpressedSection)

    builder.add_component("fill", fill, recs1=recs_input, recs2=recs_input)
    builder.alias("recommender", "fill")
    pipeline = builder.build()

    outputs = pipeline.run_all(**inputs)

    article_ids = [impression.article.article_id for impression in outputs["recommender"].impressions]

    assert len(outputs["recommender"].impressions) == 5
    assert len(article_ids) == len(set(article_ids))
