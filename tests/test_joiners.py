from uuid import uuid4

from poprox_concepts.domain import Article, ArticleSet, Click, InterestProfile
from poprox_recommender.components.filters import TopicFilter
from poprox_recommender.components.joiners import Concatenate, Fill, Interleave
from poprox_recommender.components.samplers import UniformSampler
from poprox_recommender.pipeline import RecommendationPipeline


def build_pipeline(total_slots):
    topic_filter = TopicFilter()
    sampler = UniformSampler(num_slots=total_slots)

    pipeline = RecommendationPipeline(name="random_concat")
    pipeline.add(topic_filter, inputs=["candidate", "profile"], output="topical")
    pipeline.add(sampler, inputs=["topical", "candidate"], output="sampled")

    return pipeline


total_slots = 10
inputs = {
    "candidate": ArticleSet(articles=[Article(article_id=uuid4(), title="title") for _ in range(total_slots)]),
    "profile": InterestProfile(click_history=[], onboarding_topics=[]),
}


def test_concat_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    joiner = Concatenate()

    pipeline = build_pipeline(int(total_slots / 2))
    pipeline.add(sampler2, inputs=["topical", "candidate"], output="sampled2")
    pipeline.add(joiner, inputs=["sampled", "sampled2"], output="recs")

    outputs = pipeline(inputs)

    expected_articles = outputs["sampled"].articles + outputs["sampled2"].articles
    expected_article_ids = [article.article_id for article in expected_articles]
    assert set(article.article_id for article in outputs["recs"].articles) == set(expected_article_ids)


def test_interleave_two_recs_lists():
    sampler2 = UniformSampler(num_slots=int(total_slots / 2))
    joiner = Interleave()

    pipeline = build_pipeline(int(total_slots / 2))
    pipeline.add(sampler2, inputs=["topical", "candidate"], output="sampled2")
    pipeline.add(joiner, inputs=["sampled", "sampled2"], output="recs")

    outputs = pipeline(inputs)

    recs1_article_ids = [article.article_id for article in outputs["sampled"].articles]
    recs2_article_ids = [article.article_id for article in outputs["sampled2"].articles]

    assert all([article.article_id in recs1_article_ids for article in outputs["recs"].articles[::2]])
    assert all([article.article_id in recs2_article_ids for article in outputs["recs"].articles[1::2]])


def test_fill_out_one_recs_list_from_another():
    sampled_slots = 7

    joiner = Fill(num_slots=total_slots)

    pipeline = build_pipeline(sampled_slots)
    pipeline.add(joiner, inputs=["sampled", "candidate"], output="recs")

    outputs = pipeline(inputs)

    assert len(outputs["sampled"].articles) == sampled_slots
    assert len(outputs["recs"].articles) == total_slots

    assert outputs["recs"].articles[:sampled_slots] == outputs["sampled"].articles
    assert outputs["recs"].articles[sampled_slots:] == inputs["candidate"].articles[: total_slots - sampled_slots]
