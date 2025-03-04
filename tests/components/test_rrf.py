from uuid import uuid4

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import Article, CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.components.joiners.rrf import ReciprocalRankFusion

total_slots = 10


def test_reciprocal_rank_fusion():
    inputs = {
        "recs1": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "recs2": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    recs1_input = pipeline.create_input("recs1", RecommendationList)
    recs2_input = pipeline.create_input("recs2", RecommendationList)

    pipeline.add_component("rrf", rrf, recs1=recs1_input, recs2=recs2_input)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots

    for i in range(10):
        if i % 2 == 0:
            assert outputs["recommender"].articles[i].article_id in [
                article.article_id for article in inputs["recs1"].articles
            ]
        else:
            assert outputs["recommender"].articles[i].article_id in [
                article.article_id for article in inputs["recs2"].articles
            ]


def test_reciprocal_rank_fusion_overlap():
    inputs = {
        "recs1": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "recs2": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    inputs["recs2"].articles[1] = inputs["recs1"].articles[1]

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", RecommendationList)
    in_cand2 = pipeline.create_input("recs2", RecommendationList)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots

    assert (
        outputs["recommender"].articles[0] == inputs["recs1"].articles[1]
        and outputs["recommender"].articles[0] == inputs["recs2"].articles[1]
    )


def test_reciprocal_rank_fusion_mismatched_lengths():
    inputs = {
        "recs1": RecommendationList(articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(2))]),
        "recs2": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", RecommendationList)
    in_cand2 = pipeline.create_input("recs2", RecommendationList)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots


def test_reciprocal_rank_fusion_empty_list():
    inputs = {
        "recs1": RecommendationList(articles=[]),
        "recs2": RecommendationList(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", RecommendationList)
    in_cand2 = pipeline.create_input("recs2", RecommendationList)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots
