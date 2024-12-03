from uuid import uuid4

from poprox_concepts.domain import Article, ArticleSet, InterestProfile
from poprox_recommender.components.joiners.rrf import ReciprocalRankFusion
from poprox_recommender.lkpipeline import Pipeline

total_slots = 10


def test_reciprocal_rank_fusion():
    inputs = {
        "candidate1": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "candidate2": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = Pipeline(name="rrf")
    in_cand1 = pipeline.create_input("candidate1", ArticleSet)
    in_cand2 = pipeline.create_input("candidate2", ArticleSet)

    pipeline.add_component("rrf", rrf, candidates1=in_cand1, candidates2=in_cand2)
    pipeline.alias("recommender", "rrf")

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots

    for i in range(10):
        if i % 2 == 0:
            assert outputs["recommender"].articles[i].article_id in [
                article.article_id for article in inputs["candidate1"].articles
            ]
        else:
            assert outputs["recommender"].articles[i].article_id in [
                article.article_id for article in inputs["candidate2"].articles
            ]


def test_reciprocal_rank_fusion_overlap():
    inputs = {
        "candidate1": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "candidate2": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    inputs["candidate2"].articles[1] = inputs["candidate1"].articles[1]

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = Pipeline(name="rrf")
    in_cand1 = pipeline.create_input("candidate1", ArticleSet)
    in_cand2 = pipeline.create_input("candidate2", ArticleSet)

    pipeline.add_component("rrf", rrf, candidates1=in_cand1, candidates2=in_cand2)
    pipeline.alias("recommender", "rrf")

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots

    assert (
        outputs["recommender"].articles[0] == inputs["candidate1"].articles[1]
        and outputs["recommender"].articles[0] == inputs["candidate2"].articles[1]
    )


def test_reciprocal_rank_fusion_mismatched_lengths():
    inputs = {
        "candidate1": ArticleSet(articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(2))]),
        "candidate2": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = Pipeline(name="rrf")
    in_cand1 = pipeline.create_input("candidate1", ArticleSet)
    in_cand2 = pipeline.create_input("candidate2", ArticleSet)

    pipeline.add_component("rrf", rrf, candidates1=in_cand1, candidates2=in_cand2)
    pipeline.alias("recommender", "rrf")

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots


def test_reciprocal_rank_fusion_empty_list():
    inputs = {
        "candidate1": ArticleSet(articles=[]),
        "candidate2": ArticleSet(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = Pipeline(name="rrf")
    in_cand1 = pipeline.create_input("candidate1", ArticleSet)
    in_cand2 = pipeline.create_input("candidate2", ArticleSet)

    pipeline.add_component("rrf", rrf, candidates1=in_cand1, candidates2=in_cand2)
    pipeline.alias("recommender", "rrf")

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].articles) == total_slots
