from uuid import uuid4

from lenskit.pipeline import PipelineBuilder

from poprox_concepts.domain import Article, ImpressedRecommendations, InterestProfile
from poprox_recommender.components.joiners.rrf import ReciprocalRankFusion

total_slots = 10


def test_reciprocal_rank_fusion():
    inputs = {
        "recs1": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "recs2": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    recs1_input = pipeline.create_input("recs1", ImpressedRecommendations)
    recs2_input = pipeline.create_input("recs2", ImpressedRecommendations)

    pipeline.add_component("rrf", rrf, recs1=recs1_input, recs2=recs2_input)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].impressions) == total_slots

    for i in range(10):
        if i % 2 == 0:
            assert outputs["recommender"].impressions[i].article.article_id in [
                impression.article.article_id for impression in inputs["recs1"].impressions
            ]
        else:
            assert outputs["recommender"].impressions[i].article.article_id in [
                impression.article.article_id for impression in inputs["recs2"].impressions
            ]


def test_reciprocal_rank_fusion_overlap():
    inputs = {
        "recs1": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "recs2": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    inputs["recs2"].impressions[1].article = inputs["recs1"].impressions[1].article

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", ImpressedRecommendations)
    in_cand2 = pipeline.create_input("recs2", ImpressedRecommendations)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].impressions) == total_slots

    assert (
        outputs["recommender"].impressions[0].article == inputs["recs1"].impressions[1].article
        and outputs["recommender"].impressions[0].article == inputs["recs2"].impressions[1].article
    )


def test_reciprocal_rank_fusion_mismatched_lengths():
    inputs = {
        "recs1": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(2))]
        ),
        "recs2": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", ImpressedRecommendations)
    in_cand2 = pipeline.create_input("recs2", ImpressedRecommendations)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].impressions) == total_slots


def test_reciprocal_rank_fusion_empty_list():
    inputs = {
        "recs1": ImpressedRecommendations.from_articles(articles=[]),
        "recs2": ImpressedRecommendations.from_articles(
            articles=[Article(article_id=uuid4(), headline="headline") for _ in range(int(total_slots * 2))]
        ),
        "profile": InterestProfile(click_history=[], onboarding_topics=[]),
    }

    rrf = ReciprocalRankFusion(num_slots=total_slots)

    pipeline = PipelineBuilder(name="rrf")
    in_cand1 = pipeline.create_input("recs1", ImpressedRecommendations)
    in_cand2 = pipeline.create_input("recs2", ImpressedRecommendations)

    pipeline.add_component("rrf", rrf, recs1=in_cand1, recs2=in_cand2)
    pipeline.alias("recommender", "rrf")
    pipeline = pipeline.build()

    outputs = pipeline.run_all(**inputs)

    assert len(outputs["recommender"].impressions) == total_slots
