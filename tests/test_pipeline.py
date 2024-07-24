import pytest

from poprox_concepts.domain import ArticleSet, InterestProfile
from poprox_recommender.pipeline import RecommendationPipeline


class UnannotatedArticleComponent:
    def __call__(self, articles, profile):
        return articles


class ArticleComponent:
    def __call__(self, articles: ArticleSet, profile: InterestProfile) -> ArticleSet:
        return articles


# The inconsistent parameter naming between article and profile components
# is intentional in order to test that our type checking doesn't require
# consistent naming
class UnannotatedProfileComponent:
    def __call__(self, article_set, interest_profile):
        return interest_profile


class ProfileComponent:
    def __call__(self, article_set: ArticleSet, interest_profile: InterestProfile) -> InterestProfile:
        return interest_profile


def test_construction_without_annotations():
    pipeline = RecommendationPipeline(name="matching")
    pipeline.add(UnannotatedProfileComponent(), output="profile", inputs=["candidate", "profile"])
    pipeline.add(UnannotatedArticleComponent(), output="recs", inputs=["candidate", "profile"])

    assert len(pipeline.components) == 2


def test_construction_with_matching_inputs():
    pipeline = RecommendationPipeline(name="matching")
    pipeline.add(ProfileComponent(), output="profile", inputs=["candidate", "profile"])
    pipeline.add(ArticleComponent(), output="recs", inputs=["candidate", "profile"])

    assert len(pipeline.components) == 2


def test_construction_with_mismatched_inputs():
    pipeline = RecommendationPipeline(name="matching")

    with pytest.raises(TypeError):
        pipeline.add(ProfileComponent(), output="profile", inputs=["profile", "candidate"])


def test_construction_with_new_state_values():
    pipeline = RecommendationPipeline(name="new_state")
    pipeline.add(ProfileComponent(), output="augmented_profile", inputs=["candidate", "profile"])
    pipeline.add(ArticleComponent(), output="recs", inputs=["candidate", "augmented_profile"])

    assert len(pipeline.components) == 2


def test_construction_with_mismatched_new_state_values():
    pipeline = RecommendationPipeline(name="mismatched_new_state")
    pipeline.add(ProfileComponent(), output="augmented_profile", inputs=["candidate", "profile"])

    with pytest.raises(TypeError):
        pipeline.add(ArticleComponent(), output="recs", inputs=["augmented_profile", "candidate"])


def test_construction_with_mismatched_state_overwrite():
    pipeline = RecommendationPipeline(name="mismatched_output_type")

    with pytest.raises(TypeError):
        pipeline.add(ProfileComponent(), output="candidate", inputs=["candidate", "profile"])
