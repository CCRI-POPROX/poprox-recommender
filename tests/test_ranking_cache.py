"""
Tests for LLM ranking cache functionality.
"""

import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from poprox_concepts import Article, CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.rankers.ranking_cache import RankingCacheManager, get_ranking_cache_manager


@pytest.fixture
def sample_profile():
    """Create a sample interest profile."""
    return InterestProfile(
        profile_id=uuid4(),
        click_history=[],
        onboarding_topics=[],
    )


@pytest.fixture
def sample_candidates():
    """Create a sample candidate set."""
    articles = [
        Article(article_id=uuid4(), headline=f"Article {i}", subhead=f"Subhead {i}") for i in range(5)
    ]
    return CandidateSet(articles=articles)


@pytest.fixture
def sample_ranking_output(sample_candidates):
    """Create a sample ranking output tuple."""
    recommendations = RecommendationList(articles=sample_candidates.articles[:3])
    user_model = "User is interested in tech and science"
    request_id = "test_user_123"
    llm_metrics = {"article_ranking": {"input_tokens": 100, "output_tokens": 50}}
    component_metrics = {"ranker": {"status": "success", "duration_seconds": 1.5}}
    return (recommendations, user_model, request_id, llm_metrics, component_metrics)


class TestRankingCacheManager:
    """Tests for RankingCacheManager."""

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_cache_key_generation_same_profile_and_candidates(self, mock_boto3, sample_profile, sample_candidates):
        """Test that same profile and candidates generate same cache key."""
        mock_boto3.client.return_value = MagicMock()
        cache_manager = RankingCacheManager(bucket_name="test-bucket")

        key1 = cache_manager._generate_cache_key(str(sample_profile.profile_id), sample_candidates)
        key2 = cache_manager._generate_cache_key(str(sample_profile.profile_id), sample_candidates)

        assert key1 == key2

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_cache_key_order_independence(self, mock_boto3, sample_profile):
        """Test that candidate order doesn't affect cache key."""
        mock_boto3.client.return_value = MagicMock()
        cache_manager = RankingCacheManager(bucket_name="test-bucket")

        # Use fixed UUIDs for this test
        uuid1 = uuid4()
        uuid2 = uuid4()
        uuid3 = uuid4()

        # Create two candidate sets with same articles in different order
        articles1 = [
            Article(article_id=uuid1, headline="Article 1"),
            Article(article_id=uuid2, headline="Article 2"),
            Article(article_id=uuid3, headline="Article 3"),
        ]
        articles2 = [
            Article(article_id=uuid3, headline="Article 3"),
            Article(article_id=uuid1, headline="Article 1"),
            Article(article_id=uuid2, headline="Article 2"),
        ]
        candidates1 = CandidateSet(articles=articles1)
        candidates2 = CandidateSet(articles=articles2)

        key1 = cache_manager._generate_cache_key(str(sample_profile.profile_id), candidates1)
        key2 = cache_manager._generate_cache_key(str(sample_profile.profile_id), candidates2)

        assert key1 == key2

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_cache_key_different_profiles(self, mock_boto3, sample_candidates):
        """Test that different profiles generate different cache keys."""
        mock_boto3.client.return_value = MagicMock()
        cache_manager = RankingCacheManager(bucket_name="test-bucket")

        key1 = cache_manager._generate_cache_key("user_1", sample_candidates)
        key2 = cache_manager._generate_cache_key("user_2", sample_candidates)

        assert key1 != key2

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_cache_key_different_candidates(self, mock_boto3, sample_profile):
        """Test that different candidate sets generate different cache keys."""
        mock_boto3.client.return_value = MagicMock()
        cache_manager = RankingCacheManager(bucket_name="test-bucket")

        candidates1 = CandidateSet(articles=[Article(article_id=uuid4(), headline="Article 1")])
        candidates2 = CandidateSet(articles=[Article(article_id=uuid4(), headline="Article 2")])

        key1 = cache_manager._generate_cache_key(str(sample_profile.profile_id), candidates1)
        key2 = cache_manager._generate_cache_key(str(sample_profile.profile_id), candidates2)

        assert key1 != key2

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_get_ranking_cache_manager_disabled(self, mock_boto3):
        """Test that cache manager is None when explicitly disabled."""
        with patch.dict(os.environ, {"RANKING_CACHE_ENABLED": "false"}):
            cache_manager = get_ranking_cache_manager()
            assert cache_manager is None

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_get_ranking_cache_manager_enabled_in_lambda(self, mock_boto3):
        """Test that cache manager is created in Lambda environment."""
        mock_boto3.client.return_value = MagicMock()
        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "test-function"}):
            cache_manager = get_ranking_cache_manager()
            assert cache_manager is not None
            assert isinstance(cache_manager, RankingCacheManager)

    @patch("poprox_recommender.components.rankers.ranking_cache.boto3")
    def test_s3_key_includes_date(self, mock_boto3):
        """Test that S3 key includes date for organization."""
        mock_boto3.client.return_value = MagicMock()
        cache_manager = RankingCacheManager(bucket_name="test-bucket", prefix="cache/")

        cache_key = "test_key"
        s3_key = cache_manager._get_s3_key(cache_key)

        assert "cache/" in s3_key
        assert "ranking_test_key.pkl" in s3_key
        # Should contain date in YYYY-MM-DD format
        assert len(s3_key.split("/")) >= 3  # prefix/date/filename
