import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList


class RankingCacheManager:
    """
    S3-based cache for LLM ranking outputs to ensure consistency across pipelines.

    Cache key is based on:
    - User profile ID
    - Sorted candidate article IDs (order-independent hash)
    - Model version

    Cache structure in S3:
    - {prefix}{date}/ranking_{cache_key}.pkl
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "ranking-cache/",
        model_version: str = "gpt-4.1-mini-2025-04-14",
    ):
        """
        Initialize ranking cache manager.

        Args:
            bucket_name: S3 bucket name. If None, uses environment variable or default.
            prefix: Key prefix for organizing cache in bucket
            model_version: Model version to include in cache key
        """
        if boto3 is None:
            raise ImportError("boto3 is required for RankingCacheManager. Install with: pip install boto3")

        self.s3 = boto3.client("s3")
        self.bucket = bucket_name or os.getenv(
            "RANKING_CACHE_BUCKET",
            os.getenv("PERSISTENCE_BUCKET", "poprox-default-recommender-pipeline-data-prod"),
        )
        self.prefix = prefix
        self.model_version = model_version

    def _generate_cache_key(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        model_version: Optional[str] = None,
    ) -> str:
        """
        Generate a deterministic cache key from profile ID and candidate set.

        Args:
            profile_id: User profile ID
            candidate_articles: Candidate article set
            model_version: Optional override for model version

        Returns:
            Cache key string
        """
        # Sort article IDs to ensure order-independent hashing
        article_ids = sorted([str(art.article_id) for art in candidate_articles.articles])

        # Create hash of sorted article IDs
        articles_str = ",".join(article_ids)
        articles_hash = hashlib.sha256(articles_str.encode()).hexdigest()[:16]

        # Combine profile ID, articles hash, and model version
        version = model_version or self.model_version
        cache_key = f"{profile_id}_{articles_hash}_{version}"

        return cache_key

    def _get_s3_key(self, cache_key: str) -> str:
        """
        Generate S3 key for cache entry, organized by date.

        Args:
            cache_key: Cache key from _generate_cache_key

        Returns:
            Full S3 key path
        """
        # Organize by date for easy cleanup
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{self.prefix}{date_str}/ranking_{cache_key}.pkl"

    def get_cached_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        model_version: Optional[str] = None,
    ) -> Optional[tuple[RecommendationList, str, str, dict, dict]]:
        """
        Retrieve cached ranking output if available.

        Args:
            profile_id: User profile ID
            candidate_articles: Candidate article set
            model_version: Optional override for model version

        Returns:
            Cached ranking output tuple or None if not found
        """
        cache_key = self._generate_cache_key(profile_id, candidate_articles, model_version)
        s3_key = self._get_s3_key(cache_key)

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            cached_data = pickle.loads(response["Body"].read())
            return cached_data
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                # Cache miss - not an error
                return None
            else:
                # Log error but don't fail - just return None
                print(f"Warning: Failed to retrieve cache from S3: {e}")
                return None

    def save_ranking(
        self,
        profile_id: str,
        candidate_articles: CandidateSet,
        ranking_output: tuple[RecommendationList, str, str, dict, dict],
        model_version: Optional[str] = None,
    ) -> str:
        """
        Save ranking output to cache.

        Args:
            profile_id: User profile ID
            candidate_articles: Candidate article set
            ranking_output: Full ranking output tuple to cache
            model_version: Optional override for model version

        Returns:
            Cache key used for storage
        """
        cache_key = self._generate_cache_key(profile_id, candidate_articles, model_version)
        s3_key = self._get_s3_key(cache_key)

        try:
            # Save the entire ranking output tuple
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=pickle.dumps(ranking_output),
                ContentType="application/octet-stream",
                Metadata={
                    "profile_id": profile_id,
                    "cache_key": cache_key,
                    "model_version": model_version or self.model_version,
                    "cached_at": datetime.now().isoformat(),
                },
            )
            return cache_key
        except ClientError as e:
            # Log error but don't fail the ranking operation
            print(f"Warning: Failed to save cache to S3: {e}")
            return cache_key


def get_ranking_cache_manager(model_version: str = "gpt-4.1-mini-2025-04-14") -> Optional[RankingCacheManager]:
    """
    Factory function to get ranking cache manager based on environment.

    Returns:
        RankingCacheManager if caching is enabled, None otherwise
    """
    # Check if caching is explicitly disabled
    if os.getenv("RANKING_CACHE_ENABLED", "true").lower() == "false":
        return None

    # Only enable caching in Lambda or if explicitly enabled
    if not (os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("RANKING_CACHE_ENABLED") == "true"):
        return None

    try:
        bucket = os.getenv(
            "RANKING_CACHE_BUCKET",
            os.getenv("PERSISTENCE_BUCKET", "poprox-default-recommender-pipeline-data-prod"),
        )
        prefix = os.getenv("RANKING_CACHE_PREFIX", "ranking-cache/")
        return RankingCacheManager(bucket, prefix, model_version)
    except ImportError:
        # boto3 not available - caching disabled
        return None
