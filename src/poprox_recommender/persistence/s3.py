import json
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

from poprox_concepts.domain import RecommendationList

from .base import PersistenceManager


class S3PersistenceManager(PersistenceManager):
    """
    S3-based persistence manager for AWS Lambda and production environments.

    Stores pipeline data in S3 with the following structure:
    - {prefix}{session_id}/user_model.txt
    - {prefix}{session_id}/original_recommendations.pkl
    - {prefix}{session_id}/rewritten_recommendations.pkl
    - {prefix}{session_id}/metadata.json
    """

    def __init__(self, bucket_name: str, prefix: str = "pipeline-outputs/"):
        """
        Initialize S3 persistence manager.

        Args:
            bucket_name: S3 bucket name for storing data
            prefix: Key prefix for organizing data in bucket

        Raises:
            ImportError: If boto3 is not available
        """
        if boto3 is None:
            raise ImportError("boto3 is required for S3PersistenceManager. Install with: pip install boto3")

        self.s3 = boto3.client("s3")
        self.bucket = bucket_name
        self.prefix = prefix

    def save_pipeline_data(
        self,
        request_id: str,
        user_model: str,
        original_recommendations: RecommendationList,
        rewritten_recommendations: RecommendationList,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save pipeline data to S3."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        session_id = f"{request_id}_{timestamp}"

        try:
            metadata_payload = dict(metadata or {})
            component_metrics = metadata_payload.pop("component_metrics", {})
            issues = metadata_payload.pop("issues", [])

            # Save user model
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{session_id}/user_model.txt",
                Body=user_model.encode("utf-8"),
                ContentType="text/plain",
            )

            # Save original recommendations
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{session_id}/original_recommendations.pkl",
                Body=pickle.dumps(original_recommendations),
                ContentType="application/octet-stream",
            )

            # Save rewritten recommendations
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{session_id}/rewritten_recommendations.pkl",
                Body=pickle.dumps(rewritten_recommendations),
                ContentType="application/octet-stream",
            )

            # Save metadata
            full_metadata = {
                "request_id": request_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_model_length": len(user_model),
                "num_articles": len(original_recommendations.articles),
                "pipeline_type": "llm_rank_rewrite",
                "storage_location": f"s3://{self.bucket}/{self.prefix}{session_id}/",
                **metadata_payload,
            }

            if component_metrics:
                full_metadata["component_metrics"] = component_metrics
                full_metadata["component_summary"] = {
                    name: {
                        "status": data.get("status"),
                        "duration_seconds": data.get("duration_seconds"),
                        "error_count": data.get("error_count", 0),
                    }
                    for name, data in component_metrics.items()
                }
            else:
                full_metadata["component_metrics"] = {}

            full_metadata["issues"] = issues
            full_metadata["issue_count"] = len(issues)

            # Add LLM metrics summary if available
            if "llm_metrics" in metadata_payload:
                llm_metrics = metadata_payload["llm_metrics"]
                
                # Calculate total tokens and time across all LLM calls
                total_input_tokens = 0
                total_output_tokens = 0
                total_duration = 0
                
                # Add ranker metrics
                if "ranker" in llm_metrics:
                    for metrics in llm_metrics["ranker"].values():
                        total_input_tokens += metrics.get("input_tokens", 0)
                        total_output_tokens += metrics.get("output_tokens", 0)
                        total_duration += metrics.get("duration_seconds", 0)
                
                # Add rewriter metrics
                if "rewriter" in llm_metrics:
                    for article_metrics in llm_metrics["rewriter"]:
                        total_input_tokens += article_metrics.get("input_tokens", 0)
                        total_output_tokens += article_metrics.get("output_tokens", 0)
                        total_duration += article_metrics.get("duration_seconds", 0)
                
                full_metadata["llm_summary"] = {
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                    "total_duration_seconds": total_duration,
                    "num_llm_calls": (
                        len(llm_metrics.get("ranker", {})) + 
                        len(llm_metrics.get("rewriter", []))
                    )
                }

            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"{self.prefix}{session_id}/metadata.json",
                Body=json.dumps(full_metadata, indent=2).encode("utf-8"),
                ContentType="application/json",
            )

            return session_id

        except ClientError as e:
            raise RuntimeError(f"Failed to save pipeline data to S3: {e}")

    def load_pipeline_data(self, session_id: str) -> Dict[str, Any]:
        """Load pipeline data from S3."""
        try:
            # Load user model
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{session_id}/user_model.txt")
            user_model = response["Body"].read().decode("utf-8")

            # Load original recommendations
            response = self.s3.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}{session_id}/original_recommendations.pkl"
            )
            original_recommendations = pickle.loads(response["Body"].read())

            # Load rewritten recommendations
            response = self.s3.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}{session_id}/rewritten_recommendations.pkl"
            )
            rewritten_recommendations = pickle.loads(response["Body"].read())

            # Load metadata
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{session_id}/metadata.json")
            metadata = json.loads(response["Body"].read().decode("utf-8"))

            return {
                "user_model": user_model,
                "original_recommendations": original_recommendations,
                "rewritten_recommendations": rewritten_recommendations,
                "metadata": metadata,
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Session {session_id} not found in S3")
            else:
                raise RuntimeError(f"Failed to load pipeline data from S3: {e}")

    def load_metadata(self, session_id: str) -> Dict[str, Any]:
        """Load only the metadata for a session from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{session_id}/metadata.json")
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Session {session_id} not found in S3")
            else:
                raise RuntimeError(f"Failed to load metadata from S3: {e}")

    def list_sessions(self, request_id_prefix: Optional[str] = None) -> list[str]:
        """List available sessions in S3."""
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            sessions = []

            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"):
                for prefix_info in page.get("CommonPrefixes", []):
                    session_path = prefix_info["Prefix"]
                    # Extract session_id from path: "pipeline-outputs/session_id/"
                    session_id = session_path[len(self.prefix) :].rstrip("/")

                    if request_id_prefix is None or session_id.startswith(request_id_prefix):
                        sessions.append(session_id)

            return sorted(sessions, reverse=True)  # Most recent first

        except ClientError as e:
            raise RuntimeError(f"Failed to list sessions from S3: {e}")
