import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

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
                **(metadata or {}),
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
            response = self.s3.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}{session_id}/user_model.txt"
            )
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
            response = self.s3.get_object(
                Bucket=self.bucket, Key=f"{self.prefix}{session_id}/metadata.json"
            )
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