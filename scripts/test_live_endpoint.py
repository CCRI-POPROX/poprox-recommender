#!/usr/bin/env python3
"""
Testing script that:
1. Grabs all profile requests from tests/request_data/profiles/
2. For each, hits the live endpoint for recommendations
3. Grabs the corresponding output objects from the run in the S3 bucket
4. Persists those objects locally in a dir mapped to the profile name

Usage:
    python scripts/test_live_endpoint.py [--endpoint URL] [--output-dir DIR] [--bucket BUCKET] [--dry-run]

Args:
    --endpoint: Live endpoint URL (default: auto-detect from serverless config)
    --output-dir: Local directory to save outputs (default: evaluation_data/)
    --bucket: S3 bucket name (default: from PERSISTENCE_BUCKET env var)
    --dry-run: Don't actually make requests or download files
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import boto3
import requests
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LiveEndpointTester:
    def __init__(self, endpoint_url: str, output_dir: str, bucket_name: str, dry_run: bool = False):
        self.endpoint_url = endpoint_url
        self.output_dir = Path(output_dir)
        self.bucket_name = bucket_name
        self.dry_run = dry_run

        # Setup S3 client
        self.s3 = boto3.client("s3")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get profile requests directory
        self.profiles_dir = Path("tests/request_data/profiles")
        if not self.profiles_dir.exists():
            raise FileNotFoundError(f"Profiles directory not found: {self.profiles_dir}")

    def get_profile_files(self) -> list[Path]:
        """Get all profile JSON files."""
        return list(self.profiles_dir.glob("*.json"))

    def make_recommendation_request(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a recommendation request to the live endpoint."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.dry_run:
            logger.info(f"[DRY RUN] Would send request to {self.endpoint_url}")
            return {"session_id": f"dry_run_{datetime.now().isoformat()}", "recommendations": []}

        logger.info(f"Sending request to {self.endpoint_url}")
        response = requests.post(self.endpoint_url, json=profile_data, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Received response with {len(result['recommendations']['articles'])} recommendations")
        return result

    def extract_session_id_from_response(self, response: Dict[str, Any]) -> str | None:
        """Extract session ID from response metadata."""
        # Look for session ID in various places in the response
        recommender_meta = response.get("recommender", {})
        if "session_id" in recommender_meta:
            return recommender_meta["session_id"]

        # If no session ID found, we'll need to list recent S3 objects
        # and match by timestamp (this is a fallback)
        return None

    def find_recent_s3_session(self, profile_id: str, request_time: datetime) -> str | None:
        """Find the most recent S3 session that matches our request."""
        prefix = "pipeline-outputs/"

        try:
            # List objects with the prefix, sorted by last modified
            paginator = self.s3.get_paginator("list_objects_v2")
            sessions = []

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"):
                for prefix_info in page.get("CommonPrefixes", []):
                    session_path = prefix_info["Prefix"]
                    session_id = session_path[len(prefix) :].rstrip("/")

                    # Check if this session was created around our request time
                    if profile_id in session_id or session_id.startswith(profile_id[:8]):
                        sessions.append(session_id)

            # Return the most recent session
            return sorted(sessions, reverse=True)[0] if sessions else None

        except ClientError as e:
            logger.error(f"Error finding S3 session: {e}")
            return None

    def download_s3_outputs(self, session_id: str, profile_name: str):
        """Download S3 outputs for a given session."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would download S3 outputs for session {session_id}")
            return

        profile_output_dir = self.output_dir / profile_name
        profile_output_dir.mkdir(exist_ok=True)

        # Files to download
        files_to_download = [
            "user_model.txt",
            "original_recommendations.pkl",
            "rewritten_recommendations.pkl",
        ]

        prefix = f"pipeline-outputs/{session_id}/"

        for filename in files_to_download:
            s3_key = f"{prefix}{filename}"
            local_path = profile_output_dir / filename

            try:
                logger.info(f"Downloading {s3_key} to {local_path}")
                self.s3.download_file(self.bucket_name, s3_key, str(local_path))
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    logger.warning(f"File not found in S3: {s3_key}")
                else:
                    logger.error(f"Error downloading {s3_key}: {e}")

    def process_profile(self, profile_file: Path):
        """Process a single profile file."""
        profile_name = profile_file.stem
        logger.info(f"Processing profile: {profile_name}")

        # Load profile data
        with open(profile_file, "r") as f:
            profile_data = json.load(f)

        # Extract profile ID for matching
        profile_id = profile_data.get("interest_profile", {}).get("profile_id", "")

        # Make recommendation request
        request_time = datetime.now()
        try:
            response = self.make_recommendation_request(profile_data)
        except Exception as e:
            logger.error(f"Failed to make request for {profile_name}: {e}")
            return

        # Save response locally
        response_file = self.output_dir / profile_name / "response.json"
        response_file.parent.mkdir(exist_ok=True)

        if not self.dry_run:
            with open(response_file, "w") as f:
                json.dump(response, f, indent=2)

        # Extract session ID and download S3 outputs
        session_id = self.extract_session_id_from_response(response)
        if not session_id:
            session_id = self.find_recent_s3_session(profile_id, request_time)

        if session_id:
            logger.info(f"Found session ID: {session_id}")
            self.download_s3_outputs(session_id, profile_name)
        else:
            logger.warning(f"Could not find session ID for {profile_name}")

    def warmup_endpoint(self):
        """Make a GET request to the /warmup endpoint with retry logic."""
        warmup_url = f"{self.endpoint_url.split('/?')[0]}/warmup"

        if self.dry_run:
            logger.info(f"[DRY RUN] Would send warmup request to {warmup_url}")
            return

        for attempt in range(2):  # 1 retry (2 attempts total)
            try:
                logger.info(f"Sending warmup request to {warmup_url} (attempt {attempt + 1})")
                response = requests.get(warmup_url, timeout=30)
                response.raise_for_status()
                logger.info("Warmup request successful")
                return
            except requests.exceptions.RequestException as e:
                if attempt == 0:  # First attempt failed, retry once
                    logger.warning(f"Warmup attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                else:  # Second attempt failed
                    logger.error(f"Warmup failed after 2 attempts: {e}")
                    raise

    def run(self):
        """Run the testing script for all profiles."""
        # Warmup the endpoint before processing profiles
        self.warmup_endpoint()

        profile_files = self.get_profile_files()
        logger.info(f"Found {len(profile_files)} profile files")

        for profile_file in profile_files:
            try:
                self.process_profile(profile_file)
            except Exception as e:
                logger.error(f"Error processing {profile_file}: {e}")
                continue

        logger.info("Testing completed!")


def get_endpoint_url_from_serverless() -> str:
    """Extract endpoint URL from serverless deployment info."""
    # This is a simplified approach - in practice you might need to call
    # `serverless info` or check AWS API Gateway directly
    region = os.getenv("AWS_REGION", "us-east-1")
    stage = os.getenv("STAGE", "local")

    # Default format for serverless HTTP API endpoints
    return f"https://api.{region}.amazonaws.com/{stage}/"


def main():
    parser = argparse.ArgumentParser(description="Test live endpoint with profile requests")
    parser.add_argument("--endpoint", help="Live endpoint URL")
    parser.add_argument("--output-dir", default="evaluation_data", help="Output directory")
    parser.add_argument("--bucket", help="S3 bucket name (default: from env)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")

    args = parser.parse_args()

    # Determine endpoint URL
    endpoint_url = args.endpoint
    if not endpoint_url:
        endpoint_url = get_endpoint_url_from_serverless()
        logger.info(f"Using auto-detected endpoint: {endpoint_url}")

    # Determine bucket name
    bucket_name = args.bucket or os.getenv("PERSISTENCE_BUCKET")
    if not bucket_name:
        logger.error("Bucket name must be provided via --bucket or PERSISTENCE_BUCKET env var")
        sys.exit(1)

    # Run the tester
    tester = LiveEndpointTester(
        endpoint_url=endpoint_url, output_dir=args.output_dir, bucket_name=bucket_name, dry_run=args.dry_run
    )

    tester.run()


if __name__ == "__main__":
    main()
