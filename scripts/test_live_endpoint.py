#!/usr/bin/env python3
"""
Comprehensive live endpoint testing script that:
1. Tests all available pipelines against all profile requests from tests/request_data/profiles/
2. Runs multiple iterations per pipeline-profile combination for accurate timing data
3. Records detailed timing statistics and performance metrics
4. Downloads and saves S3 outputs for each test run
5. Generates comprehensive timing summaries and aggregated results

Usage:
    python scripts/test_live_endpoint.py [--endpoint URL] [--output-dir DIR] [--bucket BUCKET] [--runs N] [--dry-run]

Args:
    --endpoint: Live endpoint URL (supports pipeline parameter for different pipelines)
    --output-dir: Local directory to save outputs (default: evaluation_data/)
    --bucket: S3 bucket name (default: from PERSISTENCE_BUCKET env var)
    --runs: Number of runs per pipeline-profile combination (default: 3)
    --dry-run: Don't actually make requests or download files
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import boto3
import requests
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def discover_pipelines() -> list[str]:
    """
    Discover the list of available pipeline configuration names.
    """
    return [
        "llm_rank_rewrite",
        "llm_rank_only", 
        "nrms_baseline",
        "nrms_baseline_rewrite",
    ]


class LiveEndpointTester:
    def __init__(self, endpoint_url: str, output_dir: str, bucket_name: str, num_runs: int = 3, dry_run: bool = False):
        self.base_endpoint_url = endpoint_url
        self.output_dir = Path(output_dir)
        self.bucket_name = bucket_name
        self.num_runs = num_runs
        self.dry_run = dry_run

        # Setup S3 client
        self.s3 = boto3.client("s3")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get profile requests directory
        self.profiles_dir = Path("tests/request_data/profiles")
        if not self.profiles_dir.exists():
            raise FileNotFoundError(f"Profiles directory not found: {self.profiles_dir}")

        # Storage for all results
        self.all_results = []

    def get_profile_files(self) -> list[Path]:
        """Get all profile JSON files."""
        return list(self.profiles_dir.glob("*.json"))

    def build_pipeline_url(self, pipeline_name: str) -> str:
        """Build endpoint URL with pipeline parameter."""
        parsed = urlparse(self.base_endpoint_url)
        query_params = parse_qs(parsed.query)
        query_params['pipeline'] = [pipeline_name]
        
        new_query = urlencode(query_params, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

    def make_recommendation_request(self, profile_data: Dict[str, Any], pipeline_url: str) -> tuple[Dict[str, Any], float]:
        """Make a recommendation request to the live endpoint and return response with timing."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.dry_run:
            logger.info(f"[DRY RUN] Would send request to {pipeline_url}")
            return {"session_id": f"dry_run_{datetime.now().isoformat()}", "recommendations": {"articles": []}}, 0.1

        logger.debug(f"Sending request to {pipeline_url}")
        start_time = time.time()
        response = requests.post(pipeline_url, json=profile_data, headers=headers, timeout=60)
        end_time = time.time()
        
        response.raise_for_status()
        execution_time = end_time - start_time

        result = response.json()
        num_recs = len(result.get('recommendations', {}).get('articles', []))
        logger.debug(f"Received response with {num_recs} recommendations in {execution_time:.3f}s")
        return result, execution_time

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

    def process_profile_pipeline_combination(self, profile_file: Path, pipeline_name: str, run_number: int):
        """Process a single profile file with a specific pipeline and run number."""
        profile_name = profile_file.stem
        logger.info(f"Processing {profile_name} with {pipeline_name} (run {run_number}/{self.num_runs})")

        # Load profile data
        with open(profile_file, "r") as f:
            profile_data = json.load(f)

        # Create unique profile ID for this run to avoid overwrites
        original_profile_id = profile_data.get("interest_profile", {}).get("profile_id", "")
        unique_profile_id = str(uuid.uuid4())
        profile_data["interest_profile"]["profile_id"] = unique_profile_id

        # Build pipeline-specific URL
        pipeline_url = self.build_pipeline_url(pipeline_name)

        # Make recommendation request with timing
        request_time = datetime.now()
        try:
            response, execution_time = self.make_recommendation_request(profile_data, pipeline_url)
        except Exception as e:
            logger.error(f"Failed to make request for {profile_name} with {pipeline_name}: {e}")
            return

        # Extract session ID and download S3 outputs
        session_id = self.extract_session_id_from_response(response)
        if not session_id:
            session_id = self.find_recent_s3_session(unique_profile_id, request_time)

        # Create run-specific directory
        run_dir = self.output_dir / f"{profile_name}_{pipeline_name}_run{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save response locally
        if not self.dry_run:
            response_file = run_dir / "response.json"
            with open(response_file, "w") as f:
                json.dump(response, f, indent=2)

        # Download S3 outputs
        if session_id:
            logger.debug(f"Found session ID: {session_id}")
            self.download_s3_outputs(session_id, f"{profile_name}_{pipeline_name}_run{run_number}")
        else:
            logger.warning(f"Could not find session ID for {profile_name} with {pipeline_name}")

        # Create structured result similar to local_req.py
        candidates = profile_data.get("candidates", {}).get("articles", [])
        recommendations = response.get("recommendations", {}).get("articles", [])
        
        # Map article_id to original headline for quick lookup
        article_id_to_headline = {article.get("article_id"): article.get("headline") for article in candidates}

        structured_output = {
            "recommendations": [
                {
                    "rank": idx + 1,
                    "headline": article.get("headline", ""),
                    "original_headline": article_id_to_headline.get(article.get("article_id"), "Unknown"),
                    "article_id": article.get("article_id"),
                }
                for idx, article in enumerate(recommendations)
            ],
            "profile_name": profile_name,
            "pipeline_name": pipeline_name,
            "run_number": run_number,
            "execution_time_seconds": execution_time,
            "timestamp": time.time(),
            "request_time": request_time.isoformat(),
            "session_id": session_id,
            "original_profile_id": original_profile_id,
            "unique_profile_id": unique_profile_id,
            "candidate_pool": [article.get("headline", "") for article in candidates],
            "num_candidates": len(candidates),
            "num_recommendations": len(recommendations),
            "recommender_meta": response.get("recommender", {}),
        }

        # Save individual run results
        if not self.dry_run:
            results_file = run_dir / "structured_results.json"
            with open(results_file, "w") as f:
                json.dump(structured_output, f, indent=2)

        # Add to global results
        self.all_results.append(structured_output)

        logger.info(f"  Completed {profile_name} with {pipeline_name} in {execution_time:.3f}s")

    def warmup_endpoint(self):
        """Make a GET request to the /warmup endpoint with retry logic."""
        warmup_url = f"{self.base_endpoint_url.split('/?')[0]}/warmup"

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
                    logger.warning(f"Warmup failed after 2 attempts: {e}")
                    logger.warning("Continuing without warmup...")

    def calculate_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate timing statistics similar to local_req.py."""
        timing_summary = {}
        for result in self.all_results:
            key = (result["profile_name"], result["pipeline_name"])
            if key not in timing_summary:
                timing_summary[key] = []
            timing_summary[key].append(result["execution_time_seconds"])

        # Calculate timing statistics
        timing_stats = {}
        for (profile, pipeline), times in timing_summary.items():
            timing_stats[f"{profile}_{pipeline}"] = {
                "mean_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "runs": len(times),
                "total_time": sum(times),
            }

        return timing_stats

    def save_results(self):
        """Save comprehensive results similar to local_req.py."""
        if self.dry_run:
            logger.info("[DRY RUN] Would save results")
            return

        # Save timing summary
        timing_stats = self.calculate_timing_statistics()
        timing_file = self.output_dir / "timing_summary.json"
        with open(timing_file, "w") as f:
            json.dump(timing_stats, f, indent=2)

        # Save all results
        all_results_file = self.output_dir / "all_results.json"
        with open(all_results_file, "w") as f:
            json.dump(self.all_results, f, indent=2)

        # Save summary statistics
        summary = {
            "total_tests": len(self.all_results),
            "profiles_tested": len(set(r["profile_name"] for r in self.all_results)),
            "pipelines_tested": len(set(r["pipeline_name"] for r in self.all_results)),
            "runs_per_combination": self.num_runs,
            "total_execution_time": sum(r["execution_time_seconds"] for r in self.all_results),
            "average_execution_time": sum(r["execution_time_seconds"] for r in self.all_results) / len(self.all_results) if self.all_results else 0,
        }
        
        summary_file = self.output_dir / "test_summary.json" 
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Timing summary: {timing_file}")
        logger.info(f"All results: {all_results_file}")
        logger.info(f"Test summary: {summary_file}")

    def run(self):
        """Run comprehensive testing for all profiles and pipelines."""
        # Try to warmup the endpoint before processing profiles (skip if it fails)
        try:
            self.warmup_endpoint()
        except Exception as e:
            logger.warning(f"Skipping warmup due to error: {e}")

        profile_files = self.get_profile_files()
        available_pipelines = discover_pipelines()
        
        logger.info("Starting comprehensive testing:")
        logger.info(f"  Profiles: {len(profile_files)}")
        logger.info(f"  Pipelines: {available_pipelines}")
        logger.info(f"  Runs per combination: {self.num_runs}")
        logger.info(f"  Total tests: {len(profile_files) * len(available_pipelines) * self.num_runs}")

        total_tests = 0
        failed_tests = 0

        # Test each profile with each pipeline multiple times
        for profile_file in profile_files:
            profile_name = profile_file.stem
            logger.info(f"\nTesting profile: {profile_name}")
            
            for pipeline_name in available_pipelines:
                logger.info(f"  Pipeline: {pipeline_name}")
                
                for run_number in range(1, self.num_runs + 1):
                    total_tests += 1
                    try:
                        self.process_profile_pipeline_combination(profile_file, pipeline_name, run_number)
                    except Exception as e:
                        failed_tests += 1
                        logger.error(f"    Error in run {run_number}: {e}")
                        continue

        # Save all results
        self.save_results()

        # Print final summary
        success_rate = (total_tests - failed_tests) / total_tests * 100 if total_tests > 0 else 0
        logger.info("\n=== Testing Completed ===")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful: {total_tests - failed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        if self.all_results:
            avg_time = sum(r["execution_time_seconds"] for r in self.all_results) / len(self.all_results)
            total_time = sum(r["execution_time_seconds"] for r in self.all_results)
            logger.info(f"Average execution time: {avg_time:.3f}s")
            logger.info(f"Total execution time: {total_time:.3f}s")

        logger.info(f"Results saved to: {self.output_dir}")


def get_endpoint_url_from_serverless() -> str:
    """Extract endpoint URL from serverless deployment info."""
    # This is a simplified approach - in practice you might need to call
    # `serverless info` or check AWS API Gateway directly
    region = os.getenv("AWS_REGION", "us-east-1")
    stage = os.getenv("STAGE", "local")

    # Default format for serverless HTTP API endpoints
    return f"https://api.{region}.amazonaws.com/{stage}/"


def main():
    parser = argparse.ArgumentParser(description="Comprehensive live endpoint testing for all pipelines and profiles")
    parser.add_argument("--endpoint", help="Live endpoint URL (supports pipeline parameter)")
    parser.add_argument("--output-dir", default="evaluation_data", help="Output directory")
    parser.add_argument("--bucket", help="S3 bucket name (default: from env)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per pipeline-profile combination")
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
        endpoint_url=endpoint_url, 
        output_dir=args.output_dir, 
        bucket_name=bucket_name, 
        num_runs=args.runs,
        dry_run=args.dry_run
    )

    tester.run()


if __name__ == "__main__":
    main()
