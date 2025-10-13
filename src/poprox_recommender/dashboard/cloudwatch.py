"""CloudWatch Logs integration for fetching Lambda execution errors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


@dataclass
class LogEvent:
    """A single CloudWatch log event."""

    timestamp: datetime
    message: str
    log_stream: str
    request_id: Optional[str] = None


class CloudWatchLogsService:
    """Service for fetching Lambda errors from CloudWatch Logs."""

    def __init__(self, function_name: Optional[str] = None, region: Optional[str] = None):
        """Initialize CloudWatch Logs client.

        Args:
            function_name: Lambda function name (defaults to env var or serverless config)
            region: AWS region (defaults to env var or us-east-1)
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is required for CloudWatch Logs integration")

        self.function_name = function_name or self._get_function_name()
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.log_group = f"/aws/lambda/{self.function_name}"

        self.client = boto3.client("logs", region_name=self.region)

    def _get_function_name(self) -> str:
        """Get Lambda function name from environment or config."""
        # Try environment variable first
        fn_name = os.getenv("LAMBDA_FUNCTION_NAME")
        if fn_name:
            return fn_name

        # Try to construct from serverless config
        service = os.getenv("SERVERLESS_SERVICE", "poprox-default-recommender")
        stage = os.getenv("SERVERLESS_STAGE", "prod")
        return f"{service}-{stage}-generateRecommendations"

    def fetch_errors_for_day(self, target_date: date, limit: int = 100) -> List[LogEvent]:
        """Fetch error logs for a specific day.

        Args:
            target_date: The date to fetch logs for
            limit: Maximum number of log events to return

        Returns:
            List of log events containing errors
        """
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        return self.fetch_errors(start_time, end_time, limit)

    def fetch_errors(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[LogEvent]:
        """Fetch error logs within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of log events to return

        Returns:
            List of log events containing errors
        """
        try:
            # Query for ERROR level logs and tracebacks
            query = """
                fields @timestamp, @message, @logStream, @requestId
                | filter @message like /ERROR/ or @message like /Traceback/ or @message like /Exception/
                | sort @timestamp desc
            """

            # Start the query
            response = self.client.start_query(
                logGroupName=self.log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query,
                limit=limit,
            )

            query_id = response["queryId"]

            # Poll for query completion
            max_attempts = 30
            for _ in range(max_attempts):
                import time

                time.sleep(0.5)

                result = self.client.get_query_results(queryId=query_id)
                status = result["status"]

                if status == "Complete":
                    return self._parse_query_results(result["results"])
                elif status == "Failed" or status == "Cancelled":
                    raise RuntimeError(f"CloudWatch Logs query {status.lower()}")

            raise TimeoutError("CloudWatch Logs query timed out")

        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to fetch CloudWatch Logs: {e}") from e

    def _parse_query_results(self, results: List[List[dict]]) -> List[LogEvent]:
        """Parse CloudWatch Insights query results into LogEvent objects."""
        events = []

        for result in results:
            field_dict = {field["field"]: field["value"] for field in result}

            timestamp_str = field_dict.get("@timestamp")
            message = field_dict.get("@message", "")
            log_stream = field_dict.get("@logStream", "")
            request_id = field_dict.get("@requestId")

            if timestamp_str:
                try:
                    # CloudWatch returns ISO format timestamps
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    # Fallback to current time if parsing fails
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            events.append(
                LogEvent(
                    timestamp=timestamp,
                    message=message,
                    log_stream=log_stream,
                    request_id=request_id,
                )
            )

        return events

    def test_connection(self) -> bool:
        """Test if the CloudWatch Logs connection works.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.client.describe_log_groups(logGroupNamePrefix=self.log_group, limit=1)
            return True
        except (BotoCoreError, ClientError):
            return False
