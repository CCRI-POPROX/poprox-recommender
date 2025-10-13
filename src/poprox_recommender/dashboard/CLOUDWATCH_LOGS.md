# CloudWatch Logs Integration

This dashboard now includes integration with AWS CloudWatch Logs to fetch Lambda execution errors for diagnostic purposes.

## Features

- **Fetch Lambda Errors by Date**: When you filter the dashboard by a specific date, you can click the "Fetch Lambda Errors" button to retrieve all error logs from Lambda executions on that day
- **Error Display**: Errors are displayed with timestamps, request IDs, log streams, and full error messages
- **CloudWatch Insights Query**: Uses CloudWatch Logs Insights to efficiently query for ERROR level logs, tracebacks, and exceptions

## Setup

### Prerequisites

1. Install boto3 (already included if you installed with the `s3` extra):
   ```bash
   pip install poprox-recommender[s3]
   # or
   uv pip install boto3
   ```

2. Configure AWS credentials with permissions to read CloudWatch Logs:
   ```bash
   aws configure
   ```

   Required IAM permissions:
   - `logs:DescribeLogGroups`
   - `logs:StartQuery`
   - `logs:GetQueryResults`

### Configuration

The CloudWatch Logs service automatically detects the Lambda function name from:

1. Environment variable `LAMBDA_FUNCTION_NAME`
2. Or constructs it from `SERVERLESS_SERVICE` and `SERVERLESS_STAGE` environment variables
3. Or defaults to `poprox-default-recommender-prod-generateRecommendations`

You can also configure the AWS region:
- Environment variable `AWS_REGION` (defaults to `us-east-1`)

## Usage

1. Start the dashboard:
   ```bash
   python -m poprox_recommender.dashboard
   ```

2. Navigate to the dashboard in your browser (typically http://localhost:8000)

3. Use the date filter to select a specific day

4. Click the "Fetch Lambda Errors" button to retrieve error logs from CloudWatch

5. Review the errors displayed below the session table

## API Endpoints

### GET /api/logs/errors
Fetch Lambda execution errors for a specific date.

**Query Parameters:**
- `date` (required): Date in YYYY-MM-DD format
- `limit` (optional): Maximum number of log events (1-500, default: 100)

**Response:**
```json
{
  "success": true,
  "date": "2025-10-13",
  "count": 5,
  "events": [
    {
      "timestamp": "2025-10-13T14:30:00Z",
      "message": "ERROR: Connection timeout...",
      "log_stream": "2025/10/13/[$LATEST]abc123",
      "request_id": "req-abc-123"
    }
  ]
}
```

### GET /api/logs/test
Test if CloudWatch Logs connection is working.

**Response:**
```json
{
  "available": true,
  "connected": true,
  "log_group": "/aws/lambda/poprox-default-recommender-prod-generateRecommendations",
  "region": "us-east-1"
}
```

## Troubleshooting

### boto3 not available
If you see an error about boto3 not being installed:
```bash
pip install boto3
```

### No permissions
If you see authentication/authorization errors, ensure your AWS credentials have the required CloudWatch Logs permissions.

### Wrong Lambda function
If no logs are found, verify the Lambda function name by checking the `/api/logs/test` endpoint.

## Implementation Details

- **Query Optimization**: Uses CloudWatch Logs Insights for efficient querying
- **Error Filtering**: Searches for messages containing "ERROR", "Traceback", or "Exception"
- **Timeout Handling**: Queries have a 15-second timeout to prevent hanging
- **Asynchronous UI**: Logs are fetched asynchronously to avoid blocking the dashboard
