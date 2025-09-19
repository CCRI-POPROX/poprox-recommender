
.PHONY: auth build-ecr deploy build-local serve-local pull-ecr serve-ecr test-warmup-local test-request-local test-warmup-live test-request-live test-live-eval full-deploy serve-dev test-warmup-dev test-request-dev test-dev-eval

# Development server
serve-dev:
	uv run uvicorn poprox_recommender.api.main:app --reload --port 8080

# Development server testing
test-warmup-dev:
	curl -X GET http://localhost:8080/warmup

test-request-dev:
	curl -X POST -H "Content-Type: application/json" "http://localhost:8080/?pipeline=llm-rank-rewrite" -d @testing_data/request-body.json

test-dev-eval:
	uv run scripts/test_live_endpoint.py --endpoint "http://localhost:8080/" --output-dir testing_data/evals --bucket poprox-default-recommender-pipeline-data-prod --runs 3

# AWS ECR Authentication
auth:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 787842565111.dkr.ecr.us-east-1.amazonaws.com

# Build and push to ECR
build-ecr:
	docker buildx build --platform linux/amd64 --provenance=false --build-arg OPENAI_API_KEY=$$(grep OPENAI_API_KEY .env | cut -d '=' -f2) -t 787842565111.dkr.ecr.us-east-1.amazonaws.com/serverless-poprox-default-recommender-prod:latest --push .

# Deploy to Lambda
deploy:
	serverless deploy --stage prod --region us-east-1 --aws-profile default

# Local development
build-local:
	docker build --build-arg OPENAI_API_KEY=$$(grep OPENAI_API_KEY .env | cut -d '=' -f2) -t poprox-recommender-local .

serve-local:
	docker run -p 9000:8080 poprox-recommender-local

# ECR image testing
pull-ecr:
	docker pull 787842565111.dkr.ecr.us-east-1.amazonaws.com/serverless-poprox-default-recommender-prod:latest

serve-ecr:
	docker run -p 9000:8080 787842565111.dkr.ecr.us-east-1.amazonaws.com/serverless-poprox-default-recommender-prod:latest

# Local testing
test-warmup-local:
	curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"version":"2.0","routeKey":"GET /warmup","rawPath":"/warmup","rawQueryString":"","headers":{"accept":"*/*","content-length":"0","host":"localhost","user-agent":"curl/7.81.0"},"requestContext":{"accountId":"anonymous","apiId":"local","http":{"method":"GET","path":"/warmup","protocol":"HTTP/1.1","sourceIp":"127.0.0.1"},"requestId":"local-request-id","routeKey":"GET /warmup","stage":"$$default","timeEpoch":1751662800000},"isBase64Encoded":false}'

test-request-local:
	jq --argjson body "$$(cat testing_data/request-body.json)" '.body = ($$body | @base64)' testing_data/event.json | curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @-

# Live endpoint testing
test-warmup-live:
	curl -X GET https://k8l1uc3s1i.execute-api.us-east-1.amazonaws.com/warmup

test-request-live:
	curl -X POST -H "Content-Type: application/json" "https://k8l1uc3s1i.execute-api.us-east-1.amazonaws.com/?pipeline=llm-rank-rewrite" -d @testing_data/request-body.json

test-live-eval:
	uv run scripts/test_live_endpoint.py --endpoint "https://k8l1uc3s1i.execute-api.us-east-1.amazonaws.com/" --output-dir testing_data/evals --bucket poprox-default-recommender-pipeline-data-prod --runs 3

# Compound commands
full-deploy: auth build-ecr deploy
	@echo "Full deployment completed!"

dev-setup: build-local serve-local
	@echo "Local development environment ready!"
