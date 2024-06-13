import json
import logging

from poprox_concepts.api.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
)
from poprox_recommender.default import select_articles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_recs(event, context):
    req = RecommendationRequest.model_validate_json(event["body"])

    algo_params = event.get("queryStringParameters", {})

    if algo_params:
        logger.info(f"Generating recommendations with parameters: {algo_params}")
    else:
        logger.info("Generating recommendations with default parameters")

    recommendations = select_articles(
        req.todays_articles,
        req.past_articles,
        req.click_histories,
        req.num_recs,
        algo_params,
    )

    resp_body = RecommendationResponse.model_validate(
        {"recommendations": recommendations}
    )

    response = {"statusCode": 200, "body": resp_body.model_dump_json()}

    return response
