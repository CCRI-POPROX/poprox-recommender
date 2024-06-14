import base64
import logging

from poprox_concepts.api.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
)
from poprox_recommender.default import select_articles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_recs(event, context):
    logger.info(f"Received event: {event}")
    body = base64.b64decode(event["body"]) if event["isBase64Encoded"] else event["body"]
    logger.info(f"Decoded body: {body}")
    req = RecommendationRequest.model_validate_json(body)

    algo_params = event.get("queryStringParameters", {})

    if algo_params:
        logger.info(f"Using parameters: {algo_params}")
    else:
        logger.info("Using default parameters")

    logger.info("Selecting articles...")
    recommendations = select_articles(
        req.todays_articles,
        req.past_articles,
        req.click_histories,
        req.num_recs,
        algo_params,
    )

    logger.info("Constructing response...")
    resp_body = RecommendationResponse.model_validate({"recommendations": recommendations})

    logger.info("Serializing response...")
    response = {"statusCode": 200, "body": resp_body.model_dump_json()}

    logger.info("Finished.")
    return response
