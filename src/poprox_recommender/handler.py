import json

from poprox_concepts.api.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
)
from poprox_recommender.default import select_articles


def generate_recs(event, context):
    # we get a couple of different formats depending on how we are called
    # if the JSON is just POSTed to Docker, it's as-is
    # if we're in Serverless, it's an event with a 'body'
    if "body" in event:
        req = RecommendationRequest.model_validate_json(event["body"])
    else:
        req = RecommendationRequest.model_validate(event)

    recommendations = select_articles(
        req.todays_articles,
        req.past_articles,
        req.click_histories,
        req.num_recs,
    )

    resp_body = RecommendationResponse.model_validate(
        {"recommendations": recommendations}
    )

    # Dumping to JSON serializes UUIDs properly but requests
    # wants a Python data structure as the body. There's gotta
    # be a better way, but this workaround bridges the gap for now.
    # MDE FIXME: serverless warns that the body is supposed to be a string
    response = {"statusCode": 200, "body": json.loads(resp_body.model_dump_json())}

    return response
