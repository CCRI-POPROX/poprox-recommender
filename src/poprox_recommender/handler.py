import base64
import logging

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
)
from poprox_recommender.default import select_articles
from poprox_recommender.topics import user_topic_preference

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_recs(event, context):
    logger.info(f"Received event: {event}")

    body = event.get("body", {})
    is_encoded = event.get("isBase64Encoded", False)
    algo_params = event.get("queryStringParameters", {})

    body = base64.b64decode(body) if is_encoded else body
    logger.info(f"Decoded body: {body}")

    req = RecommendationRequest.model_validate_json(body)

    if algo_params:
        logger.info(f"Using parameters: {algo_params}")
    else:
        logger.info("Using default parameters")

    num_candidates = len(req.todays_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    # The platform should send an ArticleSet but we'll do it here for now
    candidate_articles = ArticleSet(articles=req.todays_articles)

    # Similarly, the platform should provided pre-filtered clicked articles
    # and compute the topic counts but this shim lets us ignore that issue
    # in the actual article selection
    profile = req.interest_profile
    click_history = profile.click_history
    clicked_articles = list(filter(lambda a: a.article_id in set(click_history.article_ids), req.past_articles))
    clicked_articles = ArticleSet(articles=clicked_articles)

    profile.click_topic_counts = user_topic_preference(req.past_articles, profile.click_history)

    outputs = select_articles(
        candidate_articles,
        clicked_articles,
        profile,
        req.num_recs,
        algo_params,
    )

    logger.info("Constructing response...")
    resp_body = RecommendationResponse.model_validate(
        {
            "recommendations": {profile.profile_id: outputs.recs},
        }
    )

    logger.info("Serializing response...")
    response = {"statusCode": 200, "body": resp_body.model_dump_json()}

    logger.info("Finished.")
    return response
