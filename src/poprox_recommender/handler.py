import base64
import gzip
import logging
import os

import structlog

from poprox_concepts import CandidateSet
from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationResponse
from poprox_recommender.recommenders import select_articles

logger = logging.getLogger(__name__)


def generate_recs(event, context):
    logger.info(f"Received event: {event}")

    pipeline_params = event.get("queryStringParameters", {})
    body = event.get("body", {})
    headers = event.get("headers", {})

    is_encoded = event.get("isBase64Encoded", False)
    is_compressed = headers.get("Content-Encoding") == "gzip"

    logger.info(f"Headers: {headers}")

    # base64 encoding is applied to our requests by the AWS stack
    # and compression is applied in our code, which means that
    # we have to base64 decode first and decompress second
    # (contrary to the usual expectation)
    body = base64.b64decode(body) if is_encoded else body
    body = gzip.decompress(body) if is_compressed else body
    logger.info(f"Decoded body: {body}")

    req = RecommendationRequest.model_validate_json(body)

    if pipeline_params:
        logger.info(f"Using parameters: {pipeline_params}")
    else:
        logger.info("Using default parameters")

    num_candidates = len(req.todays_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    # The platform should send an CandidateSet but we'll do it here for now
    candidate_articles = CandidateSet(articles=req.todays_articles)

    # Similarly, the platform should provided pre-filtered clicked articles
    # and compute the topic counts but this shim lets us ignore that issue
    # in the actual article selection

    profile = req.interest_profile
    click_history = profile.click_history

    clicked_articles = list(
        filter(lambda a: a.article_id in set([c.article_id for c in click_history]), req.past_articles)
    )
    clicked_articles = CandidateSet(articles=clicked_articles)

    outputs = select_articles(
        candidate_articles,
        clicked_articles,
        profile,
        pipeline_params,
    )

    logger.info("Constructing response...")
    resp_body = RecommendationResponse.model_validate(
        {"recommendations": {profile.profile_id: outputs.default.articles}, "recommender": outputs.meta.model_dump()}
    )
    # extract properties from the articleset new fields
    # replace the articles with new recommendation object?

    logger.info("Serializing response...")
    response = {"statusCode": 200, "body": resp_body.model_dump_json()}

    logger.info("Finished.")
    return response


if "AWS_LAMBDA_FUNCTION_NAME" in os.environ and not structlog.is_configured():
    # Serverless doesn't set up logging like the AWS Lambda runtime does, so we
    # need to configure base logging ourselves. The AWS_LAMBDA_RUNTIME_API
    # environment variable is set in a real runtime environment but not the
    # local Serverless run, so we can check for that.  We will log at DEBUG
    # level for local testing.
    if "AWS_LAMBDA_RUNTIME_API" not in os.environ:
        logging.basicConfig(level=logging.DEBUG)
        # make sure we have debug for all of our code
        logging.getLogger("poprox_recommender").setLevel(logging.DEBUG)
        logger.info("local logging enabled")

    # set up structlog to dump to standard logging
    # TODO: enable JSON logs
    structlog.configure(
        [
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.MaybeTimeStamper(),
            structlog.processors.KeyValueRenderer(key_order=["event", "timestamp"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    structlog.stdlib.get_logger(__name__).info(
        "structured logging initialized",
        function=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        region=os.environ.get("AWS_REGION", None),
    )
