import logging
import os
from typing import Annotated, Any
from uuid import uuid4

import structlog
from fastapi import Body, FastAPI
from fastapi.responses import Response
from mangum import Mangum

from poprox_concepts.api.recommendations.v3 import (
    ProtocolModelV3_0,
    RecommendationList_v3,
    RecommendationRequestV3,
    RecommendationResponseSection,
    RecommendationResponseV3,
)
from poprox_concepts.domain import Impression
from poprox_recommender.api.gzip import GzipRoute
from poprox_recommender.config import default_device
from poprox_recommender.recommenders import load_all_pipelines, select_articles
from poprox_recommender.topics import user_locality_preference, user_topic_preference

logger = logging.getLogger(__name__)

app = FastAPI()
app.router.route_class = GzipRoute


logger = logging.getLogger(__name__)


@app.get("/warmup")
def warmup(response: Response):
    # Headers set on the response param get included in the response wrapped around return val
    response.headers["poprox-protocol-version"] = ProtocolModelV3_0().protocol_version.value

    # Load and cache available recommenders
    available_recommenders = load_all_pipelines(device=default_device())

    return list(available_recommenders.keys())


@app.post("/")
def root(
    body: Annotated[dict[str, Any], Body()],
    pipeline: str | None = None,
):
    logger.info(f"Decoded body: {body}")

    req = RecommendationRequestV3.model_validate(body)

    candidate_articles = req.candidates.articles
    num_candidates = len(candidate_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    profile = req.interest_profile
    profile.click_topic_counts = user_topic_preference(req.interacted.articles, profile.click_history)
    profile.click_locality_counts = user_locality_preference(req.interacted.articles, profile.click_history)

    outputs = select_articles(
        req.candidates,
        req.interacted,
        profile,
        {"pipeline": pipeline},
    )

    sections = [create_section(title="News for You!", recs=outputs)]
    resp_body = RecommendationResponseV3(recommendations=sections, recommender=outputs.meta.model_dump())

    logger.info(f"Response body: {resp_body}")
    return resp_body.model_dump()


def create_section(title: str, recs):
    articles = recs.default.articles
    newsletter_id = uuid4()
    impressions = [
        {"newsletter_id": newsletter_id, "position": i, "article": articles[i]} for i in range(len(articles))
    ]
    return RecommendationResponseSection(
        title=title,
        recommendations=RecommendationList_v3(
            impressions=impressions,
            extras=recs.default.extras,
        ),
    )


handler = Mangum(app)


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
