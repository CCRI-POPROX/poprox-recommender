import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

import structlog
from fastapi import Body, FastAPI
from fastapi.responses import Response
from mangum import Mangum

from poprox_concepts.api.recommendations.v2 import ProtocolModelV2_0, RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.gzip import GzipRoute
from poprox_recommender.config import default_device
from poprox_recommender.recommenders import load_all_pipelines, select_articles
from poprox_recommender.topics import user_locality_preference, user_topic_preference

logger = logging.getLogger(__name__)

# Global variables for background loading
_warmup_future = None
_warmup_result = None
_warmup_lock = threading.Lock()


def background_warmup():
    """Load all pipelines in background thread during Lambda initialization"""
    global _warmup_result
    try:
        logger.info("Starting background warmup of all pipelines")
        device = default_device()
        pipelines = load_all_pipelines(device=device)

        with _warmup_lock:
            _warmup_result = pipelines

        logger.info(f"Background warmup completed successfully. Loaded {len(pipelines)} pipelines")
        return pipelines
    except Exception as e:
        logger.error(f"Background warmup failed: {e}", exc_info=e)
        return None


def get_cached_pipelines():
    """Get pipelines from cache, or load them if background warmup hasn't completed"""
    global _warmup_result, _warmup_future

    with _warmup_lock:
        if _warmup_result is not None:
            return _warmup_result

    # If background warmup is still running, wait for it
    if _warmup_future and not _warmup_future.done():
        logger.info("Waiting for background warmup to complete...")
        try:
            result = _warmup_future.result(timeout=60)  # Wait up to 60s
            return result
        except Exception as e:
            logger.warning(f"Background warmup timeout or failed: {e}")

    # Fallback: load synchronously
    logger.info("Loading pipelines synchronously as fallback")
    return load_all_pipelines(device=default_device())


app = FastAPI()
app.router.route_class = GzipRoute

# Start background loading during Lambda initialization
if "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
    logger.info("Lambda detected - starting background warmup")
    executor = ThreadPoolExecutor(max_workers=1)
    _warmup_future = executor.submit(background_warmup)
else:
    logger.info("Not in Lambda - skipping background warmup")


logger = logging.getLogger(__name__)


@app.get("/warmup")
def warmup(response: Response):
    # Headers set on the response param get included in the response wrapped around return val
    response.headers["poprox-protocol-version"] = ProtocolModelV2_0().protocol_version.value

    # Get available recommenders from cache or load them
    available_recommenders = get_cached_pipelines()

    if available_recommenders is None:
        # If background warmup failed and fallback also failed, return discovered names
        logger.warning("Failed to load pipelines, returning discovered names only")
        from poprox_recommender.recommenders.load import discover_pipelines
        return discover_pipelines()

    return list(available_recommenders.keys())


@app.post("/")
def root(
    body: Annotated[dict[str, Any], Body()],
    pipeline: str | None = None,
):
    logger.info(f"Decoded body: {body}")

    req = RecommendationRequestV2.model_validate(body)

    candidate_articles = req.candidates.articles
    num_candidates = len(candidate_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    profile = req.interest_profile
    profile.click_topic_counts = user_topic_preference(req.interacted.articles, profile.click_history)
    profile.click_locality_counts = user_locality_preference(req.interacted.articles, profile.click_history)

    # Get cached pipelines to avoid loading during request
    cached_pipelines = get_cached_pipelines()

    # If we have cached pipelines, we can use them directly
    # Otherwise, select_articles will load the pipeline itself
    if cached_pipelines:
        from poprox_recommender.recommenders.load import default_pipeline
        pipeline_name = pipeline if pipeline else default_pipeline()

        if pipeline_name in cached_pipelines:
            # Use the cached pipeline directly
            selected_pipeline = cached_pipelines[pipeline_name]
            recs_node = selected_pipeline.node("recommender")
            outputs = recs_node.run(
                candidate=req.candidates,
                clicked=req.interacted,
                profile=profile,
            )
        else:
            # Fall back to normal loading if requested pipeline not cached
            outputs = select_articles(
                req.candidates,
                req.interacted,
                profile,
                {"pipeline": pipeline},
            )
    else:
        # Fall back to normal loading if no cached pipelines
        outputs = select_articles(
            req.candidates,
            req.interacted,
            profile,
            {"pipeline": pipeline},
        )

    resp_body = RecommendationResponseV2.model_validate(
        {"recommendations": outputs.default, "recommender": outputs.meta.model_dump()}
    )

    logger.info(f"Response body: {resp_body}")
    return resp_body.model_dump()


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
