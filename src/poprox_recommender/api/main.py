import logging
import os
from typing import Annotated, Any
import numpy as np
import torch
import clip
import requests
import numpy as np
import structlog
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse, Response
from mangum import Mangum

from poprox_concepts import Article
from poprox_concepts.api.recommendations.v3 import ProtocolModelV3_0, RecommendationRequestV3, RecommendationResponseV3
from poprox_recommender.api.gzip import GzipRoute
from poprox_recommender.config import default_device
from poprox_recommender.recommenders import load_all_pipelines, select_articles
from poprox_recommender.topics import user_locality_preference, user_topic_preference

logger = logging.getLogger(__name__)

app = FastAPI()
app.router.route_class = GzipRoute

logger = logging.getLogger(__name__)

# Global CLIP model cache
_clip_model = None
_clip_preprocess = None

def get_clip_model():
    """Get cached CLIP model or load it if not cached"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        device = default_device()
        logger.info(f"Loading CLIP model on device: {device}")
        _clip_model, _clip_preprocess = clip.load("ViT-L/14", device=device)  # 768 dimensions
        logger.info("CLIP model loaded successfully")
    return _clip_model, _clip_preprocess

def generate_clip_embedding(image):
    """Generate CLIP embedding for a single image """
    model, preprocess = get_clip_model()
    device = default_device()

    # Download and process image
    if not (hasattr(image, 'url') and image.url):
        raise ValueError(f"No URL found for image {image.image_id}")

    response = requests.get(image.url, timeout=10)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content)).convert('RGB')

    # Preprocess and encode
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Convert to list for JSON serialization
    embedding = image_features.cpu().numpy().flatten().tolist()
    logger.debug(f"Generated embedding of dimension {len(embedding)} for image {image.image_id}")

    return embedding

@app.get("/warmup")
def warmup(response: Response):
    # Headers set on the response param get included in the response wrapped around return val
    response.headers["poprox-protocol-version"] = ProtocolModelV3_0().protocol_version.value

    # Load and cache available recommenders
    available_recommenders = load_all_pipelines(device=default_device())

    # Load CLIP model during warmup
    try:
        get_clip_model()
        logger.info("CLIP model loaded during warmup")
    except Exception as e:
        logger.warning(f"Failed to load CLIP model during warmup: {e}")

    return list(available_recommenders.keys())

# Article -> dict[UUID, dict[str, np.array]]
@app.post("/embed")
def embed(
    body: Annotated[dict[str, Any], Body()],
    pipeline: str | None = None,
):
    logger.info(f"Embedding request received")

    article = Article.model_validate(body)
    embeddings = {}

    if article.images:
        logger.info(f"Processing {len(article.images)} images for article {article.article_id}")

        # Generate embeddings for each image - let failures propagate
        for image in article.images:
            embedding_vector = generate_clip_embedding(image)
            embeddings[image.image_id] = {"image": embedding_vector}
            logger.debug(f"Generated embedding for image {image.image_id}")
    else:
        logger.info(f"No images found for article {article.article_id}")

    logger.info(f"Generated embeddings for {len(embeddings)} images")
    return ORJSONResponse(embeddings)

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

    embeddings = req.embeddings
    # XXX: If we change the over-the-wire format to numpy instead of list of float we can probably get rid of this.
    for embedding_dict in embeddings.values():
        for key in embedding_dict:
            embedding_dict[key] = np.array(embedding_dict[key], dtype=np.float32)

    outputs = select_articles(
        req.candidates,
        req.interacted,
        profile,
        embeddings,
        {"pipeline": pipeline},
    )

    resp_body = RecommendationResponseV3.model_validate(
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
