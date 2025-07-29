import logging
from io import BytesIO
from typing import TypeAlias
from uuid import UUID, uuid4

import numpy as np
import requests
import torch
import torch as th
from PIL import Image as PILImage
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from poprox_concepts.domain import Article, CandidateSet, Click, Image, InterestProfile, RecommendationList
from poprox_recommender.components.filters.image_selector_clip import GenericImageSelector

EmbeddingLookup: TypeAlias = dict[UUID, dict[str, np.ndarray]]

logger = logging.getLogger(__name__)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26130258, 0.26130258, 0.27577711]),
    ]
)

model_name: str = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).vision_model.to("cpu")


def _load_image_from_url(image_url: str, device) -> th.Tensor | None:
    """Load and preprocess an image from a URL."""
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content)).convert("RGB")
        return transform(img).to(device)
    except Exception as e:
        logger.warning(f"Could not download or process image {image_url}: {e}")
        return None


def embed_images(image_urls: list, device) -> th.Tensor | None:
    """Generate CLIP embeddings for a list of image URLs."""
    image_tensors = []
    for url in image_urls:
        img_tensor = _load_image_from_url(url, device)
        if img_tensor is not None:
            image_tensors.append(img_tensor)

    if not image_tensors:
        logger.warning("No valid images loaded for embedding.")
        return None

    image_tensors = th.stack(image_tensors).to(device)
    with th.no_grad():
        image_features = clip_model(pixel_values=image_tensors).last_hidden_state
        image_embeddings = image_features[:, 0, :]  # Use CLS token embedding

    return image_embeddings


def test_uses_embedding_lookup():
    device = "cpu"
    selector = GenericImageSelector()

    target_image_id = uuid4()
    recommended_articles = [
        Article(
            headline="this is an article headline",
            images=[
                Image(
                    image_id=target_image_id,
                    url="https://images.pexels.com/photos/28120193/pexels-photo-28120193/free-photo-of-scenic-view-of-a-terrace-plantation.jpeg",
                    source="AP",
                ),
                Image(
                    image_id=uuid4(),
                    url="https://images.pexels.com/photos/6872754/pexels-photo-6872754.jpeg",
                    source="AP",
                ),
            ],
        )
    ]

    selected_image_id = uuid4()
    interacted_article_id = uuid4()
    interacted_articles: list[Article] = [
        Article(
            article_id=interacted_article_id,
            headline="this is technically a different article",
            preview_image_id=selected_image_id,
            images=[
                Image(
                    image_id=selected_image_id,
                    url="https://images.pexels.com/photos/28120193/pexels-photo-28120193/free-photo-of-scenic-view-of-a-terrace-plantation.jpeg",
                    source="AP",
                )
            ],
        )
    ]

    recommendations = RecommendationList(articles=recommended_articles)

    interest_profile = InterestProfile(click_history=[Click(article_id=interacted_article_id)], onboarding_topics=[])

    interacted = CandidateSet(articles=interacted_articles)

    embedding_lookup: EmbeddingLookup = {}

    # Build a lookup table of image embeddings
    article_to_embed = recommended_articles + interacted_articles
    for article in article_to_embed:
        for image in article.images or []:
            if not image.image_id or image.image_id in embedding_lookup:
                continue
            else:
                embedding = embed_images([image.url], device)
                if embedding is not None:
                    embedding_lookup[image.image_id] = {"image": embedding.numpy()}

    assert len(embedding_lookup) > 0

    # Use the embeddings from the lookup table in the __call__ method (instead of the real model)
    result = selector(recommendations, interest_profile, interacted, embedding_lookup)

    assert result.articles[0].preview_image_id == target_image_id
