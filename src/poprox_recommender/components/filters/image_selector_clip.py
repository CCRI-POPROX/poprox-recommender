import logging
import time
from io import BytesIO

import requests
import torch
from PIL import Image as PILImage
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from poprox_concepts.domain import InterestProfile, RecommendationList, CandidateSet

logger = logging.getLogger(__name__)


class ClipImageSelector:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).vision_model.to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26130258, 0.26130258, 0.27577711]
                ),
            ]
        )
        self.dim = 768

    def __call__(
        self, recommendations: RecommendationList, interest_profile: InterestProfile, interacted_articles: CandidateSet, **kwargs
    ) -> RecommendationList:
        start = time.monotonic()

        all_image_urls = []
        embeddings = []

        # TODO: Loop through either the interacted articles or the click history
        # and find the articles that are in the click history
        interest_profile.click_history

        # TODO: Embed the images from the articles preview_image_id field
        clip_user_embedding =

        # Embed images for each article in the recommendation list
        for article in recommendations.articles:
            if not article.images:
                continue

            image_urls = [img.url for img in article.images if img.url]
            if not image_urls:
                continue

            # Embed images and store
            all_image_urls.extend(image_urls)
            embeddings.append(self.embed_images(image_urls))

        end = time.monotonic()
        logger.warning(f"Image embedding of {len(all_image_urls)} images completed in {end - start} seconds")

        interest_profile.embedding = clip_user_embedding

        return recommendations

    def embed_images(self, image_urls: list):
        # Generate CLIP embeddings for a list of image URLs (for each article)
        image_tensors = []
        for url in image_urls:
            img_tensor = self._load_image_from_url(url)
            image_tensors.append(img_tensor)

        if not image_tensors:
            return None

        image_tensors = torch.stack(image_tensors)
        with torch.no_grad():
            image_features = self.clip_model(pixel_values=image_tensors).last_hidden_state
            image_embeddings = image_features[:, 0, :]  # Use CLS token embedding

        return image_embeddings

    def _load_image_from_url(self, image_url: str):
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = PILImage.open(BytesIO(response.content)).convert("RGB")
            return self.transform(img).to(self.device)
        except Exception as e:
            logger.warning(f"Could not download image {image_url}")
            return torch.zeros(3, 224, 224).to(self.device)
