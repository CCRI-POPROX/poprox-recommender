# components/filters/image_selector.py
import logging
from io import BytesIO
from time import time

import requests
import torch
from PIL import Image as PILImage
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from poprox_concepts.domain import Article, Image, RecommendationList

logger = logging.getLogger(__name__)


class ImageSelector:
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

    def _load_image_from_url(self, image_url: str):
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = PILImage.open(BytesIO(response.content)).convert("RGB")
            return self.transform(img).to(self.device)
        except Exception as e:
            return torch.zeros(3, 224, 224).to(self.device)

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

    def __call__(self, recommendations: RecommendationList, **kwargs) -> RecommendationList:
        # Embed images for each article in the recommendation list
        for article in recommendations.articles:
            if not article.images:
                continue

            image_urls = [img.url for img in article.images if img.url]
            if not image_urls:
                continue

            # Embed images and store
            embeddings = self.embed_images(image_urls)

        return recommendations
