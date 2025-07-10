import logging
from io import BytesIO

import requests
import torch
from PIL import Image as PILImage
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList

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
        self,
        recommendations: RecommendationList,
        interest_profile: InterestProfile,
        interacted_articles: CandidateSet,
        **kwargs,
    ) -> RecommendationList:
        # Generate user embedding from clicked article images
        clip_user_embedding = self._generate_user_embedding(interacted_articles)
        if clip_user_embedding is None:
            logger.warning("No valid user embedding generated. Skipping image selection.")
            return recommendations

        # Embed images for each article in the recommendation list and select the best one
        for article in recommendations.articles:
            if not article.images:
                continue

            image_urls = [img.url for img in article.images if img.url]
            if not image_urls:
                continue

            # Embed images and select the best one based on dot product with user embedding
            image_embeddings = self.embed_images(image_urls)
            if image_embeddings is not None:
                best_image = self._select_best_image(image_embeddings, clip_user_embedding, article.images)
                if best_image:
                    article.preview_image_id = best_image.image_id

        # Store user embedding in interest profile
        interest_profile.embedding = clip_user_embedding

        return recommendations

    def _generate_user_embedding(self, interacted_articles: CandidateSet) -> torch.Tensor:
        """Generate user embedding by averaging embeddings of images from the last 50 clicked articles."""
        # Limit to the last 50 articles
        recent_articles = (
            interacted_articles.articles[-50:]
            if len(interacted_articles.articles) > 50
            else interacted_articles.articles
        )

        image_urls = []
        for article in recent_articles:
            if article.images:
                image_urls.extend([img.url for img in article.images if img.url])

        if not image_urls:
            logger.warning("No valid image URLs found in the last 50 interacted articles.")
            return None

        # Embed images from clicked articles
        image_embeddings = self.embed_images(image_urls)
        if image_embeddings is None:
            return None

        # Average the embeddings to create user embedding
        user_embedding = torch.mean(image_embeddings, dim=0)
        return user_embedding

    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list) -> any:
        """Select the best image by computing dot product between image embeddings and user embedding."""
        if image_embeddings.shape[0] != len(images):
            logger.error("Mismatch between image embeddings and images list.")
            return None

        # Compute dot product scores
        scores = torch.matmul(image_embeddings, user_embedding)
        best_index = torch.argmax(scores).item()
        return images[best_index]

    def embed_images(self, image_urls: list) -> torch.Tensor:
        """Generate CLIP embeddings for a list of image URLs."""
        image_tensors = []
        for url in image_urls:
            img_tensor = self._load_image_from_url(url)
            if img_tensor is not None:
                image_tensors.append(img_tensor)

        if not image_tensors:
            logger.warning("No valid images loaded for embedding.")
            return None

        image_tensors = torch.stack(image_tensors).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model(pixel_values=image_tensors).last_hidden_state
            image_embeddings = image_features[:, 0, :]  # Use CLS token embedding

        return image_embeddings

    def _load_image_from_url(self, image_url: str) -> torch.Tensor:
        """Load and preprocess an image from a URL."""
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = PILImage.open(BytesIO(response.content)).convert("RGB")
            return self.transform(img).to(self.device)
        except Exception as e:
            logger.warning(f"Could not download or process image {image_url}: {e}")
            return None
