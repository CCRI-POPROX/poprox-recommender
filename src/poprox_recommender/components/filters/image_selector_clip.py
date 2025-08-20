import logging
from uuid import UUID

import numpy as np
import torch
from lenskit import Component

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList

logger = logging.getLogger(__name__)


class GenericImageSelector(Component):
    config: None

    def __call__(
        self,
        recommendations: RecommendationList,
        interest_profile: InterestProfile,
        interacted_articles: CandidateSet,
        embedding_lookup: dict[UUID, dict[str, np.ndarray]],
        **kwargs: object,
    ) -> RecommendationList:
        """Select personalized images for each article using CLIP embeddings."""
        # Generate user embedding from clicked article images
        clip_user_embedding = self._generate_user_embedding(interacted_articles, embedding_lookup)
        if clip_user_embedding is None:
            return recommendations

        # Select best image for each article
        for article in recommendations.articles:
            if not article.images:
                continue

            # Get embeddings for all images in this article
            valid_embeddings = []
            valid_images = []

            for img in article.images:
                if img.image_id in embedding_lookup and "image" in embedding_lookup[img.image_id]:
                    embedding_data = embedding_lookup[img.image_id]["image"]
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                    valid_embeddings.append(embedding_tensor)
                    valid_images.append(img)

            # Select best image if we have embeddings
            if valid_embeddings:
                image_embeddings = torch.stack(valid_embeddings)
                best_image = self._select_best_image(image_embeddings, clip_user_embedding, valid_images)
                if best_image:
                    article.preview_image_id = best_image.image_id

        return recommendations

    def _generate_user_embedding(self, interacted_articles: CandidateSet, embedding_lookup) -> torch.Tensor | None:
        """Generate user embedding by averaging CLIP embeddings of preview images."""
        valid_embeddings = []

        for article in interacted_articles.articles[-50:]:  # Use last 50 articles
            if article.preview_image_id and article.preview_image_id in embedding_lookup:
                if "image" in embedding_lookup[article.preview_image_id]:
                    embedding_data = embedding_lookup[article.preview_image_id]["image"]
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                    valid_embeddings.append(embedding_tensor)

        if not valid_embeddings:
            return None

        # Average and normalize
        user_embedding = torch.mean(torch.stack(valid_embeddings), dim=0)
        user_embedding = user_embedding / torch.norm(user_embedding)

        return user_embedding


    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list):
        """Select the best image using cosine similarity."""
        if len(images) == 0:
            return None

        # Normalize image embeddings
        image_embeddings_norm = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(image_embeddings_norm, user_embedding)
        best_index = torch.argmax(similarities).item()

        return images[best_index]
