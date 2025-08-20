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
        # Generating user embedding from clicked article images (only preview images)
        clip_user_embedding = self._generate_user_embedding(interacted_articles, embedding_lookup)
        if clip_user_embedding is None:
            logger.warning("No valid user embedding generated. Skipping image selection.")
            return recommendations

        # Selecting best image for each article in the recommendation list
        for article in recommendations.articles:
            if not article.images:
                continue

            # Get valid embeddings and corresponding images
            valid_embeddings = []
            valid_images = []

            for img in article.images:
                if img.image_id and img.image_id in embedding_lookup:
                    if "image" in embedding_lookup[img.image_id]:
                        embedding_data = embedding_lookup[img.image_id]["image"]

                        # Convert to tensor - ensuring it's always 1D
                        if isinstance(embedding_data, (list, np.ndarray)) and len(embedding_data) > 0:
                            embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32).flatten()

                            # Checking if embedding is not all zeros
                            if torch.sum(torch.abs(embedding_tensor)) > 0:
                                valid_embeddings.append(embedding_tensor)
                                valid_images.append(img)

            # Only proceed if we have valid embeddings
            if valid_embeddings:
                image_embeddings = torch.stack(valid_embeddings)  # Shape: [n_images, embedding_dim]
                best_image = self._select_best_image(image_embeddings, clip_user_embedding, valid_images)
                if best_image:
                    article.preview_image_id = best_image.image_id
                    logger.debug(f"Selected image {best_image.image_id} for article {article.article_id}")

        # Storing user embedding in interest profile
        interest_profile.embedding = clip_user_embedding

        return recommendations

    def _generate_user_embedding(self, interacted_articles: CandidateSet, embedding_lookup) -> torch.Tensor:
        """Generate user embedding by averaging embeddings of ONLY preview images from the last 50 clicked articles."""
        # Limit to the last 50 articles
        recent_articles = (
            interacted_articles.articles[-50:]
            if len(interacted_articles.articles) > 50
            else interacted_articles.articles
        )

        # Get only valid preview image embeddings
        valid_embeddings = []

        for article in recent_articles:
            if article.preview_image_id and article.preview_image_id in embedding_lookup:
                if "image" in embedding_lookup[article.preview_image_id]:
                    embedding_data = embedding_lookup[article.preview_image_id]["image"]

                    # Converting to tensor - ensure it's always 1D
                    if isinstance(embedding_data, (list, np.ndarray)) and len(embedding_data) > 0:
                        embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32).flatten()

                        # Checking if embedding is not all zeros
                        if torch.sum(torch.abs(embedding_tensor)) > 0:
                            valid_embeddings.append(embedding_tensor)

        if not valid_embeddings:
            logger.warning("No valid preview image embeddings found in the interacted articles.")
            return None

        # Averaging the embeddings to create user embedding - result is always 1D
        user_embedding = torch.mean(torch.stack(valid_embeddings), dim=0)  # Shape: [embedding_dim]
        logger.info(f"Generated user embedding from {len(valid_embeddings)} preview image embeddings")
        return user_embedding

    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list) -> any:
        """Select the best image by computing dot product between image embeddings and user embedding."""
        if image_embeddings.shape[0] != len(images):
            logger.error("Mismatch between image embeddings and images list.")
            return None

        if image_embeddings.shape[0] == 0:
            return None

        # Simple dot product - both tensors are guaranteed to be the right shape
        # image_embeddings: [n_images, embedding_dim]
        # user_embedding: [embedding_dim]
        # Result should be: [n_images]
        scores = torch.matmul(image_embeddings, user_embedding)

        best_index = torch.argmax(scores).item()
        best_score = scores[best_index].item()

        logger.debug(f"Selected image with score: {best_score:.4f}")
        return images[best_index]
