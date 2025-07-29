import logging

import torch

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList

logger = logging.getLogger(__name__)


class GenericImageSelector:
    def __call__(
        self,
        recommendations: RecommendationList,
        interest_profile: InterestProfile,
        interacted_articles: CandidateSet,
        embedding_lookup,
        **kwargs,
    ) -> RecommendationList:
        # Generate user embedding from clicked article images
        clip_user_embedding = self._generate_user_embedding(interacted_articles, embedding_lookup)
        if clip_user_embedding is None:
            logger.warning("No valid user embedding generated. Skipping image selection.")
            return recommendations

        # Embed images for each article in the recommendation list and select the best one
        for article in recommendations.articles:
            if not article.images:
                continue

            image_ids = [img.image_id for img in article.images if img.image_id]
            if not image_ids:
                continue

            image_embeddings = torch.stack(
                [torch.Tensor(embedding_lookup[image_id]["image"]) for image_id in image_ids]
            )

            if image_embeddings is not None:
                best_image = self._select_best_image(image_embeddings, clip_user_embedding, article.images)
                if best_image:
                    article.preview_image_id = best_image.image_id

        # Store user embedding in interest profile
        interest_profile.embedding = clip_user_embedding

        return recommendations

    def _generate_user_embedding(self, interacted_articles: CandidateSet, embedding_lookup) -> torch.Tensor:
        """Generate user embedding by averaging embeddings of images from the last 50 clicked articles."""
        # Limit to the last 50 articles
        recent_articles = (
            interacted_articles.articles[-50:]
            if len(interacted_articles.articles) > 50
            else interacted_articles.articles
        )

        image_ids = [article.preview_image_id for article in recent_articles]

        if not image_ids:
            logger.warning("No valid image URLs found in the last 50 interacted articles.")
            return None

        image_embeddings = [torch.Tensor(embedding_lookup[image_id]["image"]) for image_id in image_ids]

        if image_embeddings is None:
            return None

        # Average the embeddings to create user embedding
        user_embedding = torch.mean(torch.stack(image_embeddings), dim=0)
        return user_embedding

    def _select_best_image(self, image_embeddings: torch.Tensor, user_embedding: torch.Tensor, images: list) -> any:
        """Select the best image by computing dot product between image embeddings and user embedding."""
        if image_embeddings.shape[0] != len(images):
            logger.error("Mismatch between image embeddings and images list.")
            return None

        # Compute dot product scores
        scores = torch.matmul(image_embeddings, user_embedding.t())
        best_index = torch.argmax(scores).item()
        return images[best_index]
