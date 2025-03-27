import logging
from typing import Optional

import torch as th

from poprox_concepts.domain import Image, InterestProfile, RecommendationList
from poprox_recommender.components.embedders.caption_embedder import CaptionEmbedder
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


class ImageSelector:
    caption_embedder: CaptionEmbedder

    def __init__(self, caption_embedder: CaptionEmbedder):
        self.caption_embedder = caption_embedder

    @torch_inference
    def __call__(self, recommendations: RecommendationList, interest_profile: InterestProfile) -> RecommendationList:
        if not recommendations.articles:
            # logger.debug("No articles in recommendation list")
            return recommendations

        if interest_profile.embedding is None:
            # logger.warning("No user embedding available; skipping image selection")
            return recommendations

        user_embedding = interest_profile.embedding.squeeze(0)
        # logger.debug(f"User embedding shape: {user_embedding.shape}")

        for article in recommendations.articles:
            selected_image = self._select_article_image(article, user_embedding)
            if selected_image:
                article.preview_image_id = selected_image.image_id
                # logger.debug(f"Selected image {selected_image.image_id} for article {article.article_id}")

        return recommendations

    def _select_article_image(self, article, user_embedding) -> Optional[Image]:
        if not article.images or not any(image.caption for image in article.images):
            # logger.debug(f"No images or captions for article {article.article_id}")
            return None

        captions = [image.caption for image in article.images if image.caption]
        images = [image for image in article.images if image.caption]

        if not captions:
            # logger.debug(f"No captioned images for article {article.article_id}")
            return None

        caption_embeddings = self.caption_embedder(captions)
        # logger.debug(f"Caption embeddings shape for article {article.article_id}: {caption_embeddings.shape}")

        if caption_embeddings.shape[1] != user_embedding.shape[0]:
            logger.error(
                f"Shape mismatch: caption embeddings {caption_embeddings.shape}, user embedding {user_embedding.shape}"
            )
            return None

        scores = th.matmul(caption_embeddings, user_embedding)
        # logger.debug(f"Dot product scores for article {article.article_id}: {scores}")

        best_index = th.argmax(scores).item()
        # logger.debug(f"Selected image index for article {article.article_id}: {best_index}")
        return images[best_index]
