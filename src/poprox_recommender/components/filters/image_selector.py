import logging
from typing import Optional

import torch as th

from poprox_concepts.domain.article import Article
from poprox_concepts.domain.image import Image
from poprox_recommender.components.embedders.caption_embedder import CaptionEmbedder
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


class ImageSelector:
    caption_embedder: CaptionEmbedder

    def __init__(self, caption_embedder: CaptionEmbedder):
        self.caption_embedder = caption_embedder

    @torch_inference
    def select_image(self, article: Article, user_embedding: th.Tensor) -> Optional[Image]:
        """
        Select the most relevant image for an article based on the dot product of user embedding
        and caption embeddings.

        Args:
            article: The Article object containing a list of images.
            user_embedding: The user embedding tensor (shape: [1, embedding_size]).

        Returns:
            The most relevant Image object, or None if no images are available or no captions exist.
        """
        if not article.images or not any(image.caption for image in article.images):
            logger.debug(f"No images or captions for article {article.article_id}")
            return None

        # Extract captions and images
        captions = [image.caption for image in article.images if image.caption]
        images = [image for image in article.images if image.caption]

        if not captions:
            logger.debug(f"No captioned images for article {article.article_id}")
            return None

        # Generate embeddings for captions
        caption_embeddings = self.caption_embedder.embed_captions(captions)
        logger.debug(f"Caption embeddings shape: {caption_embeddings.shape}")

        # Ensure user_embedding is squeezed to [embedding_size]
        user_embedding = user_embedding.squeeze(0)
        logger.debug(f"User embedding shape: {user_embedding.shape}")

        # Compute dot product
        scores = th.matmul(caption_embeddings, user_embedding)
        logger.debug(f"Dot product scores: {scores}")

        # Select the image with the highest score
        best_index = th.argmax(scores).item()
        logger.debug(f"Selected image index: {best_index}")
        return images[best_index]
