# components/filters/image_selector.py
import logging

import torch as th
from transformers import AutoModel, AutoTokenizer

from poprox_concepts.domain import Article, Image, ImpressedSection, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


class ImageSelector:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)

    @torch_inference
    def __call__(self, recommendations: ImpressedSection, interest_profile: InterestProfile) -> ImpressedSection:
        if not recommendations.impressions:
            return recommendations

        if interest_profile.embedding is None:
            return recommendations

        user_embedding = interest_profile.embedding.squeeze(0)

        for impression in recommendations.impressions:
            selected_image = self._select_article_image(impression.article, user_embedding)
            if selected_image:
                impression.preview_image_id = selected_image.image_id

        return recommendations

    def _select_article_image(self, article: Article, user_embedding: th.Tensor) -> Image | None:
        if not article.images or not any(image.caption for image in article.images):
            return None

        captions = [image.caption for image in article.images if image.caption]
        images = [image for image in article.images if image.caption]

        if not captions:
            return None

        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.encoder(**tokens)
        caption_embeddings = output.last_hidden_state[:, 0, :]

        if caption_embeddings.shape[1] != user_embedding.shape[0]:
            logger.error(
                f"Shape mismatch: caption embeddings {caption_embeddings.shape}, user embedding {user_embedding.shape}"
            )
            return None

        scores = th.matmul(caption_embeddings, user_embedding)
        best_index = th.argmax(scores).item()
        return images[best_index]
