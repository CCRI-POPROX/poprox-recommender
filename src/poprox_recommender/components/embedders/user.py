from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike

import torch as th
import torch.nn.functional as F
from lenskit.pipeline import Component
from safetensors.torch import load_file

from poprox_concepts import CandidateSet, Click, InterestProfile
from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.user_encoder import UserEncoder
from poprox_recommender.pytorch.decorators import torch_inference


@dataclass
class NRMSUserEmbedderConfig:
    model_path: PathLike
    device: str = "cpu"
    max_clicks_per_user: int = 50


class NRMSUserEmbedder(Component, ABC):
    config: NRMSUserEmbedderConfig

    def __init__(self, config: NRMSUserEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        model_cfg = ModelConfig()
        checkpoint = load_file(self.config.model_path)
        self.user_encoder = UserEncoder(model_cfg.hidden_size, model_cfg.num_attention_heads)
        self.user_encoder.load_state_dict(checkpoint)
        self.user_encoder.to(self.config.device)

    @torch_inference
    def __call__(self, interacted_articles: CandidateSet, interest_profile: InterestProfile) -> InterestProfile:
        interest_profile = interest_profile.model_copy()

        if len(interest_profile.click_history) == 0:
            interest_profile.embedding = None
        else:
            embedding_lookup = {}

            # Turn the whole interacted article into them into a UUID -> embedding dictionary
            for article, article_vector in zip(
                interacted_articles.articles, interacted_articles.embeddings, strict=True
            ):
                if article.article_id not in embedding_lookup:
                    embedding_lookup[article.article_id] = article_vector

            embedding_lookup["PADDED_NEWS"] = th.zeros(
                list(embedding_lookup.values())[0].size(), device=self.config.device
            )

            # The function will filter out the embedding only for click_histoty eventually no need for double filtering
            interest_profile.embedding = self.build_user_embedding(interest_profile.click_history, embedding_lookup)

        return interest_profile

    # Compute a vector for each user
    @abstractmethod
    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        return th.Tensor


class NRMSSingleVectorUserEmbedder(NRMSUserEmbedder):
    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = list(dict.fromkeys([click.article_id for click in click_history]))[
            -self.config.max_clicks_per_user :
        ]  # deduplicate while maintaining order

        padded_positions = self.config.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
        default = article_embeddings["PADDED_NEWS"]

        # filtering out the necessary embedding-> clicked articles
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).to(self.config.device) for clicked_article in article_ids
        ]
        clicked_news_vector = (
            th.stack(
                clicked_article_embeddings,
                dim=0,
            )
            .unsqueeze(0)
            .to(self.config.device)
        )
        # using multihead attention followed by additive layer
        return th.nan_to_num(self.user_encoder(clicked_news_vector))


class NRMSMultiVectorUserEmbedder(NRMSUserEmbedder):
    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = list(dict.fromkeys([click.article_id for click in click_history]))[
            -self.config.max_clicks_per_user :
        ]  # deduplicate while maintaining order

        padded_positions = self.config.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).to(self.config.device) for clicked_article in article_ids
        ]
        clicked_news_vector = (
            th.stack(
                clicked_article_embeddings,
                dim=0,
            )
            .unsqueeze(0)
            .to(self.config.device)
        )

        clicked_news_vector = F.normalize(clicked_news_vector, dim=2)
        clicked_news_vector /= len(click_history) or 1

        return clicked_news_vector
