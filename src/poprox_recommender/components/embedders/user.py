from os import PathLike

import torch as th
from lenskit.pipeline import Component
from safetensors.torch import load_file

from poprox_concepts import ArticleSet, Click, InterestProfile
from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms.user_encoder import UserEncoder
from poprox_recommender.pytorch.decorators import torch_inference


class NRMSUserEmbedder(Component):
    def __init__(self, model_path: PathLike, device: str = "cpu", max_clicks_per_user: int = 50):
        self.model_path = model_path
        self.device = device
        self.max_clicks_per_user = max_clicks_per_user

        config = ModelConfig()
        checkpoint = load_file(model_path)
        self.user_encoder = UserEncoder(config.hidden_size, config.num_attention_heads)
        self.user_encoder.load_state_dict(checkpoint)
        self.user_encoder.to(device)

    @torch_inference
    def __call__(self, clicked_articles: ArticleSet, interest_profile: InterestProfile) -> InterestProfile:
        if len(clicked_articles.articles) == 0:
            interest_profile.embedding = None
        else:
            embedding_lookup = {}
            for article, article_vector in zip(clicked_articles.articles, clicked_articles.embeddings, strict=True):
                if article.article_id not in embedding_lookup:
                    embedding_lookup[article.article_id] = article_vector

            embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.device)

            interest_profile.embedding = self.build_user_embedding(interest_profile.click_history, embedding_lookup)

        return interest_profile

    # Compute a vector for each user
    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = list(dict.fromkeys([click.article_id for click in click_history]))[
            -self.max_clicks_per_user :
        ]  # deduplicate while maintaining order

        padded_positions = self.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).to(self.device) for clicked_article in article_ids
        ]
        clicked_news_vector = (
            th.stack(
                clicked_article_embeddings,
                dim=0,
            )
            .unsqueeze(0)
            .to(self.device)
        )
        return self.user_encoder(clicked_news_vector)
