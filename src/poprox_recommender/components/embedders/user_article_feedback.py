from dataclasses import dataclass
from datetime import datetime, timezone

import torch as th

from poprox_concepts import Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference


def feedbacked_article_conversion(article_feedbacks, clicked_articles):
    feedbacked_articles = []

    for feedbacked_article_id, feedback in article_feedbacks.items():
        for article in clicked_articles:
            if article.article_id == feedbacked_article_id:
                feedbacked_articles.append((article, feedback))

    feedbacked_articles = [
        Article(
            article_id=article.article_id,
            headline=article.headline,
            subhead=article.subhead,
            url=article.url,
            preview_image_id=article.preview_image_id,
            published_at=datetime.now(timezone.utc),  # Set current time for simplicity
            mentions=[],
            source="article_feedback",
            external_id="positive" if feedback else "negative",
            raw_data={},
        )
        for article, feedback in feedbacked_articles
    ]
    return feedbacked_articles


def virtual_pn_clicks(feedbacked_articles, feedback_type):
    virtual_clicks = []
    for feedbacked_article in feedbacked_articles:
        if feedbacked_article.external_id == feedback_type:
            virtual_clicks.extend([Click(article_id=feedbacked_article.article_id)])
    return virtual_clicks


@dataclass
class UserArticleFeedbackConfig(NRMSUserEmbedderConfig):
    feedback_type: str | None = None


class UserArticleFeedbackEmbedder(NRMSUserEmbedder):
    config: UserArticleFeedbackConfig  # type: ignore

    article_embedder: NRMSArticleEmbedder
    embedded_feedbacked_articles: CandidateSet | None = None

    def __init__(self, config: UserArticleFeedbackConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.article_embedder = NRMSArticleEmbedder(
            model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=self.config.device
        )

    @torch_inference
    def __call__(self, clicked_articles: CandidateSet, interest_profile: InterestProfile) -> InterestProfile:
        if not hasattr(interest_profile, "article_feedbacks") or len(interest_profile.article_feedbacks) == 0:
            interest_profile.embedding = None
        else:
            ##### article_feedbacks = dict[UUID --> article_id, bool --> feedback] #####
            feedbacked_articles = feedbacked_article_conversion(
                interest_profile.article_feedbacks, clicked_articles.articles
            )
            self.embedded_feedbacked_articles = self.article_embedder(CandidateSet(articles=feedbacked_articles))

            feedbacked_embeddings_by_article_id = {
                article.article_id: embedding
                for article, embedding in zip(feedbacked_articles, self.embedded_feedbacked_articles.embeddings)
            }

            article_feedback_as_clicks = virtual_pn_clicks(feedbacked_articles, self.config.feedback_type)

            feedbacked_article_lookup = {
                article_id: embedding for article_id, embedding in feedbacked_embeddings_by_article_id.items()
            }

            embedding_lookup = {**feedbacked_article_lookup}

            embedding_lookup["PADDED_NEWS"] = th.zeros(
                list(embedding_lookup.values())[0].size(), device=self.config.device
            )

            interest_profile.click_history = article_feedback_as_clicks

            interest_profile.embedding = self.build_user_embedding(interest_profile.click_history, embedding_lookup)

        return interest_profile

    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = [click.article_id for click in click_history][-self.config.max_clicks_per_user :]

        padded_positions = self.config.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).squeeze().to(self.config.device)
            for clicked_article in article_ids
        ]

        clicked_news_vector = (
            th.stack(
                clicked_article_embeddings,
                dim=0,
            )
            .unsqueeze(0)
            .to(self.config.device)
        )

        return self.user_encoder(clicked_news_vector)
