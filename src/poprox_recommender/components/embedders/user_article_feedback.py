import logging
from dataclasses import dataclass

import torch as th

from poprox_concepts import CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


@dataclass
class UserArticleFeedbackConfig(NRMSUserEmbedderConfig):
    feedback_type: bool | None = None


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
    def __call__(self, interacted_articles: CandidateSet, interest_profile: InterestProfile) -> InterestProfile:
        if not hasattr(interest_profile, "article_feedbacks") or len(interest_profile.article_feedbacks or []) == 0:
            logger.info("No feedback available, defaulting feedback embedding to None")
            interest_profile.embedding = None
        else:
            feedback = interest_profile.article_feedbacks
            logger.info(f"{len(feedback)} unfiltered article feedbacks of types {set(feedback.values())}")

            # Filter the list of feedback to only include the type this component is configured to process
            filtered_feedback = {
                article_id: feedback_type
                for article_id, feedback_type in feedback.items()
                if feedback_type == self.config.feedback_type
            }

            logger.info(f"{len(filtered_feedback)} filtered article feedbacks of type {self.config.feedback_type}")

            # Make a list of the articles with feedback and embed them
            feedbacked_articles = [
                article for article in interacted_articles.articles if article.article_id in filtered_feedback.keys()
            ]

            embedded_feedbacked_articles = self.article_embedder(CandidateSet(articles=feedbacked_articles))

            # Turn the list of embedded articles into a UUID -> embedding dictionary
            embedding_lookup = {
                article.article_id: embedding
                for article, embedding in zip(feedbacked_articles, embedded_feedbacked_articles.embeddings)
            }

            # Turn the dictionary of embedded articles into a list of clicks
            feedback_clicks = [Click(article_id=article_id) for article_id in embedding_lookup.keys()]

            if len(embedding_lookup.values()) > 0:
                embedding_lookup["PADDED_NEWS"] = th.zeros(
                    list(embedding_lookup.values())[0].size(), device=self.config.device
                )

                interest_profile.click_history = feedback_clicks

                interest_profile.embedding = self.build_user_embedding(feedback_clicks, embedding_lookup)
            else:
                interest_profile.embedding = None

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
