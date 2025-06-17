from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import torch as th
import torch.nn.functional as F

from poprox_concepts import AccountInterest, Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.multiple_users import (
    NRMSMultipleUserEmbedder,
    NRMSMultipleUserEmbedderConfig,
)
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

TOPIC_ARTICLES = [
    Article(
        article_id=uuid4(),
        headline=description,
        subhead=None,
        url=None,
        preview_image_id=None,
        published_at=datetime.now(timezone.utc),  # Set current time for simplicity
        mentions=[],
        source="topic",
        external_id=topic,
        raw_data={},
    )
    for topic, description in TOPIC_DESCRIPTIONS.items()
]


def find_implicit_topical_pref(articles: list[Article]) -> list[AccountInterest]:
    topic_names = list(TOPIC_DESCRIPTIONS.keys())
    topic_counts = defaultdict(int)

    for article in articles:
        article_topics = {mention.entity.name for mention in article.mentions}
        for topic in topic_names:
            if topic in article_topics:
                topic_counts[topic] += 1

    implicit_topical_interest = [
        AccountInterest(entity_name=topic, frequency=count) for topic, count in topic_counts.items()
    ]
    return implicit_topical_interest


def virtual_clicks_for_implicit_topical_freq(onboarding_topics, topic_articles):
    topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}
    virtual_clicks = []
    for interest in onboarding_topics:
        topic_name = interest.entity_name
        preference = interest.preference or 1

        if topic_name in topic_uuids_by_name:
            article_id = topic_uuids_by_name[topic_name]

            virtual_clicks.extend([Click(article_id=article_id)] * (preference - 1))
    return virtual_clicks


@dataclass
class UserImplicitTopicalEmbedderConfig(NRMSMultipleUserEmbedderConfig):
    max_clicks_per_user: int = 70


class UserImplicitTopicalEmbedder(NRMSMultipleUserEmbedder):
    config: UserImplicitTopicalEmbedderConfig

    article_embedder: NRMSArticleEmbedder
    embedded_topic_articles: CandidateSet | None = None

    def __init__(self, config: UserImplicitTopicalEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.article_embedder = NRMSArticleEmbedder(
            model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=self.config.device
        )

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile
    ) -> InterestProfile:
        interest_profile = interest_profile.model_copy()

        # 01: check whether the user have any click hostory
        if len(interest_profile.click_history) == 0:
            interest_profile.embedding = None
        else:
            # 02: if there is any click then find the topics of the clicked article
            users_implicit_pref_topics = find_implicit_topical_pref(clicked_articles)
            topic_clicks = virtual_clicks_for_implicit_topical_freq(users_implicit_pref_topics, TOPIC_ARTICLES)

            embeddings_from_definitions = self.build_embeddings_from_definitions()
            embeddings_from_candidates = self.build_embeddings_from_articles(candidate_articles, TOPIC_ARTICLES)

            all_topic_uuids = set(embeddings_from_definitions) | set(embeddings_from_candidates)

            topic_embeddings = {}
            for topic_uuid in all_topic_uuids:
                def_emb = embeddings_from_definitions.get(topic_uuid, th.zeros([1, 768], device=self.config.device))
                cand_emb = embeddings_from_candidates.get(topic_uuid, th.zeros([1, 768], device=self.config.device))

                avg_emb = 0.5 * def_emb + 0.5 * cand_emb
                topic_embeddings[topic_uuid] = avg_emb

            topic_lookup = {topic_uuid: emb for topic_uuid, emb in topic_embeddings.items()}

            embedding_lookup = {**topic_lookup}

            embedding_lookup["PADDED_NEWS"] = th.zeros(
                list(embedding_lookup.values())[0].size(), device=self.config.device
            )

            interest_profile.click_history = topic_clicks

            interest_profile.embedding = self.build_user_embedding(interest_profile.click_history, embedding_lookup)

            interest_profile.embedding = F.normalize(interest_profile.embedding, dim=2)

            interest_profile.embedding /= len(topic_clicks)

        return interest_profile

    def build_embeddings_from_definitions(self):
        topic_article_set = self.article_embedder(CandidateSet(articles=TOPIC_ARTICLES))

        topic_embeddings_by_uuid = {
            article.article_id: embedding for article, embedding in zip(TOPIC_ARTICLES, topic_article_set.embeddings)
        }
        return topic_embeddings_by_uuid

    def build_article_lookup(self, article_set: CandidateSet):
        embedding_lookup = {}
        for article, article_vector in zip(article_set.articles, article_set.embeddings, strict=True):
            if article.article_id not in embedding_lookup:
                embedding_lookup[article.article_id] = article_vector

        return embedding_lookup

    def find_topical_articles(self, topic: str, articles: list[Article]) -> list[Article]:
        topical_articles = []
        for article in articles:
            article_topics = {mention.entity.name for mention in article.mentions}
            if topic in article_topics:
                topical_articles.append(article)
        return topical_articles

    def build_single_user_embedding(self, click_history: list[Click], article_embeddings):
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

    def build_embeddings_from_articles(self, articles: CandidateSet, topic_articles: list[Article]):
        topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}

        topic_embeddings_by_uuid = {}
        for topic_name in TOPIC_DESCRIPTIONS.keys():
            embedding_lookup = self.build_article_lookup(articles)

            for key, emb in embedding_lookup.items():
                embedding_lookup[key] = emb[0]

            embedding_lookup["PADDED_NEWS"] = th.zeros([768], device=self.config.device)
            relevant_articles = self.find_topical_articles(topic_name, articles.articles)
            article_clicks = [Click(article_id=article.article_id) for article in relevant_articles]
            topic_embedding = self.build_single_user_embedding(article_clicks, embedding_lookup)

            if any(topic_embedding.squeeze() != embedding_lookup["PADDED_NEWS"]):
                topic_uuid = topic_uuids_by_name[topic_name]
                topic_embeddings_by_uuid[topic_uuid] = topic_embedding

        return topic_embeddings_by_uuid

    # Compute a vector for each user
    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = list(dict.fromkeys([click.article_id for click in click_history]))[
            -self.config.max_clicks_per_user :
        ]  # deduplicate while maintaining order

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
        return clicked_news_vector
