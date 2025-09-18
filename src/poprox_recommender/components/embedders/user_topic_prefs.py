from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import torch as th
from safetensors import safe_open

from poprox_concepts import Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.topical_description import TOPIC_DESCRIPTIONS
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference


def virtual_clicks(onboarding_topics, topic_articles):
    topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}
    virtual_clicks = []
    for interest in onboarding_topics:
        topic_name = interest.entity_name
        preference = interest.preference or 1

        if topic_name in topic_uuids_by_name:
            article_id = topic_uuids_by_name[topic_name]

            virtual_clicks.extend([Click(article_id=article_id)] * (preference - 1))
    return virtual_clicks


def virtual_pn_clicks(onboarding_topics, topic_articles, topic_values):
    topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}
    virtual_clicks = []
    for interest in onboarding_topics:
        topic_name = interest.entity_name
        preference = interest.preference or -1

        if preference in topic_values:
            abs_pref = abs(preference - 3) + 1
            if topic_name in topic_uuids_by_name:
                article_id = topic_uuids_by_name[topic_name]
                virtual_clicks.extend([Click(article_id=article_id)] * abs_pref)
    return virtual_clicks


def compute_topic_weights(onboarding_topics, topic_articles):
    topic_weight = {}

    topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}
    topic_preference_count = {}
    for interest in onboarding_topics:
        topic_name = interest.entity_name
        preference = interest.preference or 1

        if topic_name in topic_uuids_by_name:
            article_id = topic_uuids_by_name[topic_name]

            topic_preference_count[article_id] = (topic_name, preference)

    total_preference = sum(count[1] for count in topic_preference_count.values())
    topic_weight = {
        article_id: {"topic_name": name, "weight": preference / total_preference}
        for article_id, (name, preference) in topic_preference_count.items()
    }
    return topic_weight


@dataclass
class UserTopicEmbedderConfig(NRMSUserEmbedderConfig):
    topic_embedding: str = "nrms"
    scorer_source: str = "ArticleScorer"
    topic_pref_values: list | None = None


class UserTopicEmbedder(NRMSUserEmbedder, ABC):
    # ignore type because we are overriding a read-only property
    config: UserTopicEmbedderConfig  # type: ignore

    article_embedder: NRMSArticleEmbedder
    embedded_topic_articles: CandidateSet | None = None
    TOPIC_ARTICLES: list[Article] | None = None
    TOPIC_EMBEDDINGS: dict[str, "th.Tensor"] = {}

    def __init__(self, config: UserTopicEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.article_embedder = NRMSArticleEmbedder(
            model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=self.config.device
        )

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile
    ) -> InterestProfile:
        if self.embedded_topic_articles is None:
            self.embedded_topic_articles = self.article_embedder(CandidateSet(articles=self.TOPIC_ARTICLES))
        if self.config.topic_pref_values is not None:
            topic_clicks = virtual_pn_clicks(
                interest_profile.onboarding_topics, self.TOPIC_ARTICLES, self.config.topic_pref_values
            )
        else:
            topic_clicks = virtual_clicks(interest_profile.onboarding_topics, self.TOPIC_ARTICLES)

        topic_embeddings_by_uuid = self.compute_topic_embeddings(candidate_articles, clicked_articles)

        if self.config.scorer_source == "TopicalArticleScorer":
            click_history, topic_lookup, embedding_lookup = self.build_article_click_lookups(
                clicked_articles, interest_profile, topic_embeddings_by_uuid
            )
        elif self.config.scorer_source == "ArticleScorer":
            click_history, topic_lookup, embedding_lookup = self.build_virtual_click_lookups(
                topic_clicks, topic_embeddings_by_uuid
            )
        else:
            raise ValueError(f"Unknown scorer_source value: {self.config.scorer_source}")

        return self.update_interest_profile(
            interest_profile, topic_lookup, click_history, embedding_lookup, self.TOPIC_ARTICLES
        )

    @abstractmethod
    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        return {}

    def build_article_click_lookups(self, clicked_articles, interest_profile, topic_embeddings_by_uuid):
        topic_lookup = {topic_uuid: emb for topic_uuid, emb in topic_embeddings_by_uuid.items()}

        # Use article clicks and virtual topic clicks
        click_history = interest_profile.click_history
        embedding_lookup = self.build_article_lookup(clicked_articles)

        embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.config.device)
        return click_history, topic_lookup, embedding_lookup

    def build_virtual_click_lookups(self, topic_clicks, topic_embeddings_by_uuid):
        topic_lookup = {topic_uuid: emb for topic_uuid, emb in topic_embeddings_by_uuid.items()}

        click_history = topic_clicks
        embedding_lookup = {**topic_lookup}

        embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.config.device)
        return click_history, topic_lookup, embedding_lookup

    def update_interest_profile(
        self, interest_profile, topic_embeddings, click_history, embedding_lookup, topic_articles
    ):
        interest_profile.click_history = click_history
        interest_profile.embedding = th.nan_to_num(self.build_user_embedding(click_history, embedding_lookup))

        # adding topic_embeddings separately
        interest_profile.topic_embeddings = topic_embeddings
        interest_profile.topic_weights = compute_topic_weights(interest_profile.onboarding_topics, topic_articles)

        return interest_profile

    def build_article_lookup(self, article_set: CandidateSet):
        embedding_lookup = {}
        for article, article_vector in zip(article_set.articles, article_set.embeddings, strict=True):
            if article.article_id not in embedding_lookup:
                embedding_lookup[article.article_id] = article_vector

        return embedding_lookup

    def build_embeddings_from_articles(self, articles: CandidateSet, topic_articles: list[Article]):
        topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}

        topic_embeddings_by_uuid = {}
        for topic_name in TOPIC_DESCRIPTIONS.keys():
            embedding_lookup = self.build_article_lookup(articles)

            embedding_lookup["PADDED_NEWS"] = th.zeros([768], device=self.config.device)
            relevant_articles = self.find_topical_articles(topic_name, articles.articles)
            article_clicks = [Click(article_id=article.article_id) for article in relevant_articles]

            if self.config.topic_embedding == "nrms":
                topic_embedding = self.build_user_embedding(article_clicks, embedding_lookup)
            else:
                topic_embedding = self.average_click_embeddings(article_clicks, embedding_lookup)

            if any(topic_embedding.squeeze() != embedding_lookup["PADDED_NEWS"]):
                topic_uuid = topic_uuids_by_name[topic_name]
                topic_embeddings_by_uuid[topic_uuid] = topic_embedding
        return topic_embeddings_by_uuid

    def find_topical_articles(self, topic: str, articles: list[Article]) -> list[Article]:
        topical_articles = []
        for article in articles:
            article_topics = {mention.entity.name for mention in article.mentions}
            if topic in article_topics:
                topical_articles.append(article)
        return topical_articles

    def build_embeddings_from_definitions(self):
        topic_article_set = self.article_embedder(CandidateSet(articles=self.TOPIC_ARTICLES))

        topic_embeddings_by_uuid = {
            article.article_id: embedding
            for article, embedding in zip(self.TOPIC_ARTICLES, topic_article_set.embeddings)  # type: ignore  # noqa: E501
        }

        return topic_embeddings_by_uuid

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

    def average_click_embeddings(self, click_history: list[Click], article_embeddings):
        article_ids = [click.article_id for click in click_history][-self.config.max_clicks_per_user :]

        padded_positions = self.config.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).squeeze().to(self.config.device)
            for clicked_article in article_ids
        ]

        if len(clicked_article_embeddings) == 0:
            return default

        stacked = th.stack(
            clicked_article_embeddings,
            dim=0,
        )
        averaged_click_vector = th.mean(
            stacked,
            dim=0,
        ).to(self.config.device)

        return averaged_click_vector


class StaticDefinitionUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES = [
        Article(
            article_id=uuid4(),
            headline=f"{topic}: {description}",
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

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        return self.build_embeddings_from_definitions()


class CandidateArticleUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES = [
        Article(
            article_id=uuid4(),
            headline=f"{topic}: {description}",
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

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        embeddings_from_definitions = self.build_embeddings_from_definitions()
        embeddings_from_candidates = self.build_embeddings_from_articles(candidate_articles, self.TOPIC_ARTICLES)  # type: ignore
        topic_embeddings_by_uuid = {**embeddings_from_definitions, **embeddings_from_candidates}
        return topic_embeddings_by_uuid


class ClickedArticleUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES = [
        Article(
            article_id=uuid4(),
            headline=f"{topic}: {description}",
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

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        embeddings_from_definitions = self.build_embeddings_from_definitions()
        embeddings_from_candidates = self.build_embeddings_from_articles(candidate_articles, self.TOPIC_ARTICLES)  # type: ignore
        embeddings_from_clicked = self.build_embeddings_from_articles(clicked_articles, self.TOPIC_ARTICLES)  # type: ignore

        topic_embeddings_by_uuid = {
            **embeddings_from_definitions,
            **embeddings_from_candidates,
            **embeddings_from_clicked,
        }
        return topic_embeddings_by_uuid


class HybridUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES = [
        Article(
            article_id=uuid4(),
            headline=f"{topic}: {description}",
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

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        embeddings_from_definitions = self.build_embeddings_from_definitions()
        embeddings_from_candidates = self.build_embeddings_from_articles(candidate_articles, self.TOPIC_ARTICLES)  # type: ignore
        embeddings_from_clicked = self.build_embeddings_from_articles(clicked_articles, self.TOPIC_ARTICLES)  # type: ignore

        all_topic_uuids = (
            set(embeddings_from_definitions) | set(embeddings_from_candidates) | set(embeddings_from_clicked)
        )
        topic_embeddings_by_uuid = {}
        for topic_uuid in all_topic_uuids:
            def_emb = embeddings_from_definitions.get(topic_uuid, th.zeros(768, device=self.config.device))
            cand_emb = embeddings_from_candidates.get(topic_uuid, th.zeros(768, device=self.config.device))
            clicked_emb = embeddings_from_clicked.get(topic_uuid, th.zeros(768, device=self.config.device))

            avg_emb = 0.5 * def_emb + 0.5 * cand_emb + 0.0 * clicked_emb
            topic_embeddings_by_uuid[topic_uuid] = avg_emb
        return topic_embeddings_by_uuid


class PreLearnedStaticDefinitionUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES: list[Article] = []
    TOPIC_EMBEDDINGS: dict[object, th.Tensor] = {}

    def __init__(self, config: UserTopicEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not type(self).TOPIC_ARTICLES:
            with safe_open(model_file_path("topic_embeddings_def.safetensors"), framework="pt", device="cpu") as f:
                for topic_name in f.keys():
                    article = Article(
                        article_id=uuid4(),
                        headline="",
                        subhead=None,
                        url=None,
                        preview_image_id=None,
                        published_at=datetime.now(timezone.utc),  # Set current time for simplicity
                        mentions=[],
                        source="topic",
                        external_id=topic_name,
                        raw_data={},
                    )
                    type(self).TOPIC_ARTICLES.append(article)
                    type(self).TOPIC_EMBEDDINGS[article.article_id] = f.get_tensor(topic_name)

        self.TOPIC_ARTICLES = type(self).TOPIC_ARTICLES  # type: ignore
        self.TOPIC_EMBEDDINGS = type(self).TOPIC_EMBEDDINGS  # type: ignore

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        return self.TOPIC_EMBEDDINGS


class PreLearnedCandidateArticleUserTopicEmbedder(UserTopicEmbedder):
    TOPIC_ARTICLES: list[Article] = []
    TOPIC_EMBEDDINGS: dict[object, th.Tensor] = {}

    def __init__(self, config: UserTopicEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not type(self).TOPIC_ARTICLES:
            with safe_open(
                model_file_path("topic_embeddings_cand_30_days.safetensors"), framework="pt", device="cpu"
            ) as f:
                for topic_name in f.keys():
                    article = Article(
                        article_id=uuid4(),
                        headline="",
                        subhead=None,
                        url=None,
                        preview_image_id=None,
                        published_at=datetime.now(timezone.utc),  # Set current time for simplicity
                        mentions=[],
                        source="topic",
                        external_id=topic_name,
                        raw_data={},
                    )
                    type(self).TOPIC_ARTICLES.append(article)
                    type(self).TOPIC_EMBEDDINGS[article.article_id] = f.get_tensor(topic_name)

        self.TOPIC_ARTICLES = type(self).TOPIC_ARTICLES  # type: ignore
        self.TOPIC_EMBEDDINGS = type(self).TOPIC_EMBEDDINGS  # type: ignore

    def compute_topic_embeddings(self, candidate_articles, clicked_articles):
        return self.TOPIC_EMBEDDINGS
