from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import torch as th
import torch.nn.functional as F

from poprox_concepts import Article, CandidateSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.multiple_users import (
    NRMSMultipleUserEmbedder,
    NRMSMultipleUserEmbedderConfig,
)
from poprox_recommender.paths import model_file_path
from poprox_recommender.pytorch.decorators import torch_inference

TOPIC_DESCRIPTIONS = {
    "U.S. news": "News and events within the United States, \
        covering a wide range of topics including politics, \
        economy, social issues, cultural developments, crime, \
        education, healthcare, and other matters of national significance.",
    "World news": "News and events from across the globe, \
        focusing on international developments, global politics, \
        conflicts, diplomacy, trade, cultural exchanges, and \
        issues that affect multiple nations or regions.",
    "Politics": "The activities and functions of a governing body, \
        the administration of its internal and external affairs, \
        and the political issues that governments confront. \
        Includes governance of political entities at all levels \
        (country, state, city, etc.), and all government branches \
        (executive, judicial, legislative, military, law enforcement). \
        Also includes international governing bodies such as the UN.",
    "Business": "All commercial, industrial, financial and \
        economic activities involving individuals, corporations, \
        financial markets, governments and other organizations \
        across all countries and regions.",
    "Entertainment": "All forms of visual and performing arts, \
        design arts, books and literature, film and television, \
        music, and popular entertainment. Refers primarily to \
        the art and entertainment itself and to those who create, \
        perform, or interpret it. For business contexts, \
        see 'Media and entertainment industry'.",
    "Sports": "Organized competitive activities, usually physical in \
        nature, and the systems and practices that support them. \
        Includes all team and individual sports at all levels. \
        Also includes sports media, business, equipment, issues, and controversies.",
    "Health": "Condition, care, and treatment of the mind and body. \
        Includes diseases, illnesses, injuries, medicine, \
        medical procedures, preventive care, health services, \
        and public health issues.",
    "Science": "The ongoing discovery and increase of human knowledge \
        through systematic and disciplined experimentation, \
        and the body of knowledge thus obtained. Includes all branches \
        of natural and social sciences, scientific issues \
        and controversies, space exploration, and similar topics. \
        May include some aspects of 'applied science', \
        but for content about inventions, computers, engineering, etc., \
        Technology is often a more appropriate category.",
    "Technology": "Tools, machines, systems or techniques, \
        especially those derived from scientific knowledge and \
        often electronic or digital in nature, for implementation \
        in industry and/or everyday human activities. \
        Includes all types of technological innovations and \
        products, such as computers, communication and \
        entertainment devices, software, industrial advancements, \
        and the issues and controversies that technology gives rise to.",
    "Lifestyle": "The way a person lives, including interests, \
        attitudes, personal and domestic style, values, relationships, \
        hobbies, recreation, travel, personal care and grooming, \
        and day-to-day activities.",
    "Religion": "All topics related to religion and its place in society, \
        particularly socially and politically controversial topics. \
        See terms for individual belief systems for their \
        activities at all levels of organization.",
    "Climate and environment": "The natural or physical world,\
        and especially the relationship between nature \
        (ecosystems, wildlife, the atmosphere, water, land, etc.) \
        and human beings. Includes the effects of human activities \
        on the environment and vice versa, as well as the \
        management of nature by humans. May also include \
        discussions of the natural world that are unrelated \
        to humans or human activity.",
    "Education": "The processes of teaching and learning in \
        an institutional setting, including all topics related \
        to the establishment and management of educational institutions.",
    "Oddities": "Unusual, quirky, or strange stories that \
        capture attention due to their uniqueness, humor, or \
        unexpected nature. Often includes tales of rare occurrences, \
        peculiar behaviors, or bizarre phenomena.",
}

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
            abs_pref = abs(preference - 2) + 1
            if topic_name in topic_uuids_by_name:
                article_id = topic_uuids_by_name[topic_name]
                virtual_clicks.extend([Click(article_id=article_id)] * abs_pref)
    return virtual_clicks


@dataclass
class UserExplicitTopicalEmbedderConfig(NRMSMultipleUserEmbedderConfig):
    topic_pref_values: list | None = None
    max_clicks_per_user: int = 70


class UserExplicitTopicalEmbedder(NRMSMultipleUserEmbedder):
    config: UserExplicitTopicalEmbedderConfig

    article_embedder: NRMSArticleEmbedder
    embedded_topic_articles: CandidateSet | None = None

    def __init__(self, config: UserExplicitTopicalEmbedderConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
        self.article_embedder = NRMSArticleEmbedder(
            model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=self.config.device
        )

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile
    ) -> InterestProfile:
        interest_profile = interest_profile.model_copy()

        # 01: check which kind of topical preference interpretation is required
        if self.config.topic_pref_values is not None:
            topic_clicks = virtual_pn_clicks(
                interest_profile.onboarding_topics, TOPIC_ARTICLES, self.config.topic_pref_values
            )
        else:
            topic_clicks = virtual_clicks(interest_profile.onboarding_topics, TOPIC_ARTICLES)

        # 02: according to that interpretation check whether there is any virtual click produced or not
        if len(topic_clicks) == 0:
            interest_profile.embedding = None
        else:
            # 03: only if there is any virtual click, work on embeding.
            # It is because if there is no virtual click and still generate
            # the embedding, it will produce NaN and eventually score will turn into NaN
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
        return clicked_news_vector
