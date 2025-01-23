from datetime import datetime, timezone
from uuid import uuid4

import torch as th

from poprox_concepts import Article, ArticleSet, Click, InterestProfile
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
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


class UserOnboardingEmbedder(NRMSUserEmbedder):
    article_embedder: NRMSArticleEmbedder
    embedded_topic_articles: ArticleSet | None = None

    def __init__(self, *args, embedding_source: str = "static", topic_embedding: str = "nrms", **kwargs):
        super().__init__(*args, **kwargs)
        self.article_embedder = NRMSArticleEmbedder(
            model_file_path("nrms-mind/news_encoder.safetensors"), device=self.device
        )
        self.embedding_source = embedding_source
        self.topic_embedding = topic_embedding

    @torch_inference
    def __call__(
        self, candidate_articles: ArticleSet, clicked_articles: ArticleSet, interest_profile: InterestProfile
    ) -> InterestProfile:
        if self.embedded_topic_articles is None:
            self.embedded_topic_articles = self.article_embedder(ArticleSet(articles=TOPIC_ARTICLES))

        topic_embeddings_by_uuid = {
            article.article_id: embedding
            for article, embedding in zip(TOPIC_ARTICLES, self.embedded_topic_articles.embeddings)
        }

        topic_clicks = virtual_clicks(interest_profile.onboarding_topics, TOPIC_ARTICLES)

        embeddings_from_definitions = self.build_embeddings_from_definitions()
        embeddings_from_candidates = self.build_embeddings_from_articles(candidate_articles, TOPIC_ARTICLES)
        embeddings_from_clicked = self.build_embeddings_from_articles(clicked_articles, TOPIC_ARTICLES)

        if self.embedding_source == "static":
            topic_embeddings_by_uuid = embeddings_from_definitions
        elif self.embedding_source == "candidates":
            topic_embeddings_by_uuid = {**embeddings_from_definitions, **embeddings_from_candidates}
        elif self.embedding_source == "clicked":
            topic_embeddings_by_uuid = {
                **embeddings_from_definitions,
                **embeddings_from_candidates,
                **embeddings_from_clicked,
            }
        elif self.embedding_source == "hybrid":
            all_topic_uuids = (
                set(embeddings_from_definitions) | set(embeddings_from_candidates) | set(embeddings_from_clicked)
            )
            topic_embeddings_by_uuid = {}
            for topic_uuid in all_topic_uuids:
                def_emb = embeddings_from_definitions.get(topic_uuid, th.zeros(768, device=self.device))
                cand_emb = embeddings_from_candidates.get(topic_uuid, th.zeros(768, device=self.device))
                clicked_emb = embeddings_from_clicked.get(topic_uuid, th.zeros(768, device=self.device))

                avg_emb = 0.6 * def_emb + 0.3 * cand_emb + 0.1 * clicked_emb
                topic_embeddings_by_uuid[topic_uuid] = avg_emb
        else:
            raise ValueError(f"Unknown embedding source: {self.embedding_source}")

        combined_click_history = interest_profile.click_history + topic_clicks

        click_lookup = self.build_article_lookup(clicked_articles)
        topic_lookup = {topic_uuid: emb for topic_uuid, emb in topic_embeddings_by_uuid.items()}

        embedding_lookup = {**click_lookup, **topic_lookup}
        embedding_lookup["PADDED_NEWS"] = th.zeros(list(embedding_lookup.values())[0].size(), device=self.device)

        interest_profile.click_history = combined_click_history
        interest_profile.embedding = self.build_user_embedding(combined_click_history, embedding_lookup)

        return interest_profile

    def build_article_lookup(self, article_set: ArticleSet):
        embedding_lookup = {}
        for article, article_vector in zip(article_set.articles, article_set.embeddings, strict=True):
            if article.article_id not in embedding_lookup:
                embedding_lookup[article.article_id] = article_vector

        return embedding_lookup

    def build_embeddings_from_articles(self, articles: ArticleSet, topic_articles: list[Article]):
        topic_uuids_by_name = {article.external_id: article.article_id for article in topic_articles}

        topic_embeddings_by_uuid = {}
        for topic_name in TOPIC_DESCRIPTIONS.keys():
            embedding_lookup = self.build_article_lookup(articles)

            embedding_lookup["PADDED_NEWS"] = th.zeros([768], device=self.device)
            relevant_articles = self.find_topical_articles(topic_name, articles.articles)
            article_clicks = [Click(article_id=article.article_id) for article in relevant_articles]

            if self.topic_embedding == "nrms":
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
        topic_article_set = self.article_embedder(ArticleSet(articles=TOPIC_ARTICLES))

        topic_embeddings_by_uuid = {
            article.article_id: embedding for article, embedding in zip(TOPIC_ARTICLES, topic_article_set.embeddings)
        }

        return topic_embeddings_by_uuid

    def build_user_embedding(self, click_history: list[Click], article_embeddings):
        article_ids = [click.article_id for click in click_history][-self.max_clicks_per_user :]

        padded_positions = self.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        article_ids = ["PADDED_NEWS"] * padded_positions + article_ids
        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).squeeze().to(self.device)
            for clicked_article in article_ids
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

    def average_click_embeddings(self, click_history: list[Click], article_embeddings):
        article_ids = [click.article_id for click in click_history][-self.max_clicks_per_user :]

        padded_positions = self.max_clicks_per_user - len(article_ids)
        assert padded_positions >= 0

        default = article_embeddings["PADDED_NEWS"]
        clicked_article_embeddings = [
            article_embeddings.get(clicked_article, default).squeeze().to(self.device)
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
        ).to(self.device)

        return averaged_click_vector
